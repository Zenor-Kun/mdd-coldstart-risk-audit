"""F-RQ4 — conformal robustness (`docs/STAGE_F_PROTOCOL.md` §D).

Three scientifically distinct conformal constructions, kept apart by
construction and never merged:

===========================  ============  =================================
method                       budget        guarantee label
===========================  ============  =================================
`source_pooled_split`        S0            `EMPIRICAL_SOURCE_REFERENCE`
`weighted`                   S1            `EMPIRICAL`
`target_local`               S2            `EXACT_WITHIN_TARGET_SITE`
===========================  ============  =================================

`target_local` is the **only** method permitted a finite-sample coverage claim
(§D.3). The order-statistic arithmetic of §D.7(a) is a *deterministic threshold
construction*; for `source_pooled_split` it is not a guarantee and this module
never labels it as one — the guarantee label travels on every emitted row.

RQ4 retrains nothing. It consumes the frozen ridge-logistic probabilities
(`oof_prob_mdd`, `prob_mdd`), the frozen S1 density weights and the frozen RQ2/S2
splits. The three selective-confidence scores (`probability`, `bootstrap`,
`mahalanobis`) are **forbidden** as nonconformity scores (§D.2) and are never
read here.

This module writes nothing; persistence belongs to `stage_f_store`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import contracts
import site_bootstrap as sb
import stage_f_inputs as inputs
import stage_f_registry as reg

MDD, HC = 1, 0


class ConformalError(AssertionError):
    """A structural violation of the frozen conformal construction."""


# ---------------------------------------------------------------------------
# §D.4 — the nonconformity score
# ---------------------------------------------------------------------------
def nonconformity_scores(prob_mdd) -> tuple:
    """`(a(x, MDD), a(x, HC)) = (1 - p, p)` for the frozen `p = prob_mdd`.

    No rescaling, no ranking, no percentile mapping, no transform (§D.4).
    """
    p = np.asarray(prob_mdd, dtype=float).ravel()
    if not np.isfinite(p).all():
        raise ConformalError("a frozen probability is not finite")
    if (p < 0.0).any() or (p > 1.0).any():
        raise ConformalError("a frozen probability leaves [0, 1]")
    return 1.0 - p, p


def calibration_scores(prob_mdd, y_true) -> np.ndarray:
    """`a_i = a(x_i, y_i)` — the score of each calibration point's own label."""
    a_mdd, a_hc = nonconformity_scores(prob_mdd)
    y = np.asarray(y_true).ravel().astype(int)
    if y.shape != a_mdd.shape:
        raise ConformalError("one label per calibration probability")
    if not np.isin(y, (HC, MDD)).all():
        raise ConformalError("labels must be 1 = MDD / 0 = HC")
    a = np.where(y == MDD, a_mdd, a_hc)
    if (a < 0.0).any() or (a > 1.0).any() or not np.isfinite(a).all():
        raise ConformalError("a nonconformity score left [0, 1] (§J.3)")
    return a


# ---------------------------------------------------------------------------
# §D.7 — the threshold constructions
# ---------------------------------------------------------------------------
def conformal_index(n: int, alpha: float) -> int:
    """`k = ceil((n + 1) * (1 - alpha))` — the frozen order-statistic index."""
    if n < 1:
        raise ConformalError("the calibration sample is empty")
    return int(math.ceil((n + 1) * (1.0 - float(alpha))))


def order_statistic_threshold(scores, alpha: float) -> dict:
    """`t = a_(k)` if `k <= n`, else `+inf` (§D.7a).

    Order statistic only. `numpy.quantile` and every interpolating quantile
    function are forbidden here, so the returned threshold is always an observed
    calibration score or `+inf` — never a value between two of them.
    """
    a = np.sort(np.asarray(scores, dtype=float).ravel())
    n = int(a.size)
    k = conformal_index(n, alpha)
    if k > n:
        return {"tau": math.inf, "n_cal": n, "index": k,
                "degeneracy_code": "ALL_INCLUSIVE_THRESHOLD"}
    tau = float(a[k - 1])                       # a_(k), 1-based order statistic
    if tau not in set(a.tolist()):              # cheap anti-interpolation proof
        raise ConformalError("the threshold is not an observed calibration score")
    return {"tau": tau, "n_cal": n, "index": k, "degeneracy_code": None}


def weighted_threshold(scores, weights, alpha: float) -> dict:
    """`t = min{ a_i : sum_{j : a_j <= a_i} p_j >= 1 - alpha }` (§D.8).

    `p_i = w_i / sum_j w_j`, normalised over the calibration set **only**: the
    exact weighted split-conformal atom at `+inf` carrying the test point's own
    weight is deliberately omitted, because no target-side density-ratio weight
    is persisted anywhere and obtaining one would mean re-deploying the frozen
    S1 domain classifier (§B.2). The construction is therefore EMPIRICAL and
    carries no finite-sample guarantee — and, unlike §D.7(a), it can never
    return `+inf`, because the normalised mass reaches 1 at the largest
    calibration score. That asymmetry is structural and is not evidence that the
    weighted method "succeeded".
    """
    a = np.asarray(scores, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if a.size == 0:
        raise ConformalError("the calibration sample is empty")
    if a.shape != w.shape:
        raise ConformalError("one weight per calibration score")
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ConformalError("density weights must be finite and strictly positive")

    order = np.argsort(a, kind="mergesort")
    a_sorted, w_sorted = a[order], w[order]
    p = w_sorted / w_sorted.sum()
    # Exact-equal scores form one block, so a tie is never split by sort order.
    uniq, inverse = np.unique(a_sorted, return_inverse=True)
    block_mass = np.bincount(inverse, weights=p, minlength=uniq.size)
    cumulative = np.cumsum(block_mass)
    cumulative[-1] = 1.0                        # exact top of the weighted CDF
    target = 1.0 - float(alpha)
    idx = int(np.searchsorted(cumulative, target, side="left"))
    idx = min(idx, uniq.size - 1)
    return {"tau": float(uniq[idx]), "n_cal": int(a.size), "index": idx + 1,
            "degeneracy_code": None}


# ---------------------------------------------------------------------------
# §D.10 — prediction sets and the five rates
# ---------------------------------------------------------------------------
def prediction_set_sizes(prob_mdd, tau: float) -> tuple:
    """`C(x) = {y : a(x, y) <= t}` — membership per label and the set size."""
    a_mdd, a_hc = nonconformity_scores(prob_mdd)
    t = float(tau)
    in_mdd = a_mdd <= t
    in_hc = a_hc <= t
    return in_mdd, in_hc, in_mdd.astype(int) + in_hc.astype(int)


def class_composition_flag(y_true) -> str:
    """`BOTH_CLASSES` / `SINGLE_CLASS_MDD` / `SINGLE_CLASS_HC` (§G.3).

    A composition flag, never an estimability verdict: every registered rate is
    mathematically defined on a single-class evaluation set, and Stage F
    registers no class-conditional conformal endpoint.
    """
    y = np.asarray(y_true).ravel().astype(int)
    has_mdd, has_hc = bool((y == MDD).any()), bool((y == HC).any())
    if has_mdd and has_hc:
        return "BOTH_CLASSES"
    return "SINGLE_CLASS_MDD" if has_mdd else "SINGLE_CLASS_HC"


def evaluate_sets(prob_mdd, y_true, tau: float,
                  threshold_degeneracy: str | None = None) -> dict:
    """The five §D.10 rates on one evaluation set, plus its §G.3 fields.

    All five rates are computable whenever the evaluation set is non-empty —
    including at `t = +inf` and including on a single-class set. Neither makes
    the row `NOT_ESTIMABLE`.
    """
    y = np.asarray(y_true).ravel().astype(int)
    n = int(y.size)
    if n == 0:
        return {"n_eval": 0, "coverage": None, "avg_set_size": None,
                "singleton_rate": None, "empty_rate": None,
                "doubleton_rate": None, "status": reg.NOT_ESTIMABLE,
                "reason_code": reg.assert_reason_code("NO_EVALUATION_POINTS"),
                "degeneracy_flag": False, "degeneracy_code": None,
                "class_composition_flag": None}

    in_mdd, in_hc, size = prediction_set_sizes(prob_mdd, tau)
    if in_mdd.size != n:
        raise ConformalError("one probability per evaluation label")
    covered = np.where(y == MDD, in_mdd, in_hc)

    rates = {
        "coverage": float(covered.mean()),
        "avg_set_size": float(size.mean()),
        "singleton_rate": float((size == 1).mean()),
        "empty_rate": float((size == 0).mean()),
        "doubleton_rate": float((size == 2).mean()),
    }

    code = threshold_degeneracy
    if code is None:
        if bool((size == 2).all()):
            code = "ALL_INCLUSIVE_REALIZED"
        elif bool((size == 0).all()):
            code = "ALL_EMPTY_REALIZED"
    if code is not None and code not in reg.DEGENERACY_CODES:
        raise ConformalError(f"{code!r} is not a registered degeneracy code")
    if code == "ALL_INCLUSIVE_THRESHOLD":
        # A structural consequence of `t = +inf` (§G.3), asserted rather than
        # assumed: every set is the doubleton, so the five rates are forced.
        for name, expected in reg.ALL_INCLUSIVE_METRICS.items():
            if rates[name] != expected:
                raise ConformalError(
                    f"an all-inclusive threshold produced {name}={rates[name]}, "
                    f"the structural value is {expected}")

    return {"n_eval": n, **rates, "status": reg.ESTIMABLE, "reason_code": None,
            "degeneracy_flag": code is not None, "degeneracy_code": code,
            "class_composition_flag": class_composition_flag(y)}


# ---------------------------------------------------------------------------
# per-site drivers — one per method (§D.3)
# ---------------------------------------------------------------------------
def _site_order(sites) -> list:
    """Sites in the frozen `contracts.FROZEN_SITES` order (C15)."""
    wanted = set(map(str, sites))
    return [s for s in contracts.FROZEN_SITES if s in wanted]


def source_pooled_per_site(source_oof: pd.DataFrame, target: pd.DataFrame,
                           alpha: float) -> list:
    """`source_pooled_split` at one α, for every held-out site.

    Calibration mass is **equal** (`1/n`) over every source-OOF acquisition of
    that outer fold (§D.7b): the order-statistic construction is defined on an
    equal-mass empirical distribution, and re-weighting it by `1/n_s` would turn
    it into the separate `weighted` construction. Site balance re-enters at
    aggregation (§D.11), never here. This buys no guarantee either way (§D.3a).
    """
    rows = []
    for site in _site_order(target["held_out_site"]):
        fold = source_oof[source_oof["outer_fold_held_out_site"].astype(str) == site]
        evaluation = target[target["held_out_site"].astype(str) == site]
        if fold.empty:
            raise ConformalError(f"no source-OOF calibration rows for fold {site}")
        a = calibration_scores(fold["oof_prob_mdd"], fold["y_true"])
        threshold = order_statistic_threshold(a, alpha)
        metrics = evaluate_sets(evaluation["prob_mdd"], evaluation["y_true"],
                                threshold["tau"], threshold["degeneracy_code"])
        rows.append({"method": reg.METHOD_SOURCE_POOLED, "alpha": float(alpha),
                     "n_labels": None, "held_out_site": site,
                     "guarantee_label": reg.GUARANTEE_LABELS[reg.METHOD_SOURCE_POOLED],
                     "n_cal": threshold["n_cal"], "tau": threshold["tau"],
                     **metrics})
    return rows


def weighted_per_site(source_oof: pd.DataFrame, weights: pd.DataFrame,
                      target: pd.DataFrame, alpha: float) -> list:
    """`weighted` at one α — the frozen S1 `w_final`, used exactly as persisted.

    No density-ratio model is refitted and the clipping bounds are untouched.
    """
    keyed = weights.set_index([weights["held_out_site"].astype(str),
                               weights["canonical_id"].astype(str)])[reg.WEIGHT_COLUMN]
    rows = []
    for site in _site_order(target["held_out_site"]):
        fold = source_oof[source_oof["outer_fold_held_out_site"].astype(str) == site]
        evaluation = target[target["held_out_site"].astype(str) == site]
        index = list(zip([site] * len(fold), fold["canonical_id"].astype(str)))
        missing = [k for k in index if k not in keyed.index]
        if missing:
            raise ConformalError(
                f"{len(missing)} calibration rows of fold {site} carry no frozen "
                "density weight; a positional join is not permitted")
        w = keyed.loc[index].to_numpy(dtype=float)
        a = calibration_scores(fold["oof_prob_mdd"], fold["y_true"])
        threshold = weighted_threshold(a, w, alpha)
        metrics = evaluate_sets(evaluation["prob_mdd"], evaluation["y_true"],
                                threshold["tau"], threshold["degeneracy_code"])
        rows.append({"method": reg.METHOD_WEIGHTED, "alpha": float(alpha),
                     "n_labels": None, "held_out_site": site,
                     "guarantee_label": reg.GUARANTEE_LABELS[reg.METHOD_WEIGHTED],
                     "n_cal": threshold["n_cal"], "tau": threshold["tau"],
                     **metrics})
    return rows


def target_local_per_repeat(target: pd.DataFrame, registry, alpha: float,
                            n_labels: int) -> list:
    """`target_local` at one (α, N) — every registered site and all 100 repeats.

    The frozen RQ2/S2 splits are reused exactly: no target-label subset is
    redrawn, no RNG is introduced and `src/s2_splitting` is untouched. `N = 5` is
    excluded (§D.6) and this function refuses it.
    """
    if int(n_labels) not in reg.N_GRID:
        raise ConformalError(
            f"N = {n_labels} is not on the frozen target-local grid "
            f"{reg.N_GRID}; N = 5 is excluded by §D.6")
    frame = registry.frame if hasattr(registry, "frame") else registry._frame
    scoped = frame[frame["n_labels"].astype(int) == int(n_labels)]
    lookup = target.set_index(target["canonical_id"].astype(str))

    rows = []
    for site in _site_order(scoped["site"]):
        for repeat in range(reg.R_SPLIT):
            cell = registry.cell(site, repeat, int(n_labels))
            cal, test = list(cell["cal_ids"]), list(cell["test_ids"])
            if set(cal) & set(test):
                raise ConformalError(f"cell {(site, repeat, n_labels)} is not disjoint")
            cal_rows, test_rows = lookup.loc[cal], lookup.loc[test]
            a = calibration_scores(cal_rows["prob_mdd"], cal_rows["y_true"])
            threshold = order_statistic_threshold(a, alpha)
            metrics = evaluate_sets(test_rows["prob_mdd"], test_rows["y_true"],
                                    threshold["tau"], threshold["degeneracy_code"])
            rows.append({
                "alpha": float(alpha), "n_labels": int(n_labels),
                "held_out_site": site, "repeat": int(repeat),
                "guarantee_label": reg.GUARANTEE_LABELS[reg.METHOD_TARGET_LOCAL],
                "n_cal": threshold["n_cal"], "tau": threshold["tau"], **metrics})
    return rows


def collapse_repeats(per_repeat: list) -> list:
    """One per-site row per `(α, N, site)`: the **mean over the 100 repeats**.

    The repeats estimate split uncertainty and are never inference units (C14),
    so they are averaged here and persisted individually for audit — never
    counted as independent observations.

    Two bookkeeping conventions, neither of them scientific:
    `tau` is the mean of the repeats' thresholds (`+inf` propagates), and the
    per-site `degeneracy_flag` is set only when **every** repeat carries the same
    degeneracy code, because a site whose splits sometimes degenerate is not a
    uniformly degenerate object. The per-repeat table keeps the detail.
    """
    grouped = {}
    for row in per_repeat:
        grouped.setdefault((row["alpha"], row["n_labels"], row["held_out_site"]),
                           []).append(row)

    out = []
    for (alpha, n_labels, site), block in grouped.items():
        if len(block) != reg.R_SPLIT:
            raise ConformalError(
                f"{(site, n_labels, alpha)} carries {len(block)} repeats, frozen "
                f"is {reg.R_SPLIT}")
        estimable = [r for r in block if r["status"] == reg.ESTIMABLE]
        codes = {r["degeneracy_code"] for r in block}
        unanimous = codes.pop() if len(codes) == 1 else None
        if not estimable:
            out.append({
                "method": reg.METHOD_TARGET_LOCAL, "alpha": alpha,
                "n_labels": n_labels, "held_out_site": site,
                "guarantee_label": reg.GUARANTEE_LABELS[reg.METHOD_TARGET_LOCAL],
                "n_cal": int(n_labels), "n_eval": 0, "tau": None,
                **{r: None for r in reg.REGISTERED_RATES},
                "status": reg.NOT_ESTIMABLE,
                "reason_code": reg.assert_reason_code("NO_EVALUATION_POINTS"),
                "degeneracy_flag": False, "degeneracy_code": None,
                "class_composition_flag": None})
            continue
        compositions = {r["class_composition_flag"] for r in estimable}
        out.append({
            "method": reg.METHOD_TARGET_LOCAL, "alpha": alpha,
            "n_labels": n_labels, "held_out_site": site,
            "guarantee_label": reg.GUARANTEE_LABELS[reg.METHOD_TARGET_LOCAL],
            "n_cal": int(n_labels),
            "n_eval": int(np.mean([r["n_eval"] for r in estimable])),
            "tau": float(np.mean([r["tau"] for r in estimable])),
            **{rate: float(np.mean([r[rate] for r in estimable]))
               for rate in reg.REGISTERED_RATES},
            "status": reg.ESTIMABLE, "reason_code": None,
            "degeneracy_flag": unanimous is not None,
            "degeneracy_code": unanimous,
            "class_composition_flag": (compositions.pop() if len(compositions) == 1
                                       else "BOTH_CLASSES")})
    return sorted(out, key=lambda r: (r["n_labels"], r["alpha"],
                                      contracts.FROZEN_SITES.index(r["held_out_site"])))


# ---------------------------------------------------------------------------
# §D.11 — aggregation
# ---------------------------------------------------------------------------
def site_macro(per_site: list, method: str, alpha: float, n_labels) -> dict:
    """Unweighted mean over eligible sites — the **primary** RQ4 summary.

    Worst-site coverage is the `min` over eligible sites and is reported
    **descriptively**: it is not a multiple-testing endpoint and carries no
    interval (§D.12.4).
    """
    cell = [r for r in per_site
            if r["method"] == method and r["alpha"] == alpha
            and r["n_labels"] == n_labels]
    if not cell:
        raise ConformalError(f"no per-site rows for cell {(method, alpha, n_labels)}")
    estimable = [r for r in cell if r["status"] == reg.ESTIMABLE]
    base = {"method": method, "alpha": alpha, "n_labels": n_labels,
            "guarantee_label": reg.GUARANTEE_LABELS[method],
            "n_sites": len(cell), "n_estimable_sites": len(estimable),
            "n_degenerate_sites": sum(1 for r in estimable if r["degeneracy_flag"])}
    if not estimable:
        return {**base, "site_macro_coverage": None, "worst_site_coverage": None,
                "worst_site": None, "site_macro_avg_set_size": None,
                "site_macro_singleton_rate": None, "site_macro_empty_rate": None,
                "site_macro_doubleton_rate": None, "status": reg.NOT_ESTIMABLE,
                "reason_code": reg.assert_reason_code("NO_ESTIMABLE_SITE_IN_CELL")}
    worst = min(estimable, key=lambda r: (r["coverage"],
                                          contracts.FROZEN_SITES.index(r["held_out_site"])))
    return {**base,
            **{f"site_macro_{rate}": float(np.mean([r[rate] for r in estimable]))
               for rate in reg.REGISTERED_RATES},
            "worst_site_coverage": float(worst["coverage"]),
            "worst_site": worst["held_out_site"],
            "status": reg.ESTIMABLE, "reason_code": None}


def marginal(per_site: list, method: str, alpha: float, n_labels) -> dict:
    """Acquisition-weighted mean over eligible sites (site weight `n_s`).

    Labelled `marginal_*` in every column so it can never be read as, or
    substituted for, the site-balanced summary (§D.11 prohibition).
    """
    cell = [r for r in per_site
            if r["method"] == method and r["alpha"] == alpha
            and r["n_labels"] == n_labels]
    estimable = [r for r in cell if r["status"] == reg.ESTIMABLE]
    base = {"method": method, "alpha": alpha, "n_labels": n_labels,
            "guarantee_label": reg.GUARANTEE_LABELS[method],
            "n_acquisitions": int(sum(r["n_eval"] for r in estimable))}
    if not estimable:
        return {**base, **{f"marginal_{rate}": None for rate in reg.REGISTERED_RATES},
                "status": reg.NOT_ESTIMABLE,
                "reason_code": reg.assert_reason_code("NO_ESTIMABLE_SITE_IN_CELL")}
    n = np.array([r["n_eval"] for r in estimable], dtype=float)
    return {**base,
            **{f"marginal_{rate}": float(
                np.sum(n * np.array([r[rate] for r in estimable])) / n.sum())
               for rate in reg.REGISTERED_RATES},
            "status": reg.ESTIMABLE, "reason_code": None}


def degeneracy_counts(per_site: list, method: str, alpha: float, n_labels) -> dict:
    """Per-cell degeneracy counts. Descriptive; no interval (§D.12.4)."""
    cell = [r for r in per_site
            if r["method"] == method and r["alpha"] == alpha
            and r["n_labels"] == n_labels]
    return {"method": method, "alpha": alpha, "n_labels": n_labels,
            "n_cells": len(cell),
            **{f"n_{code.lower()}": sum(1 for r in cell
                                        if r["degeneracy_code"] == code)
               for code in reg.DEGENERACY_CODES}}


# ---------------------------------------------------------------------------
# §D.12 — the closed 80-slot CI registry
# ---------------------------------------------------------------------------
_QUANTITY_SOURCE = {
    "rq4_site_macro_coverage": ("coverage", False),
    "rq4_site_macro_avg_set_size": ("avg_set_size", False),
    "rq4_site_macro_singleton_rate": ("singleton_rate", False),
    "rq4_marginal_coverage": ("coverage", True),
}


def interval_for(quantity: str, per_site: list, method: str, alpha: float,
                 n_labels, counts_cache: dict | None = None) -> dict:
    """One registered interval, on the frozen site-cluster-bootstrap machinery.

    `B = 5000`, 95 % percentile, `numpy.quantile(method="linear")`, validity
    floor 4750, namespace `SeedSequence([20260807, 1102])` — every one of them
    reused unchanged from `src/site_bootstrap`, which is the project's single
    site-cluster-bootstrap namespace. No new seed is introduced.

    The draw matrix is a deterministic function of `n_sites` alone, so every
    quantity computed on a common eligible-site set shares bit-identical site
    draws and is paired by construction. Pairing is not claimed across different
    eligible-site sets, and no cross-method / cross-α / cross-`N` contrast is
    registered (§D.12.4), so no unpaired comparison can be produced.
    """
    if not reg.requires_ci(quantity):
        raise reg.StageFRegistryError(
            f"{quantity!r} is CI-prohibited (§D.12.4) and may not receive an interval")
    rate, acquisition_weighted = _QUANTITY_SOURCE[quantity]

    cell = [r for r in per_site
            if r["method"] == method and r["alpha"] == alpha
            and r["n_labels"] == n_labels]
    cell.sort(key=lambda r: contracts.FROZEN_SITES.index(r["held_out_site"]))
    values = np.array([np.nan if r["status"] != reg.ESTIMABLE else r[rate]
                       for r in cell], dtype=float)
    weights = (np.array([float(r["n_eval"]) for r in cell], dtype=float)
               if acquisition_weighted else None)

    n_sites = len(cell)
    cache = {} if counts_cache is None else counts_cache
    if n_sites not in cache:
        cache[n_sites] = sb.site_bootstrap_counts(n_sites)
    counts = cache[n_sites]

    base = {"quantity": quantity, "method": method, "alpha": alpha,
            "n_labels": n_labels, "ci_level": reg.CI_LEVEL, "n_sites": n_sites}
    if not np.isfinite(values).any():
        return {**base, "point_estimate": None, "ci_low": reg.NOT_ESTIMABLE,
                "ci_high": reg.NOT_ESTIMABLE,
                "ci_method": f"{int(reg.CI_LEVEL * 100)}% percentile "
                             f"(numpy.quantile method='{reg.QUANTILE_METHOD}')",
                "n_estimable_sites": 0,
                "total_bootstrap_replicates": reg.SITE_BOOTSTRAP_B,
                "valid_bootstrap_replicates": 0, "status": reg.NOT_ESTIMABLE,
                "reason_code": reg.assert_reason_code("NO_ESTIMABLE_SITE_IN_CELL")}

    summary = sb.bootstrap_summary(values, counts, weights,
                                   min_valid=reg.MIN_VALID_REPLICATES)
    # The validity floor applies to the **interval alone**; the point estimate is
    # unaffected and stays ESTIMABLE whenever it is computable (§D.12.3).
    interval_missing = summary["ci_low"] == reg.NOT_ESTIMABLE
    return {**base, "point_estimate": float(summary["point_estimate"]),
            "ci_low": summary["ci_low"], "ci_high": summary["ci_high"],
            "ci_method": summary["ci_method"],
            "n_estimable_sites": int(summary["n_estimable_sites"]),
            "total_bootstrap_replicates": int(summary["total_bootstrap_replicates"]),
            "valid_bootstrap_replicates": int(summary["valid_bootstrap_replicates"]),
            "status": reg.ESTIMABLE,
            "reason_code": (reg.assert_reason_code("CI_INSUFFICIENT_VALID_REPLICATES")
                            if interval_missing else None)}


def build_interval_registry(per_site: list) -> list:
    """All **80** registered intervals, enumerated cell by cell (§D.12.2)."""
    cache: dict = {}
    rows = [interval_for(quantity, per_site, method, alpha, n_labels, cache)
            for quantity, method, alpha, n_labels in reg.REQUIRED_INTERVALS]
    if len(rows) != reg.N_CI_SLOTS:
        raise ConformalError(
            f"the RQ4 interval registry produced {len(rows)} rows, the frozen "
            f"registry is {reg.N_CI_SLOTS}")
    return rows


# ---------------------------------------------------------------------------
# the module driver
# ---------------------------------------------------------------------------
def run(*, source_oof=None, target=None, weights=None, registry=None,
        progress=None) -> dict:
    """Every registered F-RQ4 cell. Returns the six `rq4/` tables in memory.

    Persistence, provenance and the completion barrier belong to
    `stage_f_store`; this function writes nothing.
    """
    def say(message):
        if progress is not None:
            progress(message)

    source_oof = inputs.load_source_oof_confidence() if source_oof is None else source_oof
    target = inputs.load_target_confidence() if target is None else target
    weights = inputs.load_density_weights() if weights is None else weights
    registry = inputs.load_split_registry() if registry is None else registry

    per_site, per_repeat = [], []
    for alpha in reg.ALPHA_GRID:
        say(f"rq4 source_pooled_split alpha={alpha}")
        per_site.extend(source_pooled_per_site(source_oof, target, alpha))
        say(f"rq4 weighted alpha={alpha}")
        per_site.extend(weighted_per_site(source_oof, weights, target, alpha))
    for n_labels in reg.N_GRID:
        for alpha in reg.ALPHA_GRID:
            say(f"rq4 target_local N={n_labels} alpha={alpha}")
            per_repeat.extend(target_local_per_repeat(target, registry, alpha,
                                                      n_labels))
    per_site.extend(collapse_repeats(per_repeat))

    say("rq4 aggregation")
    site_macro_rows, marginal_rows, degeneracy_rows = [], [], []
    for method, alpha, n_labels in reg.REGISTERED_CELLS:
        site_macro_rows.append(site_macro(per_site, method, alpha, n_labels))
        marginal_rows.append(marginal(per_site, method, alpha, n_labels))
        degeneracy_rows.append(degeneracy_counts(per_site, method, alpha, n_labels))

    say("rq4 site-cluster bootstrap")
    intervals = build_interval_registry(per_site)

    return {"per_site_metrics": per_site, "per_repeat_target_local": per_repeat,
            "site_macro_metrics": site_macro_rows,
            "marginal_metrics": marginal_rows,
            "site_bootstrap_intervals": intervals,
            "degeneracy": degeneracy_rows,
            "conformal_config": conformal_config()}


def conformal_config() -> dict:
    """`rq4/conformal_config.json` — the frozen constants this run deployed."""
    return {
        "protocol_version": reg.PROTOCOL_VERSION,
        "protocol_tag": reg.PROTOCOL_TAG,
        "alpha_grid": list(reg.ALPHA_GRID),
        "n_grid": list(reg.N_GRID),
        "n_excluded": [n for n in reg.S2_N_GRID if n not in reg.N_GRID],
        "nonconformity_score": reg.NONCONFORMITY_SCORE,
        "threshold_construction_rule": reg.INDEX_RULE,
        "interpolation_permitted": reg.INTERPOLATION_PERMITTED,
        "tie_inclusion": reg.TIE_INCLUSION,
        "weight_column": reg.WEIGHT_COLUMN,
        "weighted_test_point_atom_included": reg.WEIGHTED_TEST_POINT_ATOM_INCLUDED,
        "descriptive_tolerance": reg.CONFORMAL_TOLERANCE,
        "guarantee_labels": dict(reg.GUARANTEE_LABELS),
        "method_information_budget": dict(reg.METHOD_INFORMATION_BUDGET),
        "exact_guarantee_permitted_for": list(reg.EXACT_GUARANTEE_PERMITTED_FOR),
        "ci_registry": {
            "required": list(reg.F_CI_REQUIRED),
            "prohibited": list(reg.F_CI_PROHIBITED),
            "n_registered_cells": reg.N_REGISTERED_CELLS,
            "n_ci_slots": reg.N_CI_SLOTS,
            "site_bootstrap_b": reg.SITE_BOOTSTRAP_B,
            "seed_namespace": list(reg.BOOTSTRAP_ENTROPY),
            "ci_level": reg.CI_LEVEL,
            "ci_method": reg.CI_METHOD,
            "quantile_method": reg.QUANTILE_METHOD,
            "min_valid_replicates": reg.MIN_VALID_REPLICATES,
            "hypothesis_test": reg.HYPOTHESIS_TEST,
            "multiplicity_correction": reg.MULTIPLICITY_CORRECTION,
            "tost": reg.TOST,
        },
        "consumed_inputs": {p: d for p, d in sorted(reg.CONSUMED_ARTIFACTS.items())},
    }
