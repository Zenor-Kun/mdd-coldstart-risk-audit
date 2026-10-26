"""F-SENS — the six pre-registered sensitivities (`docs/STAGE_F_PROTOCOL.md` §E).

Exactly six. The set is closed and no seventh may ever be added (§A.4, §E.1).
Each changes **exactly one** factor; every other factor stays at its frozen
parent value, and none may replace, re-weight, qualify or alter a primary
conclusion of RQ2/S2 or of Stage E / RQ3.

===============================  ==========  =====================================
label                            parent      the one changed factor
===============================  ==========  =====================================
`sens1_overall_error_ucb`        RQ2/S2      one overall-error UCB, α = 0.10, no
                                             Bonferroni, instead of `U_BER`
`sens2_q_step_002`               RQ2/S2      `q` step 0.02 — 41 candidates
`sens3_rstar_pm_005`             RQ2/S2      `r*_s ± 0.05` (the rule is unchanged)
`sens4_common_site`              RQ2/S2      the 11 all-`N`-eligible sites
`sens5_s1_attribution`           RQ3         `tau_S1` replaces `tau_S0`
`sens6_realized_coverage_matched` RQ3        realized-coverage-matched `tau`
===============================  ==========  =====================================

**Implementation reuse (§E.8).** A sensitivity reuses its parent's audited
mechanism if and only if the changed factor is reachable as a parameter. The
reachability of each parent path was *verified at Stage F.2*, not asserted at
drafting time, and is recorded in `PARENT_ENTRY_POINTS`:

* SENS-5/6 change the outcome `R_i` only, and `rq3_execution.execute` takes an
  `outcome_builder` parameter — so the RQ3 GLMM, quadrature, CV, LRT, Holm and
  status machinery are invoked completely unmodified.
* SENS-1..4 run the RQ2/S2 decision + evaluation mechanism. Its **scientific**
  layer (`certification`, `shrinkage`, `s0_evaluation`, `density_ratio`) is
  parameterised in every changed factor — the candidate grid, the scoring
  functional, the risk reference and the site set are all arguments. Its
  **production artifact wrapper** (`s2_cell_worker.decide_from_frame`,
  `s2_decisions.validate_decision_artifact`) is deliberately hard-bound to the
  primary's 17-candidate grid and class-conditional bound, and Stage F persists
  tables rather than per-cell artifacts (§H.4.1), so that wrapper is not the
  reuse surface. `tests/test_stage_f_sensitivities.py` proves the parameterised
  path reproduces the parent wrappers **exactly** when handed the parent's own
  parameters — which is what makes it a parameterisation and not a fork.

Nothing here refits the disease classifier, the density-ratio model or the
shrinkage hyperparameter, and nothing here writes.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

import certification as cert
import contracts
import density_ratio as dr
import s0_evaluation as s0eval
import s2_inputs as si
import s2_splitting as spl
import shrinkage as shr
import stage_f_registry as reg

ROOT = reg.ROOT

SENS1, SENS2, SENS3, SENS4, SENS5, SENS6 = reg.SENSITIVITY_LABELS

#: §E.8 — which parent entry point each sensitivity reaches, and how.
PARENT_ENTRY_POINTS = {
    SENS1: ("certification.run_sequence_for_strategy",
            "score functional parameterised: certification.overall_error_upper"),
    SENS2: ("certification.run_sequence_for_strategy",
            "candidate grid parameterised: certification.Q_GRID_SENSITIVITY, "
            "thresholds from s0_evaluation.source_oof_thresholds / "
            "density_ratio.weighted_source_thresholds at the finer q grid"),
    SENS3: ("certification.run_sequence_for_strategy",
            "risk reference parameterised: r_star +/- 0.05"),
    SENS4: ("certification.run_sequence_for_strategy",
            "site set parameterised: the 11 common sites"),
    SENS5: ("rq3_execution.execute(outcome_builder=...)",
            "threshold table parameterised: tau_S1 replaces tau_S0"),
    SENS6: ("rq3_execution.execute(outcome_builder=...)",
            "threshold parameterised: realized-coverage-matched tau"),
}

#: The factors each sensitivity leaves at its frozen parent value. Persisted in
#: every `manifest.json` so a single-factor claim can be audited, not trusted.
UNCHANGED_FACTORS = {
    SENS1: ("candidate sets", "q grid", "sequence direction", "stop rule",
            "r_star", "c_min", "m_min", "N grid", "n0 grid", "splits",
            "all four strategies", "all three confidence methods", "100 repeats"),
    SENS2: ("direction", "stop rule", "bound", "alpha_UCB", "r_star",
            "eligibility", "splits", "strategies", "methods", "repeats"),
    SENS3: ("everything else, including the derivation of e_full,-s from "
            "outer-training sites only",),
    SENS4: ("everything else",),
    SENS5: ("disease predictions", "target confidence scores", "cohort",
            "complete-case subsets", "variable coding", "both model families",
            "GLMM numerics", "CV log-loss", "McFadden pseudo-R2", "LRT",
            "Holm within family", "alpha = 0.05"),
    SENS6: ("everything else, exactly as SENS-5",),
}

CHANGED_FACTOR = {
    SENS1: "overall-error upper confidence bound, alpha = 0.10, no Bonferroni",
    SENS2: "candidate grid: q step 0.02, 41 candidates",
    SENS3: "per-fold risk target: r*_s +/- 0.05",
    SENS4: "site set: the 11 sites eligible at all four N",
    SENS5: "abstention threshold: tau_S1 replaces tau_S0 at the same OP50 / OP30",
    SENS6: "threshold matched to realized target coverage 0.50 / 0.70",
}


class SensitivityBlocked(AssertionError):
    """A parent code path is not parameterisable for its one changed factor.

    §E.8: the sensitivity is then **blocked and reported**, never implemented by
    copying, forking or rewriting the parent — a fork would silently change a
    second factor, the code path itself.
    """


class SensitivityError(AssertionError):
    """A structural violation of a frozen sensitivity definition."""


def assert_closed_set(labels=reg.SENSITIVITY_LABELS) -> tuple:
    """Exactly six, and exactly these six (§A.4, §E.1)."""
    if tuple(labels) != reg.SENSITIVITY_LABELS:
        raise SensitivityError(
            f"the sensitivity set is closed at {reg.SENSITIVITY_LABELS}; "
            f"got {tuple(labels)}")
    if len(labels) != reg.N_SENSITIVITIES:
        raise SensitivityError("a seventh sensitivity may never be added")
    return tuple(labels)


# ===========================================================================
# SENS-1..4 — the RQ2/S2 family
# ===========================================================================
def candidate_q_grid(label: str) -> tuple:
    """The candidate grid this sensitivity deploys, in the frozen order.

    Only SENS-2 changes it, and it changes it to the parent's own registered
    `certification.Q_GRID_SENSITIVITY` — a constant frozen in the parent module
    for exactly this sensitivity, never invented here.
    """
    grid = cert.Q_GRID_SENSITIVITY if label == SENS2 else cert.Q_GRID
    if list(grid) != sorted(grid, reverse=True):
        raise SensitivityError(
            "the candidate order is lowest coverage -> highest (q descending); "
            "reordering is forbidden")
    if label == SENS2 and len(grid) != reg.SENS2_N_CANDIDATES:
        raise SensitivityError(
            f"SENS-2 must run {reg.SENS2_N_CANDIDATES} candidates, got {len(grid)}")
    return tuple(grid)


def site_set(label: str) -> tuple:
    """The eligible sites. Only SENS-4 changes them, to the 11 common sites."""
    return reg.COMMON_SITES if label == SENS4 else contracts.FROZEN_SITES


def r_star_offsets(label: str) -> tuple:
    """`{0.0}` for every sensitivity but SENS-3, which runs `{-0.05, +0.05}`."""
    return reg.SENS3_R_STAR_OFFSETS if label == SENS3 else (0.0,)


# --- the candidate threshold grids -----------------------------------------
def _shared_q(fine, coarse) -> list:
    return [q for q in fine if any(abs(q - c) < 1e-12 for c in coarse)]


def derive_threshold_grids(label: str, blocks=None,
                           s0_confidence=None) -> dict:
    """`{(site, method): taus}` for the S0 and S1 candidate grids.

    For every sensitivity but SENS-2 these are read straight from the frozen S0
    and S1 threshold tables — no derivation at all.

    SENS-2's finer grid is not persisted anywhere: the frozen tables carry the
    17 primary `q` only. The 41 thresholds are therefore produced by the
    **parent's own frozen threshold functions**, parameterised by `q_grid` —
    `s0_evaluation.source_oof_thresholds` and
    `density_ratio.weighted_source_thresholds` — applied to the frozen source-OOF
    confidence scores and the frozen `w_final` weights. Nothing is refitted.

    The two grids have **two different parents**, so they take their confidences
    from two different reads of the one frozen file, each its own parent's:

    * `tau_S0` is the RQ1/S0 mechanism, and `run_rq1_s0.py:771` reads the table
      with pandas' default float parser — so the S0 branch consumes
      `stage_f_inputs.parent_s0_threshold_confidence()`;
    * `tau_S1` is the RQ1/S1 mechanism, which reads round-trip
      (`s1_data_access._read_rows`) — so the S1 branch keeps consuming the
      round-trip confidences the `blocks` object already carries, alongside the
      frozen `w_final`.

    Feeding either branch the other's read moves the derived threshold by one to
    four ULP and the sensitivity blocks. That is the guard working, not a
    tolerance to be widened.

    The derivation is then **bound to the frozen tables**: at the 9 `q` the two
    grids share, the derived threshold must equal the frozen one exactly. A
    single mismatch blocks the sensitivity rather than shipping a threshold grid
    that the primary would not have produced.
    """
    if label != SENS2:
        return {"S0": si.source_threshold_grid(), "S1": si.weighted_threshold_grid(),
                "derived": False, "q_grid": cert.Q_GRID}

    import stage_f_inputs as inputs

    blocks = si.SourceOofBlocks() if blocks is None else blocks
    if s0_confidence is None:
        s0_confidence = inputs.parent_s0_threshold_confidence()
    grid = candidate_q_grid(label)
    frozen_s0, frozen_s1 = si.source_threshold_grid(), si.weighted_threshold_grid()
    shared = _shared_q(grid, cert.Q_GRID)

    out_s0, out_s1, mismatches = {}, {}, []
    for site in contracts.FROZEN_SITES:
        block = blocks.block(site)
        for method in cert.CONFIDENCE_METHODS:
            conf = np.asarray(block["confidence"][method], dtype=float)
            weights = np.asarray(block["weight"], dtype=float)
            conf_s0 = np.asarray(s0_confidence[(site, method)], dtype=float)
            if conf_s0.shape != conf.shape:
                raise SensitivityError(
                    f"fold {site}, method {method}: the RQ1/S0 threshold read "
                    f"holds {conf_s0.size} rows and the S2 calibration block "
                    f"{conf.size}; they are not the same frozen fold")
            derived_s0 = s0eval.source_oof_thresholds(conf_s0, q_grid=grid)
            derived_s1 = dr.weighted_source_thresholds(conf, weights, q_grid=grid)
            out_s0[(site, method)] = tuple(float(derived_s0[q]) for q in grid)
            out_s1[(site, method)] = tuple(float(derived_s1[q]) for q in grid)
            for name, derived, frozen in (("S0", derived_s0, frozen_s0),
                                          ("S1", derived_s1, frozen_s1)):
                reference = dict(zip(cert.Q_GRID, frozen[(site, method)]))
                for q in shared:
                    if float(derived[q]) != float(reference[q]):
                        mismatches.append((name, site, method, q))
    if mismatches:
        raise SensitivityBlocked(
            f"{SENS2} is blocked: the parent threshold mechanism re-derived "
            f"{len(mismatches)} of {len(shared) * 2 * len(cert.CONFIDENCE_METHODS) * len(contracts.FROZEN_SITES)} "
            f"shared thresholds differently from the frozen tables, e.g. "
            f"{mismatches[:3]}. A sensitivity may not deploy a candidate grid "
            "the primary would not have produced (§E.8)")
    return {"S0": out_s0, "S1": out_s1, "derived": True, "q_grid": grid,
            "n_shared_q_verified": len(shared)}


# --- the scoring functionals -----------------------------------------------
def persisted_score_type(label: str, strategy: str) -> str:
    """The `score_type` a decision row must persist — §E.2a, metadata only.

    The parent fixes `score_type` per **strategy**
    (`certification.STRATEGY_SCORE_TYPE`), and a sensitivity inherits that
    vocabulary unchanged. SENS-1 is the one registered exception, and it is a
    *label* correction rather than a definition change: SENS-1 replaces the
    class-conditional `U_BER = ½(U_MDD + U_HC)` at `α = 0.05` per class with the
    single overall-error bound `certification.overall_error_upper` at
    `α = 0.10`, so the rows that ran that arithmetic must say so. Attempt #5
    inherited the parent's `exact_classwise_clopper_pearson_ucb` verbatim, and a
    persisted label that contradicts the persisted computation is a provenance
    defect regardless of its numerical harmlessness.

    **No number changes.** The functional itself is chosen in
    `score_functional` / `decide_zero_label`, and this function only names what
    that choice was; the two are driven by the same condition so the label can
    never describe a functional that did not run. Rows produced by a genuinely
    class-conditional functional — every non-SENS-1 row, and SENS-1's own
    empirical tiers — keep the parent's own label.
    """
    inherited = cert.STRATEGY_SCORE_TYPE[strategy]
    if label == SENS1 and uses_overall_error_bound(strategy):
        return reg.SENS1_OVERALL_ERROR_SCORE_TYPE
    return inherited


def uses_overall_error_bound(strategy: str) -> bool:
    """Whether SENS-1's changed factor actually reaches this strategy (§E.2).

    SENS-1 changes the **exact** risk bound. The two exact-CP tiers therefore
    swap their bound; the empirical tiers (`pooled_local_shrinkage`'s
    pseudocount score, `target_unlabeled_weighted`'s ESS score) have no exact
    bound to replace and keep their own screening score, so their vocabulary is
    preserved. This is the same condition the two score-functional builders
    branch on, expressed once so the label and the arithmetic cannot diverge.
    """
    return cert.STRATEGY_SCORE_TYPE[strategy] == cert.SCORE_EXACT_CP


def score_functional(label: str, strategy: str, counts, scores=None):
    """The candidate score `score_fn(i)`, delegated entirely to the parent.

    The **only** thing SENS-1 changes is which parent bound is called: the
    class-conditional `U_BER = ½(U_MDD + U_HC)` with `α = 0.05` per class
    becomes the single overall-error bound `certification.overall_error_upper`
    at `α = 0.10` with no Bonferroni. That function is itself a frozen parent
    constant, written in `certification.py` for exactly this sensitivity
    (`RQ2_S2_PROTOCOL.md` §23.1) — Stage F does not define a bound of its own.
    """
    if label == SENS1:
        if strategy in cert.LABEL_CONSUMING_STRATEGIES and strategy != "target_local":
            # The empirical tiers keep their own screening score: SENS-1 changes
            # the *risk bound*, and the shrinkage/weighted arms have no exact
            # bound to replace. Their vocabulary is preserved (§E.2).
            return lambda i: cert.mean_class_score(scores[i][0]["score"],
                                                   scores[i][1]["score"])

        def overall(i):
            c = counts[i]
            return cert.overall_error_upper(int(c["x_MDD"]) + int(c["x_HC"]),
                                            int(c["n_keep"]),
                                            alpha=reg.SENS1_ALPHA_UCB)
        return overall

    if strategy == "target_local":
        def classwise(i):
            c = counts[i]
            return cert.u_ber(
                cert.clopper_pearson_upper(c["x_MDD"], c["n_keep_MDD"]),
                cert.clopper_pearson_upper(c["x_HC"], c["n_keep_HC"]))
        return classwise
    return lambda i: cert.mean_class_score(scores[i][0]["score"],
                                           scores[i][1]["score"])


def decide_label_consuming(label: str, *, strategy: str, method: str, site: str,
                           repeat: int, n_labels: int, y, pred, confidence,
                           taus, q_values, r_star: float, source_class_counts,
                           n0: float, r_star_offset: float = 0.0) -> dict:
    """One `(site, repeat, N, strategy, method)` decision under the sensitivity.

    Every scientific step is a parent call: retained counts by
    `certification.retained_class_counts`, eligibility by
    `certification.candidate_eligible`, the shrinkage score by
    `shrinkage.empirical_pseudocount_beta_upper_score`, the bound by
    `score_functional` (also parent functions) and the fixed sequence by
    `certification.run_sequence_for_strategy` — the frozen lowest-coverage-first
    order with the first-failure stop, unchanged.
    """
    if strategy not in cert.LABEL_CONSUMING_STRATEGIES:
        raise SensitivityError(
            f"{strategy!r} consumes no target calibration label; it is decided "
            "in the target-label-free phase")
    y = np.asarray(y).ravel().astype(int)
    pred = np.asarray(pred).ravel().astype(int)
    conf = np.asarray(confidence, dtype=float).ravel()
    if len(taus) != len(q_values):
        raise SensitivityError("one tau per candidate q")

    counts, coverage, eligible, scores = [], [], [], []
    for i, tau in enumerate(taus):
        keep = conf >= float(tau)
        c = cert.retained_class_counts(y, pred, keep)
        counts.append(c)
        cov = float(keep.mean()) if keep.size else 0.0
        coverage.append(cov)
        n_ms, x_ms, n_hs, x_hs = (int(v) for v in source_class_counts[i])
        for_eligibility = ((c["n_keep_MDD"], c["n_keep_HC"])
                           if strategy == "target_local"
                           else (c["n_keep_MDD"], c["n_keep_HC"], n_ms, n_hs))
        eligible.append(cert.candidate_eligible(cov, for_eligibility))
        if strategy == "target_local":
            scores.append(None)
        else:
            scores.append((
                shr.empirical_pseudocount_beta_upper_score(
                    c["x_MDD"], c["n_keep_MDD"], (x_ms / n_ms) if n_ms else 1.0, n0),
                shr.empirical_pseudocount_beta_upper_score(
                    c["x_HC"], c["n_keep_HC"], (x_hs / n_hs) if n_hs else 1.0, n0)))

    score_fn = score_functional(label, strategy, counts, scores)
    outcome = cert.run_sequence_for_strategy(strategy, list(q_values), eligible,
                                             score_fn, float(r_star))
    return _decision_row(label, outcome, site=site, repeat=repeat,
                         n_labels=n_labels, strategy=strategy, method=method,
                         q_values=q_values, taus=taus, coverage=coverage,
                         counts=counts, r_star=r_star,
                         r_star_offset=r_star_offset, n0=n0)


def decide_zero_label(label: str, *, strategy: str, method: str, site: str,
                      taus, q_values, r_star: float, counts, coverage,
                      diagnostics=None, r_star_offset: float = 0.0) -> dict:
    """One zero-label decision (`pooled_source` / `target_unlabeled_weighted`).

    The same parent sequence mechanism; `pooled_source` keeps the exact
    class-conditional Clopper–Pearson bound on the **source** calibration sample
    and `target_unlabeled_weighted` keeps its ESS screening score, so neither
    tier's guarantee vocabulary moves (§E.1).
    """
    if strategy not in cert.ZERO_LABEL_STRATEGIES:
        raise SensitivityError(f"{strategy!r} is not a zero-label strategy")
    n = len(q_values)

    if strategy == "pooled_source":
        eligible = [cert.candidate_eligible(coverage[i], (counts[i][0], counts[i][2]))
                    for i in range(n)]
        if label == SENS1:
            def score_fn(i):
                return cert.overall_error_upper(
                    int(counts[i][1]) + int(counts[i][3]),
                    int(counts[i][0]) + int(counts[i][2]),
                    alpha=reg.SENS1_ALPHA_UCB)
        else:
            def score_fn(i):
                return cert.u_ber(
                    cert.clopper_pearson_upper(counts[i][1], counts[i][0]),
                    cert.clopper_pearson_upper(counts[i][3], counts[i][2]))
    else:
        coverage = diagnostics["weighted_coverage"]
        eligible = [cert.candidate_eligible(coverage[i],
                                            (diagnostics["n_keep_MDD"][i],
                                             diagnostics["n_keep_HC"][i]))
                    for i in range(n)]

        def score_fn(i):
            return cert.mean_class_score(diagnostics["score_MDD"][i],
                                         diagnostics["score_HC"][i])

    outcome = cert.run_sequence_for_strategy(strategy, list(q_values), eligible,
                                             score_fn, float(r_star))
    return _decision_row(label, outcome, site=site, repeat=None, n_labels=None,
                         strategy=strategy, method=method, q_values=q_values,
                         taus=taus, coverage=coverage, counts=None,
                         r_star=r_star, r_star_offset=r_star_offset, n0=None)


def _decision_row(label, outcome, *, site, repeat, n_labels, strategy, method,
                  q_values, taus, coverage, counts, r_star, r_star_offset,
                  n0) -> dict:
    """One row of a sensitivity's `decisions.csv`, in the parent's own schema."""
    sel = outcome["selected_index"]
    reason = outcome["reason"]
    if outcome["state"] == cert.NOT_TESTABLE:
        reason = ("NO_ELIGIBLE_CANDIDATE__COVERAGE"
                  if float(coverage[-1]) < cert.C_MIN
                  else "NO_ELIGIBLE_CANDIDATE__CLASS_COUNT")
    cert.assert_state_allowed(strategy, outcome["state"])
    # The parent's own convention when nothing was selected: zeroed counts, not
    # nulls, so a sensitivity row is field-compatible with a parent decision.
    empty = {"n_keep": 0, "n_keep_MDD": 0, "n_keep_HC": 0, "x_MDD": 0, "x_HC": 0}
    selected = (counts[sel] if (counts is not None and sel is not None)
                else (empty if counts is not None else None))
    return {
        "sensitivity_label": label,
        "site": str(site), "repeat": repeat, "n_labels": n_labels,
        "strategy": strategy, "method": method,
        "r_star_offset": float(r_star_offset),
        "guarantee_tier": outcome["guarantee_tier"],
        # §E.2a — metadata only. The guarantee tier is untouched: SENS-1 swaps
        # one exact Clopper-Pearson bound for another exact one on the same
        # calibration sample, so `guarantee` stays the parent's.
        "score_type": persisted_score_type(label, strategy),
        "guarantee": outcome["guarantee"],
        "state": outcome["state"], "reason": reason,
        "selected_q": (float(q_values[sel]) if sel is not None else None),
        "selected_candidate_index": (int(sel) if sel is not None else None),
        "selected_tau": (float(taus[sel]) if sel is not None else None),
        "selected_score": (float(outcome["score_by_index"][sel])
                           if sel is not None else None),
        "n_eligible": int(outcome["n_eligible"]),
        "first_eligible_index": outcome["first_eligible_index"],
        "n_passed_candidates": int(outcome["n_passed_candidates"]),
        "stopped_at_index": outcome["stopped_at_index"],
        "n_keep_cal": (selected["n_keep"] if selected else None),
        "n_keep_cal_MDD": (selected["n_keep_MDD"] if selected else None),
        "n_keep_cal_HC": (selected["n_keep_HC"] if selected else None),
        "x_cal_MDD": (selected["x_MDD"] if selected else None),
        "x_cal_HC": (selected["x_HC"] if selected else None),
        "n0": n0, "r_star": float(r_star),
    }


def evaluate_decision(label: str, decision: dict, *, y, pred, confidence,
                      r_star: float) -> dict:
    """The realized outcome of one decision on its **disjoint** test split.

    Certification and evaluation sets never overlap: the decision saw
    `D_cal` only, and this function is handed `D_test` only. The retained
    counts and the selective BER come from the parent's own
    `certification.retained_class_counts` / `selective_ber_from_counts`, and the
    realized-outcome field name from `certification.realized_outcome_field`, so
    each strategy keeps its own tier's vocabulary.
    """
    strategy = decision["strategy"]
    tau = decision["selected_tau"]
    row = {
        "sensitivity_label": label,
        "site": decision["site"], "repeat": decision["repeat"],
        "n_labels": decision["n_labels"], "strategy": strategy,
        "method": decision["method"],
        "r_star_offset": decision["r_star_offset"],
        "guarantee_tier": decision["guarantee_tier"], "state": decision["state"],
        "selected_q": decision["selected_q"], "selected_tau": tau,
        "selected_candidate_index": decision["selected_candidate_index"],
        "realized_field": cert.realized_outcome_field(strategy),
        "r_star": float(r_star),
    }
    if tau is None:
        row.update({"coverage_test": None, "n_keep_test": 0,
                    "n_keep_test_MDD": 0, "n_keep_test_HC": 0, "ber_test": None,
                    "risk_excess": None, "realized_outcome": None})
        return row
    y = np.asarray(y).ravel().astype(int)
    pred = np.asarray(pred).ravel().astype(int)
    keep = np.asarray(confidence, dtype=float).ravel() >= float(tau)
    counts = cert.retained_class_counts(y, pred, keep)
    ber = cert.selective_ber_from_counts(counts)
    row.update({
        "coverage_test": float(keep.mean()) if keep.size else 0.0,
        "n_keep_test": counts["n_keep"],
        "n_keep_test_MDD": counts["n_keep_MDD"],
        "n_keep_test_HC": counts["n_keep_HC"],
        "ber_test": (None if ber == cert.NOT_ESTIMABLE else float(ber)),
        "risk_excess": (None if ber == cert.NOT_ESTIMABLE
                        else float(ber) - float(r_star)),
        "realized_outcome": (None if ber == cert.NOT_ESTIMABLE
                             else int(float(ber) > float(r_star))),
    })
    return row


def expected_cardinality(label: str) -> dict:
    """The frozen §H.4.2 cardinalities, re-derived from the split registry.

    `|REG| · S_few · M` decisions and `|REG| · S_few · M + |REG≥10| · S_zero · M`
    evaluations, doubled for SENS-3's two offsets and restricted to the common
    sites for SENS-4. The values are consequences of the frozen registry, not
    assertions.
    """
    s_few = len(reg.PARENT_STRATEGIES_LABEL_CONSUMING)
    s_zero = len(reg.PARENT_STRATEGIES_ZERO_LABEL)
    m = len(reg.PARENT_CONFIDENCE_METHODS)
    if label == SENS4:
        n_reg, n_reg_10 = (reg.COMMON_SITE_REGISTRY_ROWS,
                           reg.COMMON_SITE_REGISTRY_ROWS_N_GE_10)
    else:
        n_reg, n_reg_10 = reg.SPLIT_REGISTRY_ROWS, reg.SPLIT_REGISTRY_ROWS_N_GE_10
    scale = len(r_star_offsets(label))
    return {"decisions": n_reg * s_few * m * scale,
            "evaluations": (n_reg * s_few * m + n_reg_10 * s_zero * m) * scale}


def assert_expected_cardinality(label: str) -> dict:
    """Bind the derived cardinality to the frozen registry value."""
    derived = expected_cardinality(label)
    key = label.split("_")[0]
    frozen = {"decisions": reg.CARDINALITY[f"{key}_decisions"],
              "evaluations": reg.CARDINALITY[f"{key}_evaluations"]}
    if derived != frozen:
        raise SensitivityError(
            f"{label}: derived cardinality {derived} != frozen {frozen}")
    return derived


# ===========================================================================
# SENS-6 — the realized-coverage-matched threshold selector (§E.7.1)
# ===========================================================================
def select_realized_coverage_threshold(confidence, c_star: float) -> dict:
    """`tau = min { t in T : cov(t) <= c* }`, exactly as frozen.

    `T` is the **distinct observed confidence values of the group, ascending**;
    `cov(t) = #{i : conf_i >= t} / n`. Because every distinct value has
    multiplicity ≥ 1, `cov` is strictly decreasing over `T`, so no two candidates
    share a coverage and there is nothing to tie-break. Rows carrying an
    identical confidence value always move together and are never split. No
    grid, no interpolation, no randomisation, no jitter.

    **Label-free by signature**: this function takes a confidence vector and a
    requested coverage, and there is no parameter through which `y_true`,
    `prob_mdd`, `pred`, the abstention outcome or any performance quantity could
    reach it (§E.7.1). `assert_selector_is_label_free` checks that mechanically.
    """
    conf = np.asarray(confidence, dtype=float).ravel()
    n = int(conf.size)
    if n == 0:
        return {"tau_selected": None, "n_rows": 0, "n_distinct_conf": 0,
                "requested_coverage": float(c_star), "realized_coverage": None,
                "coverage_gap": None, "status": reg.NOT_ESTIMABLE,
                "reason_code": reg.assert_reason_code("NO_EVALUATION_POINTS")}
    if not np.isfinite(conf).all():
        raise SensitivityError("a confidence value is not finite")

    candidates = np.unique(conf)                       # ascending, distinct
    coverage = np.array([float((conf >= t).sum()) / n for t in candidates])
    if candidates.size > 1 and not np.all(np.diff(coverage) < 0):
        raise SensitivityError(
            "cov(t) is not strictly decreasing over the distinct candidate set; "
            "the candidate set was not deduplicated")

    feasible = np.flatnonzero(coverage <= float(c_star) + 0.0)
    if feasible.size == 0:
        return {"tau_selected": None, "n_rows": n,
                "n_distinct_conf": int(candidates.size),
                "requested_coverage": float(c_star), "realized_coverage": None,
                "coverage_gap": None, "status": reg.NOT_ESTIMABLE,
                "reason_code": reg.assert_reason_code("COVERAGE_MATCH_UNATTAINABLE")}
    idx = int(feasible[0])                             # min over an up-set
    return {"tau_selected": float(candidates[idx]), "n_rows": n,
            "n_distinct_conf": int(candidates.size),
            "requested_coverage": float(c_star),
            "realized_coverage": float(coverage[idx]),
            "coverage_gap": float(coverage[idx] - float(c_star)),
            "status": reg.ESTIMABLE, "reason_code": None}


#: Parameter names that would let a diagnosis label, a prediction or a
#: performance quantity reach the SENS-6 selector.
_LABEL_SHAPED = ("y", "y_true", "label", "labels", "diagnosis", "pred",
                 "prediction", "prob", "prob_mdd", "outcome", "r", "risk",
                 "bacc", "ber", "performance")


def assert_selector_is_label_free(selector=select_realized_coverage_threshold) -> tuple:
    """The selection function receives no label column (§E.7.1, mandatory)."""
    names = tuple(inspect.signature(selector).parameters)
    offenders = [p for p in names if p.lower() in _LABEL_SHAPED]
    if offenders:
        raise SensitivityError(
            f"the SENS-6 threshold selector accepts {offenders}; no target "
            "diagnosis label may influence tau")
    return names


def verify_worked_examples() -> list:
    """Re-derive the four frozen §E.7.2 examples, ties included.

    Arithmetic demonstrations of determinism, not results: they use no cohort
    value and touch no artifact.
    """
    checked = []
    for example in reg.SENS6_WORKED_EXAMPLES:
        got = select_realized_coverage_threshold(example["conf"], example["c_star"])
        if example["tau"] is None:
            if got["status"] != reg.NOT_ESTIMABLE or \
                    got["reason_code"] != example["reason_code"]:
                raise SensitivityError(
                    f"worked example {example} did not reproduce: {got}")
        else:
            if abs(got["tau_selected"] - example["tau"]) > 1e-12 or \
                    abs(got["realized_coverage"] - example["realized_coverage"]) > 1e-12:
                raise SensitivityError(
                    f"worked example {example} did not reproduce: {got}")
        checked.append({**example, "reproduced": True})
    return checked


def sens6_threshold_table(populations: dict, target: pd.DataFrame) -> list:
    """One threshold per `(population, held_out_site, method, operating_point)`.

    Exactly the granularity at which the parent's frozen `tau[s, M, q(OP)]` is
    defined: the site axis is preserved and no threshold is shared across sites.
    Realized coverage is computed on **the parent's own analysis rows** — the
    complete-case subset the GLMM consumes — which is what makes the
    sensitivity's wording ("matched to realized coverage 50 % / 70 %") literally
    true of the analysed data.
    """
    confidence = target.set_index(target["canonical_id"].astype(str))
    rows = []
    for population_name, population in sorted(populations.items()):
        frame = population.frame
        for site in [s for s in contracts.FROZEN_SITES
                     if s in set(frame["site"].astype(str))]:
            ids = frame.loc[frame["site"].astype(str) == site,
                            "canonical_id"].astype(str).tolist()
            for method in reg.PARENT_CONFIDENCE_METHODS:
                conf = confidence.loc[ids, f"confidence_{method}"].to_numpy(float)
                for operating_point in reg.SENS6_OPERATING_POINTS:
                    c_star = reg.SENS6_REQUESTED_COVERAGE[operating_point]
                    selection = select_realized_coverage_threshold(conf, c_star)
                    rows.append({"population": population_name,
                                 "held_out_site": site, "method": method,
                                 "operating_point": operating_point,
                                 **selection})
    if len(rows) != reg.CARDINALITY["sens6_threshold_selection_rows"]:
        raise SensitivityError(
            f"the SENS-6 threshold table holds {len(rows)} rows, the frozen "
            f"cardinality is {reg.CARDINALITY['sens6_threshold_selection_rows']}")
    return rows


# ===========================================================================
# SENS-5/6 — the RQ3 family, through the parent's `outcome_builder` parameter
# ===========================================================================
def s1_threshold_lookup() -> dict:
    """`{(site, method, OP): tau_S1}` from the frozen weighted-threshold table.

    The same table shape the parent reads for `tau_S0`, with `tau_S1` in place
    of `tau` and the operating point taken from the file's own
    `operating_point` column — `q` is never re-derived (§2.2 of the parent).
    """
    frame = pd.read_csv(ROOT / "results/rq1_s1/thresholds_weighted_source.csv",
                        float_precision="round_trip")
    for column in ("held_out_site", "method", "tau_S1", "operating_point"):
        if column not in frame.columns:
            raise SensitivityError(
                f"the frozen S1 threshold table lacks {column!r}; SENS-5 cannot "
                "substitute tau_S1 without changing a second factor")
    scoped = frame[frame["operating_point"].isin(reg.SENS6_OPERATING_POINTS)]
    lookup = {}
    for row in scoped.itertuples():
        key = (str(row.held_out_site), str(row.method), str(row.operating_point))
        if key in lookup:
            raise SensitivityError(f"duplicate S1 threshold for {key}")
        lookup[key] = float(row.tau_S1)
    expected = (len(contracts.FROZEN_SITES) * len(reg.PARENT_CONFIDENCE_METHODS)
                * len(reg.SENS6_OPERATING_POINTS))
    if len(lookup) != expected:
        raise SensitivityError(
            f"the S1 threshold table resolves {len(lookup)} (site, method, OP) "
            f"thresholds, not {expected}")
    return lookup


def _outcome_from_thresholds(population, operating_point: str, method: str,
                             tau_by_site: dict, target: pd.DataFrame):
    """`R_i = 1{confidence_M(i) < tau[s(i), M, OP]}`, aligned to the population.

    The comparison direction is the parent's and is **not** altered: retention
    is `conf >= tau`, abstention is `conf < tau`.
    """
    import rq3_data as data

    confidence = target.set_index(target["canonical_id"].astype(str))
    ids = [str(cid) for cid in population.canonical_ids]
    conf = confidence.loc[ids, f"confidence_{method}"].to_numpy(dtype=float)
    sites = confidence.loc[ids, "held_out_site"].astype(str).to_numpy()
    tau = np.array([tau_by_site[(s, method, operating_point)]
                    if (s, method, operating_point) in tau_by_site else np.nan
                    for s in sites], dtype=float)
    if not np.isfinite(tau).all():
        raise SensitivityError(
            "a population row has no substituted threshold for its site")
    return data.CellOutcome(population.name, operating_point, method,
                            (conf < tau).astype(float))


def sens5_outcome_builder(target: pd.DataFrame | None = None):
    """The SENS-5 `outcome_builder`: `tau_S1` in place of `tau_S0`.

    S1 alters **only** `tau` (`RQ1_S1_PROTOCOL.md` §§611–643), so substituting
    the threshold table and changing nothing else is exactly one factor. The
    disease predictions, the target confidence scores, the covariates, the
    complete-case subsets, the GLMM numerics, the CV scheme, the LRT, `α = 0.05`
    and the Holm procedure are all the parent's, untouched.
    """
    import stage_f_inputs as inputs

    frame = inputs.load_target_confidence() if target is None else target
    lookup = s1_threshold_lookup()

    def builder(population, operating_point: str, method: str):
        return _outcome_from_thresholds(population, operating_point, method,
                                        lookup, frame)
    return builder


def sens6_outcome_builder(threshold_rows: list, target: pd.DataFrame | None = None):
    """The SENS-6 `outcome_builder`: the realized-coverage-matched `tau`.

    A group whose threshold is `NOT_ESTIMABLE` is **excluded from the fit**,
    counted and reported — never imputed, never back-filled from the parent's
    `tau`, and never replaced by the nearest attainable coverage (§E.7.1).
    """
    import rq3_data as data
    import stage_f_inputs as inputs

    frame = inputs.load_target_confidence() if target is None else target
    confidence = frame.set_index(frame["canonical_id"].astype(str))
    selected = {(r["population"], r["held_out_site"], r["method"],
                 r["operating_point"]): r for r in threshold_rows}

    def builder(population, operating_point: str, method: str):
        ids = [str(cid) for cid in population.canonical_ids]
        conf = confidence.loc[ids, f"confidence_{method}"].to_numpy(dtype=float)
        sites = confidence.loc[ids, "held_out_site"].astype(str).to_numpy()
        R = np.full(conf.shape, np.nan, dtype=float)
        for site in np.unique(sites):
            row = selected.get((population.name, str(site), method, operating_point))
            if row is None:
                raise SensitivityError(
                    f"no SENS-6 threshold selected for "
                    f"{(population.name, site, method, operating_point)}")
            mask = sites == site
            if row["status"] == reg.NOT_ESTIMABLE:
                continue                      # excluded from the fit, not imputed
            R[mask] = (conf[mask] < float(row["tau_selected"])).astype(float)
        return data.CellOutcome(population.name, operating_point, method, R)
    return builder


def assert_rq3_outcome_builder_is_parameterised() -> dict:
    """§E.8 reachability probe for SENS-5/6, verified at implementation time.

    `rq3_execution.execute` must expose `outcome_builder` as a parameter; if it
    did not, both RQ3 sensitivities would be blocked rather than forked.
    """
    import rq3_execution as execution

    names = tuple(inspect.signature(execution.execute).parameters)
    if "outcome_builder" not in names:
        raise SensitivityBlocked(
            "rq3_execution.execute exposes no `outcome_builder` parameter; "
            "SENS-5 and SENS-6 are BLOCKED (§E.8) and may not be implemented by "
            "forking the RQ3 mechanism")
    return {"entry_point": "rq3_execution.execute", "parameter": "outcome_builder",
            "reachable": True}


def assert_s2_mechanism_is_parameterised() -> dict:
    """§E.8 reachability probe for SENS-1..4, verified at implementation time.

    The four changed factors must each be an argument of the parent's frozen
    scientific mechanism. They are: `run_sequence_for_strategy` takes the
    candidate grid, the score functional and the risk reference; the site set is
    which cells are enumerated; and `Q_GRID_SENSITIVITY` / `overall_error_upper`
    are parent constants frozen for these sensitivities.
    """
    sequence = tuple(inspect.signature(cert.run_sequence_for_strategy).parameters)
    for required in ("q_values", "score_fn", "reference"):
        if required not in sequence:
            raise SensitivityBlocked(
                f"certification.run_sequence_for_strategy exposes no {required!r} "
                "parameter; SENS-1..4 are BLOCKED (§E.8)")
    for name in ("Q_GRID_SENSITIVITY", "overall_error_upper"):
        if not hasattr(cert, name):
            raise SensitivityBlocked(
                f"certification.{name} is absent; the parent no longer registers "
                "this sensitivity's changed factor and SENS-1/2 are BLOCKED")
    if len(cert.Q_GRID_SENSITIVITY) != reg.SENS2_N_CANDIDATES:
        raise SensitivityBlocked(
            f"certification.Q_GRID_SENSITIVITY holds "
            f"{len(cert.Q_GRID_SENSITIVITY)} candidates, not "
            f"{reg.SENS2_N_CANDIDATES}")
    if cert.ALPHA_OVERALL != reg.SENS1_ALPHA_UCB:
        raise SensitivityBlocked(
            f"certification.ALPHA_OVERALL is {cert.ALPHA_OVERALL}, the frozen "
            f"SENS-1 level is {reg.SENS1_ALPHA_UCB}")
    for name in ("source_oof_thresholds",):
        if "q_grid" not in inspect.signature(getattr(s0eval, name)).parameters:
            raise SensitivityBlocked(
                f"s0_evaluation.{name} exposes no q_grid parameter; SENS-2 is "
                "BLOCKED")
    if "q_grid" not in inspect.signature(dr.weighted_source_thresholds).parameters:
        raise SensitivityBlocked(
            "density_ratio.weighted_source_thresholds exposes no q_grid "
            "parameter; SENS-2 is BLOCKED")
    return {"entry_point": "certification.run_sequence_for_strategy",
            "parameters": ["q_values", "score_fn", "reference"],
            "reachable": True}


def reachability_report() -> dict:
    """Both §E.8 probes, run together. A block is reported, never worked around."""
    return {"s2_family": assert_s2_mechanism_is_parameterised(),
            "rq3_family": assert_rq3_outcome_builder_is_parameterised(),
            "entry_points": {k: list(v) for k, v in PARENT_ENTRY_POINTS.items()}}


def manifest(label: str, **extra) -> dict:
    """One sensitivity's `manifest.json` — parent, changed factor, unchanged set."""
    parent = "RQ3 / Stage E" if label in (SENS5, SENS6) else "RQ2/S2"
    entry, how = PARENT_ENTRY_POINTS[label]
    return {
        "sensitivity_label": label,
        "protocol_version": reg.PROTOCOL_VERSION,
        "protocol_tag": reg.PROTOCOL_TAG,
        "parent": parent,
        "changed_factor": CHANGED_FACTOR[label],
        "n_changed_factors": 1,
        "unchanged_factors": list(UNCHANGED_FACTORS[label]),
        "parent_entry_point": entry,
        "parameterisation": how,
        "forked": False,
        "standing": "never primary; may not replace, re-weight, qualify or "
                    "alter a primary conclusion",
        "guarantee_vocabulary": ("parent-inherited; no sensitivity reintroduces "
                                 "a shared certification vocabulary"),
        "carries_confidence_interval": False,
        "consumed_inputs": {p: d for p, d in sorted(reg.CONSUMED_ARTIFACTS.items())},
        **extra,
    }


# ===========================================================================
# the SENS-1..4 driver
# ===========================================================================
def weighted_candidate_diagnostics(block: dict, method: str, taus) -> dict:
    """Per-candidate weighted screening inputs, at an arbitrary candidate grid.

    Every quantity comes from a parent function — `retained_class_counts` for
    the retained set and `empirical_ess_upper_score` for the score — so this is
    the parent's own derivation with the grid lifted to a parameter. The frozen
    `WeightedCandidateInputs` does the same thing at the primary's 17-candidate
    grid and additionally reproduces `weighted_source_risk.csv`; that grid is
    outside SENS-2's changed factor, so agreement with it is asserted by
    `tests/test_stage_f_sensitivities.py` rather than re-derived here.

    Nothing is refitted and no `C_domain` is re-selected: the deployed `w_final`
    weights are read exactly as persisted.
    """
    y, pred, w = block["y"], block["pred"], block["weight"]
    conf = block["confidence"][method]
    wrong = pred != y
    is_mdd, is_hc = y == 1, y == 0
    w_total = float(w.sum())
    if w_total <= 0:
        raise SensitivityError("the frozen weight sum is not positive")

    out = {k: [] for k in ("n_keep", "n_keep_MDD", "n_keep_HC",
                           "weighted_coverage", "unweighted_coverage",
                           "score_MDD", "score_HC", "n_eff_MDD", "n_eff_HC",
                           "ess_ratio_MDD", "ess_ratio_HC")}
    for tau in taus:
        keep = conf >= float(tau)
        counts = cert.retained_class_counts(y, pred, keep)
        out["n_keep"].append(int(counts["n_keep"]))
        out["n_keep_MDD"].append(int(counts["n_keep_MDD"]))
        out["n_keep_HC"].append(int(counts["n_keep_HC"]))
        out["weighted_coverage"].append(float(w[keep].sum() / w_total))
        out["unweighted_coverage"].append(float(keep.mean()) if keep.size else 0.0)
        for label, mask in (("MDD", is_mdd), ("HC", is_hc)):
            scored = cert.empirical_ess_upper_score(w[keep & mask],
                                                    wrong[keep & mask])
            out[f"score_{label}"].append(float(scored["score"]))
            out[f"n_eff_{label}"].append(float(scored["n_eff"]))
            out[f"ess_ratio_{label}"].append(float(scored["ess_ratio"]))
    return out


def target_lookup(target: pd.DataFrame) -> dict:
    """A positional index over the frozen target table, built once per run.

    Keyed by `canonical_id`, holding the label, the frozen prediction and the
    three confidence vectors as plain arrays. Building it once rather than
    re-indexing inside the 46 200-cell loop is a performance decision only; the
    values are the frozen table's, untouched.
    """
    ids = target["canonical_id"].astype(str).to_numpy()
    position = {cid: i for i, cid in enumerate(ids)}
    if len(position) != len(ids):
        raise SensitivityError("the target table holds a duplicate canonical_id")
    return {
        "position": position,
        "y": target["y_true"].to_numpy(int),
        "pred": target["pred"].to_numpy(int),
        "confidence": {m: target[f"confidence_{m}"].to_numpy(float)
                       for m in reg.PARENT_CONFIDENCE_METHODS},
    }


def _cell_frames(lookup: dict, cell: dict, method: str) -> dict:
    """The calibration and test partitions of one registered cell.

    `D_cal` and `D_test` are disjoint by the registry's own construction and are
    re-checked here: a decision may never see an evaluation row.
    """
    cal_ids = [str(i) for i in cell["cal_ids"]]
    test_ids = [str(i) for i in cell["test_ids"]]
    if set(cal_ids) & set(test_ids):
        raise SensitivityError(
            f"cell {(cell['site'], cell['repeat'], cell['n_labels'])} is not "
            "a disjoint partition")
    missing = [i for i in cal_ids + test_ids if i not in lookup["position"]]
    if missing:
        raise SensitivityError(
            f"{len(missing)} registered canonical_ids are absent from the "
            f"frozen target table, e.g. {missing[:3]}")
    cal = np.array([lookup["position"][i] for i in cal_ids], dtype=int)
    test = np.array([lookup["position"][i] for i in test_ids], dtype=int)
    conf = lookup["confidence"][method]
    return {
        "cal": {"y": lookup["y"][cal], "pred": lookup["pred"][cal],
                "conf": conf[cal]},
        "test": {"y": lookup["y"][test], "pred": lookup["pred"][test],
                 "conf": conf[test]},
    }


def run_s2_sensitivity(label: str, *, blocks=None, registry=None, target=None,
                       progress=None) -> dict:
    """One RQ2/S2 sensitivity: `decisions.csv` and `evaluations.csv`.

    The single changed factor enters as a parameter and **nothing else moves**:
    the candidate order, the eligibility rule, the stop rule, the splits, the
    100 repeats, the four strategies, the three confidence methods and every
    guarantee tier are the parent's. Point estimates only — all four RQ2
    sensitivities are CI-prohibited **by name** in
    `s2_aggregation.S2_CI_PROHIBITED` and in §D.12.4.
    """
    import stage_f_inputs as inputs

    if label not in (SENS1, SENS2, SENS3, SENS4):
        raise SensitivityError(f"{label!r} is not an RQ2/S2 sensitivity")

    def say(message):
        if progress is not None:
            progress(message)

    assert_s2_mechanism_is_parameterised()
    expected = assert_expected_cardinality(label)

    blocks = si.SourceOofBlocks() if blocks is None else blocks
    registry = inputs.load_split_registry() if registry is None else registry
    target = inputs.load_target_confidence() if target is None else target

    lookup = target_lookup(target)
    grids = derive_threshold_grids(label, blocks)
    q_values = tuple(grids["q_grid"])
    r_star_base = si.derive_r_star()["r_star"]
    sites = site_set(label)
    frame = registry.frame if hasattr(registry, "frame") else registry._frame
    scoped = frame[frame["site"].astype(str).isin(sites)]

    say(f"{label}: n0 selection")
    n0_by_site = {s: float(si.select_n0_for_fold(blocks, grids["S0"], s)["n0"])
                  for s in sites}

    say(f"{label}: source-side candidate curves")
    source_counts, weighted_diag = {}, {}
    for site in sites:
        block = blocks.block(site)
        for method in reg.PARENT_CONFIDENCE_METHODS:
            source_counts[(site, method)] = si.retained_source_counts(
                block, method, grids["S0"][(site, method)])
            weighted_diag[(site, method)] = weighted_candidate_diagnostics(
                block, method, grids["S1"][(site, method)])

    decisions, evaluations = [], []
    for offset in r_star_offsets(label):
        for site in sites:
            r_star = float(r_star_base[site]) + float(offset)
            say(f"{label}: {site} (r* offset {offset:+.2f})")

            zero_label = {}
            for method in reg.PARENT_CONFIDENCE_METHODS:
                counts = source_counts[(site, method)]
                zero_label[("pooled_source", method)] = decide_zero_label(
                    label, strategy="pooled_source", method=method, site=site,
                    taus=grids["S0"][(site, method)], q_values=q_values,
                    r_star=r_star, counts=counts["source_class_counts"],
                    coverage=counts["coverage"], r_star_offset=offset)
                zero_label[("target_unlabeled_weighted", method)] = decide_zero_label(
                    label, strategy="target_unlabeled_weighted", method=method,
                    site=site, taus=grids["S1"][(site, method)],
                    q_values=q_values, r_star=r_star, counts=None,
                    coverage=None, diagnostics=weighted_diag[(site, method)],
                    r_star_offset=offset)
            decisions.extend(zero_label.values())

            for row in scoped[scoped["site"].astype(str) == site].itertuples():
                cell = registry.cell(site, int(row.repeat), int(row.n_labels))
                for method in reg.PARENT_CONFIDENCE_METHODS:
                    partitions = _cell_frames(lookup, cell, method)
                    for strategy in reg.PARENT_STRATEGIES_LABEL_CONSUMING:
                        grid_name = si.STRATEGY_THRESHOLD_GRID[strategy]
                        taus = grids[grid_name][(site, method)]
                        decision = decide_label_consuming(
                            label, strategy=strategy, method=method, site=site,
                            repeat=int(row.repeat), n_labels=int(row.n_labels),
                            y=partitions["cal"]["y"], pred=partitions["cal"]["pred"],
                            confidence=partitions["cal"]["conf"], taus=taus,
                            q_values=q_values, r_star=r_star,
                            source_class_counts=source_counts[
                                (site, method)]["source_class_counts"],
                            n0=n0_by_site[site], r_star_offset=offset)
                        decisions.append(decision)
                        evaluations.append(evaluate_decision(
                            label, decision, y=partitions["test"]["y"],
                            pred=partitions["test"]["pred"],
                            confidence=partitions["test"]["conf"], r_star=r_star))
                    if int(row.n_labels) >= min(reg.N_GRID):
                        for strategy in reg.PARENT_STRATEGIES_ZERO_LABEL:
                            decision = dict(zero_label[(strategy, method)])
                            decision["repeat"] = int(row.repeat)
                            decision["n_labels"] = int(row.n_labels)
                            evaluations.append(evaluate_decision(
                                label, decision, y=partitions["test"]["y"],
                                pred=partitions["test"]["pred"],
                                confidence=partitions["test"]["conf"],
                                r_star=r_star))

    # Zero-label decisions are one per (site, method, strategy, offset) and are
    # not part of the frozen decisions.csv cardinality, which counts the
    # label-consuming cells only (§H.4.2).
    label_consuming = [d for d in decisions
                       if d["strategy"] in reg.PARENT_STRATEGIES_LABEL_CONSUMING]
    if len(label_consuming) != expected["decisions"]:
        raise SensitivityError(
            f"{label}: produced {len(label_consuming)} decisions, the frozen "
            f"cardinality is {expected['decisions']}")
    if len(evaluations) != expected["evaluations"]:
        raise SensitivityError(
            f"{label}: produced {len(evaluations)} evaluations, the frozen "
            f"cardinality is {expected['evaluations']}")

    return {
        "decisions.csv": label_consuming,
        "evaluations.csv": evaluations,
        "manifest.json": manifest(
            label,
            candidate_grid=list(q_values),
            n_candidates=len(q_values),
            r_star_offsets=list(r_star_offsets(label)),
            sites=list(sites),
            thresholds_derived=bool(grids["derived"]),
            n_shared_q_verified=grids.get("n_shared_q_verified"),
            zero_label_decisions=len(decisions) - len(label_consuming),
            cardinality=expected),
    }


# ===========================================================================
# the SENS-5/6 driver
# ===========================================================================
def run_rq3_sensitivity(label: str, work_root, *, outcome_builder=None,
                        populations=None, target=None, progress=None) -> dict:
    """One RQ3 sensitivity: `cells.csv`, `validation_units.csv`, `families.csv`.

    The parent's execution mechanism is invoked through its own
    `outcome_builder` parameter, so the GLMM, the mode/curvature-adaptive
    Gauss–Hermite quadrature, the K40/K60 node registry, the CV log-loss, the
    LRT, the `NOT_ESTIMABLE` semantics and the Holm procedure are the parent's
    code, unmodified. Holm stays **family-local**: it is applied within the
    sensitivity's own four families and is never pooled with the primary's, and
    it may never revise a primary Holm outcome.

    `work_root` is a non-authoritative scratch root for the parent's own record
    store, which serializes one JSON per cell. Stage F persists **tables**
    (§H.4.1), so the parent's record layout is transcribed into the three
    registered tables and the scratch store is not part of the Stage-F store.
    Stage E's conclusions are untouched by all of this.
    """
    import rq3_execution as execution
    import rq3_store as rq3_store
    import stage_f_inputs as inputs

    if label not in (SENS5, SENS6):
        raise SensitivityError(f"{label!r} is not an RQ3 sensitivity")
    assert_rq3_outcome_builder_is_parameterised()

    def say(message):
        if progress is not None:
            progress(message)

    target = inputs.load_target_confidence() if target is None else target
    populations = (execution.build_frozen_populations(progress=progress)
                   if populations is None else populations)

    threshold_rows = None
    if outcome_builder is None:
        if label == SENS5:
            outcome_builder = sens5_outcome_builder(target)
        else:
            assert_selector_is_label_free()
            threshold_rows = sens6_threshold_table(populations, target)
            outcome_builder = sens6_outcome_builder(threshold_rows, target)

    # §16.1: the caller supplies **contextual metadata only**. The RQ3 frozen
    # identity — `protocol_version` among it — is derived and persisted by the
    # RQ3 store from its own freshly verified RQ3 authority, and a caller that
    # supplies any of it is refused outright. Stage F once passed
    # `reg.PROTOCOL_VERSION` here, which is `STAGE_F_PROTOCOL/...`: not merely a
    # reserved field but the *wrong protocol's* identity. Stage F never speaks
    # for the RQ3 freeze.
    store = rq3_store.NonFormalRQ3Store.create(work_root, {
        "stage": "F.3", "run_kind": f"STAGE_F_SENSITIVITY::{label}",
        "note": ("the frozen RQ3 mechanism invoked through its outcome_builder "
                 "parameter; no RQ3 result is read, altered or replaced"),
    })
    say(f"{label}: RQ3 mechanism")
    execution.execute(store, populations, outcome_builder=outcome_builder,
                      progress=progress)

    cells = [store.get(rq3_store.CELLS_DIR, key)
             for key in sorted(store.keys(rq3_store.CELLS_DIR))]
    units = [store.get(rq3_store.VALIDATION_DIR, key)
             for key in sorted(store.keys(rq3_store.VALIDATION_DIR))]
    families = [store.get(rq3_store.FAMILIES_DIR, key)
                for key in sorted(store.keys(rq3_store.FAMILIES_DIR))]
    tables = {
        "cells.csv": [{"sensitivity_label": label, **row} for row in cells],
        "validation_units.csv": [{"sensitivity_label": label, **row}
                                 for row in units],
        "families.csv": [{"sensitivity_label": label, **row} for row in families],
    }
    assert_sens56_cardinality(tables)
    if label == SENS6:
        tables["threshold_selection.csv"] = [
            {"sensitivity_label": label, **row} for row in threshold_rows]
    tables["manifest.json"] = manifest(
        label,
        cells_logical_key=list(reg.SENS56_CELLS_LOGICAL_KEY),
        validation_units_logical_key=list(reg.SENS56_VALIDATION_UNITS_LOGICAL_KEY),
        families_logical_key=list(reg.SENS56_FAMILIES_LOGICAL_KEY),
        blocks_by_population={k: list(v) for k, v
                              in reg.SENS56_BLOCKS_BY_POPULATION.items()},
        holm_scope="family-local; never pooled with the primary family",
        parent_record_store=str(work_root))
    return tables


def assert_sens56_cardinality(tables: dict) -> dict:
    """12 cells / 24 validation units / 4 families, and `block` out of the cell key.

    `12 · 2 = 24 = rq3_protocol.N_VALIDATION_UNITS`: exactly two block-level
    validation units per cell, in every cell, in both populations. `block` is a
    validation-unit key field and never a cell key field (§H.4.3).
    """
    import rq3_protocol as proto

    expected = {
        "cells.csv": (reg.SENS56_CELLS_EXPECTED_ROWS,
                      len(reg.SENS56_POPULATIONS) * len(proto.CELL_REGISTRY)),
        "validation_units.csv": (reg.SENS56_VALIDATION_UNITS_EXPECTED_ROWS,
                                 int(proto.N_VALIDATION_UNITS)),
        "families.csv": (reg.SENS56_FAMILIES_EXPECTED_ROWS,
                         len(proto.HOLM_FAMILIES)),
    }
    for name, (frozen, derived) in expected.items():
        if frozen != derived:
            raise SensitivityError(
                f"{name}: the frozen cardinality {frozen} does not re-derive "
                f"from src/rq3_protocol ({derived})")
        rows = tables.get(name)
        if rows is not None and len(rows) != frozen:
            raise SensitivityError(
                f"{name} holds {len(rows)} rows, the frozen cardinality is {frozen}")
    if "block" in reg.SENS56_CELLS_LOGICAL_KEY:
        raise SensitivityError(
            "`block` may never be part of the 12-row cells.csv key (§H.4.3)")
    if (reg.SENS56_CELLS_EXPECTED_ROWS * reg.SENS56_VALIDATION_UNITS_PER_CELL
            != reg.SENS56_VALIDATION_UNITS_EXPECTED_ROWS):
        raise SensitivityError("12 · 2 != 24; the block axis is inconsistent")
    for population, blocks in reg.SENS56_BLOCKS_BY_POPULATION.items():
        if len(blocks) != reg.SENS56_BLOCKS_PER_POPULATION:
            raise SensitivityError(
                f"{population} carries {len(blocks)} blocks, frozen is "
                f"{reg.SENS56_BLOCKS_PER_POPULATION}")
        if tuple(blocks) != parent_blocks(population):
            raise SensitivityError(
                f"{population} blocks {blocks} != the parent's "
                f"{parent_blocks(population)}")
    return {name: frozen for name, (frozen, _) in expected.items()}


def parent_blocks(population: str) -> tuple:
    """The parent's two blocks of one population, from `rq3_protocol.BLOCK_DF`.

    `BLOCK_DF` is keyed `(population, block) -> degrees of freedom`, so the block
    axis is read off its keys rather than assumed: `full_sample -> {motion,
    diagnosis}`, `mdd_only -> {motion, clinical}` (§H.4.3).
    """
    import rq3_protocol as proto

    return tuple(block for pop, block in proto.BLOCK_DF if pop == population)
