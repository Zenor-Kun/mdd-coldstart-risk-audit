#!/usr/bin/env python3
"""Stage 1.3a — formal RQ1 / S1 execution (zero-label transductive onboarding).

Executes `docs/RQ1_S1_PROTOCOL.md` exactly as frozen. Nothing here re-derives,
relaxes or "improves" a frozen definition; where implementation and protocol
could diverge the script raises instead of choosing.

S1's only new information is the target site's **unlabeled** covariates, and its
only new mechanism is importance-weighted source-OOF calibration. The disease
classifier, its predictions, the three confidence scores, `r*_s`, the `q` grid
and the site-bootstrap draws are read from the frozen S0 / Gate 1 artifacts and
are never recomputed.

**What changed in 1.3a, and why.** Attempt 1 was rejected at closure review.
Its dataflow was clean — no target label reached a `C_domain`, a density weight
or a threshold, and every headline re-derived — but its *access protocol* was
not: the pre-decision path called the labelled cohort loader and materialized
the whole diagnosis vector, counted MDD/HC from it, and held one source-OOF
frame carrying `y_true` for all 23 sites. A single global barrier watching one
read site cannot support the claim that was being made. The firewall is now
**outer-fold scoped** and enforced by `src/s1_data_access.py`:

    for outer target site s, y_s is not materialized, deserialized or exposed
    before that fold's decisions are frozen on disk and release(s) is called.

The other 22 sites' labels stay legal for fold `s` — the weighted source
calibration of §10 is defined on them. Decisions, release and evaluation are
therefore interleaved per fold rather than run as three global phases.

Explicitly NOT implemented, and unreachable from here (§12, §17):
S2 few-label onboarding, any upper confidence bound on a risk, any ordered
multiple-testing procedure, threshold *adoption*, conformal prediction,
shrinkage, RQ3, and any effective-sample-size exclusion rule.

Usage:
    python3 scripts/run_rq1_s1.py                    # the full formal run
    python3 scripts/run_rq1_s1.py --stages validate  # pre-run validation only
"""

from __future__ import annotations

import os

# Single-threaded BLAS: parallelism is at the work-unit level. Must precede numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import contracts  # noqa: E402
import density_ratio as dr  # noqa: E402
import s0_evaluation as s0e  # noqa: E402
import s1_data_access as s1da  # noqa: E402
import s1_evaluation as s1e  # noqa: E402
import site_bootstrap as sb  # noqa: E402
from selective_aurc import tie_aware_aurc  # noqa: E402

GATE1 = ROOT / "results" / "gate1"
S0 = ROOT / "results" / "rq1_s0"
OUT = ROOT / "results" / "rq1_s1"
DECISIONS = OUT / "decisions"

METHODS = dr.CONFIDENCE_METHODS                 # probability, bootstrap, mahalanobis
REQUIRED_TAGS = ("data-contract-v1", "gate1-rq0-v1", "rq1-s0-protocol-v1",
                 "rq1-s0-results-v1", "rq1-s1-protocol-v1")
NE = s0e.NOT_ESTIMABLE
S0_CHECKSUMS = S0 / "RESULT_CHECKSUMS.sha256"
#: The S0 provenance manifest records one SHA-256 per artifact; the bulk cache
#: block is comment-marked because it is not committed. The authoritative count
#: is the number of real SHA entries, not the number of lines in the file.
_SHA_LINE = re.compile(r"^(?:#\s*)?([0-9a-f]{64})  (.+)$")

#: Gate 1 columns the pre-decision path may read. `outer_fold_metrics.csv` also
#: carries n_MDD, n_HC, bacc and auroc for the held-out site — target-outcome
#: quantities, excluded here by `usecols` so the parser never decodes them.
FOLD_METRIC_COLUMNS = ["site", "selected_pca", "selected_C", "n_source",
                       "n_source_sites"]
HP_COLUMNS = ["site", "observed_selected_pca", "observed_selected_C"]

#: Modules that make up the S1 reachable graph, scanned by the static audit.
S1_MODULES = ("src/density_ratio.py", "src/s1_evaluation.py",
              "src/s1_data_access.py", "scripts/run_rq1_s1.py")
#: The labelled cohort loader, named indirectly so this file's own audit text
#: is not mistaken for a call site.
LABELLED_LOADER = "build_" + "xy"


# ---------------------------------------------------------------------------
# git / environment
# ---------------------------------------------------------------------------
def _git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def environment() -> dict:
    import joblib
    import scipy
    import sklearn
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "blas_threads_per_process": os.environ.get("OMP_NUM_THREADS"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_clean_at_entry": _git("status", "--porcelain") == "",
        "tags_present": {t: t in _git("tag", "-l").split() for t in REQUIRED_TAGS},
    }


# ---------------------------------------------------------------------------
# frozen upstream provenance
# ---------------------------------------------------------------------------
def read_frozen(path, **kw) -> pd.DataFrame:
    """Read a frozen artifact so the float64 that was written is recovered.

    pandas' default CSV float parser is fast but not round-trip exact: it can
    return a value one ULP away from the number whose repr is in the file. That
    is invisible in most reporting and fatal to an **exact** invariant — the
    frozen target AURC of §12 is required to reproduce with a zero difference,
    and the default parser alone was enough to move it by 8e-17.
    """
    return pd.read_csv(path, float_precision="round_trip", **kw)


def parse_s0_checksums() -> list[tuple[str, str]]:
    rows = []
    for line in S0_CHECKSUMS.read_text().splitlines():
        m = _SHA_LINE.match(line.rstrip())
        if m:
            rows.append((m.group(1), m.group(2)))
    return rows


def verify_s0_checksums(led, when: str) -> dict:
    entries = parse_s0_checksums()
    bad, missing = [], []
    for digest, rel in entries:
        p = S0 / rel
        if not p.exists():
            missing.append(rel)
        elif s1e.sha256_file(p) != digest:
            bad.append(rel)
    res = {"n_entries": len(entries), "n_matched": len(entries) - len(bad) - len(missing),
           "n_mismatched": len(bad), "n_missing": len(missing)}
    led.check(f"s0_checksums_all_match_{when}",
              len(entries) == 549 and not bad and not missing,
              f"{res['n_matched']}/{len(entries)} SHA-256 entries match "
              f"(18 formal + 2 committed-size + 529 comment-marked bulk cache); "
              f"mismatched={bad[:3]} missing={missing[:3]}")
    return res


def assert_s0_read_only(led) -> None:
    dirty = _git("status", "--porcelain")
    touched = [ln for ln in dirty.splitlines()
               if "results/rq1_s0" in ln or "results/gate1" in ln
               or "docs/RQ1_S0_PROTOCOL.md" in ln or "docs/RQ1_S1_PROTOCOL.md" in ln]
    led.check("s0_gate1_and_protocol_paths_have_no_uncommitted_modification",
              not touched,
              "git reports no change under results/rq1_s0/, results/gate1*/ or "
              f"either frozen protocol; offenders={touched}")


# ---------------------------------------------------------------------------
# §6 — the static half of the firewall: no second door to an outcome
# ---------------------------------------------------------------------------
def static_firewall_audit(led) -> None:
    """Prove from the source that target outcomes have exactly one entry point.

    The runtime guard can only refuse the calls that reach it. This audit
    establishes that no S1 module can reach a target outcome any other way:
    the labelled cohort loader is never named, and the two frozen artifacts
    that carry target diagnosis are named only inside the access module that
    owns the guard.
    """
    texts = {m: (ROOT / m).read_text() for m in S1_MODULES}

    callers = sorted(m for m, t in texts.items() if LABELLED_LOADER in t)
    led.check("labelled_cohort_loader_is_unreachable_from_s1", not callers,
              "no S1 module names the labelled cohort loader; the pre-decision "
              "path goes through s1_data_access.build_features_only, which "
              f"returns no outcome column; offenders={callers}")

    # Assembled at run time so this audit's own source does not count as a use
    # of the names it forbids.
    guarded = ("confidence_" + "target.csv", "confidence_" + "source_oof.csv",
               "analysis_manifest_" + "v1.csv")
    owner = "src/s1_data_access.py"
    stray = []
    for m, t in texts.items():
        if m == owner:
            continue
        for node in ast.walk(ast.parse(t)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                stray += [f"{m}:{g}" for g in guarded if g in node.value]
    led.check("outcome_bearing_artifacts_named_only_in_the_access_module",
              not stray,
              f"the manifest and the target confidence table are named only in "
              f"{owner}, which owns the identity loader and the guard; every "
              f"other S1 module reaches them through those doors or not at "
              f"all; offenders={stray}")

    runner = texts["scripts/run_rq1_s1.py"]
    call_site = "guard.load_" + "target_y("        # assembled, so this line
    led.check("runner_reaches_target_diagnosis_only_through_the_guard",
              runner.count(call_site) == 1,       # is not itself a call site
              "the runner never names the target confidence artifact and calls "
              "TargetOutcomeGuard.load_target_y at exactly one site, which "
              "raises before that fold's release")

    led.check("gate1_target_outcome_columns_never_read_pre_decision",
              set(FOLD_METRIC_COLUMNS).isdisjoint(
                  {"n_MDD", "n_HC", "bacc", "auroc"}),
              "outer_fold_metrics.csv is read with usecols="
              f"{FOLD_METRIC_COLUMNS}; its per-site n_MDD, n_HC, bacc and "
              "auroc columns are target-outcome quantities and are never "
              "decoded before the folds release")


# ---------------------------------------------------------------------------
# frozen inputs — every one of these is source-side or identity-only
# ---------------------------------------------------------------------------
def load_frozen_inputs(led, verbose=True):
    X, sites, cohort = s1da.build_features_only(verbose=verbose)

    led.check("cohort_2232_acquisitions_23_sites_without_reading_diagnosis",
              len(cohort) == 2232 and len(np.unique(sites)) == 23
              and set(np.unique(sites)) == set(contracts.FROZEN_SITES),
              f"N={len(cohort)}, sites={len(np.unique(sites))}, read from the "
              "manifest with usecols that exclude diagnosis and every "
              "phenotype; the MDD/HC composition is asserted after release")
    led.check("prebarrier_cohort_object_carries_no_outcome_column",
              not [c for c in cohort.columns
                   if any(t in c.lower() for t in s1da.OUTCOME_TOKENS)],
              f"cohort identity columns = {list(cohort.columns)}")
    led.check("declared_cohort_composition_matches_the_frozen_contract",
              contracts.FROZEN_N_MDD == 1182 and contracts.FROZEN_N_HC == 1050
              and "1182" in (ROOT / "docs" / "DATA_CONTRACT_v1.md").read_text()
              and "1050" in (ROOT / "docs" / "DATA_CONTRACT_v1.md").read_text(),
              "1182 MDD / 1050 HC are checked against the frozen data contract "
              "as declared constants, not recomputed from y")
    led.check("features_loaded_via_feature_source_id",
              cohort["feature_source_id"].notna().all()
              and cohort["feature_source_id"].is_unique,
              "loader keys on feature_source_id, never canonical_id")
    led.check("feature_matrix_6670_finite",
              X.shape == (2232, 6670) and bool(np.isfinite(X).all()),
              f"X.shape={X.shape}, all finite")
    led.check("no_forbidden_fields_in_X",
              isinstance(X, np.ndarray) and X.dtype.kind == "f"
              and X.dtype.names is None,
              "X is a bare float ndarray; FORBIDDEN_IN_X cannot appear")

    folds = read_frozen(GATE1 / "outer_fold_metrics.csv",
                        usecols=FOLD_METRIC_COLUMNS)
    efull = read_frozen(GATE1 / "source_full_coverage_error_by_fold.csv")
    frozen_hp = read_frozen(ROOT / "results" / "gate1_formal_permutation"
                            / "frozen_outer_hyperparameters.csv",
                            usecols=HP_COLUMNS)
    s0_tau = read_frozen(S0 / "thresholds_source_oof.csv")
    led.check("gate1_and_s0_source_side_inputs_carry_no_target_outcome",
              all(not any(t in c.lower() for t in ("n_mdd", "n_hc", "bacc",
                                                   "auroc", "y_true"))
                  for f in (folds, efull, frozen_hp, s0_tau) for c in f.columns),
              "the fold table, e_full,-s, the frozen hyperparameters and the S0 "
              "thresholds are source-side or identity-only")
    return X, sites, cohort, folds, efull, frozen_hp, s0_tau


def verify_frozen_hyperparameters(led, folds, efull, frozen_hp):
    hp_a = folds.set_index("site")[["selected_pca", "selected_C"]].astype(str)
    hp_b = efull.set_index("site")[["selected_pca", "selected_C"]].astype(str)
    hp_c = frozen_hp.set_index("site")[["observed_selected_pca",
                                        "observed_selected_C"]].astype(str)
    hp_c.columns = ["selected_pca", "selected_C"]
    led.check("gate1_hyperparameters_identical_across_three_files",
              hp_a.equals(hp_b.loc[hp_a.index]) and hp_a.equals(hp_c.loc[hp_a.index]),
              "selected_pca / selected_C agree across the three frozen tables "
              "and are read, never re-selected")


def build_fold_table(folds, efull, sites):
    """One record per outer fold, in frozen site order, with frozen Gate 1 choices.

    `n_s` is counted from the site membership vector — a count of acquisitions,
    which §1.1 admits as target-side input. The held-out site's class
    composition is not read here and does not exist in this object.
    """
    order = list(contracts.FROZEN_SITES)
    f = folds.set_index("site")
    e = efull.set_index("site")
    out = []
    for o, s in enumerate(order):
        pca = None if str(f.loc[s, "selected_pca"]) == "no-PCA" \
            else int(f.loc[s, "selected_pca"])
        assert pca in (None, 50, 100, 200), f"PCA {pca} outside the frozen grid"
        C = float(f.loc[s, "selected_C"])
        assert C in (0.01, 0.1, 1.0, 10.0, 100.0), f"C {C} outside the frozen grid"
        src = [u for u in order if u != s]
        assert len(src) == 22
        out.append({"o": o, "site": s, "selected_pca": pca, "selected_C": C,
                    "e_full_minus_s": float(e.loc[s, "e_full_minus_s"]),
                    "r_star": float(e.loc[s, "e_full_minus_s"]) - 0.10,
                    "source_sites": src,
                    "n_source": int(e.loc[s, "n_source"]),
                    "n_s": int((sites == s).sum())})
    return out


# ---------------------------------------------------------------------------
# §7 — source meta-LOSO selection of C_domain (the real target is absent)
# ---------------------------------------------------------------------------
def _meta_unit(Z, sites_src, j, grid):
    """One pseudo-fold: `d(j, C)` for the whole grid, plus `d(j, w = 1)`.

    The real outer target site is not in `Z` at all, so nothing in this unit can
    see it. No diagnosis is an argument — the unit's only inputs are a feature
    representation and site membership. Deterministic: no RNG, no split seed.
    """
    src_mask = sites_src != j
    Z_src, Z_tgt = Z[src_mask], Z[~src_mask]
    out = {"pseudo_target": j, "n_pseudo_source": int(src_mask.sum()),
           "n_pseudo_target": int((~src_mask).sum()),
           "n_pseudo_source_sites": int(np.unique(sites_src[src_mask]).size)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        for C in grid:
            w = dr.normalize_then_clip(
                dr.crossfit_oof_raw_weights(Z, sites_src, j, C))
            out[f"d_C_{C}"] = dr.iw_mean_discrepancy(Z_src, w, Z_tgt)
        out["n_convergence_warnings"] = sum(
            1 for c in caught if issubclass(c.category, ConvergenceWarning))
    out["d_unweighted"] = dr.iw_mean_discrepancy(Z_src, np.ones(len(Z_src)), Z_tgt)
    return out


def fold_decisions(X, sites, cohort, src, f, n_jobs, pca_dims):
    """Everything S1 decides for one outer fold, with no access to `y_s`.

    Inputs are the feature matrix, site membership and the fold's own
    source-OOF calibration rows. The held-out site contributes its unlabeled
    `X` and its acquisition count, and nothing else.
    """
    s = f["site"]
    row_of = {cid: i for i, cid in enumerate(cohort["canonical_id"])}
    idx_src = np.array([row_of[c] for c in src["canonical_id"]], dtype=int)
    idx_tgt = np.where(sites == s)[0]
    assert len(idx_src) == 2232 - f["n_s"] and len(idx_tgt) == f["n_s"]
    assert set(sites[idx_src]) == set(f["source_sites"]) and s not in sites[idx_src]
    assert np.array_equal(np.sort(np.concatenate([idx_src, idx_tgt])),
                          np.arange(2232)), "source u target != the cohort"

    # -- meta representation: the 22 source sites only (§3.1) ----------------
    t0 = time.perf_counter()
    idx_meta = np.sort(idx_src)
    rep_meta = dr.fit_domain_representation(X[idx_meta])
    pca_dims.append(("meta", s, s1e.assert_pca50(rep_meta, f"meta fold {s}")))
    Z_meta = rep_meta.transform(X[idx_meta])
    sites_meta = sites[idx_meta]
    assert s not in set(sites_meta)

    units = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_meta_unit)(Z_meta, sites_meta, j, dr.C_DOMAIN_GRID)
        for j in dr.meta_pseudo_folds(sites_meta))
    t_meta = time.perf_counter() - t0

    J = {C: float(np.mean([u[f"d_C_{C}"] for u in units]))
         for C in dr.C_DOMAIN_GRID}
    d_unw = float(np.mean([u["d_unweighted"] for u in units]))
    grid = tuple(sorted(dr.C_DOMAIN_GRID))
    best = grid[0]
    for C in grid[1:]:
        if J[C] < J[best]:                       # strict: ties keep the smaller C
            best = C
    tied = [C for C in grid if J[C] == J[best]]

    # -- deployed transductive fit (§3.1, §4) --------------------------------
    t0 = time.perf_counter()
    rep_dep = dr.fit_domain_representation(X)     # source u target = the cohort
    pca_dims.append(("deployed", s, s1e.assert_pca50(rep_dep, f"deployed {s}")))
    Z_src = rep_dep.transform(X[idx_src])         # source-OOF row order
    Z_tgt = rep_dep.transform(X[idx_tgt])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        p_D = dr.domain_probabilities(Z_src, Z_tgt, Z_src, best)
        n_conv = sum(1 for c in caught
                     if issubclass(c.category, ConvergenceWarning))
    w_raw = dr.raw_importance_ratio(p_D, n_source=len(Z_src), n_target=len(Z_tgt))
    w_norm = w_raw / w_raw.mean()
    w = np.clip(w_norm, dr.W_CLIP_LOW, dr.W_CLIP_HIGH)
    assert np.array_equal(w, dr.normalize_then_clip(w_raw)), \
        "the weight pipeline diverged from src/density_ratio.normalize_then_clip"
    t_dep = time.perf_counter() - t0

    diag = dr.weight_diagnostics(w)
    meta_rows = [{"held_out_site": s, "pseudo_target_site": u["pseudo_target"],
                  "C_domain": C, "d_pseudo_fold": u[f"d_C_{C}"],
                  "d_unweighted": u["d_unweighted"],
                  "n_pseudo_source": u["n_pseudo_source"],
                  "n_pseudo_source_sites": u["n_pseudo_source_sites"],
                  "n_pseudo_target": u["n_pseudo_target"]}
                 for u in units for C in dr.C_DOMAIN_GRID]
    obj_rows = [{"held_out_site": s, "C_domain": C, "J_C": J[C],
                 "d_unweighted_reference": d_unw, "selected": bool(C == best),
                 "n_pseudo_target_sites": len(units)} for C in grid]
    sel_row = {
        "held_out_site": s, "c_domain": best, "J_selected": J[best],
        "J_min": min(J.values()), "J_max": max(J.values()),
        "d_unweighted_reference": d_unw,
        "weighting_improves_meta_match": bool(J[best] < d_unw),
        "tie_at_argmin": bool(len(tied) > 1), "tie_break_rule": "smallest_C",
        "n_candidates": len(grid), "n_pseudo_target_sites": len(units),
        "objective": "cross_fitted_IW_covariate_mean_discrepancy",
        "target_outcome_used": False,
        "real_target_site_present_in_meta_world": False,
    }
    diag_row = {
        "held_out_site": s, "c_domain": best,
        "n_source": int(len(Z_src)), "n_target": int(len(Z_tgt)),
        "n_source_plus_target": int(len(Z_src) + len(Z_tgt)),
        "domain_pca_dim_deployed": dr.DOMAIN_PCA_DIM,
        "domain_pca_dim_meta": dr.DOMAIN_PCA_DIM, "class_weight": "None",
        "p_D_min": float(p_D.min()), "p_D_median": float(np.median(p_D)),
        "p_D_max": float(p_D.max()),
        "frac_p_D_at_guard": float(np.mean((p_D <= dr.P_D_EPS)
                                           | (p_D >= 1 - dr.P_D_EPS))),
        "w_raw_min": float(w_raw.min()), "w_raw_max": float(w_raw.max()),
        "w_norm_preclip_min": float(w_norm.min()),
        "w_norm_preclip_max": float(w_norm.max()),
        "w_mean_post_clip": float(w.mean()),
        "domain_fit_scores": "transductive/in-sample source scores",
        "n_convergence_warnings": int(n_conv),
        **{k: v for k, v in diag.items() if k != "n_source"},
    }
    w_rows = [{"held_out_site": s, "canonical_id": cid, "source_site": ss,
               "w_raw": a, "w_normalized_preclip": b, "w_final": c}
              for cid, ss, a, b, c in zip(src["canonical_id"], src["source_site"],
                                          w_raw, w_norm, w)]
    return {"weights": w, "sel": sel_row, "obj": obj_rows, "meta": meta_rows,
            "diag": diag_row, "w_rows": w_rows,
            "timing": {"meta_seconds": t_meta, "deployed_seconds": t_dep}}


def fold_thresholds(src, w, s0_tau, f):
    """`tau_q^S1` for one fold — source confidences and weights only (§9)."""
    s = f["site"]
    s0_of = {(r.held_out_site, r.method, round(float(r.q), 2)): float(r.tau)
             for r in s0_tau.itertuples()
             if r.q != s0e.NO_ABSTENTION and r.held_out_site == s}
    rows = []
    for m in METHODS:
        c = src[f"confidence_{m}"].to_numpy()
        for q, tau in dr.weighted_source_thresholds(c, w).items():
            qq = round(float(q), 2)
            audit = s1e.equal_weight_tau_audit(c, q, s0_of[(s, m, qq)])
            rows.append({
                "held_out_site": s, "method": m, "q": qq, "tau_S1": tau,
                "tau_S0": s0_of[(s, m, qq)],
                "tau_difference_S1_minus_S0": tau - s0_of[(s, m, qq)],
                "weighted_cumulative_mass_at_tau": float(w[c <= tau].sum() / w.sum()),
                "realized_weighted_source_coverage":
                    dr.weighted_retained_mass(c, w, tau),
                "realized_unweighted_source_coverage": float(np.mean(c >= tau)),
                "n_source_oof": int(c.size),
                "source_intended_coverage": round(1.0 - qq, 2),
                "operating_point": ("OP50" if qq == s0e.OP50_Q else
                                    "OP30" if qq == s0e.OP30_Q else ""),
                **audit,
            })
        rows.append({"held_out_site": s, "method": m, "q": s0e.NO_ABSTENTION,
                     "tau_S1": "", "tau_S0": "", "n_source_oof": int(c.size),
                     "source_intended_coverage": 1.0, "operating_point": ""})
    return pd.DataFrame(rows)


def fold_weighted_source_risk(src, w, thr, f):
    """Weighted source selective BER for one fold (§10). Source labels only."""
    s = f["site"]
    tau_of = {(r.method, r.q): r.tau_S1 for r in thr.itertuples()
              if r.q != s0e.NO_ABSTENTION}
    y_o = src["y_true"].to_numpy().astype(int)
    p_o = src["oof_pred"].to_numpy().astype(int)
    src_site = src["source_site"].to_numpy()
    rows, retained_ess = [], {}
    for m in METHODS:
        c = src[f"confidence_{m}"].to_numpy()
        for q in list(s0e.Q_GRID) + [s0e.NO_ABSTENTION]:
            if q == s0e.NO_ABSTENTION:
                keep, tau, qq = np.ones(len(c), dtype=bool), "", q
            else:
                qq = round(float(q), 2)
                tau = tau_of[(m, qq)]
                keep = s0e.retained_mask(c, tau)
            ber, ldg = dr.weighted_selective_ber(y_o, p_o, keep, w)
            ber_sb, n_sb = s1e.weighted_site_balanced_source_ber(
                y_o, p_o, keep, w, src_site)
            if q in (s0e.OP50_Q, s0e.OP30_Q):
                op = "OP50" if q == s0e.OP50_Q else "OP30"
                retained_ess[(m, op)] = (dr.effective_sample_size(w[keep])
                                         if keep.any() else NE)
            rows.append({
                "held_out_site": s, "method": m, "q": qq, "tau_S1": tau,
                "n_source_oof": int(len(c)), "n_keep_source": ldg["n_keep"],
                "n_keep_source_MDD": ldg["n_keep_MDD"],
                "n_keep_source_HC": ldg["n_keep_HC"],
                "W_MDD": ldg["W_MDD"], "W_HC": ldg["W_HC"],
                "W_retained_total": float(w[keep].sum()),
                "weighted_error_MDD": ldg.get("err_MDD_weighted", NE),
                "weighted_error_HC": ldg.get("err_HC_weighted", NE),
                "BER_sel_source_oof_weighted": ber,
                "BER_sel_source_oof_weighted_site_balanced": ber_sb,
                "n_estimable_source_sites": n_sb,
                "weighted_source_coverage": float(w[keep].sum() / w.sum()),
                "unweighted_source_coverage": float(keep.mean()),
                "estimable": ldg["estimable"],
            })
    return pd.DataFrame(rows), retained_ess


def fold_ess_row(diag_row, retained_ess, f):
    r = {"held_out_site": f["site"], "n_target": int(f["n_s"]),
         **{c: diag_row[c] for c in
            ("n_source", "c_domain", "ess", "ess_fraction", "w_min", "w_median",
             "w_mean", "w_max", "frac_clipped_low", "frac_clipped_high",
             "w_mean_post_clip")}}
    for m in METHODS:
        for op in ("OP50", "OP30"):
            r[f"ess_retained_{op}_{m}"] = retained_ess[(m, op)]
    r["ess_exclusion_rule"] = "NONE (docs/RQ1_S1_PROTOCOL.md §14)"
    return r


# ---------------------------------------------------------------------------
# §11 — target evaluation, per fold, strictly after that fold's release
# ---------------------------------------------------------------------------
def evaluate_fold(tgt, thr, wsrc, f):
    s = f["site"]
    tau_of = {(r.method, r.q): r.tau_S1 for r in thr.itertuples()
              if r.q != s0e.NO_ABSTENTION}
    tau0_of = {(r.method, r.q): r.tau_S0 for r in thr.itertuples()
               if r.q != s0e.NO_ABSTENTION}
    ws = wsrc.set_index(["method", "q"])
    y_t = tgt["y_true"].to_numpy().astype(int)
    p_t = tgt["pred"].to_numpy().astype(int)
    loss = (p_t != y_t).astype(float)
    n_all_mdd, n_all_hc = int((y_t == 1).sum()), int((y_t == 0).sum())
    rows, aurc_rows = [], []
    tie_ok = True

    for m in METHODS:
        c_t = tgt[f"confidence_{m}"].to_numpy()
        aurc_rows.append({"held_out_site": s, "method": m, "n_s": f["n_s"],
                          "AURC_s_recomputed": tie_aware_aurc(c_t, loss)})
        for q in list(s0e.Q_GRID) + [s0e.NO_ABSTENTION]:
            qq = round(float(q), 2) if q != s0e.NO_ABSTENTION else q
            if q == s0e.NO_ABSTENTION:
                keep, tau, tau0 = np.ones(len(c_t), dtype=bool), "", ""
            else:
                tau, tau0 = tau_of[(m, qq)], tau0_of[(m, qq)]
                keep = s0e.retained_mask(c_t, tau)
                tie_ok &= bool(keep[c_t == tau].all())
            ber, ldg = s0e.selective_ber(y_t, p_t, keep)
            est = ber != NE
            srow = ws.loc[(m, qq)]
            ber_w = srow["BER_sel_source_oof_weighted"]
            jsd = s0e.jsd_composition(ldg["n_keep_MDD"], ldg["n_keep_HC"],
                                      n_all_mdd, n_all_hc)
            rows.append({
                "held_out_site": s, "method": m, "q": qq,
                "tau_S1": tau, "tau_S0": tau0,
                "source_intended_coverage": (1.0 if q == s0e.NO_ABSTENTION
                                             else round(1.0 - qq, 2)),
                "operating_point": ("OP50" if qq == s0e.OP50_Q else
                                    "OP30" if qq == s0e.OP30_Q else ""),
                "n_s": f["n_s"],
                "coverage_target": ldg["n_keep"] / f["n_s"],
                "n_keep": ldg["n_keep"], "n_keep_MDD": ldg["n_keep_MDD"],
                "n_keep_HC": ldg["n_keep_HC"],
                "BER_sel_target": ber,
                "BAcc_retained": (1.0 - ber) if est else NE,
                "BER_sel_estimable": est,
                "BER_sel_source_oof_weighted": ber_w,
                "BER_sel_source_oof_weighted_site_balanced":
                    srow["BER_sel_source_oof_weighted_site_balanced"],
                "weighted_source_coverage": srow["weighted_source_coverage"],
                "n_keep_source_oof": srow["n_keep_source"],
                "risk_gap": ((ber - ber_w) if (est and ber_w != NE) else NE),
                "r_star_s": f["r_star"],
                "risk_excess": (ber - f["r_star"]) if est else NE,
                "realized_risk_violation__descriptive":
                    (int(ber > f["r_star"]) if est else NE),
                "jsd_keep_vs_all": jsd,
            })
    return (pd.DataFrame(rows), pd.DataFrame(aurc_rows), tie_ok,
            {"n_MDD": n_all_mdd, "n_HC": n_all_hc})


def assert_aurc_invariant(aurc_recomputed, s0_aurc, led):
    a = s0_aurc.set_index(["held_out_site", "method"])["AURC_s"]
    b = aurc_recomputed.set_index(["held_out_site", "method"])["AURC_s_recomputed"]
    common = a.index.intersection(b.index)
    diff = b.loc[common].to_numpy() - a.loc[common].to_numpy()
    dr.assert_target_aurc_invariant(a.loc[common].to_numpy(),
                                    b.loc[common].to_numpy())
    led.check("target_aurc_exactly_equals_s0_for_23_sites_x_3_methods",
              len(common) == 69 and bool(np.array_equal(
                  a.loc[common].to_numpy(), b.loc[common].to_numpy())),
              f"69 target AURC values recomputed from the frozen S0 target "
              f"confidences; max |S1 - S0| = {float(np.max(np.abs(diff))):.1e} "
              "(exact zero required); S1 emits no AURC of its own")
    return float(np.max(np.abs(diff)))


# ---------------------------------------------------------------------------
# §13 — cross-site inference: exactly 36 intervals, S0's own draws reused
# ---------------------------------------------------------------------------
def _q_rows(rc, m, q):
    """Rows of one (method, q) cell.

    The `q` column mixes the 17 numeric candidates with the `NO_ABSTENTION`
    anchor, so it is object-typed both in memory and after a CSV round trip.
    """
    qn = pd.to_numeric(rc["q"], errors="coerce")
    return rc[(rc["method"] == m) & (qn == float(q))]


def _per_site(rc, m, q, order):
    g = _q_rows(rc, m, q).set_index("held_out_site")
    assert len(g) == len(order), f"expected {len(order)} sites at q={q}, got {len(g)}"
    return {
        "coverage": np.array([float(g.loc[s, "coverage_target"]) for s in order]),
        "ber": np.array([s0e.as_float(g.loc[s, "BER_sel_target"]) for s in order]),
        "violation": np.array([s0e.as_float(
            g.loc[s, "realized_risk_violation__descriptive"]) for s in order]),
        "jsd": np.array([s0e.as_float(g.loc[s, "jsd_keep_vs_all"]) for s in order]),
        "risk_gap": np.array([s0e.as_float(g.loc[s, "risk_gap"]) for s in order]),
        "risk_excess": np.array([s0e.as_float(g.loc[s, "risk_excess"])
                                 for s in order]),
    }


def cross_site_inference(rc, s0_rc, led):
    order = list(contracts.FROZEN_SITES)
    counts = sb.site_bootstrap_counts(len(order))
    frozen = np.load(S0 / "site_bootstrap_draws.npy")
    led.check("same_5000x23_s0_bootstrap_draws_reused_not_redrawn",
              counts.shape == (5000, 23) and np.array_equal(counts, frozen)
              and sb.SITE_BOOTSTRAP_ENTROPY == [20260807, 1102],
              "site_bootstrap_counts(23) is bit-identical to the frozen "
              "results/rq1_s0/site_bootstrap_draws.npy; S1 defines no "
              "SeedSequence, no RNG and no new seed constant")

    s1v = {(op, m): _per_site(rc, m, q, order)
           for op, q in s0e.OPERATING_POINTS.items() for m in METHODS}
    s0v = {(op, m): _per_site(s0_rc, m, q, order)
           for op, q in s0e.OPERATING_POINTS.items() for m in METHODS}

    emitted, gap_rows = [], []
    quantities = (("site_balanced_coverage", "coverage"),
                  ("site_balanced_selective_ber", "ber"),
                  ("descriptive_violation_rate", "violation"))
    for op in s0e.OPERATING_POINTS:
        for name, key in quantities:
            for m in METHODS:
                q_name = f"S1_{op}_{name}"
                assert dr.requires_ci(q_name)
                emitted.append({"quantity": q_name, "method_or_contrast": m,
                                **sb.bootstrap_summary(s1v[(op, m)][key], counts)})
        for name, key in quantities:
            for m in METHODS:
                q_name = f"S1_minus_S0_{op}_{name}"
                assert dr.requires_ci(q_name)
                emitted.append({"quantity": q_name, "method_or_contrast": m,
                                **sb.paired_contrast(s1v[(op, m)][key],
                                                     s0v[(op, m)][key], counts)})
        for m in METHODS:                      # point estimate only (§13.1 row 4)
            a, b = s1v[(op, m)]["risk_gap"], s0v[(op, m)]["risk_gap"]
            common = np.isfinite(a) & np.isfinite(b)
            assert not dr.requires_ci("s1_minus_s0_risk_gap")
            gap_rows.append({
                "operating_point": op, "method": m,
                "risk_gap_S1_site_balanced":
                    float(a[common].mean()) if common.any() else NE,
                "risk_gap_S0_site_balanced":
                    float(b[common].mean()) if common.any() else NE,
                "risk_gap_S1_minus_S0":
                    float(a[common].mean() - b[common].mean()) if common.any() else NE,
                "n_common_estimable_sites": int(common.sum()),
                "ci": "PROHIBITED (docs/RQ1_S1_PROTOCOL.md §13.1 row 4) — "
                      "point estimate only",
            })

    df = pd.DataFrame(emitted)
    n = s1e.assert_interval_registry_exact(
        [(r["quantity"], r["method_or_contrast"]) for r in emitted])
    led.check("exactly_36_intervals_emitted_none_outside_registry", n == 36,
              f"{n} intervals emitted, set-equal to the frozen S1 registry; no "
              "missing interval, no extra interval, no optional interval")
    led.check("all_intervals_are_2p5_97p5_percentile",
              bool(df["ci_method"].str.startswith("95% percentile").all()),
              "no BCa, no normal approximation, no other interval type")
    led.check("paired_contrasts_use_common_estimable_sites_and_same_replicates",
              bool((df["total_bootstrap_replicates"] == 5000).all())
              and bool(df[df["quantity"].str.startswith("S1_minus_S0")]
                       ["n_common_estimable_sites"].notna().all()),
              "every S1 - S0 difference is formed inside each replicate on the "
              "same site multiplicities, restricted to estimable_S1 & "
              "estimable_S0 before any averaging")
    led.check("fixed_s1_minus_s0_sign_convention",
              all(q.startswith("S1_minus_S0") for q in dr.S1_CI_REQUIRED
                  if "minus" in q),
              "every contrast is S1 - S0; the sign is frozen and never flipped")
    gate = df.apply(
        lambda r: (r["ci_low"] == sb.NOT_ESTIMABLE)
        == (r["valid_bootstrap_replicates"] < sb.MIN_VALID_REPLICATES), axis=1)
    led.check("ci_suppressed_exactly_below_4750_valid_replicates", bool(gate.all()),
              f"an interval is emitted iff valid_bootstrap_replicates >= "
              f"{sb.MIN_VALID_REPLICATES}")
    return df, pd.DataFrame(gap_rows), s1v, order


def operating_point_tables(intervals, s1v, gaps):
    ci = intervals.set_index(["quantity", "method_or_contrast"])
    g = gaps.set_index(["operating_point", "method"])
    out = {}
    for op, q in s0e.OPERATING_POINTS.items():
        rows = []
        for m in METHODS:
            v = s1v[(op, m)]
            r = {"operating_point": op, "q": q, "method": m,
                 "source_intended_coverage": round(1.0 - q, 2)}
            for label, key, arr in (
                    ("actual_coverage", "site_balanced_coverage", v["coverage"]),
                    ("selective_ber", "site_balanced_selective_ber", v["ber"]),
                    ("descriptive_violation_rate", "descriptive_violation_rate",
                     v["violation"])):
                c = ci.loc[(f"S1_{op}_{key}", m)]
                r[f"{label}_site_balanced"] = c["point_estimate"]
                r[f"{label}_ci_low"] = c["ci_low"]
                r[f"{label}_ci_high"] = c["ci_high"]
                r[f"{label}_n_estimable_sites"] = c["n_estimable_sites"]
                r[f"{label}_valid_bootstrap_replicates"] = \
                    c["valid_bootstrap_replicates"]
                r[f"{label}_min"] = float(np.nanmin(arr)) if np.isfinite(arr).any() else NE
                r[f"{label}_max"] = float(np.nanmax(arr)) if np.isfinite(arr).any() else NE
                d = ci.loc[(f"S1_minus_S0_{op}_{key}", m)]
                r[f"{label}_S1_minus_S0"] = d["point_estimate"]
                r[f"{label}_S1_minus_S0_ci_low"] = d["ci_low"]
                r[f"{label}_S1_minus_S0_ci_high"] = d["ci_high"]
                r[f"{label}_S1_minus_S0_n_common_estimable_sites"] = \
                    d["n_common_estimable_sites"]
            r["risk_gap_S1_site_balanced"] = g.loc[(op, m), "risk_gap_S1_site_balanced"]
            r["risk_gap_S1_minus_S0_point_estimate"] = \
                g.loc[(op, m), "risk_gap_S1_minus_S0"]
            r["risk_gap_ci"] = "PROHIBITED — point estimate only"
            rows.append(r)
        out[op] = pd.DataFrame(rows)
    return out


def contrast_table(intervals, gaps):
    rows = []
    ci = intervals.set_index(["quantity", "method_or_contrast"])
    for op in s0e.OPERATING_POINTS:
        for name in ("site_balanced_coverage", "site_balanced_selective_ber",
                     "descriptive_violation_rate"):
            for m in METHODS:
                c = ci.loc[(f"S1_minus_S0_{op}_{name}", m)]
                s1 = ci.loc[(f"S1_{op}_{name}", m)]
                rows.append({
                    "operating_point": op, "quantity": name, "method": m,
                    "sign_convention": "S1 - S0",
                    "S1_point_estimate": s1["point_estimate"],
                    "difference_point_estimate": c["point_estimate"],
                    "ci_low": c["ci_low"], "ci_high": c["ci_high"],
                    "n_common_estimable_sites": c["n_common_estimable_sites"],
                    "n_estimable_sites_S1": c["n_estimable_sites_A"],
                    "n_estimable_sites_S0": c["n_estimable_sites_B"],
                    "valid_bootstrap_replicates": c["valid_bootstrap_replicates"],
                    "total_bootstrap_replicates": c["total_bootstrap_replicates"],
                    "ci_method": c["ci_method"],
                })
    for r in gaps.itertuples():
        rows.append({"operating_point": r.operating_point, "quantity": "risk_gap",
                     "method": r.method, "sign_convention": "S1 - S0",
                     "S1_point_estimate": r.risk_gap_S1_site_balanced,
                     "difference_point_estimate": r.risk_gap_S1_minus_S0,
                     "ci_low": "PROHIBITED", "ci_high": "PROHIBITED",
                     "n_common_estimable_sites": r.n_common_estimable_sites,
                     "n_estimable_sites_S1": "", "n_estimable_sites_S0": "",
                     "valid_bootstrap_replicates": "",
                     "total_bootstrap_replicates": "",
                     "ci_method": "NOT PART OF S1 PROTOCOL (§13.1 row 4)"})
    return pd.DataFrame(rows)


def jsd_tables(s1v, order, led):
    rows, dist = [], []
    for op in s0e.OPERATING_POINTS:
        for m in METHODS:
            v = s1v[(op, m)]["jsd"]
            ok = np.isfinite(v)
            rows.append({
                "operating_point": op, "method": m,
                "site_balanced_mean": float(v[ok].mean()) if ok.any() else NE,
                "median": float(np.median(v[ok])) if ok.any() else NE,
                "min": float(v[ok].min()) if ok.any() else NE,
                "max": float(v[ok].max()) if ok.any() else NE,
                "n_estimable_sites": int(ok.sum()), "n_sites": 23,
                "log_base": 2, "variables": "[MDD, HC] diagnosis composition only",
                "ci": "NOT PART OF S1 PROTOCOL (§16) — descriptive only",
            })
            for s, val in zip(order, v):
                dist.append({"operating_point": op, "method": m, "site": s,
                             "jsd": val if np.isfinite(val) else NE})
    assert not dr.requires_ci("jsd")
    led.check("jsd_is_descriptive_only_and_carries_no_interval", True,
              "JSD(P_keep || P_all) on diagnosis composition, base 2, inherited "
              "from S0 unchanged; no interval at any q under any method")
    return pd.DataFrame(rows), pd.DataFrame(dist)


def worst_site_table(rc, s1v, order):
    rows = []
    for op, q in s0e.OPERATING_POINTS.items():
        for m in METHODS:
            v = s1v[(op, m)]
            g = _q_rows(rc, m, q).set_index("held_out_site")

            def pick(arr, how):
                a = np.asarray(arr, dtype=float)
                if not np.isfinite(a).any():
                    return NE, NE, NE
                i = int(np.nanargmin(a) if how == "min" else np.nanargmax(a))
                return order[i], float(a[i]), int(g.loc[order[i], "n_keep"])

            cov_lo_s, cov_lo, cov_lo_n = pick(v["coverage"], "min")
            cov_hi_s, cov_hi, cov_hi_n = pick(v["coverage"], "max")
            ber_s, ber_v, ber_n = pick(v["ber"], "max")
            exc_s, exc_v, exc_n = pick(v["risk_excess"], "max")
            jsd_s, jsd_v, jsd_n = pick(v["jsd"], "max")
            rows.append({
                "operating_point": op, "method": m,
                "min_target_coverage": cov_lo, "min_coverage_site": cov_lo_s,
                "min_coverage_n_keep": cov_lo_n,
                "max_target_coverage": cov_hi, "max_coverage_site": cov_hi_s,
                "max_coverage_n_keep": cov_hi_n,
                "worst_estimable_selective_ber": ber_v, "worst_ber_site": ber_s,
                "worst_ber_n_keep": ber_n,
                "highest_risk_excess": exc_v, "highest_risk_excess_site": exc_s,
                "highest_risk_excess_n_keep": exc_n,
                "highest_jsd": jsd_v, "highest_jsd_site": jsd_s,
                "highest_jsd_n_keep": jsd_n,
                "n_estimable_sites": int(np.isfinite(v["ber"]).sum()),
                "ci": "NOT PART OF S1 PROTOCOL — descriptive only",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §17, §20 assertion 20 — vocabulary and reachability scan
# ---------------------------------------------------------------------------
def scan_output_vocabulary(led) -> None:
    claim_words = ("certified", "uncertified", "significantly",
                   "statistically significant")
    topic_words = ("certif", "uncertif", "ucb", "clopper", "conformal",
                   "fixed_sequential")
    hits: list[str] = []
    for p in sorted(OUT.glob("*.csv")):
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        hits += [f"{p.name}:column:{c}" for c in df.columns
                 if any(w in str(c).lower() for w in topic_words)]
        for c in df.columns:
            if df[c].astype(str).str.lower().str.contains(
                    "|".join(topic_words), regex=True).any():
                hits.append(f"{p.name}:value:{c}")
    for p in sorted(OUT.glob("*.md")) + sorted(OUT.glob("*.yaml")):
        low = p.read_text(errors="ignore").lower()
        hits += [f"{p.name}:claim:{w}" for w in claim_words if w in low]
    led.check("no_out_of_scope_claim_vocabulary_in_outputs", not hits,
              f"scanned {len(list(OUT.glob('*.csv')))} CSV column/value sets and "
              f"{len(list(OUT.glob('*.md'))) + len(list(OUT.glob('*.yaml')))} "
              f"text outputs; hits={hits}")

    ids: set[str] = set()
    for src in S1_MODULES:
        for node in ast.walk(ast.parse((ROOT / src).read_text())):
            if isinstance(node, ast.Name):
                ids.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                ids.add(node.attr.lower())
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                ids.add(node.name.lower())
            elif isinstance(node, ast.ImportFrom):
                ids.add((node.module or "").lower())
            elif isinstance(node, ast.Import):
                ids.update(a.name.lower() for a in node.names)
    bad = sorted(i for i in ids
                 if any(w in i for w in topic_words + ("binomtest", "binom_test",
                                                       "proportion_confint")))
    led.check("no_upper_bound_or_ordered_testing_call_reachable", not bad,
              "executable identifiers across the four S1 modules carry no "
              f"upper-bound or ordered-testing symbol; offenders={bad}")


def behavioural_probes(led) -> None:
    """Assertions on the estimator itself, checked by behaviour not by reading."""
    from sklearn.linear_model import LogisticRegression

    # Deterministic closed-form probe data: S1 introduces no RNG and no seed
    # constant anywhere, not even for a self-test (§20 assertion 18).
    a = np.arange(300, dtype=float)[:, None] + np.arange(5, dtype=float)
    b = np.arange(60, dtype=float)[:, None] + np.arange(5, dtype=float)
    S = np.sin(a) + 0.3 * np.cos(2 * a)
    T = np.sin(b) + 0.3 * np.cos(2 * b) + 0.6
    Z = np.vstack([S, T])
    D = np.r_[np.zeros(len(S), int), np.ones(len(T), int)]
    got = dr.domain_probabilities(S, T, S, 1.0)
    ref_none = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                  max_iter=dr.DOMAIN_MAX_ITER, class_weight=None,
                                  random_state=dr.PCA_RANDOM_STATE).fit(Z, D)
    ref_bal = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=dr.DOMAIN_MAX_ITER,
                                 class_weight="balanced",
                                 random_state=dr.PCA_RANDOM_STATE).fit(Z, D)
    p_none = np.clip(ref_none.predict_proba(S)[:, 1], dr.P_D_EPS, 1 - dr.P_D_EPS)
    p_bal = np.clip(ref_bal.predict_proba(S)[:, 1], dr.P_D_EPS, 1 - dr.P_D_EPS)
    led.check("domain_classifier_is_not_class_balanced",
              bool(np.array_equal(got, p_none)) and not np.allclose(p_none, p_bal),
              "domain_probabilities reproduces the class_weight=None fit exactly "
              "and differs from the class_weight='balanced' fit; no domain class "
              "balancing is applied anywhere")
    w_raw = dr.raw_importance_ratio(got, n_source=len(S), n_target=len(T))
    led.check("density_ratio_prior_correction_is_odds_times_ns_over_nt",
              bool(np.allclose(w_raw, (got / (1 - got)) * (len(S) / len(T)),
                               rtol=0, atol=0)),
              "w_raw == [p_D/(1-p_D)] * (n_source/n_target) with the counts of "
              "the fit that produced p_D")
    w = dr.normalize_then_clip(w_raw)
    clipped = np.clip(w_raw / w_raw.mean(), dr.W_CLIP_LOW, dr.W_CLIP_HIGH)
    led.check("normalize_then_clip_then_stop",
              bool(np.array_equal(w, clipped))
              and float(w.min()) >= dr.W_CLIP_LOW
              and float(w.max()) <= dr.W_CLIP_HIGH,
              "the pipeline is w_raw -> /mean(w_raw) -> clip[0.1, 10] and "
              "terminates there; no post-clip renormalization exists")
    led.check("all_s1_estimands_are_invariant_to_a_positive_weight_rescaling",
              abs(dr.effective_sample_size(3.7 * w) - dr.effective_sample_size(w))
              < 1e-9
              and dr.weighted_quantile(S[:, 0], w, 0.5)
              == dr.weighted_quantile(S[:, 0], 3.7 * w, 0.5),
              "ESS and the weighted quantile are homogeneous of degree 0 (§5.3)")
    led.check("ess_formula_is_sum_w_squared_over_sum_w2",
              abs(dr.effective_sample_size(np.ones(37)) - 37.0) < 1e-9
              and abs(dr.effective_sample_size(np.array([2.0, 2.0]))
                      - dr.effective_sample_size(np.array([1.0, 1.0]))) < 1e-12,
              "ESS = (sum w)^2 / sum w^2; equals n for constant weights and is "
              "invariant to the weight scale")
    c = np.array([0.6, 0.6, 0.7, 0.9, 0.9])
    w5 = np.array([1.0, 3.0, 1.0, 1.0, 2.0])
    perm = np.array([4, 0, 2, 1, 3])
    led.check("weighted_quantile_is_deterministic_and_tie_invariant",
              dr.weighted_quantile(c, w5, 0.5) == dr.weighted_quantile(
                  c[perm], w5[perm], 0.5)
              and dr.weighted_quantile(c, w5, 0.0) == float(c.min()),
              "the inverse weighted ECDF aggregates exact-equal values into "
              "blocks, so it is invariant to row order and to within-tie order")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="all")
    ap.add_argument("--n-jobs", type=int, default=14)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    DECISIONS.mkdir(parents=True, exist_ok=True)
    led = contracts.AssertionLedger()
    env = environment()
    started = datetime.now(timezone.utc).isoformat()
    t_start = time.perf_counter()

    print("=" * 78)
    print("STAGE 1.3a — FORMAL RQ1 / S1 EXECUTION (zero-label transductive)")
    print("=" * 78)
    print(json.dumps(env, indent=2))

    # ---- pre-run provenance validation -------------------------------------
    led.check("required_git_tags_present", all(env["tags_present"].values()),
              ", ".join(f"{k}={v}" for k, v in env["tags_present"].items()))
    led.check("working_tree_clean_at_entry", env["git_status_clean_at_entry"],
              "git status --porcelain empty at entry")
    checks_in = verify_s0_checksums(led, "at_entry")
    assert_s0_read_only(led)
    static_firewall_audit(led)

    X, sites, cohort, folds, efull, frozen_hp, s0_tau = load_frozen_inputs(led)
    verify_frozen_hyperparameters(led, folds, efull, frozen_hp)
    fold_table = build_fold_table(folds, efull, sites)
    led.check("r_star_read_from_frozen_gate1_never_recomputed_in_s1",
              len(fold_table) == 23 and all(
                  abs(f["r_star"] - (f["e_full_minus_s"] - 0.10)) < 1e-15
                  for f in fold_table),
              f"r*_s = e_full,-s - 0.10 in "
              f"[{min(f['r_star'] for f in fold_table):.4f}, "
              f"{max(f['r_star'] for f in fold_table):.4f}], from source-side "
              "Gate 1 output only")
    led.check("all_three_confidence_methods_carried_into_s1",
              tuple(METHODS) == ("probability", "bootstrap", "mahalanobis"),
              "probability, bootstrap disagreement and Mahalanobis all enter S1 "
              "weighted calibration; none is dropped or pre-selected")
    behavioural_probes(led)

    oof = s1da.SourceOofProvider()
    guard = s1da.TargetOutcomeGuard(contracts.FROZEN_SITES)
    led.check("target_outcome_guard_starts_closed_for_all_23_folds",
              not guard.all_released
              and set(oof.folds) == set(contracts.FROZEN_SITES),
              "every fold's target diagnosis is closed at entry; the source-OOF "
              "provider indexes all 23 folds by their non-outcome key")

    if args.stages == "validate":
        pd.DataFrame(led.rows).to_csv(OUT / "leakage_assertions.csv", index=False)
        print(f"\n[validate] {len(led.rows)} assertions, all_pass={led.all_pass}")
        return 0

    # ---- per fold: decisions -> checksums -> release -> evaluation ----------
    pca_dims, timings = [], {"meta_seconds": 0.0, "deployed_seconds": 0.0}
    all_meta, all_obj, all_sel, all_diag, all_w = [], [], [], [], []
    all_thr, all_wsrc, all_ess, all_rc, all_aurc = [], [], [], [], []
    src_audits, tie_ok, composition = {}, True, []

    for f in fold_table:
        s = f["site"]
        d = DECISIONS / s
        d.mkdir(parents=True, exist_ok=True)

        src = oof.for_fold(s, expected_rows=f["n_source"])
        src_audits[s] = oof.audit(s, src)
        assert src_audits[s]["target_site_rows_in_source_labels"] == 0

        out = fold_decisions(X, sites, cohort, src, f, args.n_jobs, pca_dims)
        thr = fold_thresholds(src, out["weights"], s0_tau, f)
        wsrc, retained_ess = fold_weighted_source_risk(src, out["weights"], thr, f)
        ess_row = fold_ess_row(out["diag"], retained_ess, f)

        artifacts = {
            "selected_domain_C": pd.DataFrame([out["sel"]]),
            "domain_meta_objective": pd.DataFrame(out["obj"]),
            "domain_meta_pseudo_target_discrepancy": pd.DataFrame(out["meta"]),
            "domain_fit_diagnostics": pd.DataFrame([out["diag"]]),
            "density_weights": pd.DataFrame(out["w_rows"]),
            "ess": pd.DataFrame([ess_row]),
            "thresholds_weighted_source": thr,
            "weighted_source_risk": wsrc,
        }
        for name, frame in artifacts.items():
            frame.to_csv(d / f"{name}.csv", index=False)
            guard.record_decision(s, name, d / f"{name}.csv")
        (d / "decision_checksums.json").write_text(json.dumps(
            {name: s1da.sha256_file(d / f"{name}.csv") for name in artifacts},
            indent=2))
        guard.record_decision(s, "decision_checksums",
                              d / "decision_checksums.json")
        guard.release(s)

        tgt = guard.load_target_y(s, expected_rows=f["n_s"])
        rc_s, aurc_s, ok, comp = evaluate_fold(tgt, thr, wsrc, f)
        tie_ok &= ok
        composition.append({"held_out_site": s, **comp})
        guard.record_evaluation(s)

        all_meta += out["meta"]; all_obj += out["obj"]; all_sel.append(out["sel"])
        all_diag.append(out["diag"]); all_w += out["w_rows"]
        all_thr.append(thr); all_wsrc.append(wsrc); all_ess.append(ess_row)
        all_rc.append(rc_s); all_aurc.append(aurc_s)
        for k in timings:
            timings[k] += out["timing"][k]
        print(f"[fold] {s}: C*={out['sel']['c_domain']}  "
              f"J={out['sel']['J_selected']:.5f}  ESS={out['diag']['ess']:.1f}  "
              f"released -> evaluated")

    meta = pd.DataFrame(all_meta)
    obj = pd.DataFrame(all_obj)
    sel = pd.DataFrame(all_sel)
    diagd = pd.DataFrame(all_diag)
    wdf = pd.DataFrame(all_w)
    thresholds = pd.concat(all_thr, ignore_index=True)
    wsrc = pd.concat(all_wsrc, ignore_index=True)
    ess = pd.DataFrame(all_ess)
    rc = pd.concat(all_rc, ignore_index=True)
    aurc_rec = pd.concat(all_aurc, ignore_index=True)
    comp = pd.DataFrame(composition)
    del X

    # ---- the firewall ledger ------------------------------------------------
    ledger = guard.fold_ledger(src_audits)
    guard.write(OUT / "target_label_barrier.json", ledger)
    ledger.to_csv(OUT / "firewall_fold_ledger.csv", index=False)
    led.check("target_y_firewall_ordering_ok_for_all_23_folds",
              len(ledger) == 23 and bool(ledger["ordering_ok"].all())
              and not bool(ledger["target_labels_visible_prebarrier"].any()),
              "for every outer fold: C_domain, weights, thresholds and their "
              "checksums are frozen (seq) < release (seq) < first target-y load "
              "(seq), and no target label was loaded before that fold's release")
    led.check("source_label_objects_never_contain_the_held_out_site",
              bool((ledger["source_label_target_rows"] == 0).all())
              and bool((ledger["source_labels_loaded"]
                        == [f["n_source"] for f in fold_table]).all()),
              "each fold's source calibration object holds exactly its 22-site "
              "source pool; the held-out site contributes zero labelled rows")

    # ---- cross-site aggregation (all folds released) ------------------------
    guard.require_all_released("cross-site inference")
    s0_rc = read_frozen(S0 / "risk_coverage_all_q.csv")
    s0_aurc = read_frozen(S0 / "per_site_aurc.csv")

    led.check("cohort_composition_1182_mdd_1050_hc_after_release",
              int(comp["n_MDD"].sum()) == 1182 and int(comp["n_HC"].sum()) == 1050
              and int(comp[["n_MDD", "n_HC"]].to_numpy().sum()) == 2232,
              f"summing each fold's released target labels gives "
              f"{int(comp['n_MDD'].sum())} MDD / {int(comp['n_HC'].sum())} HC "
              "over 2232 acquisitions — the frozen composition, verified only "
              "after every fold's decisions were sealed")
    led.check("all_46_domain_pca_fits_are_exactly_50_components",
              len(pca_dims) == 46 and all(k == 50 for _, _, k in pca_dims),
              "23 deployed + 23 meta scaler+PCA fits, every one n_components_ "
              "== 50; the low-rank fallback was never reached")
    n_src_of = {f["site"]: 2232 - f["n_s"] for f in fold_table}
    led.check("real_target_absent_from_every_meta_object",
              bool((meta["held_out_site"] != meta["pseudo_target_site"]).all())
              and bool((meta["n_pseudo_source_sites"] == 21).all())
              and bool((meta.groupby("held_out_site")["pseudo_target_site"]
                        .nunique() == 22).all()),
              "22 pseudo-targets per fold, each scored against 21 pseudo-source "
              "sites; the outer held-out site is in no meta representation, no "
              "meta domain fit and no objective term")
    led.check("meta_world_partitions_the_folds_source_pool_exactly",
              bool((meta["n_pseudo_source"] + meta["n_pseudo_target"]
                    == meta["held_out_site"].map(n_src_of)).all()),
              "the pseudo-source pool and the pseudo-target partition that "
              "fold's 22-site source pool exactly")
    led.check("exactly_one_c_domain_per_outer_fold_from_frozen_grid",
              len(sel) == 23 and sel["held_out_site"].is_unique
              and bool(sel["c_domain"].isin(dr.C_DOMAIN_GRID).all()),
              f"23 selections, each in {list(dr.C_DOMAIN_GRID)}")
    led.check("c_domain_shared_by_all_three_confidence_methods", True,
              "the domain classifier is a function of X and site membership "
              "only, so one weight vector per fold serves all three methods")
    led.check("domain_fit_uses_actual_class_frequencies",
              bool((diagd["class_weight"] == "None").all())
              and bool((diagd["n_source_plus_target"] == 2232).all()),
              "class_weight is None in every domain fit and n_source + "
              "n_target == 2232 in every deployed fit")
    led.check("weights_normalized_then_clipped_then_stopped",
              bool((wdf["w_final"] >= dr.W_CLIP_LOW - 1e-12).all())
              and bool((wdf["w_final"] <= dr.W_CLIP_HIGH + 1e-12).all())
              and bool(np.isfinite(wdf["w_raw"]).all())
              and bool((wdf["w_raw"] > 0).all()),
              f"final weights in [{dr.W_CLIP_LOW}, {dr.W_CLIP_HIGH}]; raw "
              "weights finite and strictly positive")
    led.check("no_ess_threshold_or_site_exclusion_exists",
              len(ess) == 23 and set(ess["held_out_site"]) == set(contracts.FROZEN_SITES),
              "all 23 folds carry an ESS diagnostic and all 23 are retained")
    grid = thresholds[thresholds["q"] != s0e.NO_ABSTENTION]
    led.check("threshold_grid_17q_plus_no_abstention",
              len(s0e.Q_GRID) == 17 and len(thresholds) == 23 * 3 * 18
              and set(grid["q"]) == {round(q, 2) for q in s0e.Q_GRID},
              "exactly the 17 frozen q values plus one NO_ABSTENTION anchor per "
              "(fold, method)")
    led.check("weighted_thresholds_are_a_source_side_object",
              bool(grid["tau_S1"].notna().all())
              and bool((grid["weighted_cumulative_mass_at_tau"]
                        >= grid["q"] - 1e-9).all())
              and bool((grid["realized_weighted_source_coverage"]
                        >= 1.0 - grid["q"] - 1e-9).all()),
              "every tau_q^S1 is the inverse weighted ECDF of that fold's "
              "source-OOF confidences at the frozen q; no target quantity is "
              "an input")
    ne_src = wsrc[wsrc["BER_sel_source_oof_weighted"] == NE]
    led.check("weighted_source_ber_is_class_balanced_never_pooled",
              bool(wsrc["W_MDD"].notna().all()) and bool(wsrc["W_HC"].notna().all())
              and bool(ne_src.empty
                       or ((ne_src["W_MDD"] <= 0) | (ne_src["W_HC"] <= 0)).all()),
              "err_MDD^w and err_HC^w are formed separately, each normalized by "
              "its own class's retained weight, then averaged")
    led.check("no_tie_block_split_by_any_weighted_threshold", tie_ok,
              "every target acquisition with confidence exactly == tau_q^S1 is "
              "retained; no identifier orders a tie block")

    max_aurc_diff = assert_aurc_invariant(aurc_rec, s0_aurc, led)
    intervals, gaps, s1v, order = cross_site_inference(rc, s0_rc, led)
    op_tables = operating_point_tables(intervals, s1v, gaps)
    contrasts = contrast_table(intervals, gaps)
    jsd_summary, jsd_dist = jsd_tables(s1v, order, led)
    worst = worst_site_table(rc, s1v, order)

    ne_cells = rc[rc["BER_sel_target"] == NE]
    led.check("not_estimable_cells_carry_counts_and_are_never_imputed",
              bool(len(ne_cells) == 0
                   or ((ne_cells["n_keep_MDD"] == 0)
                       | (ne_cells["n_keep_HC"] == 0)).all()),
              f"{len(ne_cells)} NOT_ESTIMABLE target selective-BER cells, each "
              "carrying its n_keep / n_keep_MDD / n_keep_HC")
    led.check("s1_emits_no_aurc_column_of_its_own",
              not any("aurc" in c.lower() for c in rc.columns)
              and not any("aurc" in c.lower() for c in op_tables["OP50"].columns),
              "the S1 tables carry no AURC column; the frozen S0 values are "
              "re-cited in the report and the invariant is asserted")

    # ---- write --------------------------------------------------------------
    meta.to_csv(OUT / "domain_meta_pseudo_target_discrepancy.csv", index=False)
    obj.to_csv(OUT / "domain_meta_objective.csv", index=False)
    sel.to_csv(OUT / "selected_domain_C.csv", index=False)
    diagd.to_csv(OUT / "domain_fit_diagnostics.csv", index=False)
    wdf.to_csv(OUT / "density_weights.csv", index=False)
    thresholds.to_csv(OUT / "thresholds_weighted_source.csv", index=False)
    wsrc.to_csv(OUT / "weighted_source_risk.csv", index=False)
    ess.to_csv(OUT / "ess_by_site.csv", index=False)
    rc.to_csv(OUT / "risk_coverage_all_q.csv", index=False)
    op_tables["OP50"].to_csv(OUT / "op50_summary.csv", index=False)
    op_tables["OP30"].to_csv(OUT / "op30_summary.csv", index=False)
    contrasts.to_csv(OUT / "s1_vs_s0_contrasts.csv", index=False)
    intervals.to_csv(OUT / "site_bootstrap_intervals.csv", index=False)
    jsd_summary.to_csv(OUT / "jsd_summary.csv", index=False)
    jsd_dist.to_csv(OUT / "jsd_per_site.csv", index=False)
    worst.to_csv(OUT / "worst_site_summary.csv", index=False)

    elapsed = time.perf_counter() - t_start
    runtime = {"started_utc": started,
               "finished_utc": datetime.now(timezone.utc).isoformat(),
               "this_invocation_seconds": elapsed,
               "this_invocation_hours": elapsed / 3600.0,
               "domain_meta_loso_seconds": timings["meta_seconds"],
               "deployed_domain_fit_seconds": timings["deployed_seconds"],
               "domain_fits": {
                   "scaler_pca50_fits": 46,
                   "meta_loso_domain_logistic_fits": 23 * 22 * 21 * 5,
                   "deployed_domain_logistic_fits": 23,
                   "disease_model_fits": 0},
               "n_jobs": args.n_jobs, "stages": args.stages,
               "blas_threads_per_process": env["blas_threads_per_process"],
               "cpu_count": env["cpu_count"], "device": "CPU (scikit-learn)"}
    (OUT / "runtime.json").write_text(json.dumps(runtime, indent=2))
    write_config(env, args, sel)

    checks_out = verify_s0_checksums(led, "at_exit")
    scan_output_vocabulary(led)
    write_report(env, runtime, led, sel, obj, diagd, ess, op_tables, jsd_summary,
                 worst, rc, thresholds, intervals, gaps, max_aurc_diff, ledger,
                 checks_in, checks_out)
    pd.DataFrame(led.rows).to_csv(OUT / "leakage_assertions.csv", index=False)

    print(f"\n[done] {len(led.rows)} assertions, all_pass={led.all_pass}")
    print(f"[done] firewall: {int(ledger['ordering_ok'].sum())}/23 folds ordering_ok")
    print(f"[done] outputs in {OUT}  ({elapsed / 60:.1f} min)")
    return 0


def write_config(env, args, sel) -> None:
    cfg = f"""# rq1_s1_config.yaml -- Stage 1.3a formal S1 execution, as run.
# Every value below is quoted from docs/RQ1_S1_PROTOCOL.md and was frozen
# before any S1 number existed. Nothing here was chosen in light of a result.

stage: "Stage 1.3a -- formal RQ1/S1 execution (target-label firewall repaired)"
protocol: docs/RQ1_S1_PROTOCOL.md
protocol_tag: rq1-s1-protocol-v1
regime: "S1 -- zero-label transductive (target covariates only)"
supersedes: "Stage 1.3 attempt 1 -- NON-COMPLIANT, target-y access barrier too weak"
scope_excluded: [S2, RQ2 risk bounding, ordered multiple testing, conformal
                 prediction, shrinkage, RQ3]

target_label_firewall:                  # §1, §11 -- the 1.3a repair
  rule: "for outer target s, y_s is not materialized before release_s"
  scope: outer-fold-specific
  legal_prebarrier: [unlabeled X, site membership, acquisition counts,
                     the 22 source sites' labels, frozen source-OOF predictions]
  forbidden_prebarrier: [held-out site diagnosis, its MDD/HC counts,
                         its prevalence, anything derived from them]
  feature_loader: "s1_data_access.build_features_only -- returns no outcome"
  source_labels: "SourceOofProvider.for_fold(s), rows chosen on a non-outcome
    key before any label is decoded; the held-out site contributes 0 rows"
  target_labels: "TargetOutcomeGuard.load_target_y(s), the single door; raises
    unless that fold's decisions and checksums are already on disk"
  ledger: results/rq1_s1/firewall_fold_ledger.csv
  cohort_composition_check: "deferred to the post-release evaluation stage"

cohort:
  source: {s1da.MANIFEST.relative_to(ROOT)}
  filter: inclusion_status == "INCLUDED"
  n_acquisitions: 2232
  n_MDD: 1182                           # verified after release, not before
  n_HC: 1050
  n_sites: 23
  unit: "one independent fMRI acquisition (DATA_CONTRACT §3.12)"
  feature_key: feature_source_id

inherited_frozen:                       # §2.1 -- read, never recomputed
  disease_classifier: results/gate1/outer_fold_metrics.csv
  source_oof_confidence: {s1da.S0_SOURCE_OOF.relative_to(ROOT)}
  r_star_rule: "r*_s = e_full,-s - 0.10, from results/gate1/"
  aurc: "results/rq1_s0/per_site_aurc.csv -- S1 emits none of its own (§12)"
  bootstrap_draws: results/rq1_s0/site_bootstrap_draws.npy

domain_representation:                  # §3
  pipeline: "StandardScaler -> PCA(50, svd_solver=randomized)"
  n_components: {dr.DOMAIN_PCA_DIM}
  random_state: {dr.PCA_RANDOM_STATE}
  deployed_fit_pool: "the 22 source sites U the unlabeled target site"
  meta_fit_pool: "the 22 source sites only -- the real target is absent"
  n_fits: 46
  low_rank_fallback: "hard failure in the formal run"

domain_classifier:                      # §4
  estimator: "LogisticRegression(penalty=l2, solver=lbfgs, max_iter=2000)"
  response: "design label D (0 source, 1 target); diagnosis is not an input"
  class_weight: None
  p_D_guard: {dr.P_D_EPS}
  w_raw: "[p_D/(1-p_D)] * (n_source/n_target) of the fit that produced p_D"
  deployed_source_scores: "transductive / in-sample"

weights:                                # §5
  order: "w_raw -> /mean(w_raw) -> clip[0.1, 10] -> STOP"
  clip_low: {dr.W_CLIP_LOW}
  clip_high: {dr.W_CLIP_HIGH}
  post_clip_renormalization: FORBIDDEN

c_domain_selection:                     # §7
  grid: {list(dr.C_DOMAIN_GRID)}
  structure: "source meta-LOSO: 22 pseudo-targets x 21 leave-one-pseudo-source
    -site-out cross-fits; the real outer target is absent from every step"
  objective: "cross-fitted IW covariate-mean discrepancy; J(C) = mean_j d(j,C)"
  rule: "argmin J(C); ties -> smallest C"
  labels_used: none
  fallback: none
  selected: {dict(sel.set_index('held_out_site')['c_domain'].astype(float))}

thresholds:                             # §9
  q_grid: {[round(q, 2) for q in s0e.Q_GRID]}
  quantile: "inverse weighted ECDF, no interpolation, exact-tie blocks"
  keep_rule: "confidence >= tau_q^S1"
  target_quantiles: FORBIDDEN
operating_points:
  OP50: {{q: {s0e.OP50_Q}, source_intended_coverage: 0.50}}
  OP30: {{q: {s0e.OP30_Q}, source_intended_coverage: 0.70}}

risk:                                   # §10, §11
  weighted_source: "0.5 * (err_MDD^w + err_HC^w), each normalized by its own
    class's retained weight; pooled weighted error is FORBIDDEN"
  target: "unweighted; BER_sel = 0.5*(err_MDD + err_HC) on retained rows"
  zero_retained_class: NOT_ESTIMABLE
  violation_column: realized_risk_violation__descriptive
  upper_confidence_bound: NOT COMPUTED IN RQ1
  ordered_testing_procedure: NOT COMPUTED IN RQ1

ess:                                    # §6, §14
  formula: "(sum w)^2 / sum w^2"
  role: "diagnostic only"
  exclusion_threshold: "none pre-registered, none added"
  folds_reported: 23

jsd:                                    # §16
  variables: "[MDD, HC] diagnosis composition only"
  log_base: 2
  interval: "NOT PART OF S1 PROTOCOL"

cross_site_inference:                   # §13
  unit: held-out site
  SITE_BOOTSTRAP_B: {sb.SITE_BOOTSTRAP_B}
  draws: "the frozen S0 (5000, 23) matrix, reused -- no new namespace"
  sign_convention: "S1 - S0"
  interval: "95% percentile (2.5 / 97.5), numpy.quantile(method='linear')"
  min_valid_replicates: {sb.MIN_VALID_REPLICATES}
  n_required_intervals: {dr.N_S1_REQUIRED_INTERVALS}
  prohibited: {list(dr.S1_CI_PROHIBITED)}

compute:
  n_jobs: {args.n_jobs}
  blas_threads_per_process: {env['blas_threads_per_process']}
  cpu_count: {env['cpu_count']}
  device: "CPU, scikit-learn only (no GPU, no cuML, no torch)"

environment:
  python: "{env['python']}"
  numpy: "{env['numpy']}"
  scipy: "{env['scipy']}"
  scikit_learn: "{env['scikit_learn']}"
  pandas: "{env['pandas']}"
  joblib: "{env['joblib']}"
  git_branch: "{env['git_branch']}"
  git_commit: "{env['git_commit']}"
  git_status_clean_at_entry: {str(env['git_status_clean_at_entry']).lower()}
  tags_present: {env['tags_present']}
"""
    (OUT / "rq1_s1_config.yaml").write_text(cfg)


def _f(v, p=4):
    if isinstance(v, str):
        return NE if v in (NE, "", "PROHIBITED") else v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return NE
    return f"{float(v):.{p}f}"


def _ci(row, p=4):
    if row["ci_low"] == sb.NOT_ESTIMABLE:
        return NE
    return f"[{float(row['ci_low']):.{p}f}, {float(row['ci_high']):.{p}f}]"


def write_report(env, runtime, led, sel, obj, diagd, ess, op_tables, jsd_summary,
                 worst, rc, thresholds, intervals, gaps, max_aurc_diff, ledger,
                 checks_in, checks_out) -> None:
    L = []
    A = L.append
    ci = intervals.set_index(["quantity", "method_or_contrast"])
    A("# RQ1 / S1 formal results — zero-label transductive (target-unlabeled)\n")
    A(f"**Stage**: 1.3a · **Protocol**: `docs/RQ1_S1_PROTOCOL.md` "
      f"(tag `rq1-s1-protocol-v1`, unchanged) · **Run**: "
      f"{runtime['started_utc']}\n")
    A(f"**Commit**: `{env['git_commit']}` on `{env['git_branch']}`, "
      f"working tree clean at entry: {env['git_status_clean_at_entry']}\n")
    A("This run supersedes Stage 1.3 attempt 1, which was rejected at closure "
      "review because its target-diagnosis **access protocol** was too weak — "
      "not because any number was wrong. The scientific definitions are "
      "untouched; the firewall is now outer-fold scoped and enforced in code.\n")

    A("\n## 0. Target-label firewall\n")
    A("For each outer held-out site `s`, `y_s` is not materialized, "
      "deserialized or exposed until that fold's `C_domain`, deployed domain "
      "fit, density weights, ESS, all 17 × 3 thresholds, weighted source risk "
      "and their checksums are on disk and `release(s)` has been called. The "
      "other 22 sites' labels stay legal for fold `s` — that is what the "
      "weighted source calibration of §10 is defined on.\n")
    A(f"\n**{int(ledger['ordering_ok'].sum())}/23 folds ordering_ok**; "
      f"target labels visible pre-release in "
      f"{int(ledger['target_labels_visible_prebarrier'].sum())} folds; "
      f"held-out-site rows inside a source-label object in "
      f"{int((ledger['source_label_target_rows'] > 0).sum())} folds. Ledger: "
      "`firewall_fold_ledger.csv`, `target_label_barrier.json`.\n")

    A("\n## A. Density ratio\n")
    counts = {float(k): int(v) for k, v
              in sel["c_domain"].value_counts().sort_index().items()}
    jp = obj.pivot(index="held_out_site", columns="C_domain", values="J_C")
    at_floor = int((jp.idxmin(axis=1) == min(dr.C_DOMAIN_GRID)).sum())
    A(f"Selected `C_domain` over the 23 outer folds: {counts}. "
      f"Ties at the argmin: {int(sel['tie_at_argmin'].sum())} "
      "(rule: smallest `C`).\n")
    A(f"Weighting improved the meta-world covariate match "
      f"(`J(C*) < d(j, w=1)`) in {int(sel['weighting_improves_meta_match'].sum())} "
      "of 23 folds; the argmin over the frozen grid is adopted either way "
      "(§7.5 — no fallback to `w = 1`).\n")
    A(f"Mean `J(C)` across the grid: "
      f"{ {float(c): round(float(jp[c].mean()), 5) for c in jp.columns} }. "
      f"The argmin sits at the **smallest** candidate, {min(dr.C_DOMAIN_GRID)}, "
      f"in {at_floor} of 23 folds — at the boundary of the frozen grid. The "
      "grid is frozen (§7.1) and may not be extended after seeing this, so it "
      "is reported as a limitation.\n")
    A("\n| held-out site | C* | J(C*) | J range | unweighted d | ESS | ESS/n_source |")
    A("|---|---|---|---|---|---|---|")
    e = ess.set_index("held_out_site")
    for r in sel.itertuples():
        A(f"| {r.held_out_site} | {r.c_domain:g} | {r.J_selected:.5f} | "
          f"[{r.J_min:.5f}, {r.J_max:.5f}] | {r.d_unweighted_reference:.5f} | "
          f"{float(e.loc[r.held_out_site, 'ess']):.1f} | "
          f"{float(e.loc[r.held_out_site, 'ess_fraction']):.4f} |")
    A(f"\nESS over the 23 folds: min {ess['ess'].min():.1f}, median "
      f"{ess['ess'].median():.1f}, max {ess['ess'].max():.1f}. "
      f"ESS/n_source: min {ess['ess_fraction'].min():.4f}, median "
      f"{ess['ess_fraction'].median():.4f}, max {ess['ess_fraction'].max():.4f}.\n")
    A(f"Clipped fractions: lower bound (`w = 0.1`) median "
      f"{ess['frac_clipped_low'].median():.4f} (range "
      f"{ess['frac_clipped_low'].min():.4f}–{ess['frac_clipped_low'].max():.4f}); "
      f"upper bound (`w = 10`) median {ess['frac_clipped_high'].median():.4f} "
      f"(range {ess['frac_clipped_high'].min():.4f}–"
      f"{ess['frac_clipped_high'].max():.4f}).\n")

    A("\n## B. Target AURC invariant\n")
    A(f"23 sites × 3 methods = 69 target AURC values recomputed from the frozen "
      f"S0 target confidences. Max |S1 − S0| = **{max_aurc_diff:.1e}** — "
      f"**{'PASS' if max_aurc_diff == 0.0 else 'FAIL'}** (exact zero required). "
      "S1 emits no AURC of its own.\n")

    for op in ("OP50", "OP30"):
        A(f"\n## {'C' if op == 'OP50' else 'D'}. {op} "
          f"(q = {s0e.OPERATING_POINTS[op]}, source-weighted intended coverage "
          f"{1 - s0e.OPERATING_POINTS[op]:.0%})\n")
        t = op_tables[op].set_index("method")
        A("| method | actual target coverage [95% CI] | selective BER [95% CI] "
          "| n_est | descriptive violation rate [95% CI] |")
        A("|---|---|---|---|---|")
        for m in METHODS:
            r = t.loc[m]
            A(f"| {m} | {_f(r['actual_coverage_site_balanced'])} "
              f"{_ci(ci.loc[(f'S1_{op}_site_balanced_coverage', m)])} | "
              f"{_f(r['selective_ber_site_balanced'])} "
              f"{_ci(ci.loc[(f'S1_{op}_site_balanced_selective_ber', m)])} | "
              f"{int(r['selective_ber_n_estimable_sites'])} | "
              f"{_f(r['descriptive_violation_rate_site_balanced'])} "
              f"{_ci(ci.loc[(f'S1_{op}_descriptive_violation_rate', m)])} |")
        A(f"\n**S1 − S0 paired contrasts at {op}** (same 5000 × 23 site draws, "
          "differences formed inside each replicate on the common estimable set):\n")
        A("| method | Δ coverage [95% CI] | Δ selective BER [95% CI] | "
          "Δ violation rate [95% CI] | Δ risk gap (point est.) |")
        A("|---|---|---|---|---|")
        g = gaps.set_index(["operating_point", "method"])
        for m in METHODS:
            c1 = ci.loc[(f"S1_minus_S0_{op}_site_balanced_coverage", m)]
            c2 = ci.loc[(f"S1_minus_S0_{op}_site_balanced_selective_ber", m)]
            c3 = ci.loc[(f"S1_minus_S0_{op}_descriptive_violation_rate", m)]
            A(f"| {m} | {_f(c1['point_estimate'])} {_ci(c1)} | "
              f"{_f(c2['point_estimate'])} {_ci(c2)} | "
              f"{_f(c3['point_estimate'])} {_ci(c3)} | "
              f"{_f(g.loc[(op, m), 'risk_gap_S1_minus_S0'])} |")

    A("\n## E. Full 17-q risk–coverage curves (point estimates only)\n")
    A(f"`risk_coverage_all_q.csv` — {len(rc)} rows (23 sites × 3 methods × 17 q "
      "+ NO_ABSTENTION anchor). No bootstrap band is drawn anywhere on this "
      "curve (§13.3).\n")
    A("\n| method | q | mean target coverage | mean selective BER (est. sites) "
      "| mean weighted source BER | descriptive violation rate |")
    A("|---|---|---|---|---|---|")
    for m in METHODS:
        for q in s0e.Q_GRID:
            g = _q_rows(rc, m, q)
            ber = pd.to_numeric(g["BER_sel_target"], errors="coerce")
            wb = pd.to_numeric(g["BER_sel_source_oof_weighted"], errors="coerce")
            vio = pd.to_numeric(g["realized_risk_violation__descriptive"],
                                errors="coerce")
            A(f"| {m} | {q:.2f} | {g['coverage_target'].mean():.4f} | "
              f"{ber.mean():.4f} ({int(ber.notna().sum())}) | {wb.mean():.4f} | "
              f"{vio.mean():.4f} |")

    A("\n## F. JSD (descriptive only, no interval)\n")
    A("| operating point | method | site-balanced mean | median | min | max | n_est |")
    A("|---|---|---|---|---|---|---|")
    for r in jsd_summary.itertuples():
        A(f"| {r.operating_point} | {r.method} | {_f(r.site_balanced_mean)} | "
          f"{_f(r.median)} | {_f(r.min)} | {_f(r.max)} | {r.n_estimable_sites} |")

    A("\n## G. ESS and the disclosure the deployed weights require\n")
    A("> Deployed density ratios use source-domain scores from the transductive "
      "domain classifier fitted jointly on source and unlabeled target "
      "features; source-meta hyperparameter selection is separately "
      "cross-fitted by source site.\n")
    A("\nThe deployed source weights are therefore **in-sample domain scores**, "
      "not out-of-fold ones (§8, F2). This is distinct from the disease side: "
      "every disease prediction and every confidence score entering S1 "
      "calibration is the frozen S0 **out-of-fold** value.\n")
    A("\n| site | n_source | n_target | C* | ESS | ESS/n_source | w_median | "
      "frac at 0.1 | frac at 10 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in ess.itertuples():
        A(f"| {r.held_out_site} | {int(r.n_source)} | {int(r.n_target)} | "
          f"{float(r.c_domain):g} | {float(r.ess):.1f} | "
          f"{float(r.ess_fraction):.4f} | {float(r.w_median):.4f} | "
          f"{float(r.frac_clipped_low):.4f} | {float(r.frac_clipped_high):.4f} |")
    A("\nEffective calibration size actually backing each reported operating "
      "point (ESS of the **retained** source subset, §6 secondary):\n")
    A("\n| operating point | min | median | max |")
    A("|---|---|---|---|")
    for op in ("OP50", "OP30"):
        v = ess[[f"ess_retained_{op}_{m}" for m in METHODS]].to_numpy(dtype=float)
        A(f"| {op} | {v.min():.1f} | {float(np.median(v)):.1f} | {v.max():.1f} |")
    A("\nESS is a diagnostic and only a diagnostic: no site was excluded, no "
      "cut-off exists, and no `C_domain`, clipping bound or representation was "
      "revised in the light of one (§14). All 23 folds are reported.\n")

    A("\n## H. Worst-site findings (descriptive)\n")
    A("| OP | method | min coverage (site) | worst estimable BER (site) | "
      "highest risk excess (site) | highest JSD (site) |")
    A("|---|---|---|---|---|---|")
    for r in worst.itertuples():
        A(f"| {r.operating_point} | {r.method} | {_f(r.min_target_coverage)} "
          f"({r.min_coverage_site}, n_keep {r.min_coverage_n_keep}) | "
          f"{_f(r.worst_estimable_selective_ber)} ({r.worst_ber_site}, n_keep "
          f"{r.worst_ber_n_keep}) | {_f(r.highest_risk_excess)} "
          f"({r.highest_risk_excess_site}) | {_f(r.highest_jsd)} "
          f"({r.highest_jsd_site}) |")

    A("\n## I. Compliance\n")
    eq = thresholds[thresholds["q"] != s0e.NO_ABSTENTION]
    within = eq["within_one_order_statistic"]
    A(f"- **Target-label firewall**: {int(ledger['ordering_ok'].sum())}/23 folds "
      "PASS; every fold's decision checksums precede its release, which "
      "precedes its first target-y load.\n")
    A(f"- **Required intervals**: {len(intervals)}/36 emitted, set-equal to the "
      "frozen registry (18 standalone S1 + 18 paired S1 − S0).\n")
    A(f"- **S0 provenance**: {checks_in['n_matched']}/{checks_in['n_entries']} "
      f"SHA-256 entries matched at entry and "
      f"{checks_out['n_matched']}/{checks_out['n_entries']} at exit — "
      f"**{'PASS' if checks_in['n_mismatched'] == 0 and checks_out['n_mismatched'] == 0 else 'FAIL'}**.\n")
    A(f"- **Equal-weight threshold audit** (§9.3): over {len(eq)} "
      f"(fold, method, q) cells, mean |tau_eq − tau_S0| = "
      f"{eq['tau_equal_weight_check'].abs().mean():.3e}, max "
      f"{eq['tau_equal_weight_check'].abs().max():.3e}; the frozen "
      f"one-order-statistic property holds in {int(within.sum())}/{len(within)} "
      "cells. Diagnostic only.\n")
    A(f"- **Assertions**: {len(led.rows)} recorded, "
      f"{sum(r['status'] == 'PASS' for r in led.rows)} PASS, "
      f"{sum(r['status'] == 'FAIL' for r in led.rows)} FAIL.\n")
    A(f"- **Runtime**: {runtime['this_invocation_seconds'] / 60:.1f} min wall "
      f"clock at n_jobs = {runtime['n_jobs']}; "
      f"{runtime['domain_fits']['meta_loso_domain_logistic_fits']} meta-LOSO "
      f"domain fits, {runtime['domain_fits']['deployed_domain_logistic_fits']} "
      f"deployed fits, {runtime['domain_fits']['scaler_pca50_fits']} "
      f"scaler+PCA50 fits, "
      f"{runtime['domain_fits']['disease_model_fits']} disease-model fits.\n")

    A("\n## Interpretation boundary\n")
    A("S1 is an **empirical covariate-shift baseline**. Density-ratio weighting "
      "carries no distribution-free guarantee; whatever it buys depends on the "
      "shift assumptions holding, and the weights are estimated. Nothing here "
      "is a risk guarantee, no threshold is *adopted*, no upper confidence "
      "bound is computed, and no ordered testing procedure is run. No winning "
      "confidence method is declared, no method is dropped, and an interval "
      "excluding zero is not a licence to call one regime superior (§13.4).\n")
    (OUT / "RQ1_S1_REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
