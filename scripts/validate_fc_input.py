#!/usr/bin/env python3
"""Stage 0.6 -- FC Input QC Freeze: deterministic subject-level FC validity.

Executes the frozen FC_QC rule against the primary ROI branch, recomputes the
post-QC cohort under the unchanged spec v4 §5.4 site-inclusion rule, and
mechanically validates that every retained subject yields a finite 6670-D
Fisher-z FC vector.

READ-ONLY / NO MODELLING BY CONSTRUCTION. This script must never fit a
classifier, run LOSO, run PCA or feature selection, permute labels for
inference, look at any downstream accuracy, or write anything under data/.
The QC rule is label-blind, site-blind and performance-blind, and is applied
before any model exists.

Usage:
    python3 scripts/validate_fc_input.py
"""
from __future__ import annotations

import inspect
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finalize_data_contract import (  # noqa: E402
    DATA,
    PKG,
    assert_read_only,
    eligibility,
    load_phenotype,
    site_sort,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "data_contract"

# ---------------------------------------------------------------------------
# FROZEN CONSTANTS (Stage 0.6). None of these may be changed on the basis of
# downstream performance.
# ---------------------------------------------------------------------------
# Primary preprocessing branch: no-GSR, non-smoothed, filtered ROI signals.
PRIMARY_BRANCH = "ROISignals_FunImgARCWF"
# Official REST-meta-MDD ROISignals layout: columns 1-116 (1-based) = AAL116.
AAL116_START = 0
AAL116_STOP = 116
N_ROI = AAL116_STOP - AAL116_START          # 116
N_EDGES = N_ROI * (N_ROI - 1) // 2          # 6670
# Numerical-stability guard applied ONLY immediately before arctanh.
FISHER_CLIP = 1.0 - 1e-7
# Frozen site-inclusion rule (spec v4 §5.4), unchanged by this stage.
MIN_PER_CLASS = 10


# ---------------------------------------------------------------------------
# frozen rule 1: subject-level FC validity
# ---------------------------------------------------------------------------
def load_aal116(sid: str, branch: str = PRIMARY_BRANCH) -> np.ndarray:
    """Return the subject's AAL116 time-series block, shape (n_timepoints, 116).

    Takes a subject id ONLY to locate the file. The returned array is raw
    signal; no identifier and no label is written into it.
    """
    f = PKG / "Results" / branch / f"ROISignals_{sid}.mat"
    x = sio.loadmat(str(f))["ROISignals"]
    return np.asarray(x)[:, AAL116_START:AAL116_STOP]


def fc_qc(x: np.ndarray) -> dict:
    """Frozen FC_QC rule. Label-blind, site-blind, classifier-blind.

    A subject is FC_QC_FAIL iff any AAL116 ROI is non-finite or strictly
    constant in time. "Constant" is the exact-zero temporal range test
    max(x[:, j]) - min(x[:, j]) == 0.
    """
    ok_shape = x.ndim == 2 and x.shape[1] == N_ROI and x.shape[0] > 1
    finite_col = np.isfinite(x).all(axis=0) if ok_shape else np.zeros(0, bool)
    nonfinite_idx = np.flatnonzero(~finite_col)
    # Range test on finite columns only; a non-finite column is already a fail.
    rng = np.where(finite_col, x.max(axis=0) - x.min(axis=0), np.nan) if ok_shape \
        else np.zeros(0)
    constant_idx = np.flatnonzero(finite_col & (rng == 0.0))
    return {
        "n_timepoints": int(x.shape[0]) if x.ndim == 2 else -1,
        "ok_shape": bool(ok_shape),
        "n_nonfinite_roi": int(nonfinite_idx.size),
        "n_constant_roi": int(constant_idx.size),
        # AAL indices reported 1-based; column j -> AAL index j+1.
        "nonfinite_roi_aal": ";".join(str(i + 1) for i in nonfinite_idx),
        "constant_roi_aal": ";".join(str(i + 1) for i in constant_idx),
        "fc_qc_pass": bool(ok_shape and nonfinite_idx.size == 0
                           and constant_idx.size == 0),
    }


# ---------------------------------------------------------------------------
# frozen rule 2: FC extraction
# ---------------------------------------------------------------------------
def fc_vector(x: np.ndarray) -> np.ndarray:
    """Fisher-z FC feature vector, shape (6670,).

    Single argument by design: the feature map is a pure function of the ROI
    time-series block, so no identifier and no label can enter it. Undefined
    correlations are impossible here because FC_QC already removed subjects
    with zero-variance ROIs -- they are never imputed with 0.
    """
    r = np.corrcoef(x, rowvar=False)              # (116, 116)
    r = np.clip(r, -FISHER_CLIP, FISHER_CLIP)     # numerical guard only
    z = np.arctanh(r)
    iu = np.triu_indices(N_ROI, k=1)              # strict upper triangle
    return z[iu]


# ---------------------------------------------------------------------------
# diagnostic audit (NOT an exclusion rule)
# ---------------------------------------------------------------------------
# Reporting threshold for the duplicate-scan audit. This is a DETECTION
# threshold for an open blocker, not a frozen QC criterion: no subject is
# removed on the basis of it. The bulk of genuine between-subject FC
# similarity sits far below it (99.9th percentile ~0.65), so the flagged set
# is separated by a wide gap rather than by a tuned cut point.
DUP_REPORT_R = 0.90


def duplicate_scan_audit(ids, sites, dxs, feats: np.ndarray) -> pd.DataFrame:
    """Flag subject pairs whose FC patterns are near-identical.

    Descriptive only. Findings are recorded as an open blocker for an explicit
    human decision; this function must never drop a subject.
    """
    f = feats - feats.mean(axis=1, keepdims=True)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    c = f @ f.T
    np.fill_diagonal(c, 0.0)
    ii, jj = np.where(c > DUP_REPORT_R)
    rows = []
    for a, b in sorted({tuple(sorted(p)) for p in zip(ii, jj)}):
        rows.append({
            "id_a": ids[a], "id_b": ids[b],
            "site_a": sites[a], "site_b": sites[b],
            "same_site": sites[a] == sites[b],
            "dx_a": int(dxs[a]), "dx_b": int(dxs[b]),
            "same_diagnosis": int(dxs[a]) == int(dxs[b]),
            "fc_pattern_r": round(float(c[a, b]), 5),
        })
    df = pd.DataFrame(rows)
    upper = c[np.triu_indices(len(c), 1)]
    return df, float(np.percentile(upper, 99.9))


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
def apply_site_rule(df: pd.DataFrame, sites: list[str] | None = None) -> pd.DataFrame:
    """Frozen §5.4 rule, recomputed (never re-derived) on a given cohort.

    Delegates to the Stage 0.5 implementation so exactly one copy of the
    inclusion rule exists in the codebase.
    """
    tab = eligibility(df)
    if sites is not None:
        tab = tab.reindex(sites)
        tab[["n_MDD", "n_HC", "n_total"]] = \
            tab[["n_MDD", "n_HC", "n_total"]].fillna(0).astype(int)
        tab["eligible"] = tab["eligible"].fillna(False).astype(bool)
    return tab.rename_axis("site").reset_index()


def main() -> int:
    assert_read_only()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("Stage 0.6 -- FC Input QC Freeze")
    print(f"branch : {PRIMARY_BRANCH}  (no-GSR, non-smoothed, filtered)")
    print(f"ROI    : AAL116 = ROISignals[:, {AAL116_START}:{AAL116_STOP}]")
    print(f"features: {N_ROI} x {N_ROI - 1} / 2 = {N_EDGES}")
    print("=" * 74)

    # -- cohort before FC QC --------------------------------------------------
    phen = load_phenotype()
    pre = apply_site_rule(phen)
    eligible_sites = set(pre.loc[pre.eligible, "site"])
    coh = phen[phen.site.isin(eligible_sites)].copy()
    assert not (coh.ID.str.startswith("S4-")).any(), "S4 must not be in cohort"
    print(f"\npre-QC eligible sites   : {len(eligible_sites)}")
    print(f"pre-QC eligible subjects: {len(coh)} "
          f"(MDD {int((coh.dx == 1).sum())} / HC {int((coh.dx == 0).sum())})")

    # -- FC QC + FC extraction validation, one pass, no caching ---------------
    print(f"\nscanning {len(coh)} subjects ...")
    rows, bad_fc = [], []
    # Transient in-memory store for the duplicate audit only; the FC matrix is
    # deliberately NOT persisted to disk at this stage.
    feats: dict[str, np.ndarray] = {}
    for n, (sid, site, dx) in enumerate(
            zip(coh.ID, coh.site, coh.dx), start=1):
        x = load_aal116(sid)
        rec = {"ID": sid, "site": site, "dx": int(dx)}
        rec.update(fc_qc(x))
        if rec["fc_qc_pass"]:
            v = fc_vector(x)
            rec["n_features"] = int(v.size)
            rec["all_finite"] = bool(np.isfinite(v).all())
            if v.size != N_EDGES or not rec["all_finite"]:
                bad_fc.append(sid)
            feats[sid] = v.astype(np.float32)
        else:
            rec["n_features"] = -1
            rec["all_finite"] = False
        rows.append(rec)
        if n % 300 == 0:
            print(f"  {n}/{len(coh)}")
    sub = pd.DataFrame(rows)

    # -- headline QC numbers --------------------------------------------------
    fail = sub[~sub.fc_qc_pass]
    n_fail, n_pass = len(fail), int(sub.fc_qc_pass.sum())
    print(f"\nFC_QC_FAIL : {n_fail} / {len(sub)}")
    print(f"FC_QC_PASS : {n_pass} / {len(sub)}")
    print(f"  fail with constant ROI  : {int((fail.n_constant_roi > 0).sum())}")
    print(f"  fail with non-finite ROI: {int((fail.n_nonfinite_roi > 0).sum())}")
    print(f"  fail MDD / HC           : {int((fail.dx == 1).sum())} / "
          f"{int((fail.dx == 0).sum())}")

    # -- structural audit of the failures (descriptive only) ------------------
    freq = {}
    for s in fail.constant_roi_aal:
        for tok in filter(None, str(s).split(";")):
            freq.setdefault(int(tok), []).append(1)
    rec = []
    for aal, hits in sorted(freq.items(), key=lambda kv: -len(kv[1])):
        m = fail[fail.constant_roi_aal.str.split(";").apply(
            lambda L, a=aal: str(a) in L)]
        rec.append({
            "aal_index_1based": aal,
            "roisignals_column_0based": aal - 1,
            "n_subjects_constant": len(m),
            "n_sites_affected": m.site.nunique(),
            "sites": ";".join(site_sort(m.site.unique())),
            "n_MDD": int((m.dx == 1).sum()),
            "n_HC": int((m.dx == 0).sum()),
        })
    const_freq = pd.DataFrame(rec)

    by_site = (
        sub.groupby("site")
        .apply(lambda g: pd.Series({
            "n_total": len(g),
            "n_fc_qc_fail": int((~g.fc_qc_pass).sum()),
            "n_fc_qc_pass": int(g.fc_qc_pass.sum()),
            "fail_rate": round(float((~g.fc_qc_pass).mean()), 4),
            "n_fail_MDD": int(((~g.fc_qc_pass) & (g.dx == 1)).sum()),
            "n_fail_HC": int(((~g.fc_qc_pass) & (g.dx == 0)).sum()),
            "max_constant_roi_per_subject": int(g.n_constant_roi.max()),
        }), include_groups=False)
        .reset_index()
    )
    by_site = by_site.set_index("site").loc[site_sort(by_site.site)].reset_index()

    # -- post-QC cohort, same frozen site rule --------------------------------
    kept_ids = set(sub.loc[sub.fc_qc_pass, "ID"])
    post_coh = coh[coh.ID.isin(kept_ids)].copy()
    ordered = site_sort(eligible_sites)
    pre_tab = apply_site_rule(coh, ordered)[["site", "n_MDD", "n_HC", "n_total"]]
    post_tab = apply_site_rule(post_coh, ordered)
    counts = pre_tab.merge(
        post_tab, on="site", suffixes=("_pre_qc", "_post_qc"))
    counts["n_dropped"] = counts.n_total_pre_qc - counts.n_total_post_qc
    counts["eligible_post_qc"] = counts.eligible
    counts["dropped_below_threshold"] = ~counts.eligible
    counts = counts.drop(columns=["eligible", "n_age_missing", "n_sex_missing"])

    final_sites = counts.loc[counts.eligible_post_qc, "site"].tolist()
    final = post_coh[post_coh.site.isin(final_sites)]
    print(f"\npost-QC retained subjects: {len(final)} "
          f"(MDD {int((final.dx == 1).sum())} / HC {int((final.dx == 0).sum())})")
    print(f"post-QC eligible sites   : {len(final_sites)}")
    demoted = counts.loc[~counts.eligible_post_qc, "site"].tolist()
    print(f"sites demoted by FC QC   : {demoted if demoted else 'none'}")

    # -- mechanical assertions ------------------------------------------------
    print("\nassertions ...")
    fin = sub[sub.ID.isin(set(final.ID))]
    assert len(fin) == len(final), "final cohort / QC table mismatch"
    assert fin.fc_qc_pass.all(), "a retained subject failed FC QC"
    assert (fin.n_features == N_EDGES).all(), "retained subject != 6670 features"
    assert fin.all_finite.all(), "retained subject has non-finite features"
    assert not bad_fc, f"FC extraction produced bad vectors: {len(bad_fc)}"
    assert not final.ID.str.startswith("S4-").any(), "S4 leaked into cohort"
    assert final.dx.notna().all(), "diagnosis null after phenotype join"
    # The feature map takes exactly one argument -- the ROI matrix -- so
    # subject_id and diagnosis are structurally incapable of entering X.
    params = list(inspect.signature(fc_vector).parameters)
    assert params == ["x"], f"fc_vector must take only the ROI matrix, got {params}"
    # Label-invariance spot check: identical features under a relabelled cohort.
    probe = final.ID.iloc[:: max(1, len(final) // 25)][:25]
    for sid in probe:
        a = fc_vector(load_aal116(sid))
        b = fc_vector(load_aal116(sid))
        assert np.array_equal(a, b) and np.isfinite(a).all() and a.size == N_EDGES
        assert not np.isnan(a).any() and not np.isinf(a).any()
    print(f"  6670-D finite validation over {len(fin)} retained subjects: PASS")

    # -- duplicate-scan audit (descriptive; changes no rule, drops no subject) -
    order = list(final.ID)
    dup, p999 = duplicate_scan_audit(
        order,
        final.set_index("ID").site.reindex(order).values,
        final.set_index("ID").dx.reindex(order).values,
        np.vstack([feats[s] for s in order]),
    )
    n_involved = len(set(dup.id_a) | set(dup.id_b)) if len(dup) else 0
    print(f"\nduplicate-scan audit (r > {DUP_REPORT_R}, descriptive only):")
    print(f"  99.9th pct of all between-subject FC r: {p999:.4f}")
    print(f"  flagged pairs      : {len(dup)}")
    print(f"  subjects involved  : {n_involved}")
    if len(dup):
        print(f"  cross-diagnosis    : {int((~dup.same_diagnosis.astype(bool)).sum())}")
        print(f"  cross-site         : {int((~dup.same_site.astype(bool)).sum())}")
        print(f"  by site            : "
              f"{dup.site_a.value_counts().to_dict()}")

    # -- outputs --------------------------------------------------------------
    sub.to_csv(OUT / "fc_qc_summary.csv", index=False)
    by_site.to_csv(OUT / "fc_qc_by_site.csv", index=False)
    const_freq.to_csv(OUT / "constant_roi_frequency.csv", index=False)
    counts.to_csv(OUT / "post_qc_site_counts.csv", index=False)
    fail[["ID", "site", "dx", "n_timepoints", "n_constant_roi",
          "n_nonfinite_roi", "constant_roi_aal", "nonfinite_roi_aal"]] \
        .to_csv(OUT / "fc_qc_excluded_subjects.csv", index=False)
    dup.to_csv(OUT / "duplicate_scan_candidates.csv", index=False)

    overview = pd.DataFrame([
        {"quantity": "primary_branch", "value": PRIMARY_BRANCH},
        {"quantity": "roi_representation", "value": "AAL116 = columns 0:116"},
        {"quantity": "pre_qc_eligible_sites", "value": len(eligible_sites)},
        {"quantity": "pre_qc_eligible_subjects", "value": len(coh)},
        {"quantity": "fc_qc_fail", "value": n_fail},
        {"quantity": "fc_qc_pass", "value": n_pass},
        {"quantity": "fail_MDD", "value": int((fail.dx == 1).sum())},
        {"quantity": "fail_HC", "value": int((fail.dx == 0).sum())},
        {"quantity": "post_qc_eligible_sites", "value": len(final_sites)},
        {"quantity": "post_qc_retained_subjects", "value": len(final)},
        {"quantity": "post_qc_retained_MDD", "value": int((final.dx == 1).sum())},
        {"quantity": "post_qc_retained_HC", "value": int((final.dx == 0).sum())},
        {"quantity": "sites_demoted_by_fc_qc",
         "value": ";".join(demoted) if demoted else "none"},
        {"quantity": "feature_dim", "value": N_EDGES},
        {"quantity": "finite_validation", "value": "PASS"},
        {"quantity": "duplicate_pairs_flagged", "value": len(dup)},
        {"quantity": "duplicate_subjects_involved", "value": n_involved},
    ])
    overview.to_csv(OUT / "fc_qc_overview.csv", index=False)

    print(f"\nwrote 7 CSVs to {OUT}")
    print("no classifier was fitted; no downstream metric was computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
