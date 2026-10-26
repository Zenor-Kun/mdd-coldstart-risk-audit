#!/usr/bin/env python3
"""Stage 0.7 -- Official cohort and duplicate reconciliation.

Cross-checks the frozen Stage 0.6 cohort against the *official* Yan et al.
(2019, PNAS) REST-meta-MDD subject lists published in
Chaogan-Yan/PaperScripts/Yan_2019_PNAS/StatsSubInfo, and gathers
diagnosis-independent imaging evidence about the 41 within-site duplicate-scan
candidates found in Stage 0.6.

IDENTITY AUDIT ONLY -- NO MODELLING BY CONSTRUCTION. This script must never
fit a classifier, run LOSO, run PCA or feature selection, permute labels,
look at any downstream accuracy, or write anything under data/. Nothing here
changes the frozen cohort: Stage 0.7 produces evidence and *simulates* the
consequences of three pre-declared de-duplication strategies. Adopting one is
an explicit human decision, recorded separately.

Network use is download-only and public: it fetches the consortium's own
published .mat subject lists from raw.githubusercontent.com. No phenotype,
no subject identifier and no derived value is ever uploaded.

Usage:
    python3 scripts/reconcile_official_subjects_and_duplicates.py
"""
from __future__ import annotations

import hashlib
import itertools
import sys
import urllib.request
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
    load_phenotype,
    site_sort,
)
from validate_fc_input import (  # noqa: E402
    PRIMARY_BRANCH,
    apply_site_rule,
    fc_vector,
    load_aal116,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "data_contract"
# Reference material downloaded from a public repository. Kept outside
# results/data_contract/ because .mat blobs are not contract evidence; the
# parsed, reviewable form is official_subject_reconciliation.csv.
EXTERNAL = ROOT / "results" / "external" / "yan2019"

OFFICIAL_BASE = (
    "https://raw.githubusercontent.com/Chaogan-Yan/PaperScripts/master/"
    "Yan_2019_PNAS/StatsSubInfo"
)
# The two official lists that carry a subject-level inclusion decision.
#   Info_1789_943_846      -- the sample surviving the consortium's own image
#                             quality exclusions (943 MDD / 846 NC).
#   Stat_Sub_Info_848MDD.. -- the narrower primary MDD-vs-NC analysis sample.
# The first is the reference for "did the consortium consider this scan
# usable"; the second additionally reflects analysis-specific restrictions.
OFFICIAL_FILES = {
    "official_1789": "Info_1789_943_846.mat",
    "official_1642": "Stat_Sub_Info_848MDDvs794NC.mat",
}

# GSR branch, used ONLY as an independently-processed cross-check of scan
# identity. This is not a change to the frozen primary branch (§6 of the data
# contract): no model, no label and no accuracy is involved.
GSR_BRANCH = "ROISignals_FunImgARglobalCWF"

# Negative-reference sampling. Fixed seed; the sample is drawn before any
# similarity value is looked at.
NEG_SEED = 20270807
NEG_PAIRS_PER_SITE = 300


# ---------------------------------------------------------------------------
# Task 1 -- official Yan 2019 subject lists
# ---------------------------------------------------------------------------
def fetch_official() -> dict[str, Path]:
    """Download the public subject lists if absent. Returns local paths."""
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, name in OFFICIAL_FILES.items():
        p = EXTERNAL / name
        if not p.exists():
            url = f"{OFFICIAL_BASE}/{name}"
            print(f"  downloading {name} ...")
            with urllib.request.urlopen(url, timeout=60) as r:
                p.write_bytes(r.read())
        paths[key] = p
    return paths


def parse_official(path: Path) -> pd.DataFrame:
    """Parse one StatsSubInfo .mat into a tidy frame.

    Dx is coded 1 = MDD, -1 = NC in the official files; it is recoded to the
    project convention (1 = MDD, 0 = HC) purely for comparison.
    """
    m = sio.loadmat(str(path))
    ids = [str(x[0][0]) for x in m["SubID"]]
    return pd.DataFrame({
        "ID": ids,
        "site_official": ["S" + str(s) for s in m["Site"].ravel()],
        "dx_official": [1 if d == 1 else 0 for d in m["Dx"].ravel()],
        "age_official": m["Age"].ravel().astype(float),
        "sex_official": m["Sex"].ravel().astype(int),
        "motion_official": m["Motion"].ravel().astype(float),
    })


def reconcile_official(phen: pd.DataFrame, official: dict[str, pd.DataFrame],
                       qc: pd.DataFrame) -> pd.DataFrame:
    """Subject-level union of phenotype and official lists.

    The official list is reference/QC evidence. It is recorded alongside the
    frozen cohort, never substituted for it.
    """
    ref = official["official_1789"]
    ids = sorted(set(phen.ID) | set(ref.ID),
                 key=lambda s: (int(s.split("-")[0][1:]), s))
    df = pd.DataFrame({"ID": ids})
    df["site"] = df.ID.str.split("-").str[0]
    df = df.merge(phen[["ID", "dx", "Age", "Sex"]], on="ID", how="left")
    df = df.rename(columns={"dx": "dx_phenotype"})
    df["in_phenotype"] = df.ID.isin(set(phen.ID))
    for key, o in official.items():
        df[f"in_{key}"] = df.ID.isin(set(o.ID))
    df = df.merge(ref[["ID", "dx_official"]], on="ID", how="left")
    df["dx_agree"] = np.where(
        df.in_phenotype & df.in_official_1789,
        df.dx_phenotype == df.dx_official,
        np.nan,
    )
    df = df.merge(qc[["ID", "fc_qc_pass"]], on="ID", how="left")
    return df


# ---------------------------------------------------------------------------
# Task 4 -- diagnosis-independent imaging evidence
# ---------------------------------------------------------------------------
def load_rp(sid: str) -> np.ndarray | None:
    """Realignment parameters, shape (n_volumes, 6), or None if absent.

    This is the strongest available identity evidence that is independent of
    the ROI pipeline: rp.txt is estimated directly from the raw 4-D series,
    upstream of nuisance regression, parcellation and any label.
    """
    f = PKG / "RealignParameter" / sid / "rp.txt"
    if not f.exists():
        return None
    try:
        return np.loadtxt(f)
    except Exception:
        return None


def load_fd_series(sid: str) -> np.ndarray | None:
    f = PKG / "RealignParameter" / sid / f"FD_Jenkinson_{sid}.txt"
    if not f.exists():
        return None
    try:
        return np.loadtxt(f).ravel()
    except Exception:
        return None


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size != b.size or a.size < 2:
        return np.nan
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


class SignalCache:
    """Lazily loads per-subject signals once, for the audited sites only."""

    def __init__(self) -> None:
        self._aal: dict[str, np.ndarray] = {}
        self._fc: dict[str, np.ndarray] = {}
        self._gsr: dict[str, np.ndarray] = {}
        self._rp: dict[str, np.ndarray | None] = {}
        self._fd: dict[str, np.ndarray | None] = {}

    def aal(self, sid):
        if sid not in self._aal:
            self._aal[sid] = load_aal116(sid)
        return self._aal[sid]

    def fc(self, sid):
        if sid not in self._fc:
            self._fc[sid] = fc_vector(self.aal(sid))
        return self._fc[sid]

    def gsr_fc(self, sid):
        if sid not in self._gsr:
            try:
                self._gsr[sid] = fc_vector(load_aal116(sid, branch=GSR_BRANCH))
            except Exception:
                self._gsr[sid] = None
        return self._gsr[sid]

    def rp(self, sid):
        if sid not in self._rp:
            self._rp[sid] = load_rp(sid)
        return self._rp[sid]

    def fd(self, sid):
        if sid not in self._fd:
            self._fd[sid] = load_fd_series(sid)
        return self._fd[sid]


def pair_evidence(a: str, b: str, cache: SignalCache) -> dict:
    """Similarity metrics for one subject pair. Never touches diagnosis."""
    xa, xb = cache.aal(a), cache.aal(b)
    same_len = xa.shape[0] == xb.shape[0]
    out = {
        "n_timepoints_a": int(xa.shape[0]),
        "n_timepoints_b": int(xb.shape[0]),
        "same_scan_length": bool(same_len),
        "fc_r": _corr(cache.fc(a), cache.fc(b)),
    }

    # per-ROI raw time-series correspondence (only meaningful at equal length)
    if same_len:
        za = xa - xa.mean(0)
        zb = xb - xb.mean(0)
        na = np.linalg.norm(za, axis=0)
        nb = np.linalg.norm(zb, axis=0)
        ok = (na > 0) & (nb > 0)
        r = np.full(xa.shape[1], np.nan)
        r[ok] = (za[:, ok] * zb[:, ok]).sum(0) / (na[ok] * nb[ok])
        out.update({
            "roi_ts_r_mean": float(np.nanmean(r)),
            "roi_ts_r_median": float(np.nanmedian(r)),
            "roi_ts_r_min": float(np.nanmin(r)),
            "roi_ts_frac_gt_0.99": float(np.nanmean(r > 0.99)),
            "ts_flat_r": _corr(xa, xb),
        })
    else:
        out.update({k: np.nan for k in
                    ["roi_ts_r_mean", "roi_ts_r_median", "roi_ts_r_min",
                     "roi_ts_frac_gt_0.99", "ts_flat_r"]})

    # independent pipeline: GSR branch FC
    ga, gb = cache.gsr_fc(a), cache.gsr_fc(b)
    out["gsr_fc_r"] = _corr(ga, gb) if ga is not None and gb is not None else np.nan

    # fully upstream evidence: realignment parameters and FD time course
    ra, rb = cache.rp(a), cache.rp(b)
    if ra is not None and rb is not None and ra.shape == rb.shape:
        out["rp_r"] = _corr(ra, rb)
        out["rp_max_abs_diff"] = float(np.abs(ra - rb).max())
        out["rp_n_volumes"] = int(ra.shape[0])
    else:
        out["rp_r"] = np.nan
        out["rp_max_abs_diff"] = np.nan
        out["rp_n_volumes"] = int(ra.shape[0]) if ra is not None else -1

    fa, fb = cache.fd(a), cache.fd(b)
    out["fd_series_r"] = (_corr(fa, fb) if fa is not None and fb is not None
                          and fa.size == fb.size else np.nan)
    out["mean_fd_a"] = float(np.nanmean(fa)) if fa is not None else np.nan
    out["mean_fd_b"] = float(np.nanmean(fb)) if fb is not None else np.nan
    out["mean_fd_abs_diff"] = abs(out["mean_fd_a"] - out["mean_fd_b"]) \
        if np.isfinite(out["mean_fd_a"]) and np.isfinite(out["mean_fd_b"]) else np.nan
    return out


def negative_reference(sites: list[str], phen: pd.DataFrame,
                       exclude: set[str], cache: SignalCache) -> pd.DataFrame:
    """Same metrics on random non-candidate within-site pairs.

    Sampling is seeded and drawn before any similarity is computed, so the
    reference distribution cannot be shaped by what it is compared against.
    """
    rng = np.random.default_rng(NEG_SEED)
    rows = []
    for s in sites:
        ids = sorted(phen.loc[phen.site == s, "ID"])
        allp = [p for p in itertools.combinations(ids, 2) if p not in exclude]
        if not allp:
            continue
        k = min(NEG_PAIRS_PER_SITE, len(allp))
        idx = rng.choice(len(allp), size=k, replace=False)
        for i in idx:
            a, b = allp[int(i)]
            rows.append({"site": s, "id_a": a, "id_b": b,
                         **pair_evidence(a, b, cache)})
    return pd.DataFrame(rows)


def distribution_summary(df: pd.DataFrame, group: str,
                         metrics: list[str]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        v = pd.to_numeric(df[m], errors="coerce").dropna().to_numpy()
        if v.size == 0:
            continue
        rows.append({
            "group": group, "metric": m, "n_pairs": int(v.size),
            "min": v.min(), "p50": np.percentile(v, 50),
            "p95": np.percentile(v, 95), "p99": np.percentile(v, 99),
            "p99.9": np.percentile(v, 99.9), "max": v.max(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task 5 -- duplicate graph
# ---------------------------------------------------------------------------
def connected_components(pairs: list[tuple[str, str]]) -> list[list[str]]:
    """Union-find over the undirected duplicate-candidate graph.

    Pairs are NOT assumed independent: a subject may appear in several pairs,
    in which case pairwise handling would double-count.
    """
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps: dict[str, list[str]] = {}
    for node in parent:
        comps.setdefault(find(node), []).append(node)
    return [sorted(v) for v in sorted(comps.values(), key=lambda c: sorted(c)[0])]


# ---------------------------------------------------------------------------
# Tasks 6-7 -- pre-declared strategies, simulated only
# ---------------------------------------------------------------------------
def classify_components(comps: list[list[str]], official: set[str],
                        dx: dict[str, int]) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(comps, 1):
        keep = [s for s in c if s in official]
        rows.append({
            "component_id": i,
            "site": c[0].split("-")[0],
            "size": len(c),
            "members": ";".join(c),
            "n_in_official_1789": len(keep),
            "official_resolves": len(keep) == 1,
            "n_MDD": sum(dx.get(s, -1) == 1 for s in c),
            "n_HC": sum(dx.get(s, -1) == 0 for s in c),
            "diagnosis_conflict": len({dx[s] for s in c if s in dx}) > 1,
        })
    return pd.DataFrame(rows)


def simulate_strategies(post_qc: pd.DataFrame,
                        comp_tab: pd.DataFrame,
                        official: set[str]) -> tuple[pd.DataFrame, dict]:
    """Cohort counts only. No classifier, no performance, nothing adopted."""
    sites = site_sort(post_qc.site.unique())
    plans: dict[str, set[str]] = {}

    # A -- official-list canonicalisation where it is unambiguous.
    drop_a: set[str] = set()
    for _, r in comp_tab.iterrows():
        members = r.members.split(";")
        if r.official_resolves:
            drop_a |= {s for s in members if s not in official}
    plans["A_official_canonical"] = drop_a

    # B -- drop the whole component wherever the official list does not
    #      uniquely determine a canonical identity.
    drop_b: set[str] = set()
    for _, r in comp_tab.iterrows():
        members = r.members.split(";")
        if r.official_resolves:
            drop_b |= {s for s in members if s not in official}
        else:
            drop_b |= set(members)
    plans["B_drop_ambiguous_component"] = drop_b

    # C -- retain every observation, bind components into identity groups.
    plans["C_grouped_identity"] = set()

    rows = []
    summary = {}
    for name, drop in plans.items():
        keep = post_qc[~post_qc.ID.isin(drop)]
        tab = apply_site_rule(keep, sites)
        tab.insert(0, "strategy", name)
        rows.append(tab)
        summary[name] = {
            "n_dropped": len(drop),
            "N": len(keep),
            "MDD": int((keep.dx == 1).sum()),
            "HC": int((keep.dx == 0).sum()),
            "eligible_sites": int(tab.eligible.sum()),
            "sites_below_threshold": [
                r.site for r in tab.itertuples() if not r.eligible
            ],
        }
    return pd.concat(rows, ignore_index=True), summary


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
def main() -> int:
    assert_read_only()
    OUT.mkdir(parents=True, exist_ok=True)
    phen = load_phenotype()
    qc = pd.read_csv(OUT / "fc_qc_summary.csv")
    cand = pd.read_csv(OUT / "duplicate_scan_candidates.csv")

    # ---------------- Task 1 ----------------
    print("== Task 1 -- official Yan 2019 subject lists")
    paths = fetch_official()
    for k, p in paths.items():
        print(f"  {p.name}  md5={hashlib.md5(p.read_bytes()).hexdigest()}")
    official = {k: parse_official(p) for k, p in paths.items()}
    o1789 = set(official["official_1789"].ID)
    o1642 = set(official["official_1642"].ID)

    rec = reconcile_official(phen, official, qc)
    rec.to_csv(OUT / "official_subject_reconciliation.csv", index=False)

    pid = set(phen.ID)
    mapped = o1789 & pid
    unmapped = sorted(o1789 - pid)
    extra = sorted(pid - o1789)
    print(f"  official final list N      : {len(o1789)} "
          f"({int(official['official_1789'].dx_official.sum())} MDD / "
          f"{int((official['official_1789'].dx_official == 0).sum())} HC)")
    print(f"  mapped to phenotype        : {len(mapped)}")
    print(f"  official but not phenotyped: {len(unmapped)} "
          f"-> sites {sorted({s.split('-')[0] for s in unmapped})}")
    print(f"  phenotyped but not official: {len(extra)}")
    dis = rec[(rec.dx_agree == 0)]
    print(f"  diagnosis disagreements    : {len(dis)}")

    by_site = (rec.assign(one=1)
               .groupby("site")
               .agg(n_phenotype=("in_phenotype", "sum"),
                    n_official_1789=("in_official_1789", "sum"),
                    n_official_1642=("in_official_1642", "sum"),
                    n_mapped=("ID", lambda s: len(set(s) & mapped)),
                    n_official_only=("ID", lambda s: len(set(s) & set(unmapped))),
                    n_phenotype_only=("ID", lambda s: len(set(s) & set(extra))))
               .reindex(site_sort(rec.site.unique()))
               .rename_axis("site").reset_index())
    by_site.to_csv(OUT / "official_reconciliation_by_site.csv", index=False)

    # ---------------- Task 2 ----------------
    print("\n== Task 2 -- the 67 FC_QC_FAIL subjects vs the official list")
    fail = qc[~qc.fc_qc_pass]
    fail_off = fail.ID.isin(o1789)
    print(f"  FC_QC_FAIL total           : {len(fail)}")
    print(f"    in official final list   : {int(fail_off.sum())}")
    print(f"    not in official list     : {int((~fail_off).sum())}")
    site_tab = (fail.assign(in_official=fail_off.values)
                .groupby("site")
                .agg(n_fail=("ID", "size"), n_fail_in_official=("in_official", "sum"))
                .reindex(site_sort(fail.site.unique())))
    print(site_tab.to_string())
    # Fraction of the site's officially-included subjects that we drop anyway.
    print("  NOTE: the constant-ROI rule is unchanged. FC is mathematically "
          "undefined for these scans regardless of official status.")

    # ---------------- Tasks 3-4 ----------------
    print("\n== Tasks 3-4 -- duplicate candidate evidence")
    cache = SignalCache()
    ph_idx = phen.set_index("ID")
    rows = []
    for _, r in cand.iterrows():
        a, b = r.id_a, r.id_b
        ev = pair_evidence(a, b, cache)
        pa, pb = ph_idx.loc[a], ph_idx.loc[b]
        rows.append({
            "subject_a": a, "subject_b": b, "site": r.site_a,
            "diagnosis_a": int(pa.dx), "diagnosis_b": int(pb.dx),
            "diagnosis_compatible": int(pa.dx) == int(pb.dx),
            "official_list_a": a in o1789, "official_list_b": b in o1789,
            "official_1642_a": a in o1642, "official_1642_b": b in o1642,
            "age_a": pa.Age, "age_b": pb.Age,
            "age_abs_diff": abs(float(pa.Age) - float(pb.Age)),
            "sex_a": int(pa.Sex), "sex_b": int(pb.Sex),
            "sex_compatible": int(pa.Sex) == int(pb.Sex),
            **ev,
        })
    pair = pd.DataFrame(rows)
    pair.to_csv(OUT / "duplicate_pair_reconciliation.csv", index=False)

    both = int((pair.official_list_a & pair.official_list_b).sum())
    only_a = int((pair.official_list_a & ~pair.official_list_b).sum())
    only_b = int((~pair.official_list_a & pair.official_list_b).sum())
    neither = int((~pair.official_list_a & ~pair.official_list_b).sum())
    print(f"  pairs: both official {both} | only A {only_a} | only B {only_b} "
          f"| neither {neither}   (total {len(pair)})")
    print(f"  same scan length           : {int(pair.same_scan_length.sum())}/{len(pair)}")
    for m in ["fc_r", "ts_flat_r", "roi_ts_r_min", "rp_r", "fd_series_r",
              "gsr_fc_r", "mean_fd_abs_diff", "age_abs_diff"]:
        v = pd.to_numeric(pair[m], errors="coerce").dropna()
        print(f"    {m:<18} n={len(v):>3}  min={v.min():.5f}  "
              f"median={v.median():.5f}  max={v.max():.5f}")
    print(f"  sex compatible             : {int(pair.sex_compatible.sum())}/{len(pair)}")
    conflict = pair[~pair.diagnosis_compatible]
    print(f"  diagnosis-conflict pairs   : {len(conflict)}")
    for _, r in conflict.iterrows():
        print(f"    {r.subject_a} (dx={r.diagnosis_a}, official={r.official_list_a}) "
              f"<-> {r.subject_b} (dx={r.diagnosis_b}, official={r.official_list_b}) "
              f"rp_r={r.rp_r:.5f} fc_r={r.fc_r:.5f}")

    # negative reference within the same sites
    audit_sites = site_sort(set(cand.site_a))
    exclude = {tuple(sorted((a, b))) for a, b in zip(cand.id_a, cand.id_b)}
    post_qc_ids = set(qc.loc[qc.fc_qc_pass, "ID"])
    phen_qc = phen[phen.ID.isin(post_qc_ids)]
    neg = negative_reference(audit_sites, phen_qc, exclude, cache)
    metrics = ["fc_r", "ts_flat_r", "roi_ts_r_mean", "roi_ts_r_min", "rp_r",
               "fd_series_r", "gsr_fc_r", "mean_fd_abs_diff"]
    ref = pd.concat([
        distribution_summary(pair, "duplicate_candidate", metrics),
        distribution_summary(neg, "within_site_non_candidate", metrics),
    ], ignore_index=True)
    ref.to_csv(OUT / "duplicate_similarity_reference.csv", index=False)
    print(f"\n  negative reference: {len(neg)} within-site non-candidate pairs "
          f"across {len(audit_sites)} sites")
    print(ref.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # ---------------- Task 5 ----------------
    print("\n== Task 5 -- duplicate graph")
    comps = connected_components(list(zip(cand.id_a, cand.id_b)))
    dx = dict(zip(phen.ID, phen.dx))
    comp_tab = classify_components(comps, o1789, dx)
    comp_tab.to_csv(OUT / "duplicate_component_summary.csv", index=False)
    sizes = comp_tab["size"].value_counts().sort_index()
    print(f"  connected components : {len(comps)}  (subjects "
          f"{int(comp_tab['size'].sum())})")
    print(f"  size distribution    : {sizes.to_dict()}")
    print(f"  all size 2           : {bool((comp_tab['size'] == 2).all())}")
    print(f"  components with >1 official member : "
          f"{int((comp_tab.n_in_official_1789 > 1).sum())}")
    print(f"  components official resolves to 1  : "
          f"{int(comp_tab.official_resolves.sum())}")
    print(f"  components with 0 official members : "
          f"{int((comp_tab.n_in_official_1789 == 0).sum())}")
    print(f"  diagnosis-conflict components      : "
          f"{int(comp_tab.diagnosis_conflict.sum())}")
    print(comp_tab.groupby("site")["size"].agg(["size", "sum"])
          .rename(columns={"size": "n_components", "sum": "n_subjects"}).to_string())

    # ---------------- Tasks 6-7 ----------------
    print("\n== Tasks 6-7 -- simulated cohort impact (no strategy adopted)")
    impact, summary = simulate_strategies(phen_qc, comp_tab, o1789)
    impact.to_csv(OUT / "dedup_strategy_cohort_impact.csv", index=False)
    for name, s in summary.items():
        print(f"  {name:<28} dropped={s['n_dropped']:>3}  N={s['N']}  "
              f"MDD={s['MDD']}  HC={s['HC']}  eligible_sites={s['eligible_sites']}"
              + (f"  BELOW THRESHOLD: {s['sites_below_threshold']}"
                 if s["sites_below_threshold"] else "  (no site below >=10/>=10)"))
    for s in ["S9", "S11"]:
        sub = impact[impact.site == s][["strategy", "n_MDD", "n_HC", "n_total"]]
        print(f"  {s}:")
        print(sub.to_string(index=False))

    # ---------------- assertions ----------------
    print("\n== assertions")
    assert len(o1789) == 1789, "official list size changed"
    assert not (rec.dx_agree == 0).any(), "official/phenotype diagnosis mismatch"
    assert len(pair) == len(cand) == 41, "candidate pair count changed"
    assert int(comp_tab["size"].sum()) == 82, "component membership lost subjects"
    assert summary["C_grouped_identity"]["N"] == 2275, "Strategy C changed N"
    for name, s in summary.items():
        tot = int(impact.loc[impact.strategy == name, "n_total"].sum())
        assert tot == s["N"], f"{name}: per-site counts do not sum to N"
    assert OUT.resolve().is_relative_to(ROOT / "results"), "output escaped results/"
    print("  all assertions passed")

    print("\nwrote:")
    for f in ["official_subject_reconciliation.csv",
              "official_reconciliation_by_site.csv",
              "duplicate_pair_reconciliation.csv",
              "duplicate_component_summary.csv",
              "duplicate_similarity_reference.csv",
              "dedup_strategy_cohort_impact.csv"]:
        print(f"  results/data_contract/{f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
