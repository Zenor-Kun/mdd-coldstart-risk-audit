#!/usr/bin/env python3
"""Stage 0.5 -- Data Contract Freeze: read-only verification of REST-meta-MDD.

Verifies the data facts that back docs/DATA_CONTRACT_v1.md and emits the
machine-readable summaries under results/data_contract/.

READ-ONLY BY CONSTRUCTION. This script must never fit a model, select
features, permute labels, or write anything under data/. It only reads
raw data and writes summary CSVs under results/.

Usage:
    python3 scripts/finalize_data_contract.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PKG = DATA / "REST-meta-MDD"
PHENO_XLSX = DATA / "REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx"
OUT = ROOT / "results" / "data_contract"

ROI_BRANCHES = ["ROISignals_FunImgARCWF", "ROISignals_FunImgARglobalCWF"]
FD_VARIANTS = ["FD_Jenkinson", "FD_Power", "FD_VanDijk"]
PRIMARY_FD = "FD_Jenkinson"
SENTINEL = -9999
EMPTY_STRINGS = ["[]", "", " ", "NA", "n/a", "NaN"]
QC_ENCODING = "gb18030"

# Frozen eligibility rule, spec v4 §5.4.
MIN_PER_CLASS = 10


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------
def assert_read_only() -> None:
    """Fail loudly if the output target would touch raw data."""
    out = OUT.resolve()
    if DATA.resolve() in out.parents or out == DATA.resolve():
        raise RuntimeError("refusing to write inside data/ -- raw data is read-only")


# --------------------------------------------------------------------------
# cohort
# --------------------------------------------------------------------------
def normalise_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Map every missing encoding used in this workbook to NaN.

    Three distinct encodings coexist and all three must be handled:
      1. numeric sentinel -9999
      2. genuinely blank cells (already NaN)
      3. the literal string '[]' (MATLAB empty) in object-dtype columns
         -- Illness duration (months), HAMD, HAMA
    Handling only (1) leaves '[]' as a string and silently turns a numeric
    clinical variable into an unusable object column.
    """
    out = df.copy()
    for c in cols:
        s = out[c].replace(EMPTY_STRINGS, np.nan)
        if s.dtype == object:
            s = pd.to_numeric(s, errors="coerce")
        out[c] = s.replace(SENTINEL, np.nan)
    return out


def load_phenotype() -> pd.DataFrame:
    """Phenotype is the ONLY source of truth for diagnosis."""
    xl = pd.ExcelFile(PHENO_XLSX)
    mdd = xl.parse("MDD")
    hc = xl.parse("Controls")
    mdd["dx"] = 1  # MDD
    hc["dx"] = 0  # HC
    phen = pd.concat([mdd, hc], ignore_index=True, sort=False)
    phen["site"] = phen["ID"].str.split("-").str[0]
    clinical = [c for c in phen.columns if c not in ("ID", "site", "dx")]
    phen = normalise_missing(phen, clinical)
    assert phen["ID"].is_unique, "duplicate phenotype IDs"
    assert phen["dx"].notna().all(), "phenotype diagnosis must never be null"
    assert not [c for c in clinical if phen[c].dtype == object], \
        "clinical columns must be numeric after missing-value normalisation"
    return phen


def clinical_missingness(phen: pd.DataFrame) -> pd.DataFrame:
    """Missingness per clinical variable, MDD-only where HC has no such field."""
    raw = pd.ExcelFile(PHENO_XLSX)
    rawm = raw.parse("MDD")
    rows = []
    mdd = phen[phen.dx == 1]
    for c in [c for c in rawm.columns if c != "ID"]:
        s_raw = rawm[c]
        rows.append(
            {
                "variable": c,
                "n_MDD": len(mdd),
                "n_missing": int(mdd[c].isna().sum()),
                "missing_rate": round(float(mdd[c].isna().mean()), 4),
                "n_sentinel_-9999": int((s_raw == SENTINEL).sum()),
                "n_empty_string": int(s_raw.isin(EMPTY_STRINGS).sum()),
                "n_blank_cell": int(s_raw.isna().sum()),
                "raw_dtype": str(s_raw.dtype),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False)


def site_sort(sites) -> list:
    return sorted(sites, key=lambda s: int(s[1:]))


def eligibility(phen: pd.DataFrame) -> pd.DataFrame:
    """Frozen rule (§5.4): >=10 MDD and >=10 HC, with site/dx/age/sex complete."""
    g = phen.groupby("site")
    tab = pd.DataFrame(
        {
            "n_MDD": g["dx"].sum().astype(int),
            "n_HC": g["dx"].apply(lambda s: int((s == 0).sum())),
            "n_total": g.size(),
            "n_age_missing": g["Age"].apply(lambda s: int(s.isna().sum())),
            "n_sex_missing": g["Sex"].apply(lambda s: int(s.isna().sum())),
        }
    ).reindex(site_sort(phen["site"].unique()))
    tab["eligible"] = (
        (tab.n_MDD >= MIN_PER_CLASS)
        & (tab.n_HC >= MIN_PER_CLASS)
        & (tab.n_age_missing == 0)
        & (tab.n_sex_missing == 0)
    )
    return tab


# --------------------------------------------------------------------------
# Task A -- ROI structure
# --------------------------------------------------------------------------
def scan_roi(phen: pd.DataFrame) -> pd.DataFrame:
    """Header-only scan (whosmat) of every phenotype subject in every branch."""
    rows = []
    for branch in ROI_BRANCHES:
        bdir = PKG / "Results" / branch
        for sid in phen["ID"]:
            f = bdir / f"ROISignals_{sid}.mat"
            rec = {"branch": branch, "ID": sid, "present": int(f.exists())}
            if f.exists():
                info = sio.whosmat(str(f))
                shapes = {v[0]: v[1] for v in info}
                dtypes = {v[0]: v[2] for v in info}
                shp = shapes.get("ROISignals")
                rec.update(
                    n_vars=len(info),
                    var_names="|".join(shapes),
                    n_timepoints=shp[0] if shp else np.nan,
                    n_roi=shp[1] if shp and len(shp) > 1 else np.nan,
                    n_dim=len(shp) if shp else np.nan,
                    mat_dtype=dtypes.get("ROISignals", ""),
                )
            rows.append(rec)
    scan = pd.DataFrame(rows)
    scan["malformed"] = (
        (scan.present == 1)
        & ~((scan.n_dim == 2) & (scan.var_names == "ROISignals") & (scan.n_vars == 1))
    ).astype(int)
    return scan.merge(phen[["ID", "site", "dx"]], on="ID", how="left")


def roi_summary(scan: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (branch, site), g in scan.groupby(["branch", "site"]):
        present = g[g.present == 1]
        out.append(
            {
                "branch": branch,
                "site": site,
                "n_subjects_phenotype": len(g),
                "n_roi_files_present": int(g.present.sum()),
                "n_roi_files_missing": int((g.present == 0).sum()),
                "n_malformed": int(g.malformed.sum()),
                "n_roi_distinct": "|".join(
                    str(int(v)) for v in sorted(present.n_roi.dropna().unique())
                ),
                "n_roi": int(present.n_roi.mode().iloc[0]) if len(present) else -1,
                "n_timepoints_distinct": "|".join(
                    str(int(v)) for v in sorted(present.n_timepoints.dropna().unique())
                ),
                "n_timepoints": int(present.n_timepoints.mode().iloc[0]) if len(present) else -1,
                "mat_dtype": "|".join(present.mat_dtype.dropna().unique()),
            }
        )
    df = pd.DataFrame(out)
    df["site_order"] = df.site.str[1:].astype(int)
    return df.sort_values(["branch", "site_order"]).drop(columns="site_order")


def trace_subjects(phen: pd.DataFrame, sites=("S1", "S13", "S20", "S5")) -> list[dict]:
    """Full provenance trace for one subject per site (Task A)."""
    traces = []
    for site in sites:
        pool = phen[phen.site == site]
        if pool.empty:
            continue
        sid = pool.sample(1, random_state=0)["ID"].iloc[0]
        f = PKG / "Results" / ROI_BRANCHES[0] / f"ROISignals_{sid}.mat"
        mat = sio.loadmat(str(f))
        arr = mat["ROISignals"]
        traces.append(
            {
                "phenotype_id": sid,
                "site": site,
                "file": str(f.relative_to(ROOT)),
                "keys": [k for k in mat if not k.startswith("__")],
                "type": type(arr).__name__,
                "dtype": str(arr.dtype),
                "shape": arr.shape,
                "n_time": arr.shape[0],
                "n_roi": arr.shape[1],
                "n_nan": int(np.isnan(arr).sum()),
            }
        )
    return traces


def column_fingerprint(phen: pd.DataFrame, site: str, n: int = 20) -> np.ndarray:
    """Per-column scale profile, averaged over subjects and scale-normalised.

    Different atlases carve different ROI volumes, so the column-wise std
    profile acts as a layout fingerprint that is comparable across sites
    even when scan length differs. Used only to test whether two sites share
    a column layout -- never as a model feature.
    """
    prof = []
    for sid in phen[phen.site == site]["ID"].head(n):
        arr = sio.loadmat(str(PKG / "Results" / ROI_BRANCHES[0] / f"ROISignals_{sid}.mat"))["ROISignals"]
        s = arr.std(axis=0)
        prof.append(s / np.median(s[s > 0]))
    return np.mean(prof, axis=0)


def roi_layout_probe(phen: pd.DataFrame, odd_site: str, ref_sites=("S1", "S20", "S8")) -> pd.DataFrame:
    """Test whether `odd_site`'s narrower column space is a slice of the wide one."""
    fp_odd = column_fingerprint(phen, odd_site)
    k = len(fp_odd)
    refs = {s: column_fingerprint(phen, s) for s in ref_sites}
    rows = []
    names = list(refs)
    for i, a in enumerate(names):  # baseline: wide-vs-wide agreement
        for b in names[i + 1:]:
            rows.append({"comparison": f"{a} vs {b} (both wide, full)", "kind": "baseline",
                         "r": round(float(np.corrcoef(refs[a], refs[b])[0, 1]), 4)})
    for s, fp in refs.items():  # is the narrow layout a prefix / suffix / any slice?
        rows.append({"comparison": f"{odd_site} vs {s}[:{k}] (prefix)", "kind": "prefix",
                     "r": round(float(np.corrcoef(fp_odd, fp[:k])[0, 1]), 4)})
        rows.append({"comparison": f"{odd_site} vs {s}[-{k}:] (suffix)", "kind": "suffix",
                     "r": round(float(np.corrcoef(fp_odd, fp[-k:])[0, 1]), 4)})
        best_r, best_o = max(
            (float(np.corrcoef(fp_odd, fp[o:o + k])[0, 1]), o) for o in range(len(fp) - k + 1)
        )
        rows.append({"comparison": f"{odd_site} vs {s}[best offset={best_o}]", "kind": "best_slice",
                     "r": round(best_r, 4)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Task B -- motion
# --------------------------------------------------------------------------
def mean_fd(sid: str, variant: str = PRIMARY_FD) -> float:
    f = PKG / "RealignParameter" / sid / f"{variant}_{sid}.txt"
    if not f.exists():
        return np.nan
    try:
        v = np.loadtxt(f)
    except Exception:
        return np.nan
    v = np.atleast_1d(v)
    return float(np.mean(v)) if v.size else np.nan


def scan_motion(phen: pd.DataFrame) -> pd.DataFrame:
    rec = []
    for sid in phen["ID"]:
        row = {"ID": sid}
        for variant in FD_VARIANTS:
            row[f"mean_{variant}"] = mean_fd(sid, variant)
        row["realign_dir_exists"] = int((PKG / "RealignParameter" / sid).is_dir())
        rec.append(row)
    return pd.DataFrame(rec).merge(phen[["ID", "site", "dx"]], on="ID", how="left")


def load_qc(name: str = "RawFunImgQC") -> pd.DataFrame:
    # QC tables carry Chinese reviewer comments in GBK, not UTF-8.
    qc = pd.read_csv(PKG / "QC" / f"{name}.tsv", sep="\t", encoding=QC_ENCODING)
    qc.columns = [c.strip() for c in qc.columns]
    return qc.rename(columns={"Subject ID": "ID", "QC Score": "qc_score"})


# --------------------------------------------------------------------------
# Task D -- preprocessing branches
# --------------------------------------------------------------------------
def branch_inventory(phen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in sorted((PKG / "Results").iterdir()):
        if not d.is_dir():
            continue
        files = list(d.iterdir())
        name = d.name
        metric = name.split("_")[0]
        suffix = name.split("_", 1)[1] if "_" in name else ""
        stems = {f.name for f in files}
        # Subject ID is the final underscore-delimited token before the extension.
        def sid_of(fname: str) -> str:
            return fname.split(".")[0].split("_")[-1]

        by_sid: dict[str, int] = {}
        for s in stems:
            by_sid[sid_of(s)] = by_sid.get(sid_of(s), 0) + 1
        covered = sum(1 for sid in phen["ID"] if sid in by_sid)
        extras = sorted(set(by_sid) - set(phen["ID"]))
        example = sorted(stems)[0] if stems else ""
        rows.append(
            {
                "directory": str(d.relative_to(ROOT)),
                "metric": metric,
                "pipeline_suffix": suffix,
                "gsr": "GSR" if "global" in suffix else "no-GSR",
                "smoothing_flag_in_name": "S" if suffix.endswith("S") else "none",
                "file_pattern": example.replace(sid_of(example), "<SubID>") if example else "",
                "n_files": len(files),
                "n_files_per_subject": round(len(files) / max(len(by_sid), 1), 2),
                "n_distinct_subjects": len(by_sid),
                "n_phenotype_subjects_covered": covered,
                "n_phenotype_subjects_missing": len(phen) - covered,
                "n_extra_subjects_not_in_phenotype": len(extras),
                "extra_subject_sites": "|".join(sorted({e.split("-")[0] for e in extras})),
                "is_roi_timeseries": int(metric == "ROISignals"),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main() -> int:
    assert_read_only()
    OUT.mkdir(parents=True, exist_ok=True)

    phen = load_phenotype()
    print(f"[cohort] phenotype N={len(phen)} MDD={int(phen.dx.sum())} HC={int((phen.dx==0).sum())}")
    elig = eligibility(phen)
    eligible = list(elig.index[elig.eligible])
    excluded = list(elig.index[~elig.eligible])
    print(f"[cohort] sites={len(elig)} eligible={len(eligible)} excluded={excluded}")
    elig.to_csv(OUT / "site_cohort_table.csv")

    print("[roi] scanning headers for both branches ...")
    scan = scan_roi(phen)
    summ = roi_summary(scan)
    summ.to_csv(OUT / "roi_structure_summary.csv", index=False)
    prim = scan[scan.branch == ROI_BRANCHES[0]]
    print(f"[roi] present={int(prim.present.sum())}/{len(prim)} "
          f"missing={int((prim.present==0).sum())} malformed={int(prim.malformed.sum())}")
    print(f"[roi] distinct n_roi = {sorted(prim.n_roi.dropna().unique().astype(int))}")
    for n, g in prim.groupby("n_roi"):
        print(f"       n_roi={int(n)}: {len(g)} subjects, sites={site_sort(g.site.unique())}")

    odd = [s for s in prim.site.unique() if prim[prim.site == s].n_roi.iloc[0] != prim.n_roi.mode().iloc[0]]
    if odd:
        print(f"[roi] column-layout probe for divergent site(s) {odd} ...")
        probe = roi_layout_probe(phen, odd[0])
        probe.to_csv(OUT / "roi_column_layout_probe.csv", index=False)
        print(probe.to_string(index=False))

    print("[roi] subject traces:")
    for t in trace_subjects(phen):
        print(f"       {t['phenotype_id']} -> {t['file']} keys={t['keys']} "
              f"shape={t['shape']} (time={t['n_time']}, roi={t['n_roi']}) nan={t['n_nan']}")

    print("[motion] computing mean FD ...")
    mot = scan_motion(phen)
    mot.to_csv(OUT / "mean_fd_per_subject.csv", index=False)
    col = f"mean_{PRIMARY_FD}"
    valid = int(mot[col].notna().sum())
    print(f"[motion] valid={valid} missing={len(mot)-valid} rate={(len(mot)-valid)/len(mot):.4f}")

    # Disambiguate two DIFFERENT gaps that are easy to conflate:
    #   visual QC rating coverage  vs  mean-FD (motion) coverage.
    qc = load_qc("RawFunImgQC")
    absent_qc = sorted(set(phen["ID"]) - set(qc["ID"]))
    qc_sites = pd.Series([s.split("-")[0] for s in absent_qc]).value_counts().to_dict()
    print(f"[qc] phenotype subjects absent from RawFunImgQC.tsv = {len(absent_qc)} by site {qc_sites}")
    print(f"[qc] phenotype subjects missing mean FD             = {len(mot)-valid}")
    print(f"[qc] -> the two gaps are {'THE SAME' if len(absent_qc)==len(mot)-valid and len(absent_qc)>0 else 'NOT equivalent'}")
    pd.DataFrame({"ID": absent_qc}).to_csv(OUT / "phenotype_subjects_absent_from_funcQC.csv", index=False)

    by_site = (
        mot.groupby("site")
        .apply(
            lambda g: pd.Series(
                {
                    "n_total": len(g),
                    "n_valid_FD": int(g[col].notna().sum()),
                    "n_missing_FD": int(g[col].isna().sum()),
                    "missing_rate": round(float(g[col].isna().mean()), 4),
                }
            ),
            include_groups=False,
        )
        .reindex(site_sort(mot.site.unique()))
    )
    by_site["eligible"] = elig["eligible"]
    by_site.to_csv(OUT / "mean_fd_missingness_by_site.csv")
    print(by_site.to_string())

    cm = clinical_missingness(phen)
    cm.to_csv(OUT / "clinical_missingness.csv", index=False)
    print("[clinical] top missing variables (MDD sheet):")
    print(cm.head(8).to_string(index=False))
    multi = cm[(cm["n_sentinel_-9999"] > 0) & (cm["n_empty_string"] > 0)]
    print(f"[clinical] variables using MORE THAN ONE missing encoding: {list(multi.variable)}")

    print("[branches] inventory ...")
    inv = branch_inventory(phen)
    inv.to_csv(OUT / "preprocessing_branch_inventory.csv", index=False)
    print(inv[["directory", "metric", "gsr", "n_files", "n_phenotype_subjects_covered"]].to_string(index=False))

    print(f"\nwrote -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
