#!/usr/bin/env python3
"""Stage G — build the manuscript-facing source-data layer.

READ-ONLY with respect to every CLOSED-stage frozen authority. This script
performs **no scientific analysis**: it reads frozen artifacts, applies only
aggregations that are already frozen in the protocol documents, and writes
compact source-data CSVs under ``results/stage_g/source_data/`` that the
Stage-G figures, tables and number ledger are rendered from.

**No Stage-G aggregate estimand exists.** Every aggregate written here is one
that the frozen protocols already authorise: the §9 site-balanced selective-risk
aggregate (equal-weight mean over the estimable sites) and the §10.2
patient-weighted / site-balanced **AURC** pair, the latter read verbatim from
``results/rq1_s0/aurc_summary.csv`` together with its registered intervals.
Acquisition-count weighting is applied to nothing but AURC, which is where the
frozen protocol defines it. No new threshold, test, confidence interval, model,
weighting rule or estimator is introduced.

Regenerate with::

    python3 scripts/stage_g_build_source_data.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "stage_g"
SRC = OUT / "source_data"

# --------------------------------------------------------------------------
# Frozen authority inventory. Nothing outside this table may be read.
# --------------------------------------------------------------------------
AUTHORITIES = {
    "data_contract": {
        "tag": "data-contract-v1",
        "report": "docs/DATA_CONTRACT_v1.md",
        "store": "results/data_contract/",
    },
    "gate1_rq0": {
        "tag": "gate1-rq0-v1",
        "report": "results/gate1_formal_permutation/RQ0_FORMAL_REPORT.md",
        "store": "results/gate1/ + results/gate1_formal_permutation/",
    },
    "rq1_s0": {
        "tag": "rq1-s0-results-v1",
        "protocol_tag": "rq1-s0-protocol-v1",
        "report": "results/rq1_s0/RQ1_S0_REPORT.md",
        "store": "results/rq1_s0/",
    },
    "rq1_s1": {
        "tag": "rq1-s1-results-v1",
        "protocol_tag": "rq1-s1-protocol-v1",
        "report": "results/rq1_s1/RQ1_S1_REPORT.md",
        "store": "results/rq1_s1/",
    },
    "rq2_s2": {
        "tag": "rq2-s2-results-v2",
        "protocol_tag": "rq2-s2-protocol-v1",
        "report": "results/rq2_s2_freeze/RQ2_S2_REPORT.md",
        "store": "results/rq2_s2/",
    },
    "rq3": {
        "tag": "rq3-results-v2",
        "protocol_tag": "rq3-protocol-v2",
        "report": "results/rq3_freeze/RQ3_REPORT.md",
        "store": "results/rq3/",
    },
    "stage_f": {
        "tag": "stage-f-results-v2",
        "protocol_tag": "stage-f-protocol-v2",
        "report": "docs/STAGE_F_REPORT.md",
        "store": "results/stage_f_v2/",
    },
}

_READS: dict[str, str] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(rel: str) -> pd.DataFrame:
    p = ROOT / rel
    _READS[rel] = sha256(p)
    return pd.read_csv(p)


def read_json(rel: str):
    p = ROOT / rel
    _READS[rel] = sha256(p)
    return json.loads(p.read_text())


def write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(SRC / name, index=False)
    print(f"  wrote {name}  ({len(df)} rows)")


# --------------------------------------------------------------------------
# Frozen aggregation rules (RQ1/S0 protocol §9 and §10.2)
# --------------------------------------------------------------------------
def site_balanced(values: np.ndarray) -> float:
    """§9: equal-weight mean over the ESTIMABLE sites."""
    v = values[np.isfinite(values)]
    return float(np.mean(v)) if v.size else float("nan")


# --------------------------------------------------------------------------
# Fig. 1 — risk-coverage curves and per-site acceptance
# --------------------------------------------------------------------------
def build_fig1() -> None:
    # ---- Fig. 1a: the FROZEN patient-weighted vs site-balanced AURC ------
    # Read verbatim from the frozen RQ1/S0 store. Both aggregations, both
    # registered 95% site-cluster-bootstrap percentile intervals and the frozen
    # per-site distribution summary are already computed there; Stage G
    # recomputes nothing.
    a = read_csv("results/rq1_s0/aurc_summary.csv")
    per_site = read_csv("results/rq1_s0/per_site_aurc.csv")
    iv = read_csv("results/rq1_s0/site_bootstrap_intervals.csv")

    rows = []
    for _, r in a.iterrows():
        n_sites = int((per_site.method == r["method"]).sum())
        sum_n_s = float(per_site.loc[per_site.method == r["method"], "n_s"].sum())
        for agg, val, lo, hi, rule, denom_name, denom in (
            ("patient_weighted", r["patient_weighted_aurc"],
             r["patient_weighted_aurc_ci_low"], r["patient_weighted_aurc_ci_high"],
             "sum_s n_s * AURC_s / sum_s n_s  (RQ1_S0_PROTOCOL 10.2)",
             "acquisitions_summed_over_all_23_sites", sum_n_s),
            ("site_balanced", r["site_balanced_aurc"],
             r["site_balanced_aurc_ci_low"], r["site_balanced_aurc_ci_high"],
             "mean_s AURC_s over 23 sites  (RQ1_S0_PROTOCOL 10.2)",
             "sites_equally_weighted", float(n_sites)),
        ):
            reg = iv[(iv["quantity"] == f"{agg}_aurc")
                     & (iv["method_or_contrast"] == r["method"])]
            assert len(reg) == 1, (agg, r["method"])
            reg = reg.iloc[0]
            assert abs(float(reg["point_estimate"]) - float(val)) <= 1e-15
            rows.append({
                "method": r["method"],
                "aggregation": agg,
                "aurc": float(val),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "ci_definition": reg["ci_method"],
                "ci_level": 0.95,
                "bootstrap_replicates_valid": int(reg["valid_bootstrap_replicates"]),
                "bootstrap_replicates_total": int(reg["total_bootstrap_replicates"]),
                "n_sites": n_sites,
                "n_estimable_sites": int(reg["n_estimable_sites"]),
                "aggregation_denominator_name": denom_name,
                "aggregation_denominator_value": denom,
                "aggregation_rule": rule,
                "min_site_aurc": float(r["min_site_aurc"]),
                "median_site_aurc": float(r["median_site_aurc"]),
                "max_site_aurc": float(r["max_site_aurc"]),
                "loss": "0/1 classification loss — ranking quality, NOT the "
                        "certification balanced-error risk",
                "provenance": "FROZEN_REGISTERED_QUANTITY",
                "source_artifact": "results/rq1_s0/aurc_summary.csv; "
                                   "results/rq1_s0/site_bootstrap_intervals.csv",
                "source_authority": AUTHORITIES["rq1_s0"]["tag"],
            })
    write(pd.DataFrame(rows), "fig1_aurc_aggregation.csv")

    write(per_site.assign(
        provenance="FROZEN_REGISTERED_QUANTITY",
        loss="0/1 classification loss",
        source_artifact="results/rq1_s0/per_site_aurc.csv",
        source_authority=AUTHORITIES["rq1_s0"]["tag"],
    ), "fig1_per_site_aurc.csv")

    # ---- Fig. 1b: the FROZEN site-balanced selective risk-coverage curve --
    # RQ1_S0_PROTOCOL section 9 authorises exactly two selective-risk
    # aggregations: site-balanced (equal-weight mean over the ESTIMABLE sites)
    # and worst-site. Only the site-balanced aggregate is carried here. No
    # acquisition-count weighting of selective risk exists in this artifact.
    frames = []
    for regime, rel in (
        ("S0", "results/rq1_s0/risk_coverage_all_q.csv"),
        ("S1", "results/rq1_s1/risk_coverage_all_q.csv"),
    ):
        d = read_csv(rel)
        d = d[d["q"].astype(str) != "NO_ABSTENTION"].copy()
        d["q"] = d["q"].astype(float)
        for (method, q), g in d.groupby(["method", "q"], sort=True):
            ber = pd.to_numeric(g["BER_sel_target"], errors="coerce").to_numpy(float)
            cov = pd.to_numeric(g["coverage_target"], errors="coerce").to_numpy(float)
            frames.append({
                "regime": regime,
                "method": method,
                "q": q,
                "source_intended_coverage": float(g["source_intended_coverage"].iloc[0]),
                "n_sites_in_fold_set": int(len(g)),
                # denominators are SITE COUNTS: the frozen aggregation weights
                # sites equally. No acquisition-count denominator applies here.
                "ber_site_balanced": site_balanced(ber),
                "ber_denominator_sites": int(np.isfinite(ber).sum()),
                "coverage_site_balanced": site_balanced(cov),
                "coverage_denominator_sites": int(np.isfinite(cov).sum()),
                "aggregation_rule": "equal-weight mean over the ESTIMABLE sites "
                                    "(RQ1_S0_PROTOCOL section 9)",
                "provenance": "FROZEN_AGGREGATE__RQ1_S0_PROTOCOL_SECTION_9",
                "source_artifact": rel,
                "source_authority": AUTHORITIES["rq1_s0" if regime == "S0" else "rq1_s1"]["tag"],
            })
    write(pd.DataFrame(frames).sort_values(["regime", "method", "q"],
                                           ascending=[True, True, False]),
          "fig1_risk_coverage_site_balanced.csv")

    # ---- Fig. 1c: per-site acceptance (actual target coverage) -----------
    cohort = read_csv("results/data_contract/post_dedup_site_counts.csv")
    cohort = cohort[cohort["eligible_post_dedup"]].set_index("site")

    rows = []
    for regime, rel in (
        ("S0", "results/rq1_s0/risk_coverage_all_q.csv"),
        ("S1", "results/rq1_s1/risk_coverage_all_q.csv"),
    ):
        d = read_csv(rel)
        d = d[d["operating_point"].isin(["OP50", "OP30"])].copy()
        for _, r in d.iterrows():
            site = r["held_out_site"]
            rows.append(
                {
                    "regime": regime,
                    "operating_point": r["operating_point"],
                    "method": r["method"],
                    "site": site,
                    "n_s": int(cohort.loc[site, "n_total_acq"]),
                    "n_MDD": int(cohort.loc[site, "n_MDD_acq"]),
                    "n_HC": int(cohort.loc[site, "n_HC_acq"]),
                    "source_intended_coverage": float(r["source_intended_coverage"]),
                    "acceptance_rate": float(r["coverage_target"]),
                    "n_keep": int(r["n_keep"]),
                    "ber_sel_target": pd.to_numeric(pd.Series([r["BER_sel_target"]]),
                                                    errors="coerce").iloc[0],
                    "ber_sel_estimable": bool(r["BER_sel_estimable"]),
                    "provenance": "FROZEN_PER_SITE_QUANTITY",
                    "source_artifact": rel,
                    "source_authority": AUTHORITIES["rq1_s0" if regime == "S0" else "rq1_s1"]["tag"],
                }
            )
    acc = pd.DataFrame(rows)
    # frozen manuscript ordering convention: site sample size, descending
    acc = acc.sort_values(["regime", "operating_point", "method", "n_s", "site"],
                          ascending=[True, True, True, False, True])
    write(acc, "fig1_right_site_acceptance.csv")


# --------------------------------------------------------------------------
# Fig. 2 — RQ2 / S2 certification and label cost
# --------------------------------------------------------------------------
TIER_REGIME = {
    "pooled_source": ("S0", "source_reference"),
    "target_unlabeled_weighted": ("S1", "empirical_screening"),
    "target_local": ("S2", "formal_target_certification"),
    "pooled_local_shrinkage": ("S2", "empirical_screening"),
}


def build_fig2() -> None:
    iv = read_csv("results/rq2_s2/registered_intervals.csv")
    keep = [
        "formal_certification_probability",
        "empirical_pass_probability",
        "source_reference_pass_rate",
        "zero_label_empirical_pass_rate",
        "formal_attainable_coverage",
        "empirical_attainable_coverage",
        "formal_realized_violation_rate",
        "formal_p_censored_min_cert_n",
        "empirical_p_censored_min_pass_n",
    ]
    d = iv[iv["quantity"].isin(keep)].copy()
    d["information_regime"] = d["strategy"].map(lambda s: TIER_REGIME[s][0])
    d["guarantee_tier"] = d["strategy"].map(lambda s: TIER_REGIME[s][1])
    d["status"] = np.where(d["ci_low"].astype(str) == "NOT_ESTIMABLE",
                           "NOT_ESTIMABLE", "ESTIMABLE")
    d["source_artifact"] = "results/rq2_s2/registered_intervals.csv"
    d["source_authority"] = AUTHORITIES["rq2_s2"]["tag"]
    d["ci_definition"] = "95% site-level cluster-bootstrap percentile CI (B=5000)"
    cols = [
        "quantity", "information_regime", "guarantee_tier", "strategy", "n_labels",
        "method", "point_estimate", "ci_low", "ci_high", "n_sites",
        "n_estimable_sites", "valid_bootstrap_replicates", "status",
        "ci_definition", "source_artifact", "source_authority",
    ]
    write(d[cols].sort_values(["quantity", "strategy", "n_labels", "method"]),
          "fig2_certification_intervals.csv")

    # ---- uncertified-site burden vs N, with frozen reason split -----------
    fps = read_json("results/rq2_s2/formal_point_sections.json")
    payload = fps["expected_uncertified_site_burden"]["payload"]["by_method_n"]
    rows = []
    for key, v in payload.items():
        method, n = key.split("|")
        split = v.get("not_certified_reason_split", {})
        rows.append(
            {
                "strategy": "target_local",
                "information_regime": "S2",
                "guarantee_tier": "formal_target_certification",
                "method": method,
                "n_labels": int(n),
                "exploratory": bool(v["exploratory"]),
                "n_eligible_sites": int(v["n_sites"]),
                "expected_uncertified_site_burden": float(v["expected_uncertified_site_burden"]),
                "n_sites_certified_at_least_once": int(v["n_sites_at_one"]),
                "n_sites_never_certified": int(v["n_sites_at_zero"]),
                "decisions_NO_ELIGIBLE_CANDIDATE": int(split.get("NO_ELIGIBLE_CANDIDATE", 0)),
                "decisions_NO_PASSING_CANDIDATE": int(split.get("NO_PASSING_CANDIDATE", 0)),
                "source_artifact": "results/rq2_s2/formal_point_sections.json",
                "source_authority": AUTHORITIES["rq2_s2"]["tag"],
            }
        )
    write(pd.DataFrame(rows).sort_values(["method", "n_labels"]),
          "fig2_uncertified_burden.csv")


# --------------------------------------------------------------------------
# Table 1 — cohort / site block + RQ0 gate block
# --------------------------------------------------------------------------
def build_table1() -> None:
    cohort = read_csv("results/data_contract/post_dedup_site_counts.csv")
    cohort = cohort[cohort["eligible_post_dedup"]].copy()
    folds = read_csv("results/gate1/outer_fold_metrics.csv").set_index("site")
    rstar_src = read_csv("results/gate1/source_full_coverage_error_by_fold.csv").set_index("site")

    rows = []
    for _, c in cohort.iterrows():
        s = c["site"]
        rows.append(
            {
                "site": s,
                "n_acquisitions": int(c["n_total_acq"]),
                "n_MDD": int(c["n_MDD_acq"]),
                "n_HC": int(c["n_HC_acq"]),
                "outer_loso_bacc": float(folds.loc[s, "bacc"]),
                "outer_loso_auroc": float(folds.loc[s, "auroc"]),
                "e_full_minus_s": float(rstar_src.loc[s, "e_full_minus_s"]),
                "r_star_s": float(rstar_src.loc[s, "e_full_minus_s"]) - 0.10,
                "source_artifact": "results/data_contract/post_dedup_site_counts.csv; "
                                   "results/gate1/outer_fold_metrics.csv; "
                                   "results/gate1/source_full_coverage_error_by_fold.csv",
                "source_authority": f'{AUTHORITIES["data_contract"]["tag"]}; '
                                    f'{AUTHORITIES["gate1_rq0"]["tag"]}',
            }
        )
    sites = pd.DataFrame(rows).sort_values("n_acquisitions", ascending=False)
    write(sites, "table1_sites.csv")

    g1 = read_json("results/gate1/gate1_results.json")
    perm = read_json("results/gate1_formal_permutation/permutation_summary.json")
    obs = g1["observed"]["macro_site_bacc"]
    dec = g1["site_decodability"]
    e = g1["observed"]["e_full_minus_s"]

    gates = [
        dict(gate="RQ0(a) outer LOSO-site macro BAcc", quantity="macro_site_bacc",
             value=obs["mean"], dispersion_sd=obs["sd"], median=obs["median"],
             minimum=obs["min"], maximum=obs["max"], n_units=obs["n"],
             unit="balanced accuracy", aggregation="macro mean over 23 held-out sites",
             reference="0.500 (chance)",
             source_artifact="results/gate1/gate1_results.json",
             source_authority=AUTHORITIES["gate1_rq0"]["tag"]),
        dict(gate="RQ0(b) site decodability", quantity="site_balanced_accuracy",
             value=dec["balanced_accuracy"], dispersion_sd=None,
             median=None, minimum=None, maximum=None, n_units=dec["n_sites"],
             unit="balanced accuracy (23-class)",
             aggregation="5-fold stratified over 2232 acquisitions",
             reference=f'{dec["chance_reference"]:.6f} (1/23 chance)',
             source_artifact="results/gate1/site_decodability.csv",
             source_authority=AUTHORITIES["gate1_rq0"]["tag"]),
        dict(gate="RQ0(b) site decodability", quantity="site_macro_f1",
             value=dec["macro_f1"], dispersion_sd=None, median=None,
             minimum=None, maximum=None, n_units=dec["n_sites"],
             unit="macro F1 (23-class)",
             aggregation="5-fold stratified over 2232 acquisitions",
             reference="—",
             source_artifact="results/gate1/site_decodability.csv",
             source_authority=AUTHORITIES["gate1_rq0"]["tag"]),
        dict(gate="RQ0(c) within-site permutation null", quantity="null_macro_site_bacc",
             value=perm["null_mean"], dispersion_sd=perm["null_sd"],
             median=perm["null_median"], minimum=perm["null_min"],
             maximum=perm["null_max"], n_units=perm["B"],
             unit="balanced accuracy",
             aggregation="B=200 reduced-cost within-site label permutations",
             reference="observed 0.5819861923367704",
             source_artifact="results/gate1_formal_permutation/permutation_summary.json",
             source_authority=AUTHORITIES["gate1_rq0"]["tag"]),
        dict(gate="RQ0(c) within-site permutation null", quantity="permutation_p_value",
             value=perm["p_value"], dispersion_sd=None, median=None,
             minimum=None, maximum=None, n_units=perm["B"],
             unit="p-value (upper-tailed, finite-sample corrected)",
             aggregation="(1 + #{null >= observed}) / (B + 1), #{}=0",
             reference="—",
             source_artifact="results/gate1_formal_permutation/permutation_summary.json",
             source_authority=AUTHORITIES["gate1_rq0"]["tag"]),
        dict(gate="RQ0 support quantity", quantity="e_full_minus_s",
             value=e["mean"], dispersion_sd=e["sd"], median=e["median"],
             minimum=e["min"], maximum=e["max"], n_units=e["n"],
             unit="balanced error rate",
             aggregation="per outer fold, source sites only (full coverage)",
             reference="r*_s = e_full,-s - 0.10",
             source_artifact="results/gate1/source_full_coverage_error_by_fold.csv",
             source_authority=AUTHORITIES["gate1_rq0"]["tag"]),
    ]
    write(pd.DataFrame(gates), "table1_rq0_gates.csv")


# --------------------------------------------------------------------------
# Table 2 — RQ3 block-wise attribution
# --------------------------------------------------------------------------
BLOCK_OF = {"full_sample": ["motion", "diagnosis"], "mdd_only": ["motion", "clinical"]}
POP_N = {"full_sample": (2222, 23), "mdd_only": (484, 9)}


def build_table2() -> None:
    fam = {}
    for pop, blocks in BLOCK_OF.items():
        for b in blocks:
            fam[(pop, b)] = read_json(f"results/rq3/families/{pop}__{b}.json")

    rows = []
    for pop, blocks in BLOCK_OF.items():
        n_sub, n_sites = POP_N[pop]
        for op in ("OP50", "OP30"):
            for method in ("probability", "bootstrap", "mahalanobis"):
                cell = read_json(f"results/rq3/cells/{pop}__{op}__{method}.json")
                for b in blocks:
                    blk = cell["blocks"][b]
                    slots = fam[(pop, b)]["slots"]
                    slot = next(s for s in slots
                                if s["method"] == method and s["operating_point"] == op)
                    est = blk["status"] == "ESTIMABLE"
                    rows.append(
                        {
                            "population": pop,
                            "n_complete_case": n_sub,
                            "n_sites": n_sites,
                            "block": b,
                            "operating_point": op,
                            "confidence_method": method,
                            "status": blk["status"],
                            "failure_code": blk.get("failing_condition") or "",
                            "df": blk["df"],
                            "lrt_statistic": blk["lrt_statistic"] if est else None,
                            "p_raw": slot["p_raw"] if est else "NOT_ESTIMABLE",
                            "p_holm": slot["p_holm"] if est else "NOT_ESTIMABLE",
                            "holm_significant": bool(slot["significant"]),
                            "delta_mcfadden": blk["delta_mcfadden"] if est else None,
                            "delta_cv_logloss": blk["delta_cv_logloss"] if est else None,
                            "sigma_site_sq": cell["sigma_site_sq"],
                            "sigma_site_sq_interpretation": cell["sigma_site_sq_interpretation"],
                            "holm_family": f"{pop}__{b}",
                            "holm_family_size": fam[(pop, b)]["m"],
                            "holm_alpha": fam[(pop, b)]["alpha"],
                            "node": "K40 (authoritative)",
                            "source_artifact": f"results/rq3/cells/{pop}__{op}__{method}.json; "
                                               f"results/rq3/families/{pop}__{b}.json",
                            "source_authority": AUTHORITIES["rq3"]["tag"],
                        }
                    )
    t2 = pd.DataFrame(rows)
    write(t2, "table2_rq3_attribution.csv")

    sig = (
        t2[["population", "operating_point", "confidence_method",
            "sigma_site_sq", "sigma_site_sq_interpretation"]]
        .drop_duplicates()
        .sort_values(["population", "operating_point", "confidence_method"])
    )
    # a NOT_ESTIMABLE cell contributes no sigma^2; mark it explicitly
    ne = (
        t2.groupby(["population", "operating_point", "confidence_method"])["status"]
        .apply(lambda s: (s == "ESTIMABLE").any())
        .rename("any_block_estimable")
        .reset_index()
    )
    sig = sig.merge(ne, on=["population", "operating_point", "confidence_method"])
    sig["sigma_site_sq_reportable"] = np.where(sig["any_block_estimable"],
                                               sig["sigma_site_sq"], np.nan)
    sig["source_artifact"] = "results/rq3/cells/*.json"
    sig["source_authority"] = AUTHORITIES["rq3"]["tag"]
    write(sig, "table2_rq3_sigma_site.csv")


# --------------------------------------------------------------------------
# Stage-F secondary block (RQ4 + sensitivities + SVM)
# --------------------------------------------------------------------------
def build_stage_f() -> None:
    sm = read_csv("results/stage_f_v2/rq4/site_macro_metrics.csv")
    mg = read_csv("results/stage_f_v2/rq4/marginal_metrics.csv")
    svm = read_json("results/stage_f_v2/svm/site_macro_bacc.json")
    per_site = read_csv("results/stage_f_v2/svm/per_site_bacc.csv")

    write(sm.assign(source_artifact="results/stage_f_v2/rq4/site_macro_metrics.csv",
                    source_authority=AUTHORITIES["stage_f"]["tag"]),
          "stage_f_rq4_site_macro.csv")
    write(mg.assign(source_artifact="results/stage_f_v2/rq4/marginal_metrics.csv",
                    source_authority=AUTHORITIES["stage_f"]["tag"]),
          "stage_f_rq4_marginal.csv")

    svm_row = pd.DataFrame([{
        "quantity": "linear_svm_site_macro_bacc",
        "status": svm.get("status"),
        "site_macro_bacc": svm.get("site_macro_bacc"),
        "n_outer_folds": len(per_site),
        "n_estimable_outer_folds": int((per_site["bacc"].notna()).sum()),
        "n_not_estimable_outer_folds": int((per_site["bacc"].isna()).sum()),
        "n_svm_not_converged": int((~per_site["converged"].astype(bool)).sum()),
        "headline_reason_code": svm.get("reason") or svm.get("reason_code"),
        "source_artifact": "results/stage_f_v2/svm/site_macro_bacc.json; "
                           "results/stage_f_v2/svm/per_site_bacc.csv",
        "source_authority": AUTHORITIES["stage_f"]["tag"],
    }])
    write(svm_row, "stage_f_svm_status.csv")


# --------------------------------------------------------------------------
def main() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    print("Stage G — building source data from frozen authorities")
    build_fig1()
    build_fig2()
    build_table1()
    build_table2()
    build_stage_f()

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    tags = {}
    for name, a in AUTHORITIES.items():
        for k in ("tag", "protocol_tag"):
            if k in a:
                c = subprocess.run(["git", "rev-list", "-n1", a[k]], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip()
                tags[a[k]] = c
    prov = {
        "stage": "G",
        "purpose": "manuscript-facing statistical synthesis; presentation only",
        "new_scientific_experiment_performed": False,
        "head_at_build": head,
        "authority_inventory": AUTHORITIES,
        "tag_commits": tags,
        "input_artifact_sha256": dict(sorted(_READS.items())),
    }
    (OUT / "PROVENANCE.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"  wrote PROVENANCE.json ({len(_READS)} frozen inputs recorded)")


if __name__ == "__main__":
    main()
