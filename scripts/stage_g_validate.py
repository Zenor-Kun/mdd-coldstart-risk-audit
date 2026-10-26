#!/usr/bin/env python3
"""Stage G — deterministic validation of every Stage-G artifact.

Three independent things are checked:

1. **Frozen-authority integrity.** Every CLOSED-stage checksum ledger is
   re-verified; Stage G must not have touched a single byte of any earlier
   result store.
2. **Number provenance.** Every value in a Stage-G source-data file is
   re-derived directly from the frozen authority and compared for exact
   equality (bit-exact for floats read back through the same CSV round trip,
   ``<= 1e-12`` for values re-aggregated by a frozen rule).
3. **Figure / table fidelity.** Every row a figure plots, and every cell a
   table renders, is matched back to a source-data row; denominators, statuses
   and site counts are checked; no hidden filtering, no interpolation.

Exit code 0 = PASS. Any failure prints and exits 1.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SG = ROOT / "results" / "stage_g"
SRC = SG / "source_data"
TAB = SG / "tables"

FAILURES: list[str] = []
CHECKS = 0


def check(cond: bool, label: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(f"{label}" + (f"  ({detail})" if detail else ""))


def close(a, b, tol=1e-12) -> bool:
    a, b = float(a), float(b)
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol)


# ==========================================================================
# 1. Frozen-authority integrity
# ==========================================================================
def validate_frozen_stores() -> None:
    print("\n[1] frozen CLOSED-stage authority integrity")
    stores = ["stage_f_v2", "stage_f_v2_freeze", "stage_f", "stage_f_freeze",
              "rq3_freeze", "rq2_s2_freeze", "rq1_s0", "rq1_s1"]
    for d in stores:
        led = ROOT / "results" / d / "RESULT_CHECKSUMS.sha256"
        if not led.exists():
            print(f"    {d:22s} no checksum ledger (skipped)")
            continue
        r = subprocess.run(["sha256sum", "-c", "RESULT_CHECKSUMS.sha256"],
                           cwd=ROOT / "results" / d, capture_output=True, text=True)
        n = len(led.read_text().strip().splitlines())
        check(r.returncode == 0, f"checksum ledger {d}", r.stdout[-300:])
        print(f"    {d:22s} {n:4d} entries  "
              f"{'OK' if r.returncode == 0 else 'FAILED'}")

    r = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                       capture_output=True, text=True)
    dirty = [ln for ln in r.stdout.splitlines()
             if not re.search(r"(results/stage_g/|scripts/stage_g_|"
                              r"docs/STAGE_G_REPORT\.md|docs/EXPERIMENT_LOG\.md|"
                              r"\.gitignore)", ln)]
    check(not dirty, "no non-Stage-G working-tree change", "; ".join(dirty[:5]))
    print(f"    working tree: {len(dirty)} non-Stage-G modifications")


# ==========================================================================
# 2. Number provenance
# ==========================================================================
OUT_OF_SCOPE_TOKENS = ("patient_weighted_ber", "ber_patient_weighted",
                       "patient_weighted_selective", "sum_n_s",
                       "STAGE_G_DERIVED_AGGREGATION")


def validate_fig1_source() -> None:
    print("\n[2a] Fig. 1 source data vs frozen RQ1/S0 and RQ1/S1 stores")

    # ---- scope guard: the out-of-scope aggregate must be entirely gone -----
    check(not (SRC / "fig1_left_risk_coverage.csv").exists(),
          "the superseded fig1_left_risk_coverage.csv is removed")
    for f in sorted(SG.rglob("*.csv")) + sorted(SG.rglob("*.md")) + \
             sorted(SG.rglob("*.json")):
        txt = f.read_text(errors="ignore")
        for tok in OUT_OF_SCOPE_TOKENS:
            check(tok not in txt,
                  f"out-of-scope token '{tok}' absent from {f.relative_to(ROOT)}")
    rc = pd.read_csv(SRC / "fig1_risk_coverage_site_balanced.csv")
    check(not [c for c in rc.columns if "patient" in c.lower()],
          "no patient-weighted column in the risk-coverage source data")
    check(not [c for c in rc.columns if "weight" in c.lower()
               and c != "aggregation_rule"],
          "no acquisition-count-weighted selective-risk field survives")
    print("    out-of-scope patient-weighted selective-BER aggregate: absent")

    # ---- Fig. 1a: the frozen AURC comparison, verbatim --------------------
    a = pd.read_csv(SRC / "fig1_aurc_aggregation.csv")
    raw = pd.read_csv(ROOT / "results/rq1_s0/aurc_summary.csv").set_index("method")
    iv = pd.read_csv(ROOT / "results/rq1_s0/site_bootstrap_intervals.csv")
    per_site = pd.read_csv(SRC / "fig1_per_site_aurc.csv")
    raw_ps = pd.read_csv(ROOT / "results/rq1_s0/per_site_aurc.csv")

    check(len(a) == 6, "6 AURC rows (3 methods x 2 aggregations)", str(len(a)))
    check(set(a.provenance) == {"FROZEN_REGISTERED_QUANTITY"},
          "every AURC row is a frozen registered quantity")
    for _, r in a.iterrows():
        col = ("patient_weighted_aurc" if r.aggregation == "patient_weighted"
               else "site_balanced_aurc")
        check(close(raw.loc[r.method, col], r.aurc, 0.0),
              f"frozen {r.aggregation} AURC reproduces exactly for {r.method}")
        check(close(raw.loc[r.method, col + "_ci_low"], r.ci_low, 0.0),
              f"frozen {r.aggregation} AURC CI low exact for {r.method}")
        check(close(raw.loc[r.method, col + "_ci_high"], r.ci_high, 0.0),
              f"frozen {r.aggregation} AURC CI high exact for {r.method}")
        reg = iv[(iv["quantity"] == f"{r.aggregation}_aurc")
                 & (iv["method_or_contrast"] == r.method)]
        check(len(reg) == 1, f"one registered interval for {r.aggregation}/{r.method}")
        reg = reg.iloc[0]
        check(close(reg["point_estimate"], r.aurc, 0.0),
              f"registered-interval point estimate matches for {r.method}")
        check(close(reg["ci_low"], r.ci_low, 0.0) and close(reg["ci_high"], r.ci_high, 0.0),
              f"registered interval bounds match aurc_summary for {r.method}")
        check(int(reg["valid_bootstrap_replicates"]) == 5000
              and int(reg["n_estimable_sites"]) == 23,
              f"registered CI ledger for {r.method}")
        for stat in ("min", "median", "max"):
            check(close(raw.loc[r.method, f"{stat}_site_aurc"],
                        r[f"{stat}_site_aurc"], 0.0),
                  f"frozen {stat} per-site AURC exact for {r.method}")
        ps = raw_ps[raw_ps.method == r.method]
        check(len(ps) == 23, f"23 per-site AURC values for {r.method}")
        check(close(ps.n_s.sum(), 2232), f"per-site n_s sums to 2232 for {r.method}")
        if r.aggregation == "patient_weighted":
            check(close(r.aggregation_denominator_value, 2232),
                  "patient-weighted AURC denominator is the full 2232 acquisitions")
            check(r.aggregation_denominator_name
                  == "acquisitions_summed_over_all_23_sites",
                  "patient-weighted denominator is explicitly named")
        else:
            check(close(r.aggregation_denominator_value, 23),
                  "site-balanced AURC denominator is 23 equally weighted sites")
    check(len(per_site) == 69 and set(per_site.method) == {"probability",
                                                           "bootstrap",
                                                           "mahalanobis"},
          "per-site AURC source data is the frozen 69-row table")
    print("    frozen AURC comparison: 6 estimates + 6 registered CIs reproduce "
          "bit-exactly")

    # ---- Fig. 1b: lawful site-balanced risk-coverage only -----------------
    check(len(rc) == 2 * 3 * 17, "risk-coverage row count", str(len(rc)))
    check(set(rc.provenance) == {"FROZEN_AGGREGATE__RQ1_S0_PROTOCOL_SECTION_9"},
          "every risk-coverage row is the frozen section-9 aggregate")
    for regime, rel in (("S0", "results/rq1_s0/risk_coverage_all_q.csv"),
                        ("S1", "results/rq1_s1/risk_coverage_all_q.csv")):
        src = pd.read_csv(ROOT / rel)
        src = src[src["q"].astype(str) != "NO_ABSTENTION"].copy()
        src["q"] = src["q"].astype(float)
        for _, row in rc[rc.regime == regime].iterrows():
            g = src[(src.method == row.method) & (src.q == row.q)]
            check(len(g) == 23, f"{regime}/{row.method}/q={row.q} has 23 sites")
            ber = pd.to_numeric(g["BER_sel_target"], errors="coerce").to_numpy(float)
            cov = pd.to_numeric(g["coverage_target"], errors="coerce").to_numpy(float)
            mb = np.isfinite(ber)
            check(int(mb.sum()) == int(row.ber_denominator_sites),
                  f"{regime}/{row.method}/q={row.q} BER site denominator")
            check(int(np.isfinite(cov).sum()) == int(row.coverage_denominator_sites),
                  f"{regime}/{row.method}/q={row.q} coverage site denominator")
            check(close(np.mean(ber[mb]), row.ber_site_balanced),
                  f"{regime}/{row.method}/q={row.q} site-balanced BER")
            check(close(np.mean(cov), row.coverage_site_balanced),
                  f"{regime}/{row.method}/q={row.q} site-balanced coverage")
    varying = rc[rc.ber_denominator_sites < rc.n_sites_in_fold_set]
    check(len(varying) == 33, "33 varying-denominator BER points are labelled",
          str(len(varying)))
    check(bool((varying.ber_denominator_sites < 23).all()),
          "every varying-denominator row records its actual site denominator")
    print(f"    site-balanced risk-coverage: 102 points; "
          f"{len(varying)} carry a reduced site denominator, all recorded")

    rep = (ROOT / "results/rq1_s0/RQ1_S0_REPORT.md").read_text()
    n_hit = sum(
        1 for _, row in rc[rc.regime == "S0"].iterrows()
        if f"| {row.coverage_site_balanced:.4f} | {row.ber_site_balanced:.4f} |" in rep
    )
    check(n_hit == 51, "all 51 S0 curve points reproduce the frozen S0 report",
          f"matched {n_hit}/51")
    print(f"    S0 curve points reproducing the frozen S0 report: {n_hit}/51")

    # ---- Fig. 1c: per-site acceptance ------------------------------------
    acc = pd.read_csv(SRC / "fig1_right_site_acceptance.csv")
    check(len(acc) == 2 * 2 * 3 * 23, "acceptance row count", str(len(acc)))
    cohort = pd.read_csv(ROOT / "results/data_contract/post_dedup_site_counts.csv")
    cohort = cohort[cohort.eligible_post_dedup].set_index("site")
    for _, r in acc.iterrows():
        check(int(cohort.loc[r.site, "n_total_acq"]) == int(r.n_s),
              f"acceptance n_s for {r.site}")
    raw = pd.read_csv(ROOT / "results/rq1_s0/risk_coverage_all_q.csv")
    for _, r in acc[(acc.regime == "S0") & (acc.operating_point == "OP50")].iterrows():
        g = raw[(raw.held_out_site == r.site) & (raw.method == r.method)
                & (raw.operating_point == "OP50")]
        check(len(g) == 1 and close(float(g.coverage_target.iloc[0]), r.acceptance_rate),
              f"OP50 acceptance {r.site}/{r.method}")


def validate_fig2_source() -> None:
    print("\n[2b] Fig. 2 source data vs the frozen RQ2/S2 store")
    iv = pd.read_csv(SRC / "fig2_certification_intervals.csv")
    raw = pd.read_csv(ROOT / "results/rq2_s2/registered_intervals.csv")
    for _, r in iv.iterrows():
        m = ((raw.quantity == r.quantity) & (raw.strategy == r.strategy)
             & (raw.method == r.method))
        m &= raw.n_labels.isna() if pd.isna(r.n_labels) else (raw.n_labels == r.n_labels)
        g = raw[m]
        check(len(g) == 1, f"unique registered interval for {r.quantity}/{r.strategy}")
        if len(g) != 1:
            continue
        g = g.iloc[0]
        check(str(g.ci_low) == str(r.ci_low) and str(g.ci_high) == str(r.ci_high),
              f"CI verbatim for {r.quantity}/{r.strategy}/N={r.n_labels}/{r.method}")
        check(int(g.n_sites) == int(r.n_sites), f"n_sites {r.quantity}")
        if r.status == "ESTIMABLE":
            check(close(g.point_estimate, r.point_estimate),
                  f"point estimate {r.quantity}")
        else:
            check(float(g.n_estimable_sites) == 0.0,
                  f"NOT_ESTIMABLE has 0 estimable sites: {r.quantity}")
    check(int((iv.status == "ESTIMABLE").sum()) == 36, "36 estimable Fig-2 intervals",
          str(int((iv.status == "ESTIMABLE").sum())))
    check(int((iv.status == "NOT_ESTIMABLE").sum()) == 36,
          "36 NOT_ESTIMABLE Fig-2 intervals")
    est = iv[iv.status == "ESTIMABLE"]
    zero = est[~est.quantity.str.contains("censored")]
    check(bool((zero.point_estimate.astype(float) == 0.0).all()),
          "every non-censoring estimable Fig-2 estimate is exactly 0.000")
    cens = est[est.quantity.str.contains("censored")]
    check(bool((cens.point_estimate.astype(float) == 1.0).all()),
          "every censoring probability is exactly 1.000")
    print(f"    intervals verified verbatim: {len(iv)}  "
          f"(36 ESTIMABLE, 36 NOT_ESTIMABLE)")

    bur = pd.read_csv(SRC / "fig2_uncertified_burden.csv")
    fps = json.loads((ROOT / "results/rq2_s2/formal_point_sections.json").read_text())
    payload = fps["expected_uncertified_site_burden"]["payload"]["by_method_n"]
    check(len(bur) == len(payload) == 12, "12 burden cells")
    for _, r in bur.iterrows():
        v = payload[f"{r.method}|{int(r.n_labels)}"]
        check(close(v["expected_uncertified_site_burden"],
                    r.expected_uncertified_site_burden),
              f"burden {r.method}/N={r.n_labels}")
        check(int(v["n_sites"]) == int(r.n_eligible_sites), f"eligible {r.method}/N={r.n_labels}")
        check(int(v["n_sites_at_one"]) == 0, f"0 certified {r.method}/N={r.n_labels}")
    expect = {5: 23, 10: 22, 20: 21, 40: 11}
    for n, k in expect.items():
        g = bur[bur.n_labels == n]
        check(set(g.expected_uncertified_site_burden) == {float(k)},
              f"burden at N={n} is {k}")
    print(f"    uncertified burden by N: "
          f"{ {int(n): int(v) for n, v in expect.items()} }")


def validate_table_sources() -> None:
    print("\n[2c] Table source data vs the frozen Gate-1 and RQ3 stores")
    sites = pd.read_csv(SRC / "table1_sites.csv")
    folds = pd.read_csv(ROOT / "results/gate1/outer_fold_metrics.csv").set_index("site")
    rst = pd.read_csv(
        ROOT / "results/gate1/source_full_coverage_error_by_fold.csv").set_index("site")
    check(len(sites) == 23 and int(sites.n_acquisitions.sum()) == 2232,
          "23 sites / 2232 acquisitions")
    check(int(sites.n_MDD.sum()) == 1182 and int(sites.n_HC.sum()) == 1050,
          "1182 MDD / 1050 HC")
    for r in sites.itertuples():
        check(close(folds.loc[r.site, "bacc"], r.outer_loso_bacc), f"LOSO BAcc {r.site}")
        check(close(rst.loc[r.site, "e_full_minus_s"] - 0.10, r.r_star_s),
              f"r*_s {r.site}")
    g1 = json.loads((ROOT / "results/gate1/gate1_results.json").read_text())
    check(close(sites.outer_loso_bacc.mean(), g1["observed"]["macro_site_bacc"]["mean"]),
          "per-site BAcc mean equals the frozen macro BAcc")
    print(f"    macro BAcc re-derived from 23 per-site values: "
          f"{sites.outer_loso_bacc.mean():.16f}")

    gates = pd.read_csv(SRC / "table1_rq0_gates.csv").set_index("quantity")
    perm = json.loads(
        (ROOT / "results/gate1_formal_permutation/permutation_summary.json").read_text())
    check(close(gates.loc["permutation_p_value", "value"], perm["p_value"]),
          "permutation p verbatim")
    check(close(gates.loc["permutation_p_value", "value"], 1.0 / 201.0),
          "permutation p equals (1+0)/(200+1)")
    check(perm["n_null_ge_observed"] == 0 and perm["B"] == 200,
          "0 of 200 null draws reach the observed statistic")
    check(close(gates.loc["site_balanced_accuracy", "value"],
                g1["site_decodability"]["balanced_accuracy"]), "site decodability")
    check("permutation_smoke" not in gates.index.tolist(),
          "smoke permutation is NOT used as a final number")

    t2 = pd.read_csv(SRC / "table2_rq3_attribution.csv")
    check(len(t2) == 24, "24 RQ3 attribution units")
    check(int((t2.status == "ESTIMABLE").sum()) == 18, "18 ESTIMABLE")
    check(int((t2.status == "NOT_ESTIMABLE").sum()) == 6, "6 NOT_ESTIMABLE")
    check(int(t2.holm_significant.sum()) == 6, "6 Holm-supported")
    for _, r in t2.iterrows():
        cell = json.loads(
            (ROOT / f"results/rq3/cells/{r.population}__{r.operating_point}"
                    f"__{r.confidence_method}.json").read_text())
        blk = cell["blocks"][r.block]
        check(blk["status"] == r.status, f"status {r.population}/{r.block}")
        fam = json.loads(
            (ROOT / f"results/rq3/families/{r.population}__{r.block}.json").read_text())
        slot = next(s for s in fam["slots"]
                    if s["method"] == r.confidence_method
                    and s["operating_point"] == r.operating_point)
        if r.status == "ESTIMABLE":
            check(close(blk["lrt_statistic"], r.lrt_statistic), "LRT")
            check(close(slot["p_holm"], float(r.p_holm)), "Holm p")
            check(close(blk["delta_mcfadden"], r.delta_mcfadden), "delta McFadden")
            check(close(blk["delta_cv_logloss"], r.delta_cv_logloss), "delta CV")
        else:
            check(slot["p_holm"] == "NOT_ESTIMABLE", "NOT_ESTIMABLE Holm slot")
            check(bool(slot["significant"]) is False,
                  "NOT_ESTIMABLE slot is never significant")
        check(bool(slot["significant"]) == bool(r.holm_significant), "significance flag")
    check(set(t2.n_complete_case) == {2222, 484}, "frozen complete-case counts")
    print(f"    RQ3 units verified: 24  (18 ESTIMABLE / 6 NOT_ESTIMABLE / "
          f"6 Holm-supported)")

    svm = pd.read_csv(SRC / "stage_f_svm_status.csv").iloc[0]
    raw = json.loads(
        (ROOT / "results/stage_f_v2/svm/site_macro_bacc.json").read_text())
    check(raw["site_macro_bacc"] is None and raw["status"] == "NOT_ESTIMABLE",
          "SVM macro BAcc is null / NOT_ESTIMABLE")
    check(int(svm.n_estimable_outer_folds) == 0
          and int(svm.n_not_estimable_outer_folds) == 23
          and int(svm.n_svm_not_converged) == 23,
          "SVM 0 / 23 / 23")
    print("    SVM: 0 ESTIMABLE, 23 NOT_ESTIMABLE, 23 SVM_NOT_CONVERGED")


# ==========================================================================
# 3. Figure / table fidelity
# ==========================================================================
FORBIDDEN_SVM_BACC = "0.5791186452230204"


def validate_renderings() -> None:
    print("\n[3] figure and table renderings")
    for stem in ("fig1_risk_coverage_and_site_acceptance",
                 "fig2_certification_label_cost"):
        for ext in ("pdf", "png"):
            p = SG / "figures" / f"{stem}.{ext}"
            check(p.exists() and p.stat().st_size > 0, f"{stem}.{ext} exists")
        check((SG / "figures" / f"{stem}.pdf").read_bytes()[:5] == b"%PDF-",
              f"{stem}.pdf is vector PDF")

    # Fig. 1 plots exactly the source-data rows, nothing filtered away
    rc = pd.read_csv(SRC / "fig1_risk_coverage_site_balanced.csv")
    acc = pd.read_csv(SRC / "fig1_right_site_acceptance.csv")
    check(rc.groupby(["regime", "method"]).size().eq(17).all(),
          "17 q values plotted per regime and method (no coverage-grid change)")
    fig_src = (ROOT / "scripts/stage_g_figures.py").read_text()
    check("fig1_left_risk_coverage" not in fig_src,
          "the figure script no longer reads the superseded source file")
    check("ber_patient_weighted" not in fig_src and "patient_weighted(" not in fig_src,
          "the figure script plots no patient-weighted selective-risk quantity")
    check("fig1_aurc_aggregation.csv" in fig_src,
          "Fig. 1a is driven by the frozen AURC source-data artifact")
    plotted = acc[(acc.regime == "S0") & (acc.operating_point == "OP50")]
    check(plotted.site.nunique() == 23, "all 23 sites are drawn in Fig. 1c",
          "no site may be hidden")
    check(plotted.groupby("site").size().eq(3).all(),
          "all 3 confidence methods drawn per site")

    # Fig. 2 draws no fabricated value for a NOT_ESTIMABLE quantity
    iv = pd.read_csv(SRC / "fig2_certification_intervals.csv")
    ne = iv[iv.status == "NOT_ESTIMABLE"]
    check(ne.point_estimate.isna().all(),
          "NOT_ESTIMABLE Fig-2 quantities carry no numeric point estimate")

    # tables
    t1 = (TAB / "table1_cohort_and_rq0.md").read_text()
    t1c = (TAB / "table1_compact.md").read_text()
    t2 = (TAB / "table2_rq3_attribution.md").read_text()
    for tok in ("2232", "1182", "1050", "0.582", "0.689", "0.004975", "0.501"):
        check(tok in t1 or tok in t1c, f"Table 1 carries {tok}")
    check("reduced-cost" in t1, "Table 1 labels the permutation as reduced-cost")
    check("smoke" in t1 and "not** a final result" in t1,
          "Table 1 excludes the smoke permutation explicitly")
    sites = pd.read_csv(SRC / "table1_sites.csv")
    for r in sites.itertuples():
        check(f"| {r.site} | {int(r.n_acquisitions)} | {int(r.n_MDD)} | "
              f"{int(r.n_HC)} |" in t1, f"Table 1 row {r.site}")
    check("NOT_ESTIMABLE" in t2 and "missing evidence, not null evidence" in t2,
          "Table 2 keeps NOT_ESTIMABLE as missing evidence")
    check("site-associated acquisition heterogeneity" in t2,
          "Table 2 uses the frozen heterogeneity wording")
    check("scanner effect" not in t2.lower().replace("scanner-effect wording is", ""),
          "Table 2 does not assert scanner effects")
    check("same complete-case" in t2, "Table 2 states the same-complete-case rule")

    t2src = pd.read_csv(SRC / "table2_rq3_attribution.csv")
    for _, r in t2src[t2src.status == "ESTIMABLE"].iterrows():
        check(f"{float(r.lrt_statistic):.4f}" in t2,
              f"Table 2 LRT cell {r.population}/{r.block}/"
              f"{r.operating_point}/{r.confidence_method}")

    # the invalid historical SVM number must appear nowhere in Stage G
    for p in list(SG.rglob("*.csv")) + list(SG.rglob("*.md")) + \
             list(SG.rglob("*.json")) + [ROOT / "docs/STAGE_G_REPORT.md"]:
        if not p.exists():
            continue
        txt = p.read_text(errors="ignore")
        check(FORBIDDEN_SVM_BACC not in txt,
              f"historical invalid SVM BAcc absent from {p.relative_to(ROOT)}")
    print("    historical v1 SVM BAcc: absent from every Stage-G artifact")


def validate_ledger() -> None:
    print("\n[4] final manuscript-number ledger")
    led = pd.read_csv(SG / "final_number_ledger.csv")
    required = {"PRIMARY", "SECONDARY", "SENSITIVITY", "LIMITATION", "NOT_ESTIMABLE"}
    check(required <= set(led.classification.unique()),
          "all five ledger classifications present")
    check(led.source_artifact.notna().all() and (led.source_artifact != "").all(),
          "no orphan number: every ledger row names a source artifact")
    check(led.source_frozen_authority.notna().all(),
          "no orphan number: every ledger row names a frozen authority")
    check(led.key.is_unique, "ledger keys are unique",
          str(led.key[led.key.duplicated()].tolist()[:5]))
    for prefix in ("cohort.", "rq0.", "s0.", "s1.", "s2.", "rq3.", "rq4.",
                   "svm.", "stage_f.sens"):
        check(led.key.str.startswith(prefix).any(),
              f"ledger covers {prefix}")
    rq3 = led[led.key.str.startswith("rq3.")
              & (led.value_full_precision != "NOT_ESTIMABLE")]
    check(len(rq3) == 18, "18 estimable RQ3 ledger rows", str(len(rq3)))
    for _, r in rq3.iterrows():
        v = float(r.value_full_precision)
        want = f"{v:.4e}" if v < 1e-3 else f"{v:.4f}"
        check(str(r.manuscript_rounded) == want,
              f"RQ3 manuscript_rounded obeys the rounding policy: {r.key}",
              f"{r.manuscript_rounded} != {want}")
        check(str(r.value_full_precision) == repr(v) or
              close(float(r.value_full_precision), v, 0.0),
              f"RQ3 full-precision value preserved: {r.key}")
    aurc_rows = led[led.key.str.contains("aurc")]
    check(len(aurc_rows) == 15, "6 AURC estimates + 9 per-site summaries",
          str(len(aurc_rows)))
    check(led.key.str.contains("patient").sum() == 3
          and bool(led[led.key.str.contains("patient")].key.str.contains("aurc").all()),
          "every patient-weighted ledger row is an AURC row")
    check((led[led.classification == "LIMITATION"].key == "svm.site_macro_bacc").all(),
          "SVM is the LIMITATION row")
    check(led[led.key == "svm.site_macro_bacc"].value_full_precision.iloc[0]
          == "NOT_ESTIMABLE", "SVM value is NOT_ESTIMABLE")
    tags = set(led.source_frozen_authority.unique())
    print(f"    {len(led)} rows; classifications "
          f"{dict(led.classification.value_counts())}")
    print(f"    frozen authorities cited: {sorted(tags)}")


def main() -> int:
    print("=" * 74)
    print("STAGE G — DETERMINISTIC VALIDATION")
    print("=" * 74)
    validate_frozen_stores()
    validate_fig1_source()
    validate_fig2_source()
    validate_table_sources()
    validate_renderings()
    validate_ledger()
    print("\n" + "=" * 74)
    if FAILURES:
        print(f"FAIL — {len(FAILURES)} of {CHECKS} checks failed:")
        for f in FAILURES[:60]:
            print("  - " + f)
        return 1
    print(f"PASS — {CHECKS} checks, 0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
