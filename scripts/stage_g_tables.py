#!/usr/bin/env python3
"""Stage G — render Table 1, Table 2 and the final manuscript-number ledger.

Consumes only the Stage-G source-data CSVs (which are themselves traceable to
frozen CLOSED-stage authorities) plus the frozen registered-interval stores for
the ledger rows that no figure needs. No analysis, no new statistic, no new
test, no new interval.

Outputs Markdown + LaTeX renderings under ``results/stage_g/tables/`` and the
ledger at ``results/stage_g/final_number_ledger.csv``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SG = ROOT / "results" / "stage_g"
SRC = SG / "source_data"
TAB = SG / "tables"

AUTH = json.loads((SG / "PROVENANCE.json").read_text())["authority_inventory"]

# --------------------------------------------------------------------------
# Manuscript rounding policy — applied at render time only. The ledger keeps
# full precision.
# --------------------------------------------------------------------------
def p3(x) -> str:
    """proportions / coverage / BAcc / BER: 3 decimals"""
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{float(x):.3f}"


def pct1(x) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100*float(x):.1f}%"


def pval(x) -> str:
    """p-values keep the frozen convention: exact for the permutation p,
    3 significant digits (scientific below 1e-3) for Holm-adjusted p."""
    if isinstance(x, str):
        return x
    x = float(x)
    if x < 1e-3:
        return f"{x:.4e}"
    return f"{x:.4f}"


def ci(lo, hi, fmt=p3) -> str:
    if isinstance(lo, str) or isinstance(hi, str):
        return "NOT_ESTIMABLE"
    return f"[{fmt(lo)}, {fmt(hi)}]"


def md_table(rows, header, align=None) -> str:
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ==========================================================================
# Table 1
# ==========================================================================
def table1() -> tuple[str, str]:
    sites = pd.read_csv(SRC / "table1_sites.csv")
    gates = pd.read_csv(SRC / "table1_rq0_gates.csv")

    assert len(sites) == 23
    assert int(sites.n_acquisitions.sum()) == 2232
    assert int(sites.n_MDD.sum()) == 1182
    assert int(sites.n_HC.sum()) == 1050

    site_rows = [
        [r.site, int(r.n_acquisitions), int(r.n_MDD), int(r.n_HC),
         p3(r.outer_loso_bacc), p3(r.r_star_s)]
        for r in sites.itertuples()
    ]
    site_rows.append(["**All 23**", "**2232**", "**1182**", "**1050**",
                      f"**{p3(sites.outer_loso_bacc.mean())}**",
                      f"**{p3(sites.r_star_s.mean())}**"])

    g = gates.set_index("quantity")
    gate_rows = [
        ["(a) outer LOSO-site macro BAcc",
         f"{p3(g.loc['macro_site_bacc','value'])} "
         f"(SD {p3(g.loc['macro_site_bacc','dispersion_sd'])}; "
         f"range {p3(g.loc['macro_site_bacc','minimum'])}–{p3(g.loc['macro_site_bacc','maximum'])})",
         "23 held-out sites", "chance 0.500"],
        ["(b) site decodability, balanced accuracy",
         p3(g.loc["site_balanced_accuracy", "value"]),
         "2232 acquisitions, 23 classes",
         f"chance {p3(g.loc['site_balanced_accuracy','value']*0+1/23)}"],
        ["(b) site decodability, macro F1",
         p3(g.loc["site_macro_f1", "value"]),
         "2232 acquisitions, 23 classes", "—"],
        ["(c) within-site permutation null, mean BAcc",
         f"{p3(g.loc['null_macro_site_bacc','value'])} "
         f"(SD {p3(g.loc['null_macro_site_bacc','dispersion_sd'])})",
         "B = 200 reduced-cost permutations", "0 / 200 reach the observed value"],
        ["(c) permutation p-value",
         f"{g.loc['permutation_p_value','value']:.6f}",
         "(1 + 0) / (200 + 1), upper-tailed", "—"],
        ["source-side full-coverage balanced error e_full,−s",
         f"{p3(g.loc['e_full_minus_s','value'])} "
         f"(range {p3(g.loc['e_full_minus_s','minimum'])}–{p3(g.loc['e_full_minus_s','maximum'])})",
         "23 outer folds, source sites only",
         "gives r*_s = e_full,−s − 0.10"],
    ]

    full = f"""### Table 1 — cohort, eligible sites and the RQ0 information / leakage gates

**A. Frozen analysis cohort — 2232 independent fMRI acquisitions, 23 eligible sites.**
One acquisition = one observational unit (`DATA_CONTRACT_v1.md` §3.12, Strategy D).

{md_table(site_rows, ["site", "n", "MDD", "HC", "LOSO BAcc", "r*_s"],
          ["---", "---:", "---:", "---:", "---:", "---:"])}

*Sites ordered by acquisition count. `LOSO BAcc` is the held-out-site balanced
accuracy of the frozen ridge-logistic pipeline; `r*_s = e_full,−s − 0.10` is the
per-fold target risk, computed from outer-training sites only.*

**B. RQ0 gates.**

{md_table(gate_rows, ["gate", "value", "unit / denominator", "reference"],
          ["---", "---", "---", "---"])}

*The permutation shuffles diagnosis labels independently within each site,
preserving site structure and site prevalence, and is a **reduced-cost**
permutation (per-fold PCA dimension and ridge C frozen from the observed run;
outer training and held-out-site prediction re-run in full), pre-declared by
spec §4 because 200 × T = 107.9 h exceeded the 24 h budget (T = 1942.1 s). The
20-run Gate-1 smoke permutation is **not** a final result and is not reported.*
"""

    compact_rows = [
        ["Eligible sites", "23", "of 24 phenotyped sites; §5.4 rule ≥10 MDD and ≥10 HC"],
        ["Independent acquisitions", "2232", "1182 MDD / 1050 HC"],
        ["Site size",
         f"{int(sites.n_acquisitions.min())} – {int(sites.n_acquisitions.max())}",
         f"median {int(sites.n_acquisitions.median())} acquisitions"],
        ["Features", "6670", "Fisher-z AAL116 FC upper triangle, `ROISignals_FunImgARCWF`"],
        ["(a) outer LOSO-site macro BAcc",
         f"{p3(g.loc['macro_site_bacc','value'])} "
         f"({p3(g.loc['macro_site_bacc','minimum'])}–{p3(g.loc['macro_site_bacc','maximum'])})",
         "macro mean over 23 held-out sites; chance 0.500"],
        ["(b) site decodability BAcc", p3(g.loc["site_balanced_accuracy", "value"]),
         "23-class; chance 0.043"],
        ["(c) within-site permutation null",
         f"{p3(g.loc['null_macro_site_bacc','value'])} ± {p3(g.loc['null_macro_site_bacc','dispersion_sd'])}",
         "B = 200 reduced-cost; 0/200 ≥ observed"],
        ["(c) permutation p", f"{g.loc['permutation_p_value','value']:.6f}",
         "(1 + 0)/(200 + 1), upper-tailed"],
        ["e_full,−s (source, full coverage)",
         f"{p3(g.loc['e_full_minus_s','value'])} "
         f"({p3(g.loc['e_full_minus_s','minimum'])}–{p3(g.loc['e_full_minus_s','maximum'])})",
         f"r*_s = e_full,−s − 0.10, mean {p3(sites.r_star_s.mean())}"],
    ]
    compact = f"""### Table 1 (compact main-paper form) — cohort and RQ0 gates

{md_table(compact_rows, ["quantity", "value", "definition / denominator"],
          ["---", "---:", "---"])}

*Per-site counts, per-site LOSO BAcc and per-site r*_s for all 23 sites are in
`results/stage_g/source_data/table1_sites.csv` and in the full Table 1 rendering.*
"""
    return full, compact


# ==========================================================================
# Table 2
# ==========================================================================
BLOCK_TITLE = {
    ("full_sample", "motion"): "Full sample · motion block (mean FD), df 1",
    ("full_sample", "diagnosis"): "Full sample · diagnosis block, df 1",
    ("mdd_only", "motion"): "MDD-only · motion block (mean FD), df 1",
    ("mdd_only", "clinical"): "MDD-only · clinical block (HAMD, episodicity, medication, duration), df 4",
}
POP_NOTE = {"full_sample": "n = 2222 acquisitions, 23 sites",
            "mdd_only": "n = 484 acquisitions, 9 sites"}


def table2() -> str:
    t = pd.read_csv(SRC / "table2_rq3_attribution.csv")
    sig = pd.read_csv(SRC / "table2_rq3_sigma_site.csv")
    assert len(t) == 24
    assert int((t.status == "ESTIMABLE").sum()) == 18
    assert int((t.status == "NOT_ESTIMABLE").sum()) == 6
    assert int(t.holm_significant.sum()) == 6

    parts = []
    for (pop, blk), title in BLOCK_TITLE.items():
        g = t[(t.population == pop) & (t.block == blk)]
        rows = []
        for op in ("OP50", "OP30"):
            for method in ("probability", "bootstrap", "mahalanobis"):
                r = g[(g.operating_point == op) & (g.confidence_method == method)].iloc[0]
                if r.status != "ESTIMABLE":
                    rows.append([op, method, f"**NOT_ESTIMABLE** (`{r.failure_code}`)",
                                 "—", "—", "—", "—"])
                else:
                    rows.append([
                        op, method, "ESTIMABLE",
                        f"{float(r.lrt_statistic):.4f}",
                        pval(float(r.p_holm)) + (" **†**" if r.holm_significant else ""),
                        f"{float(r.delta_mcfadden):.6f}",
                        f"{float(r.delta_cv_logloss):+.6f}",
                    ])
        n_est = int((g.status == "ESTIMABLE").sum())
        n_sig = int(g.holm_significant.sum())
        parts.append(
            f"**{title}** — {POP_NOTE[pop]}; {n_est}/6 ESTIMABLE, "
            f"{n_sig}/6 Holm-supported (family size m = 6, α = 0.05)\n\n"
            + md_table(rows, ["OP", "confidence method", "status", "LRT",
                              "Holm p", "Δ McFadden R²", "ΔCV log-loss"],
                       ["---", "---", "---", "---:", "---:", "---:", "---:"])
        )

    srows = []
    sig = sig.assign(_op=sig.operating_point.map({"OP50": 0, "OP30": 1}),
                     _m=sig.confidence_method.map(
                         {"probability": 0, "bootstrap": 1, "mahalanobis": 2}))
    sig = sig.sort_values(["population", "_op", "_m"], ascending=[True, True, True])
    for r in sig.itertuples():
        v = "NOT_ESTIMABLE" if not r.any_block_estimable else f"{float(r.sigma_site_sq):.6f}"
        srows.append([r.population, r.operating_point, r.confidence_method, v])

    return f"""### Table 2 — RQ3 block-wise attribution of source-intended abstention

Outcome is `R_i = 1{{confidence < τ}}`: R = 1 means the acquisition was **rejected**
by the frozen source-intended S0 rule. Every nested and block comparison inside a
population uses **exactly the same complete-case rows** (`RQ3_PROTOCOL` §5.3);
missingness-driven sample changes are not treated as explanatory gains. The two
populations use different subsets and are never compared to each other. All values
are the authoritative K = 40 node.

{parts[0]}

{parts[1]}

{parts[2]}

{parts[3]}

**Residual site-associated acquisition heterogeneity** (σ²_site of the full model,
descriptive only — no variance-component test, no reference scale, and never
compared on a raw scale against incremental McFadden pseudo-R²):

{md_table(srows, ["population", "OP", "confidence method", "σ²_site"],
          ["---", "---", "---", "---:"])}

**† Holm-supported at α = 0.05 within its own family of m = 6.**

*Reading rules that must survive into the manuscript.* (i) `NOT_ESTIMABLE` is
**missing evidence, not null evidence** — no test was performed in those 6 of 24
units, and none is reported as non-significant, as p = 1 or as a zero effect.
(ii) The full-sample motion association is **method-dependent**: 4 of 5 estimable
cells are Holm-supported and all four are bootstrap or Mahalanobis; the
probability method is non-significant at OP50 and not estimable at OP30.
(iii) **No multiplicity-controlled evidence of a diagnosis association** was
detected (smallest Holm p in that family 0.077); this is not evidence that
diagnosis has no effect. (iv) MDD-only clinical attribution is **unsupported and
partly not estimable**, on a subset in which 236 of 484 acquisitions (48.8%) come
from a single site, and in both Holm-supported MDD-only motion cells the ΔCV
log-loss is **negative**, so in-sample likelihood evidence is not accompanied by
held-out-site predictive improvement. (v) Wording is *site-associated acquisition
heterogeneity* and *motion-associated*; **no scanner, vendor, field-strength,
sequence or protocol variable entered any model**, so scanner-effect wording is
not licensed. (vi) No standard error, coefficient CI or odds ratio is registered
in the frozen fit schema and none may be reported.
"""


# ==========================================================================
# Final manuscript-number ledger
# ==========================================================================
def ledger() -> pd.DataFrame:
    rows: list[dict] = []

    def add(cls, key, value, definition, unit, agg, ci_txt, artifact, authority,
            rounded, note=""):
        rows.append(dict(classification=cls, key=key, value_full_precision=value,
                         definition=definition, unit_scale=unit,
                         aggregation_level=agg, confidence_interval=ci_txt,
                         source_artifact=artifact, source_frozen_authority=authority,
                         manuscript_rounded=rounded, note=note))

    dc, g1 = AUTH["data_contract"]["tag"], AUTH["gate1_rq0"]["tag"]
    a_s0, a_s1 = AUTH["rq1_s0"]["tag"], AUTH["rq1_s1"]["tag"]
    a_s2, a_r3, a_sf = AUTH["rq2_s2"]["tag"], AUTH["rq3"]["tag"], AUTH["stage_f"]["tag"]

    # ---- cohort ---------------------------------------------------------
    sites = pd.read_csv(SRC / "table1_sites.csv")
    for k, v in (("n_acquisitions", 2232), ("n_MDD", 1182),
                 ("n_HC", 1050), ("n_eligible_sites", 23)):
        add("PRIMARY", f"cohort.{k}", v,
            "frozen analysis cohort after Strategy-D acquisition collapse",
            "count", "cohort", "n/a",
            "results/data_contract/post_dedup_site_counts.csv", dc, str(v))
    add("PRIMARY", "cohort.site_size_min_max", "24;531",
        "smallest and largest eligible site", "count", "site", "n/a",
        "results/data_contract/post_dedup_site_counts.csv", dc, "24 – 531")
    add("PRIMARY", "cohort.n_features", 6670,
        "Fisher-z AAL116 FC upper triangle", "count", "acquisition", "n/a",
        "docs/DATA_CONTRACT_v1.md §3.7–§3.9", dc, "6670")

    # ---- RQ0 ------------------------------------------------------------
    gates = pd.read_csv(SRC / "table1_rq0_gates.csv").set_index("quantity")
    add("PRIMARY", "rq0.macro_site_bacc", float(gates.loc["macro_site_bacc", "value"]),
        "outer LOSO-site macro-averaged balanced accuracy", "balanced accuracy",
        "macro mean over 23 held-out sites", "none registered",
        "results/gate1/gate1_results.json", g1,
        p3(gates.loc["macro_site_bacc", "value"]),
        "SD 0.066290, range 0.462–0.692")
    add("PRIMARY", "rq0.site_decodability_bacc",
        float(gates.loc["site_balanced_accuracy", "value"]),
        "23-class site decodability", "balanced accuracy",
        "5-fold stratified over 2232 acquisitions", "none registered",
        "results/gate1/site_decodability.csv", g1,
        p3(gates.loc["site_balanced_accuracy", "value"]), "chance 1/23 = 0.043")
    add("PRIMARY", "rq0.site_decodability_macro_f1",
        float(gates.loc["site_macro_f1", "value"]),
        "23-class site decodability", "macro F1",
        "5-fold stratified over 2232 acquisitions", "none registered",
        "results/gate1/site_decodability.csv", g1,
        p3(gates.loc["site_macro_f1", "value"]))
    add("PRIMARY", "rq0.permutation_null_mean",
        float(gates.loc["null_macro_site_bacc", "value"]),
        "within-site label-permutation null of the macro BAcc", "balanced accuracy",
        "B = 200 reduced-cost permutations", "none registered (null distribution)",
        "results/gate1_formal_permutation/permutation_summary.json", g1,
        p3(gates.loc["null_macro_site_bacc", "value"]),
        "SD 0.014268; 0/200 null draws reach the observed statistic")
    add("PRIMARY", "rq0.permutation_p_value",
        float(gates.loc["permutation_p_value", "value"]),
        "(1 + #{null >= observed}) / (B + 1), upper-tailed, finite-sample corrected",
        "p-value", "one test on the macro statistic", "n/a",
        "results/gate1_formal_permutation/permutation_summary.json", g1,
        f'{gates.loc["permutation_p_value","value"]:.6f}',
        "reduced-cost permutation; must be described as such in the paper")
    add("PRIMARY", "rq0.e_full_minus_s_mean",
        float(gates.loc["e_full_minus_s", "value"]),
        "source-side full-coverage balanced error, outer-training sites only",
        "balanced error rate", "mean over 23 outer folds", "none registered",
        "results/gate1/source_full_coverage_error_by_fold.csv", g1,
        p3(gates.loc["e_full_minus_s", "value"]), "range 0.393–0.419")
    add("PRIMARY", "rq0.r_star_mean", float(sites.r_star_s.mean()),
        "per-fold target risk r*_s = e_full,-s - 0.10", "balanced error rate",
        "mean over 23 outer folds", "none registered (rule is frozen, not the number)",
        "results/gate1/source_full_coverage_error_by_fold.csv", g1,
        p3(sites.r_star_s.mean()),
        f"range {p3(sites.r_star_s.min())}–{p3(sites.r_star_s.max())}")
    add("PRIMARY", "rq0.wall_clock_T_seconds", 1942.1416002930055,
        "one complete nested observed pipeline (spec §5.5 / §18)", "seconds",
        "single run", "n/a", "results/gate1/gate1_results.json", g1, "1942.1 s",
        "triggered the pre-declared reduced-cost permutation variant")

    # ---- S0 / S1 AURC and operating points -------------------------------
    aurc = pd.read_csv(SRC / "fig1_aurc_aggregation.csv")
    for _, r in aurc.iterrows():
        add("PRIMARY", f"s0.{r['aggregation']}_aurc.{r['method']}",
            float(r["aurc"]),
            "AURC — confidence-ranking quality under 0/1 loss; NOT the "
            "certification balanced-error risk", "AURC",
            r["aggregation_rule"],
            f'{ci(r["ci_low"], r["ci_high"], lambda v: f"{float(v):.4f}")} '
            f'(registered 95% site-cluster-bootstrap percentile, B=5000, '
            f'{int(r["bootstrap_replicates_valid"])} valid)',
            "results/rq1_s0/aurc_summary.csv", a_s0,
            f'{float(r["aurc"]):.3f}',
            f'denominator {r["aggregation_denominator_name"]} = '
            f'{r["aggregation_denominator_value"]:g}')
    for _, r in aurc[aurc.aggregation == "site_balanced"].iterrows():
        for stat in ("min", "median", "max"):
            add("PRIMARY", f"s0.per_site_aurc_{stat}.{r['method']}",
                float(r[f"{stat}_site_aurc"]),
                f"{stat} of the 23 per-site AURC_s values — the frozen "
                f"distribution both aggregates are built from", "AURC",
                "per held-out site", "no interval registered for this summary",
                "results/rq1_s0/aurc_summary.csv; results/rq1_s0/per_site_aurc.csv",
                a_s0, f'{float(r[f"{stat}_site_aurc"]):.3f}')

    for opname, f in (("OP50", "results/rq1_s0/op50_summary.csv"),
                      ("OP30", "results/rq1_s0/op30_summary.csv")):
        d = pd.read_csv(ROOT / f)
        for _, r in d.iterrows():
            for q, base in (("coverage", "actual_coverage"),
                            ("selective_ber", "selective_ber"),
                            ("descriptive_violation_rate", "descriptive_violation_rate")):
                col = f"{base}_site_balanced"
                add("PRIMARY", f"s0.{opname}.{q}.{r['method']}", float(r[col]),
                    {"coverage": "actual target acceptance rate",
                     "selective_ber": "selective balanced error on retained cases",
                     "descriptive_violation_rate":
                         "descriptive share of sites with BER_sel > r*_s "
                         "(NOT certification, NOT a formal violation rate)"}[q],
                    "proportion", "site-balanced over estimable sites",
                    f'{ci(r[base + "_ci_low"], r[base + "_ci_high"])} '
                    f'(95% site-cluster-bootstrap percentile, B=5000; '
                    f'n_estimable = {int(r[base + "_n_estimable_sites"])})',
                    f, a_s0, p3(r[col]))

    s1iv = pd.read_csv(ROOT / "results/rq1_s1/site_bootstrap_intervals.csv")
    for _, r in s1iv.iterrows():
        add("PRIMARY", f"s1.{r['quantity']}.{r.get('method_or_contrast', '')}",
            float(r["point_estimate"]) if pd.notna(r["point_estimate"]) else None,
            "S1 zero-label transductive (density-ratio weighted source calibration)",
            "proportion", "site-balanced over estimable sites",
            f'{ci(r["ci_low"], r["ci_high"])} '
            f'(95% site-cluster-bootstrap percentile, B=5000)',
            "results/rq1_s1/site_bootstrap_intervals.csv", a_s1,
            p3(r["point_estimate"]) if pd.notna(r["point_estimate"]) else "—")

    # ---- S2 -------------------------------------------------------------
    f2 = pd.read_csv(SRC / "fig2_certification_intervals.csv")
    for _, r in f2.iterrows():
        est = r["status"] == "ESTIMABLE"
        cls = "PRIMARY" if r["quantity"].startswith("formal") else "SECONDARY"
        if not est:
            cls = "NOT_ESTIMABLE"
        add(cls,
            f"s2.{r['quantity']}.{r['strategy']}"
            f"{'.N' + str(int(r['n_labels'])) if pd.notna(r['n_labels']) else ''}"
            f".{r['method']}",
            float(r["point_estimate"]) if est else "NOT_ESTIMABLE",
            f"{r['guarantee_tier']} tier, information regime {r['information_regime']}",
            "probability", "site-level, over eligible sites",
            (f"{ci(r['ci_low'], r['ci_high'])} (95% site-cluster-bootstrap "
             f"percentile, B=5000)") if est
            else f"NOT_ESTIMABLE (n_estimable_sites = {int(float(r['n_estimable_sites']))})",
            "results/rq2_s2/registered_intervals.csv", a_s2,
            p3(r["point_estimate"]) if est else "NOT_ESTIMABLE",
            "" if est else "zero adopted thresholds; a zero must never be drawn here")

    bur = pd.read_csv(SRC / "fig2_uncertified_burden.csv")
    for n in sorted(bur.n_labels.unique()):
        g = bur[bur.n_labels == n]
        add("PRIMARY", f"s2.expected_uncertified_site_burden.N{int(n)}",
            float(g.expected_uncertified_site_burden.iloc[0]),
            "expected number of uncertified sites (target_local, formal tier)",
            "site count", "site", "point section, no CI registered",
            "results/rq2_s2/formal_point_sections.json", a_s2,
            f"{int(g.expected_uncertified_site_burden.iloc[0])}"
            f"/{int(g.n_eligible_sites.iloc[0])}",
            "identical across the three confidence methods; N=5 is exploratory")

    # ---- RQ3 ------------------------------------------------------------
    t2 = pd.read_csv(SRC / "table2_rq3_attribution.csv")
    for _, r in t2.iterrows():
        est = r["status"] == "ESTIMABLE"
        add("SECONDARY" if est else "NOT_ESTIMABLE",
            f"rq3.{r['population']}.{r['block']}.{r['operating_point']}."
            f"{r['confidence_method']}.p_holm",
            r["p_holm"] if est else "NOT_ESTIMABLE",
            f"block LRT, Holm-corrected within family {r['holm_family']} "
            f"(m = {r['holm_family_size']}, alpha = {r['holm_alpha']}), df = {r['df']}",
            "p-value", f"{r['population']} complete-case subset "
                       f"(n = {r['n_complete_case']}, {r['n_sites']} sites)",
            "LRT + Holm is the registered uncertainty channel; no Wald SE or CI exists",
            f"results/rq3/families/{r['population']}__{r['block']}.json", a_r3,
            pval(float(r["p_holm"])) if est else "NOT_ESTIMABLE",
            "" if est else f"failure code {r['failure_code']}; missing evidence, not null evidence")

    # ---- Stage F: RQ4 secondary + SVM limitation -------------------------
    rq4 = pd.read_csv(SRC / "stage_f_rq4_site_macro.csv")
    for _, r in rq4.iterrows():
        nlab = "" if pd.isna(r["n_labels"]) else f".N{int(r['n_labels'])}"
        add("SECONDARY",
            f"rq4.site_macro_coverage.{r['method']}{nlab}.alpha{r['alpha']}",
            float(r["site_macro_coverage"]),
            "split-conformal site-macro coverage at nominal 1 - alpha",
            "proportion", "site-macro over 23 sites",
            "descriptive 95% site-cluster-bootstrap (hypothesis_test: false, "
            "tost: false, no multiplicity correction)",
            "results/stage_f_v2/rq4/site_macro_metrics.csv", a_sf,
            p3(r["site_macro_coverage"]),
            f"aggregate property only — NOT a site-conditional guarantee; "
            f"worst site {r['worst_site']} at {p3(r['worst_site_coverage'])}; "
            f"doubleton rate {p3(r['site_macro_doubleton_rate'])}")

    svm = pd.read_csv(SRC / "stage_f_svm_status.csv").iloc[0]
    add("LIMITATION", "svm.site_macro_bacc", "NOT_ESTIMABLE",
        "pre-specified linear-SVM robustness comparator", "balanced accuracy",
        "site-macro over outer LOSO folds",
        "none — no estimate exists",
        "results/stage_f_v2/svm/site_macro_bacc.json", a_sf, "NOT_ESTIMABLE",
        f"{int(svm.n_estimable_outer_folds)} ESTIMABLE / "
        f"{int(svm.n_not_estimable_outer_folds)} NOT_ESTIMABLE outer folds, "
        f"{int(svm.n_svm_not_converged)} SVM_NOT_CONVERGED. The historical v1 "
        f"BAcc is scientifically invalid for final reporting and is omitted.")

    # ---- Stage F sensitivities ------------------------------------------
    sens = [
        ("SENS-1", "overall-error Clopper-Pearson UCB estimand",
         "380 / 15522 = 0.0244814", "QUALIFIES",
         "results/stage_f_v2/sensitivities/sens1_overall_error_ucb/decisions.csv"),
        ("SENS-2", "q step 0.02 (41 candidates)", "0 / 15522", "CONFIRMS",
         "results/stage_f_v2/sensitivities/sens2_q_step_002/decisions.csv"),
        ("SENS-3", "r* ± 0.05", "0 / 15522 at both offsets", "CONFIRMS",
         "results/stage_f_v2/sensitivities/sens3_rstar_pm_005/decisions.csv"),
        ("SENS-4", "common-site subset (11 sites)", "0 / 9519", "CONFIRMS",
         "results/stage_f_v2/sensitivities/sens4_common_site/decisions.csv"),
        ("SENS-5", "S1 attribution", "10/24 ESTIMABLE, 4 Holm-supported "
                                     "(all full-sample motion)", "CONFIRMS (partial)",
         "results/stage_f_v2/sensitivities/sens5_s1_attribution/validation_units.csv"),
        ("SENS-6", "realized-coverage matching", "0/24 ESTIMABLE units",
         "DOES_NOT_ADDRESS",
         "results/stage_f_v2/sensitivities/sens6_realized_coverage_matched/validation_units.csv"),
    ]
    for name, factor, result, cls, art in sens:
        add("SENSITIVITY", f"stage_f.{name.lower().replace('-', '')}", result,
            f"one changed factor: {factor}", "target-local certifications / "
            "evaluable cells", "decision cell", "no CI registered",
            art, a_sf, result, f"classification: {cls}")

    return pd.DataFrame(rows)


# ==========================================================================
def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    print("Stage G — rendering tables and the final number ledger")

    full, compact = table1()
    (TAB / "table1_cohort_and_rq0.md").write_text(full)
    (TAB / "table1_compact.md").write_text(compact)
    print("  wrote table1_cohort_and_rq0.md / table1_compact.md")

    (TAB / "table2_rq3_attribution.md").write_text(table2())
    print("  wrote table2_rq3_attribution.md")

    led = ledger()
    led.to_csv(SG / "final_number_ledger.csv", index=False)
    print(f"  wrote final_number_ledger.csv ({len(led)} rows)")
    print(led.classification.value_counts().to_string())


if __name__ == "__main__":
    main()
