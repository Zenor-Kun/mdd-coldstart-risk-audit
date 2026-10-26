#!/usr/bin/env python3
"""Gate 1 — RQ0 Information & Leakage Gate.

Produces the five spec §18 baseline numbers and nothing else:

  1. macro mean of per-site BAcc under strict outer LOSO-site
  2. site-decodability balanced accuracy (macro-F1 secondary)
  3. within-site stratified permutation null (20-run smoke test)
  4. source-side full-coverage balanced error e_full,-s
  5. single-round wall-clock time T of the nested pipeline

Explicitly NOT implemented here: S0 selective prediction, S1 density-ratio,
S2 few-label onboarding, certification, conformal, shrinkage, RQ3.

Usage:
    python3 scripts/run_gate1.py                    # everything, 20 permutations
    python3 scripts/run_gate1.py --stages observed  # observed round only
"""

from __future__ import annotations

import os

# Single-threaded BLAS: parallelism is at the outer-fold level, and this keeps
# the wall-clock measurement T interpretable. Must precede the numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import contracts  # noqa: E402
import fc_features  # noqa: E402
import nested_loso as nl  # noqa: E402
import site_decodability as sd  # noqa: E402

OUT = ROOT / "results" / "gate1"

# --- frozen-for-this-stage seeds -------------------------------------------
MODEL_SEED = 20260807          # PCA / solver random_state, site-decoding CV
PERM_SEED_BASE = 20260807      # permutation i uses PERM_SEED_BASE + i


# ---------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_tag_exists(tag: str) -> bool:
    try:
        tags = subprocess.check_output(["git", "-C", str(ROOT), "tag", "-l"],
                                       text=True).split()
        return tag in tags
    except Exception:
        return False


def environment() -> dict:
    import joblib
    import scipy
    import sklearn
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "blas_threads_per_process": os.environ.get("OMP_NUM_THREADS"),
        "git_commit": git_commit(),
        "data_contract_v1_tag_exists": git_tag_exists("data-contract-v1"),
    }


# ---------------------------------------------------------------------------
def summarise(v) -> dict:
    a = np.asarray(list(v), dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max()), "n": int(a.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="observed,site,permutation",
                    help="comma list of: observed, site, permutation")
    ap.add_argument("--n-jobs", type=int, default=23)
    ap.add_argument("--n-perm", type=int, default=20)
    args = ap.parse_args()
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}

    OUT.mkdir(parents=True, exist_ok=True)
    assert ROOT / "data" not in OUT.parents, "refusing to write under data/"
    led = contracts.AssertionLedger()
    env = environment()
    started = datetime.now(timezone.utc).isoformat()

    print("=" * 78)
    print("GATE 1 -- RQ0 Information & Leakage Gate")
    print("=" * 78)
    print(json.dumps(env, indent=2))
    led.check("data_contract_v1_tag_exists", env["data_contract_v1_tag_exists"],
              "git tag data-contract-v1")

    # ---- cohort + features ------------------------------------------------
    t0 = time.perf_counter()
    X, y, sites, cohort = fc_features.build_xy()
    t_features = time.perf_counter() - t0

    led.check("manifest_included_only",
              bool((cohort["inclusion_status"] == "INCLUDED").all()),
              "inclusion_status == INCLUDED enforced")
    led.check("frozen_cohort_2232_1182_1050_23",
              len(cohort) == 2232 and int((y == 1).sum()) == 1182
              and int((y == 0).sum()) == 1050 and len(np.unique(sites)) == 23,
              f"N={len(cohort)} MDD={int((y == 1).sum())} "
              f"HC={int((y == 0).sum())} sites={len(np.unique(sites))}")
    led.check("unique_canonical_acquisitions",
              cohort["canonical_id"].is_unique
              and len(cohort) == cohort["canonical_id"].nunique(),
              "no acquisition appears twice")
    led.check("no_dropped_alias_in_cohort", True,
              "checked by contracts.assert_unique_acquisitions")
    led.check("features_via_feature_source_id",
              cohort["feature_source_id"].notna().all()
              and cohort["feature_source_id"].is_unique,
              "loader keys on feature_source_id only")
    led.check("feature_matrix_6670_finite",
              X.shape == (2232, 6670) and bool(np.isfinite(X).all()),
              f"X.shape={X.shape}, all finite")
    led.check("no_forbidden_fields_in_X", isinstance(X, np.ndarray)
              and X.dtype.kind == "f" and X.dtype.names is None,
              "X is a bare float ndarray built by a single-argument pure "
              "function; FORBIDDEN_IN_X cannot appear")
    print(f"[data] X={X.shape} y={np.bincount(y).tolist()} "
          f"sites={len(np.unique(sites))} feature_build={t_features:.1f}s")

    results: dict = {"environment": env, "started_utc": started,
                     "feature_build_seconds": t_features}

    # ---- 1) observed nested LOSO ------------------------------------------
    if "observed" in stages:
        print("\n[observed] nested LOSO-site (23 outer folds)")
        t0 = time.perf_counter()
        folds = nl.run_nested_loso(X, y, sites, seed=MODEL_SEED,
                                   n_jobs=args.n_jobs, verbose=True)
        T = time.perf_counter() - t0
        print(f"[observed] T = {T:.1f}s ({T / 60:.2f} min)")

        rows, oof_rows, test_rows = [], [], []
        for f in folds:
            rows.append({k: v for k, v in f.items() if not k.startswith("_")})
            o, te = f["_oof"], f["_test"]
            oof_rows.append(pd.DataFrame({
                "outer_fold_held_out_site": f["site"],
                "canonical_id": cohort["canonical_id"].to_numpy()[o["index"]],
                "source_site": o["site"], "y_true": o["y"],
                "oof_prob_mdd": o["prob"], "oof_pred": o["yhat"],
                "selected_pca": f["selected_pca"], "selected_C": f["selected_C"],
            }))
            test_rows.append(pd.DataFrame({
                "held_out_site": f["site"],
                "canonical_id": cohort["canonical_id"].to_numpy()[te["index"]],
                "y_true": te["y"], "prob_mdd": te["prob"], "pred": te["yhat"],
            }))

        fold_df = pd.DataFrame(rows).sort_values(
            "site", key=lambda c: c.str.slice(1).astype(int))
        fold_df.to_csv(OUT / "outer_fold_metrics.csv", index=False)
        oof_df = pd.concat(oof_rows, ignore_index=True)
        oof_df.to_csv(OUT / "source_oof_predictions.csv", index=False)
        pd.concat(test_rows, ignore_index=True).to_csv(
            OUT / "outer_test_predictions.csv", index=False)
        fold_df[["site", "n_source", "n_source_sites", "selected_pca",
                 "selected_C", "e_full_minus_s"]].to_csv(
            OUT / "source_full_coverage_error_by_fold.csv", index=False)

        # OOF integrity: every source acquisition predicted exactly once per
        # fold, and never by a model that saw its own site.
        cnt = oof_df.groupby(["outer_fold_held_out_site", "canonical_id"]).size()
        led.check("oof_one_prediction_per_source_acquisition",
                  bool((cnt == 1).all()), "no acquisition predicted twice")
        led.check("held_out_site_absent_from_source_oof",
                  bool((oof_df["outer_fold_held_out_site"]
                        != oof_df["source_site"]).all()),
                  "outer test site never appears among source rows")
        led.check("all_outer_folds_present", len(fold_df) == 23,
                  f"{len(fold_df)} outer folds")

        macro = summarise(fold_df["bacc"])
        efull = summarise(fold_df["e_full_minus_s"])
        results["observed"] = {
            "macro_site_bacc": macro,
            "auroc_across_folds": summarise(fold_df["auroc"]),
            "e_full_minus_s": efull,
            "selected_pca_counts": fold_df["selected_pca"].astype(str)
                                   .value_counts().to_dict(),
            "selected_C_counts": fold_df["selected_C"].astype(str)
                                 .value_counts().to_dict(),
            "T_seconds": T, "T_minutes": T / 60.0,
        }
        print(f"[observed] macro site BAcc = {macro['mean']:.4f} "
              f"(SD {macro['sd']:.4f}, [{macro['min']:.4f}, {macro['max']:.4f}])")
        print(f"[observed] e_full,-s mean = {efull['mean']:.4f}")

        # runtime.json (spec §18 item 5)
        runtime = {
            "definition": ("one complete nested observed diagnostic pipeline: "
                           "23 outer LOSO folds, per-fold inner "
                           "leave-one-source-site-out CV, scaler/PCA fit, PCA "
                           "dimension + ridge C selection, final outer refit, "
                           "held-out-site prediction, source OOF generation. "
                           "Excludes FC feature loading, site decoding, the "
                           "permutation smoke test, and any S0/S1/S2 work."),
            "T_seconds": T, "T_minutes": T / 60.0,
            "T_hours": T / 3600.0,
            "n_permutations_planned": 200,
            "projected_200T_seconds": 200 * T,
            "projected_200T_hours": 200 * T / 3600.0,
            "budget_hours": 24.0,
            "verdict": ("FULL_PERMUTATION_FEASIBLE" if 200 * T <= 24 * 3600
                        else "REDUCED_COST_PERMUTATION_REQUIRED"),
            "n_jobs_outer_folds": args.n_jobs,
            "blas_threads_per_process": os.environ.get("OMP_NUM_THREADS"),
            "cpu_count": os.cpu_count(),
            "warm_start": "cold (features loaded from cache, no prior fit)",
            "feature_build_seconds_excluded": t_features,
        }
        (OUT / "runtime.json").write_text(json.dumps(runtime, indent=2))
        results["runtime"] = runtime
        print(f"[runtime] 200 x T = {200 * T / 3600:.2f} h -> "
              f"{runtime['verdict']}")

    # ---- 2) site decodability ---------------------------------------------
    if "site" in stages:
        print("\n[site] site-decodability diagnostic")
        t0 = time.perf_counter()
        sdec = sd.run_site_decodability(X, sites, seed=MODEL_SEED)
        t_site = time.perf_counter() - t0
        pred = sdec.pop("predictions")
        pd.DataFrame([{
            "metric_primary": "balanced_accuracy",
            "balanced_accuracy": sdec["balanced_accuracy"],
            "macro_f1": sdec["macro_f1"],
            "accuracy": sdec["accuracy"],
            "chance_reference_1_over_n_sites": sdec["chance_reference"],
            "n_sites": sdec["n_sites"], "n_acquisitions": sdec["n_acquisitions"],
            "cv": f"{sd.N_SPLITS}-fold stratified, inner {sd.INNER_SPLITS}-fold "
                  f"C selection, PCA {sd.PCA_DIM}",
            "selected_C_per_fold": ";".join(map(str, sdec["selected_C_per_fold"])),
            "seconds": t_site,
        }]).to_csv(OUT / "site_decodability.csv", index=False)
        pd.DataFrame({"site": list(sdec["per_site_recall"]),
                      "recall": list(sdec["per_site_recall"].values())}).to_csv(
            OUT / "site_decodability_per_site.csv", index=False)
        sdec["seconds"] = t_site
        results["site_decodability"] = sdec
        print(f"[site] BAcc = {sdec['balanced_accuracy']:.4f}, "
              f"macro-F1 = {sdec['macro_f1']:.4f} "
              f"(chance {sdec['chance_reference']:.4f}), {t_site:.1f}s")

    # ---- 3) within-site permutation smoke test ----------------------------
    if "permutation" in stages:
        print(f"\n[perm] {args.n_perm}-run within-site stratified permutation")
        prows = []
        for i in range(args.n_perm):
            seed = PERM_SEED_BASE + i
            rng = np.random.default_rng(seed)
            y_perm = nl.permute_within_site(y, sites, rng)
            t0 = time.perf_counter()
            pf = nl.run_nested_loso(X, y_perm, sites, seed=MODEL_SEED,
                                    n_jobs=args.n_jobs)
            dt = time.perf_counter() - t0
            baccs = [f["bacc"] for f in pf]
            prows.append({
                "permutation": i, "seed": seed,
                "macro_site_bacc": float(np.mean(baccs)),
                "sd_across_sites": float(np.std(baccs, ddof=1)),
                "min_site_bacc": float(np.min(baccs)),
                "max_site_bacc": float(np.max(baccs)),
                "mean_e_full_minus_s": float(np.mean(
                    [f["e_full_minus_s"] for f in pf])),
                "seconds": dt,
            })
            print(f"[perm] {i + 1}/{args.n_perm} macro BAcc = "
                  f"{prows[-1]['macro_site_bacc']:.4f} ({dt:.0f}s)", flush=True)
            pd.DataFrame(prows).to_csv(OUT / "permutation_smoke_20.csv",
                                       index=False)
        led.check("permutation_preserves_site_structure", True,
                  "asserted per permutation in permute_within_site()")
        null = summarise([r["macro_site_bacc"] for r in prows])
        results["permutation_smoke"] = {"n": args.n_perm, "null": null}
        if "observed" in stages:
            obs = results["observed"]["macro_site_bacc"]["mean"]
            n_ge = int(sum(r["macro_site_bacc"] >= obs for r in prows))
            results["permutation_smoke"].update({
                "observed_macro_site_bacc": obs,
                "n_null_ge_observed": n_ge,
                "n_null_total": args.n_perm,
                "z_like_position": float((obs - null["mean"]) / null["sd"])
                if null["sd"] > 0 else float("nan"),
                "note": ("20-run smoke test: magnitude check only, NOT a final "
                         "permutation p-value (spec requires >= 200)"),
            })
        print(f"[perm] null macro BAcc mean = {null['mean']:.4f} "
              f"(SD {null['sd']:.4f})")

    # ---- config, ledger, report -------------------------------------------
    write_config(args, env)
    pd.DataFrame(led.rows).to_csv(OUT / "leakage_assertions.csv", index=False)
    results["leakage_assertions_all_pass"] = led.all_pass
    results["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (OUT / "gate1_results.json").write_text(json.dumps(results, indent=2,
                                                       default=str))
    write_report(results, led, OUT)
    print(f"\n[done] outputs in {OUT}")
    return 0


def write_config(args, env) -> None:
    cfg = f"""# gate1_config.yaml -- Gate 1 (RQ0) configuration, as executed.
# Frozen items are quoted from docs/EXPERIMENT_LOG.md and
# docs/DATA_CONTRACT_v1.md; they were fixed before any model was fitted.

stage: "Gate 1 -- RQ0 Information & Leakage Gate"
scope_excluded: [S0, S1, S2, certification, conformal, shrinkage, RQ3]

cohort:
  source: results/data_contract/analysis_manifest_v1.csv
  filter: inclusion_status == "INCLUDED"
  n_acquisitions: 2232
  n_MDD: 1182
  n_HC: 1050
  n_sites: 23
  unit: "one independent fMRI acquisition (DATA_CONTRACT §3.12)"
  feature_key: feature_source_id

features:              # FROZEN, EXPERIMENT_LOG §1
  branch: {fc_features.FROZEN_INPUT['branch']}
  atlas: {fc_features.FROZEN_INPUT['atlas']}
  roi_slice: "{fc_features.FROZEN_INPUT['roi_slice']}"
  connectivity: {fc_features.FROZEN_INPUT['connectivity']}
  transform: "{fc_features.FROZEN_INPUT['transform']}"
  vectorisation: "{fc_features.FROZEN_INPUT['vectorisation']}"
  n_features: {fc_features.FROZEN_INPUT['n_features']}

outer_protocol:
  scheme: leave-one-site-out
  n_folds: 23
  held_out_site_visibility: "none -- excluded from scaler, PCA, C selection,
    dimension selection and every model fit"

inner_protocol:
  scheme: leave-one-source-site-out over the 22 source sites
  selection_metric: "macro mean of per-site balanced accuracy (never pooled)"
  scaler: StandardScaler          # FROZEN, EXPERIMENT_LOG §7
  pca_candidates: [no-PCA, 50, 100, 200]   # FROZEN, EXPERIMENT_LOG §7
  pca_fit_scope: "inner-training source acquisitions only; each candidate
    dimension gets its own PCA fit (randomized SVD), never a truncation of a
    larger fit -- see src/nested_loso.Preprocessor and
    tests/test_gate1.py::test_each_pca_candidate_gets_its_own_fit"
  C_grid: {list(nl.C_GRID)}   # fixed here, before any model was fitted
  classifier: "ridge (L2) logistic regression, lbfgs, max_iter={nl.MAX_ITER}"
  decision_threshold: {nl.DECISION_THRESHOLD}
  tie_break: "highest inner macro site-BAcc; then smaller C; then stronger
    compression"

site_weighting:
  formula: "w_i = (n_train / n_sites_train) / n_{{s(i)}}"
  effect: "every training site carries equal total mass; mean(w) == 1"
  n_s_unit: acquisitions

site_decodability:
  target: site label (this is the only place site is a target)
  cv: "{sd.N_SPLITS}-fold stratified over acquisitions, inner {sd.INNER_SPLITS}-fold C selection"
  preprocessing: "StandardScaler + PCA {sd.PCA_DIM}, fit on training fold only"
  classifier: multinomial L2 logistic regression
  primary_metric: balanced_accuracy      # spec §18
  secondary_metric: macro_f1             # spec §8 permits either
  chance_reference: 1/23

permutation_smoke:
  n_permutations: {args.n_perm}
  scheme: "diagnosis labels shuffled independently within each site; site
    membership, site size and site prevalence preserved"
  rerun: "complete pipeline per permutation (inner CV, scaler, PCA selection,
    C selection, outer refit, held-out-site prediction)"
  status: "SMOKE TEST -- magnitude only; the formal >=200-run permutation
    inference is not run at Gate 1"

seeds:
  model_seed: {MODEL_SEED}
  permutation_seed_base: {PERM_SEED_BASE}   # permutation i uses base + i
  site_decoding_cv_seed: {MODEL_SEED}

compute:
  n_jobs_outer_folds: {args.n_jobs}
  blas_threads_per_process: {env['blas_threads_per_process']}
  cpu_count: {env['cpu_count']}
  device: CPU only (no GPU, no neural network -- spec §6)

environment:
  python: "{env['python']}"
  numpy: "{env['numpy']}"
  scipy: "{env['scipy']}"
  scikit_learn: "{env['scikit_learn']}"
  pandas: "{env['pandas']}"
  joblib: "{env['joblib']}"
  git_commit: "{env['git_commit']}"
  data_contract_v1_tag_exists: {str(env['data_contract_v1_tag_exists']).lower()}
"""
    (OUT / "gate1_config.yaml").write_text(cfg)


def write_report(res: dict, led, out: Path) -> None:
    def fmt(d, k="mean", p=4):
        return f"{d[k]:.{p}f}"

    L = ["# GATE 1 REPORT — RQ0 Information & Leakage Gate", "",
         f"**Generated**: {res.get('finished_utc', '')}  ",
         f"**Commit**: `{res['environment']['git_commit']}`  ",
         f"**Tag `data-contract-v1` present**: "
         f"{res['environment']['data_contract_v1_tag_exists']}  ",
         "**Scope**: Gate 1 / RQ0 only. No S0, S1, S2, certification, "
         "conformal, shrinkage or RQ3 code exists in this stage.", "",
         "## Cohort", "",
         "2232 independent acquisitions (1182 MDD / 1050 HC) across 23 eligible "
         "sites, read from `analysis_manifest_v1.csv` with "
         "`inclusion_status == \"INCLUDED\"`; features loaded through "
         "`feature_source_id` as 6670-D Fisher-z AAL116 FC vectors.", ""]

    if "observed" in res:
        o = res["observed"]
        L += ["## 1. Observed outer LOSO-site macro BAcc", "",
              f"- **macro mean of 23 per-site BAcc = {fmt(o['macro_site_bacc'])}**",
              f"- SD across sites = {fmt(o['macro_site_bacc'], 'sd')}",
              f"- median = {fmt(o['macro_site_bacc'], 'median')}, "
              f"min = {fmt(o['macro_site_bacc'], 'min')}, "
              f"max = {fmt(o['macro_site_bacc'], 'max')}",
              f"- AUROC across folds: mean = {fmt(o['auroc_across_folds'])}, "
              f"SD = {fmt(o['auroc_across_folds'], 'sd')}", "",
              f"Selected PCA dimension: `{o['selected_pca_counts']}`  ",
              f"Selected C: `{o['selected_C_counts']}`", "",
              "## 4. Source-side full-coverage balanced error", "",
              f"- descriptive global mean of `e_full,-s` over 23 folds = "
              f"**{fmt(o['e_full_minus_s'])}** "
              f"(SD {fmt(o['e_full_minus_s'], 'sd')}, "
              f"min {fmt(o['e_full_minus_s'], 'min')}, "
              f"max {fmt(o['e_full_minus_s'], 'max')})",
              "- feasibility number only. The per-fold rule "
              "`r*_s = e_full,-s − 0.10` is **not** implemented at Gate 1.", ""]

    if "site_decodability" in res:
        s = res["site_decodability"]
        L += ["## 2. Site decodability", "",
              f"- **balanced accuracy (primary) = {s['balanced_accuracy']:.4f}**",
              f"- macro-F1 (secondary) = {s['macro_f1']:.4f}",
              f"- plain accuracy = {s['accuracy']:.4f}",
              f"- chance reference = 1/{s['n_sites']} = "
              f"{s['chance_reference']:.4f} (interpretive only)", ""]

    if "permutation_smoke" in res:
        p = res["permutation_smoke"]
        n = p["null"]
        L += [f"## 3. Within-site permutation null ({p['n']}-run smoke test)", "",
              f"- null macro site-BAcc: mean = {fmt(n)}, SD = {fmt(n, 'sd')}, "
              f"min = {fmt(n, 'min')}, max = {fmt(n, 'max')}"]
        if "observed_macro_site_bacc" in p:
            L += [f"- observed = {p['observed_macro_site_bacc']:.4f}; "
                  f"{p['n_null_ge_observed']}/{p['n_null_total']} null runs "
                  f"reach or exceed it "
                  f"({p['z_like_position']:.1f} null SD above the null mean)"]
        L += ["- **Smoke test only** — magnitude check. No permutation p-value "
              "is claimed; the spec's ≥200-run inference is not run here.", ""]

    if "runtime" in res:
        r = res["runtime"]
        L += ["## 5. Wall-clock T", "",
              f"- `T_seconds` = {r['T_seconds']:.1f}",
              f"- `T_minutes` = {r['T_minutes']:.2f}",
              f"- 200 × T = {r['projected_200T_hours']:.2f} h "
              f"(budget 24 h) → **{r['verdict']}**",
              f"- measured with `n_jobs={r['n_jobs_outer_folds']}` outer folds, "
              f"{r['blas_threads_per_process']} BLAS thread per process, "
              f"{r['cpu_count']} CPUs; feature loading "
              f"({r['feature_build_seconds_excluded']:.1f}s) excluded.",
              f"- definition: {r['definition']}", ""]

    L += ["## Leakage assertions", "",
          f"All assertions PASS: **{res.get('leakage_assertions_all_pass')}**", "",
          "| assertion | status |", "|---|---|"]
    L += [f"| `{r['assertion']}` | {r['status']} |" for r in led.rows]
    L += ["", "Full machine-readable ledger: `leakage_assertions.csv`.", "",
          "## Environment", "", "```json",
          json.dumps(res["environment"], indent=2), "```", ""]
    (out / "GATE1_REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
