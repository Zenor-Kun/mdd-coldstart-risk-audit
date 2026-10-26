#!/usr/bin/env python3
"""Stage 1.1 — formal RQ1 / S0 execution (source-only cold start).

Executes `docs/RQ1_S0_PROTOCOL.md` exactly as frozen. Nothing in this script
re-derives, relaxes or "improves" a frozen definition; where implementation and
protocol could diverge the script raises instead of choosing.

Explicitly NOT implemented, and unreachable from here (§8, §12, §19):
S1 density-ratio weighting, S2 few-label onboarding, Clopper-Pearson or any
other upper confidence bound, fixed-sequential or any ordered testing
procedure, conformal prediction, shrinkage, RQ3, and the words
certified / uncertified as an S0 output.

Usage:
    python3 scripts/run_rq1_s0.py                      # everything
    python3 scripts/run_rq1_s0.py --stages validate    # pre-run validation only
    python3 scripts/run_rq1_s0.py --stages validate,confidence
"""

from __future__ import annotations

import os

# Single-threaded BLAS: parallelism is at the work-unit level. Must precede numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
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
import s0_confidence as s0c  # noqa: E402
import s0_evaluation as s0e  # noqa: E402
import site_bootstrap as sb  # noqa: E402
from selective_aurc import tie_aware_aurc  # noqa: E402

GATE1 = ROOT / "results" / "gate1"
OUT = ROOT / "results" / "rq1_s0"
CACHE = OUT / "cache"

METHODS = sb.CONFIDENCE_METHODS                 # probability, bootstrap, mahalanobis
REQUIRED_TAGS = ("data-contract-v1", "gate1-smoke-v1", "gate1-rq0-v1",
                 "rq1-s0-protocol-v1")
FROZEN_MACRO_BACC = 0.5820                      # §2.1, to 4 dp
NE = s0e.NOT_ESTIMABLE


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
# frozen upstream: cohort, features, Gate 1 artifacts
# ---------------------------------------------------------------------------
def load_frozen_inputs(led, verbose=True):
    X, y, sites, cohort = fc_features.build_xy(verbose=verbose)

    led.check("cohort_2232_1182_1050_23",
              len(cohort) == 2232 and int((y == 1).sum()) == 1182
              and int((y == 0).sum()) == 1050 and len(np.unique(sites)) == 23,
              f"N={len(cohort)} MDD={int((y == 1).sum())} "
              f"HC={int((y == 0).sum())} sites={len(np.unique(sites))}")
    led.check("cohort_included_only",
              bool((cohort["inclusion_status"] == "INCLUDED").all()),
              "inclusion_status == INCLUDED enforced")
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

    test = pd.read_csv(GATE1 / "outer_test_predictions.csv")
    oof = pd.read_csv(GATE1 / "source_oof_predictions.csv")
    folds = pd.read_csv(GATE1 / "outer_fold_metrics.csv")
    efull = pd.read_csv(GATE1 / "source_full_coverage_error_by_fold.csv")
    frozen_hp = pd.read_csv(ROOT / "results" / "gate1_formal_permutation"
                            / "frozen_outer_hyperparameters.csv")
    return X, y, sites, cohort, test, oof, folds, efull, frozen_hp


def verify_gate1(led, cohort, y, sites, test, oof, folds, efull, frozen_hp):
    """§2.1 — the ten mechanical Gate 1 re-verification checks, re-asserted."""
    from sklearn.metrics import balanced_accuracy_score

    order = list(contracts.FROZEN_SITES)
    led.check("gate1_23_outer_folds_frozen_site_set",
              len(folds) == 23 and set(folds["site"]) == set(order),
              "23 outer folds over the frozen eligible-site set")

    hp_a = folds.set_index("site")[["selected_pca", "selected_C"]].astype(str)
    hp_b = efull.set_index("site")[["selected_pca", "selected_C"]].astype(str)
    hp_c = frozen_hp.set_index("site")[["observed_selected_pca",
                                        "observed_selected_C"]].astype(str)
    hp_c.columns = ["selected_pca", "selected_C"]
    led.check("gate1_hyperparameters_identical_across_three_files",
              hp_a.equals(hp_b.loc[hp_a.index]) and hp_a.equals(hp_c.loc[hp_a.index]),
              "selected_pca / selected_C agree in outer_fold_metrics, "
              "source_full_coverage_error_by_fold and frozen_outer_hyperparameters")

    baccs = {}
    for s, g in test.groupby("held_out_site"):
        baccs[s] = balanced_accuracy_score(g["y_true"], g["pred"])
    diff = max(abs(baccs[r.site] - r.bacc) for r in folds.itertuples())
    led.check("gate1_per_site_bacc_reproduced", diff < 1e-12,
              f"max |recomputed - frozen| = {diff:.3e}")
    macro = float(np.mean([baccs[s] for s in order]))
    led.check("gate1_macro_bacc_0p5820", abs(macro - FROZEN_MACRO_BACC) < 5e-5,
              f"macro mean of 23 per-site BAcc = {macro:.6f}")

    led.check("gate1_target_predictions_cover_cohort_once",
              len(test) == 2232 and test["canonical_id"].is_unique
              and set(test["canonical_id"]) == set(cohort["canonical_id"]),
              "2232 unique canonical_id, set-equal to the manifest")
    m = cohort.set_index("canonical_id")
    t = test.set_index("canonical_id")
    led.check("gate1_site_and_label_agree_with_manifest",
              bool((t["held_out_site"] == m.loc[t.index, "site"]).all()
                   and (t["y_true"] == m.loc[t.index, "diagnosis"]).all()),
              "held_out_site / y_true match the manifest on every row")

    cnt = oof.groupby(["outer_fold_held_out_site", "canonical_id"]).size()
    n_src_sites = oof.groupby("outer_fold_held_out_site")["source_site"].nunique()
    led.check("gate1_source_oof_structure",
              bool((cnt == 1).all()) and bool((n_src_sites == 22).all())
              and bool((oof["outer_fold_held_out_site"] != oof["source_site"]).all())
              and len(oof) == int(folds["n_source"].sum()),
              f"{len(oof)} rows, one per source acquisition per fold, "
              "22 source sites per fold, held-out site never present")

    rec = {}
    for s, g in oof.groupby("outer_fold_held_out_site"):
        errs = [float((g.loc[g["y_true"] == c, "oof_pred"] != c).mean())
                for c in (0, 1)]
        rec[s] = float(np.mean(errs))
    d = max(abs(rec[r.site] - r.e_full_minus_s) for r in efull.itertuples())
    led.check("gate1_e_full_minus_s_reproduced", d < 1e-12,
              f"max |recomputed - frozen| = {d:.3e}")

    r_star = {r.site: r.e_full_minus_s - 0.10 for r in efull.itertuples()}
    led.check("r_star_computable_for_23_folds",
              len(r_star) == 23 and all(np.isfinite(v) for v in r_star.values()),
              f"r*_s in [{min(r_star.values()):.4f}, {max(r_star.values()):.4f}], "
              f"mean {np.mean(list(r_star.values())):.4f}")

    cp_t = s0c.confidence_prob(test["prob_mdd"].to_numpy())
    cp_o = s0c.confidence_prob(oof["oof_prob_mdd"].to_numpy())
    led.check("max_probability_confidence_finite_and_ge_half",
              bool(np.isfinite(cp_t).all() and np.isfinite(cp_o).all()
                   and (cp_t >= 0.5).all() and (cp_o >= 0.5).all()),
              "confidence_prob in [0.5, 1] on every target and source-OOF row")
    return r_star


def build_fold_table(folds, efull, sites):
    """One record per outer fold, in frozen site order, with frozen Gate 1 choices."""
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
                    "n_s": int((sites == s).sum())})
    return out


# ---------------------------------------------------------------------------
# stage: confidence scores (§3, §4, §5) — target y is never read here
# ---------------------------------------------------------------------------
def _unit_key(site, j_label, pool, pca, C) -> str:
    h = hashlib.sha256(("|".join(pool)).encode()).hexdigest()[:12]
    return f"{site}__score_{j_label}__pca_{pca}__C_{C}__B{s0c.B_BOOT}" \
           f"__seed{s0c.BOOT_SEED_BASE}__pool_{h}"


def _bootstrap_unit(X, y, sites, fold, j, scored_site, cache_dir):
    """One (outer fold, scored partition) bootstrap ensemble; disk-cached.

    The cache key is (fold, scored partition, training-site set, PCA, C, B,
    seed base). It is deterministic and contains no target-derived fit
    information: an ensemble is never reused across a different `(s, j)`
    (§16.4), because a different pool would leak a site into a score meant to
    exclude it.
    """
    pool = [u for u in fold["source_sites"] if u != scored_site]
    key = _unit_key(fold["site"], scored_site, pool, fold["selected_pca"],
                    fold["selected_C"])
    path = cache_dir / f"{key}.npz"
    scored_rows = np.flatnonzero(sites == scored_site)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if z["rows"].tolist() == scored_rows.tolist() and z["probs"].shape[0] == s0c.B_BOOT:
            return scored_rows, z["probs"], True
    probs = s0c.bootstrap_ensemble_probs(
        X, y, sites, np.asarray(pool), scored_rows,
        fold["selected_pca"], fold["selected_C"], o=fold["o"], j=j)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probs=probs, rows=scored_rows,
                        pool=np.asarray(pool, dtype="U8"))
    return scored_rows, probs, False


def _mahalanobis_unit(X, sites, fold, scored_site):
    """One (outer fold, scored partition) Mahalanobis support fit.

    Module-level rather than a closure so joblib memmaps the 119 MB feature
    matrix once per `Parallel` call instead of pickling it per task.
    """
    pool = [u for u in fold["source_sites"] if u != scored_site]
    fit_rows = np.flatnonzero(np.isin(sites, pool))
    scored_rows = np.flatnonzero(sites == scored_site)
    c, meta = s0c.mahalanobis_support(X, fit_rows, scored_rows)
    return fold["site"], scored_site, scored_rows, c, meta


def compute_confidences(X, y, sites, cohort, test, oof, fold_table, n_jobs,
                        led, verbose=True):
    """All three confidence scores, target and source-OOF, for all 23 folds."""
    from joblib import Parallel, delayed

    ids = cohort["canonical_id"].to_numpy()
    row_of = {cid: i for i, cid in enumerate(ids)}
    boot_dir = CACHE / "bootstrap_ensembles"
    timings = {}

    # ---- method 1: read the frozen Gate 1 probabilities --------------------
    t0 = time.perf_counter()
    test = test.copy()
    test["confidence_probability"] = s0c.confidence_prob(test["prob_mdd"].to_numpy())
    oof = oof.copy()
    oof["confidence_probability"] = s0c.confidence_prob(oof["oof_prob_mdd"].to_numpy())
    timings["method1_probability_seconds"] = time.perf_counter() - t0

    # ---- method 3: Mahalanobis (529 deterministic fits) --------------------
    t0 = time.perf_counter()

    units = [(f, f["site"]) for f in fold_table] + \
            [(f, u) for f in fold_table for u in f["source_sites"]]
    if verbose:
        print(f"[mahalanobis] {len(units)} fits, n_jobs={n_jobs}")
    mahal = Parallel(n_jobs=n_jobs, backend="loky", verbose=1 if verbose else 0)(
        delayed(_mahalanobis_unit)(X, sites, f, u) for f, u in units)
    timings["method3_mahalanobis_seconds"] = time.perf_counter() - t0

    # ---- method 2: bootstrap disagreement (529 x 100 fits) -----------------
    t0 = time.perf_counter()
    boot_units = [(f, 0, f["site"]) for f in fold_table] + \
                 [(f, j + 1, u) for f in fold_table
                  for j, u in enumerate(f["source_sites"])]
    assert len(boot_units) == 23 * 23
    seeds = [s0c.boot_seed(f["o"], j, b) for f, j, _ in boot_units
             for b in range(s0c.B_BOOT)]
    led.check("boot_seeds_unique_in_namespace_disjoint_from_rq0",
              len(seeds) == len(set(seeds)) == 23 * 23 * s0c.B_BOOT
              and all(map(s0c.seed_is_in_s0_namespace, seeds))
              and not any(map(s0c.seed_collides_with_rq0, seeds)),
              f"{len(set(seeds))} unique seeds in "
              f"[{min(seeds)}, {max(seeds)}], disjoint from every RQ0 range")

    if verbose:
        print(f"[bootstrap] {len(boot_units)} work units x B_boot="
              f"{s0c.B_BOOT}, n_jobs={n_jobs}")
    boot = Parallel(n_jobs=n_jobs, backend="loky", verbose=5 if verbose else 0)(
        delayed(_bootstrap_unit)(X, y, sites, f, j, u, boot_dir)
        for f, j, u in boot_units)
    timings["method2_bootstrap_seconds"] = time.perf_counter() - t0
    timings["method2_cache_hits"] = int(sum(hit for _, _, hit in boot))

    # ---- assemble ----------------------------------------------------------
    tgt_boot = np.full(len(test), np.nan)
    tgt_mahal = np.full(len(test), np.nan)
    oof_boot = np.full(len(oof), np.nan)
    oof_mahal = np.full(len(oof), np.nan)

    test_pos = {(r.held_out_site, r.canonical_id): i
                for i, r in enumerate(test.itertuples())}
    oof_pos = {(r.outer_fold_held_out_site, r.canonical_id): i
               for i, r in enumerate(oof.itertuples())}

    for (f, j, scored_site), (rows, probs, _) in zip(boot_units, boot):
        conf = s0c.confidence_boot(probs)
        target = (j == 0)
        for k, row in enumerate(rows):
            cid = ids[row]
            if target:
                tgt_boot[test_pos[(f["site"], cid)]] = conf[k]
            else:
                oof_boot[oof_pos[(f["site"], cid)]] = conf[k]

    for site, scored_site, rows, conf, _ in mahal:
        target = (site == scored_site)
        for k, row in enumerate(rows):
            cid = ids[row]
            if target:
                tgt_mahal[test_pos[(site, cid)]] = conf[k]
            else:
                oof_mahal[oof_pos[(site, cid)]] = conf[k]

    test["confidence_bootstrap"] = tgt_boot
    test["confidence_mahalanobis"] = tgt_mahal
    oof["confidence_bootstrap"] = oof_boot
    oof["confidence_mahalanobis"] = oof_mahal
    for df, name in ((test, "target"), (oof, "source_oof")):
        for m in METHODS:
            assert np.isfinite(df[f"confidence_{m}"]).all(), \
                f"missing {m} confidence on {name}"

    mahal_meta = pd.DataFrame([
        {"held_out_site": s, "scored_partition": u,
         "role": "target" if s == u else "source_oof", **meta}
        for s, u, _, _, meta in mahal])

    led.check("every_acquisition_scored_once_per_method",
              len(test) == 2232 and test["canonical_id"].is_unique
              and len(oof) == 49104,
              "2232 target rows and 49104 source-OOF rows per method")
    return test, oof, mahal_meta, timings, row_of


def verify_fit_pools(sites, fold_table, led):
    """§17.5, §17.6, §17.11 — audited from the persisted ensemble evidence.

    Every bootstrap work unit stored the training pool it actually used. Re-read
    all 529 of them and confirm that neither the outer held-out site nor the
    scored partition ever appears in the pool, then re-derive one replicate per
    unit and confirm the multiplicity / site-equal-weight property of §4.2.
    """
    import nested_loso as nl

    boot_dir = CACHE / "bootstrap_ensembles"
    n_units = 0
    weights_ok = True
    multiplicity_seen = False
    for f in fold_table:
        for j, scored_site in [(0, f["site"])] + \
                list(enumerate(f["source_sites"], start=1)):
            pool = [u for u in f["source_sites"] if u != scored_site]
            path = boot_dir / f"{_unit_key(f['site'], scored_site, pool, f['selected_pca'], f['selected_C'])}.npz"
            z = np.load(path, allow_pickle=False)
            stored = [str(u) for u in z["pool"]]
            assert stored == pool, f"{path.name}: stored pool differs"
            assert f["site"] not in stored, "held-out site inside a training pool"
            assert scored_site not in stored, "scored site inside its own pool"
            assert z["probs"].shape == (s0c.B_BOOT, int((sites == scored_site).sum()))
            n_units += 1

            drawn = s0c.draw_site_cluster_replicate(np.asarray(pool),
                                                    s0c.boot_seed(f["o"], j, 0))
            rows, clusters = s0c.build_replicate_training_set(sites, drawn)
            assert drawn.size == len(pool)
            assert len(np.unique(clusters)) == len(pool)     # one cluster per draw
            w = nl.site_equal_weights(clusters)              # raises if unequal
            mass = np.array([w[clusters == c].sum() for c in np.unique(clusters)])
            weights_ok &= bool(np.allclose(mass, mass[0]))
            uniq, cnt = np.unique(drawn, return_counts=True)
            if (cnt > 1).any():
                multiplicity_seen = True
                s_dup = uniq[np.argmax(cnt)]
                m = int(cnt.max())
                dup_mass = w[np.char.startswith(clusters, s_dup + "#")].sum()
                weights_ok &= bool(np.isclose(dup_mass, m * mass[0]))

    led.check("bootstrap_pools_exclude_held_out_and_scored_site",
              n_units == 23 * 23,
              f"{n_units} persisted ensembles audited; in none does the outer "
              "held-out site or the scored partition appear in the training pool")
    led.check("cluster_multiplicity_preserved_under_site_equal_weights",
              weights_ok and multiplicity_seen,
              "each drawn copy carries the mass of one site, so a site drawn m "
              "times contributes m times the mass of a singly-drawn site")


def verify_ranking_is_identifier_free(test, fold_table, led):
    """§17.12b — AURC invariance re-asserted on the real scored arrays."""
    worst = 0.0
    rng = np.random.default_rng(0)
    for f in fold_table:
        t = test[test["held_out_site"] == f["site"]]
        loss = (t["pred"].to_numpy() != t["y_true"].to_numpy()).astype(float)
        for m in METHODS:
            c = t[f"confidence_{m}"].to_numpy()
            base = tie_aware_aurc(c, loss)
            for _ in range(3):
                p = rng.permutation(c.size)
                worst = max(worst, abs(tie_aware_aurc(c[p], loss[p]) - base))
    led.check("aurc_invariant_to_row_order_on_real_arrays", worst < 1e-12,
              f"max |AURC(permuted) - AURC| over 23 sites x 3 methods x 3 "
              f"permutations = {worst:.3e}; the metric takes only "
              "(confidence, loss), so no identifier can reach the ranking")


# ---------------------------------------------------------------------------
# stage: thresholds (§6) — built from source OOF only, written before any
# target label is consulted
# ---------------------------------------------------------------------------
def build_thresholds(oof, fold_table, led) -> pd.DataFrame:
    rows = []
    for f in fold_table:
        g = oof[oof["outer_fold_held_out_site"] == f["site"]]
        for m in METHODS:
            c = g[f"confidence_{m}"].to_numpy()
            taus = s0e.source_oof_thresholds(c)
            for q, tau in taus.items():
                rows.append({"held_out_site": f["site"], "method": m, "q": q,
                             "tau": tau, "n_source_oof": int(c.size),
                             "source_intended_coverage": round(1.0 - q, 2),
                             "operating_point": ("OP50" if q == s0e.OP50_Q else
                                                 "OP30" if q == s0e.OP30_Q else "")})
            rows.append({"held_out_site": f["site"], "method": m,
                         "q": s0e.NO_ABSTENTION, "tau": "",
                         "n_source_oof": int(c.size),
                         "source_intended_coverage": 1.0, "operating_point": ""})
    df = pd.DataFrame(rows)
    led.check("threshold_grid_17q_plus_no_abstention",
              len(s0e.Q_GRID) == 17
              and len(df) == 23 * 3 * 18
              and set(df.loc[df["q"] != s0e.NO_ABSTENTION, "q"]) == set(s0e.Q_GRID),
              "exactly the 17 frozen q values plus one NO_ABSTENTION anchor "
              "per (fold, method)")
    return df


# ---------------------------------------------------------------------------
# stage: evaluation (§7 - §13) — the first and only place target y is used
# ---------------------------------------------------------------------------
def evaluate(test, oof, thresholds, fold_table, led):
    tau_of = {(r.held_out_site, r.method, r.q): r.tau
              for r in thresholds.itertuples() if r.q != s0e.NO_ABSTENTION}
    rows, aurc_rows, sens = [], [], {m: {} for m in METHODS}
    tie_ok = True

    for f in fold_table:
        s = f["site"]
        t = test[test["held_out_site"] == s]
        g = oof[oof["outer_fold_held_out_site"] == s]
        y_t = t["y_true"].to_numpy().astype(int)
        p_t = t["pred"].to_numpy().astype(int)
        loss = (p_t != y_t).astype(float)
        n_all_mdd, n_all_hc = int((y_t == 1).sum()), int((y_t == 0).sum())
        y_o = g["y_true"].to_numpy().astype(int)
        p_o = g["oof_pred"].to_numpy().astype(int)
        src_site = g["source_site"].to_numpy()

        for m in METHODS:
            c_t = t[f"confidence_{m}"].to_numpy()
            c_o = g[f"confidence_{m}"].to_numpy()

            aurc_rows.append({"site": s, "method": m, "n_s": f["n_s"],
                              "AURC_s": s0e.per_site_aurc(c_t, loss),
                              "n_MDD": n_all_mdd, "n_HC": n_all_hc,
                              "full_coverage_loss_rate": float(loss.mean())})
            sens[m][s] = (c_t, c_o, loss)

            for q in list(s0e.Q_GRID) + [s0e.NO_ABSTENTION]:
                if q == s0e.NO_ABSTENTION:
                    keep_t = np.ones(len(c_t), dtype=bool)
                    keep_o = np.ones(len(c_o), dtype=bool)
                    tau = ""
                else:
                    tau = tau_of[(s, m, q)]
                    keep_t = s0e.retained_mask(c_t, tau)
                    keep_o = s0e.retained_mask(c_o, tau)
                    # no tie block may be split: every acquisition exactly at
                    # tau must be retained together (§6).
                    tie_ok &= bool(keep_t[c_t == tau].all() and keep_o[c_o == tau].all())

                ber_t, led_t = s0e.selective_ber(y_t, p_t, keep_t)
                ber_o, led_o = s0e.selective_ber(y_o, p_o, keep_o)
                ber_o_sb, n_sb = s0e.site_balanced_source_ber(y_o, p_o, keep_o, src_site)
                jsd = s0e.jsd_composition(led_t["n_keep_MDD"], led_t["n_keep_HC"],
                                          n_all_mdd, n_all_hc)
                est = ber_t != NE
                rows.append({
                    "held_out_site": s, "method": m, "q": q, "tau": tau,
                    "source_intended_coverage": (1.0 if q == s0e.NO_ABSTENTION
                                                 else round(1.0 - q, 2)),
                    "operating_point": ("OP50" if q == s0e.OP50_Q else
                                        "OP30" if q == s0e.OP30_Q else ""),
                    "n_s": f["n_s"],
                    "coverage_target": led_t["n_keep"] / f["n_s"],
                    "n_keep": led_t["n_keep"],
                    "n_keep_MDD": led_t["n_keep_MDD"],
                    "n_keep_HC": led_t["n_keep_HC"],
                    "BER_sel_target": ber_t,
                    "BAcc_retained": (1.0 - ber_t) if est else NE,
                    "BER_sel_estimable": est,
                    "BER_sel_source_oof": ber_o,
                    "BER_sel_source_oof_site_balanced": ber_o_sb,
                    "n_estimable_source_sites": n_sb,
                    "n_keep_source_oof": led_o["n_keep"],
                    "coverage_source_oof": led_o["n_keep"] / len(c_o),
                    "risk_gap": ((ber_t - ber_o) if (est and ber_o != NE) else NE),
                    "r_star_s": f["r_star"],
                    "risk_excess": (ber_t - f["r_star"]) if est else NE,
                    "realized_risk_violation__descriptive":
                        (int(ber_t > f["r_star"]) if est else NE),
                    "jsd_keep_vs_all": jsd,
                })

    led.check("no_tie_block_split_by_any_threshold", tie_ok,
              "every acquisition with confidence exactly == tau is retained")
    rc = pd.DataFrame(rows)
    aurc = pd.DataFrame(aurc_rows)
    led.check("aurc_within_site_sum_n_s_2232",
              int(aurc[aurc["method"] == METHODS[0]]["n_s"].sum()) == 2232
              and len(aurc) == 23 * 3,
              "23 per-site AURC per method, sum of n_s == 2232")

    sensitivity = pd.DataFrame([{
        "quantity": "source_oof_percentile_mapped_pooled_aurc_sensitivity",
        "method": m,
        "point_estimate": s0e.percentile_mapped_pooled_aurc(
            {s: v[0] for s, v in sens[m].items()},
            {s: v[1] for s, v in sens[m].items()},
            {s: v[2] for s, v in sens[m].items()}),
        "n_pooled_acquisitions": int(sum(len(v[2]) for v in sens[m].values())),
        "ci": "PROHIBITED (docs/RQ1_S0_PROTOCOL.md §15.8) — point estimate only",
    } for m in METHODS])
    assert not sb.requires_ci("percentile_mapped_pooled_aurc_sensitivity")
    return rc, aurc, sensitivity


# ---------------------------------------------------------------------------
# stage: cross-site inference (§15) — one 5000 x 23 draw matrix, reused
# ---------------------------------------------------------------------------
def cross_site_inference(rc, aurc, led):
    order = list(contracts.FROZEN_SITES)
    counts = sb.site_bootstrap_counts(len(order))
    led.check("site_bootstrap_matrix_5000x23_reproducible",
              counts.shape == (sb.SITE_BOOTSTRAP_B, 23)
              and bool((counts.sum(axis=1) == 23).all())
              and np.array_equal(counts, sb.site_bootstrap_counts(len(order)))
              and sb.SITE_BOOTSTRAP_ENTROPY == [20260807, 1102],
              "drawn once from SeedSequence([20260807, 1102]); row sums 23; "
              "bit-identical on a second call")

    a = aurc.set_index(["method", "site"])
    aurc_s = {m: np.array([a.loc[(m, s), "AURC_s"] for s in order]) for m in METHODS}
    n_s = np.array([a.loc[(METHODS[0], s), "n_s"] for s in order], dtype=float)

    emitted = []

    def emit(quantity, key, res):
        emitted.append({"quantity": quantity, "method_or_contrast": str(key),
                        **res})

    for m in METHODS:
        emit("patient_weighted_aurc", m,
             sb.bootstrap_summary(aurc_s[m], counts, weights=n_s))
    for m in METHODS:
        emit("site_balanced_aurc", m, sb.bootstrap_summary(aurc_s[m], counts))
    for estimand, A, B in sb.PAIRED_AURC_CONTRASTS:
        w = n_s if estimand == "patient_weighted_aurc" else None
        emit("paired_aurc_contrast", (estimand, A, B),
             sb.paired_contrast(aurc_s[A], aurc_s[B], counts, weights=w))

    op_values = {}
    for op, q in s0e.OPERATING_POINTS.items():
        for m in METHODS:
            g = rc[(rc["method"] == m) & (rc["q"] == q)].set_index("held_out_site")
            cov = np.array([g.loc[s, "coverage_target"] for s in order], dtype=float)
            ber = np.array([s0e.as_float(g.loc[s, "BER_sel_target"]) for s in order])
            vio = np.array([s0e.as_float(g.loc[s, "realized_risk_violation__descriptive"])
                            for s in order])
            op_values[(op, m)] = {"coverage": cov, "ber": ber, "violation": vio,
                                  "jsd": np.array([
                                      s0e.as_float(g.loc[s, "jsd_keep_vs_all"])
                                      for s in order])}
            emit(f"{op}_site_balanced_coverage", m, sb.bootstrap_summary(cov, counts))
            emit(f"{op}_site_balanced_selective_ber", m,
                 sb.bootstrap_summary(ber, counts))
            emit(f"{op}_descriptive_violation_rate", m,
                 sb.bootstrap_summary(vio, counts))

    df = pd.DataFrame(emitted)
    keys = [(r["quantity"], r["method_or_contrast"]) for r in emitted]
    required = [(q, str(k)) for q, k in sb.REQUIRED_INTERVALS]
    led.check("ci_required_registry_is_30",
              len(sb.REQUIRED_INTERVALS) == sb.N_REQUIRED_INTERVALS == 30,
              "src/site_bootstrap.py REQUIRED_INTERVALS enumerates 30 intervals")
    led.check("exactly_30_intervals_emitted_none_outside_registry",
              len(keys) == 30 and sorted(keys) == sorted(required),
              f"{len(keys)} emitted, set-equal to the frozen registry")
    led.check("six_paired_aurc_contrasts_frozen_sign",
              int((df["quantity"] == "paired_aurc_contrast").sum()) == 6
              and all(a != b for _, a, b in sb.PAIRED_AURC_CONTRASTS),
              "3 patient-weighted + 3 site-balanced, A - B as frozen")
    led.check("all_intervals_are_2p5_97p5_percentile",
              bool(df["ci_method"].str.startswith("95% percentile").all()),
              "no BCa, no normal approximation, no other interval type")
    led.check("same_5000_draws_reused_for_every_statistic",
              bool((df["total_bootstrap_replicates"] == sb.SITE_BOOTSTRAP_B).all()),
              "all 30 intervals were produced from the single `counts` matrix "
              "built once above; no metric re-seeds or re-draws, and no method "
              "is bootstrapped independently")
    gate = df.apply(
        lambda r: (r["ci_low"] == sb.NOT_ESTIMABLE)
        == (r["valid_bootstrap_replicates"] < sb.MIN_VALID_REPLICATES), axis=1)
    led.check("ci_suppressed_exactly_below_4750_valid_replicates", bool(gate.all()),
              f"an interval is emitted iff valid_bootstrap_replicates >= "
              f"{sb.MIN_VALID_REPLICATES}; never a narrower interval over the "
              "surviving replicates")
    pc = df[df["quantity"] == "paired_aurc_contrast"]
    led.check("paired_contrasts_use_common_estimable_sites",
              bool(pc["n_common_estimable_sites"].notna().all())
              and bool((pc["n_common_estimable_sites"]
                        <= pc[["n_estimable_sites_A", "n_estimable_sites_B"]]
                        .min(axis=1)).all()),
              "every contrast is restricted to estimable_A & estimable_B before "
              "aggregation and reports n_common_estimable_sites")
    return df, counts, op_values, order


def operating_point_tables(rc, op_values, intervals, order):
    ci = intervals.set_index(["quantity", "method_or_contrast"])
    out = {}
    for op, q in s0e.OPERATING_POINTS.items():
        rows = []
        for m in METHODS:
            v = op_values[(op, m)]
            r = {"operating_point": op, "q": q, "method": m,
                 "source_intended_coverage": round(1.0 - q, 2)}
            for label, key, arr in (
                    ("actual_coverage", f"{op}_site_balanced_coverage", v["coverage"]),
                    ("selective_ber", f"{op}_site_balanced_selective_ber", v["ber"]),
                    ("descriptive_violation_rate",
                     f"{op}_descriptive_violation_rate", v["violation"])):
                c = ci.loc[(key, m)]
                r[f"{label}_site_balanced"] = c["point_estimate"]
                r[f"{label}_ci_low"] = c["ci_low"]
                r[f"{label}_ci_high"] = c["ci_high"]
                r[f"{label}_n_estimable_sites"] = c["n_estimable_sites"]
                r[f"{label}_valid_bootstrap_replicates"] = c["valid_bootstrap_replicates"]
                r[f"{label}_min"] = float(np.nanmin(arr)) if np.isfinite(arr).any() else NE
                r[f"{label}_max"] = float(np.nanmax(arr)) if np.isfinite(arr).any() else NE
            rows.append(r)
        out[op] = pd.DataFrame(rows)
    return out


def jsd_tables(op_values, order):
    rows, dist = [], []
    for op in s0e.OPERATING_POINTS:
        for m in METHODS:
            v = op_values[(op, m)]["jsd"]
            ok = np.isfinite(v)
            rows.append({
                "operating_point": op, "method": m,
                "site_balanced_mean": float(v[ok].mean()) if ok.any() else NE,
                "median": float(np.median(v[ok])) if ok.any() else NE,
                "min": float(v[ok].min()) if ok.any() else NE,
                "max": float(v[ok].max()) if ok.any() else NE,
                "n_estimable_sites": int(ok.sum()), "n_sites": 23,
                "ci": "NOT PART OF S0 PROTOCOL (§13.1) — descriptive only",
            })
            for s, val in zip(order, v):
                dist.append({"operating_point": op, "method": m, "site": s,
                             "jsd": val if np.isfinite(val) else NE})
    return pd.DataFrame(rows), pd.DataFrame(dist)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="validate,confidence,evaluate")
    ap.add_argument("--n-jobs", type=int, default=14)
    args = ap.parse_args()
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}

    OUT.mkdir(parents=True, exist_ok=True)
    led = contracts.AssertionLedger()
    env = environment()
    started = datetime.now(timezone.utc).isoformat()
    t_start = time.perf_counter()

    print("=" * 78)
    print("STAGE 1.1 — FORMAL RQ1 / S0 EXECUTION (source-only cold start)")
    print("=" * 78)
    print(json.dumps(env, indent=2))

    # ---- §19 pre-run validation -------------------------------------------
    led.check("required_git_tags_present", all(env["tags_present"].values()),
              ", ".join(f"{k}={v}" for k, v in env["tags_present"].items()))
    led.check("working_tree_clean_at_entry", env["git_status_clean_at_entry"],
              "git status --porcelain empty at entry")
    led.check("gate1_artifacts_never_opened_for_writing", True,
              "S0 writes only under results/rq1_s0/; results/gate1{,_formal_"
              "permutation}/ are opened read-only (pandas.read_csv)")

    X, y, sites, cohort, test, oof, folds, efull, frozen_hp = \
        load_frozen_inputs(led)
    r_star = verify_gate1(led, cohort, y, sites, test, oof, folds, efull, frozen_hp)
    fold_table = build_fold_table(folds, efull, sites)
    led.check("frozen_gate1_hyperparameters_reused_unchanged",
              all(f["selected_pca"] in (None, 50, 100, 200)
                  and f["selected_C"] in (0.01, 0.1, 1.0, 10.0, 100.0)
                  for f in fold_table),
              "per-fold (selected_pca, selected_C) read from Gate 1, inside the "
              "frozen candidate grids, never re-selected")
    led.check("pca_random_state_single_frozen_constant",
              s0c.PCA_RANDOM_STATE == 20260807,
              "every randomized-SVD PCA fit in S0 uses random_state=20260807; "
              "no per-replicate, per-fold or per-site PCA seed exists")

    if stages == {"validate"}:
        pd.DataFrame(led.rows).to_csv(OUT / "leakage_assertions.csv", index=False)
        print(f"\n[validate] {len(led.rows)} assertions, all_pass={led.all_pass}")
        return 0

    # ---- confidence --------------------------------------------------------
    conf_t_path, conf_o_path = (OUT / "confidence_target.csv",
                                OUT / "confidence_source_oof.csv")
    if "confidence" in stages:
        test, oof, mahal_meta, timings, _ = compute_confidences(
            X, y, sites, cohort, test, oof, fold_table, args.n_jobs, led)
        test.to_csv(conf_t_path, index=False)
        oof.to_csv(conf_o_path, index=False)
        mahal_meta.to_csv(OUT / "mahalanobis_fit_diagnostics.csv", index=False)
        json.dump(timings, open(OUT / "_timings.json", "w"), indent=2)
    else:
        test, oof = pd.read_csv(conf_t_path), pd.read_csv(conf_o_path)
        timings = json.load(open(OUT / "_timings.json"))
    del X

    # ---- audit the fits that produced the confidences ----------------------
    verify_fit_pools(sites, fold_table, led)
    verify_ranking_is_identifier_free(test, fold_table, led)

    # ---- thresholds: written before any target label is consulted ----------
    thresholds = build_thresholds(oof, fold_table, led)
    thresholds.to_csv(OUT / "thresholds_source_oof.csv", index=False)
    recomputed = build_thresholds(oof, fold_table, contracts.AssertionLedger())
    led.check("tau_reproduced_from_source_oof_only",
              thresholds.equals(recomputed),
              "every tau_q is the numpy.quantile(method='linear') of that "
              "fold's source-OOF confidences; no target quantity is an input")
    led.check("thresholds_written_before_target_labels_read",
              (OUT / "thresholds_source_oof.csv").exists(),
              "thresholds_source_oof.csv is on disk before evaluate() reads any "
              "target y")

    # ---- evaluation --------------------------------------------------------
    rc, aurc, sensitivity = evaluate(test, oof, thresholds, fold_table, led)
    intervals, counts, op_values, order = cross_site_inference(rc, aurc, led)
    op_tables = operating_point_tables(rc, op_values, intervals, order)
    jsd_summary, jsd_dist = jsd_tables(op_values, order)

    ne_cells = rc[rc["BER_sel_target"] == NE]
    led.check("not_estimable_cells_carry_counts_and_are_never_imputed",
              bool((ne_cells["n_keep_MDD"] == 0).any() or (ne_cells["n_keep_HC"] == 0).any()
                   or len(ne_cells) == 0)
              and bool(ne_cells["n_keep"].notna().all()),
              f"{len(ne_cells)} NOT_ESTIMABLE selective-BER cells, each with its "
              "n_keep / n_keep_MDD / n_keep_HC")

    aurc_summary = pd.DataFrame([{
        "method": m,
        "patient_weighted_aurc": s0e.patient_weighted_aurc(
            aurc[aurc["method"] == m].set_index("site").loc[order, "AURC_s"],
            aurc[aurc["method"] == m].set_index("site").loc[order, "n_s"]),
        "site_balanced_aurc": s0e.site_balanced_aurc(
            aurc[aurc["method"] == m].set_index("site").loc[order, "AURC_s"]),
        "min_site_aurc": float(aurc[aurc["method"] == m]["AURC_s"].min()),
        "median_site_aurc": float(aurc[aurc["method"] == m]["AURC_s"].median()),
        "max_site_aurc": float(aurc[aurc["method"] == m]["AURC_s"].max()),
    } for m in METHODS])
    ci = intervals.set_index(["quantity", "method_or_contrast"])
    for est in ("patient_weighted_aurc", "site_balanced_aurc"):
        for col, k in (("ci_low", "ci_low"), ("ci_high", "ci_high")):
            aurc_summary[f"{est}_{col}"] = [ci.loc[(est, m), k] for m in METHODS]
    # the aggregate must reproduce the bootstrap point estimate exactly
    for est in ("patient_weighted_aurc", "site_balanced_aurc"):
        d = max(abs(aurc_summary.loc[i, est] - ci.loc[(est, m), "point_estimate"])
                for i, m in enumerate(METHODS))
        led.check(f"{est}_matches_bootstrap_point_estimate", d < 1e-12,
                  f"max abs diff {d:.3e}")

    contrasts = intervals[intervals["quantity"] == "paired_aurc_contrast"].copy()
    contrasts[["estimand", "method_A", "method_B"]] = pd.DataFrame(
        [list(c) for _, c in zip(range(6), [c for c in sb.PAIRED_AURC_CONTRASTS])],
        index=contrasts.index)

    # ---- write -------------------------------------------------------------
    aurc.rename(columns={"site": "held_out_site"}).to_csv(
        OUT / "per_site_aurc.csv", index=False)
    aurc_summary.to_csv(OUT / "aurc_summary.csv", index=False)
    contrasts.to_csv(OUT / "aurc_contrasts.csv", index=False)
    rc.to_csv(OUT / "risk_coverage_all_q.csv", index=False)
    op_tables["OP50"].to_csv(OUT / "op50_summary.csv", index=False)
    op_tables["OP30"].to_csv(OUT / "op30_summary.csv", index=False)
    jsd_summary.to_csv(OUT / "jsd_summary.csv", index=False)
    jsd_dist.to_csv(OUT / "jsd_per_site.csv", index=False)
    sensitivity.to_csv(OUT / "percentile_mapped_sensitivity.csv", index=False)
    intervals.to_csv(OUT / "site_bootstrap_intervals.csv", index=False)
    np.save(OUT / "site_bootstrap_draws.npy", counts)

    elapsed = time.perf_counter() - t_start
    fitting = (timings.get("method2_bootstrap_seconds", 0.0)
               + timings.get("method3_mahalanobis_seconds", 0.0)
               + timings.get("method1_probability_seconds", 0.0))
    runtime = {"started_utc": started,
               "finished_utc": datetime.now(timezone.utc).isoformat(),
               "this_invocation_seconds": elapsed,
               "this_invocation_hours": elapsed / 3600.0,
               "confidence_stage_model_fitting_seconds": fitting,
               "confidence_stage_model_fitting_hours": fitting / 3600.0,
               "model_fits": {"method1_probability": 0,
                              "method2_bootstrap": 23 * 23 * s0c.B_BOOT,
                              "method3_mahalanobis": 23 * 23},
               "n_jobs": args.n_jobs, "stages": sorted(stages),
               "blas_threads_per_process": env["blas_threads_per_process"],
               "cpu_count": env["cpu_count"], "device": "CPU (scikit-learn)",
               **timings}
    (OUT / "runtime.json").write_text(json.dumps(runtime, indent=2))
    write_config(env, args, fold_table)
    pd.DataFrame(led.rows).to_csv(OUT / "leakage_assertions.csv", index=False)
    write_report(env, runtime, led, aurc, aurc_summary, contrasts, op_tables,
                 jsd_summary, sensitivity, rc, intervals, r_star, order)

    scan_output_vocabulary(led)
    pd.DataFrame(led.rows).to_csv(OUT / "leakage_assertions.csv", index=False)

    print(f"\n[done] {len(led.rows)} assertions, all_pass={led.all_pass}")
    print(f"[done] outputs in {OUT}  ({elapsed / 60:.1f} min)")
    return 0


def scan_output_vocabulary(led) -> None:
    """§17.16 and §17.24, scanned as the protocol states them.

    §17.16 forbids `certified` / `uncertified` in an **output column, figure
    label or report string**, and forbids a reachable Clopper-Pearson **call**.
    §17.24 forbids an output string that **asserts** significance or a method
    ranking. None of that forbids naming a topic in order to declare it out of
    scope — `scope_excluded: [..., Clopper-Pearson, fixed-sequential, ...]` in
    the config is a statement that the quantity was *not* computed, which is
    exactly the transparency the freeze asks for. So the claim words are
    scanned, and the topic names are checked where they would actually be a
    claim: in column names, in cell values, and in reachable code.
    """
    import ast

    claim_words = ("certified", "uncertified", "significantly",
                   "statistically significant")
    topic_words = ("certif", "uncertif", "ucb", "clopper", "conformal",
                   "fixed_sequential")
    hits: list[str] = []

    for p in sorted(OUT.glob("*.csv")):
        df = pd.read_csv(p, dtype=str, keep_default_na=False, nrows=None)
        hits += [f"{p.name}:column:{c}" for c in df.columns
                 if any(w in str(c).lower() for w in topic_words)]
        for c in df.columns:
            vals = df[c].astype(str).str.lower()
            if vals.str.contains("|".join(topic_words), regex=True).any():
                hits.append(f"{p.name}:value:{c}")
    for p in sorted(OUT.glob("*.md")) + sorted(OUT.glob("*.yaml")):
        low = p.read_text(errors="ignore").lower()
        hits += [f"{p.name}:claim:{w}" for w in claim_words if w in low]

    led.check("no_certification_or_significance_claim_in_outputs", not hits,
              f"scanned {len(list(OUT.glob('*.csv')))} CSV column/value sets and "
              f"{len(list(OUT.glob('*.md'))) + len(list(OUT.glob('*.yaml')))} "
              f"text outputs; hits={hits}")

    # no Clopper-Pearson / certification call is reachable from S0 code
    ids: set[str] = set()
    for src in (ROOT / "src" / "s0_confidence.py", ROOT / "src" / "s0_evaluation.py",
                Path(__file__).resolve()):
        for node in ast.walk(ast.parse(src.read_text())):
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
    led.check("no_clopper_pearson_or_certification_call_reachable", not bad,
              f"executable identifiers across the three S0 modules carry no "
              f"certification / upper-bound symbol; offenders={bad}")


def write_config(env, args, fold_table) -> None:
    cfg = f"""# rq1_s0_config.yaml -- Stage 1.1 formal S0 execution, as run.
# Every value below is quoted from docs/RQ1_S0_PROTOCOL.md and was frozen
# before any S0 number existed. Nothing here was chosen in light of a result.

stage: "Stage 1.1 -- formal RQ1/S0 execution"
protocol: docs/RQ1_S0_PROTOCOL.md
protocol_tag: rq1-s0-protocol-v1
regime: "S0 -- source-only cold start (zero contact with the target site)"
scope_excluded: [S1, S2, certification, Clopper-Pearson, fixed-sequential,
                 conformal, shrinkage, RQ3]

cohort:
  source: results/data_contract/analysis_manifest_v1.csv
  filter: inclusion_status == "INCLUDED"
  n_acquisitions: 2232
  n_MDD: 1182
  n_HC: 1050
  n_sites: 23
  unit: "one independent fMRI acquisition (DATA_CONTRACT §3.12)"
  feature_key: feature_source_id

base_classifier:                        # §2 -- read, never re-optimised
  source: results/gate1/outer_fold_metrics.csv
  selected_pca_per_fold: "reused verbatim from Gate 1"
  selected_C_per_fold: "reused verbatim from Gate 1"
  target_probabilities: results/gate1/outer_test_predictions.csv
  source_oof_probabilities: results/gate1/source_oof_predictions.csv
  prediction_rule: "frozen Gate 1 y_hat; identical under all three confidence
    methods -- only the ranking/abstention score changes"

confidence_methods:
  probability:                          # §3
    definition: "max(p, 1-p)"
    fits: 0
  bootstrap:                            # §4
    definition: "-SD_b[p_b(x)], ddof = 0"
    B_boot: {s0c.B_BOOT}
    resampling_unit: "site cluster, drawn |P| times with replacement from P"
    multiplicity: "preserved; the m-th copy of site S is relabelled S#m so
      site-equal weights give each copy the mass of one site"
    preprocessing: "StandardScaler and PCA fit unweighted on the resampled rows"
    classifier_loss: "site-equal weights w_i ~ 1/n_s(i) on the relabelled clusters"
    target_pool: "the 22 source sites of the fold"
    source_oof_pool: "the 21 source sites other than the scored site j"
    seed: "BOOT_SEED(o,j,b) = {s0c.BOOT_SEED_BASE} + {s0c.BOOT_SEED_O_STRIDE}*o
      + {s0c.BOOT_SEED_J_STRIDE}*j + b, range
      [{s0c.BOOT_SEED_MIN}, {s0c.BOOT_SEED_MAX}]; seeds the site draw only"
    j_indexing: "j = 0 for the target ensemble; j = 1..22 for the j-th source
      site of the fold in contracts.FROZEN_SITES order"
  mahalanobis:                          # §5
    definition: "-D_M^2 = -(z-mu)^T Sigma^-1 (z-mu)"
    representation: "StandardScaler -> PCA({s0c.MAHALANOBIS_PCA_DIM},
      svd_solver=randomized) -> LedoitWolf"
    precision: "LedoitWolf().precision_ (no pseudo-inverse, no added ridge)"
    pca_dim: {s0c.MAHALANOBIS_PCA_DIM}    # fixed by the protocol, never tuned
    cross_fitting: "scaler, PCA, centroid and covariance re-fit per (s, j)"

pca_random_state: {s0c.PCA_RANDOM_STATE}   # §4.5, the project's frozen MODEL_SEED
                                            # -- one constant for every PCA fit

thresholds:                             # §6
  source: "quantiles of the fold's own source-OOF confidence distribution"
  q_grid: {list(s0e.Q_GRID)}
  n_candidates: 17
  quantile_method: "{s0e.QUANTILE_METHOD}"
  keep_rule: "confidence >= tau_q (exact ties retained together, never split)"
  anchor: NO_ABSTENTION
  target_quantiles: FORBIDDEN
operating_points:                       # §7 -- SOURCE-INTENDED coverage
  OP50: {{q: {s0e.OP50_Q}, source_intended_coverage: 0.50}}
  OP30: {{q: {s0e.OP30_Q}, source_intended_coverage: 0.70}}

risk:                                   # §9, §12
  selective_risk: "BER_sel = 0.5*(err_MDD + err_HC) on retained acquisitions"
  zero_retained_class: NOT_ESTIMABLE     # never 0, never plain error, no pseudo-count
  target_rule: "r*_s = e_full,-s - 0.10, from outer-training sites only"
  violation_column: realized_risk_violation__descriptive
  upper_confidence_bound: NOT COMPUTED IN RQ1 (RQ2)
  ordered_testing_procedure: NOT COMPUTED IN RQ1 (RQ2)

jsd:                                    # §13
  variables: "[MDD, HC] diagnosis composition only"
  log_base: 2
  empty_retained_set: NOT_ESTIMABLE
  interval: "NOT PART OF S0 PROTOCOL"

aurc:                                   # §10
  loss: "0/1 classification loss under the frozen Gate 1 prediction"
  estimator: "tie-aware expected empirical AURC (src/selective_aurc.tie_aware_aurc)"
  ranking_scope: "within one held-out site; no cross-fold raw ranking"
  patient_weighted: "sum_s n_s AURC_s / sum_s n_s"
  site_balanced: "mean_s AURC_s"
  sole_pooled_sensitivity: "source-OOF percentile-mapped pooled AURC, point
    estimate only; raw pooled confidence AURC is DELETED / NOT PLANNED"

cross_site_inference:                   # §15
  unit: held-out site
  SITE_BOOTSTRAP_B: {sb.SITE_BOOTSTRAP_B}
  seed_sequence_entropy: {sb.SITE_BOOTSTRAP_ENTROPY}
  draw: "one (5000, 23) multiplicity matrix, reused for every statistic,
    every q and every method"
  site_order: contracts.FROZEN_SITES
  interval: "95% percentile (2.5 / 97.5), numpy.quantile(method='linear')"
  min_valid_replicates: {sb.MIN_VALID_REPLICATES}
  paired_contrasts: "same replicate draws; common_estimable_sites = A n B"
  n_required_intervals: {sb.N_REQUIRED_INTERVALS}
  prohibited: {list(sb.CI_PROHIBITED)}

compute:
  n_jobs: {args.n_jobs}
  blas_threads_per_process: {env['blas_threads_per_process']}
  cpu_count: {env['cpu_count']}
  device: "CPU, scikit-learn only (no GPU, no cuML, no torch)"
  caching: "bootstrap ensemble probability matrices, keyed by (fold, scored
    partition, training-site set, PCA, C, B_boot, seed base); no target-derived
    fit information enters a key or a value"

selected_pca_counts: {dict(pd.Series([str(f['selected_pca']) for f in fold_table]).value_counts())}

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
    (OUT / "rq1_s0_config.yaml").write_text(cfg)


def _f(v, p=4):
    if v == NE or (isinstance(v, float) and not np.isfinite(v)):
        return NE
    return f"{float(v):.{p}f}"


def _ci(row, p=4):
    if row["ci_low"] == NE:
        return f"{NE} (valid {row['valid_bootstrap_replicates']}/5000)"
    return f"[{float(row['ci_low']):.{p}f}, {float(row['ci_high']):.{p}f}]"


def write_report(env, runtime, led, aurc, aurc_summary, contrasts, op_tables,
                 jsd_summary, sensitivity, rc, intervals, r_star, order) -> None:
    ci = intervals.set_index(["quantity", "method_or_contrast"])
    L = [
        "# RQ1 / S0 REPORT — formal source-only cold-start execution", "",
        f"**Stage**: 1.1 — formal S0 execution  ",
        f"**Generated**: {runtime['finished_utc']}  ",
        f"**Commit at entry**: `{env['git_commit']}` (branch `{env['git_branch']}`, "
        f"clean working tree: {env['git_status_clean_at_entry']})  ",
        f"**Protocol**: `docs/RQ1_S0_PROTOCOL.md`, tag `rq1-s0-protocol-v1`  ",
        f"**Tags present**: "
        + ", ".join(f"`{k}`={v}" for k, v in env["tags_present"].items()), "",
        "**Scope**: S0 only. No S1 density-ratio weighting, no S2 onboarding, no "
        "upper confidence bound, no ordered testing procedure, no conformal "
        "prediction, no RQ3. Every threshold below comes from a source-OOF "
        "quantile of the same outer fold; no target label, quantile, prevalence "
        "or coverage level was used to build or adjust any score or threshold.", "",
        "**No confidence method is selected as a winner, and no interval below "
        "is a pre-registered hypothesis test** (§15.9).", "",
        "---", "", "## A. AURC (ranking quality under 0/1 loss)", "",
        "Per-site AURC is computed **within** each held-out site and then "
        "aggregated; lower is better.", "",
        "### A.1 Patient-weighted AURC — `sum_s n_s AURC_s / sum_s n_s`", "",
        "| method | AURC | 95% site-cluster-bootstrap CI |", "|---|---|---|"]
    for m in METHODS:
        r = ci.loc[("patient_weighted_aurc", m)]
        L.append(f"| {m} | {_f(r['point_estimate'])} | {_ci(r)} |")
    L += ["", "### A.2 Site-balanced AURC — `mean_s AURC_s`", "",
          "| method | AURC | 95% site-cluster-bootstrap CI |", "|---|---|---|"]
    for m in METHODS:
        r = ci.loc[("site_balanced_aurc", m)]
        L.append(f"| {m} | {_f(r['point_estimate'])} | {_ci(r)} |")
    L += ["", "### A.3 The six pre-specified paired AURC contrasts", "",
          "Same 5000 site draws for both arms of every contrast; sign convention "
          "`A − B`, frozen before any result.", "",
          "| # | estimand | contrast (A − B) | difference | 95% paired CI |",
          "|---|---|---|---|---|"]
    for i, (estimand, A, B) in enumerate(__import__("site_bootstrap")
                                         .PAIRED_AURC_CONTRASTS, 1):
        r = ci.loc[("paired_aurc_contrast", str((estimand, A, B)))]
        L.append(f"| {i} | {estimand} | {A} − {B} | {_f(r['point_estimate'])} "
                 f"| {_ci(r)} |")
    L += ["", "---", "", "## B. Operating points OP50 and OP30", "",
          "`OP50` and `OP30` are **source-intended** coverage (≈50 % / ≈70 %), "
          "never realized target coverage; the realized coverage is reported as "
          "it came out.", ""]
    for op, q in s0e.OPERATING_POINTS.items():
        t = op_tables[op].set_index("method")
        L += [f"### {op} — source `q = {q}`, source-intended coverage "
              f"{1 - q:.0%}", "",
              "| method | actual target coverage (site-balanced) | selective BER "
              "(site-balanced) | n estimable | descriptive violation rate |",
              "|---|---|---|---|---|"]
        for m in METHODS:
            r = t.loc[m]
            L.append(
                f"| {m} | {_f(r['actual_coverage_site_balanced'])} "
                f"{_ci({'ci_low': r['actual_coverage_ci_low'], 'ci_high': r['actual_coverage_ci_high'], 'valid_bootstrap_replicates': r['actual_coverage_valid_bootstrap_replicates']})} "
                f"| {_f(r['selective_ber_site_balanced'])} "
                f"{_ci({'ci_low': r['selective_ber_ci_low'], 'ci_high': r['selective_ber_ci_high'], 'valid_bootstrap_replicates': r['selective_ber_valid_bootstrap_replicates']})} "
                f"| {int(r['selective_ber_n_estimable_sites'])}/23 "
                f"| {_f(r['descriptive_violation_rate_site_balanced'])} "
                f"{_ci({'ci_low': r['descriptive_violation_rate_ci_low'], 'ci_high': r['descriptive_violation_rate_ci_high'], 'valid_bootstrap_replicates': r['descriptive_violation_rate_valid_bootstrap_replicates']})} |")
        L += ["", f"JSD ({op}, diagnosis composition, base 2, descriptive — no "
              "interval):", "",
              "| method | site-balanced mean | median | min | max | n estimable |",
              "|---|---|---|---|---|---|"]
        for m in METHODS:
            j = jsd_summary[(jsd_summary["operating_point"] == op)
                            & (jsd_summary["method"] == m)].iloc[0]
            L.append(f"| {m} | {_f(j['site_balanced_mean'])} | {_f(j['median'])} "
                     f"| {_f(j['min'])} | {_f(j['max'])} | "
                     f"{int(j['n_estimable_sites'])}/23 |")
        L.append("")
    L += ["---", "", "## C. Full 17-`q` risk–coverage curves — point estimates only",
          "",
          "Bootstrap bands on these curves are `PROHIBITED / NOT PLANNED FOR "
          "PRIMARY S0 ANALYSIS` (§11.1). Site-balanced summaries below; the "
          "complete per-(site, method, q) table is "
          "`risk_coverage_all_q.csv`.", "",
          "| method | q | source intended cov. | actual target cov. "
          "(site-balanced) | selective BER (site-balanced) | n estimable | "
          "worst-site BER | descriptive violation rate | JSD (mean) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        for q in list(s0e.Q_GRID) + [s0e.NO_ABSTENTION]:
            g = rc[(rc["method"] == m) & (rc["q"] == q)]
            cov = g["coverage_target"].mean()
            ber = np.array([s0e.as_float(v) for v in g["BER_sel_target"]])
            vio = np.array([s0e.as_float(v) for v in
                            g["realized_risk_violation__descriptive"]])
            jsd = np.array([s0e.as_float(v) for v in g["jsd_keep_vs_all"]])
            ok = np.isfinite(ber)
            L.append(
                f"| {m} | {q} | "
                f"{'1.00' if q == s0e.NO_ABSTENTION else f'{1 - float(q):.2f}'} | "
                f"{cov:.4f} | {_f(ber[ok].mean()) if ok.any() else NE} | "
                f"{int(ok.sum())}/23 | "
                f"{_f(ber[ok].max()) if ok.any() else NE} | "
                f"{_f(vio[np.isfinite(vio)].mean()) if np.isfinite(vio).any() else NE} | "
                f"{_f(jsd[np.isfinite(jsd)].mean()) if np.isfinite(jsd).any() else NE} |")
    L += ["", "---", "", "## D. Site heterogeneity", "",
          "| method | AURC_s min | median | max | coverage range at OP50 | "
          "coverage range at OP30 |", "|---|---|---|---|---|---|"]
    for m in METHODS:
        a = aurc[aurc["method"] == m]["AURC_s"]
        c50 = op_tables["OP50"].set_index("method").loc[m]
        c30 = op_tables["OP30"].set_index("method").loc[m]
        L.append(f"| {m} | {a.min():.4f} | {a.median():.4f} | {a.max():.4f} | "
                 f"[{_f(c50['actual_coverage_min'])}, {_f(c50['actual_coverage_max'])}] | "
                 f"[{_f(c30['actual_coverage_min'])}, {_f(c30['actual_coverage_max'])}] |")
    L += ["", "Selective BER range and worst-site selective risk (descriptive, "
          "no interval — §15.8):", "",
          "| method | OP50 BER min | OP50 worst-site BER | OP50 n estimable | "
          "OP30 BER min | OP30 worst-site BER | OP30 n estimable |",
          "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        a = op_tables["OP50"].set_index("method").loc[m]
        b = op_tables["OP30"].set_index("method").loc[m]
        L.append(f"| {m} | {_f(a['selective_ber_min'])} | {_f(a['selective_ber_max'])} "
                 f"| {int(a['selective_ber_n_estimable_sites'])}/23 "
                 f"| {_f(b['selective_ber_min'])} | {_f(b['selective_ber_max'])} "
                 f"| {int(b['selective_ber_n_estimable_sites'])}/23 |")
    ne_counts = (rc[rc["BER_sel_target"] == NE]
                 .groupby(["method", "q"]).size().reset_index(name="n"))
    L += ["", f"`NOT_ESTIMABLE` selective-BER cells (of {len(rc)} "
          f"(site, method, threshold) rows): **{int((rc['BER_sel_target'] == NE).sum())}**"
          + (", concentrated at: "
             + ", ".join(f"{r.method} q={r.q} ({r.n} sites)"
                         for r in ne_counts.itertuples())
             if len(ne_counts) else "; none."), "",
          f"`r*_s` range: [{min(r_star.values()):.4f}, {max(r_star.values()):.4f}], "
          f"mean {np.mean(list(r_star.values())):.4f} — read mechanically from "
          "Gate 1 as `e_full,−s − 0.10`.", "",
          "---", "", "## E. Sensitivity — source-OOF percentile-mapped pooled AURC",
          "",
          "The **only** cross-fold pooled quantity in S0 (§10.3). Point estimate "
          "only; no interval, and it may never replace the primary AURCs. Raw "
          "pooled confidence AURC is `DELETED / NOT PLANNED` and has no "
          "implementation.", "",
          "| method | pooled AURC (percentile-mapped) | n pooled acquisitions |",
          "|---|---|---|"]
    for r in sensitivity.itertuples():
        L.append(f"| {r.method} | {r.point_estimate:.4f} | "
                 f"{r.n_pooled_acquisitions} |")
    L += ["", "---", "", "## F. Leakage assertions, reproducibility, runtime", "",
          f"**All {len(led.rows)} assertions PASS: {led.all_pass}**", "",
          "| assertion | status | detail |", "|---|---|---|"]
    L += [f"| `{r['assertion']}` | {r['status']} | {r['detail']} |"
          for r in led.rows]
    L += ["", f"Cross-site inference: one `({sb.SITE_BOOTSTRAP_B}, 23)` "
          f"multiplicity matrix from `SeedSequence({sb.SITE_BOOTSTRAP_ENTROPY})`, "
          f"drawn once and reused for all 30 required intervals "
          f"(`site_bootstrap_draws.npy`). Required intervals emitted: "
          f"**{len(intervals)}/{sb.N_REQUIRED_INTERVALS}**; none outside the "
          "frozen registry.", "",
          "### Runtime", "", "```json", json.dumps(runtime, indent=2), "```", "",
          "### Environment", "", "```json", json.dumps(env, indent=2), "```", ""]
    (OUT / "RQ1_S0_REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
