"""Reduced-cost permutation machinery (spec §4 RQ0, 算力预算与降级方案).

The spec pre-declares this variant and the condition that triggers it:

    200 × T ≤ 24 小时 → 完整重跑
    超出 → 降级变体：超参从观测运行冻结，置换只重跑外层（训练 + 校准），
           不重跑内层调参；次数 200；论文中明写为 reduced-cost permutation

Gate 1 measured `T = 1942.1 s`, so `200 × T = 107.9 h > 24 h` and the reduced
variant is mechanically required — it is not a discretionary optimisation.

What "reduced" means here, precisely:

* the PCA dimension and ridge `C` of every outer fold are **read from the
  observed run** (`results/gate1/outer_fold_metrics.csv`) and never re-selected;
* the inner leave-one-source-site-out CV is therefore not re-run;
* everything else is unchanged — same cohort, same FC, same site weighting,
  same decision threshold, same per-fold outer refit on permuted source labels,
  same held-out-site prediction, same statistic (macro mean of 23 per-site
  BAcc).

What is *not* reduced, and must not be:

* the held-out site still enters no scaler fit, no PCA fit and no model fit;
* the estimator stack stays scikit-learn CPU (`nested_loso.fit_predict`), so
  the null and the observed statistic come from numerically the same estimator.

Caching: the source-fitted StandardScaler/PCA depends on `X` and the source
partition only — never on `y` — so it is identical across permutations and is
computed once per outer fold. `assert_transform_equivalence` proves that the
cached projection equals a from-scratch refit; if that ever fails, the caller
must fall back to refitting per permutation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

import contracts
import nested_loso as nl

# --- seeds, reserved once and disjoint by construction ----------------------
# Gate 1 smoke permutations used 20260807 + i, i = 0..19.
FORMAL_SEED_BASE = 20270101          # formal permutation b uses BASE + b
FORMAL_B = 200
# Runtime-only benchmark seeds. Reserved so a benchmark permutation can never
# be mistaken for, or reused as, one of the 200 formal null draws.
BENCHMARK_SEEDS: tuple[int, ...] = (90000001, 90000002, 90000003)


def formal_seeds(b: int = FORMAL_B) -> list[int]:
    """The B permutation seeds, fixed before the run and recorded verbatim."""
    seeds = [FORMAL_SEED_BASE + i for i in range(b)]
    assert len(set(seeds)) == b, "formal seeds are not unique"
    assert not set(seeds) & set(BENCHMARK_SEEDS), "benchmark seed leaked into B"
    return seeds


def smoke_seeds(n: int = 20, base: int = 20260807) -> list[int]:
    """Gate 1 smoke seeds, re-derived from `scripts/run_gate1.py` (base + i)."""
    return [base + i for i in range(n)]


# ---------------------------------------------------------------------------
# frozen per-fold hyperparameters
# ---------------------------------------------------------------------------
def load_frozen_hyperparameters(path: Path) -> pd.DataFrame:
    """Per-outer-fold (PCA, C) as selected by the observed Gate 1 run.

    Read mechanically from `outer_fold_metrics.csv`; nothing here re-selects,
    re-orders or re-optimises anything.
    """
    df = pd.read_csv(path)
    assert len(df) == contracts.FROZEN_N_SITES, f"{len(df)} outer folds, need 23"
    assert set(df["site"]) == set(contracts.FROZEN_SITES), "outer fold set moved"
    df = df.sort_values("site", key=lambda c: c.str.slice(1).astype(int))
    df = df.reset_index(drop=True)

    def _pca(v):
        if str(v) == "no-PCA":
            return None
        return int(v)

    # object dtype on purpose: a plain map would coerce the `no-PCA` Nones to
    # float NaN and turn `200` into `200.0`, which sklearn's PCA rejects.
    df["pca"] = pd.Series(np.array([_pca(v) for v in df["selected_pca"]],
                                   dtype=object), index=df.index, dtype=object)
    df["C"] = df["selected_C"].astype(float)
    assert df["pca"].map(lambda p: p is None or p in nl.PCA_CANDIDATES).all(), \
        "a frozen PCA choice is outside the frozen candidate set"
    assert df["C"].isin(nl.C_GRID).all(), \
        "a frozen C is outside the frozen C grid"
    return df


# ---------------------------------------------------------------------------
# per-fold source-fitted transformation cache
# ---------------------------------------------------------------------------
@dataclass
class FoldPlan:
    """Everything one outer fold needs, with `y` deliberately absent."""

    site: str
    pca: int | None
    C: float
    tr_idx: np.ndarray
    te_idx: np.ndarray
    weights: np.ndarray          # site-equalising, source partition, order-aligned
    ztr_path: str
    zte_path: str


def fit_source_projection(X: np.ndarray, sites: np.ndarray, held_out: str,
                          pca: int | None, seed: int):
    """Source-only StandardScaler (+PCA), identical to the observed outer refit.

    Mirrors `nested_loso.run_outer_fold`'s refit step exactly:
    `Preprocessor((best_pca,)).fit(X_src, seed)`.
    """
    te = sites == held_out
    tr = ~te
    contracts.assert_held_out_site_invisible(np.unique(sites[tr]), held_out)
    pre = nl.Preprocessor((pca,)).fit(X[tr], seed)
    return pre, tr, te


def build_fold_caches(X: np.ndarray, sites: np.ndarray, frozen: pd.DataFrame,
                      seed: int, cache_dir: Path,
                      verbose: bool = True) -> list[FoldPlan]:
    """Precompute the source-fitted projection of every outer fold, once.

    The projection is a function of `X` and the source partition only, so it is
    invariant to label permutation — which is exactly why it may be reused.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    plans: list[FoldPlan] = []
    for row in frozen.itertuples():
        pre, tr, te = fit_source_projection(X, sites, row.site, row.pca, seed)
        Ztr = pre.transform(X[tr], row.pca)
        Zte = pre.transform(X[te], row.pca)
        assert np.isfinite(Ztr).all() and np.isfinite(Zte).all(), \
            f"non-finite projection at {row.site}"
        ztr_p = cache_dir / f"ztr_{row.site}.npy"
        zte_p = cache_dir / f"zte_{row.site}.npy"
        np.save(ztr_p, np.ascontiguousarray(Ztr))
        np.save(zte_p, np.ascontiguousarray(Zte))
        plans.append(FoldPlan(
            site=row.site, pca=row.pca, C=float(row.C),
            tr_idx=np.flatnonzero(tr), te_idx=np.flatnonzero(te),
            weights=nl.site_equal_weights(sites[tr]),
            ztr_path=str(ztr_p), zte_path=str(zte_p),
        ))
        if verbose:
            print(f"[cache] {row.site}: Ztr{Ztr.shape} Zte{Zte.shape} "
                  f"pca={row.pca} C={row.C}", flush=True)
    return plans


_MMAP: dict[str, np.ndarray] = {}


def _load(path: str) -> np.ndarray:
    """Read-only memmap, one handle per worker process."""
    z = _MMAP.get(path)
    if z is None:
        z = np.load(path, mmap_mode="r")
        _MMAP[path] = z
    return z


# ---------------------------------------------------------------------------
# one permutation
# ---------------------------------------------------------------------------
def fold_bacc(plan: FoldPlan, y_perm: np.ndarray, seed: int,
              Ztr: np.ndarray | None = None,
              Zte: np.ndarray | None = None) -> float:
    """Refit on permuted source labels, predict the permuted held-out site."""
    if Ztr is None:
        Ztr = _load(plan.ztr_path)
    if Zte is None:
        Zte = _load(plan.zte_path)
    prob = nl.fit_predict(Ztr, y_perm[plan.tr_idx], plan.weights,
                          Zte, plan.C, seed)
    yhat = (prob >= nl.DECISION_THRESHOLD).astype(int)
    return float(balanced_accuracy_score(y_perm[plan.te_idx], yhat))


def reduced_cost_statistic(plans: list[FoldPlan], y_perm: np.ndarray,
                           seed: int) -> tuple[float, list[float]]:
    """Macro mean of the 23 held-out-site BAcc values — the RQ0 statistic.

    Identical in definition to the observed statistic: an unweighted mean over
    outer folds of per-site balanced accuracy. Never patient-pooled.
    """
    baccs = [fold_bacc(p, y_perm, seed) for p in plans]
    assert len(baccs) == len(plans)          # 23 in production; the runner
    assert np.isfinite(baccs).all()          # asserts the count separately
    return float(np.mean(baccs)), baccs


def run_permutation(seed: int, plans: list[FoldPlan], y: np.ndarray,
                    sites: np.ndarray, model_seed: int) -> dict:
    """One complete reduced-cost null draw. Pure function of `seed`."""
    rng = np.random.default_rng(seed)
    y_perm = nl.permute_within_site(y, sites, rng)   # asserts within-site
    contracts.assert_permutation_within_site(y, y_perm, sites)
    stat, baccs = reduced_cost_statistic(plans, y_perm, model_seed)
    return {
        "seed": seed,
        "macro_site_bacc": stat,
        "sd_across_sites": float(np.std(baccs, ddof=1)),
        "min_site_bacc": float(np.min(baccs)),
        "max_site_bacc": float(np.max(baccs)),
        "per_site_bacc": {p.site: b for p, b in zip(plans, baccs)},
    }


# ---------------------------------------------------------------------------
# cached-vs-refitted equivalence (mandatory before the formal run)
# ---------------------------------------------------------------------------
def assert_transform_equivalence(X, sites, plans, seed, rtol=1e-10,
                                 atol=1e-10) -> list[dict]:
    """Refit each fold's scaler/PCA from scratch and compare to the cache.

    If any fold fails, the cache may not be used and the caller must refit the
    projection inside every permutation.
    """
    rows = []
    for p in plans:
        pre, tr, te = fit_source_projection(X, sites, p.site, p.pca, seed)
        Ztr_new = pre.transform(X[tr], p.pca)
        Zte_new = pre.transform(X[te], p.pca)
        Ztr_old, Zte_old = _load(p.ztr_path), _load(p.zte_path)
        d_tr = float(np.max(np.abs(np.asarray(Ztr_old) - Ztr_new)))
        d_te = float(np.max(np.abs(np.asarray(Zte_old) - Zte_new)))
        ok = (np.allclose(Ztr_old, Ztr_new, rtol=rtol, atol=atol)
              and np.allclose(Zte_old, Zte_new, rtol=rtol, atol=atol))
        rows.append({"site": p.site, "pca": "no-PCA" if p.pca is None else p.pca,
                     "max_abs_diff_train": d_tr, "max_abs_diff_test": d_te,
                     "allclose": bool(ok)})
        assert ok, (f"cached transform != refitted transform at {p.site} "
                    f"(train {d_tr:g}, test {d_te:g})")
    return rows


def permutation_pvalue(observed: float, null: np.ndarray, b: int) -> dict:
    """Finite-sample corrected permutation p-value, upper-tailed.

        p = (1 + #{null >= observed}) / (B + 1)
    """
    null = np.asarray(null, dtype=float)
    assert null.size == b, f"{null.size} null statistics, expected {b}"
    assert np.isfinite(null).all(), "non-finite null statistic"
    n_ge = int((null >= observed).sum())
    return {
        "B": int(b),
        "observed": float(observed),
        "n_null_ge_observed": n_ge,
        "p_value": (1.0 + n_ge) / (b + 1.0),
        "p_value_formula": "(1 + #{null >= observed}) / (B + 1)",
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_median": float(np.median(null)),
        "null_min": float(null.min()),
        "null_max": float(null.max()),
        "null_p2.5": float(np.percentile(null, 2.5)),
        "null_p95": float(np.percentile(null, 95)),
        "null_p97.5": float(np.percentile(null, 97.5)),
        "observed_rank_high_to_low": n_ge + 1,
        "observed_percentile_in_null": float((null < observed).mean() * 100.0),
    }


def save_plans(plans: list[FoldPlan], path: Path) -> None:
    path.write_text(json.dumps(
        [{"site": p.site, "pca": p.pca, "C": p.C,
          "n_train": int(p.tr_idx.size), "n_test": int(p.te_idx.size),
          "ztr": p.ztr_path, "zte": p.zte_path} for p in plans], indent=2))
