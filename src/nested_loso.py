"""Nested LOSO-site protocol (spec §7) — Gate 1 diagnostic pipeline only.

For each outer held-out site `s`:

  1. site `s` is fully removed;
  2. an inner leave-one-source-site-out CV over the remaining 22 sites selects
     the feature-compression dimension and the ridge `C`, and produces
     out-of-fold source predictions;
  3. the selected configuration is refit on all 22 source sites and applied to
     site `s`.

Nothing in this module implements S0/S1/S2, certification, conformal or
shrinkage — those are out of Gate 1's scope.

Frozen elsewhere, used here:
  * scaler = StandardScaler, PCA candidates {50, 100, 200, no-PCA}, fit only
    within the corresponding training partition (EXPERIMENT_LOG §7);
  * training weights w_i ∝ 1/n_{s(i)} with n_s counted in acquisitions
    (spec §10, DATA_CONTRACT §3.12.8(4));
  * the inner selection metric is the macro mean of per-site balanced accuracy
    (spec §4 RQ0), never pooled accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import contracts

# --- pre-fixed hyperparameter grids ---------------------------------------
# PCA candidates are the frozen EXPERIMENT_LOG §7 set. `None` = no compression.
PCA_CANDIDATES: tuple[int | None, ...] = (None, 50, 100, 200)
# Ridge C grid: not previously frozen anywhere in the project, fixed here
# before any model was fitted and recorded in gate1_config.yaml.
C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
DECISION_THRESHOLD = 0.5
MAX_ITER = 2000


# ---------------------------------------------------------------------------
# weights and metrics
# ---------------------------------------------------------------------------
def site_equal_weights(sites: np.ndarray) -> np.ndarray:
    """w_i = (n / n_sites) / n_{s(i)}  — sites weighted equally, mean(w) == 1.

    Each site contributes total mass n/n_sites regardless of its size, so the
    relative site weighting is exactly 1/n_{s(i)}; the constant only fixes the
    normalisation so that the mean sample weight is 1.
    """
    uniq, counts = np.unique(sites, return_counts=True)
    n_per_site = dict(zip(uniq, counts))
    scale = len(sites) / len(uniq)
    w = np.array([scale / n_per_site[s] for s in sites], dtype=float)
    contracts.assert_site_weights(w, sites)
    return w


def per_site_bacc(y: np.ndarray, yhat: np.ndarray, sites: np.ndarray) -> dict:
    """Per-site balanced accuracy on unique acquisitions (§3.12.8(5))."""
    out = {}
    for s in np.unique(sites):
        m = sites == s
        if len(np.unique(y[m])) < 2:      # undefined without both classes
            continue
        out[str(s)] = float(balanced_accuracy_score(y[m], yhat[m]))
    return out


def macro_site_bacc(y: np.ndarray, yhat: np.ndarray, sites: np.ndarray) -> float:
    """The RQ0 primary statistic: macro mean of per-site BAcc."""
    v = list(per_site_bacc(y, yhat, sites).values())
    return float(np.mean(v)) if v else float("nan")


def balanced_error(y: np.ndarray, yhat: np.ndarray) -> float:
    """Full-coverage balanced error rate = mean of the two class error rates."""
    errs = []
    for c in (0, 1):
        m = y == c
        if m.sum() == 0:
            continue
        errs.append(float((yhat[m] != c).mean()))
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# one fit
# ---------------------------------------------------------------------------
@dataclass
class Preprocessor:
    """StandardScaler (+ one PCA per candidate), fit on a training partition only.

    Each PCA candidate gets its own fit rather than being sliced out of a
    single larger fit: `svd_solver="randomized"` is approximate, so the leading
    k columns of a rank-200 fit are *not* numerically identical to a rank-k fit,
    and the frozen candidate set {50, 100, 200, no-PCA} is defined as four
    dimensions, not one truncation ladder.
    """

    candidates: tuple
    scaler: StandardScaler = field(default=None, repr=False)
    pcas: dict = field(default_factory=dict, repr=False)

    def fit(self, X_tr: np.ndarray, seed: int) -> "Preprocessor":
        self.scaler = StandardScaler().fit(X_tr)
        Z = self.scaler.transform(X_tr)
        for p in self.candidates:
            if p is None:
                continue
            k = min(p, min(Z.shape) - 1)
            self.pcas[p] = PCA(n_components=k, svd_solver="randomized",
                               random_state=seed).fit(Z)
        return self

    def transform(self, X: np.ndarray, n_components: int | None) -> np.ndarray:
        Z = self.scaler.transform(X)
        if n_components is None:
            return Z
        return self.pcas[n_components].transform(Z)


def fit_predict(X_tr, y_tr, w_tr, X_te, C: float, seed: int) -> np.ndarray:
    """Ridge (L2) logistic regression; returns P(y=1) on X_te."""
    clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                             max_iter=MAX_ITER, random_state=seed)
    clf.fit(X_tr, y_tr, sample_weight=w_tr)
    return clf.predict_proba(X_te)[:, 1]


# ---------------------------------------------------------------------------
# one outer fold
# ---------------------------------------------------------------------------
def run_outer_fold(X, y, sites, held_out: str, seed: int,
                   pca_candidates=PCA_CANDIDATES, c_grid=C_GRID) -> dict:
    """Inner CV -> selection -> refit -> predict the held-out site."""
    te = sites == held_out
    tr = ~te
    contracts.assert_disjoint(np.flatnonzero(tr), np.flatnonzero(te))

    X_src, y_src, s_src = X[tr], y[tr], sites[tr]
    source_sites = np.unique(s_src)
    contracts.assert_held_out_site_invisible(source_sites, held_out)

    # ---- inner leave-one-source-site-out ---------------------------------
    n_src = len(y_src)
    oof_prob = {(p, c): np.full(n_src, np.nan)
                for p in pca_candidates for c in c_grid}

    for u in source_sites:
        iva = s_src == u
        itr = ~iva
        contracts.assert_held_out_site_invisible(np.unique(s_src[itr]), held_out)
        contracts.assert_held_out_site_invisible(np.unique(s_src[itr]), u)

        pre = Preprocessor(pca_candidates).fit(X_src[itr], seed)
        w = site_equal_weights(s_src[itr])
        for p in pca_candidates:
            Ztr = pre.transform(X_src[itr], p)
            Zva = pre.transform(X_src[iva], p)
            for c in c_grid:
                oof_prob[(p, c)][iva] = fit_predict(
                    Ztr, y_src[itr], w, Zva, c, seed)

    # ---- selection on the inner OOF predictions --------------------------
    scores = {}
    for key, prob in oof_prob.items():
        assert np.isfinite(prob).all(), "inner OOF prediction missing"
        scores[key] = macro_site_bacc(
            y_src, (prob >= DECISION_THRESHOLD).astype(int), s_src)
    # deterministic tie-break: highest inner score, then smaller C, then
    # stronger compression (fewer components; no-PCA last).
    def _rank(k):
        p, c = k
        return (-scores[k], c_grid.index(c), pca_candidates.index(p))
    best = sorted(scores, key=_rank)[0]
    best_pca, best_c = best

    # ---- refit on all source sites, predict the held-out site ------------
    pre = Preprocessor((best_pca,)).fit(X_src, seed)
    w = site_equal_weights(s_src)
    prob_te = fit_predict(pre.transform(X_src, best_pca), y_src, w,
                          pre.transform(X[te], best_pca), best_c, seed)
    yhat_te = (prob_te >= DECISION_THRESHOLD).astype(int)

    y_te = y[te]
    auroc = (float(roc_auc_score(y_te, prob_te))
             if len(np.unique(y_te)) == 2 else float("nan"))

    # OOF source predictions under the selected configuration
    oof_sel = oof_prob[best]
    oof_hat = (oof_sel >= DECISION_THRESHOLD).astype(int)

    return {
        "site": held_out,
        "n_test": int(te.sum()),
        "n_MDD": int((y_te == 1).sum()),
        "n_HC": int((y_te == 0).sum()),
        "selected_pca": "no-PCA" if best_pca is None else int(best_pca),
        "selected_C": float(best_c),
        "inner_macro_site_bacc": float(scores[best]),
        "bacc": float(balanced_accuracy_score(y_te, yhat_te)),
        "auroc": auroc,
        "e_full_minus_s": balanced_error(y_src, oof_hat),
        "n_source": int(tr.sum()),
        "n_source_sites": int(len(source_sites)),
        "_oof": {"prob": oof_sel, "yhat": oof_hat, "y": y_src, "site": s_src,
                 "index": np.flatnonzero(tr)},
        "_test": {"prob": prob_te, "yhat": yhat_te, "y": y_te,
                  "index": np.flatnonzero(te)},
    }


def run_nested_loso(X, y, sites, seed: int, n_jobs: int = 1,
                    pca_candidates=PCA_CANDIDATES, c_grid=C_GRID,
                    verbose: bool = False) -> list[dict]:
    """All outer folds. Returns one result dict per held-out site."""
    from joblib import Parallel, delayed

    order = sorted(np.unique(sites), key=lambda s: int(s[1:]))
    if n_jobs == 1:
        out = []
        for s in order:
            if verbose:
                print(f"[outer] {s}", flush=True)
            out.append(run_outer_fold(X, y, sites, s, seed,
                                      pca_candidates, c_grid))
        return out
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=5 if verbose else 0)(
        delayed(run_outer_fold)(X, y, sites, s, seed, pca_candidates, c_grid)
        for s in order)


# ---------------------------------------------------------------------------
# within-site stratified label permutation (RQ0 negative control)
# ---------------------------------------------------------------------------
def permute_within_site(y: np.ndarray, sites: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Shuffle diagnosis independently inside each site.

    Site membership, site sample size and site prevalence are preserved by
    construction (a permutation of a site's own label vector).
    """
    y_perm = y.copy()
    for s in np.unique(sites):
        idx = np.flatnonzero(sites == s)
        y_perm[idx] = y[idx][rng.permutation(len(idx))]
    contracts.assert_permutation_within_site(y, y_perm, sites)
    return y_perm
