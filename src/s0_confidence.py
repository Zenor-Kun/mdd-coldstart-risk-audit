"""S0 confidence scores (RQ1/S0, frozen `docs/RQ1_S0_PROTOCOL.md` §3–§5).

Three confidence definitions, all label-blind on the scored partition and all
built strictly inside the S0 information budget of §1:

* **method 1 — probability**  `max(p, 1-p)`, read from the frozen Gate 1 CSVs;
  no model is fitted here (§3).
* **method 2 — bootstrap disagreement**  `-SD_b[p_b(x)]`, `ddof = 0`, over
  `B_boot = 100` site-cluster bootstrap replicates of the legitimate training
  pool (§4).
* **method 3 — Mahalanobis source support**  `-D_M(x)^2` under
  StandardScaler -> PCA50 -> Ledoit-Wolf, fit on the legitimate training pool
  (§5).

The scored partition — the outer held-out site, or the inner held-out source
site `j` — is **never** part of any scaler mean/variance, PCA basis,
covariance or classifier fit produced by this module. That property is asserted
by the callers (`scripts/run_rq1_s0.py`) and by `tests/test_rq1_s0.py`.

No target label, no target population statistic and no threshold is used or
produced here.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

import nested_loso as nl

# --- frozen constants (docs/RQ1_S0_PROTOCOL.md §4, §5, §4.5) ----------------
B_BOOT = 100                       # §4.1, never reduced
BOOT_DDOF = 0                      # §4.4
PCA_RANDOM_STATE = 20260807        # §4.5 — the project's frozen MODEL_SEED
MAHALANOBIS_PCA_DIM = 50           # §5, fixed by the protocol, never tuned

#: `BOOT_SEED(o, j, b) = 20290001 + 10000*o + 200*j + b`, range [20290001, 20514500].
BOOT_SEED_BASE = 20290001
BOOT_SEED_O_STRIDE = 10000
BOOT_SEED_J_STRIDE = 200
BOOT_SEED_MIN = BOOT_SEED_BASE
BOOT_SEED_MAX = BOOT_SEED_BASE + BOOT_SEED_O_STRIDE * 22 + BOOT_SEED_J_STRIDE * 22 + 99

#: Seed ranges consumed by RQ0; the S0 namespace must stay disjoint from them.
RQ0_SEED_RANGES = ((20260807, 20260826), (20270101, 20270300), (90000001, 90000003))


def boot_seed(o: int, j: int, b: int) -> int:
    """Resampling seed for outer-fold index `o`, pool index `j`, replicate `b`.

    `o` is the held-out site's index in `contracts.FROZEN_SITES` (0..22); `j` is
    0 for the target ensemble and 1..22 for the `j`-th inner held-out source
    site in that fold's source-site list, taken in frozen site order; `b` is
    0..99. The map is injective (§4.5) and seeds only the site draw.
    """
    if not 0 <= o <= 22:
        raise ValueError(f"outer fold index out of range: {o}")
    if not 0 <= j <= 22:
        raise ValueError(f"pool index out of range: {j}")
    if not 0 <= b < B_BOOT:
        raise ValueError(f"replicate index out of range: {b}")
    return BOOT_SEED_BASE + BOOT_SEED_O_STRIDE * o + BOOT_SEED_J_STRIDE * j + b


def seed_is_in_s0_namespace(seed: int) -> bool:
    return BOOT_SEED_MIN <= seed <= BOOT_SEED_MAX


def seed_collides_with_rq0(seed: int) -> bool:
    return any(lo <= seed <= hi for lo, hi in RQ0_SEED_RANGES)


# ---------------------------------------------------------------------------
# method 1 — max probability (§3)
# ---------------------------------------------------------------------------
def confidence_prob(prob_mdd) -> np.ndarray:
    """`max(p, 1-p)`; higher = more confident. Range [0.5, 1.0]."""
    p = np.asarray(prob_mdd, dtype=float).ravel()
    if not np.isfinite(p).all():
        raise ValueError("probabilities must be finite")
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return np.maximum(p, 1.0 - p)


# ---------------------------------------------------------------------------
# method 2 — site-cluster bootstrap disagreement (§4)
# ---------------------------------------------------------------------------
def draw_site_cluster_replicate(pool_sites, seed: int) -> np.ndarray:
    """Draw `|P|` sites from `P` uniformly with replacement (§4.1 step 1).

    Returns the drawn site labels **in draw order, with repetitions kept**;
    the caller turns each drawn position into a distinct training cluster so
    that multiplicity survives the site-equal weighting (§4.2 trap).
    """
    pool = np.asarray(pool_sites)
    rng = np.random.default_rng(seed)
    return pool[rng.integers(0, pool.size, size=pool.size)]


def build_replicate_training_set(sites: np.ndarray, drawn_sites):
    """Row indices and per-copy cluster labels for one bootstrap replicate.

    A site drawn `m` times contributes its rows `m` times (§4.1 step 3), and the
    `m`-th copy is relabelled `S#m` so that `nested_loso.site_equal_weights`
    gives **each copy** the mass of one site. Computing `1/n_{s(i)}` on the
    replicated counts instead would cancel the multiplicity exactly, which §4.2
    forbids.
    """
    sites = np.asarray(sites)
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    seen: dict = {}
    for s in drawn_sites:
        copy = seen.get(s, 0)
        seen[s] = copy + 1
        idx = np.flatnonzero(sites == s)
        if idx.size == 0:
            raise ValueError(f"drawn site {s!r} has no acquisitions")
        rows.append(idx)
        labels.append(np.full(idx.size, f"{s}#{copy}", dtype=object))
    return np.concatenate(rows), np.concatenate(labels).astype(str)


def bootstrap_ensemble_probs(X, y, sites, pool_sites, scored_rows,
                             selected_pca, selected_C, o: int, j: int,
                             b_boot: int = B_BOOT) -> np.ndarray:
    """`(b_boot, n_scored)` matrix of ensemble probabilities `p_b(x)` (§4.3).

    Per replicate: site-cluster resample of `pool_sites`, ordinary unweighted
    `StandardScaler` and PCA fits on the resampled rows, ridge logistic
    regression at the fold's **frozen** Gate 1 `(selected_pca, selected_C)` with
    site-equal weights applied per drawn copy. No hyperparameter is re-selected.

    `scored_rows` must be disjoint from every row of every `pool_sites` site —
    the caller asserts this before calling.
    """
    sites = np.asarray(sites)
    pool = np.asarray(pool_sites)
    scored_rows = np.asarray(scored_rows)
    if np.isin(sites[scored_rows], pool).any():
        raise AssertionError("scored partition intersects its own training pool")

    X_scored = X[scored_rows]
    out = np.empty((b_boot, scored_rows.size), dtype=float)
    for b in range(b_boot):
        drawn = draw_site_cluster_replicate(pool, boot_seed(o, j, b))
        rows, clusters = build_replicate_training_set(sites, drawn)
        X_tr = X[rows]
        pre = nl.Preprocessor((selected_pca,)).fit(X_tr, PCA_RANDOM_STATE)
        w = nl.site_equal_weights(clusters)
        out[b] = nl.fit_predict(pre.transform(X_tr, selected_pca), y[rows], w,
                                pre.transform(X_scored, selected_pca),
                                selected_C, PCA_RANDOM_STATE)
        del X_tr, pre
    return out


def confidence_boot(ensemble_probs) -> np.ndarray:
    """`-SD_b[p_b(x)]` with `ddof = 0`; higher = more confident (§4.4)."""
    P = np.asarray(ensemble_probs, dtype=float)
    if P.ndim != 2:
        raise ValueError("expected a (B_boot, n_scored) matrix")
    if P.shape[0] != B_BOOT:
        raise ValueError(f"B_boot must be {B_BOOT}, got {P.shape[0]}")
    return -P.std(axis=0, ddof=BOOT_DDOF)


# ---------------------------------------------------------------------------
# method 3 — Mahalanobis source support (§5)
# ---------------------------------------------------------------------------
def mahalanobis_support(X, fit_rows, scored_rows,
                        n_components: int = MAHALANOBIS_PCA_DIM):
    """`-D_M^2` for `scored_rows` against the support of `fit_rows` (§5).

    StandardScaler -> PCA(50, randomized, `random_state = 20260807`) ->
    `LedoitWolf`, all fit on `fit_rows` only. `Sigma^{-1}` is
    `LedoitWolf().precision_`: no manual regularisation, no pseudo-inverse, no
    added ridge term. Method 3 draws no random sample; given its training set
    the score is fully deterministic.
    """
    fit_rows = np.asarray(fit_rows)
    scored_rows = np.asarray(scored_rows)
    if np.intersect1d(fit_rows, scored_rows).size:
        raise AssertionError("scored rows leaked into the Mahalanobis fit")

    pre = nl.Preprocessor((n_components,)).fit(X[fit_rows], PCA_RANDOM_STATE)
    Z_fit = pre.transform(X[fit_rows], n_components)
    Z_scored = pre.transform(X[scored_rows], n_components)

    mu = Z_fit.mean(axis=0)
    lw = LedoitWolf().fit(Z_fit)
    diff = Z_scored - mu
    d2 = np.einsum("ij,jk,ik->i", diff, lw.precision_, diff)
    return -d2, {"shrinkage": float(lw.shrinkage_), "n_fit": int(fit_rows.size),
                 "pca_dim": int(Z_fit.shape[1]),
                 "mean_d2_fit": float(np.einsum(
                     "ij,jk,ik->i", Z_fit - mu, lw.precision_, Z_fit - mu).mean())}
