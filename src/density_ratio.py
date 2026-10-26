"""S1 density-ratio utilities (RQ1/S1, frozen `docs/RQ1_S1_PROTOCOL.md`).

Scope: the *only* new mechanism S1 introduces — importance-weighted source OOF
calibration under a target-unlabeled information budget. This module

* fits the 50-D domain representation and the binary domain classifier (§3, §4);
* turns its probabilities into density-ratio weights with the frozen prior
  correction, the frozen normalize-then-clip order, and nothing after it (§5);
* reports the effective sample size and the weight diagnostics (§6);
* selects one `C_domain` per outer fold from a label-free, cross-fitted
  source meta-LOSO objective (§7);
* provides the weighted quantile, the weighted class-balanced selective error,
  and the S1 cross-site CI registry (§9, §10, §13).

Deliberately absent, and unreachable from here (§12, §17): any upper confidence
bound, any ordered testing procedure, threshold *adoption*, the certification
vocabulary, any target-label argument, any RNG or seed of its own, and any
effective-sample-size exclusion rule.

**No diagnosis label is an argument to any function in the domain-fitting path.**
The response of the domain classifier is the design label `D` (0 = source,
1 = target), constructed inside this module from the shapes of the two matrices
it is given.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from s0_evaluation import NO_ABSTENTION, OPERATING_POINTS, Q_GRID  # noqa: F401

NOT_ESTIMABLE = "NOT_ESTIMABLE"

# --- frozen constants (docs/RQ1_S1_PROTOCOL.md §3, §4, §5, §7) --------------
#: Density-ratio representation. Fixed by the protocol; not a grid, never tuned.
DOMAIN_PCA_DIM = 50
#: The project's existing frozen MODEL_SEED. S1 introduces no seed of its own.
PCA_RANDOM_STATE = 20260807
DOMAIN_MAX_ITER = 2000
#: Numerical guard on the domain probability before the odds are formed (§4.2).
P_D_EPS = 1e-6
#: v4 §RQ1 item 3. Normalize, then clip, then stop (§5).
W_CLIP_LOW = 0.1
W_CLIP_HIGH = 10.0
#: Frozen before any S1 result; may not be extended (§7.1).
C_DOMAIN_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
#: All three S0 methods are carried into S1; none may be dropped (§15).
CONFIDENCE_METHODS = ("probability", "bootstrap", "mahalanobis")


# ---------------------------------------------------------------------------
# input guards
# ---------------------------------------------------------------------------
def assert_domain_matrix(X) -> np.ndarray:
    """The domain path accepts a bare finite float matrix and nothing else.

    A labelled table cannot reach a fit through this door: anything carrying
    column names — identity, `site`, `diagnosis`, phenotype — is rejected here
    rather than silently converted (`DATA_CONTRACT_v1.md` §7).
    """
    if hasattr(X, "columns") or hasattr(X, "dtypes"):
        raise TypeError("the domain path takes a bare ndarray, not a table")
    A = np.asarray(X)
    if A.dtype.kind not in "fc" and A.dtype.kind != "i":
        raise TypeError(f"non-numeric domain matrix: dtype {A.dtype}")
    A = A.astype(float, copy=False)
    if A.ndim != 2:
        raise ValueError(f"domain matrix must be 2-D, got ndim {A.ndim}")
    if not np.isfinite(A).all():
        raise ValueError("domain matrix contains NaN or Inf")
    return A


# ---------------------------------------------------------------------------
# §3 domain representation
# ---------------------------------------------------------------------------
@dataclass
class DomainRepresentation:
    """StandardScaler -> PCA50, fit on the allowed `X` of one estimation context.

    Fit pools are frozen by §3.1: the *deployed* representation of fold `s` sees
    the 22 source sites and the target site; the *meta* representation sees the
    22 source sites only. Diagnosis enters neither.
    """

    n_components: int = DOMAIN_PCA_DIM
    random_state: int = PCA_RANDOM_STATE
    scaler: StandardScaler = field(default=None, repr=False)
    pca: PCA = field(default=None, repr=False)

    def fit(self, X_allowed) -> "DomainRepresentation":
        A = assert_domain_matrix(X_allowed)
        self.scaler = StandardScaler().fit(A)
        Z = self.scaler.transform(A)
        k = min(self.n_components, min(Z.shape) - 1)
        self.pca = PCA(n_components=k, svd_solver="randomized",
                       random_state=self.random_state).fit(Z)
        return self

    def transform(self, X) -> np.ndarray:
        A = assert_domain_matrix(X)
        return self.pca.transform(self.scaler.transform(A))


def fit_domain_representation(X_allowed, n_components: int = DOMAIN_PCA_DIM
                              ) -> DomainRepresentation:
    """Fit the frozen density-ratio representation on the allowed `X` (§3)."""
    return DomainRepresentation(n_components=n_components).fit(X_allowed)


# ---------------------------------------------------------------------------
# §4 domain classifier and the density ratio
# ---------------------------------------------------------------------------
def domain_probabilities(Z_source_fit, Z_target_fit, Z_score, C: float
                         ) -> np.ndarray:
    """`p_D(x) = P(D = 1 | x)` on `Z_score`, guarded to `[eps, 1 - eps]`.

    The response is built here from the two matrices' row counts —
    `D = 0` for every row of `Z_source_fit`, `D = 1` for every row of
    `Z_target_fit`. There is no label argument, so no diagnosis can reach the
    fit. Trained on the **actual** sample frequencies: `class_weight` is `None`,
    because re-balancing would absorb the prior that `raw_importance_ratio`
    then corrects for (§4.4).
    """
    S = assert_domain_matrix(Z_source_fit)
    T = assert_domain_matrix(Z_target_fit)
    Q = assert_domain_matrix(Z_score)
    if S.shape[1] != T.shape[1] or S.shape[1] != Q.shape[1]:
        raise ValueError("source, target and scored matrices differ in width")
    if len(S) == 0 or len(T) == 0:
        raise ValueError("both domain classes must be non-empty")
    X = np.vstack([S, T])
    D = np.concatenate([np.zeros(len(S), dtype=int), np.ones(len(T), dtype=int)])
    clf = LogisticRegression(penalty="l2", C=float(C), solver="lbfgs",
                             max_iter=DOMAIN_MAX_ITER, class_weight=None,
                             random_state=PCA_RANDOM_STATE)
    clf.fit(X, D)
    p = clf.predict_proba(Q)[:, 1]
    return np.clip(p, P_D_EPS, 1.0 - P_D_EPS)


def raw_importance_ratio(p_d, n_source: int, n_target: int) -> np.ndarray:
    """`w_raw = [p_D/(1-p_D)] * (n_source/n_target)` (§4.3).

    `n_source` and `n_target` are the two domain-class sizes **of the fit that
    produced `p_d`** — for a cross-fitted meta sub-fit that is the sub-fit's own
    pool, never the outer fold's. The factor converts the mixture odds into the
    density ratio `p_T(x)/p_S(x)`.
    """
    p = np.asarray(p_d, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("empty probability vector")
    if not np.isfinite(p).all() or (p <= 0).any() or (p >= 1).any():
        raise ValueError("p_D must be finite and strictly inside (0, 1); "
                         "apply the eps guard first")
    if n_source < 1 or n_target < 1:
        raise ValueError("both domain classes must be non-empty")
    return (p / (1.0 - p)) * (float(n_source) / float(n_target))


def normalize_then_clip(w_raw) -> np.ndarray:
    """The frozen weight pipeline: normalize to mean 1, clip, **stop** (§5).

    v4 §RQ1 item 3 fixes the order as normalize-then-clip and names `[0.1, 10]`
    as the state of the weights afterwards, so no renormalization follows the
    clip. Every S1 estimand is invariant to a positive rescaling of `w` (§5.3),
    so the omission is a bookkeeping fact rather than a scientific choice.
    """
    w = np.asarray(w_raw, dtype=float).ravel()
    if w.size == 0:
        raise ValueError("empty weight vector")
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("raw weights must be finite and strictly positive")
    m = float(w.mean())
    if not np.isfinite(m) or m <= 0:
        raise ValueError("degenerate weight mean")
    return np.clip(w / m, W_CLIP_LOW, W_CLIP_HIGH)


def deployed_source_weights(Z_source, Z_target, C: float) -> np.ndarray:
    """Weights for a fold's own source rows, from one fit on source vs target.

    The transductive, in-sample estimator of v4: one domain classifier per outer
    fold, scoring the same source acquisitions it was fitted on. Cross-fitting
    is a requirement on the *disease* prediction and confidence score (frozen in
    S0), not on the domain classifier (§8).
    """
    p = domain_probabilities(Z_source, Z_target, Z_source, C)
    return normalize_then_clip(
        raw_importance_ratio(p, n_source=len(np.asarray(Z_source)),
                             n_target=len(np.asarray(Z_target))))


# ---------------------------------------------------------------------------
# §6 effective sample size and weight diagnostics
# ---------------------------------------------------------------------------
def effective_sample_size(w) -> float:
    """`ESS = (sum w)^2 / sum w^2` — a diagnostic, and only a diagnostic (§6).

    Scale-invariant, equal to `n` for constant weights, and never used to drop,
    filter or re-select anything (§14).
    """
    a = np.asarray(w, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("empty weight vector")
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("weights must be finite and non-negative")
    denom = float((a ** 2).sum())
    if denom <= 0:
        raise ValueError("all weights are zero")
    return float(a.sum() ** 2 / denom)


def weight_diagnostics(w) -> dict:
    """The mandatory per-fold weight report of §6.

    Returns descriptive statistics only. There is no exclusion flag, no
    threshold and no verdict field, because §14 pre-registers no
    effective-sample-size cut-off of any kind.
    """
    a = np.asarray(w, dtype=float).ravel()
    ess = effective_sample_size(a)
    return {
        "n_source": int(a.size),
        "ess": ess,
        "ess_fraction": float(ess / a.size),
        "w_min": float(a.min()),
        "w_median": float(np.median(a)),
        "w_mean": float(a.mean()),
        "w_max": float(a.max()),
        "frac_clipped_low": float(np.mean(a == W_CLIP_LOW)),
        "frac_clipped_high": float(np.mean(a == W_CLIP_HIGH)),
    }


# ---------------------------------------------------------------------------
# §7 source meta-LOSO selection of C_domain
# ---------------------------------------------------------------------------
def meta_pseudo_folds(sites) -> tuple:
    """The pseudo-target sites of a fold's source meta-LOSO, in frozen order.

    `sites` carries the **source** sites of one outer fold only; the real outer
    target never appears in it, and this function never sees a diagnosis.
    """
    return tuple(np.unique(np.asarray(sites).astype(str)).tolist())


def crossfit_oof_raw_weights(Z, sites, pseudo_target_site: str, C: float
                             ) -> np.ndarray:
    """Out-of-fold raw weights over the pseudo-source pool of one pseudo-fold.

    For each held-out pseudo-source site `h`, the domain classifier is fit on
    `D = 1`: the pseudo-target site, `D = 0`: the pseudo-source sites other than
    `h`; `h` is then scored by a fit that never saw it (§7.2). Returns one raw
    weight per pseudo-source row, aligned to the rows of `Z` for which
    `sites != pseudo_target_site`.
    """
    A = assert_domain_matrix(Z)
    s = np.asarray(sites).astype(str)
    if s.size != len(A):
        raise ValueError("one site label per row")
    j = str(pseudo_target_site)
    target = A[s == j]
    if len(target) == 0:
        raise ValueError(f"pseudo-target site {j!r} has no rows")
    src_mask = s != j
    src_sites = s[src_mask]
    Z_src = A[src_mask]
    held_out = np.unique(src_sites)
    if held_out.size < 2:
        raise ValueError("meta-LOSO needs at least two pseudo-source sites")

    w_raw = np.empty(len(Z_src), dtype=float)
    for h in held_out:
        score_mask = src_sites == h
        fit_mask = ~score_mask
        p = domain_probabilities(Z_src[fit_mask], target, Z_src[score_mask], C)
        w_raw[score_mask] = raw_importance_ratio(
            p, n_source=int(fit_mask.sum()), n_target=len(target))
    return w_raw


def iw_mean_discrepancy(Z_source, w, Z_target) -> float:
    """The §7.3 objective for one pseudo-fold: `d(j, C)`.

    The density ratio is defined by `E_S[w f] = E_T[f]`; this evaluates that
    moment condition on the 50 representation coordinates, standardized by the
    pseudo-source SD so no single component dominates. Weights enter only
    through `sum(w z)/sum(w)`, so the value is invariant to their scale.
    """
    S = assert_domain_matrix(Z_source)
    T = assert_domain_matrix(Z_target)
    a = np.asarray(w, dtype=float).ravel()
    if a.size != len(S):
        raise ValueError("one weight per source row")
    if not np.isfinite(a).all() or (a < 0).any() or a.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    if S.shape[1] != T.shape[1]:
        raise ValueError("source and target representations differ in width")
    m_w = (a[:, None] * S).sum(axis=0) / a.sum()
    m_t = T.mean(axis=0)
    sd = S.std(axis=0, ddof=0)
    if (sd <= 0).any():
        raise ValueError("a representation coordinate is constant on the "
                         "pseudo-source pool")
    return float(np.sqrt(np.mean(((m_w - m_t) / sd) ** 2)))


def meta_objective(Z, sites, C: float) -> float:
    """`J(C)` — the mean cross-fitted discrepancy over the pseudo-folds (§7.3).

    Label-free: `Z` and site membership are its only inputs, which is why one
    `C_domain` serves all three confidence methods.
    """
    A = assert_domain_matrix(Z)
    s = np.asarray(sites).astype(str)
    vals = []
    for j in meta_pseudo_folds(s):
        src_mask = s != j
        w_raw = crossfit_oof_raw_weights(A, s, j, C)
        w = normalize_then_clip(w_raw)
        vals.append(iw_mean_discrepancy(A[src_mask], w, A[s == j]))
    return float(np.mean(vals))


def unweighted_discrepancy(Z, sites) -> float:
    """`mean_j d(j, w = 1)` — the descriptive `no-weighting` reference (§7.3).

    Reported beside `J(C)`; it has **no** selection role and is never a fallback
    (§7.5).
    """
    A = assert_domain_matrix(Z)
    s = np.asarray(sites).astype(str)
    vals = []
    for j in meta_pseudo_folds(s):
        src = A[s != j]
        vals.append(iw_mean_discrepancy(src, np.ones(len(src)), A[s == j]))
    return float(np.mean(vals))


def select_c_domain(Z, sites, c_grid=C_DOMAIN_GRID) -> dict:
    """One `C_domain` for one outer fold: `argmin J(C)`, ties -> smallest `C`.

    The grid is frozen (§7.1) and carries no `no-weighting` entry: the selection
    always returns a member of it, even where weighting does not improve the
    source-side match (§7.5).
    """
    grid = tuple(sorted(float(c) for c in c_grid))
    if not grid:
        raise ValueError("empty C_domain grid")
    objective = {c: meta_objective(Z, sites, c) for c in grid}
    best = grid[0]
    for c in grid[1:]:
        if objective[c] < objective[best]:      # strict: ties keep the smaller C
            best = c
    return {
        "c_domain": best,
        "objective": objective,
        "tie_break": "smallest_C",
        "n_candidates": len(grid),
    }


# ---------------------------------------------------------------------------
# §9 weighted source thresholds
# ---------------------------------------------------------------------------
def weighted_quantile(values, weights, q: float) -> float:
    """The frozen weighted quantile of §9.2 — inverse weighted ECDF, no interpolation.

    Ascending sort; exact-equal values aggregated into blocks (exact numeric
    equality, no tolerance); cumulative **normalized** block weights with the
    last entry set to exactly 1.0; `tau = v_{k*}`, `k* = min{k : F_k >= q}`.

    Deterministic, invariant to row order and to any within-tie ordering, and
    free of every identifier — the ID encodes diagnosis, so it may never order a
    calibration set.
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("empty value vector")
    if v.size != w.size:
        raise ValueError("one weight per value")
    if not np.isfinite(v).all():
        raise ValueError("values must be finite")
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("weights must be finite and strictly positive")
    qq = float(q)
    if not 0.0 <= qq <= 1.0:
        raise ValueError(f"q out of range: {qq}")

    uniq, inv = np.unique(v, return_inverse=True)          # ascending, exact ties
    block_w = np.bincount(inv, weights=w, minlength=uniq.size)
    F = np.cumsum(block_w) / block_w.sum()
    F[-1] = 1.0                                            # exact top of the CDF
    k = int(np.searchsorted(F, qq, side="left"))
    k = min(k, uniq.size - 1)
    return float(uniq[k])


def weighted_source_thresholds(confidence, weights, q_grid=Q_GRID) -> dict:
    """`tau_q^S1` for the 17 frozen `q`, from source scores and weights only.

    The target confidence distribution is not an argument and cannot become one:
    a threshold in this project is a source-side object (§9.4).
    """
    return {q: weighted_quantile(confidence, weights, q) for q in q_grid}


def weighted_retained_mass(confidence, weights, tau: float) -> float:
    """Realized **weighted** source coverage at a threshold — an audit column."""
    c = np.asarray(confidence, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if c.size != w.size:
        raise ValueError("one weight per confidence")
    keep = c >= float(tau)
    return float(w[keep].sum() / w.sum())


# ---------------------------------------------------------------------------
# §10 weighted source selective balanced error
# ---------------------------------------------------------------------------
def weighted_selective_ber(y_true, y_pred, keep, weights):
    """`0.5 * (err_MDD^w + err_HC^w)` on retained source rows, or `NOT_ESTIMABLE`.

    The two class error rates are formed **separately** and then averaged.
    Pooling all retained rows into one weight-weighted error rate is forbidden:
    that quantity is not the balanced error of v4 §4.2.1.

    Zero retained weight in either class -> `NOT_ESTIMABLE` (equivalently, zero
    retained rows of that class, since every weight is >= 0.1 after clipping).
    No pseudo-count, no continuity correction, no imputation.
    """
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(y_pred).ravel().astype(int)
    k = np.asarray(keep).ravel().astype(bool)
    w = np.asarray(weights, dtype=float).ravel()
    if not (y.size == p.size == k.size == w.size):
        raise ValueError("y_true, y_pred, keep and weights must align")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and non-negative")

    yk, pk, wk = y[k], p[k], w[k]
    is_mdd, is_hc = yk == 1, yk == 0
    W_mdd, W_hc = float(wk[is_mdd].sum()), float(wk[is_hc].sum())
    ledger = {
        "n_keep": int(k.sum()),
        "n_keep_MDD": int(is_mdd.sum()),
        "n_keep_HC": int(is_hc.sum()),
        "W_MDD": W_mdd,
        "W_HC": W_hc,
    }
    if W_mdd <= 0 or W_hc <= 0:
        ledger["estimable"] = False
        return NOT_ESTIMABLE, ledger
    err_mdd = float(wk[is_mdd][pk[is_mdd] != 1].sum() / W_mdd)
    err_hc = float(wk[is_hc][pk[is_hc] != 0].sum() / W_hc)
    ledger["estimable"] = True
    ledger["err_MDD_weighted"] = err_mdd
    ledger["err_HC_weighted"] = err_hc
    return 0.5 * (err_mdd + err_hc), ledger


# ---------------------------------------------------------------------------
# §12 the target AURC invariant
# ---------------------------------------------------------------------------
def assert_target_aurc_invariant(aurc_s0, aurc_s1) -> None:
    """S1 changes no target confidence, so the target AURC is bit-identical (§12).

    S1 emits no AURC of its own; this guard exists so that a recomputation can
    be checked against the frozen S0 values with an **exact** zero difference.
    """
    a = np.asarray(aurc_s0, dtype=float).ravel()
    b = np.asarray(aurc_s1, dtype=float).ravel()
    if a.shape != b.shape:
        raise AssertionError("AURC vectors differ in length")
    if not np.array_equal(a, b):
        worst = float(np.max(np.abs(a - b)))
        raise AssertionError(
            f"target AURC moved between S0 and S1 (max |diff| = {worst:.3e}); "
            "S1 may not change a target ranking")


# ---------------------------------------------------------------------------
# §13.3 the S1 cross-site CI scope, mirrored so it can be asserted
# ---------------------------------------------------------------------------
_OP_QUANTITIES = (
    "site_balanced_coverage",
    "site_balanced_selective_ber",
    "descriptive_violation_rate",
)

S1_CI_REQUIRED = (
    tuple(f"S1_{op}_{q}" for op in OPERATING_POINTS for q in _OP_QUANTITIES)
    + tuple(f"S1_minus_S0_{op}_{q}"
            for op in OPERATING_POINTS for q in _OP_QUANTITIES)
)

S1_CI_PROHIBITED = (
    "s1_risk_coverage_curve_17q",
    "per_site_value",
    "worst_site_selective_risk",
    "jsd",
    "risk_gap",
    "s1_minus_s0_risk_gap",
    "weighted_source_selective_ber",
    "weighted_source_selective_ber_site_balanced",
    "ess",
    "weight_diagnostic",
    "meta_selection_objective",
    "tau_equal_weight_check",
    "aurc",
)

#: 36 = (2 operating points x 3 quantities x 3 methods) standalone S1
#:      + the same 18 as paired S1 - S0 contrasts.
S1_REQUIRED_INTERVALS = tuple(
    (quantity, method)
    for quantity in S1_CI_REQUIRED
    for method in CONFIDENCE_METHODS
)
N_S1_REQUIRED_INTERVALS = len(S1_REQUIRED_INTERVALS)


def requires_ci(quantity: str) -> bool:
    """True iff `quantity` carries a 95% site-bootstrap CI in the S1 report.

    Raises on an unregistered name: S1 has no "optional" or default-off
    category, so a new quantity must be classified in the protocol first.
    """
    if quantity in S1_CI_REQUIRED:
        return True
    if quantity in S1_CI_PROHIBITED:
        return False
    raise KeyError(
        f"{quantity!r} is not in the frozen S1 CI scope table "
        f"(docs/RQ1_S1_PROTOCOL.md §13.3); classify it there first")
