"""S1 evaluation helpers (RQ1/S1, frozen `docs/RQ1_S1_PROTOCOL.md`).

Scope: the pieces of the S1 arm that are *not* the density ratio itself —

* the weighted **site-balanced** source selective BER (§10 secondary);
* the equal-weight threshold audit of §9.3;
* an independent re-implementation of the §7.3 discrepancy, used to audit the
  standardization denominator (which pool, which sites, no diagnosis);
* the exact-registry check behind the §13.3 interval scope.

The target-diagnosis firewall used to live here as a single global barrier. It
was too weak to support the claim it made and has been replaced by the
outer-fold-scoped `TargetOutcomeGuard` in `src/s1_data_access.py`; nothing in
this module touches an outcome artifact.

Deliberately absent and unreachable from here, exactly as in `s0_evaluation`
and `density_ratio` (§12, §17): any upper confidence bound, any ordered
multiple-testing procedure, threshold *adoption*, the vocabulary of RQ2, any
target-label argument, and any effective-sample-size exclusion rule.

Every S1 quantity defined here is a function of frozen S0 artifacts, source
labels and the density-ratio weights. No new seed, no RNG.
"""

from __future__ import annotations

import hashlib

import numpy as np

import density_ratio as dr
from s0_evaluation import NOT_ESTIMABLE

__all__ = [
    "NOT_ESTIMABLE",
    "assert_interval_registry_exact",
    "assert_pca50",
    "equal_weight_tau_audit",
    "reference_iw_discrepancy",
    "sha256_file",
    "weighted_site_balanced_source_ber",
]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# §13.3 — the interval registry, checked exactly and in one place
# ---------------------------------------------------------------------------
def assert_interval_registry_exact(emitted, required=dr.S1_REQUIRED_INTERVALS,
                                   expected_n: int = 36) -> int:
    """The emitted intervals are the frozen 36, no more and no fewer.

    Written as three separate comparisons on purpose. The first version of the
    provenance script spelled this `len(iv) != N != 36`, which Python evaluates
    as `(len(iv) != N) and (N != 36)` — with `N == 36` the right conjunct is
    always false, so the guard could never fire however many intervals were
    emitted. A count check that cannot fail is worse than none.
    """
    req = sorted((str(q), str(k)) for q, k in required)
    got = sorted((str(q), str(k)) for q, k in emitted)
    if len(req) != expected_n:
        raise AssertionError(
            f"the frozen registry enumerates {len(req)} intervals, not "
            f"{expected_n} (docs/RQ1_S1_PROTOCOL.md §13.3)")
    if len(got) != expected_n:
        raise AssertionError(
            f"{len(got)} intervals emitted, expected exactly {expected_n}")
    if got != req:
        extra = sorted(set(got) - set(req))
        missing = sorted(set(req) - set(got))
        raise AssertionError(
            f"emitted intervals differ from the frozen registry; "
            f"extra={extra[:4]} missing={missing[:4]}")
    return expected_n


# ---------------------------------------------------------------------------
# §3, §5 — the PCA50 guard the formal run must not fall back from
# ---------------------------------------------------------------------------
def assert_pca50(representation, context: str) -> int:
    """Every formal S1 domain fit is exactly 50-D — no defensive fallback.

    `DomainRepresentation.fit` clamps `n_components` to what a degenerate pool
    could support. On the frozen 2232-acquisition cohort that clamp is never
    active, so in the formal run a fitted dimension other than 50 means the
    input pool was wrong. It is a hard failure here, never a silent reduction.
    """
    k = int(representation.pca.n_components_)
    if k != dr.DOMAIN_PCA_DIM:
        raise AssertionError(
            f"{context}: domain PCA fitted {k} components, not "
            f"{dr.DOMAIN_PCA_DIM}; the formal cohort must never reach the "
            "low-rank fallback")
    return k


# ---------------------------------------------------------------------------
# §10 secondary — the site-balanced weighted source risk
# ---------------------------------------------------------------------------
def weighted_site_balanced_source_ber(y_true, y_pred, keep, weights,
                                      source_sites):
    """Per-source-site weighted BER, then an equal-weight mean over sites.

    Mirrors `s0_evaluation.site_balanced_source_ber` with the density-ratio
    weights in place of counts. A site whose retained set lacks a class is not
    estimable and is left out of the mean rather than imputed; the surviving
    denominator travels with the value. Weights are not renormalized within a
    site — every weighted class error is a ratio of weight sums, so the value is
    invariant to the scale (§5.3).
    """
    s = np.asarray(source_sites)
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    k = np.asarray(keep)
    w = np.asarray(weights, dtype=float)
    vals = []
    for site in np.unique(s):
        m = s == site
        ber, _ = dr.weighted_selective_ber(y[m], p[m], k[m], w[m])
        if ber != NOT_ESTIMABLE:
            vals.append(ber)
    if not vals:
        return NOT_ESTIMABLE, 0
    return float(np.mean(vals)), len(vals)


# ---------------------------------------------------------------------------
# §9.3 — the equal-weight threshold audit
# ---------------------------------------------------------------------------
def equal_weight_tau_audit(confidence, q: float, tau_s0: float) -> dict:
    """Measure the S0/S1 quantile-convention gap instead of arguing about it.

    S0 uses `numpy.quantile(method="linear")` (interpolating, Type 7); S1 uses
    the inverse weighted ECDF (step). The frozen claim of §9.3 is that the two
    differ by **at most one order statistic**. That claim is checked here as a
    property of the observed confidence multiset: no source-OOF confidence lies
    strictly between the two thresholds. It is a diagnostic — it may not adjust
    either convention, and neither threshold is recomputed from it.
    """
    c = np.asarray(confidence, dtype=float).ravel()
    tau_eq = dr.weighted_quantile(c, np.ones(c.size), q)
    lo, hi = (tau_eq, float(tau_s0)) if tau_eq <= tau_s0 else (float(tau_s0), tau_eq)
    strictly_between = int(np.count_nonzero((c > lo) & (c < hi)))
    return {
        "tau_equal_weight": float(tau_eq),
        "tau_equal_weight_check": float(tau_eq) - float(tau_s0),
        "tau_equal_weight_is_observed_value": bool(np.any(c == tau_eq)),
        "n_source_values_strictly_between": strictly_between,
        "within_one_order_statistic": strictly_between == 0,
    }


# ---------------------------------------------------------------------------
# §7.3 / §9 of the run brief — the discrepancy denominator audit
# ---------------------------------------------------------------------------
def reference_iw_discrepancy(Z_pseudo_source, w, Z_pseudo_target) -> dict:
    """An independent transcription of `d(j, C)`, written from the protocol.

    Exists so the implementation in `density_ratio.iw_mean_discrepancy` can be
    checked against the frozen formula rather than against itself, and so the
    **standardization denominator** is visible: the population SD (`ddof = 0`)
    of each of the 50 meta-representation coordinates over the *pseudo-source*
    pool only — the pseudo-target site is excluded from it, the real outer
    target is absent from the entire meta world, and no diagnosis enters.
    """
    S = np.asarray(Z_pseudo_source, dtype=float)
    T = np.asarray(Z_pseudo_target, dtype=float)
    a = np.asarray(w, dtype=float).ravel()
    m_w = np.array([float((a * S[:, d]).sum() / a.sum()) for d in range(S.shape[1])])
    m_t = np.array([float(T[:, d].mean()) for d in range(T.shape[1])])
    sd = np.array([float(np.sqrt(((S[:, d] - S[:, d].mean()) ** 2).mean()))
                   for d in range(S.shape[1])])
    d = float(np.sqrt(np.mean(((m_w - m_t) / sd) ** 2)))
    return {
        "d": d,
        "n_denominator_rows": int(S.shape[0]),
        "n_dimensions": int(S.shape[1]),
        "denominator_pool": "pseudo_source_pool_only",
        "denominator_ddof": 0,
        "diagnosis_used": False,
    }
