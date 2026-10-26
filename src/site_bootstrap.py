"""Site-level cluster bootstrap for cross-site inference (RQ1/S0, frozen Stage 1.0.2).

Scope: uncertainty quantification only. This module implements the resampling
scheme, the percentile-interval convention, the `NOT_ESTIMABLE` bookkeeping and
the paired method-contrast rule of `docs/RQ1_S0_PROTOCOL.md` §14. It runs no S0
experiment, reads no manifest, no feature cache and no Gate 1 artifact, and
contains no hypothesis test — RQ1 pre-registers no family-wise testing
procedure (§14.8).

The inference unit is the **held-out site**. Acquisitions inside a site are
never resampled; thresholds `q`, bootstrap classifiers and FC edges are never
inference units.

One 5000 x 23 multiplicity matrix is drawn once, from a fixed SeedSequence, and
reused for every statistic and every confidence method. That is what makes all
method-vs-method contrasts paired by construction: two methods can never see
different site draws.
"""

from __future__ import annotations

import numpy as np

# --- frozen constants (docs/RQ1_S0_PROTOCOL.md §14.1) -----------------------
SITE_BOOTSTRAP_B = 5000
# Independent namespace derived from the project's existing frozen seed.
SITE_BOOTSTRAP_ENTROPY = [20260807, 1102]
CI_LEVEL = 0.95
CI_QUANTILES = (0.025, 0.975)
QUANTILE_METHOD = "linear"
# A percentile interval is reported only if at least 95% of replicates are valid.
MIN_VALID_REPLICATES = 4750
NOT_ESTIMABLE = "NOT_ESTIMABLE"


# --- frozen CI scope (docs/RQ1_S0_PROTOCOL.md §15.8, Stage 1.0.3) -----------
# The protocol's authoritative scope table, mirrored so it can be asserted
# rather than remembered. Closed in both directions: a quantity is in exactly
# one list, and `requires_ci` raises on anything unregistered rather than
# silently defaulting.
CONFIDENCE_METHODS = ("probability", "bootstrap", "mahalanobis")
OPERATING_POINTS = ("OP50", "OP30")

#: The three method pairs, in the frozen `A - B` sign convention.
METHOD_PAIRS = (
    ("probability", "bootstrap"),
    ("probability", "mahalanobis"),
    ("bootstrap", "mahalanobis"),
)
AURC_ESTIMANDS = ("patient_weighted_aurc", "site_balanced_aurc")

#: Exactly six paired AURC contrasts: three per primary AURC estimand.
PAIRED_AURC_CONTRASTS = tuple(
    (estimand, a, b) for estimand in AURC_ESTIMANDS for a, b in METHOD_PAIRS
)

CI_REQUIRED = (
    "patient_weighted_aurc",
    "site_balanced_aurc",
    "paired_aurc_contrast",
    "OP50_site_balanced_coverage",
    "OP50_site_balanced_selective_ber",
    "OP50_descriptive_violation_rate",
    "OP30_site_balanced_coverage",
    "OP30_site_balanced_selective_ber",
    "OP30_descriptive_violation_rate",
)

CI_PROHIBITED = (
    "risk_coverage_curve_17q",
    "per_site_value",
    "worst_site_selective_risk",
    "jsd",
    "risk_gap",
    "percentile_mapped_pooled_aurc_sensitivity",
)

#: The 30 required intervals, enumerated one by one rather than counted by a
#: formula — the arithmetic shortcut is what let a stale "24" survive in the
#: prose (Stage 1.0.3a). Composition is unchanged: 3 patient-weighted AURC +
#: 3 site-balanced AURC + 6 paired contrasts + 9 at OP50 + 9 at OP30.
_OP_QUANTITIES = (
    "site_balanced_coverage",
    "site_balanced_selective_ber",
    "descriptive_violation_rate",
)
REQUIRED_INTERVALS = (
    tuple(("patient_weighted_aurc", m) for m in CONFIDENCE_METHODS)
    + tuple(("site_balanced_aurc", m) for m in CONFIDENCE_METHODS)
    + tuple(("paired_aurc_contrast", c) for c in PAIRED_AURC_CONTRASTS)
    + tuple((f"{op}_{q}", m)
            for op in OPERATING_POINTS
            for q in _OP_QUANTITIES
            for m in CONFIDENCE_METHODS)
)

N_REQUIRED_INTERVALS = len(REQUIRED_INTERVALS)


def requires_ci(quantity: str) -> bool:
    """True iff `quantity` carries a 95% site-bootstrap CI in the S0 report.

    Raises on an unregistered name: there is no "optional" or default-off
    category, so a new quantity must be classified in the protocol before it
    can be emitted.
    """
    if quantity in CI_REQUIRED:
        return True
    if quantity in CI_PROHIBITED:
        return False
    raise KeyError(
        f"{quantity!r} is not in the frozen CI scope table "
        f"(docs/RQ1_S0_PROTOCOL.md §15.8); classify it there first")


def site_bootstrap_counts(n_sites: int, B: int = SITE_BOOTSTRAP_B) -> np.ndarray:
    """The frozen (B, n_sites) matrix of site multiplicities.

    Replicate ``b`` draws ``n_sites`` site indices uniformly with replacement
    from ``range(n_sites)``; ``counts[b, s]`` is how often site ``s`` was drawn,
    so multiplicity is preserved rather than deduplicated. Row sums are exactly
    ``n_sites``.

    Deterministic: the same ``(n_sites, B)`` always yields the same matrix, on
    any machine, with no dependence on call order or on any S0 result.

    Callers must pass site-level vectors in the **frozen eligible-site order**
    (`contracts.FROZEN_SITES`); column ``s`` of this matrix is site ``s`` of
    that order.
    """
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    if B < 1:
        raise ValueError("B must be >= 1")
    seq = np.random.SeedSequence(SITE_BOOTSTRAP_ENTROPY)
    rng = np.random.Generator(np.random.PCG64(seq))
    draws = rng.integers(0, n_sites, size=(B, n_sites), dtype=np.int64)
    counts = np.zeros((B, n_sites), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(B), n_sites), draws.ravel()), 1)
    return counts


def _clean(values, weights, n_sites):
    v = np.asarray(values, dtype=float).ravel()
    if v.size != n_sites:
        raise ValueError(f"expected {n_sites} site values, got {v.size}")
    if weights is None:
        w = np.ones(n_sites, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != n_sites:
            raise ValueError("weights must be one per site")
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
    ok = np.isfinite(v)          # NaN encodes NOT_ESTIMABLE at the site level
    return v, w, ok


def replicate_statistics(values, counts, weights=None) -> np.ndarray:
    """Per-replicate weighted mean over the drawn site entries.

    ``stat_b = sum_s m_bs w_s v_s / sum_s m_bs w_s`` restricted to sites whose
    value is estimable. A draw landing on a `NOT_ESTIMABLE` site enters neither
    numerator nor denominator; a replicate with no estimable draw is `NaN`.

    With ``weights=None`` and every site estimable the denominator is exactly
    ``n_sites``, so this reproduces ``sum_s m_bs v_s / n_sites``. With
    ``weights=n_s`` it reproduces ``sum_s m_bs n_s v_s / sum_s m_bs n_s``.
    """
    counts = np.asarray(counts)
    v, w, ok = _clean(values, weights, counts.shape[1])
    num = counts @ np.where(ok, w * v, 0.0)
    den = counts @ np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out


def point_estimate(values, weights=None) -> float:
    """Observed statistic: weighted mean over the estimable sites, each once."""
    v, w, ok = _clean(values, weights, np.asarray(values, float).ravel().size)
    den = w[ok].sum()
    return float((w[ok] * v[ok]).sum() / den) if den > 0 else float("nan")


def percentile_ci(replicates, min_valid: int = MIN_VALID_REPLICATES) -> dict:
    """95% percentile interval, or `NOT_ESTIMABLE` if too few valid replicates.

    No BCa, no normal approximation, no bias correction, and no switching
    between interval types on the basis of a result.
    """
    r = np.asarray(replicates, dtype=float).ravel()
    valid = r[np.isfinite(r)]
    out = {
        "total_bootstrap_replicates": int(r.size),
        "valid_bootstrap_replicates": int(valid.size),
        "ci_method": f"{int(CI_LEVEL * 100)}% percentile "
                     f"(numpy.quantile method='{QUANTILE_METHOD}')",
    }
    if valid.size < min_valid:
        out["ci_low"] = NOT_ESTIMABLE
        out["ci_high"] = NOT_ESTIMABLE
        return out
    lo, hi = np.quantile(valid, CI_QUANTILES, method=QUANTILE_METHOD)
    out["ci_low"] = float(lo)
    out["ci_high"] = float(hi)
    return out


def bootstrap_summary(values, counts, weights=None,
                      min_valid: int = MIN_VALID_REPLICATES) -> dict:
    """Point estimate + estimability ledger + percentile CI for one method."""
    v = np.asarray(values, dtype=float).ravel()
    rep = replicate_statistics(v, counts, weights)
    out = {
        "point_estimate": point_estimate(v, weights),
        "n_estimable_sites": int(np.isfinite(v).sum()),
        "n_sites": int(v.size),
    }
    out.update(percentile_ci(rep, min_valid))
    return out


def paired_contrast(values_a, values_b, counts, weights=None,
                    min_valid: int = MIN_VALID_REPLICATES) -> dict:
    """Paired A − B contrast on the **same** site draws and the common set.

    A site enters only if the metric is estimable under *both* methods
    (`common_estimable_sites = estimable_A & estimable_B`); the restriction is
    applied to both arms before anything is averaged, so the contrast is never
    a difference between two means taken over different site sets.
    """
    a = np.asarray(values_a, dtype=float).ravel()
    b = np.asarray(values_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("the two methods must supply one value per site")
    common = np.isfinite(a) & np.isfinite(b)
    a_c = np.where(common, a, np.nan)
    b_c = np.where(common, b, np.nan)

    rep = replicate_statistics(a_c, counts, weights) - \
        replicate_statistics(b_c, counts, weights)
    out = {
        "point_estimate": point_estimate(a_c, weights) - point_estimate(b_c, weights),
        "n_common_estimable_sites": int(common.sum()),
        "n_estimable_sites_A": int(np.isfinite(a).sum()),
        "n_estimable_sites_B": int(np.isfinite(b).sum()),
        "n_sites": int(a.size),
    }
    out.update(percentile_ci(rep, min_valid))
    return out
