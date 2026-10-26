"""Tie-aware expected empirical AURC (RQ1/S0, frozen Stage 1.0.1).

Scope: this module defines the *ranking-quality* metric of
`docs/RQ1_S0_PROTOCOL.md` §10 and nothing else. It runs no S0 experiment,
reads no manifest, no feature cache and no Gate 1 artifact, and knows nothing
about sites, identifiers or diagnosis. It receives two aligned arrays —
confidence and per-acquisition 0/1 loss — and returns a number.

Why tie-aware rather than tie-broken (Stage 1.0.1 Fix 2): REST-meta-MDD
`canonical_id` / `feature_source_id` strings encode diagnosis in the middle
field (`S<site>-<1|2>-<running>`, 1 = MDD, 2 = HC), so ordering tied
acquisitions by identifier would let a label-derived token decide a prediction
ranking. That violates the identity-never-in-X / label-independent-evaluation
principle of `DATA_CONTRACT_v1.md` §7. The exact expectation over tie-internal
orderings is deterministic, needs no identifier and no random draw, and reduces
exactly to the standard empirical AURC when no ties are present.
"""

from __future__ import annotations

import numpy as np


def tie_aware_aurc(confidence, loss) -> float:
    """Expected empirical AURC over all tie-internal orderings.

    Parameters
    ----------
    confidence : array_like, shape (n,)
        Confidence score; higher = more confident. Compared by **exact**
        numeric equality — no tolerance is applied, and none may be added.
    loss : array_like, shape (n,)
        Per-acquisition 0/1 classification loss ``1{y_hat != y}``.

    Returns
    -------
    float
        ``(1/n) * sum over the n positions of the expected cumulative risk``.

    Notes
    -----
    Confidences are grouped into tie blocks by descending unique value. For a
    block of ``m`` acquisitions carrying ``e`` errors, entered with ``n0``
    acquisitions and ``e0`` errors already accumulated, the expected number of
    errors among the first ``j`` tied acquisitions is ``j * e / m`` under a
    uniformly random tie-internal ordering, so position ``j`` contributes an
    expected cumulative risk of ``(e0 + j*e/m) / (n0 + j)``.

    With every block of size 1 this is identically
    ``(1/n) * sum_k cumulative_error_k / k`` — the standard empirical AURC.

    The result depends only on the multiset of ``(confidence, loss)`` pairs: it
    is invariant to row order and carries no dependence on any identifier,
    site, or diagnosis-derived quantity.
    """
    conf = np.asarray(confidence, dtype=float).ravel()
    los = np.asarray(loss, dtype=float).ravel()

    if conf.shape != los.shape:
        raise ValueError("confidence and loss must have the same shape")
    n = conf.size
    if n == 0:
        raise ValueError("AURC is undefined on an empty set")
    if not np.isfinite(conf).all():
        raise ValueError("confidence must be finite")
    if not np.isin(los, (0.0, 1.0)).all():
        raise ValueError("loss must be 0/1")

    total = 0.0
    n0 = 0
    e0 = 0.0
    # np.unique returns ascending exact-equality groups; walk them high to low.
    for level in np.unique(conf)[::-1]:
        block = conf == level
        m = int(block.sum())
        e = float(los[block].sum())
        j = np.arange(1, m + 1, dtype=float)
        total += float(np.sum((e0 + j * e / m) / (n0 + j)))
        n0 += m
        e0 += e
    return total / n


def source_oof_percentile_map(target_confidence, source_oof_confidence):
    """Mid-rank empirical CDF of a fold's source-OOF confidence distribution.

    ``u(c) = [ #(source < c) + 0.5 * #(source == c) ] / N_source_OOF``

    Higher ``u`` = higher relative confidence within that fold. This is the
    only cross-fold pooling map the protocol allows (Stage 1.0.2 Fix 6), and it
    feeds a **sensitivity** analysis only — never a primary AURC.

    Uses the frozen source-OOF scores of the *same* outer fold, so it consumes
    no target information beyond the target confidences being mapped.
    """
    t = np.asarray(target_confidence, dtype=float).ravel()
    s = np.sort(np.asarray(source_oof_confidence, dtype=float).ravel())
    if s.size == 0:
        raise ValueError("source-OOF confidence distribution is empty")
    if not (np.isfinite(t).all() and np.isfinite(s).all()):
        raise ValueError("confidences must be finite")
    lt = np.searchsorted(s, t, side="left")        # # strictly below
    le = np.searchsorted(s, t, side="right")       # # below or equal
    return (lt + 0.5 * (le - lt)) / s.size


def standard_aurc_from_order(loss_in_confidence_order) -> float:
    """Standard empirical AURC for an already-ordered loss vector.

    Reference implementation used by the tests to pin `tie_aware_aurc` down; it
    presupposes a total order and is therefore never used on data with ties.
    """
    los = np.asarray(loss_in_confidence_order, dtype=float).ravel()
    k = np.arange(1, los.size + 1, dtype=float)
    return float(np.mean(np.cumsum(los) / k))
