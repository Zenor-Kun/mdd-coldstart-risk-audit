"""S0 deployment evaluation (RQ1/S0, frozen `docs/RQ1_S0_PROTOCOL.md` §6–§13).

Source-derived thresholds, selective balanced error with its zero-class
`NOT_ESTIMABLE` rule, diagnosis-composition JSD, per-site AURC and the two
primary AURC aggregates.

Deliberately absent, and unreachable from this module (§8, §12):
Clopper-Pearson or any other upper confidence bound, fixed-sequential or any
ordered testing procedure, threshold *adoption*, and the words
certified / uncertified. Those belong to RQ2.
"""

from __future__ import annotations

import numpy as np

from selective_aurc import source_oof_percentile_map, tie_aware_aurc

NOT_ESTIMABLE = "NOT_ESTIMABLE"

# --- frozen threshold grid (§6) ---------------------------------------------
#: 17 candidates, 0.80 down to 0.00 in steps of 0.05.
Q_GRID = tuple(round(0.80 - 0.05 * i, 2) for i in range(17))
NO_ABSTENTION = "NO_ABSTENTION"
QUANTILE_METHOD = "linear"

# --- frozen operating points (§7), source-*intended* coverage ---------------
OP50_Q = 0.50      # source intended coverage ~50%
OP30_Q = 0.30      # source intended coverage ~70%
OPERATING_POINTS = {"OP50": OP50_Q, "OP30": OP30_Q}

assert OP50_Q in Q_GRID and OP30_Q in Q_GRID
assert len(Q_GRID) == 17 and len(set(Q_GRID)) == 17


def source_oof_thresholds(source_oof_confidence, q_grid=Q_GRID) -> dict:
    """`tau_q` = the source-OOF confidence quantile, for each of the 17 `q` (§6).

    The **only** admissible threshold source in S0. No target confidence, no
    target coverage level and no target class proportion enters this function —
    its single argument is the fold's own source-OOF score vector.
    """
    c = np.asarray(source_oof_confidence, dtype=float).ravel()
    if c.size == 0:
        raise ValueError("empty source-OOF confidence distribution")
    if not np.isfinite(c).all():
        raise ValueError("source-OOF confidence must be finite")
    return {q: float(np.quantile(c, q, method=QUANTILE_METHOD)) for q in q_grid}


def retained_mask(confidence, tau) -> np.ndarray:
    """`keep(x) = 1{confidence >= tau}` (§6).

    `>=` keeps an exact-tie block together on the retained side. A tie block
    straddling `tau` makes realized coverage jump; that jump is reported as the
    coverage it produces and is never resolved by splitting the block.
    """
    c = np.asarray(confidence, dtype=float).ravel()
    return c >= float(tau)


# ---------------------------------------------------------------------------
# selective balanced error (§9)
# ---------------------------------------------------------------------------
def selective_ber(y_true, y_pred, keep):
    """`0.5 * (err_MDD + err_HC)` on retained acquisitions, or `NOT_ESTIMABLE`.

    Returns `(ber_or_NOT_ESTIMABLE, ledger)` where the ledger always carries
    `n_keep`, `n_keep_MDD`, `n_keep_HC` and `estimable`, including on the
    `NOT_ESTIMABLE` branch. No pseudo-count, no continuity correction, no
    substitution of plain error rate, no imputed value.
    """
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(y_pred).ravel().astype(int)
    k = np.asarray(keep).ravel().astype(bool)
    yk, pk = y[k], p[k]
    n_mdd = int((yk == 1).sum())
    n_hc = int((yk == 0).sum())
    ledger = {"n_keep": int(k.sum()), "n_keep_MDD": n_mdd, "n_keep_HC": n_hc}
    if n_mdd == 0 or n_hc == 0:
        ledger["estimable"] = False
        return NOT_ESTIMABLE, ledger
    err_mdd = float((pk[yk == 1] != 1).mean())
    err_hc = float((pk[yk == 0] != 0).mean())
    ledger["estimable"] = True
    ledger["err_MDD"] = err_mdd
    ledger["err_HC"] = err_hc
    return 0.5 * (err_mdd + err_hc), ledger


def as_float(value) -> float:
    """`NOT_ESTIMABLE` -> `nan`, which is how estimability travels into arrays."""
    return float("nan") if value == NOT_ESTIMABLE else float(value)


def site_balanced_source_ber(y_true, y_pred, keep, source_sites):
    """§8 secondary: per-source-site BER, then an equal-weight mean.

    Sites whose retained set lacks a class are not estimable and are excluded
    from the mean rather than imputed; the count is returned alongside.
    """
    s = np.asarray(source_sites)
    vals = []
    for site in np.unique(s):
        m = s == site
        ber, _ = selective_ber(np.asarray(y_true)[m], np.asarray(y_pred)[m],
                               np.asarray(keep)[m])
        if ber != NOT_ESTIMABLE:
            vals.append(ber)
    if not vals:
        return NOT_ESTIMABLE, 0
    return float(np.mean(vals)), len(vals)


# ---------------------------------------------------------------------------
# JSD on diagnosis composition (§13)
# ---------------------------------------------------------------------------
def jsd_composition(n_keep_mdd: int, n_keep_hc: int, n_all_mdd: int,
                    n_all_hc: int):
    """Jensen-Shannon divergence, **base 2**, between retained and full composition.

    Only the `[MDD, HC]` composition of the current target site is involved —
    no age, sex, motion or site variable enters (§13). An empty retained set is
    `NOT_ESTIMABLE`, never 0. A retained set with one class absent but
    `n_keep > 0` is a legitimate, estimable, maximal-shift JSD, even where §9
    marks the same cell's selective BER `NOT_ESTIMABLE`.
    """
    n_keep = int(n_keep_mdd) + int(n_keep_hc)
    n_all = int(n_all_mdd) + int(n_all_hc)
    if n_keep == 0:
        return NOT_ESTIMABLE
    if n_all == 0:
        raise ValueError("the site has no acquisitions")
    P = np.array([n_keep_mdd / n_keep, n_keep_hc / n_keep], dtype=float)
    Q = np.array([n_all_mdd / n_all, n_all_hc / n_all], dtype=float)
    M = 0.5 * (P + Q)

    def _kl(a, m):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / m[nz])))

    return max(0.0, 0.5 * _kl(P, M) + 0.5 * _kl(Q, M))


# ---------------------------------------------------------------------------
# AURC aggregates (§10.2, §10.3)
# ---------------------------------------------------------------------------
def per_site_aurc(confidence, loss) -> float:
    """`AURC_s` — the frozen tie-aware AURC, ranked **within one held-out site**.

    Takes only `(confidence, loss)`: no identifier, site or diagnosis token can
    reach the ranking (§10.1, Stage 1.0.1 Fix 2).
    """
    return tie_aware_aurc(confidence, loss)


def patient_weighted_aurc(aurc_per_site, n_per_site) -> float:
    """`sum_s n_s AURC_s / sum_s n_s` (§10.2)."""
    a = np.asarray(aurc_per_site, dtype=float).ravel()
    n = np.asarray(n_per_site, dtype=float).ravel()
    if a.shape != n.shape:
        raise ValueError("one AURC and one acquisition count per site")
    return float((n * a).sum() / n.sum())


def site_balanced_aurc(aurc_per_site) -> float:
    """`mean_s AURC_s` (§10.2)."""
    a = np.asarray(aurc_per_site, dtype=float).ravel()
    return float(a.mean())


def percentile_mapped_pooled_aurc(fold_target_confidence, fold_source_oof_confidence,
                                  fold_loss) -> float:
    """The **sole** pooled sensitivity of §10.3 — point estimate, never a CI.

    Each fold's target confidences are mapped through that fold's own source-OOF
    mid-rank ECDF, the mapped scores of all 23 held-out sites are pooled, and
    the pooled set is scored with the frozen tie-aware AURC. Each acquisition
    enters exactly once.

    Raw pooled confidence AURC is `DELETED / NOT PLANNED` (§18 F1) and has no
    implementation anywhere in this codebase.
    """
    u_all, loss_all = [], []
    for key in fold_target_confidence:
        u_all.append(source_oof_percentile_map(fold_target_confidence[key],
                                               fold_source_oof_confidence[key]))
        loss_all.append(np.asarray(fold_loss[key], dtype=float).ravel())
    return tie_aware_aurc(np.concatenate(u_all), np.concatenate(loss_all))
