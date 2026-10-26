"""Pooled-local shrinkage — an EMPIRICAL few-label baseline (RQ2/S2 §12-§13).

**Stage 2.0.1 correction.** Stage 2.0 described the quantity below as a
"Bayesian upper credible bound under a source-centred prior". A methodological
audit rejected that description: the shape parameters do not correspond to a
properly specified source-centred prior updated by the local data, so the
Bayesian reading is a misdescription rather than a weaker guarantee. The
estimator is unchanged; the *claim* is removed.

What this module now provides is a **deterministic empirical screening score**:

    empirical_pseudocount_beta_upper_score
        = Beta.ppf(0.95, x_c + n0*r_src + 1, (n_c - x_c) + n0*(1 - r_src))

with, stated plainly and enforced by tests:

* no exact frequentist coverage guarantee;
* no nominal alpha;
* no family-wise error control;
* **not** a posterior credible interval from a properly specified prior;
* no inheritance argument of any kind from the `n0 = 0` endpoint.

The `n0 = 0` identity with exact Clopper-Pearson is retained only as a
numerical sanity check on the expression — an **endpoint identity**. It is not
evidence that `n0 > 0` inherits Clopper-Pearson coverage, and this module must
never be read as claiming so.

Two things this module also does not do: it does not transplant the
finite-sample corrections of the segmentation conformal-risk-control
literature, and it does not feed its fractional pseudo-counts into the exact
Clopper-Pearson path (which raises on a non-integer count by construction).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta

from certification import (ALPHA_CLASS, GUARANTEE_BY_SCORE,
                           SCORE_EMPIRICAL_PSEUDOCOUNT)

# ---------------------------------------------------------------------------
# frozen constants (§12.5, §13.2) — unchanged by Stage 2.0.1
# ---------------------------------------------------------------------------
N0_GRID = (5, 10, 20, 50, 100)
R_META = 20
META_SPLIT_ENTROPY = [20260807, 2001]
N0_TIE_BREAK = "smallest"

assert len(N0_GRID) == len(set(N0_GRID)) == 5
assert 0 not in N0_GRID, "n0 = 0 would duplicate target_local (§12.5)"
assert tuple(sorted(N0_GRID)) == N0_GRID, "n0 grid must be ascending"


# ---------------------------------------------------------------------------
# §12.2 — the shrunk class-risk curve (a deterministic blend, nothing more)
# ---------------------------------------------------------------------------
def shrinkage_lambda(n_local: int, n0: float) -> float:
    """`lambda_c = n_c / (n_c + n0)` — the local weight in the blend.

    v4 writes `lambda_local = N / (N + n0)`; the label budget is replaced by
    the per-class retained count at that candidate, which is the quantity that
    governs the local estimate's precision. This removes a free parameter: the
    adaptation to the label budget is automatic, so one `n0` serves all `N`.
    """
    n_local, n0 = float(n_local), float(n0)
    if n_local < 0 or n0 < 0:
        raise ValueError("n_local and n0 must be >= 0")
    if n_local + n0 <= 0:
        raise ValueError("n_local + n0 must be > 0")
    return n_local / (n_local + n0)


def shrunk_class_risk(x_local: int, n_local: int, r_source: float,
                      n0: float) -> float:
    """`(x_c + n0 * r_src) / (n_c + n0)` = `lam * r_loc + (1 - lam) * r_src`.

    A deterministic convex blend of two observed risk curves. It is a point
    summary, not an estimator with a stated sampling distribution.
    """
    _validate(x_local, n_local, r_source, n0)
    return ((float(x_local) + float(n0) * float(r_source))
            / (float(n_local) + float(n0)))


def _validate(x_local, n_local, r_source, n0) -> None:
    x, n = int(x_local), int(n_local)
    if x < 0 or n < 0 or x > n:
        raise ValueError(f"invalid local counts x={x_local}, n={n_local}")
    if not 0.0 <= float(r_source) <= 1.0:
        raise ValueError(f"source class risk {r_source!r} outside [0, 1]")
    if float(n0) < 0:
        raise ValueError("n0 must be >= 0")


# ---------------------------------------------------------------------------
# §12.3 — the empirical screening score
# ---------------------------------------------------------------------------
def empirical_pseudocount_beta_upper_score(x_local: int, n_local: int,
                                           r_source: float, n0: float,
                                           alpha: float = ALPHA_CLASS) -> dict:
    """A deterministic pseudo-count Beta upper score. **No guarantee.**

    ``Beta.ppf(1 - alpha, x_c + n0*r_src + 1, (n_c - x_c) + n0*(1 - r_src))``.

    Read it as: a monotone score that starts from the local error count, pulls
    it toward the source class risk with strength `n0`, and adds a
    Beta-shaped upper allowance whose width falls as `n_c + n0` grows. That is
    all it is. It is not a confidence bound, not a credible interval, and it
    carries no nominal level: `alpha` here selects a fixed quantile of a fixed
    expression, and no coverage statement follows from it.

    A degenerate shape parameter yields `score = 1` (nothing can pass) and
    never removes the candidate from the sequence.
    """
    _validate(x_local, n_local, r_source, n0)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    x, n, n0 = int(x_local), int(n_local), float(n0)
    a = x + n0 * float(r_source) + 1.0
    b = (n - x) + n0 * (1.0 - float(r_source))
    if b <= 0.0 or a <= 0.0 or not np.isfinite(a) or not np.isfinite(b):
        score = 1.0
    else:
        score = float(beta.ppf(1.0 - alpha, a, b))
    return {
        "score": score,
        "a": a,
        "b": b,
        "n0": n0,
        "lambda_local": shrinkage_lambda(n, n0),
        "r_shrunk": shrunk_class_risk(x, n, r_source, n0),
        "r_local": (x / n) if n > 0 else float("nan"),
        "r_source": float(r_source),
        "n_local": n,
        "x_local": x,
        "score_type": SCORE_EMPIRICAL_PSEUDOCOUNT,
        "guarantee": GUARANTEE_BY_SCORE[SCORE_EMPIRICAL_PSEUDOCOUNT],
    }


def endpoint_identity_check(x_local: int, n_local: int, r_source: float,
                            alpha: float = ALPHA_CLASS) -> float:
    """Numerical sanity check: at `n0 = 0` the expression equals exact CP.

    **Endpoint identity only.** It says the algebra at one boundary agrees with
    a known quantity; it establishes nothing whatsoever about the behaviour or
    the coverage of the score at the `n0` values actually used, all of which
    are strictly positive. Any inheritance argument from this identity is
    invalid and is forbidden in the protocol, the code and the report.
    """
    from certification import clopper_pearson_upper
    got = empirical_pseudocount_beta_upper_score(x_local, n_local, r_source,
                                                 0.0, alpha)["score"]
    return abs(got - clopper_pearson_upper(x_local, n_local, alpha))


#: §24 — a certified-adjacent diagnostic threshold for *reporting only*: it
#: enters no eligibility test, no score and no selection.
PRIOR_DOMINANCE_LAMBDA = 0.25
#: The reporting name for the same idea, without the prior vocabulary.
SOURCE_DOMINATED_LAMBDA = PRIOR_DOMINANCE_LAMBDA


# ---------------------------------------------------------------------------
# §13 — source meta-LOSO selection of n0 (unchanged; wording corrected)
# ---------------------------------------------------------------------------
def meta_split_permutation(outer_index: int, pseudo_index: int, repeat: int,
                           n_pseudo_target: int) -> np.ndarray:
    """The frozen meta-world permutation.

    Deterministic in `(outer_index, pseudo_index, repeat)`, machine-independent,
    with no dependence on call order, no global RNG and no `spawn()`.
    """
    for name, v in (("outer_index", outer_index), ("pseudo_index", pseudo_index),
                    ("repeat", repeat)):
        if int(v) < 0:
            raise ValueError(f"{name} must be >= 0")
    seq = np.random.SeedSequence(
        META_SPLIT_ENTROPY + [int(outer_index), int(pseudo_index), int(repeat)])
    rng = np.random.Generator(np.random.PCG64(seq))
    return rng.permutation(int(n_pseudo_target))


def meta_objective(terms_by_n0: dict) -> dict:
    """`d(s, n0) = sqrt(mean of (r_shrunk - r_ref)^2)`; argmin, ties -> smallest.

    A pure out-of-sample prediction-error criterion for the blend: it selects
    where the blended curve sits, using only source-side sites. It contains no
    prior, no posterior, no certification count and no target outcome. Every
    `n0` candidate must contribute the identical number of terms — eligibility
    and the reference risk are both independent of `n0`, so a mismatch means
    the caller has let `n0` influence the term set.
    """
    missing = sorted(set(N0_GRID) - set(terms_by_n0))
    if missing:
        raise AssertionError(f"meta objective missing n0 candidates: {missing}")
    extra = sorted(set(terms_by_n0) - set(N0_GRID))
    if extra:
        raise AssertionError(f"meta objective has off-grid n0 candidates: {extra}")

    sizes = {n0: len(np.asarray(terms_by_n0[n0]).ravel()) for n0 in N0_GRID}
    if len(set(sizes.values())) != 1:
        raise AssertionError(
            "the n0 candidates were scored on different numbers of terms "
            f"({sizes}); §13.3 requires a common support because eligibility "
            "and the reference risk are both independent of n0")
    if next(iter(sizes.values())) == 0:
        raise AssertionError("meta objective has no terms")

    d = {n0: float(np.sqrt(np.mean(np.asarray(terms_by_n0[n0], dtype=float) ** 2)))
         for n0 in N0_GRID}
    best = min(d.values())
    selected = next(n0 for n0 in N0_GRID if d[n0] <= best)
    return {
        "d_by_n0": d,
        "selected_n0": int(selected),
        "n_terms": int(next(iter(sizes.values()))),
        "n_ties_at_argmin": int(sum(1 for n0 in N0_GRID if d[n0] == best)),
    }
