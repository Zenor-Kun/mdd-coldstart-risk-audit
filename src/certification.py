"""Selective-risk decision primitives (RQ2/S2, `docs/RQ2_S2_PROTOCOL.md`).

**Stage 2.0.1 guarantee-semantics correction.** Stage 2.0 gave all four
calibration strategies a single `CERTIFIED` state and a single certification
vocabulary. A methodological audit rejected that: an approximate
importance-weight score and a pseudo-count Beta score do not justify the same
nominal frequentist statement as an exact class-conditional Clopper-Pearson
bound on integer target counts. The scientific constants are unchanged; the
*claims* are now tiered.

Three guarantee tiers, and no generic `CERTIFIED` state anywhere:

===========================  ==========================  =========================
tier                         strategy                    states
===========================  ==========================  =========================
formal_target_certification  target_local                TARGET_CERTIFIED /
                                                         TARGET_UNCERTIFIED
source_reference             pooled_source               SOURCE_REFERENCE_PASS /
                                                         SOURCE_REFERENCE_FAIL
empirical_screening          target_unlabeled_weighted,  EMPIRICAL_PASS /
                             pooled_local_shrinkage      EMPIRICAL_NO_PASS
===========================  ==========================  =========================

`NOT_TESTABLE` is shared: no candidate satisfied `c_min` / `m_min`, so nothing
was tested under any tier.

Only the formal tier may use: `reject H_k`, FWER, the nominal `alpha_UCB = 0.10`
comparison, "minimum certifiable N", and the word *certification*. The module
enforces that by construction — `state_for` raises if a strategy tries to emit a
state outside its tier, and `clopper_pearson_upper` raises on a non-integer
count so no approximate quantity can reach the exact path.

This module fits no model, reads no artifact and touches no file.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta

# ---------------------------------------------------------------------------
# frozen scientific constants — UNCHANGED by the Stage 2.0.1 correction
# ---------------------------------------------------------------------------
ALPHA_UCB = 0.10
ALPHA_CLASS = 0.05
ALPHA_OVERALL = 0.10          # §23.1 single-bound sensitivity, no Bonferroni

C_MIN = 0.20
M_MIN = 3                     # EXPERIMENT_LOG.md §6, frozen 2026-08-07

Q_GRID = tuple(round(0.80 - 0.05 * i, 2) for i in range(17))
Q_GRID_SENSITIVITY = tuple(round(0.80 - 0.02 * i, 2) for i in range(41))

N_GRID = (5, 10, 20, 40)
EXPLORATORY_N = 5
PRIMARY_N = (10, 20, 40)
R_SPLIT = 100

# ---------------------------------------------------------------------------
# guarantee tiers (Stage 2.0.1 §1-§4)
# ---------------------------------------------------------------------------
TIER_FORMAL = "formal_target_certification"
TIER_SOURCE_REFERENCE = "source_reference"
TIER_EMPIRICAL = "empirical_screening"

STRATEGIES = (
    "pooled_source",              # zero target labels, source-side exact CP
    "target_unlabeled_weighted",  # zero target labels, empirical screening
    "target_local",               # N target labels, FORMAL certification
    "pooled_local_shrinkage",     # N target labels, empirical screening
)
ZERO_LABEL_STRATEGIES = ("pooled_source", "target_unlabeled_weighted")
#: The two strategies whose decision consumes target calibration labels, and
#: therefore the only two that run inside an isolated decision cell.
LABEL_CONSUMING_STRATEGIES = ("target_local", "pooled_local_shrinkage")

STRATEGY_TIER = {
    "target_local": TIER_FORMAL,
    "pooled_source": TIER_SOURCE_REFERENCE,
    "target_unlabeled_weighted": TIER_EMPIRICAL,
    "pooled_local_shrinkage": TIER_EMPIRICAL,
}

#: The single strategy permitted to make a formal target-risk statement.
FORMAL_STRATEGY = "target_local"

CONFIDENCE_METHODS = ("probability", "bootstrap", "mahalanobis")

# ---------------------------------------------------------------------------
# the state enum — there is deliberately no generic CERTIFIED token
# ---------------------------------------------------------------------------
TARGET_CERTIFIED = "TARGET_CERTIFIED"
TARGET_UNCERTIFIED = "TARGET_UNCERTIFIED"
SOURCE_REFERENCE_PASS = "SOURCE_REFERENCE_PASS"
SOURCE_REFERENCE_FAIL = "SOURCE_REFERENCE_FAIL"
EMPIRICAL_PASS = "EMPIRICAL_PASS"
EMPIRICAL_NO_PASS = "EMPIRICAL_NO_PASS"
NOT_TESTABLE = "NOT_TESTABLE"

#: (pass state, fail state) per tier; `NOT_TESTABLE` is legal under all three.
TIER_STATES = {
    TIER_FORMAL: (TARGET_CERTIFIED, TARGET_UNCERTIFIED),
    TIER_SOURCE_REFERENCE: (SOURCE_REFERENCE_PASS, SOURCE_REFERENCE_FAIL),
    TIER_EMPIRICAL: (EMPIRICAL_PASS, EMPIRICAL_NO_PASS),
}
ALL_STATES = tuple(s for pair in TIER_STATES.values() for s in pair) + (NOT_TESTABLE,)
PASS_STATES = tuple(pair[0] for pair in TIER_STATES.values())

#: Why nothing was selected. `NO_ELIGIBLE_CANDIDATE` goes with `NOT_TESTABLE`;
#: `NO_PASSING_CANDIDATE` goes with the tier's fail state.
REASON_NO_ELIGIBLE = "NO_ELIGIBLE_CANDIDATE"
REASON_NO_PASS = "NO_PASSING_CANDIDATE"

# ---------------------------------------------------------------------------
# score vocabulary — one label per tier, carried on every emitted row
# ---------------------------------------------------------------------------
SCORE_EXACT_CP = "exact_classwise_clopper_pearson_ucb"
SCORE_EMPIRICAL_ESS = "empirical_ess_upper_score"
SCORE_EMPIRICAL_PSEUDOCOUNT = "empirical_pseudocount_beta_upper_score"

GUARANTEE_BY_SCORE = {
    SCORE_EXACT_CP: "EXACT_CP_ON_ITS_OWN_CALIBRATION_SAMPLE",
    SCORE_EMPIRICAL_ESS: "EMPIRICAL_NO_COVERAGE_GUARANTEE",
    SCORE_EMPIRICAL_PSEUDOCOUNT: "EMPIRICAL_NO_COVERAGE_GUARANTEE",
}
STRATEGY_SCORE_TYPE = {
    "target_local": SCORE_EXACT_CP,
    "pooled_source": SCORE_EXACT_CP,
    "target_unlabeled_weighted": SCORE_EMPIRICAL_ESS,
    "pooled_local_shrinkage": SCORE_EMPIRICAL_PSEUDOCOUNT,
}

#: Only this strategy may be compared to the nominal one-sided level.
NOMINAL_ALPHA_COMPARABLE = (FORMAL_STRATEGY,)

#: The realized-outcome field name each strategy is allowed to emit (§7).
REALIZED_FIELD = {
    "target_local": "formal_realized_violation",
    "pooled_source": "transported_target_risk_exceedance",
    "target_unlabeled_weighted": "post_pass_target_risk_exceedance",
    "pooled_local_shrinkage": "post_pass_target_risk_exceedance",
}

NOT_ESTIMABLE = "NOT_ESTIMABLE"

assert len(Q_GRID) == 17 == len(set(Q_GRID))
assert len(Q_GRID_SENSITIVITY) == 41 == len(set(Q_GRID_SENSITIVITY))
assert Q_GRID[0] == 0.80 and Q_GRID[-1] == 0.00, "sequence direction moved"
assert abs(ALPHA_CLASS * 2 - ALPHA_UCB) < 1e-12, "Bonferroni split broken"
assert set(STRATEGY_TIER) == set(STRATEGIES) == set(STRATEGY_SCORE_TYPE)
assert set(REALIZED_FIELD) == set(STRATEGIES)
assert sum(1 for t in STRATEGY_TIER.values() if t == TIER_FORMAL) == 1


# ---------------------------------------------------------------------------
# the guarantee-state type system
# ---------------------------------------------------------------------------
def tier_of(strategy: str) -> str:
    if strategy not in STRATEGY_TIER:
        raise KeyError(f"{strategy!r} is not a frozen S2 strategy")
    return STRATEGY_TIER[strategy]


def allowed_states(strategy: str) -> tuple:
    """The complete set of states `strategy` may ever emit."""
    return TIER_STATES[tier_of(strategy)] + (NOT_TESTABLE,)


def assert_state_allowed(strategy: str, state: str) -> None:
    """Raise if a strategy emits a state outside its guarantee tier.

    This is the barrier that stops `target_unlabeled_weighted` or
    `pooled_local_shrinkage` from ever reporting `TARGET_CERTIFIED`, and stops
    `pooled_source` from converting an exact *source-side* decision into a
    target-risk certification.
    """
    if state not in ALL_STATES:
        raise KeyError(f"{state!r} is not a frozen S2 state token")
    if state not in allowed_states(strategy):
        raise AssertionError(
            f"{strategy!r} (tier {tier_of(strategy)}) may not emit {state!r}; "
            f"its states are {allowed_states(strategy)} "
            "(docs/RQ2_S2_PROTOCOL.md §0.2, §15)")


def state_for(strategy: str, passed: bool, testable: bool = True) -> str:
    """The tier-correct state token for one decision cell."""
    if not testable:
        return NOT_TESTABLE
    pass_state, fail_state = TIER_STATES[tier_of(strategy)]
    return pass_state if passed else fail_state


def is_pass(state: str) -> bool:
    """True for the *pass* state of any tier. Deliberately tier-agnostic.

    A caller that needs "did this certify?" must ask
    `is_target_certified`, which is true for the formal tier only.
    """
    if state not in ALL_STATES:
        raise KeyError(f"{state!r} is not a frozen S2 state token")
    return state in PASS_STATES


def is_target_certified(state: str) -> bool:
    """True only for `TARGET_CERTIFIED` — the one formal certification state."""
    if state not in ALL_STATES:
        raise KeyError(f"{state!r} is not a frozen S2 state token")
    return state == TARGET_CERTIFIED


def assert_nominal_comparison_allowed(strategy: str) -> None:
    """Only the formal tier may be set against the nominal `alpha_UCB = 0.10`."""
    if strategy not in NOMINAL_ALPHA_COMPARABLE:
        raise AssertionError(
            f"{strategy!r} may not be compared to the nominal alpha_UCB; that "
            "statement belongs to formal target-local certification alone "
            "(docs/RQ2_S2_PROTOCOL.md §7)")


def realized_outcome_field(strategy: str) -> str:
    """The only realized-outcome column name this strategy may emit (§7)."""
    return REALIZED_FIELD[strategy]


# ---------------------------------------------------------------------------
# the exact class-conditional Clopper-Pearson bound (formal + source-side)
# ---------------------------------------------------------------------------
def _require_integer(value, name: str) -> int:
    """Reject anything that is not an exact non-negative integer count."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer count, got a bool")
    if isinstance(value, (int, np.integer)):
        out = int(value)
    elif isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            raise TypeError(
                f"{name}={value!r} is not an integer count; exact "
                "Clopper-Pearson requires integer binomial counts, and no "
                "empirical score may reach this path "
                "(docs/RQ2_S2_PROTOCOL.md §10, §11, §12)")
        out = int(value)
    else:
        raise TypeError(f"{name} must be an integer count, got {type(value)}")
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {out}")
    return out


def clopper_pearson_upper(x, n, alpha: float = ALPHA_CLASS) -> float:
    """One-sided exact Clopper-Pearson upper bound on a binomial proportion.

    ``U = 1`` when ``x == n``; otherwise ``Beta.ppf(1 - alpha, x + 1, n - x)``.

    Integer counts only — a non-integer raises, which is what keeps the Kish
    effective size of §11 and the pseudo-counts of §12 out of the exact path.
    """
    x = _require_integer(x, "x")
    n = _require_integer(n, "n")
    if x > n:
        raise ValueError(f"x={x} exceeds n={n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if x == n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, x + 1, n - x))


def _mean_pair(a: float, b: float, name_a: str, name_b: str) -> float:
    for name, v in ((name_a, a), (name_b, b)):
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError(f"{name}={v!r} outside [0, 1]")
    return 0.5 * (float(a) + float(b))


def u_ber(u_mdd: float, u_hc: float) -> float:
    """`U_BER = 0.5 (U_MDD + U_HC)` — the FORMAL certification quantity.

    Both arguments must be exact Clopper-Pearson class bounds. The Bonferroni
    split is `alpha_UCB = 0.10` over the two classes, `0.05` each.
    """
    return _mean_pair(u_mdd, u_hc, "U_MDD", "U_HC")


def mean_class_score(s_mdd: float, s_hc: float) -> float:
    """The empirical analogue of `U_BER` — same arithmetic, no guarantee.

    Deliberately a different function with a different name, so that an
    empirical screening score can never be printed under the symbol `U_BER`.
    """
    return _mean_pair(s_mdd, s_hc, "score_MDD", "score_HC")


def overall_error_upper(x, n, alpha: float = ALPHA_OVERALL) -> float:
    """§23.1 sensitivity: one exact bound on the pooled retained error rate.

    One-sided, `alpha = 0.10`, no Bonferroni. Formal only where its inputs are
    formal: on `target_local` it is a formal sensitivity, on the empirical
    strategies its analogue remains an empirical sensitivity.
    """
    return clopper_pearson_upper(x, n, alpha)


# ---------------------------------------------------------------------------
# §11 — the weighted EMPIRICAL screening score (no guarantee, no CP claim)
# ---------------------------------------------------------------------------
def kish_effective_size(weights) -> float:
    """`(sum w)^2 / sum w^2` — a diagnostic effective size, not a trial count."""
    w = np.asarray(weights, dtype=float).ravel()
    if w.size == 0:
        return 0.0
    if (w < 0).any() or not np.isfinite(w).all():
        raise ValueError("weights must be finite and non-negative")
    s2 = float((w ** 2).sum())
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0


def empirical_ess_upper_score(weights, wrong, alpha: float = ALPHA_CLASS) -> dict:
    """A CP-**shaped** empirical screening score for the weighted strategy.

    A self-normalized importance-weighted risk under estimated weights is not a
    binomial proportion. This quantity substitutes a Kish effective size into
    the Clopper-Pearson functional form and is therefore:

    * **not** an exact Clopper-Pearson bound,
    * **not** a confidence bound of any kind,
    * carrying **no** nominal `alpha` and **no** family-wise guarantee.

    It exists to order and screen candidates on a comparable scale. Both
    roundings (`floor` on the effective size, `ceil` on the effective error
    count) are conservative, which makes the score harder to pass, not valid.
    """
    w = np.asarray(weights, dtype=float).ravel()
    e = np.asarray(wrong).ravel().astype(bool)
    if w.shape != e.shape:
        raise ValueError("one weight and one error flag per retained row")
    if (w < 0).any() or not np.isfinite(w).all():
        raise ValueError("weights must be finite and non-negative")

    s1 = float(w.sum())
    n_eff = kish_effective_size(w)
    err_w = float(w[e].sum() / s1) if s1 > 0 else float("nan")

    n_tilde = int(math.floor(n_eff))
    if n_tilde < 1 or not math.isfinite(err_w):
        score, x_tilde = 1.0, n_tilde
    else:
        x_tilde = min(n_tilde, int(math.ceil(err_w * n_tilde)))
        # the CP functional form, evaluated on effective counts. This is the
        # score's *shape*, not a coverage claim (see the docstring).
        score = clopper_pearson_upper(x_tilde, n_tilde, alpha)

    return {
        "score": float(score),
        "n_keep": int(w.size),
        "weight_sum": s1,
        "n_eff": n_eff,
        "n_eff_floor": n_tilde,
        "x_eff_ceil": int(x_tilde),
        "err_weighted": err_w,
        "ess_ratio": (n_eff / w.size) if w.size else float("nan"),
        "score_type": SCORE_EMPIRICAL_ESS,
        "guarantee": GUARANTEE_BY_SCORE[SCORE_EMPIRICAL_ESS],
    }


# ---------------------------------------------------------------------------
# §8 — candidate eligibility (UNCHANGED by Stage 2.0.1)
# ---------------------------------------------------------------------------
def candidate_eligible(coverage_cal: float, class_counts, c_min: float = C_MIN,
                       m_min: int = M_MIN) -> bool:
    """`coverage >= c_min` and every supplied raw class count `>= m_min`.

    Effective sizes and pseudo-counts never enter here: they are not monotone
    in the retained set, and a non-monotone eligibility rule would break the
    suffix property.
    """
    if not math.isfinite(float(coverage_cal)):
        raise ValueError("calibration coverage must be finite")
    if float(coverage_cal) < float(c_min):
        return False
    for n_c in class_counts:
        if _require_integer(n_c, "n_keep_class") < int(m_min):
            return False
    return True


def assert_eligible_is_suffix(eligible) -> int:
    """The eligible candidates must be a contiguous **suffix** of the order.

    Returns the index of the first eligible candidate, or `len(eligible)` if
    none is. An eligible candidate followed by an ineligible one would mean the
    prefix filter had become a mid-sequence skip; the nesting of the retained
    sets makes that impossible, so it is a hard failure.
    """
    flags = [bool(f) for f in eligible]
    first = next((i for i, f in enumerate(flags) if f), len(flags))
    if not all(flags[first:]):
        bad = [i for i in range(first, len(flags)) if not flags[i]]
        raise AssertionError(
            "eligibility is not a contiguous suffix of the frozen candidate "
            f"order; ineligible candidates at {bad} follow an eligible one "
            "(docs/RQ2_S2_PROTOCOL.md §8.5)")
    return first


# ---------------------------------------------------------------------------
# the shared sequential mechanism, with tier-specific vocabulary on top
# ---------------------------------------------------------------------------
def _run_sequence(q_values, eligible, score_fn, reference: float,
                  strategy: str) -> dict:
    """The mechanical core: identical order and stop rule for every tier.

    The *mechanism* is shared so the four strategies remain comparable on a
    common candidate order. The *claim* is not shared: the caller-facing
    wrappers below attach tier-correct states and vocabulary.
    """
    qs = list(q_values)
    flags = [bool(f) for f in eligible]
    if len(qs) != len(flags):
        raise ValueError("one eligibility flag per candidate")
    if list(qs) != sorted(qs, reverse=True):
        raise ValueError(
            "candidates are not in the frozen lowest-coverage-to-highest order "
            "(q descending); reordering is forbidden")

    first = assert_eligible_is_suffix(flags)
    out = {
        "strategy": strategy,
        "guarantee_tier": tier_of(strategy),
        "score_type": STRATEGY_SCORE_TYPE[strategy],
        "guarantee": GUARANTEE_BY_SCORE[STRATEGY_SCORE_TYPE[strategy]],
        "state": None,
        "reason": None,
        "selected_index": None,
        "selected_q": None,
        "n_eligible": len(qs) - first,
        "first_eligible_index": first if first < len(qs) else None,
        "n_passed_candidates": 0,
        "evaluated_indices": [],
        "stopped_at_index": None,
        "score_by_index": {},
    }
    if first >= len(qs):
        out["state"] = state_for(strategy, False, testable=False)
        out["reason"] = REASON_NO_ELIGIBLE
        return out

    selected = None
    for i in range(first, len(qs)):
        s = float(score_fn(i))
        out["evaluated_indices"].append(i)
        out["score_by_index"][i] = s
        if s <= float(reference):
            selected = i
            out["n_passed_candidates"] += 1
            continue
        out["stopped_at_index"] = i
        break

    if selected is None:
        out["state"] = state_for(strategy, False)
        out["reason"] = REASON_NO_PASS
        return out
    out["state"] = state_for(strategy, True)
    out["selected_index"] = selected
    out["selected_q"] = qs[selected]
    return out


def certify_fixed_sequence(q_values, eligible, u_ber_fn, r_star: float) -> dict:
    """**FORMAL** fixed-sequential certification — `target_local` only.

    Null hypotheses `H_k : Risk_target_sel(tau_k) > r*_s`, tested from the
    lowest-coverage candidate upward. `H_k` is **rejected** iff
    `U_BER <= r*_s`; testing continues only while `H_k` is rejected and stops
    at the **first failure to reject**; the selected threshold is the **last
    rejected** candidate. This is the only place in S2 where the words
    *reject*, *FWER* and *certification* apply.

    `u_ber_fn(i)` must return an exact class-conditional `U_BER` (§10) and is
    called lazily, so `evaluated_indices` proves nothing after the stop was
    looked at.
    """
    out = _run_sequence(q_values, eligible, u_ber_fn, r_star, FORMAL_STRATEGY)
    out["n_rejected_hypotheses"] = out["n_passed_candidates"]
    out["nominal_alpha_ucb"] = ALPHA_UCB
    assert_state_allowed(FORMAL_STRATEGY, out["state"])
    return out


def source_reference_sequence(q_values, eligible, u_ber_fn,
                              r_star: float) -> dict:
    """Source-side fixed sequence for `pooled_source`.

    Exact Clopper-Pearson applies to the **source calibration sample**. No
    exchangeability or transport argument to an unseen target site has been
    established, so the outcome is a *source reference*, never a target
    certification, and its target-side behaviour is reported descriptively.
    """
    out = _run_sequence(q_values, eligible, u_ber_fn, r_star, "pooled_source")
    out["scope"] = "source_calibration_sample"
    assert_state_allowed("pooled_source", out["state"])
    return out


def empirical_screening_sequence(q_values, eligible, score_fn,
                                 reference: float, strategy: str) -> dict:
    """**EMPIRICAL screening sequence** — weighted and shrinkage only.

    Mechanically identical to the formal sequence (same frozen order, same
    first-failure stop) so the four strategies stay comparable, but it tests no
    hypothesis, rejects nothing, controls no error rate and carries no nominal
    level. `reference` is the same `r*_s` value, reused as a screening
    reference for comparability and **not** as a guaranteed risk target.
    """
    if tier_of(strategy) != TIER_EMPIRICAL:
        raise AssertionError(
            f"{strategy!r} is not an empirical-screening strategy; the formal "
            "and source-reference tiers have their own entry points")
    out = _run_sequence(q_values, eligible, score_fn, reference, strategy)
    out["screening_reference"] = float(reference)
    assert_state_allowed(strategy, out["state"])
    return out


def run_sequence_for_strategy(strategy: str, q_values, eligible, score_fn,
                              reference: float) -> dict:
    """Dispatch to the tier-correct entry point. No generic path exists."""
    tier = tier_of(strategy)
    if tier == TIER_FORMAL:
        return certify_fixed_sequence(q_values, eligible, score_fn, reference)
    if tier == TIER_SOURCE_REFERENCE:
        return source_reference_sequence(q_values, eligible, score_fn, reference)
    return empirical_screening_sequence(q_values, eligible, score_fn, reference,
                                        strategy)


# ---------------------------------------------------------------------------
# retained-set bookkeeping shared by every strategy
# ---------------------------------------------------------------------------
def retained_class_counts(y_true, y_pred, keep) -> dict:
    """`(n_c, x_c)` per class on the retained rows.

    `y`: 1 = MDD, 0 = HC. `x_c` counts rows the **frozen Gate 1 prediction**
    gets wrong; S2 never re-predicts.
    """
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(y_pred).ravel().astype(int)
    k = np.asarray(keep).ravel().astype(bool)
    if not (y.shape == p.shape == k.shape):
        raise ValueError("y_true, y_pred and keep must align")
    yk, pk = y[k], p[k]
    return {
        "n_keep": int(k.sum()),
        "n_keep_MDD": int((yk == 1).sum()),
        "n_keep_HC": int((yk == 0).sum()),
        "x_MDD": int(((yk == 1) & (pk != 1)).sum()),
        "x_HC": int(((yk == 0) & (pk != 0)).sum()),
    }


def selective_ber_from_counts(counts) -> object:
    """`0.5 (err_MDD + err_HC)` on retained rows, or `NOT_ESTIMABLE`."""
    n_m, n_h = int(counts["n_keep_MDD"]), int(counts["n_keep_HC"])
    if n_m == 0 or n_h == 0:
        return NOT_ESTIMABLE
    return 0.5 * (counts["x_MDD"] / n_m + counts["x_HC"] / n_h)


# ---------------------------------------------------------------------------
# §17 — the post-hoc oracle diagnostic
# ---------------------------------------------------------------------------
def oracle_infeasible(coverage_test, class_count_pairs, ber_test, r_star: float,
                      c_min: float = C_MIN, m_min: int = M_MIN) -> dict:
    """Was *any* candidate deployable on this test split? Post-hoc only.

    Takes part in no selection, no decision, no `n0` choice and no headline
    aggregate. `uncertified` and `oracle_infeasible` may never be equated.
    Test-side `m_min` applies, because the question is deployability.
    """
    n = len(coverage_test)
    if not (len(class_count_pairs) == len(ber_test) == n):
        raise ValueError("one entry per candidate in every argument")
    feasible = []
    for i in range(n):
        n_m, n_h = class_count_pairs[i]
        if float(coverage_test[i]) < float(c_min):
            continue
        if int(n_m) < int(m_min) or int(n_h) < int(m_min):
            continue
        b = ber_test[i]
        if b == NOT_ESTIMABLE or b is None or not math.isfinite(float(b)):
            continue
        if float(b) <= float(r_star):
            feasible.append(i)
    return {
        "oracle_infeasible": int(len(feasible) == 0),
        "n_feasible_candidates": len(feasible),
        "feasible_indices": feasible,
    }
