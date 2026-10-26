"""Repeat -> site -> cross-site aggregation and the CI registry (RQ2/S2 §18-§22).

**Stage 2.0.1 rebuild.** Stage 2.0's registry of 228 intervals was built on a
single certification semantics shared by all four strategies. That semantics no
longer exists, so the registry could not be carried over mechanically: an
interval on "certification probability" that spanned an exact target-local
bound, an exact source-side bound and two empirical screening scores would put
four different claims on one axis. The registry is rebuilt around the three
guarantee tiers, and its count is recomputed rather than preserved.

Two rules the module enforces rather than documents:

* 100 repeats quantify **split instability** inside a site; the unit of
  inference is the **site**. Only one function sees the repeat dimension, and
  it collapses it.
* A quantity carries a tier in its name, and `assert_quantity_strategy_allowed`
  refuses to compute a formal quantity for a non-formal strategy. There is no
  generic "certification probability" that four strategies can share.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import contracts
import site_bootstrap
from certification import (CONFIDENCE_METHODS, FORMAL_STRATEGY, N_GRID,
                           NOT_ESTIMABLE, PRIMARY_N, R_SPLIT, STRATEGIES,
                           TIER_EMPIRICAL, TIER_FORMAL, TIER_SOURCE_REFERENCE,
                           ZERO_LABEL_STRATEGIES, assert_state_allowed,
                           is_pass, is_target_certified, realized_outcome_field,
                           tier_of)

CENSORED = "CENSORED"
NOT_CERTIFIED_WITHIN_GRID = "NOT_CERTIFIED_WITHIN_FEASIBLE_GRID"
NOT_CERTIFIED_WITHIN_40 = "NOT_CERTIFIED_WITHIN_40"
NO_EMPIRICAL_PASS_WITHIN_GRID = "NO_EMPIRICAL_PASS_WITHIN_FEASIBLE_GRID"
NO_EMPIRICAL_PASS_WITHIN_40 = "NO_EMPIRICAL_PASS_WITHIN_40"
#: Zero-label strategies have no label-cost axis at all (§6).
NOT_APPLICABLE_ZERO_LABEL = "NOT_APPLICABLE_ZERO_LABEL"

CENSORING_ORDER_STATISTICS = {"q1": 25, "median": 50, "q3": 75}
SUBSTANTIAL_CENSORING = 0.5

#: The single strategy with an empirical few-label label-cost axis (§6).
EMPIRICAL_FEW_LABEL_STRATEGY = "pooled_local_shrinkage"


# ---------------------------------------------------------------------------
# §19 — repeat -> site, with a tier-correct name on every output
# ---------------------------------------------------------------------------
PASS_RATE_FIELD = {
    TIER_FORMAL: "formal_certification_probability",
    TIER_SOURCE_REFERENCE: "source_reference_pass_rate",
    TIER_EMPIRICAL: "empirical_pass_probability",
}
COVERAGE_FIELD = {
    TIER_FORMAL: "formal_attainable_coverage",
    TIER_SOURCE_REFERENCE: "source_reference_attainable_coverage",
    TIER_EMPIRICAL: "empirical_attainable_coverage",
}


def pass_rate_site(strategy: str, states) -> dict:
    """The site's pass rate under **its own** tier's vocabulary.

    `n_valid_repeats` is the repeat count itself: every repeat of an
    `N`-eligible site is valid and there is no other validity condition, so the
    denominator can never depend on an outcome. `NOT_TESTABLE` repeats count as
    non-passing and are reported separately.
    """
    sts = list(states)
    if not sts:
        raise ValueError("a site with no repeats cannot be summarised")
    for s in sts:
        assert_state_allowed(strategy, s)
    n_pass = sum(1 for s in sts if is_pass(s))
    n_not_testable = sum(1 for s in sts if s == "NOT_TESTABLE")
    field = PASS_RATE_FIELD[tier_of(strategy)]
    return {
        "strategy": strategy,
        "guarantee_tier": tier_of(strategy),
        "n_valid_repeats": len(sts),
        "n_pass_repeats": n_pass,
        "n_not_testable_repeats": n_not_testable,
        field: n_pass / len(sts),
        f"{field}__complement": 1.0 - n_pass / len(sts),
        "pass_rate_field": field,
    }


def assert_zero_label_repeat_invariance(strategy: str, states, selected) -> None:
    """A zero-label strategy's decision may not vary across repeats.

    `pooled_source` and `target_unlabeled_weighted` decide from source data
    alone, so their state and selected threshold are functions of the outer
    fold. Variation would mean a zero-label decision had seen a split.
    """
    if strategy not in ZERO_LABEL_STRATEGIES:
        raise AssertionError(f"{strategy!r} is not a zero-label strategy")
    sts, sel = list(states), list(selected)
    if len(set(sts)) > 1 or len(set(map(repr, sel))) > 1:
        raise AssertionError(
            f"zero-label strategy {strategy!r} produced repeat-dependent "
            f"decisions ({sorted(set(sts))}); its decision must not depend on "
            "the target split (docs/RQ2_S2_PROTOCOL.md §3.1, §19.2)")


def attainable_coverage_site(strategy: str, states, coverage_test) -> dict:
    """Coverage of the selected threshold on `D_test`, over passing repeats only.

    `NOT_ESTIMABLE` when the site passes in no repeat. The pass rate always
    travels with the value: a high coverage among 3 passing repeats of 100 is
    not a capability.
    """
    sts = list(states)
    cov = np.asarray(coverage_test, dtype=float).ravel()
    if len(sts) != cov.size:
        raise ValueError("one coverage per repeat")
    summary = pass_rate_site(strategy, sts)
    mask = np.array([is_pass(s) for s in sts])
    vals = cov[mask]
    vals = vals[np.isfinite(vals)]
    field = COVERAGE_FIELD[tier_of(strategy)]
    summary["coverage_field"] = field
    if vals.size == 0:
        summary[field] = NOT_ESTIMABLE
        summary[f"{field}__median"] = NOT_ESTIMABLE
    else:
        summary[field] = float(vals.mean())
        summary[f"{field}__median"] = float(np.median(vals))
    return summary


def realized_outcome_fraction_site(strategy: str, states, outcomes) -> dict:
    """The strategy's realized target-side fraction, under its own field name.

    * `target_local`   -> `formal_realized_violation` (the only quantity that
      may later be set against the nominal `alpha_UCB = 0.10`);
    * `pooled_source`  -> `transported_target_risk_exceedance` (descriptive);
    * weighted / shrinkage -> `post_pass_target_risk_exceedance` (descriptive).

    A passing repeat whose retained test set lacks a class contributes to
    neither numerator nor denominator, and the threshold is never changed.
    """
    sts, vals = list(states), list(outcomes)
    if len(sts) != len(vals):
        raise ValueError("one realized outcome per repeat")
    for s in sts:
        assert_state_allowed(strategy, s)
    num = den = 0
    for s, v in zip(sts, vals):
        if not is_pass(s):
            continue
        if v == NOT_ESTIMABLE or v is None:
            continue
        den += 1
        num += int(v)
    field = realized_outcome_field(strategy)
    out = {
        "strategy": strategy,
        "guarantee_tier": tier_of(strategy),
        "realized_field": field,
        "n_pass_repeats": sum(1 for s in sts if is_pass(s)),
        "n_estimable_pass_repeats": den,
        f"{field}_fraction": (num / den) if den else NOT_ESTIMABLE,
    }
    return out


# ---------------------------------------------------------------------------
# §5, §6 — the two label-cost estimands, kept apart by construction
# ---------------------------------------------------------------------------
def minimum_certifiable_n(strategy: str, state_by_n: dict, feasible_n) -> dict:
    """`N_min_cert` — the FORMAL label cost. `target_local` only.

    The minimum feasible `N` at which `target_local` obtains
    `TARGET_CERTIFIED` for that repeat. No other strategy has this estimand,
    and asking for one raises: `pooled_source` and
    `target_unlabeled_weighted` spend zero labels, and
    `pooled_local_shrinkage` screens rather than certifies.
    """
    if strategy != FORMAL_STRATEGY:
        raise AssertionError(
            f"minimum certifiable N is defined for {FORMAL_STRATEGY!r} alone; "
            f"{strategy!r} has no formal target-risk certification semantics "
            "(docs/RQ2_S2_PROTOCOL.md §5)")
    return _minimum_n(strategy, state_by_n, feasible_n, is_target_certified,
                      NOT_CERTIFIED_WITHIN_40, NOT_CERTIFIED_WITHIN_GRID,
                      "n_min_cert")


def minimum_empirical_pass_n(strategy: str, state_by_n: dict,
                             feasible_n) -> dict:
    """`N_min_empirical_pass` — an EMPIRICAL label-budget descriptor.

    Defined for `pooled_local_shrinkage` only, and deliberately named so that
    it can never be plotted or tabulated as a certification cost. Calling it
    for `target_local` raises, which is the field-mixing guard.
    """
    if strategy != EMPIRICAL_FEW_LABEL_STRATEGY:
        raise AssertionError(
            f"minimum empirical-pass N is defined for "
            f"{EMPIRICAL_FEW_LABEL_STRATEGY!r} alone; {strategy!r} either "
            "certifies formally or spends no target labels "
            "(docs/RQ2_S2_PROTOCOL.md §6)")
    return _minimum_n(strategy, state_by_n, feasible_n, is_pass,
                      NO_EMPIRICAL_PASS_WITHIN_40, NO_EMPIRICAL_PASS_WITHIN_GRID,
                      "n_min_empirical_pass")


def zero_label_cost(strategy: str) -> str:
    """Zero-label strategies have no `N` axis — they report status only (§6)."""
    if strategy not in ZERO_LABEL_STRATEGIES:
        raise AssertionError(f"{strategy!r} is not a zero-label strategy")
    return NOT_APPLICABLE_ZERO_LABEL


def _minimum_n(strategy, state_by_n, feasible_n, predicate, token_40, token_grid,
               field):
    feasible = tuple(sorted(int(n) for n in feasible_n))
    missing = [n for n in feasible if n not in state_by_n]
    if missing:
        raise ValueError(f"no state for feasible N {missing}")
    for n in feasible:
        assert_state_allowed(strategy, state_by_n[n])
    hits = [n for n in feasible if predicate(state_by_n[n])]
    max_feasible = feasible[-1] if feasible else None
    if not hits:
        token = token_40 if max_feasible == 40 else token_grid
        return {field: token, "censored": True, "max_feasible_N": max_feasible,
                "non_monotone": False, "strategy": strategy}
    non_monotone = any(not predicate(state_by_n[n]) for n in feasible
                       if n > hits[0])
    return {field: int(hits[0]), "censored": False,
            "max_feasible_N": max_feasible, "non_monotone": bool(non_monotone),
            "strategy": strategy}


def censored_order_statistic_summary(values, r_split: int = R_SPLIT) -> dict:
    """Q1 / median / Q3 as 1-indexed order statistics, censored ranked `+inf`.

    Each is emitted as a number when that order statistic is finite and as the
    literal `CENSORED` when it is not. No imputation to 40 or infinity, no
    interpolation, no survival model, and no repeat is dropped from the
    denominator.
    """
    vals = list(values)
    if len(vals) != int(r_split):
        raise ValueError(f"expected {r_split} repeat values, got {len(vals)}")
    finite = sorted(int(v) for v in vals if not isinstance(v, str))
    n_censored = len(vals) - len(finite)
    out = {"n_repeats": len(vals), "n_censored": n_censored,
           "p_censored": n_censored / len(vals)}
    for name, k in CENSORING_ORDER_STATISTICS.items():
        out[name] = float(finite[k - 1]) if len(finite) >= k else CENSORED
    return out


def expected_uncertified_site_burden(p_uncertified_by_site) -> float:
    """`sum_s (1 - formal certification probability)` — formal tier only.

    The caller must have produced the per-site values through
    `pass_rate_site("target_local", ...)`; the quantity has no meaning for a
    screening tier, which is why the burden is not registered for one.
    """
    v = np.asarray(p_uncertified_by_site, dtype=float).ravel()
    if not np.isfinite(v).all():
        raise ValueError("the formal certification probability is defined for "
                         "every eligible site")
    return float(v.sum())


# ---------------------------------------------------------------------------
# §12 — the rebuilt authoritative CI registry
# ---------------------------------------------------------------------------
#: Cross-strategy contrasts are formed on a **common observable target
#: outcome** only, never on a pass rate that spans guarantee tiers.
DESCRIPTIVE_CONTRAST_QUANTITY = "descriptive_target_selective_ber"
PRIMARY_CONTRASTS = (
    ("target_local", "pooled_source"),
    ("pooled_local_shrinkage", "pooled_source"),
    ("pooled_local_shrinkage", "target_local"),
    ("target_unlabeled_weighted", "pooled_source"),
)
DESCRIPTIVE_QUANTITIES = (
    "descriptive_target_coverage",
    "descriptive_target_selective_ber",
    "descriptive_target_risk_exceedance",
)

#: quantity -> the strategies it may be computed for. A formal quantity has
#: exactly one legal strategy; this mapping is the type system of the registry.
QUANTITY_STRATEGY_SCOPE = {
    "formal_certification_probability": (FORMAL_STRATEGY,),
    "formal_attainable_coverage": (FORMAL_STRATEGY,),
    "formal_realized_violation_rate": (FORMAL_STRATEGY,),
    "formal_p_censored_min_cert_n": (FORMAL_STRATEGY,),
    "empirical_pass_probability": (EMPIRICAL_FEW_LABEL_STRATEGY,),
    "empirical_attainable_coverage": (EMPIRICAL_FEW_LABEL_STRATEGY,),
    "empirical_p_censored_min_pass_n": (EMPIRICAL_FEW_LABEL_STRATEGY,),
    "source_reference_pass_rate": ("pooled_source",),
    "zero_label_empirical_pass_rate": ("target_unlabeled_weighted",),
    "descriptive_target_coverage": STRATEGIES,
    "descriptive_target_selective_ber": STRATEGIES,
    "descriptive_target_risk_exceedance": STRATEGIES,
    "descriptive_contrast": STRATEGIES,
}

S2_CI_REQUIRED = tuple(QUANTITY_STRATEGY_SCOPE)

S2_CI_PROHIBITED = (
    "per_site_value",
    "per_repeat_value",
    "candidate_ledger",
    "expected_uncertified_site_burden",
    "n_min_cert_median",
    "n_min_cert_iqr",
    "n_min_empirical_pass_median",
    "n_min_site_distribution",
    "pass_rate_extremes",
    "not_testable_reason_split",
    "oracle_infeasible",
    "meta_objective_table",
    "ess_and_weight_diagnostics",
    "shrinkage_lambda_diagnostics",
    "jsd",
    "n_to_n_contrast",
    "cross_tier_pass_rate_contrast",       # forbidden outright, see §11
    "sensitivity_overall_error_ucb",
    "sensitivity_q_step_002",
    "sensitivity_r_star_shift",
    "sensitivity_common_sites",
)

S2_REQUIRED_INTERVALS = (
    # A — formal target-local certification (39)
    tuple(("formal_certification_probability", n, m)
          for n in N_GRID for m in CONFIDENCE_METHODS)
    + tuple(("formal_attainable_coverage", n, m)
            for n in N_GRID for m in CONFIDENCE_METHODS)
    + tuple(("formal_realized_violation_rate", n, m)
            for n in N_GRID for m in CONFIDENCE_METHODS)
    + tuple(("formal_p_censored_min_cert_n", m) for m in CONFIDENCE_METHODS)
    # B — empirical few-label screening, shrinkage only (27)
    + tuple(("empirical_pass_probability", n, m)
            for n in N_GRID for m in CONFIDENCE_METHODS)
    + tuple(("empirical_attainable_coverage", n, m)
            for n in N_GRID for m in CONFIDENCE_METHODS)
    + tuple(("empirical_p_censored_min_pass_n", m) for m in CONFIDENCE_METHODS)
    # C — zero-label reference statuses, no N axis (6)
    + tuple(("source_reference_pass_rate", m) for m in CONFIDENCE_METHODS)
    + tuple(("zero_label_empirical_pass_rate", m) for m in CONFIDENCE_METHODS)
    # D — descriptive transported target outcomes, all four strategies (108)
    + tuple((q, n, s, m) for q in DESCRIPTIVE_QUANTITIES
            for n in PRIMARY_N for s in STRATEGIES for m in CONFIDENCE_METHODS)
    # E — descriptive paired contrasts on one common observable (36)
    + tuple(("descriptive_contrast", DESCRIPTIVE_CONTRAST_QUANTITY, c, m, n)
            for c in PRIMARY_CONTRASTS for m in CONFIDENCE_METHODS
            for n in PRIMARY_N)
)

N_REQUIRED_INTERVALS = len(S2_REQUIRED_INTERVALS)

assert N_REQUIRED_INTERVALS == 216, N_REQUIRED_INTERVALS
assert len(set(S2_REQUIRED_INTERVALS)) == N_REQUIRED_INTERVALS
assert not set(S2_CI_REQUIRED) & set(S2_CI_PROHIBITED)


def requires_ci(quantity: str) -> bool:
    """True iff `quantity` carries a 95% site-bootstrap CI. Raises if unknown.

    There is no "optional", "if useful" or default-off category anywhere in S2.
    """
    if quantity in S2_CI_REQUIRED:
        return True
    if quantity in S2_CI_PROHIBITED:
        return False
    raise KeyError(
        f"{quantity!r} is not in the frozen CI scope table "
        f"(docs/RQ2_S2_PROTOCOL.md §21.2); classify it there first")


def assert_quantity_strategy_allowed(quantity: str, strategy: str) -> None:
    """A formal quantity may only be computed for the formal strategy."""
    if quantity not in QUANTITY_STRATEGY_SCOPE:
        raise KeyError(f"{quantity!r} is not a registered S2 quantity")
    allowed = QUANTITY_STRATEGY_SCOPE[quantity]
    if strategy not in allowed:
        raise AssertionError(
            f"{quantity!r} may not be computed for {strategy!r}; its scope is "
            f"{allowed} (docs/RQ2_S2_PROTOCOL.md §11, §12)")


# ---------------------------------------------------------------------------
# Stage 2.0.2 — the full CI key, and the single gate every interval goes through
# ---------------------------------------------------------------------------
class CIRegistryError(AssertionError):
    """Raised when an interval is requested outside the frozen registry."""


FAMILY_FORMAL = "formal"
FAMILY_EMPIRICAL_FEW_LABEL = "empirical_few_label"
FAMILY_ZERO_LABEL = "zero_label"
FAMILY_DESCRIPTIVE = "descriptive"
FAMILY_DESCRIPTIVE_CONTRAST = "descriptive_contrast"

#: Descriptive observables carry no guarantee tier of their own — they are the
#: one thing all four strategies may be compared on.
TIER_DESCRIPTIVE = "descriptive_observable"

_FAMILY_OF_QUANTITY = {
    "formal_certification_probability": FAMILY_FORMAL,
    "formal_attainable_coverage": FAMILY_FORMAL,
    "formal_realized_violation_rate": FAMILY_FORMAL,
    "formal_p_censored_min_cert_n": FAMILY_FORMAL,
    "empirical_pass_probability": FAMILY_EMPIRICAL_FEW_LABEL,
    "empirical_attainable_coverage": FAMILY_EMPIRICAL_FEW_LABEL,
    "empirical_p_censored_min_pass_n": FAMILY_EMPIRICAL_FEW_LABEL,
    "source_reference_pass_rate": FAMILY_ZERO_LABEL,
    "zero_label_empirical_pass_rate": FAMILY_ZERO_LABEL,
    "descriptive_target_coverage": FAMILY_DESCRIPTIVE,
    "descriptive_target_selective_ber": FAMILY_DESCRIPTIVE,
    "descriptive_target_risk_exceedance": FAMILY_DESCRIPTIVE,
}


@dataclass(frozen=True)
class CIKey:
    """The complete coordinates of one registered interval.

    Validating a *quantity name* is not enough: a formal quantity requested for
    the wrong strategy, a registered quantity at an unregistered `N`, or a
    contrast between an unregistered strategy pair each name a legitimate
    string and an illegitimate interval. The gate validates the whole tuple.
    """

    family: str
    quantity: str
    strategy: str
    n_labels: object = None          # int, or None for the N-free quantities
    method: str = None
    contrast: tuple = None           # (A, B) for a contrast, else None
    tier: str = None

    def as_tuple(self) -> tuple:
        return (self.family, self.quantity, self.strategy, self.n_labels,
                self.method, self.contrast, self.tier)


def _tier_for(family: str, strategy: str) -> str:
    if family in (FAMILY_DESCRIPTIVE, FAMILY_DESCRIPTIVE_CONTRAST):
        return TIER_DESCRIPTIVE
    return tier_of(strategy)


def _key_from_registry_entry(entry: tuple) -> CIKey:
    """Derive the canonical key of one frozen registry entry.

    The 216 entries are **not** re-enumerated here: the keys are derived from
    the frozen tuples, so the count and composition cannot drift apart from the
    registry the protocol document states.
    """
    head = entry[0]
    if head == "descriptive_contrast":
        _h, quantity, contrast, method, n = entry
        return CIKey(FAMILY_DESCRIPTIVE_CONTRAST, quantity, contrast[0], int(n),
                     method, tuple(contrast), TIER_DESCRIPTIVE)
    family = _FAMILY_OF_QUANTITY[head]
    if family == FAMILY_DESCRIPTIVE:
        quantity, n, strategy, method = entry
        return CIKey(family, quantity, strategy, int(n), method, None,
                     TIER_DESCRIPTIVE)
    strategy = QUANTITY_STRATEGY_SCOPE[head][0]
    if len(entry) == 3:                       # quantity, N, method
        _q, n, method = entry
        return CIKey(family, head, strategy, int(n), method, None,
                     _tier_for(family, strategy))
    _q, method = entry                        # quantity, method (no N axis)
    return CIKey(family, head, strategy, None, method, None,
                 _tier_for(family, strategy))


S2_REQUIRED_CI_KEYS = frozenset(_key_from_registry_entry(e)
                                for e in S2_REQUIRED_INTERVALS)

assert len(S2_REQUIRED_CI_KEYS) == N_REQUIRED_INTERVALS == 216


def ci_key(quantity: str, *, strategy: str, method: str, n_labels=None,
           contrast=None) -> CIKey:
    """Build a full key from named coordinates, deriving family and tier."""
    if contrast is not None:
        contrast = tuple(contrast)
        return CIKey(FAMILY_DESCRIPTIVE_CONTRAST, quantity, contrast[0],
                     None if n_labels is None else int(n_labels), method,
                     contrast, TIER_DESCRIPTIVE)
    family = _FAMILY_OF_QUANTITY.get(quantity)
    if family is None:
        raise CIRegistryError(
            f"{quantity!r} is not a registered S2 CI quantity "
            "(docs/RQ2_S2_PROTOCOL.md §21.2)")
    return CIKey(family, quantity, strategy,
                 None if n_labels is None else int(n_labels), method, None,
                 _tier_for(family, strategy))


def assert_ci_registered(key: CIKey) -> CIKey:
    """The gate. Raises unless the **complete** key is one of the frozen 216."""
    if not isinstance(key, CIKey):
        raise CIRegistryError(f"expected a CIKey, got {type(key)}")
    if key in S2_REQUIRED_CI_KEYS:
        return key
    # a targeted diagnosis, so a caller learns which coordinate is wrong
    same_quantity = [k for k in S2_REQUIRED_CI_KEYS if k.quantity == key.quantity]
    if not same_quantity:
        raise CIRegistryError(
            f"{key.quantity!r} is not a registered CI quantity; classify it in "
            "the protocol before it can be emitted")
    if key.contrast is not None and not any(
            k.contrast == key.contrast for k in same_quantity):
        raise CIRegistryError(
            f"contrast {key.contrast} is not a pre-registered strategy pair "
            f"(frozen pairs: {PRIMARY_CONTRASTS})")
    if not any(k.strategy == key.strategy for k in same_quantity):
        raise CIRegistryError(
            f"{key.quantity!r} may not be computed for {key.strategy!r}; its "
            f"scope is {QUANTITY_STRATEGY_SCOPE.get(key.quantity)}")
    if not any(k.n_labels == key.n_labels for k in same_quantity):
        raise CIRegistryError(
            f"{key.quantity!r} carries no registered interval at N="
            f"{key.n_labels!r} (registered: "
            f"{sorted({k.n_labels for k in same_quantity}, key=str)})")
    if not any(k.method == key.method for k in same_quantity):
        raise CIRegistryError(f"{key.method!r} is not a frozen confidence method")
    raise CIRegistryError(f"{key.as_tuple()} is not one of the frozen 216 keys")


# ---------------------------------------------------------------------------
# cross-site inference — the site is the only resampling unit
# ---------------------------------------------------------------------------
def site_draws(n_sites: int) -> np.ndarray:
    """The frozen multiplicity matrix for an analysis over `n_sites` sites."""
    return site_bootstrap.site_bootstrap_counts(int(n_sites))


def _check_site_vector(values, n_sites):
    v = np.asarray(values, dtype=float).ravel()
    k = int(n_sites) if n_sites is not None else v.size
    if v.size != k:
        raise ValueError(f"expected {k} site values, got {v.size}")
    if not 1 <= k <= len(contracts.FROZEN_SITES):
        raise ValueError(
            f"{k} sites is outside the frozen eligible-site set "
            f"(1..{len(contracts.FROZEN_SITES)})")
    return v, k


class RegisteredIntervalResult:
    """The value `compute_registered_interval` returns (Stage 2.0.5).

    A plain immutable value type. **No protocol guarantee rests on a caller
    being unable to construct one**: the formal report accepts no interval
    values at all, so a fabricated result has nowhere to go. Mutating a returned
    object likewise changes nothing — `build_formal_s2_report` recomputes every
    interval from site-level data and copies primitives into its own rows.
    """

    __slots__ = ("_key", "_payload")

    def __init__(self, key: CIKey, payload: dict):
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_payload", dict(payload))

    def __setattr__(self, name, value):
        raise CIRegistryError("a registered interval result is immutable")

    @property
    def key(self) -> CIKey:
        return self._key

    def as_dict(self) -> dict:
        return dict(self._payload)

    def __repr__(self) -> str:
        return f"RegisteredIntervalResult({self._key.as_tuple()})"


def compute_registered_interval(key: CIKey, site_values, site_values_b=None,
                                n_sites: int = None):
    """**The single public gate.** Validate the full key, then bootstrap.

    Order is fixed: validate the exact registry key -> validate the
    quantity/strategy tier scope -> apply the estimability and common-site
    rules -> resample sites -> return the value. The formal report calls this
    itself, once per registered key; it never accepts a result from a caller.
    """
    assert_ci_registered(key)
    if key.contrast is None:
        assert_quantity_strategy_allowed(key.quantity, key.strategy)
        v, k = _check_site_vector(site_values, n_sites)
        out = site_bootstrap.bootstrap_summary(v, site_draws(k))
        out.update({"quantity": key.quantity, "strategy": key.strategy,
                    "family": key.family, "tier": key.tier,
                    "n_labels": key.n_labels, "method": key.method,
                    "n_sites": k, "ci_key": key.as_tuple()})
    else:
        if key.quantity not in DESCRIPTIVE_QUANTITIES:
            raise CIRegistryError(
                f"{key.quantity!r} may not be contrasted across strategies; only "
                "a common observable target outcome may be, and a pass rate "
                "spans guarantee tiers (docs/RQ2_S2_PROTOCOL.md §11, §22)")
        a_name, b_name = key.contrast
        for s in (a_name, b_name):
            assert_quantity_strategy_allowed(key.quantity, s)
        a, k = _check_site_vector(site_values, n_sites)
        b, _ = _check_site_vector(site_values_b, k)
        out = site_bootstrap.paired_contrast(a, b, site_draws(k))
        out.update({"quantity": key.quantity, "strategy_A": a_name,
                    "strategy_B": b_name, "family": key.family, "tier": key.tier,
                    "n_labels": key.n_labels, "method": key.method, "n_sites": k,
                    "guarantee_tier_A": tier_of(a_name),
                    "guarantee_tier_B": tier_of(b_name),
                    "tiers_differ": tier_of(a_name) != tier_of(b_name),
                    "ci_key": key.as_tuple()})

    return RegisteredIntervalResult(key, out)


def cross_site_summary(quantity: str, strategy: str, site_values, *,
                       method: str, n_labels=None, n_sites: int = None) -> dict:
    """Registered site-balanced interval. Every coordinate is validated.

    Stage 2.0.2: `method` and `n_labels` are **required** coordinates, not
    decoration. Validating only the quantity name let a registered quantity be
    computed at an unregistered `N`, or for a strategy outside its scope.
    """
    key = ci_key(quantity, strategy=strategy, method=method, n_labels=n_labels)
    return compute_registered_interval(key, site_values, n_sites=n_sites).as_dict()


def cross_site_contrast(quantity: str, strategy_a: str, strategy_b: str,
                        values_a, values_b, *, method: str, n_labels=None,
                        n_sites: int = None) -> dict:
    """Registered paired `A - B` contrast. Only pre-registered pairs exist.

    A free-form `contrast_name`, an unregistered strategy pair, an unregistered
    quantity, a wrong `N` (including `N = 5`, which carries no contrast
    interval) and any cross-tier pass-rate contrast each raise.
    """
    key = ci_key(quantity, strategy=strategy_a, method=method,
                 n_labels=n_labels, contrast=(strategy_a, strategy_b))
    return compute_registered_interval(key, values_a, values_b,
                                       n_sites=n_sites).as_dict()
