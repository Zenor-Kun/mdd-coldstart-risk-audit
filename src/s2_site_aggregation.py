"""Authoritative site-level aggregation (RQ2/S2, Stage 2.0.8).

The formal report used to be built from a `SiteLevelResults` the caller
constructed. Its provenance fields were two strings, so a stream of arbitrary
site vectors labelled `decision_manifest_checksum = "FAKE"` produced a formal
report with a straight face. Provenance a caller types is decoration.

Site-level results are now **derived**, here, from frozen artifacts:

    ProductionS2Context
      -> authoritative decision store        (exact 46,200 + the 138 references)
      -> authoritative evaluation store      (exact expected set, each artifact
                                              bound to its frozen decision)
      -> re-derived decision + evaluation manifests
      -> site-level summaries                <- this module
      -> the frozen 216-key CI registry

The caller supplies no site vector, no site mean, no `p_cert` array, no BER
array and no coverage array. It cannot: the only public production entry point
takes a context and this run's ledger, and everything else is read off disk.

Stage 2.0.7 removes the last two ways in: the `evaluation_store` parameter (a
caller-supplied root meant a caller-supplied set of artifacts) and the
`ledger=None` default (a missing ledger meant no ordering assertion at all).
The store is resolved from the context; the ledger is required and must be this
run's allocator, already carrying every prerequisite event.

Two identity checks make "looks complete" insufficient (§10, §11):

* for every `(strategy, method, N, quantity)` the **site set** must equal the
  authoritative eligible set for that `N` — one site short, one site too many,
  a duplicate, or another `N`'s site list all fail;
* for every `(site, strategy, method, N)` the **repeat set** must be exactly
  `0..99`. A metric that is `NOT_ESTIMABLE` still has an artifact; estimability
  is a value, never a reason to drop a cell and shrink a denominator.
"""

from __future__ import annotations

import numpy as np

import certification as cert
import s2_aggregation as agg
import s2_evaluation_store as evalstore
import s2_ledger as ledgermod
import s2_splitting as spl

NOT_ESTIMABLE = cert.NOT_ESTIMABLE


class SiteAggregationError(AssertionError):
    """Raised when the evaluated cells cannot support the frozen registry."""


# ---------------------------------------------------------------------------
# the authoritative site scope
# ---------------------------------------------------------------------------
def authoritative_site_scope(context) -> dict:
    """The eligible-site identities, re-derived from the production registry.

    Not read from a constant and not supplied: the site sizes come from the
    registry's own rosters and the frozen §4 eligibility rule is applied to
    them, so the per-`N` site *identities* — not merely the counts 23/22/21/11 —
    are what every aggregate is checked against.
    """
    registry = context.registry
    sizes = {site: len(registry.site_roster(site)) for site in registry.sites}
    by_n = spl.eligible_sites_by_n(sizes)
    return {"by_n": {int(n): tuple(v) for n, v in by_n.items()},
            "all_sites": tuple(registry.sites),
            "per_n_site_counts": {int(n): len(v) for n, v in by_n.items()},
            "site_sizes": sizes}


def sites_for_key(key: agg.CIKey, scope: dict) -> tuple:
    """The exact site identities one registered interval must be computed on."""
    if key.n_labels is None:
        return scope["all_sites"]
    n = int(key.n_labels)
    if n not in scope["by_n"]:
        raise SiteAggregationError(f"N={n} is not on the frozen grid")
    return scope["by_n"][n]


def feasible_n_for_site(site: str, scope: dict) -> tuple:
    return tuple(n for n in cert.N_GRID if site in scope["by_n"][n])


# ---------------------------------------------------------------------------
# identity validation (§10, §11)
# ---------------------------------------------------------------------------
def index_evaluation_records(records) -> dict:
    """`{(strategy, method, N): {site: {repeat: payload}}}`, with no merging.

    Accepts a mapping or an iterable of `(cell key, payload)` pairs. A repeated
    cell raises rather than overwriting: two artifacts for one cell would let a
    site contribute a repeat twice, which §18 forbids.
    """
    index = {}
    items = records.items() if hasattr(records, "items") else records
    for key, payload in items:
        site, repeat, n_labels, strategy, method = key
        bucket = index.setdefault((strategy, method, int(n_labels)), {})
        by_repeat = bucket.setdefault(site, {})
        if repeat in by_repeat:
            raise SiteAggregationError(
                f"duplicate evaluation record for {key}")
        by_repeat[int(repeat)] = payload
    return index


def assert_cell_identities(index: dict, scope: dict, expected_cells,
                           r_split: int = cert.R_SPLIT) -> dict:
    """Site identities and repeat identities, per `(strategy, method, N)`.

    Counts are not enough and were never enough: 22 sites at `N = 10` with one
    eligible site replaced by an ineligible one has the right count and the
    wrong estimand.
    """
    expected_groups = {}
    for (site, repeat, n_labels, strategy, method) in expected_cells:
        expected_groups.setdefault((strategy, method, int(n_labels)),
                                   set()).add(site)

    if set(index) != set(expected_groups):
        missing = sorted(set(expected_groups) - set(index))
        extra = sorted(set(index) - set(expected_groups))
        raise SiteAggregationError(
            f"the evaluated (strategy, method, N) groups are not the "
            f"authoritative set: {len(missing)} missing, {len(extra)} "
            f"unexpected (e.g. missing {missing[:2]}, extra {extra[:2]})")

    checked = 0
    for group, by_site in sorted(index.items()):
        strategy, method, n_labels = group
        authoritative = set(scope["by_n"][n_labels])
        if set(by_site) != authoritative:
            missing = sorted(authoritative - set(by_site))
            extra = sorted(set(by_site) - authoritative)
            raise SiteAggregationError(
                f"{strategy}/{method} at N={n_labels} was aggregated over "
                f"{len(by_site)} sites, but the authoritative eligible set has "
                f"{len(authoritative)}: missing {missing}, unexpected {extra}")
        if set(by_site) != set(expected_groups[group]):
            raise SiteAggregationError(
                f"{group}: the evaluated site identities differ from the "
                "authoritative expected identities")
        for site, by_repeat in by_site.items():
            if set(by_repeat) != set(range(r_split)):
                missing = sorted(set(range(r_split)) - set(by_repeat))
                extra = sorted(set(by_repeat) - set(range(r_split)))
                raise SiteAggregationError(
                    f"{strategy}/{method} at N={n_labels}, site {site}: "
                    f"repeats are not exactly 0..{r_split - 1} "
                    f"({len(by_repeat)} present, missing {missing[:3]}, "
                    f"unexpected {extra[:3]}). A NOT_ESTIMABLE metric still "
                    "has an artifact; a cell is never dropped")
            checked += 1
    return {"n_groups": len(index), "n_site_repeat_blocks": checked,
            "r_split": int(r_split)}


def assert_zero_label_identities(zero_label: dict, scope: dict) -> dict:
    """The 138 references must cover every frozen site, method and strategy."""
    expected = {(site, strategy, method)
                for site in scope["all_sites"]
                for strategy in cert.ZERO_LABEL_STRATEGIES
                for method in cert.CONFIDENCE_METHODS}
    if set(zero_label) != expected:
        raise SiteAggregationError(
            f"the zero-label reference decisions do not cover the "
            f"authoritative (site, strategy, method) set: "
            f"{len(expected - set(zero_label))} missing, "
            f"{len(set(zero_label) - expected)} unexpected")
    return {"n_zero_label": len(zero_label)}


# ---------------------------------------------------------------------------
# per-site quantities
# ---------------------------------------------------------------------------
def _as_float(value) -> float:
    """`NOT_ESTIMABLE` becomes `NaN`, which the site bootstrap already honours."""
    if value is None or value == NOT_ESTIMABLE:
        return float("nan")
    return float(value)


def _ordered(by_repeat: dict, r_split: int) -> list:
    return [by_repeat[r] for r in range(r_split)]


def _pass_probability(rows, strategy) -> float:
    states = [r["decision_state"] for r in rows]
    summary = agg.pass_rate_site(strategy, states)
    return float(summary[summary["pass_rate_field"]])


def _attainable_coverage(rows, strategy) -> float:
    states = [r["decision_state"] for r in rows]
    coverage = [_as_float(r["coverage_test"]) for r in rows]
    summary = agg.attainable_coverage_site(strategy, states, coverage)
    return _as_float(summary[summary["coverage_field"]])


def _realized_outcome_fraction(rows, strategy) -> float:
    states = [r["decision_state"] for r in rows]
    outcomes = [NOT_ESTIMABLE if r["realized_outcome"] is None
                else int(r["realized_outcome"]) for r in rows]
    summary = agg.realized_outcome_fraction_site(strategy, states, outcomes)
    field = f"{summary['realized_field']}_fraction"
    return _as_float(summary[field])


def _p_censored_min_n(site, strategy, method, index, scope, r_split) -> float:
    """The site's `p_censored` for its tier's minimum-`N` estimand.

    The whole feasible `N` grid of that site is used, so a site eligible only
    at `N = 5` contributes a censored trajectory rather than being dropped.
    """
    feasible = feasible_n_for_site(site, scope)
    values = []
    for repeat in range(r_split):
        state_by_n = {n: index[(strategy, method, n)][site][repeat][
            "decision_state"] for n in feasible}
        if strategy == cert.FORMAL_STRATEGY:
            out = agg.minimum_certifiable_n(strategy, state_by_n, feasible)
            values.append(out["n_min_cert"])
        else:
            out = agg.minimum_empirical_pass_n(strategy, state_by_n, feasible)
            values.append(out["n_min_empirical_pass"])
    return float(agg.censored_order_statistic_summary(values, r_split)["p_censored"])


def _descriptive(rows, quantity) -> float:
    """A transported target observable, over the repeats that adopted a threshold.

    A repeat with no adopted threshold contributed no selective prediction, and
    a retained-class-empty repeat is `NOT_ESTIMABLE`: both leave numerator and
    denominator, and neither is imputed (§16).
    """
    if quantity == "descriptive_target_coverage":
        vals = [_as_float(r["coverage_test"]) for r in rows
                if r["evaluation_status"] != evalstore.EVAL_STATUS_NO_THRESHOLD]
    elif quantity == "descriptive_target_selective_ber":
        vals = [_as_float(r["ber_test"]) for r in rows
                if r["evaluation_status"] == evalstore.EVAL_STATUS_COMPLETE]
    elif quantity == "descriptive_target_risk_exceedance":
        vals = [float(r["realized_outcome"]) for r in rows
                if r["evaluation_status"] == evalstore.EVAL_STATUS_COMPLETE]
    else:
        raise SiteAggregationError(f"{quantity!r} is not a descriptive observable")
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _zero_label_pass(site, strategy, method, zero_label) -> float:
    payload = zero_label[(site, strategy, method)]
    return 1.0 if cert.is_pass(payload["state"]) else 0.0


def site_vector_for_key(key: agg.CIKey, index: dict, zero_label: dict,
                        scope: dict, r_split: int = cert.R_SPLIT) -> dict:
    """The site-level input of exactly one registered interval.

    Returns `site_values` (one per authoritative eligible site, in frozen site
    order, `NaN` where the site is `NOT_ESTIMABLE`), `n_sites` and — for a
    registered contrast — the second arm on the **same** site identities.
    """
    agg.assert_ci_registered(key)
    sites = sites_for_key(key, scope)
    quantity, method = key.quantity, key.method

    if key.contrast is not None:
        a, b = key.contrast
        values_a = [_descriptive(_ordered(index[(a, method, int(key.n_labels))][s],
                                          r_split), quantity) for s in sites]
        values_b = [_descriptive(_ordered(index[(b, method, int(key.n_labels))][s],
                                          r_split), quantity) for s in sites]
        return {"site_values": values_a, "site_values_b": values_b,
                "n_sites": len(sites), "site_ids": tuple(sites)}

    strategy = key.strategy
    values = []
    for site in sites:
        if quantity in ("formal_p_censored_min_cert_n",
                        "empirical_p_censored_min_pass_n"):
            values.append(_p_censored_min_n(site, strategy, method, index,
                                            scope, r_split))
            continue
        if quantity in ("source_reference_pass_rate",
                        "zero_label_empirical_pass_rate"):
            values.append(_zero_label_pass(site, strategy, method, zero_label))
            continue
        rows = _ordered(index[(strategy, method, int(key.n_labels))][site],
                        r_split)
        if quantity in ("formal_certification_probability",
                        "empirical_pass_probability"):
            values.append(_pass_probability(rows, strategy))
        elif quantity in ("formal_attainable_coverage",
                          "empirical_attainable_coverage"):
            values.append(_attainable_coverage(rows, strategy))
        elif quantity == "formal_realized_violation_rate":
            values.append(_realized_outcome_fraction(rows, strategy))
        elif quantity in agg.DESCRIPTIVE_QUANTITIES:
            values.append(_descriptive(rows, quantity))
        else:
            raise SiteAggregationError(
                f"{quantity!r} has no site-level aggregation rule")
    return {"site_values": values, "n_sites": len(sites),
            "site_ids": tuple(sites)}


#: Public names for the frozen per-site formulas. `s2_point_sections` reports
#: the same site-level quantities the registry bootstraps, and must call the
#: same functions rather than keep a second copy of them.
as_float = _as_float
ordered_repeat_rows = _ordered
site_pass_probability = _pass_probability
site_attainable_coverage = _attainable_coverage
site_realized_outcome_fraction = _realized_outcome_fraction
site_descriptive_value = _descriptive
site_zero_label_pass = _zero_label_pass


# ---------------------------------------------------------------------------
# the internal value object and the production entry point
# ---------------------------------------------------------------------------
class SiteLevelResults:
    """Site-level inputs of the 216 intervals. **An internal value object.**

    Stage 2.0.5 treated this type as evidence, and Stage 2.0.6 withdraws that:
    it carries no guarantee, and constructing one proves nothing. The formal
    report has no parameter through which one can be supplied — it obtains its
    own from `aggregate_authoritative_site_level_results`, which reads frozen
    artifacts. What makes the numbers trustworthy is where they came from, not
    what type they arrived in.
    """

    __slots__ = ("_values", "_provenance")

    REQUIRED_PROVENANCE = ("protocol_version", "decision_manifest_checksum",
                           "evaluation_manifest_checksum")

    def __init__(self, values: dict, provenance: dict) -> None:
        import s2_decisions as sd
        missing = [f for f in self.REQUIRED_PROVENANCE if not provenance.get(f)]
        if missing:
            raise SiteAggregationError(
                f"site-level results lack provenance {missing}")
        if provenance["protocol_version"] != sd.PROTOCOL_VERSION:
            raise SiteAggregationError(
                f"site-level results were produced under "
                f"{provenance['protocol_version']!r}, not {sd.PROTOCOL_VERSION!r}")
        for key, payload in values.items():
            if not isinstance(key, agg.CIKey):
                raise SiteAggregationError("site-level results are keyed by CIKey")
            agg.assert_ci_registered(key)
            for banned in ("ci_low", "ci_high", "point_estimate", "interval"):
                if banned in payload:
                    raise SiteAggregationError(
                        f"site-level input for {key.as_tuple()} carries "
                        f"{banned!r}; intervals are computed, never carried")
            if "site_values" not in payload or "n_sites" not in payload:
                raise SiteAggregationError(
                    f"site-level input for {key.as_tuple()} needs site_values "
                    "and n_sites")
        object.__setattr__(self, "_values", dict(values))
        object.__setattr__(self, "_provenance", dict(provenance))

    def __setattr__(self, name, value):
        raise SiteAggregationError("site-level results are immutable")

    @property
    def provenance(self) -> dict:
        return dict(self._provenance)

    def keys(self) -> tuple:
        return tuple(sorted(self._values, key=lambda k: k.as_tuple()))

    def for_key(self, key: agg.CIKey) -> dict:
        if key not in self._values:
            raise SiteAggregationError(
                f"no site-level input for registered key {key.as_tuple()}")
        return dict(self._values[key])

    def assert_covers_registry(self) -> None:
        missing = agg.S2_REQUIRED_CI_KEYS - set(self._values)
        extra = set(self._values) - agg.S2_REQUIRED_CI_KEYS
        if missing or extra:
            raise SiteAggregationError(
                f"site-level results do not cover the frozen registry: "
                f"{len(missing)} missing, {len(extra)} unregistered")


def aggregate_authoritative_site_level_results(context, ledger
                                               ) -> SiteLevelResults:
    """**The only production source of formal site-level results.**

    Two arguments, both authority, neither data: a run's context and a run's
    ledger. There is no `evaluation_store` parameter — Stage 2.0.6 still had
    one, and a complete fake store at an arbitrary root could therefore be
    aggregated — and `ledger` has no default, because `ledger=None` skipped the
    ordering assertion outright.

    Order is fixed and none of it is optional:

    1. the context re-derives as authoritative;
    2. the ledger *is* this run's allocator and has reached
       `EVALUATION_MANIFEST_FROZEN`;
    3. the decision store *is* the authoritative 46,200, and the 138 zero-label
       references are complete;
    4. the evaluation store — resolved from the context, never named by the
       caller — *is* the authoritative expected set, every artifact bound to the
       decision the store actually holds;
    5. the decision manifest is recomputed from disk;
    6. the evaluation manifest is recomputed from disk and must equal the one
       the ledger froze;
    7. the cells are aggregated into site-level summaries, with the site and
       repeat identities of every group asserted first.
    """
    import s2_firewall as fw
    import s2_decisions as sd
    from s2_production import ProductionAuthorityError, ProductionS2Context

    if not isinstance(context, ProductionS2Context):
        raise ProductionAuthorityError(
            "formal site-level results are aggregated under a "
            "ProductionS2Context")
    context.assert_still_authoritative()
    ledgermod.assert_run_ledger_for_context(ledger, context)
    ledger.append(ledgermod.EV_AGGREGATION_STARTED)

    decisions = fw.assert_decision_store_is_complete(context)
    zero_label = evalstore.load_zero_label_decisions(context)
    scan = evalstore.assert_evaluation_store_is_complete(context)
    frozen_manifest = ledger.provenance.get("evaluation_manifest_checksum")
    if frozen_manifest != scan["evaluation_manifest_checksum"]:
        raise ProductionAuthorityError(
            f"the ledger froze evaluation manifest {frozen_manifest!r}; the "
            f"authoritative store at {scan['evaluation_root']} recomputes to "
            f"{scan['evaluation_manifest_checksum']!r}")

    scope = authoritative_site_scope(context)
    index = index_evaluation_records(scan["records"])
    assert_cell_identities(index, scope, scan["expected"])
    assert_zero_label_identities(zero_label, scope)

    values = {}
    for key in sorted(agg.S2_REQUIRED_CI_KEYS, key=lambda k: k.as_tuple()):
        values[key] = site_vector_for_key(key, index, zero_label, scope)
    if len(values) != agg.N_REQUIRED_INTERVALS:
        raise SiteAggregationError(
            f"{len(values)} site-level inputs produced, not "
            f"{agg.N_REQUIRED_INTERVALS}")

    provenance = {
        "protocol_version": sd.PROTOCOL_VERSION,
        "decision_manifest_checksum": decisions["decision_manifest_checksum"],
        "evaluation_manifest_checksum": scan["evaluation_manifest_checksum"],
        "n_decision_cells": decisions["n_completed"],
        "n_evaluation_cells": scan["n_evaluated"],
        "n_zero_label_decisions": len(zero_label),
        "per_n_site_counts": scope["per_n_site_counts"],
        "run_id": context.run_id,
        "ledger_id": ledger.ledger_id,
        "evaluation_root": scan["evaluation_root"],
        "evaluation_root_identity": scan["evaluation_root_identity"],
    }
    ledger.bind(
        decision_manifest_checksum=provenance["decision_manifest_checksum"],
        evaluation_manifest_checksum=provenance["evaluation_manifest_checksum"])
    return SiteLevelResults(values, provenance)
