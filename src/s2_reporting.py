"""The formal RQ2/S2 report builder (Stage 2.0.8).

Stage 2.0.5 removed the caller's ability to supply *intervals*. It left the
ability to supply the numbers they are computed from: `build_formal_s2_report`
took a `SiteLevelResults` whose provenance was two strings, so

    SiteLevelResults(arbitrary_site_vectors,
                     {"decision_manifest_checksum": "FAKE",
                      "evaluation_manifest_checksum": "FAKE", ...})

produced a formal report. Stage 2.0.6 applies the same remedy one layer down:
**the report has no data parameter at all.**

    build_formal_s2_report(context, ledger)
        -> authoritative decision store        (exact 46,200 + 138 references)
        -> authoritative evaluation store      (exact expected set, each
                                                artifact bound to its decision)
        -> re-derived decision + evaluation manifests
        -> internal site-level aggregation     (site + repeat identities checked)
        -> the frozen 216-key registry, walked here
        -> compute_registered_interval()       (once per key, computed inside)
        -> S2Report

There is no `site_level_results` argument, no `intervals` argument, no
`decision_manifest_checksum` argument and no `set_interval` /`add_interval` /
`load_interval_store` / `set_ci` / `build_from_registered_store`. A caller who
wants a different reported number has to write different artifacts to the
evaluation store, and those fail their decision binding.

Stage 2.0.7: nor can a caller say *which* evaluation store, and nor can it omit
the ledger. The store is resolved from the context inside the aggregation, and
`ledger` is a required argument that must be this run's allocator carrying every
prerequisite event — an empty or foreign ledger fails before any artifact is
read.
"""

from __future__ import annotations

import s2_aggregation as agg
import s2_decisions as sd
import s2_evaluation_store as evalstore
import s2_ledger as ledgermod
import s2_point_sections as psec
import s2_site_aggregation as sagg

#: Re-exported so callers have one name for the internal value object. It is
#: **not** an input to anything here; see `s2_site_aggregation.SiteLevelResults`.
SiteLevelResults = sagg.SiteLevelResults


class ReportSchemaError(AssertionError):
    """Raised when report provenance or the 216 do not close."""


#: Quantities the report carries **without** an interval (§21.2). Every one is
#: registered, classified and derived by `s2_point_sections`; the sixteen the
#: formal run must produce are `psec.REQUIRED_FORMAL_SECTION_NAMES`, and the
#: four §23 sensitivities are executed in the later sensitivity stage.
POINT_ESTIMATE_SECTIONS = (
    "candidate_ledger", "per_site_value", "per_repeat_value",
    "expected_uncertified_site_burden", "n_min_cert_median", "n_min_cert_iqr",
    "n_min_empirical_pass_median", "n_min_site_distribution",
    "pass_rate_extremes", "not_testable_reason_split", "oracle_infeasible",
    "meta_objective_table", "ess_and_weight_diagnostics",
    "shrinkage_lambda_diagnostics", "jsd", "n_to_n_contrast",
    "sensitivity_overall_error_ucb", "sensitivity_q_step_002",
    "sensitivity_r_star_shift", "sensitivity_common_sites",
)

#: Fields every interval row carries. Copied primitives — the report never
#: retains a caller-visible mutable result object.
INTERVAL_ROW_FIELDS = ("ci_key", "quantity", "family", "tier", "strategy",
                       "n_labels", "method", "point_estimate", "ci_low",
                       "ci_high", "n_sites", "n_estimable_sites",
                       "valid_bootstrap_replicates",
                       "total_bootstrap_replicates")


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
class S2Report:
    """Immutable formal report. Built only by `build_formal_s2_report`."""

    __slots__ = ("_rows", "_point_sections", "_provenance")

    def __init__(self, token, rows, point_sections, provenance):
        if token is not _REPORT_TOKEN:
            raise ReportSchemaError(
                "an S2Report is produced by build_formal_s2_report() alone")
        object.__setattr__(self, "_rows", dict(rows))
        object.__setattr__(self, "_point_sections", dict(point_sections))
        object.__setattr__(self, "_provenance", dict(provenance))

    def __setattr__(self, name, value):
        raise ReportSchemaError("the report is immutable once built")

    @property
    def provenance(self) -> dict:
        return dict(self._provenance)

    @property
    def point_sections(self) -> dict:
        return dict(self._point_sections)

    def interval_rows(self) -> tuple:
        return tuple(dict(self._rows[k]) for k in
                     sorted(self._rows, key=lambda k: k.as_tuple()))

    def interval(self, key: agg.CIKey) -> dict:
        agg.assert_ci_registered(key)
        return dict(self._rows[key])

    def assert_complete(self) -> dict:
        missing = agg.S2_REQUIRED_CI_KEYS - set(self._rows)
        extra = set(self._rows) - agg.S2_REQUIRED_CI_KEYS
        if missing or extra:
            raise ReportSchemaError(
                f"{len(missing)} required CI rows unfilled, {len(extra)} "
                "unregistered")
        psec.assert_sections_complete(self._point_sections)
        return {"n_intervals": len(self._rows),
                "n_point_sections": len(self._point_sections),
                "n_registered_point_sections":
                    len(psec.REGISTERED_SECTION_NAMES)}


_REPORT_TOKEN = object()


def build_formal_s2_report(context, ledger) -> S2Report:
    """**The only formal report entry point.** Takes no data from its caller.

    Its two arguments are a run's authority and a run's ledger; neither carries
    a number that reaches a reported quantity, and neither is optional. The
    ledger must be *this* run's allocator and must already have reached
    `EVALUATION_MANIFEST_FROZEN`: `None`, a fresh ledger and another run's
    ledger all fail before anything is read. The evaluation artifacts are found
    by `resolve_authoritative_evaluation_store(context)`, so a report cannot be
    pointed at a store either.

    Stage 2.1b removes the last data parameter. `point_sections` was a caller
    dict whose *names* were checked and whose *values* were not, so a caller
    could report its own `n_min_cert_median` or `oracle_infeasible`. The
    sections are now derived by
    `s2_point_sections.derive_authoritative_formal_point_sections(context,
    ledger)` from the same frozen artifacts the intervals come from, and their
    set must equal `REQUIRED_FORMAL_SECTION_NAMES` exactly.
    """
    from s2_production import ProductionAuthorityError, ProductionS2Context

    if not isinstance(context, ProductionS2Context):
        raise ProductionAuthorityError(
            "a formal report is built under a ProductionS2Context")
    context.assert_still_authoritative()
    ledgermod.assert_run_ledger_for_context(ledger, context)

    if agg.N_REQUIRED_INTERVALS != 216:
        raise ReportSchemaError(
            f"the CI registry holds {agg.N_REQUIRED_INTERVALS} keys, not 216")

    site_level = sagg.aggregate_authoritative_site_level_results(context, ledger)
    site_level.assert_covers_registry()
    if site_level.provenance.get("ledger_id") != ledger.ledger_id:
        raise ReportSchemaError(
            "the aggregated results were produced under a different allocator")

    rows = _compute_registry_rows(site_level)

    sections = psec.derive_authoritative_formal_point_sections(context, ledger)
    for name in sections:
        if name not in POINT_ESTIMATE_SECTIONS:
            raise ReportSchemaError(
                f"{name!r} is not a registered point-estimate section")
        if agg.requires_ci(name):
            raise ReportSchemaError(f"{name!r} requires an interval, not a section")
    psec.assert_sections_complete(sections)

    provenance = dict(site_level.provenance)
    provenance.update({
        "ci_registry_checksum": context.snapshot["ci_registry_checksum"],
        "split_registry_checksum": context.snapshot["split_registry_checksum"],
        "n_intervals": len(rows),
        "global_ledger_id": ledger.ledger_id,
        "barrier_seqs": ledger.barrier_seqs(),
        "n_point_sections": len(sections),
        "required_point_sections": list(psec.REQUIRED_FORMAL_SECTION_NAMES),
        "later_stage_point_sections": list(psec.LATER_STAGE_SECTION_NAMES),
    })
    report = S2Report(_REPORT_TOKEN, rows, sections, provenance)
    report.assert_complete()
    ledger.append(ledgermod.EV_REPORT_BUILT, n_intervals=len(rows))
    ledgermod.assert_global_ordering(
        ledger, required=ledgermod.AGGREGATION_PREREQUISITE_EVENTS + (
            ledgermod.EV_AGGREGATION_STARTED, ledgermod.EV_REPORT_BUILT))
    return report


def _compute_registry_rows(site_level) -> dict:
    """Walk the frozen 216-key registry and compute every interval **here**.

    Internal on purpose. It returns rows, never an `S2Report`: a caller who
    reaches it can compute numbers for itself, which it could always do with
    `compute_registered_interval` anyway, but it cannot mint a formal report —
    only `build_formal_s2_report` can, and that takes no data.
    """
    if not isinstance(site_level, sagg.SiteLevelResults):
        raise ReportSchemaError(
            "the registry walk consumes aggregated SiteLevelResults")
    site_level.assert_covers_registry()
    rows = {}
    for key in report_schema_slots():
        payload = site_level.for_key(key)
        result = agg.compute_registered_interval(
            key, payload["site_values"], payload.get("site_values_b"),
            n_sites=payload["n_sites"])
        rows[key] = _freeze_row(key, result.as_dict())
    if len(rows) != 216:
        raise ReportSchemaError(f"{len(rows)} intervals produced, not 216")
    return rows


def _freeze_row(key: agg.CIKey, result: dict) -> dict:
    """Copy the primitives the report keeps; retain no shared mutable object."""
    row = {"ci_key": key.as_tuple()}
    for field in INTERVAL_ROW_FIELDS[1:]:
        value = result.get(field)
        row[field] = tuple(value) if isinstance(value, list) else value
    return row


def report_schema_slots() -> tuple:
    """The 216 slots, derived from the frozen registry — never re-enumerated."""
    return tuple(sorted(agg.S2_REQUIRED_CI_KEYS, key=lambda k: k.as_tuple()))


def preflight_ci_schema() -> dict:
    """Schema-level preflight: 216 keys, exact composition, one slot each."""
    slots = report_schema_slots()
    if len(slots) != agg.N_REQUIRED_INTERVALS != 216:
        raise ReportSchemaError(
            f"{len(slots)} report slots vs {agg.N_REQUIRED_INTERVALS} registry keys")
    if len(set(slots)) != len(slots):
        raise ReportSchemaError("duplicate report slot")
    composition = {}
    for key in slots:
        composition[key.family] = composition.get(key.family, 0) + 1
    expected = {agg.FAMILY_FORMAL: 39, agg.FAMILY_EMPIRICAL_FEW_LABEL: 27,
                agg.FAMILY_ZERO_LABEL: 6, agg.FAMILY_DESCRIPTIVE: 108,
                agg.FAMILY_DESCRIPTIVE_CONTRAST: 36}
    if composition != expected:
        raise ReportSchemaError(
            f"registry composition is {composition}, not {expected}")
    for key in slots:
        agg.assert_ci_registered(key)
    return {"n_slots": len(slots), "composition": composition,
            "schema_checksum": sd.sha256_bytes(sd.canonical_json(
                sorted(str(k.as_tuple()) for k in slots)))}


#: Modules whose formal interval creation must route through the gate.
CI_CLOSED_MODULES = ("src/s2_reporting.py", "src/s2_aggregation.py",
                     "src/s2_site_aggregation.py", "scripts/run_rq2_s2.py")
