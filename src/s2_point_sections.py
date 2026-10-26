"""The frozen point-estimate sections of the formal RQ2/S2 report (Stage 2.1b).

Stage 2.0.8 froze `build_formal_s2_report(context, ledger, point_sections)`, and
that third argument was the last place a caller could put a scientific number
into a formal report: the builder checked only that a section's *name* was
registered, so

    build_formal_s2_report(context, ledger,
                           {"n_min_cert_median": {"anything": 1},
                            "oracle_infeasible": {"anything": 0}})

produced a formal report carrying caller-computed values. Stage 2.1b removes the
parameter. Every section is derived **here**, from artifacts the run itself
froze, and the report calls this module rather than accepting its output from
outside.

    context + ledger
        -> authoritative decision store   (46,200 cells + 138 references)
        -> authoritative evaluation store (78,600 artifacts, decision-bound)
        -> FrozenS2Inputs.from_frozen_context(context)  (frozen S0/S1/Gate 1)
        -> the 20 registered sections, each with its own provenance

Two of the twenty are functions of the frozen upstream alone and exist before
phase 1 (`meta_objective_table`, `ess_and_weight_diagnostics`). Fourteen are
functions of the formal decisions and evaluations and cannot exist until the
evaluation manifest is frozen. The remaining four are the §23 sensitivities,
which the frozen protocol *defines* here and **executes in a later sensitivity
stage** — they are registered and classified `later_stage_sensitivity`, and no
placeholder is emitted for them. `REQUIRED_FORMAL_SECTION_NAMES` is therefore
the sixteen the formal S2 run must produce, and the report requires exact set
equality against it: a missing section, an extra section, a duplicated section
and an unregistered name each fail.

**Stage 2.1c — two calculations, corrected.** An independent audit of the
Stage 2.1b wiring found two sections computing a frozen quantity wrongly under a
correct-looking label, and only those two changed:

* `shrinkage_lambda_diagnostics` reported the class *average* of `λ` and the
  pooled `n_c` / `n_c^src`. §12.1-12.2 index the whole blend by the class, and a
  class average hides the configuration §24.10's diagnostic exists to expose —
  one class carrying the blend locally while the other is swamped by the source
  term. It is now class-specific throughout, and the source-dominance flag is
  the frozen **strict** `λ_c < 0.25`, not `<=`.
* `n_to_n_contrast` filtered each arm's estimable sites separately and
  subtracted the two means, so the arms could describe different site sets while
  the row reported a common denominator. Each quantity is now paired on its own
  `isfinite(A) & isfinite(B)` mask.

Neither changed a formula, a threshold, a grid, a count, a guarantee tier, an
interval or an estimand, and no section was added or removed.

Nothing here re-fits, re-estimates or re-selects anything. Pass rates,
attainable coverage, realized-outcome fractions and descriptive observables come
from `s2_site_aggregation`'s own per-site functions; minimum-`N`, censoring and
the burden come from `s2_aggregation`; `λ`, `n0`, ESS and JSD are read from
frozen artifacts; and the oracle diagnostic is read from the evaluation
artifacts, where the frozen `certification.oracle_infeasible` primitive put it.
"""

from __future__ import annotations

import numpy as np

import certification as cert
import s2_aggregation as agg
import s2_decisions as sd
import s2_evaluation_store as evalstore
import s2_inputs as s2inputs
import s2_ledger as ledgermod
import s2_site_aggregation as sagg
import shrinkage as shr
import site_bootstrap as sb

NOT_ESTIMABLE = cert.NOT_ESTIMABLE


class PointSectionError(AssertionError):
    """Raised when a registered section cannot be derived from the artifacts."""


# ---------------------------------------------------------------------------
# the machine-readable section registry
# ---------------------------------------------------------------------------
#: A section exists before phase 1 (a function of the frozen upstream), after
#: phase 2 (a function of this run's own frozen artifacts), or in the later
#: sensitivity stage the frozen §23 assigns it to.
PHASE_INPUT_DERIVED = "input_derived"
PHASE_POST_PHASE2 = "post_phase2"
PHASE_LATER_STAGE = "later_stage_sensitivity"


class SectionSpec:
    """One registered point-estimate section, described mechanically."""

    __slots__ = ("name", "phase", "protocol_ref", "authoritative_source",
                 "required_keys", "key_structure", "derivation")

    def __init__(self, name, phase, protocol_ref, authoritative_source,
                 required_keys, key_structure, derivation):
        self.name = name
        self.phase = phase
        self.protocol_ref = protocol_ref
        self.authoritative_source = authoritative_source
        self.required_keys = tuple(required_keys)
        self.key_structure = key_structure
        self.derivation = derivation

    def as_dict(self) -> dict:
        return {"section": self.name, "phase": self.phase,
                "protocol_ref": self.protocol_ref,
                "authoritative_source": self.authoritative_source,
                "required_keys": list(self.required_keys),
                "key_structure": self.key_structure,
                "derivation": self.derivation}


#: `strategy|method|N` for a per-`N` quantity, `method|N` where the quantity is
#: tier-fixed, `method|site` for a per-site tier quantity.
_K_SMN = "strategy|method|N"
_K_MN = "method|N"
_K_MS = "method|site"

SECTION_REGISTRY = (
    SectionSpec(
        "candidate_ledger", PHASE_POST_PHASE2, "§24.3, §24.4",
        "authoritative decision store (46,200 cells + 138 zero-label references)",
        ("q_values", "cells", "zero_label", "n_cells", "n_zero_label",
         "n_suffix_checks", "fields"),
        "cell tuple (site, repeat, N, strategy, method)",
        "decision artifact ledger fields + certification.assert_eligible_is_suffix"),
    SectionSpec(
        "per_site_value", PHASE_POST_PHASE2, "§19, §24.6, §24.8",
        "authoritative evaluation store + zero-label references",
        ("by_strategy_method_n", "zero_label_by_site", "n_rows",
         "n_valid_repeats", "fields"),
        _K_SMN,
        "s2_site_aggregation.site_* per-site functions"),
    SectionSpec(
        "per_repeat_value", PHASE_POST_PHASE2, "§18, §24.5, §24.7",
        "authoritative evaluation store",
        ("rows", "n_rows", "fields", "repeats", "inference_unit"),
        "cell tuple (site, repeat, N, strategy, method)",
        "frozen evaluation artifact fields, no repeat treated as an inference unit"),
    SectionSpec(
        "expected_uncertified_site_burden", PHASE_POST_PHASE2, "§19, §24.6",
        "authoritative evaluation + decision stores, target_local only",
        ("by_method_n", "strategy", "n_rows"), _K_MN,
        "s2_aggregation.expected_uncertified_site_burden"),
    SectionSpec(
        "n_min_cert_median", PHASE_POST_PHASE2, "§5, §20, §24.7",
        "authoritative decision states across the site's feasible N grid",
        ("by_method_site", "cross_site_by_method", "strategy", "field",
         "n_grid", "exploratory_n"), _K_MS,
        "s2_aggregation.minimum_certifiable_n + censored_order_statistic_summary"),
    SectionSpec(
        "n_min_cert_iqr", PHASE_POST_PHASE2, "§20, §24.7",
        "authoritative decision states across the site's feasible N grid",
        ("by_method_site", "strategy", "field", "order_statistics"), _K_MS,
        "s2_aggregation.censored_order_statistic_summary (Q1, Q3)"),
    SectionSpec(
        "n_min_empirical_pass_median", PHASE_POST_PHASE2, "§6, §20, §24.7",
        "authoritative decision states across the site's feasible N grid",
        ("by_method_site", "cross_site_by_method", "strategy", "field",
         "n_grid", "exploratory_n"), _K_MS,
        "s2_aggregation.minimum_empirical_pass_n + censored_order_statistic_summary"),
    SectionSpec(
        "n_min_site_distribution", PHASE_POST_PHASE2, "§20, §24.7",
        "authoritative decision states across the site's feasible N grid",
        ("formal", "empirical_few_label", "per_repeat_tokens"), _K_MS,
        "per-repeat minimum-N tokens, max_feasible_N_s, non-monotonicity"),
    SectionSpec(
        "pass_rate_extremes", PHASE_POST_PHASE2, "§19, §21.2",
        "authoritative evaluation store + zero-label references",
        ("by_strategy_method_n", "zero_label", "n_rows"), _K_SMN,
        "site-level tier pass rates, extremes and degenerate-site counts"),
    SectionSpec(
        "not_testable_reason_split", PHASE_POST_PHASE2, "§15, §19, §24.6",
        "authoritative decision store",
        ("by_strategy_method_n", "by_site", "reasons", "n_rows"), _K_SMN,
        "s2_decisions reason vocabulary counted over frozen decision artifacts"),
    SectionSpec(
        "oracle_infeasible", PHASE_POST_PHASE2, "§16, §17, §24.5",
        "authoritative evaluation store (post-release diagnostic field)",
        ("by_strategy_method_n", "by_strategy", "grid_by_strategy",
         "n_cells", "vocabulary", "diagnostic_only"), _K_SMN,
        "certification.oracle_infeasible, frozen per-strategy candidate grid"),
    SectionSpec(
        "meta_objective_table", PHASE_INPUT_DERIVED, "§13, §24.9",
        "frozen Gate 1 / RQ1-S0 source-OOF blocks (FrozenS2Inputs)",
        ("d_by_site_and_n0", "selected_n0_by_site", "n0_grid", "tie_break_rule",
         "meta_repeats", "meta_split_entropy", "common_support"),
        "outer_site|n0",
        "s2_inputs.FrozenS2Inputs.meta_objective_section"),
    SectionSpec(
        "ess_and_weight_diagnostics", PHASE_INPUT_DERIVED, "§11, §24.10",
        "frozen RQ1-S1 ess_by_site.csv, selected_domain_C.csv, weights",
        ("ess_by_site", "ess_by_site_columns", "per_candidate_by_site_and_method",
         "per_candidate_fields", "q_values", "weighted_taus_by_site_and_method",
         "c_domain_by_site", "n_frozen_s1_reproduction_checks"),
        "site (artifact row) and site|method (per candidate)",
        "s2_inputs.FrozenS2Inputs.ess_and_weight_diagnostics_section"),
    SectionSpec(
        "shrinkage_lambda_diagnostics", PHASE_POST_PHASE2, "§12, §24.10",
        "authoritative decision store (pooled_local_shrinkage) + frozen n0/source counts",
        ("by_method_n_site", "class_labels", "overall_by_class", "n0_by_site",
         "source_dominated_lambda", "source_dominated_rule",
         "lambda_primitive", "strategy", "n_rows", "n_repeat_rows"),
        "method|N|site, then one row per repeat",
        "class-specific lambda_c = shrinkage_lambda(n_c, n0), n_c, n_c^src and "
        "the source-dominated share (lambda_c < 0.25, strict), per class"),
    SectionSpec(
        "jsd", PHASE_POST_PHASE2, "§24.5",
        "authoritative evaluation store (jsd_keep_vs_all)",
        ("by_strategy_method_n", "n_rows", "primitive"), _K_SMN,
        "frozen s0_evaluation.jsd_composition values, read not recomputed"),
    SectionSpec(
        "n_to_n_contrast", PHASE_POST_PHASE2, "§21.3, §22",
        "authoritative evaluation store, target_local, sites where both arms "
        "are estimable for the quantity",
        ("by_method_pair", "strategy", "pairs", "quantities", "sign_convention",
         "estimability_rule", "carries_ci"), "method|N_low->N_high",
        "site-balanced means on the per-quantity common-estimable site set "
        "(paired), point estimates only"),
    SectionSpec(
        "sensitivity_overall_error_ucb", PHASE_LATER_STAGE, "§23.1", None,
        (), None, None),
    SectionSpec(
        "sensitivity_q_step_002", PHASE_LATER_STAGE, "§23.2", None,
        (), None, None),
    SectionSpec(
        "sensitivity_r_star_shift", PHASE_LATER_STAGE, "§23.3", None,
        (), None, None),
    SectionSpec(
        "sensitivity_common_sites", PHASE_LATER_STAGE, "§23.4", None,
        (), None, None),
)

SECTION_SPECS = {spec.name: spec for spec in SECTION_REGISTRY}

#: Every registered point-estimate name, classified. Mirrors — and is asserted
#: equal to — `s2_reporting.POINT_ESTIMATE_SECTIONS`.
REGISTERED_SECTION_NAMES = tuple(spec.name for spec in SECTION_REGISTRY)

#: The sections the **formal S2 run** must produce. The four §23 sensitivities
#: are excluded by the frozen protocol's own words — "each is defined here,
#: executed in a later sensitivity stage" — not by a choice made here, and they
#: appear in no §24 required-output line.
REQUIRED_FORMAL_SECTION_NAMES = tuple(
    spec.name for spec in SECTION_REGISTRY if spec.phase != PHASE_LATER_STAGE)
LATER_STAGE_SECTION_NAMES = tuple(
    spec.name for spec in SECTION_REGISTRY if spec.phase == PHASE_LATER_STAGE)

if len(set(REGISTERED_SECTION_NAMES)) != len(REGISTERED_SECTION_NAMES):
    raise PointSectionError("a section name is registered twice")

#: The three §22 tier-A N-to-N pairs: ascending, `N >= 10`, no `N = 5` contrast.
N_TO_N_PAIRS = tuple((a, b) for i, a in enumerate(cert.N_GRID[1:])
                     for b in cert.N_GRID[1 + i + 1:])
N_TO_N_QUANTITIES = ("formal_certification_probability",
                     "formal_attainable_coverage",
                     "formal_realized_violation_rate")

#: `N = 5` is exploratory (§4); primary interpretation begins at `N >= 10`.
EXPLORATORY_N = 5


def section_registry_table() -> list:
    """The machine-readable registry, for preflight and audit."""
    return [spec.as_dict() for spec in SECTION_REGISTRY]


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------
def _finite_mean(values) -> float:
    finite = [float(v) for v in values
              if v is not None and v != NOT_ESTIMABLE and np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _key(*parts) -> str:
    return "|".join(str(p) for p in parts)


def _cell_str(key) -> str:
    return "|".join(str(p) for p in key)


def _groups(index) -> list:
    return sorted(index, key=lambda g: (g[0], g[1], g[2]))


# ---------------------------------------------------------------------------
# §24.3 / §24.4 — the candidate ledger
# ---------------------------------------------------------------------------
#: The ledger fields copied out of each frozen decision artifact. Every one of
#: them is written by the decision worker and validated by
#: `s2_decisions.validate_decision_artifact` before it reaches disk.
CANDIDATE_LEDGER_FIELDS = (
    "state", "reason", "score_type", "guarantee", "guarantee_tier",
    "selected_q", "selected_candidate_index", "selected_tau", "selected_score",
    "n_eligible", "first_eligible_index", "n_passed_candidates",
    "stopped_at_index", "evaluated_q", "eligible_q", "score_by_q", "r_star",
)
#: Present on exactly one tier each; copied when the artifact carries them.
CANDIDATE_LEDGER_OPTIONAL_FIELDS = (
    "n_rejected_hypotheses", "nominal_alpha_ucb", "screening_reference",
    "n0", "lambda_local_selected", "scope",
)


def _suffix_check(payload: dict) -> int:
    """§24.4 — eligibility is a contiguous suffix of the frozen order.

    Re-derived from the artifact's own eligible-`q` ledger and checked with the
    frozen primitive, then required to agree with the artifact's recorded
    `first_eligible_index`. The check is a re-verification of a frozen artifact,
    never a repair of one.

    `assert_eligible_is_suffix` returns the raw sentinel `len(flags)` when no
    candidate is eligible, and the frozen producer stores that same case as
    `first_eligible_index = None` (`certification.py`, `first if first < len(qs)
    else None`). The comparison is therefore made against the producer's
    normalisation of the recomputed value, not against the raw sentinel — an
    empty eligible set is `None` on both sides. This normalises the
    *representation* only: the eligibility predicate, the suffix requirement and
    the `n_eligible` cross-check below are untouched.
    """
    eligible = {round(float(q), 2) for q in payload["eligible_q"]}
    flags = [round(float(q), 2) in eligible for q in cert.Q_GRID]
    raw_first = cert.assert_eligible_is_suffix(flags)
    first = raw_first if raw_first < len(flags) else None
    recorded = payload["first_eligible_index"]
    if (first is None) != (recorded is None) or (
            first is not None and int(first) != int(recorded)):
        raise PointSectionError(
            f"decision artifact records first_eligible_index={recorded!r} but "
            f"its eligibility ledger begins at {first!r}")
    if int(payload["n_eligible"]) != sum(flags):
        raise PointSectionError(
            f"decision artifact records n_eligible={payload['n_eligible']!r} "
            f"but its eligibility ledger holds {sum(flags)}")
    return sum(flags)


def _ledger_row(key, payload: dict) -> dict:
    row = {"cell": _cell_str(key)}
    for field in CANDIDATE_LEDGER_FIELDS:
        row[field] = payload[field]
    for field in CANDIDATE_LEDGER_OPTIONAL_FIELDS:
        if field in payload:
            row[field] = payload[field]
    row["n_evaluated_candidates"] = len(payload["evaluated_q"])
    row["suffix_property_ok"] = True
    return row


def _candidate_ledger(decisions: dict, zero_label: dict) -> dict:
    cells, checks = [], 0
    for key in sorted(decisions):
        payload = decisions[key]
        checks += 1
        _suffix_check(payload)
        cells.append(_ledger_row(key, payload))
    references = []
    for key in sorted(zero_label):
        payload = zero_label[key]
        checks += 1
        _suffix_check(payload)
        references.append(_ledger_row(key, payload))
    return {"q_values": list(cert.Q_GRID), "cells": cells,
            "zero_label": references, "n_cells": len(cells),
            "n_zero_label": len(references), "n_suffix_checks": checks,
            "fields": list(CANDIDATE_LEDGER_FIELDS)
                      + list(CANDIDATE_LEDGER_OPTIONAL_FIELDS)}


# ---------------------------------------------------------------------------
# §19 / §24.6 / §24.8 — per-site values
# ---------------------------------------------------------------------------
PER_SITE_FIELDS = (
    "pass_rate_field", "pass_rate", "n_valid_repeats", "n_not_testable",
    "coverage_field", "attainable_coverage", "realized_field",
    "realized_outcome_fraction", "descriptive_target_coverage",
    "descriptive_target_selective_ber", "descriptive_target_risk_exceedance",
)


def _per_site_value(index, zero_label, scope, r_split) -> dict:
    by_group, n_rows = {}, 0
    for group in _groups(index):
        strategy, method, n_labels = group
        rows_by_site = {}
        for site in scope["by_n"][n_labels]:
            rows = sagg.ordered_repeat_rows(index[group][site], r_split)
            states = [r["decision_state"] for r in rows]
            pass_summary = agg.pass_rate_site(strategy, states)
            coverage_summary = agg.attainable_coverage_site(
                strategy, states,
                [sagg.as_float(r["coverage_test"]) for r in rows])
            realized_summary = agg.realized_outcome_fraction_site(
                strategy, states,
                [NOT_ESTIMABLE if r["realized_outcome"] is None
                 else int(r["realized_outcome"]) for r in rows])
            realized_field = realized_summary["realized_field"]
            record = {
                "pass_rate_field": pass_summary["pass_rate_field"],
                "pass_rate": float(pass_summary[pass_summary["pass_rate_field"]]),
                "n_valid_repeats": int(r_split),
                "n_not_testable": sum(1 for s in states
                                      if s == cert.NOT_TESTABLE),
                "coverage_field": coverage_summary["coverage_field"],
                "attainable_coverage": sagg.as_float(
                    coverage_summary[coverage_summary["coverage_field"]]),
                "realized_field": realized_field,
                "realized_outcome_fraction": sagg.as_float(
                    realized_summary[f"{realized_field}_fraction"]),
            }
            for quantity in agg.DESCRIPTIVE_QUANTITIES:
                record[quantity] = sagg.site_descriptive_value(rows, quantity)
            rows_by_site[site] = record
            n_rows += 1
        by_group[_key(strategy, method, n_labels)] = rows_by_site
    zero = {}
    for (site, strategy, method), payload in sorted(zero_label.items()):
        zero[_key(strategy, method, site)] = {
            "state": payload["state"],
            "pass_rate": sagg.site_zero_label_pass(site, strategy, method,
                                                   zero_label),
            "selected_q": payload["selected_q"],
            "selected_tau": payload["selected_tau"],
            "guarantee_tier": payload["guarantee_tier"],
        }
    return {"by_strategy_method_n": by_group, "zero_label_by_site": zero,
            "n_rows": n_rows, "n_valid_repeats": int(r_split),
            "fields": list(PER_SITE_FIELDS)}


# ---------------------------------------------------------------------------
# §18 / §24.5 / §24.7 — per-repeat values
# ---------------------------------------------------------------------------
PER_REPEAT_FIELDS = (
    "decision_state", "evaluation_status", "selected_q", "selected_tau",
    "coverage_test", "n_test", "n_keep_test", "n_keep_test_MDD",
    "n_keep_test_HC", "ber_test", "risk_excess", "realized_field",
    "realized_outcome", "jsd_keep_vs_all", "oracle_infeasible",
    "n_oracle_feasible_candidates",
)


def _per_repeat_value(index, scope, r_split) -> dict:
    rows = {}
    for group in _groups(index):
        strategy, method, n_labels = group
        for site in scope["by_n"][n_labels]:
            by_repeat = index[group][site]
            for repeat in range(r_split):
                payload = by_repeat[repeat]
                key = _cell_str((site, repeat, n_labels, strategy, method))
                rows[key] = {field: payload.get(field)
                             for field in PER_REPEAT_FIELDS}
    return {"rows": rows, "n_rows": len(rows),
            "fields": list(PER_REPEAT_FIELDS),
            "repeats": [0, int(r_split) - 1],
            "inference_unit": "site"}


# ---------------------------------------------------------------------------
# §19 / §24.6 — the formal uncertified-site burden
# ---------------------------------------------------------------------------
def _reason_family(reason) -> str:
    if reason is None:
        return "NONE"
    if reason == sd.REASON_NO_PASS:
        return cert.REASON_NO_PASS
    if reason in sd.NOT_TESTABLE_REASONS:
        return cert.REASON_NO_ELIGIBLE
    raise PointSectionError(f"{reason!r} is not a frozen decision reason")


def _burden(index, decisions, scope, r_split) -> dict:
    strategy = cert.FORMAL_STRATEGY
    out, n_rows = {}, 0
    for method in cert.CONFIDENCE_METHODS:
        for n_labels in cert.N_GRID:
            sites = scope["by_n"][n_labels]
            p_uncertified, at_zero, at_one, split = [], [], [], {}
            for site in sites:
                rows = sagg.ordered_repeat_rows(
                    index[(strategy, method, n_labels)][site], r_split)
                p = sagg.site_pass_probability(rows, strategy)
                p_uncertified.append(1.0 - p)
                if p == 0.0:
                    at_zero.append(site)
                if p == 1.0:
                    at_one.append(site)
                for repeat in range(r_split):
                    payload = decisions[(site, repeat, n_labels, strategy,
                                         method)]
                    if cert.is_pass(payload["state"]):
                        continue
                    family = _reason_family(payload["reason"])
                    split[family] = split.get(family, 0) + 1
            out[_key(method, n_labels)] = {
                "expected_uncertified_site_burden":
                    agg.expected_uncertified_site_burden(p_uncertified),
                "n_sites": len(sites),
                "sites": list(sites),
                "n_sites_at_zero": len(at_zero),
                "sites_at_zero": at_zero,
                "n_sites_at_one": len(at_one),
                "sites_at_one": at_one,
                "not_certified_reason_split": split,
                "exploratory": bool(n_labels == EXPLORATORY_N),
            }
            n_rows += 1
    return {"by_method_n": out, "strategy": strategy, "n_rows": n_rows}


# ---------------------------------------------------------------------------
# §5 / §6 / §20 / §24.7 — the minimum-`N` family
# ---------------------------------------------------------------------------
def _min_n_trajectories(decisions, scope, strategy, method, site,
                        r_split) -> dict:
    """The site's per-repeat minimum-`N` tokens, with censoring diagnostics."""
    feasible = sagg.feasible_n_for_site(site, scope)
    field = ("n_min_cert" if strategy == cert.FORMAL_STRATEGY
             else "n_min_empirical_pass")
    values, non_monotone, max_feasible = [], 0, None
    for repeat in range(r_split):
        state_by_n = {n: decisions[(site, repeat, n, strategy, method)]["state"]
                      for n in feasible}
        if strategy == cert.FORMAL_STRATEGY:
            out = agg.minimum_certifiable_n(strategy, state_by_n, feasible)
        else:
            out = agg.minimum_empirical_pass_n(strategy, state_by_n, feasible)
        values.append(out[field])
        non_monotone += int(bool(out["non_monotone"]))
        max_feasible = out["max_feasible_N"]
    summary = agg.censored_order_statistic_summary(values, r_split)
    return {"field": field, "values": values, "summary": summary,
            "max_feasible_N": max_feasible, "n_non_monotone": non_monotone,
            "feasible_n": list(feasible)}


def _min_n_tables(decisions, scope, r_split) -> dict:
    """One pass over both tiers; the four registered sections read from it."""
    tables = {}
    for strategy in (cert.FORMAL_STRATEGY, agg.EMPIRICAL_FEW_LABEL_STRATEGY):
        per_site = {}
        for method in cert.CONFIDENCE_METHODS:
            for site in scope["all_sites"]:
                per_site[_key(method, site)] = _min_n_trajectories(
                    decisions, scope, strategy, method, site, r_split)
        tables[strategy] = per_site
    return tables


def _median_section(tables, strategy) -> dict:
    per_site = tables[strategy]
    field = ("n_min_cert" if strategy == cert.FORMAL_STRATEGY
             else "n_min_empirical_pass")
    by_site, cross = {}, {}
    for key, entry in sorted(per_site.items()):
        by_site[key] = {
            f"{field}_median": entry["summary"]["median"],
            "n_censored_repeats": entry["summary"]["n_censored"],
            "p_censored": entry["summary"]["p_censored"],
            "max_feasible_N": entry["max_feasible_N"],
            "n_non_monotone_repeats": entry["n_non_monotone"],
            "feasible_n": entry["feasible_n"],
        }
    for method in cert.CONFIDENCE_METHODS:
        medians = [by_site[_key(method, s)][f"{field}_median"]
                   for s in sorted(
                       {k.split("|")[1] for k in per_site
                        if k.split("|")[0] == method})]
        finite = [float(m) for m in medians if not isinstance(m, str)]
        cross[method] = {
            "site_medians": medians,
            "n_sites": len(medians),
            "n_censored_sites": sum(1 for m in medians if isinstance(m, str)),
            "median_of_site_medians": (float(np.median(finite)) if finite
                                       else agg.CENSORED),
            "site_balanced_p_censored": float(np.mean(
                [by_site[_key(method, s)]["p_censored"]
                 for s in sorted({k.split("|")[1] for k in per_site
                                  if k.split("|")[0] == method})])),
        }
    return {"by_method_site": by_site, "cross_site_by_method": cross,
            "strategy": strategy, "field": field,
            "n_grid": list(cert.N_GRID), "exploratory_n": EXPLORATORY_N}


def _iqr_section(tables) -> dict:
    strategy = cert.FORMAL_STRATEGY
    per_site = tables[strategy]
    by_site = {}
    for key, entry in sorted(per_site.items()):
        q1, q3 = entry["summary"]["q1"], entry["summary"]["q3"]
        both_finite = not isinstance(q1, str) and not isinstance(q3, str)
        by_site[key] = {
            "n_min_cert_q1": q1, "n_min_cert_q3": q3,
            "n_min_cert_iqr": (float(q3) - float(q1)) if both_finite
                              else agg.CENSORED,
            "n_censored_repeats": entry["summary"]["n_censored"],
        }
    return {"by_method_site": by_site, "strategy": strategy,
            "field": "n_min_cert",
            "order_statistics": dict(agg.CENSORING_ORDER_STATISTICS)}


def _distribution_section(tables) -> dict:
    out, tokens = {}, {}
    for strategy, label in ((cert.FORMAL_STRATEGY, "formal"),
                            (agg.EMPIRICAL_FEW_LABEL_STRATEGY,
                             "empirical_few_label")):
        per_site = tables[strategy]
        field = ("n_min_cert" if strategy == cert.FORMAL_STRATEGY
                 else "n_min_empirical_pass")
        by_site = {}
        for key, entry in sorted(per_site.items()):
            by_site[key] = {
                "field": field,
                "median": entry["summary"]["median"],
                "q1": entry["summary"]["q1"], "q3": entry["summary"]["q3"],
                "n_repeats": entry["summary"]["n_repeats"],
                "n_censored": entry["summary"]["n_censored"],
                "p_censored": entry["summary"]["p_censored"],
                "max_feasible_N": entry["max_feasible_N"],
                "n_non_monotone_repeats": entry["n_non_monotone"],
            }
            tokens[_key(strategy, key)] = list(entry["values"])
        out[label] = {"strategy": strategy, "field": field,
                      "by_method_site": by_site}
    return {"formal": out["formal"],
            "empirical_few_label": out["empirical_few_label"],
            "per_repeat_tokens": tokens}


# ---------------------------------------------------------------------------
# §19 / §21.2 — pass-rate extremes
# ---------------------------------------------------------------------------
def _pass_rate_extremes(index, zero_label, scope, r_split) -> dict:
    out, n_rows = {}, 0
    for group in _groups(index):
        strategy, method, n_labels = group
        sites = scope["by_n"][n_labels]
        rates = {}
        for site in sites:
            rows = sagg.ordered_repeat_rows(index[group][site], r_split)
            rates[site] = sagg.site_pass_probability(rows, strategy)
        lo = min(rates.values())
        hi = max(rates.values())
        out[_key(strategy, method, n_labels)] = {
            "pass_rate_field": agg.PASS_RATE_FIELD[cert.tier_of(strategy)],
            "min_pass_rate": lo,
            "min_pass_rate_sites": sorted(s for s, v in rates.items() if v == lo),
            "max_pass_rate": hi,
            "max_pass_rate_sites": sorted(s for s, v in rates.items() if v == hi),
            "n_sites_at_zero": sum(1 for v in rates.values() if v == 0.0),
            "n_sites_at_one": sum(1 for v in rates.values() if v == 1.0),
            "n_sites": len(sites),
        }
        n_rows += 1
    # Zero-label pass rates have no `N` axis and are degenerate in {0, 1}
    # (§3.1, §19); they are reported beside the per-`N` extremes, never inside
    # them, so a degenerate reference can never be read as a site distribution.
    zero = {}
    for (site, strategy, method), payload in sorted(zero_label.items()):
        entry = zero.setdefault(_key(strategy, method), {
            "pass_rate_field": agg.PASS_RATE_FIELD[cert.tier_of(strategy)],
            "n_sites_at_zero": 0, "n_sites_at_one": 0,
            "n_sites": 0, "degenerate": True})
        passing = cert.is_pass(payload["state"])
        entry["n_sites_at_one"] += int(passing)
        entry["n_sites_at_zero"] += int(not passing)
        entry["n_sites"] += 1
    for entry in zero.values():
        if entry["n_sites_at_zero"] and entry["n_sites_at_one"]:
            entry["degenerate"] = False
    return {"by_strategy_method_n": out, "zero_label": zero, "n_rows": n_rows}


# ---------------------------------------------------------------------------
# §15 / §19 / §24.6 — the NOT_TESTABLE reason split
# ---------------------------------------------------------------------------
def _not_testable_reason_split(decisions, zero_label) -> dict:
    by_group, by_site, n_rows = {}, {}, 0
    for key in sorted(decisions):
        site, repeat, n_labels, strategy, method = key
        payload = decisions[key]
        reason = payload["reason"]
        if reason not in sd.VALID_REASONS:
            raise PointSectionError(f"{reason!r} is not a frozen reason")
        group = by_group.setdefault(_key(strategy, method, n_labels), {
            "n_cells": 0, "n_not_testable": 0, "by_state": {},
            "by_reason": {}, "by_reason_family": {}})
        group["n_cells"] += 1
        group["by_state"][payload["state"]] = \
            group["by_state"].get(payload["state"], 0) + 1
        if payload["state"] == cert.NOT_TESTABLE:
            group["n_not_testable"] += 1
        if reason is not None:
            group["by_reason"][reason] = group["by_reason"].get(reason, 0) + 1
            family = _reason_family(reason)
            group["by_reason_family"][family] = \
                group["by_reason_family"].get(family, 0) + 1
            cell = by_site.setdefault(_key(strategy, method, n_labels, site), {})
            cell[reason] = cell.get(reason, 0) + 1
    n_rows = len(by_group)
    zero = {}
    for (site, strategy, method), payload in sorted(zero_label.items()):
        entry = zero.setdefault(_key(strategy, method), {})
        label = payload["reason"] or "NONE"
        entry[label] = entry.get(label, 0) + 1
    return {"by_strategy_method_n": by_group, "by_site": by_site,
            "zero_label": zero,
            "reasons": list(sd.VALID_REASONS), "n_rows": n_rows}


# ---------------------------------------------------------------------------
# §16 / §17 / §24.5 — the oracle diagnostic
# ---------------------------------------------------------------------------
ORACLE_VOCABULARY = (
    "oracle-infeasible is a post-hoc deployability diagnostic on the released "
    "target test split. It is not `uncertified` (a calibration-side failure to "
    "reject) and not a `realized violation` (an adopted threshold whose actual "
    "risk exceeded r*_s); the three may never be equated or summed.")


def _oracle_section(index, scope, r_split) -> dict:
    by_group, by_strategy, n_cells = {}, {}, 0
    for group in _groups(index):
        strategy, method, n_labels = group
        per_site = {}
        for site in scope["by_n"][n_labels]:
            rows = sagg.ordered_repeat_rows(index[group][site], r_split)
            flags, feasible = [], []
            for row in rows:
                if row.get("oracle_infeasible") is None:
                    raise PointSectionError(
                        f"evaluation artifact for {site}/{group} carries no "
                        "oracle diagnostic; the frozen §24.5 output requires "
                        "one for every evaluated cell")
                flags.append(int(row["oracle_infeasible"]))
                feasible.append(int(row["n_oracle_feasible_candidates"]))
            per_site[site] = {
                "oracle_infeasible_fraction": float(np.mean(flags)),
                "n_oracle_infeasible_repeats": int(sum(flags)),
                "mean_feasible_candidates": float(np.mean(feasible)),
                "n_repeats": len(flags),
            }
            n_cells += len(flags)
        by_group[_key(strategy, method, n_labels)] = per_site
        agg_strategy = by_strategy.setdefault(
            strategy, {"n_repeats": 0, "n_oracle_infeasible": 0})
        for record in per_site.values():
            agg_strategy["n_repeats"] += record["n_repeats"]
            agg_strategy["n_oracle_infeasible"] += \
                record["n_oracle_infeasible_repeats"]
    for record in by_strategy.values():
        record["oracle_infeasible_fraction"] = (
            record["n_oracle_infeasible"] / record["n_repeats"]
            if record["n_repeats"] else float("nan"))
    grid = {s: s2inputs.STRATEGY_THRESHOLD_GRID[s] for s in cert.STRATEGIES}
    return {"by_strategy_method_n": by_group, "by_strategy": by_strategy,
            "grid_by_strategy": grid, "n_cells": n_cells,
            "vocabulary": ORACLE_VOCABULARY, "diagnostic_only": True,
            "c_min": cert.C_MIN, "m_min": cert.M_MIN}


# ---------------------------------------------------------------------------
# §12 / §24.10 — the shrinkage diagnostics
# ---------------------------------------------------------------------------
#: §12.1–12.2 index the whole blend by the class `c`: `r_c^loc`, `r_c^src`,
#: `lambda_c = n_c / (n_c + n0)` and `r_c^shrunk` are per-class quantities, and
#: §24.10 requires `λ_c`, `n_c`, `n_c^src` and the source-dominated share as
#: diagnostics. Stage 2.1c restores that index: Stage 2.1b reported the mean of
#: the two class lambdas and the pooled counts, which is not `λ_c` and can hide
#: a class whose local calibration count is swamped by the source prior while
#: the other class is locally dominated.
SHRINKAGE_CLASSES = ("MDD", "HC")

#: `(n_c, x_c, ...)` layout of one frozen source-class-count row, as written by
#: the frozen S0 block and consumed by `s2_cell_worker`:
#: `(n_MDD^src, x_MDD^src, n_HC^src, x_HC^src)`.
SOURCE_COUNT_COLUMN = {"MDD": 0, "HC": 2}

#: §12.5 / F4 — *strictly* below. `λ_c = 0.25` is **not** a source-dominated
#: pass: the frozen text reads `λ_c < 0.25`, and at `λ_c = 0.25` the local block
#: carries a third of the blend's weight relative to the prior, exactly on the
#: boundary the protocol placed outside the flag.
SOURCE_DOMINATED_RULE = "lambda_c < 0.25 (strict; equality is not dominated)"


def _is_source_dominated(lam: float) -> bool:
    """The frozen §12.5 flag, per class, strict."""
    return bool(float(lam) < float(shr.SOURCE_DOMINATED_LAMBDA))


def _class_lambda_row(payload, source_counts, n0, cell) -> dict:
    """One passing cell's class-specific `λ_c`, `n_c`, `n_c^src` and flags.

    `λ_c` is not read off the artifact — the artifact's `lambda_local_selected`
    is the mean of the two classes, which is the quantity this stage had to stop
    reporting. It is re-derived per class from the frozen primitive
    `shrinkage.shrinkage_lambda(n_c, n0)` on the artifact's own retained
    calibration counts and its own `n0` at the selected candidate, and the mean
    of the two is then checked against the artifact so the derivation is
    provably the frozen one decomposed, not a second definition.
    """
    index = payload["selected_candidate_index"]
    row = {"repeat": int(payload["repeat"]), "state": payload["state"],
           "selected_candidate_index": int(index),
           "selected_q": payload["selected_q"], "n0": float(n0)}
    lambdas = []
    for cls in SHRINKAGE_CLASSES:
        n_local = int(payload[f"n_keep_cal_{cls}"])
        n_src = int(source_counts[index][SOURCE_COUNT_COLUMN[cls]])
        lam = float(shr.shrinkage_lambda(n_local, n0))
        lambdas.append(lam)
        row[f"lambda_{cls}"] = lam
        row[f"n_{cls}"] = n_local
        row[f"n_{cls}_source"] = n_src
        row[f"source_dominated_{cls}"] = _is_source_dominated(lam)
    frozen = payload["lambda_local_selected"]
    if frozen is None:
        raise PointSectionError(
            f"shrinkage decision {cell} selected a candidate but carries no "
            "lambda")
    if abs(0.5 * (lambdas[0] + lambdas[1]) - float(frozen)) > 1e-12:
        raise PointSectionError(
            f"shrinkage decision {cell}: the class-specific lambdas "
            f"{lambdas} do not reproduce the artifact's frozen "
            f"lambda_local_selected {frozen!r}")
    row["source_dominated_any_class"] = any(
        row[f"source_dominated_{c}"] for c in SHRINKAGE_CLASSES)
    row["source_dominated_both_classes"] = all(
        row[f"source_dominated_{c}"] for c in SHRINKAGE_CLASSES)
    return row


def _shrinkage_section(decisions, inputs, scope, r_split) -> dict:
    """§24.10 — `λ_c`, `n_c`, `n_c^src` and the source-dominated share, per class.

    One row per `site × N × repeat × method`. A repeat that adopted no
    threshold is kept with `null` class quantities and no flag: it is not a
    passing cell, so it enters neither numerator nor denominator of the share,
    and it is never read as `λ = 0`.
    """
    strategy = agg.EMPIRICAL_FEW_LABEL_STRATEGY
    out, n_rows, n_repeat_rows = {}, 0, 0
    overall = {cls: {"n_passing_cells": 0, "n_source_dominated_cells": 0}
               for cls in SHRINKAGE_CLASSES}
    for method in cert.CONFIDENCE_METHODS:
        for n_labels in cert.N_GRID:
            for site in scope["by_n"][n_labels]:
                n0 = float(inputs.n0_for(site))
                source_counts = inputs.source_class_counts(site, method)
                per_repeat, passing = [], []
                for repeat in range(r_split):
                    cell = (site, repeat, n_labels, strategy, method)
                    payload = decisions[cell]
                    # §12.5 — one n0 per outer fold, shared across N, methods
                    # and repeats. The artifact's own n0 is what produced its
                    # lambda, so it is what the class decomposition must use;
                    # it has to *be* the fold's frozen value.
                    if float(payload["n0"]) != n0:
                        raise PointSectionError(
                            f"shrinkage decision {_cell_str(cell)} carries n0 "
                            f"{payload['n0']!r}; the frozen fold value is {n0}")
                    index = payload["selected_candidate_index"]
                    is_pass = cert.is_pass(payload["state"])
                    if (index is None) == is_pass:
                        raise PointSectionError(
                            f"shrinkage decision {_cell_str(cell)}: state "
                            f"{payload['state']!r} disagrees with "
                            f"selected_candidate_index {index!r}")
                    if index is None:
                        row = {"repeat": repeat, "state": payload["state"],
                               "selected_candidate_index": None,
                               "selected_q": None, "n0": n0,
                               "source_dominated_any_class": None,
                               "source_dominated_both_classes": None}
                        for cls in SHRINKAGE_CLASSES:
                            row[f"lambda_{cls}"] = None
                            row[f"n_{cls}"] = None
                            row[f"n_{cls}_source"] = None
                            row[f"source_dominated_{cls}"] = None
                    else:
                        row = _class_lambda_row(payload, source_counts, n0,
                                                _cell_str(cell))
                        passing.append(row)
                    per_repeat.append(row)
                    n_repeat_rows += 1

                record = {
                    "n_repeats": r_split,
                    "n_passing_repeats": len(passing),
                    "n0": n0,
                    "per_repeat": per_repeat,
                }
                for cls in SHRINKAGE_CLASSES:
                    dominated = sum(1 for row in passing
                                    if row[f"source_dominated_{cls}"])
                    record[f"mean_lambda_{cls}"] = _finite_mean(
                        [row[f"lambda_{cls}"] for row in passing])
                    record[f"mean_n_{cls}"] = _finite_mean(
                        [row[f"n_{cls}"] for row in passing])
                    record[f"mean_n_{cls}_source"] = _finite_mean(
                        [row[f"n_{cls}_source"] for row in passing])
                    record[f"n_source_dominated_{cls}"] = dominated
                    record[f"source_dominated_share_{cls}"] = (
                        dominated / len(passing) if passing else float("nan"))
                    overall[cls]["n_passing_cells"] += len(passing)
                    overall[cls]["n_source_dominated_cells"] += dominated
                for both in ("any_class", "both_classes"):
                    hits = sum(1 for row in passing
                               if row[f"source_dominated_{both}"])
                    record[f"n_source_dominated_{both}"] = hits
                    record[f"source_dominated_share_{both}"] = (
                        hits / len(passing) if passing else float("nan"))
                out[_key(method, n_labels, site)] = record
                n_rows += 1

    for cls in SHRINKAGE_CLASSES:
        cells = overall[cls]["n_passing_cells"]
        overall[cls]["source_dominated_share"] = (
            overall[cls]["n_source_dominated_cells"] / cells if cells
            else float("nan"))
    return {"by_method_n_site": out,
            "class_labels": list(SHRINKAGE_CLASSES),
            "overall_by_class": overall,
            "n0_by_site": {s: inputs.n0_for(s) for s in scope["all_sites"]},
            "source_dominated_lambda": float(shr.SOURCE_DOMINATED_LAMBDA),
            "source_dominated_rule": SOURCE_DOMINATED_RULE,
            "lambda_primitive": "shrinkage.shrinkage_lambda(n_c, n0), per class",
            "strategy": strategy, "n_rows": n_rows,
            "n_repeat_rows": n_repeat_rows}


# ---------------------------------------------------------------------------
# §24.5 — JSD
# ---------------------------------------------------------------------------
def _jsd_section(index, scope, r_split) -> dict:
    out, n_rows = {}, 0
    for group in _groups(index):
        strategy, method, n_labels = group
        per_site = {}
        for site in scope["by_n"][n_labels]:
            rows = sagg.ordered_repeat_rows(index[group][site], r_split)
            values = [r["jsd_keep_vs_all"] for r in rows]
            per_site[site] = {
                "mean_jsd_keep_vs_all": _finite_mean(values),
                "n_estimable_repeats": sum(1 for v in values if v is not None),
                "n_repeats": len(values),
            }
        out[_key(strategy, method, n_labels)] = per_site
        n_rows += 1
    return {"by_strategy_method_n": out, "n_rows": n_rows,
            "primitive": "s0_evaluation.jsd_composition (frozen, read only)"}


# ---------------------------------------------------------------------------
# §21.3 / §22 — the tier-A N-to-N contrasts
# ---------------------------------------------------------------------------
def _site_quantity(rows, quantity, strategy) -> float:
    """One site's value of one N-to-N quantity, `NaN` where `NOT_ESTIMABLE`."""
    if quantity == "formal_certification_probability":
        return sagg.site_pass_probability(rows, strategy)
    if quantity == "formal_attainable_coverage":
        return sagg.site_attainable_coverage(rows, strategy)
    if quantity == "formal_realized_violation_rate":
        return sagg.site_realized_outcome_fraction(rows, strategy)
    raise PointSectionError(f"{quantity!r} is not an N-to-N quantity")


def _paired_point_contrast(sites, values_low, values_high, low, high) -> dict:
    """One quantity's paired point contrast on its own common-finite site set.

    `sites` is the site universe of the pair; `values_low` / `values_high` carry
    one value per site in that order, `NaN` where the site is `NOT_ESTIMABLE`.
    Both arms are restricted to `isfinite(A) & isfinite(B)` **before** anything
    is averaged, so the reported arm means, the difference and the denominator
    all describe the same sites.
    """
    a = np.asarray(values_low, dtype=float).ravel()
    b = np.asarray(values_high, dtype=float).ravel()
    if a.shape != b.shape or a.size != len(sites):
        raise PointSectionError(
            "an N-to-N contrast needs one value per site in each arm")
    common = np.isfinite(a) & np.isfinite(b)
    mean_low = sb.point_estimate(np.where(common, a, np.nan))
    mean_high = sb.point_estimate(np.where(common, b, np.nan))
    paired = sb.point_estimate(np.where(common, b - a, np.nan))
    if np.isfinite(paired) and abs(paired - (mean_high - mean_low)) > 1e-9:
        raise PointSectionError(
            f"N{low}->N{high}: the paired mean difference {paired!r} disagrees "
            f"with the difference of the restricted arm means — the two arms "
            "were not averaged over the same sites")
    return {
        f"N{low}": mean_low,
        f"N{high}": mean_high,
        "difference": paired,
        "n_common_estimable_sites": int(common.sum()),
        "common_estimable_sites": [site for site, keep in zip(sites, common)
                                   if keep],
        f"n_estimable_sites_N{low}": int(np.isfinite(a).sum()),
        f"n_estimable_sites_N{high}": int(np.isfinite(b).sum()),
    }


def _n_to_n(index, scope, r_split) -> dict:
    """§21.3 / §22 — the tier-A N-to-N contrasts, **paired on common-finite sites**.

    Stage 2.1b filtered each arm's estimable sites separately and subtracted the
    two means. That is a difference of means over *different* site sets: a site
    that is `NOT_ESTIMABLE` at one `N` still moved the other arm's mean, and the
    reported denominator was the eligible-site count rather than the number of
    sites that actually contributed to both. Stage 2.1c builds one mask per
    quantity,

        common_Q = isfinite(A_Q) & isfinite(B_Q)

    restricts **both** arms to it before anything is averaged — the same rule
    `site_bootstrap.paired_contrast` applies to the registered intervals — and
    reports `n_common_estimable_sites = sum(common_Q)`.

    Each quantity gets its own mask. Coverage and the violation rate become
    `NOT_ESTIMABLE` under different conditions, and the frozen protocol asks for
    a common-*estimable* comparison per contrast, not a common estimability
    class across quantities. Nothing here is imputed: a `NOT_ESTIMABLE` site is
    dropped from that quantity's pair, not read as zero, and it keeps its other
    quantities.

    The estimand is unchanged and still point-only: no interval, no p-value, no
    test, no registry entry.
    """
    strategy = cert.FORMAL_STRATEGY
    out = {}
    for method in cert.CONFIDENCE_METHODS:
        for low, high in N_TO_N_PAIRS:
            universe = sorted(set(scope["by_n"][low]) & set(scope["by_n"][high]))
            record = {"n_sites_eligible_both_arms": len(universe),
                      "sites_eligible_both_arms": universe}
            rows_by_n = {
                n: [sagg.ordered_repeat_rows(
                    index[(strategy, method, n)][site], r_split)
                    for site in universe]
                for n in (low, high)}
            for quantity in N_TO_N_QUANTITIES:
                a = [_site_quantity(rows, quantity, strategy)
                     for rows in rows_by_n[low]]
                b = [_site_quantity(rows, quantity, strategy)
                     for rows in rows_by_n[high]]
                record[quantity] = _paired_point_contrast(universe, a, b,
                                                          low, high)
            out[f"{method}|N{low}->N{high}"] = record
    return {"by_method_pair": out, "strategy": strategy,
            "pairs": [list(p) for p in N_TO_N_PAIRS],
            "quantities": list(N_TO_N_QUANTITIES),
            "sign_convention":
                "higher_N minus lower_N, paired on the sites where both arms "
                "are estimable for that quantity",
            "estimability_rule":
                "per-quantity mask isfinite(A_Q) & isfinite(B_Q); "
                "NOT_ESTIMABLE is never imputed and never dropped from the "
                "site's other quantities",
            "carries_ci": False}


# ---------------------------------------------------------------------------
# the production entry point
# ---------------------------------------------------------------------------
def derive_authoritative_formal_point_sections(context, ledger) -> dict:
    """**The only production source of the formal report's point sections.**

    Two arguments, both authority, neither data. There is no `point_sections`
    parameter, no section dict, no dataframe and no array through which a caller
    could supply one; a caller who wants a different reported section has to
    write different decision or evaluation artifacts, and those fail their
    binding.

    It runs after the run's own barrier, and it verifies that it did: the ledger
    must be this run's allocator, must carry every aggregation prerequisite
    (through `EVALUATION_MANIFEST_FROZEN`), the decision store must *be* the
    authoritative 46,200 with its 138 references, and the evaluation store must
    *be* the authoritative expected set with the manifest checksum the ledger
    froze. The oracle diagnostic is read from those evaluation artifacts, which
    is the only place it can come from — nothing here opens a target outcome.
    """
    import s2_firewall as fw
    from s2_production import ProductionAuthorityError, ProductionS2Context

    if not isinstance(context, ProductionS2Context):
        raise ProductionAuthorityError(
            "formal point sections are derived under a ProductionS2Context")
    context.assert_still_authoritative()
    ledgermod.assert_run_ledger_for_context(ledger, context)
    ledgermod.assert_global_ordering(
        ledger, required=ledgermod.AGGREGATION_PREREQUISITE_EVENTS)

    decision_scan = fw.assert_decision_store_is_complete(context)
    decisions = decision_scan["records"]
    zero_label = evalstore.load_zero_label_decisions(context)
    scan = evalstore.assert_evaluation_store_is_complete(context)
    frozen_manifest = ledger.provenance.get("evaluation_manifest_checksum")
    if frozen_manifest != scan["evaluation_manifest_checksum"]:
        raise ProductionAuthorityError(
            f"the ledger froze evaluation manifest {frozen_manifest!r}; the "
            f"authoritative store recomputes to "
            f"{scan['evaluation_manifest_checksum']!r}")

    scope = sagg.authoritative_site_scope(context)
    index = sagg.index_evaluation_records(scan["records"])
    sagg.assert_cell_identities(index, scope, scan["expected"])
    sagg.assert_zero_label_identities(zero_label, scope)
    r_split = cert.R_SPLIT

    inputs = s2inputs.FrozenS2Inputs.from_frozen_context(context)
    min_n_tables = _min_n_tables(decisions, scope, r_split)

    payloads = {
        "candidate_ledger": _candidate_ledger(decisions, zero_label),
        "per_site_value": _per_site_value(index, zero_label, scope, r_split),
        "per_repeat_value": _per_repeat_value(index, scope, r_split),
        "expected_uncertified_site_burden": _burden(index, decisions, scope,
                                                    r_split),
        "n_min_cert_median": _median_section(min_n_tables,
                                             cert.FORMAL_STRATEGY),
        "n_min_cert_iqr": _iqr_section(min_n_tables),
        "n_min_empirical_pass_median": _median_section(
            min_n_tables, agg.EMPIRICAL_FEW_LABEL_STRATEGY),
        "n_min_site_distribution": _distribution_section(min_n_tables),
        "pass_rate_extremes": _pass_rate_extremes(index, zero_label, scope,
                                                  r_split),
        "not_testable_reason_split": _not_testable_reason_split(
            decisions, zero_label),
        "oracle_infeasible": _oracle_section(index, scope, r_split),
        "meta_objective_table": inputs.meta_objective_section(),
        "ess_and_weight_diagnostics": inputs.ess_and_weight_diagnostics_section(),
        "shrinkage_lambda_diagnostics": _shrinkage_section(decisions, inputs,
                                                           scope, r_split),
        "jsd": _jsd_section(index, scope, r_split),
        "n_to_n_contrast": _n_to_n(index, scope, r_split),
    }

    provenance_base = {
        "protocol_version": sd.PROTOCOL_VERSION,
        "run_id": context.run_id,
        "ledger_id": ledger.ledger_id,
        "decision_manifest_checksum":
            decision_scan["decision_manifest_checksum"],
        "evaluation_manifest_checksum": scan["evaluation_manifest_checksum"],
        "split_registry_checksum":
            context.snapshot["split_registry_checksum"],
    }
    upstream = {row["artifact"]: row["sha256"]
                for row in inputs.provenance["artifacts"]}

    sections = {}
    for name, payload in payloads.items():
        sections[name] = _build_section(name, payload, provenance_base,
                                        upstream)
    assert_sections_complete(sections)
    return sections


def _build_section(name: str, payload: dict, provenance_base: dict,
                   upstream: dict) -> dict:
    """Wrap one derived payload with the provenance §26 requires it to carry."""
    spec = SECTION_SPECS.get(name)
    if spec is None:
        raise PointSectionError(f"{name!r} is not a registered point section")
    if spec.phase == PHASE_LATER_STAGE:
        raise PointSectionError(
            f"{name!r} is a §23 sensitivity, executed in the later sensitivity "
            "stage; the formal S2 report does not derive one")
    missing = [key for key in spec.required_keys if key not in payload]
    if missing:
        raise PointSectionError(f"section {name!r} is missing {missing}")
    provenance = dict(provenance_base)
    provenance.update({
        "section": name,
        "phase": spec.phase,
        "protocol_ref": spec.protocol_ref,
        "authoritative_source": spec.authoritative_source,
        "key_structure": spec.key_structure,
        "derivation": spec.derivation,
    })
    if spec.phase == PHASE_INPUT_DERIVED:
        provenance["upstream_artifact_checksums"] = dict(upstream)
    return {"payload": payload, "provenance": provenance}


def assert_sections_complete(sections) -> dict:
    """Exact set equality against the frozen required section names."""
    actual = set(sections)
    required = set(REQUIRED_FORMAL_SECTION_NAMES)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing or extra:
        raise PointSectionError(
            f"the formal report's point sections are not the frozen required "
            f"set: {len(missing)} missing {missing[:3]}, {len(extra)} "
            f"unexpected {extra[:3]}")
    for name, section in sections.items():
        if set(section) != {"payload", "provenance"}:
            raise PointSectionError(
                f"section {name!r} is not a (payload, provenance) pair")
        if section["provenance"].get("section") != name:
            raise PointSectionError(
                f"section {name!r} carries provenance for "
                f"{section['provenance'].get('section')!r}")
    return {"n_sections": len(sections),
            "n_registered": len(REGISTERED_SECTION_NAMES),
            "n_later_stage": len(LATER_STAGE_SECTION_NAMES)}


def preflight_section_schema() -> dict:
    """Schema-level preflight: the registry, its classification, its coverage."""
    import s2_reporting as rep

    if set(REGISTERED_SECTION_NAMES) != set(rep.POINT_ESTIMATE_SECTIONS):
        raise PointSectionError(
            "the section registry and s2_reporting.POINT_ESTIMATE_SECTIONS "
            "name different sets")
    if len(REGISTERED_SECTION_NAMES) != len(rep.POINT_ESTIMATE_SECTIONS):
        raise PointSectionError("a registered section name is duplicated")
    for name in REQUIRED_FORMAL_SECTION_NAMES:
        if agg.requires_ci(name):
            raise PointSectionError(
                f"{name!r} is a registered interval, not a point section")
    by_phase = {}
    for spec in SECTION_REGISTRY:
        by_phase[spec.phase] = by_phase.get(spec.phase, 0) + 1
    return {"n_registered": len(REGISTERED_SECTION_NAMES),
            "n_required": len(REQUIRED_FORMAL_SECTION_NAMES),
            "by_phase": by_phase,
            "required": list(REQUIRED_FORMAL_SECTION_NAMES),
            "later_stage": list(LATER_STAGE_SECTION_NAMES),
            "registry": section_registry_table(),
            "schema_checksum": sd.sha256_bytes(sd.canonical_json(
                [spec.as_dict() for spec in SECTION_REGISTRY]))}
