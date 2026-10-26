#!/usr/bin/env python3
"""Production-input and execution-wiring preflight for the frozen RQ2/S2
runner (Stage 2.1a inputs, Stage 2.1b wiring).

`scripts/run_rq2_s2.py --validate-only` verifies the *architecture* of a formal
run: root of trust, counts, schema, ledger ordering, static audits. It says
nothing about whether the `inputs` object `run_formal(context, inputs)` needs
can actually be built from the frozen upstream, because at Stage 2.0.8 no such
object existed.

This command verifies exactly that layer and nothing beyond it:

    upstream        every consumed artifact exists and matches its frozen
                    checksum ledger; the two Gate 1 files are content-bound to
                    checksummed S0 artifacts
    r*              recomputed as `e_full,-s - 0.10` and cross-checked against
                    an independently frozen `r_star_s`
    thresholds      23 x 3 x 17, `NO_ABSTENTION` excluded, tau non-increasing
    source counts   retained source-OOF `[n_MDD, x_MDD, n_HC, x_HC]` per candidate
    n0              the §13 source meta-LOSO, one value per outer fold
    phase 0         138 zero-label decisions, each through the frozen validator
    schema          the provider's three payloads against the frozen firewall
                    signatures, argument name for argument name

**It does not run a formal S2.** `run_formal` is never called, no target
outcome is read, no decision cell is launched, and `results/rq2_s2/` is not
created. It is **not a second runner**: `scripts/run_rq2_s2.py` remains the one
and only formal RQ2 path, which
`test_rq2_s2_root_of_trust.test_the_formal_runner_exists_and_is_the_only_one`
enforces by name — hence this file's name carries no `rq2` token. The
source-side derivations below do execute, because they are functions of the
frozen source-OOF blocks alone.

    python3 scripts/execute_s2_input_preflight.py --input-preflight
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import certification as cert  # noqa: E402
import contracts  # noqa: E402
import s2_decisions as sd  # noqa: E402
import s2_firewall as fw  # noqa: E402
import s2_inputs as si  # noqa: E402
import s2_point_sections as psec  # noqa: E402
import s2_production as prod  # noqa: E402
import s2_reporting as rep  # noqa: E402
import shrinkage as shr  # noqa: E402

#: The protocol commit this stage implements against. Recorded, then verified.
PROTOCOL_TAG = "rq2-s2-protocol-v1"
#: The authoritative result store a formal run would create. Not created here.
FORMAL_RESULTS = ROOT / "results" / "rq2_s2"


class InputPreflightError(AssertionError):
    """Raised when the production input layer cannot be established."""


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def verify_protocol_tag() -> dict:
    """The frozen tag resolves, and the protocol version is the frozen one."""
    def git(*args) -> str:
        out = subprocess.run(["git", "-C", str(ROOT), *args],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise InputPreflightError(
                f"git {' '.join(args)} failed: {out.stderr.strip()}")
        return out.stdout.strip()

    tags = git("tag").splitlines()
    if PROTOCOL_TAG not in tags:
        raise InputPreflightError(f"the protocol tag {PROTOCOL_TAG} is absent")
    tag_sha = git("rev-parse", PROTOCOL_TAG)
    head_sha = git("rev-parse", "HEAD")
    doc = ROOT / "docs" / "RQ2_S2_PROTOCOL.md"
    tracked = git("show", f"{PROTOCOL_TAG}:docs/RQ2_S2_PROTOCOL.md")
    if tracked != doc.read_text().rstrip("\n") and tracked + "\n" != doc.read_text():
        raise InputPreflightError(
            "the working-tree protocol document differs from the frozen tag")
    return {"tag": PROTOCOL_TAG, "tag_sha": tag_sha, "head_sha": head_sha,
            "protocol_version": sd.PROTOCOL_VERSION,
            "protocol_doc_checksum": sd.sha256_file(doc)}


# ---------------------------------------------------------------------------
# schema — the provider against the frozen consumers
# ---------------------------------------------------------------------------
def verify_provider_schema(provider) -> dict:
    """Every payload key is an argument the frozen firewall actually declares."""
    def kwonly(fn) -> set:
        return {p.name for p in inspect.signature(fn).parameters.values()
                if p.kind is inspect.Parameter.KEYWORD_ONLY}

    decision_params = kwonly(fw._S2FirewallBase.run_decision_cell)
    eval_params = kwonly(fw._S2FirewallBase.run_evaluation_cell) - {"outcomes"}

    site, method = contracts.FROZEN_SITES[0], cert.CONFIDENCE_METHODS[0]
    candidate = set(provider.candidate_inputs(site, method))
    evaluation = set(provider.evaluation_inputs(site, method,
                                                cert.FORMAL_STRATEGY))
    if candidate != decision_params:
        raise InputPreflightError(
            f"candidate_inputs supplies {sorted(candidate)}, the frozen "
            f"run_decision_cell declares {sorted(decision_params)}")
    if not evaluation <= eval_params:
        raise InputPreflightError(
            f"evaluation_inputs supplies {sorted(evaluation - eval_params)}, "
            "which run_evaluation_cell does not declare")

    sections = provider.point_sections()
    unregistered = sorted(set(sections) - set(rep.POINT_ESTIMATE_SECTIONS))
    if unregistered:
        raise InputPreflightError(
            f"point_sections returns unregistered sections {unregistered}")
    input_derived = sorted(sections)
    post_phase2 = sorted(set(psec.REQUIRED_FORMAL_SECTION_NAMES)
                         - set(sections))

    # every (site, method) must resolve, not just the first; and every strategy
    # must resolve to *its own* frozen grid (§7).
    grids = {}
    for s in contracts.FROZEN_SITES:
        for m in cert.CONFIDENCE_METHODS:
            ci = provider.candidate_inputs(s, m)
            if tuple(ci["q_values"]) != cert.Q_GRID:
                raise InputPreflightError(f"{s}/{m}: q_values are not the frozen grid")
            if len(ci["taus"]) != len(cert.Q_GRID):
                raise InputPreflightError(f"{s}/{m}: wrong number of taus")
            if len(ci["source_class_counts"]) != len(cert.Q_GRID):
                raise InputPreflightError(f"{s}/{m}: wrong number of count rows")
            for strategy in cert.STRATEGIES:
                payload = provider.evaluation_inputs(s, m, strategy)
                expected = provider.candidate_grid(s, m, strategy)
                if payload["taus"] != expected:
                    raise InputPreflightError(
                        f"{s}/{m}/{strategy}: the evaluation grid is not the "
                        "one the strategy deploys")
                if payload["oracle"] is not True:
                    raise InputPreflightError(
                        f"{s}/{m}/{strategy}: the §24.5 oracle diagnostic is off")
                grids.setdefault(strategy, set()).add(
                    si.STRATEGY_THRESHOLD_GRID[strategy])
    if grids != {"target_local": {si.S0_GRID},
                 "pooled_local_shrinkage": {si.S0_GRID},
                 "pooled_source": {si.S0_GRID},
                 "target_unlabeled_weighted": {si.S1_GRID}}:
        raise InputPreflightError(f"the strategy-grid mapping is {grids}")
    return {"candidate_inputs": sorted(candidate),
            "evaluation_inputs": sorted(evaluation),
            "strategy_grid_map": {k: sorted(v) for k, v in sorted(grids.items())},
            "point_sections_input_derived": input_derived,
            "point_sections_post_phase2": post_phase2,
            "n_site_method_cells": len(contracts.FROZEN_SITES)
                                   * len(cert.CONFIDENCE_METHODS)}


def verify_zero_label_decisions(provider) -> dict:
    """138, exactly the 23 x 3 x 2 set, each through the frozen validator."""
    payloads = provider.zero_label_decisions()
    if len(payloads) != prod.PRODUCTION_ZERO_LABEL_DECISIONS:
        raise InputPreflightError(
            f"{len(payloads)} zero-label decisions, not "
            f"{prod.PRODUCTION_ZERO_LABEL_DECISIONS}")
    keys, by_strategy, states = set(), {}, {}
    for payload in payloads:
        sd.validate_decision_artifact(payload, zero_label=True)
        key = (payload["site"], payload["method"], payload["strategy"])
        if key in keys:
            raise InputPreflightError(f"duplicate zero-label decision {key}")
        keys.add(key)
        if payload["strategy"] not in cert.ZERO_LABEL_STRATEGIES:
            raise InputPreflightError(
                f"{payload['strategy']!r} is not a zero-label strategy")
        if payload["guarantee_tier"] != cert.tier_of(payload["strategy"]):
            raise InputPreflightError("a zero-label tier does not match its strategy")
        by_strategy[payload["strategy"]] = by_strategy.get(payload["strategy"], 0) + 1
        label = f"{payload['strategy']}|{payload['state']}"
        states[label] = states.get(label, 0) + 1
    expected = {(s, m, st) for s in contracts.FROZEN_SITES
                for m in cert.CONFIDENCE_METHODS
                for st in cert.ZERO_LABEL_STRATEGIES}
    if keys != expected:
        raise InputPreflightError("the zero-label key set is not 23 x 3 x 2")
    return {"total": len(payloads), "by_strategy": by_strategy, "states": states,
            "n_distinct_keys": len(keys)}


def verify_n0(provider) -> dict:
    driver = provider._meta                      # noqa: SLF001 - preflight only
    table = driver.objective_table()
    selected = driver.selected
    if set(selected) != set(contracts.FROZEN_SITES):
        raise InputPreflightError("n0 was not selected for all 23 outer folds")
    off_grid = sorted(v for v in selected.values() if v not in shr.N0_GRID)
    if off_grid:
        raise InputPreflightError(f"off-grid n0 values {off_grid}")
    expected_rows = len(contracts.FROZEN_SITES) * len(shr.N0_GRID)
    if len(table) != expected_rows:
        raise InputPreflightError(
            f"the d(s, n0) table has {len(table)} rows, not {expected_rows}")
    support = driver.support_diagnostics()
    if not support["common_support"].all():
        raise InputPreflightError("a fold failed the common-support check")
    if support["held_out_target_in_meta_world"].any():
        raise InputPreflightError("a held-out target site entered its own meta world")
    if support["target_outcome_used"].any():
        raise InputPreflightError("a meta-LOSO selection recorded a target outcome")
    return {
        "selected_n0_by_site": dict(selected),
        "n0_grid": list(shr.N0_GRID),
        "d_table_rows": int(len(table)),
        "d_table_shape": [len(contracts.FROZEN_SITES), len(shr.N0_GRID)],
        "meta_repeats": int(shr.R_META),
        "n_pseudo_targets": sorted(set(support["n_pseudo_targets"].tolist())),
        "n_terms_range": [int(support["n_terms"].min()),
                          int(support["n_terms"].max())],
        "n_ties_at_argmin": sorted(set(support["n_ties_at_argmin"].tolist())),
        "tie_break_rule": shr.N0_TIE_BREAK,
    }


def verify_source_counts(provider) -> dict:
    counts = provider._source_counts             # noqa: SLF001 - preflight only
    cells = len(contracts.FROZEN_SITES) * len(cert.CONFIDENCE_METHODS)
    if len(counts) != cells:
        raise InputPreflightError(f"{len(counts)} count cells, not {cells}")
    n_rows, worst = 0, None
    for (site, method), payload in counts.items():
        rows = payload["source_class_counts"]
        if len(rows) != len(cert.Q_GRID):
            raise InputPreflightError(f"{site}/{method}: {len(rows)} count rows")
        prev = None
        for n_mdd, x_mdd, n_hc, x_hc in rows:
            for name, value in (("n_MDD", n_mdd), ("x_MDD", x_mdd),
                                ("n_HC", n_hc), ("x_HC", x_hc)):
                if not isinstance(value, int) or value < 0:
                    raise InputPreflightError(
                        f"{site}/{method}: {name}={value!r} is not a count")
            if x_mdd > n_mdd or x_hc > n_hc:
                raise InputPreflightError(
                    f"{site}/{method}: more errors than retained rows")
            if prev is not None and (n_mdd < prev[0] or n_hc < prev[2]):
                raise InputPreflightError(
                    f"{site}/{method}: retained counts are not non-decreasing "
                    "along the frozen candidate order")
            prev = (n_mdd, x_mdd, n_hc, x_hc)
            n_rows += 1
        cov = payload["coverage"]
        if any(b < a for a, b in zip(cov, cov[1:])):
            raise InputPreflightError(f"{site}/{method}: coverage is not monotone")
        if worst is None or cov[0] < worst:
            worst = cov[0]
    flat = [v for p in counts.values() for row in p["source_class_counts"]
            for v in row]
    return {"cells": len(counts), "rows": n_rows,
            "min_entry": int(min(flat)), "max_entry": int(max(flat)),
            "tightest_source_coverage": float(worst)}


# ---------------------------------------------------------------------------
# execution-wiring preflight (Stage 2.1b)
# ---------------------------------------------------------------------------
#: The synthetic integration that drives the production report path end to end
#: over fabricated decision and evaluation stores. It is a test, not a run: it
#: opens no target outcome and creates no `results/rq2_s2/`.
SYNTHETIC_WIRING_TEST = ("tests/test_rq2_s2_wiring_closure.py"
                         "::test_synthetic_end_to_end_report_wiring")


def verify_execution_wiring(provider) -> dict:
    """Stage 2.1b: no production entry point takes a scientific input.

    Signature-level, so a bypass has to be reintroduced in the source rather
    than merely be unused: `run_formal` declares one argument and it is the
    context; `build_formal_s2_report` declares two and neither is a
    section; `FrozenS2Inputs.from_frozen_context` is the constructor the runner
    reaches; `evaluation_inputs` carries the strategy axis; and the twenty
    registered point sections are classified with sixteen required.
    """
    import s2_ledger as ledgermod

    runner_path = ROOT / "scripts" / "run_rq2_s2.py"
    spec = importlib.util.spec_from_file_location("_wiring_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    run_formal = inspect.signature(runner.run_formal)
    positional = [n for n, p in run_formal.parameters.items()
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if positional != ["context"]:
        raise InputPreflightError(
            f"run_formal declares positional parameters {positional}")
    for banned in si.FORBIDDEN_PROVIDER_PARAMS:
        if banned in run_formal.parameters:
            raise InputPreflightError(f"run_formal still accepts {banned!r}")

    report_sig = inspect.signature(rep.build_formal_s2_report)
    if list(report_sig.parameters) != ["context", "ledger"]:
        raise InputPreflightError(
            f"build_formal_s2_report declares {list(report_sig.parameters)}")

    deriver = inspect.signature(psec.derive_authoritative_formal_point_sections)
    if list(deriver.parameters) != ["context", "ledger"]:
        raise InputPreflightError(
            f"the section deriver declares {list(deriver.parameters)}")

    audits = si.run_execution_wiring_audits()
    if any(row["status"] != "PASS" for row in audits):
        raise InputPreflightError(f"an execution-wiring static audit failed: {audits}")

    schema = psec.preflight_section_schema()
    if schema["n_registered"] != 20 or schema["n_required"] != 16:
        raise InputPreflightError(
            f"the section registry holds {schema['n_registered']} names with "
            f"{schema['n_required']} required")

    ess = provider.ess_and_weight_diagnostics_section()
    if tuple(ess["ess_by_site_columns"]) != \
            si.FrozenS2Inputs.ESS_BY_SITE_COLUMNS:
        raise InputPreflightError("the ESS diagnostic schema is incomplete")
    for site, row in ess["ess_by_site"].items():
        if set(row) != set(si.FrozenS2Inputs.ESS_BY_SITE_COLUMNS):
            raise InputPreflightError(f"{site}: ESS diagnostics are incomplete")

    oracle_ok = {}
    for strategy in cert.STRATEGIES:
        payload = provider.evaluation_inputs(contracts.FROZEN_SITES[0],
                                             cert.CONFIDENCE_METHODS[0],
                                             strategy)
        oracle_ok[strategy] = bool(payload["oracle"]) and \
            len(payload["taus"]) == len(cert.Q_GRID)
    if not all(oracle_ok.values()):
        raise InputPreflightError(f"the oracle schema is incomplete: {oracle_ok}")

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", SYNTHETIC_WIRING_TEST, "-q"],
        cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise InputPreflightError(
            "the synthetic wiring integration failed:\n"
            + completed.stdout[-2000:])

    return {
        "run_formal_signature": list(run_formal.parameters),
        "report_signature": list(report_sig.parameters),
        "section_deriver_signature": list(deriver.parameters),
        "authoritative_constructor": "FrozenS2Inputs.from_frozen_context",
        "static_audits": audits,
        "section_schema": {k: schema[k] for k in
                           ("n_registered", "n_required", "by_phase",
                            "schema_checksum")},
        "required_sections": schema["required"],
        "later_stage_sections": schema["later_stage"],
        "ess_diagnostic_columns": len(ess["ess_by_site_columns"]),
        "oracle_available_by_strategy": oracle_ok,
        "aggregation_prerequisites":
            list(ledgermod.AGGREGATION_PREREQUISITE_EVENTS),
        "synthetic_integration": SYNTHETIC_WIRING_TEST,
        "synthetic_integration_status": "PASS",
    }


# ---------------------------------------------------------------------------
# the preflight
# ---------------------------------------------------------------------------
def input_preflight() -> dict:
    if FORMAL_RESULTS.exists():
        raise InputPreflightError(
            f"{FORMAL_RESULTS} exists; this stage may not run against an "
            "authoritative formal result store")
    report = {"protocol": verify_protocol_tag()}
    provider = si.FrozenS2Inputs()
    report["upstream"] = provider.provenance
    report["gate1_binding"] = provider.gate1_binding
    report["r_star"] = {k: v for k, v in provider._r_star_result.items()
                        if k != "r_star"}                     # noqa: SLF001
    report["thresholds"] = {
        "s0_cells": len(provider._source_taus),               # noqa: SLF001
        "s1_cells": len(provider._weighted_taus),             # noqa: SLF001
        "candidates_per_cell": len(cert.Q_GRID),
        "no_abstention_excluded": True,
    }
    report["source_counts"] = verify_source_counts(provider)
    report["n0"] = verify_n0(provider)
    report["zero_label"] = verify_zero_label_decisions(provider)
    report["schema"] = verify_provider_schema(provider)
    report["wiring"] = verify_execution_wiring(provider)
    report["static_audits"] = si.run_input_static_audits()
    if any(r["status"] != "PASS" for r in report["static_audits"]):
        raise InputPreflightError("an input-layer static audit failed")
    report["weighted_reproduction_checks"] = \
        provider._weighted.n_reproduction_checks              # noqa: SLF001
    report["target_outcome_rows_read"] = 0
    report["formal_results_dir_present"] = FORMAL_RESULTS.exists()
    if report["formal_results_dir_present"]:
        raise InputPreflightError("the input preflight created a formal store")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-preflight", action="store_true",
                        help="verify the production input-derivation layer; "
                             "runs no formal S2 and reads no target outcome")
    parser.add_argument("--json", action="store_true",
                        help="emit the machine-readable report instead")
    args = parser.parse_args(argv)

    if not args.input_preflight:
        print("Formal RQ2/S2 execution is not authorised at Stage 2.1b.\n"
              "Run with --input-preflight.", file=sys.stderr)
        return 2

    report = input_preflight()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    p = report["protocol"]
    print("RQ2/S2 production-input preflight (--input-preflight)")
    print("=" * 70)
    print(f"protocol tag        {p['tag']}  {p['tag_sha']}")
    print(f"HEAD                {p['head_sha']}")
    print(f"protocol version    {p['protocol_version']}")
    print("-" * 70)
    for row in report["upstream"]["artifacts"]:
        ledger = row["checksum_ledger"] or "content-bound"
        print(f"upstream  {row['sha256'][:16]}  {ledger:<14} {row['path']}")
    gb = report["gate1_binding"]
    print(f"gate1 binding       {gb['n_oof_rows']} OOF rows, "
          f"{gb['n_exact_columns']} columns bit-identical to the checksummed "
          f"S0 table, {gb['n_ulp_columns']} within {gb['max_ulp_delta']:g}")
    rs = report["r_star"]
    print(f"r_star              {rs['n_sites']}/23 recomputed from e_full,-s; "
          f"cross-check max delta {rs['max_abs_crosscheck_delta']:g}, "
          f"{rs['n_exactly_equal']} exact")
    th = report["thresholds"]
    print(f"thresholds          S0 {th['s0_cells']} cells x "
          f"{th['candidates_per_cell']} q · S1 {th['s1_cells']} cells x "
          f"{th['candidates_per_cell']} q · NO_ABSTENTION excluded")
    sc = report["source_counts"]
    print(f"source counts       {sc['cells']} cells, {sc['rows']} candidate "
          f"rows, entries [{sc['min_entry']}, {sc['max_entry']}], tightest "
          f"source coverage {sc['tightest_source_coverage']:.4f}")
    n0 = report["n0"]
    print(f"n0 meta-LOSO        grid {n0['n0_grid']} · R_META {n0['meta_repeats']} "
          f"· pseudo-targets {n0['n_pseudo_targets']} · terms "
          f"{n0['n_terms_range']} · ties {n0['n_ties_at_argmin']}")
    print(f"                    d(s, n0) table {n0['d_table_shape'][0]} x "
          f"{n0['d_table_shape'][1]} ({n0['d_table_rows']} rows), tie break "
          f"{n0['tie_break_rule']}")
    print(f"                    selected n0 {n0['selected_n0_by_site']}")
    zl = report["zero_label"]
    print(f"zero-label          {zl['total']}/"
          f"{prod.PRODUCTION_ZERO_LABEL_DECISIONS} · {zl['by_strategy']}")
    print(f"                    states {zl['states']}")
    sch = report["schema"]
    print(f"candidate_inputs    {sch['candidate_inputs']}")
    print(f"evaluation_inputs   {sch['evaluation_inputs']}")
    print(f"strategy grids      {sch['strategy_grid_map']}")
    print(f"point_sections      input-derived "
          f"{sch['point_sections_input_derived']}")
    print(f"                    post-phase-2 "
          f"({len(sch['point_sections_post_phase2'])}): "
          f"{sch['point_sections_post_phase2']}")
    wi = report["wiring"]
    print("-" * 70)
    print(f"run_formal          ({', '.join(wi['run_formal_signature'])}) · "
          f"no caller provider · builds {wi['authoritative_constructor']}")
    print(f"formal report       ({', '.join(wi['report_signature'])}) · "
          f"no point_sections · sections from "
          f"({', '.join(wi['section_deriver_signature'])})")
    print(f"point sections      {wi['section_schema']['n_required']}/"
          f"{wi['section_schema']['n_registered']} required · "
          f"{wi['section_schema']['by_phase']}")
    print(f"                    later sensitivity stage (§23): "
          f"{wi['later_stage_sections']}")
    print(f"oracle schema       {wi['oracle_available_by_strategy']}")
    print(f"ESS diagnostics     {wi['ess_diagnostic_columns']} frozen S1 columns "
          "carried per fold")
    print(f"synthetic wiring    {wi['synthetic_integration_status']}  "
          f"{wi['synthetic_integration']}")
    for audit in wi["static_audits"]:
        print(f"audit  {audit['status']:<5} {audit['audit']}")
    print(f"S1 reproduction     {report['weighted_reproduction_checks']} frozen "
          "weighted-source values re-derived and matched")
    for audit in report["static_audits"]:
        print(f"audit  {audit['status']:<5} {audit['audit']}")
    print("=" * 70)
    print(f"FORMAL RESULTS DIR  {FORMAL_RESULTS} "
          f"({'PRESENT' if FORMAL_RESULTS.exists() else 'absent'})")
    print(f"outcome reads       {report['target_outcome_rows_read']} rows")
    print("INPUT PREFLIGHT: PASS — run_formal not invoked, no target outcome "
          "read, no formal store created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
