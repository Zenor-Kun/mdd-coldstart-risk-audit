#!/usr/bin/env python3
"""The single formal RQ2/S2 runner (protocol 2.0.8, wiring Stage 2.1b).

There is no alternate runner. Every formal S2 quantity — decision, evaluation,
aggregate and interval — is produced by the call chain below, so the chain can
be audited statically and the CI registry cannot be walked around by a script
someone wrote beside it.

    CONTEXT   build ProductionS2Context      (root of trust, recomputed)
              GlobalRunLedger.for_context()  (the run's ONE seq allocator)
    PHASE 0   verify frozen upstream checksums
              derive + freeze the production split manifest
              assert 46,200 expected decision cells and their decomposition
              zero-label reference decisions (no target label opened)
    PHASE 1   cell-isolated decision workers, target_local + shrinkage only
                                             -> PHASE1_STARTED
    RELEASE   complete_production_phase1()   -> ALL_DECISIONS_FROZEN
    OPEN      open_phase2()                  -> RELEASE_AUTHORITY_VERIFIED
                                             -> PHASE2_OPEN
    OUTCOMES  the parent, and only now, materializes the target outcomes
                                             -> FIRST_TARGET_OUTCOME_READ
    PHASE 2   pure evaluation workers -> frozen evaluation artifacts
              complete_production_phase2()   -> EVALUATION_COMPLETE
                                             -> EVALUATION_MANIFEST_FROZEN
    AGGREGATE authoritative site-level results, read off the artifact stores
                                             -> AGGREGATION_STARTED
    CI        compute_registered_interval() and nothing else
    SECTIONS  derive_authoritative_formal_point_sections(context, ledger)
    REPORT    build_formal_s2_report(context, ledger)   <- no store, no None,
                                                           no point sections
                                             -> REPORT_BUILT

Every one of those sequence numbers is issued by the same allocator, so the
`ALL_DECISIONS_FROZEN < RELEASE_AUTHORITY_VERIFIED < FIRST_TARGET_OUTCOME_READ`
claim is a comparison and not a coincidence.

Two modes, and the difference between them is authority, not scale.

`--validate-only` reads no target outcome, launches no decision job, produces no
science value and creates no authoritative `results/rq2_s2/`; it verifies
architecture, root of trust and schema inside a temporary directory.

`--formal` is the authoritative run. Stage 2.1d added it as **execution wiring
only**: it runs the same frozen preflight first, refuses to start unless the
authoritative result root is pristine, builds the production context *on* that
root so every store, ledger binding, manifest and artifact the existing
framework already persists lands under `results/rq2_s2/`, and then calls the
one audited chain — `run_formal(context)` — unchanged. It adds no scientific
step, no second chain, no threshold, no override and no path option.

    python3 scripts/run_rq2_s2.py --validate-only
    python3 scripts/run_rq2_s2.py --formal
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import certification as cert  # noqa: E402
import s2_aggregation as agg  # noqa: E402
import s2_decisions as sd  # noqa: E402
import s2_evaluation_store as evalstore  # noqa: E402
import s2_firewall as fw  # noqa: E402
import s2_inputs  # noqa: E402
import s2_ledger as ledgermod  # noqa: E402
import s2_point_sections as psec  # noqa: E402
import s2_outcomes as outcomes_mod  # noqa: E402
import s2_production as prod  # noqa: E402
import s2_reporting as rep  # noqa: E402
import s2_site_aggregation as sagg  # noqa: E402

#: The authoritative output location of a formal run. **Not created by
#: `--validate-only`**, which works entirely inside a temporary directory.
FORMAL_RESULTS = ROOT / "results" / "rq2_s2"


# ---------------------------------------------------------------------------
# preflight — everything this stage is allowed to do
# ---------------------------------------------------------------------------
def preflight(workdir=None) -> dict:
    """Root-of-trust, count and schema verification. Opens no target outcome."""
    report = {}

    context = prod.ProductionS2Context.build(workdir)
    snapshot = context.assert_still_authoritative()   # re-derives everything
    report["protocol_version"] = snapshot["protocol_version"]
    report["protocol_checksum"] = snapshot["protocol_doc_checksum"][:16]
    report["cohort_checksum"] = snapshot["cohort_checksum"][:16]
    report["roster_checksum"] = snapshot["roster_checksum"][:16]
    report["outcome_source"] = snapshot["outcome_source"]
    report["split_manifest_checksum"] = snapshot["split_registry_checksum"][:16]

    invariants = context.registry.assert_production_invariants()
    report["registry"] = invariants

    decomposition = prod.assert_expected_decomposition(context.expected_cells)
    report["expected_decisions"] = decomposition
    if decomposition["total"] != prod.PRODUCTION_EXPECTED_DECISIONS:
        raise prod.ProductionAuthorityError("expected-decision count moved")

    expected_eval = evalstore.derive_authoritative_expected_evaluation_cells(
        context)
    report["expected_evaluations"] = \
        evalstore.assert_expected_evaluation_decomposition(expected_eval)

    scope = sagg.authoritative_site_scope(context)
    report["site_identities"] = {
        "per_n_site_counts": scope["per_n_site_counts"],
        "n_all_sites": len(scope["all_sites"]),
        "n_expected_repeats": cert.R_SPLIT,
    }
    if scope["per_n_site_counts"] != prod.PRODUCTION_SITE_COUNTS_BY_N:
        raise prod.ProductionAuthorityError("per-N eligible-site identities moved")

    report["ci_schema"] = rep.preflight_ci_schema()
    report["ci_registry_checksum"] = snapshot["ci_registry_checksum"][:16]
    report["point_section_schema"] = psec.preflight_section_schema()
    report["wiring_audits"] = s2_inputs.run_execution_wiring_audits()
    if any(r["status"] != "PASS" for r in report["wiring_audits"]):
        raise AssertionError("an execution-wiring static audit failed")
    report["global_ledger"] = _global_ledger_preflight()
    report["evaluation_root"] = _evaluation_root_preflight(context)
    report["phase2_architecture"] = _phase2_architecture_preflight()
    report["static_audits"] = fw.run_static_audits().to_dict("records")
    if any(r["status"] != "PASS" for r in report["static_audits"]):
        raise AssertionError("a production static audit failed")

    report["target_outcome_rows_read"] = outcomes_mod.total_rows_read()
    if report["target_outcome_rows_read"] != 0:
        raise AssertionError("preflight must read no target outcome")
    report["zero_label_decisions_expected"] = prod.PRODUCTION_ZERO_LABEL_DECISIONS
    report["scientific_constants"] = {
        "m_min": cert.M_MIN, "c_min": cert.C_MIN, "alpha_ucb": cert.ALPHA_UCB,
        "alpha_class": cert.ALPHA_CLASS, "n_grid": list(cert.N_GRID),
        "r_split": cert.R_SPLIT, "n_candidates": len(cert.Q_GRID),
        "n_methods": len(cert.CONFIDENCE_METHODS),
        "n_strategies": len(cert.STRATEGIES),
    }
    return report


def _global_ledger_preflight() -> dict:
    """One allocator, the frozen event schema, and a refusal to mix two runs.

    Reads no target outcome: it exercises the ordering machinery on an empty
    ledger and on two deliberately independent ones, so `--validate-only` can
    show that a separate-counter stream cannot pass.
    """
    ledger = ledgermod.GlobalRunLedger("preflight")
    ledger.append(ledgermod.EV_PHASE1_STARTED)
    ledger.append(ledgermod.EV_ALL_DECISIONS_FROZEN)
    ledger.append(ledgermod.EV_RELEASE_AUTHORITY_VERIFIED)
    ledger.append(ledgermod.EV_PHASE2_OPEN)
    ledger.append(ledgermod.EV_OUTCOME_READ_AUTHORIZED)
    ledger.append(ledgermod.EV_FIRST_TARGET_OUTCOME_READ, rows_read=0)
    ordering = ledgermod.assert_global_ordering(ledger)

    other = ledgermod.GlobalRunLedger("preflight-second-allocator")
    other.append(ledgermod.EV_FIRST_TARGET_OUTCOME_READ, rows_read=0)
    merged = list(ledger.events) + list(other.events)
    try:
        ledgermod.assert_event_stream_ordering(merged, ledger.ledger_id,
                                               ledger.run_id)
    except ledgermod.LedgerError:
        mixed_rejected = True
    else:
        mixed_rejected = False
    if not mixed_rejected:
        raise AssertionError(
            "two independent allocators produced an accepted ordering proof")
    return {
        "critical_events": list(ledgermod.GLOBAL_EVENT_ORDER),
        "required_barrier_events": list(ledgermod.REQUIRED_BARRIER_EVENTS),
        "barrier_seqs": ordering["barrier_seqs"],
        "separate_counter_stream_rejected": mixed_rejected,
        "single_allocator_audit":
            fw.audit_single_global_sequence_allocator()["ok"],
    }


def _evaluation_root_preflight(context) -> dict:
    """The evaluation root is a derivation, and the ledger is not optional.

    Reads no artifact: it resolves the authoritative root from the context,
    re-derives its identity, and checks the two production signatures — the
    aggregation and the report — on the real objects, so `--validate-only` can
    show that neither a caller-supplied store nor a missing ledger has an entry
    point left.
    """
    import inspect

    store = evalstore.resolve_authoritative_evaluation_store(context)
    root = context.assert_evaluation_root_authoritative()
    if Path(store.root) != Path(root) or Path(root).parent != Path(context.workdir):
        raise AssertionError("the evaluation root is not the context's own")

    aggregate = list(inspect.signature(
        sagg.aggregate_authoritative_site_level_results).parameters)
    report_params = inspect.signature(rep.build_formal_s2_report).parameters
    if aggregate != ["context", "ledger"]:
        raise AssertionError(f"aggregation signature is {aggregate}")
    if list(report_params)[:2] != ["context", "ledger"]:
        raise AssertionError(f"report signature is {list(report_params)}")
    for name in ("context", "ledger"):
        if report_params[name].default is not inspect.Parameter.empty:
            raise AssertionError(f"the report's {name} has a default")

    # Repeated resolution must return the binding's one id. Conversely, an
    # independently constructed same-run allocator remains unauthoritative even
    # when its provenance and complete event stream are otherwise correct.
    official = ledgermod.GlobalRunLedger.for_context(context)
    repeated = ledgermod.GlobalRunLedger.for_context(context)
    if official.ledger_id != repeated.ledger_id:
        raise AssertionError("repeated for_context minted a second ledger id")
    binding = ledgermod.read_authoritative_ledger_binding(context)
    snapshot = context.snapshot
    forged = ledgermod.GlobalRunLedger(
        context.run_id,
        context_snapshot_checksum=sd.sha256_bytes(sd.canonical_json(snapshot)),
        split_registry_checksum=snapshot["split_registry_checksum"],
        ledger_binding_checksum=binding["binding_checksum"],
    )
    for event in (ledgermod.EV_PHASE1_STARTED,
                  ledgermod.EV_ALL_DECISIONS_FROZEN,
                  ledgermod.EV_RELEASE_AUTHORITY_VERIFIED,
                  ledgermod.EV_PHASE2_OPEN,
                  ledgermod.EV_OUTCOME_READ_AUTHORIZED,
                  ledgermod.EV_FIRST_TARGET_OUTCOME_READ,
                  ledgermod.EV_EVALUATION_COMPLETE,
                  ledgermod.EV_EVALUATION_MANIFEST_FROZEN):
        forged.append(event)
    try:
        ledgermod.assert_run_ledger_for_context(forged, context)
    except ledgermod.LedgerError as exc:
        if "authoritative ledger-id mismatch" not in str(exc):
            raise AssertionError(
                "the complete second allocator failed for the wrong reason") from exc
        second_rejected = True
    else:
        raise AssertionError("a complete same-run second allocator was accepted")

    # An empty same-id resolver cannot support completed-run ordering; None and
    # non-ledgers remain unavailable modes.
    for bad in (None, repeated, object()):
        try:
            ledgermod.assert_run_ledger_for_context(bad, context)
        except (ledgermod.LedgerError, prod.ProductionAuthorityError):
            continue
        raise AssertionError(f"{bad!r} was accepted as this run's ledger")

    return {
        "evaluation_root": str(root),
        "evaluation_root_identity":
            context.snapshot["evaluation_root_identity"][:16],
        "ledger_binding": str(
            ledgermod.authoritative_ledger_binding_path(context)),
        "authoritative_ledger_id": official.ledger_id,
        "repeated_for_context_same_id": True,
        "complete_second_ledger_rejected": second_rejected,
        "root_is_derived_not_supplied": True,
        "aggregation_signature": aggregate,
        "report_signature": list(report_params),
        "none_empty_and_foreign_ledgers_refused": True,
        "no_caller_supplied_store_audit":
            fw.audit_no_caller_supplied_evaluation_store()["ok"],
        "no_optional_ledger_audit":
            fw.audit_no_optional_production_ledger()["ok"],
    }


def _phase2_architecture_preflight() -> dict:
    """Phase-2 capability gating exists and has no command-line entry point."""
    import s2_cell_worker
    import s2_eval_worker
    checks = {
        "no_standalone_evaluation_cli": not hasattr(s2_eval_worker, "main"),
        "eval_worker_has_no_outcome_access":
            fw.audit_eval_worker_has_no_outcome_access()["ok"],
        "sole_parent_outcome_read_path":
            fw.audit_single_outcome_read_path()["ok"],
        "report_accepts_no_external_ci":
            fw.audit_report_accepts_no_external_ci()["ok"],
        "report_takes_no_site_level_input":
            fw.audit_report_takes_no_site_level_input()["ok"],
        "evaluation_store_is_derived":
            fw.audit_evaluation_store_is_derived()["ok"],
        "no_caller_supplied_evaluation_store":
            fw.audit_no_caller_supplied_evaluation_store()["ok"],
        "no_optional_production_ledger":
            fw.audit_no_optional_production_ledger()["ok"],
        "no_target_outcome_read_in_preflight":
            outcomes_mod.total_rows_read() == 0,
        "evaluation_requires_capability":
            "capability" in fw.ProductionS2Firewall.run_evaluation_cell.__code__.co_varnames,
        "decision_cli_rejects_evaluation_mode":
            "no command-line entry point" in (s2_cell_worker.main.__doc__ or "")
            or "no command-line entry point" in Path(
                ROOT / "src" / "s2_cell_worker.py").read_text(),
        "open_phase2_is_the_only_gate": hasattr(fw.ProductionS2Firewall,
                                                "open_phase2"),
        "phase_states": [fw.PHASE_INIT, fw.PHASE_1, fw.PHASE_1_COMPLETE,
                         fw.PHASE_2_OPEN],
    }
    for name, ok in checks.items():
        if name != "phase_states" and not ok:
            raise AssertionError(f"phase-2 architecture check failed: {name}")
    return checks


# ---------------------------------------------------------------------------
# the formal phases (present, auditable, not executed by this stage)
# ---------------------------------------------------------------------------
def run_formal(context) -> dict:
    """The formal execution path. Never reached under `--validate-only`.

    **One parameter, and it is authority, not data.** Stage 2.0.8 froze
    `run_formal(context, inputs)`, so any object with four methods was a
    production input provider: a hand-built one returning a fabricated
    `tau`, `r*`, `n0` and zero-label decision entered the formal run through the
    front door. Stage 2.1b removes the parameter. The provider is constructed
    here, from the frozen artifacts alone:

        inputs = s2_inputs.FrozenS2Inputs.from_frozen_context(context)

    There is no `inputs`, `provider`, `candidate_provider`, `evaluation_provider`,
    `point_sections`, `fixture` or `input_factory` argument, no keyword-only
    argument and no catch-all through which one could arrive.

    One context, one ledger, one firewall. Phase 0's zero-label decisions and
    Phase 1's cell decisions are frozen through the authoritative validator; the
    release is issued only by `complete_production_phase1()`; Phase 2 opens
    after `open_phase2()` re-derives authority; each evaluation freezes an
    artifact including its strategy-correct oracle diagnostic; and the report is
    built from those artifacts, not from anything this function is holding.
    """
    # The production input provider, constructed here and nowhere else. It
    # resolves the frozen S0/S1/Gate-1 artifacts itself, verifies their
    # checksums and provenance, and opens no target outcome.
    inputs = s2_inputs.FrozenS2Inputs.from_frozen_context(context)

    # The run's ONE allocator, created here and nowhere else. Every component
    # below receives this object; none of them may build its own (§17.25).
    ledger = ledgermod.GlobalRunLedger.for_context(context)
    firewall = fw.ProductionS2Firewall(context, ledger)

    for payload in inputs.zero_label_decisions():                 # PHASE 0
        firewall.record_zero_label_decision(payload)

    for cell in sorted(context.expected_cells):                   # PHASE 1
        site, repeat, n_labels, strategy, method = cell
        firewall.run_decision_cell(site, repeat, n_labels, strategy, method,
                                   **inputs.candidate_inputs(site, method))

    firewall.complete_production_phase1()                         # RELEASE
    capability = firewall.open_phase2()          # full authority re-derivation

    # the FIRST target outcome read of the whole run, and the only one
    target_outcomes = outcomes_mod.load_authoritative_target_outcomes_for_phase2(
        context, firewall, ledger)

    expected_eval = evalstore.derive_authoritative_expected_evaluation_cells(
        context)
    for cell in sorted(expected_eval):                            # PHASE 2
        site, repeat, n_labels, strategy, method = cell
        # strategy-aware: the §17 oracle diagnostic must sweep the frozen
        # candidate grid *that* strategy deploys — tau_q^S0 for three of them,
        # tau_q^S1 for the weighted arm (§7).
        firewall.run_evaluation_cell(
            capability, site, repeat, n_labels, strategy, method,
            outcomes=target_outcomes,
            **inputs.evaluation_inputs(site, method, strategy))

    manifest = firewall.complete_production_phase2()              # EVAL MANIFEST
    # AGGREGATE + CI + SECTIONS + REPORT are all inside: the report takes no
    # data argument, no evaluation store, no point sections and no optional
    # ledger — the same `ledger` object that opened phase 1 is the one that must
    # close the report.
    report = rep.build_formal_s2_report(context, ledger)
    ledgermod.assert_full_run_ordering(ledger)
    return {"firewall": firewall, "report": report,
            "evaluation_manifest": manifest,
            "global_ledger": ledger,
            "outcome_ledger": outcomes_mod.phase_ordering_ledger(firewall)}


# ---------------------------------------------------------------------------
# Stage 2.1d — formal execution wiring
#
# Everything below is mechanism: where the authoritative run puts its files,
# what it refuses to start on top of, and what it writes down so the result can
# be audited without re-running it. No scientific quantity is computed here;
# `run_formal` above is reached unchanged and remains the only chain.
# ---------------------------------------------------------------------------
#: Files the wiring persists on top of what the production framework already
#: writes under the run's workdir (split registry, ledger binding, decision
#: store, zero-label references, evaluation store, global release).
PROVENANCE_FILENAME = "formal_run_provenance.json"
COMPLETION_FILENAME = "formal_run_completion.json"
INTERVALS_FILENAME = "registered_intervals.csv"
SECTIONS_FILENAME = "formal_point_sections.json"
LEDGER_FILENAME = "global_run_ledger.csv"
EVAL_MANIFEST_FILENAME = "evaluation_manifest.json"
CHECKSUM_FILENAME = "RESULT_CHECKSUMS.sha256"

#: What the checksum manifest covers, digested from the bytes on disk. The
#: completion marker is written *after* the manifest and is therefore not a
#: member — see `_write_checksum_manifest`.
CHECKSUM_MANIFEST_MEMBERS = (PROVENANCE_FILENAME, INTERVALS_FILENAME,
                             SECTIONS_FILENAME, LEDGER_FILENAME,
                             EVAL_MANIFEST_FILENAME)

#: The immutable execution freeze this wiring amends. Recorded, never trusted:
#: the run records what HEAD actually is alongside it.
EXECUTION_BASELINE_TAG = "rq2-s2-execution-v1"


class FormalRunRefused(RuntimeError):
    """Raised when the authoritative run may not start or may not be trusted."""


def _assert_formal_root_pristine() -> dict:
    """Fail closed on any pre-existing authoritative state.

    A formal run is not resumable and never overwrites: the release, the
    manifests and the ledger binding are all statements about *one* execution,
    so a directory that already holds part of one is a refusal, not a starting
    point. Recovery is an explicit human decision to move the old tree aside.
    """
    if not FORMAL_RESULTS.exists():
        return {"root": str(FORMAL_RESULTS), "pre_existing_entries": 0}
    entries = sorted(p.name for p in FORMAL_RESULTS.iterdir())
    if entries:
        raise FormalRunRefused(
            f"{FORMAL_RESULTS} already holds {len(entries)} entries "
            f"(e.g. {entries[:5]}); a formal run neither resumes nor overwrites "
            f"authoritative results. Move the existing tree aside first.")
    return {"root": str(FORMAL_RESULTS), "pre_existing_entries": 0}


def _assert_root_containment(context) -> dict:
    """The run's whole tree must lie under the authoritative result root."""
    workdir = Path(context.workdir).resolve()
    expected = FORMAL_RESULTS.resolve()
    if workdir != expected:
        raise FormalRunRefused(
            f"the production context was built on {workdir}, not the "
            f"authoritative {expected}")
    evaluation_root = Path(context.assert_evaluation_root_authoritative())
    if evaluation_root.parent.resolve() != expected:
        raise FormalRunRefused(
            f"the evaluation root {evaluation_root} is outside {expected}")
    registry_path = Path(context.registry.path).resolve()
    if registry_path.parent != expected:
        raise FormalRunRefused(
            f"the split registry {registry_path} is outside {expected}")
    binding = Path(ledgermod.authoritative_ledger_binding_path(context)).resolve()
    if binding.parent != expected:
        raise FormalRunRefused(
            f"the ledger binding {binding} is outside {expected}")
    return {"workdir": str(workdir), "evaluation_root": str(evaluation_root),
            "split_registry": str(registry_path), "ledger_binding": str(binding)}


def _environment_provenance() -> dict:
    """Python and library versions, so the run can be placed in time."""
    import platform
    from importlib import metadata

    versions = {}
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "statsmodels",
                    "joblib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:      # pragma: no cover - env drift
            versions[package] = "ABSENT"
    return {"python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": versions}


def _repository_provenance() -> dict:
    """Tag and commit provenance, through the frozen preflight's own checker.

    `verify_protocol_tag()` is the existing mechanism: it fails closed when the
    protocol tag is missing or when the working-tree protocol document has
    drifted from it. Nothing new is invented here; the execution baseline tag is
    resolved beside it so the amendment's own drift is visible in the record.
    """
    import importlib.util
    import subprocess

    spec = importlib.util.spec_from_file_location(
        "s2_input_preflight", ROOT / "scripts" / "execute_s2_input_preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provenance = dict(module.verify_protocol_tag())

    def git(*argv):
        done = subprocess.run(["git", "-C", str(ROOT), *argv],
                              capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else None

    provenance.update({
        "protocol_tag": provenance.pop("tag"),
        "protocol_tag_sha": provenance.pop("tag_sha"),
        "execution_baseline_tag": EXECUTION_BASELINE_TAG,
        "execution_baseline_sha": git("rev-parse", EXECUTION_BASELINE_TAG),
        "head_sha": provenance.get("head_sha"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
    })
    return provenance


def _pre_run_provenance(context) -> dict:
    """Everything an auditor needs about the run's authority, before Phase 0.

    Written before the first decision artifact exists. Its source of truth is
    the production context, the frozen upstream verifier and the frozen expected
    decompositions — this function derives no scientific quantity of its own.
    """
    expected_eval = evalstore.derive_authoritative_expected_evaluation_cells(
        context)
    scope = sagg.authoritative_site_scope(context)
    return {
        "stage": "formal_s2",
        "run_id": context.run_id,
        "protocol_version": context.protocol_version,
        "repository": _repository_provenance(),
        "environment": _environment_provenance(),
        "authority_snapshot": context.snapshot,
        "upstream": s2_inputs.verify_frozen_upstream(),
        "expected_decisions": prod.assert_expected_decomposition(
            context.expected_cells),
        "expected_zero_label_decisions": prod.PRODUCTION_ZERO_LABEL_DECISIONS,
        "expected_evaluations":
            evalstore.assert_expected_evaluation_decomposition(expected_eval),
        "expected_intervals": agg.N_REQUIRED_INTERVALS,
        "expected_point_sections": len(psec.REQUIRED_FORMAL_SECTION_NAMES),
        "later_stage_point_sections": list(psec.LATER_STAGE_SECTION_NAMES),
        "site_identities": {
            "per_n_site_counts": scope["per_n_site_counts"],
            "all_sites": list(scope["all_sites"]),
            "n_repeats": cert.R_SPLIT,
        },
        "scientific_constants": {
            "m_min": cert.M_MIN, "c_min": cert.C_MIN,
            "alpha_ucb": cert.ALPHA_UCB, "alpha_class": cert.ALPHA_CLASS,
            "n_grid": list(cert.N_GRID), "r_split": cert.R_SPLIT,
            "q_grid": list(cert.Q_GRID),
            "confidence_methods": list(cert.CONFIDENCE_METHODS),
            "strategies": list(cert.STRATEGIES),
        },
        "authoritative_roots": _assert_root_containment(context),
        "authoritative_ledger_binding_path":
            str(ledgermod.authoritative_ledger_binding_path(context)),
        "target_outcome_rows_read_before_phase0":
            outcomes_mod.total_rows_read(),
    }


def _write_json(path: Path, payload) -> str:
    path.write_bytes(sd.canonical_json(payload))
    return sd.sha256_file(path)


def _persist_intervals(path: Path, report) -> str:
    """The 216 registered intervals, one row each, in registry order."""
    import csv

    rows = report.interval_rows()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rep.INTERVAL_ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field)
                             for field in rep.INTERVAL_ROW_FIELDS})
    return sd.sha256_file(path)


def _write_checksum_manifest(root: Path) -> dict:
    """Digest every persisted artifact **from its bytes on disk**, then verify.

    The digests must be of the files an auditor will actually check, so they are
    read back off the filesystem rather than carried down from whatever produced
    the artifact. A logical payload hash is not a file digest: the provenance
    record, for one, cannot contain its own file's checksum, and an earlier
    revision of this function put that logical hash in the manifest and made
    `sha256sum -c` fail on the one artifact written before the run started.

    The completion marker is deliberately **not** a member: it is written after
    this manifest, and it is what this manifest makes trustworthy rather than
    one of the things it covers.
    """
    digests = {}
    for name in CHECKSUM_MANIFEST_MEMBERS:
        path = root / name
        if not path.is_file():
            raise FormalRunRefused(
                f"{name} is missing; the checksum manifest may not describe a "
                f"partially persisted run")
        digests[name] = sd.sha256_file(path)

    manifest_path = root / CHECKSUM_FILENAME
    manifest_path.write_text("".join(f"{digests[name]}  {name}\n"
                                     for name in sorted(digests)))

    # A manifest nobody verified is a claim, not evidence. Re-read every file
    # and compare before the run is allowed to call itself complete.
    for name, digest in digests.items():
        if sd.sha256_file(root / name) != digest:
            raise FormalRunRefused(
                f"{name} changed while the checksum manifest was being written")
    return {"digests": digests,
            "checksum_manifest_checksum": sd.sha256_file(manifest_path)}


def _persist_formal_outputs(context, ledger, report, evaluation_manifest,
                            outcome_ledger) -> dict:
    """Make the finished run durable and self-describing.

    Source of truth throughout: the report the frozen builder returned, the
    run's one allocator, and the manifest the firewall froze. Nothing is
    recomputed and nothing is reformatted into a competing record.

    **Order is a guarantee here.** Evidence first, then the checksum manifest
    over all of it, and only then the completion marker. Nothing is wrapped in a
    recovery branch, so a failure at any step propagates and no completion
    artifact is left behind: the presence of `formal_run_completion.json` means
    every prior step succeeded, and it means nothing else.
    """
    root = Path(context.workdir)
    _persist_intervals(root / INTERVALS_FILENAME, report)
    _write_json(root / SECTIONS_FILENAME, report.point_sections)
    ledger.frame().to_csv(root / LEDGER_FILENAME, index=False)
    _write_json(root / EVAL_MANIFEST_FILENAME, evaluation_manifest)

    checksums = _write_checksum_manifest(root)

    completion = {
        "run_id": context.run_id,
        "protocol_version": context.protocol_version,
        "report_provenance": report.provenance,
        "report_completeness": report.assert_complete(),
        "barrier_seqs": ledger.barrier_seqs(),
        "ledger_ordering": ledgermod.assert_full_run_ordering(ledger),
        "outcome_ordering": outcome_ledger,
        "target_outcome_rows_read": outcomes_mod.total_rows_read(),
        "evaluation_manifest_checksum":
            evaluation_manifest["evaluation_manifest_checksum"],
        "n_evaluations": len(evaluation_manifest["evaluation_manifest"]),
        # raises rather than returning False: the authority is re-derived and
        # compared field by field inside the context itself.
        "authority_snapshot_re_derived":
            bool(context.assert_still_authoritative()),
        # the binding's own field name — this record consumes the authority
        # framework's schema, it does not define one of its own.
        "authoritative_ledger_id": ledgermod.read_authoritative_ledger_binding(
            context)["authoritative_ledger_id"],
        "artifact_checksums": dict(checksums["digests"]),
        "checksum_manifest_checksum": checksums["checksum_manifest_checksum"],
        "provenance_checksum": checksums["digests"][PROVENANCE_FILENAME],
    }
    # last, and only now: the run's single success marker
    _write_json(root / COMPLETION_FILENAME, completion)
    return {"completion": completion, "files": dict(checksums["digests"])}


def execute_formal_run() -> dict:
    """The authoritative run. One frozen chain, one authoritative root.

    The order is the point: refuse a dirty root, pass the frozen preflight,
    build the context *on* `results/rq2_s2/`, record provenance while the tree
    still holds no result, and only then enter `run_formal`.
    """
    _assert_formal_root_pristine()
    with tempfile.TemporaryDirectory(prefix="s2_preflight_") as tmp:
        preflight(Path(tmp))

    FORMAL_RESULTS.mkdir(parents=True, exist_ok=True)
    context = prod.ProductionS2Context.build(FORMAL_RESULTS)
    context.assert_still_authoritative()
    _assert_root_containment(context)

    provenance = _pre_run_provenance(context)
    _write_json(FORMAL_RESULTS / PROVENANCE_FILENAME, provenance)

    outcome = run_formal(context)

    persisted = _persist_formal_outputs(
        context, outcome["global_ledger"], outcome["report"],
        outcome["evaluation_manifest"], outcome["outcome_ledger"])
    return {"context": context, "provenance": provenance, **outcome,
            "persisted": persisted}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="root-of-trust and schema preflight; reads no "
                           "target outcome and produces no science value")
    mode.add_argument("--formal", action="store_true",
                      help="the authoritative formal S2 run; writes "
                           "results/rq2_s2/ and refuses a non-pristine root")
    args = parser.parse_args(argv)

    if args.formal:
        return _run_formal_cli()

    with tempfile.TemporaryDirectory(prefix="s2_preflight_") as tmp:
        report = preflight(Path(tmp))

    print("RQ2/S2 production preflight (--validate-only)")
    print("=" * 66)
    print(f"protocol            {report['protocol_version']}")
    print(f"protocol checksum   {report['protocol_checksum']}...")
    print(f"cohort checksum     {report['cohort_checksum']}...")
    print(f"outcome source      {report['outcome_source']}")
    print(f"split manifest      {report['split_manifest_checksum']}...  "
          f"{report['registry']['n_cells']} cells, "
          f"{report['registry']['n_sites']} sites, "
          f"{report['registry']['n_repeats']} repeats")
    print(f"per-N sites         {report['registry']['per_n_site_counts']}")
    dec = report["expected_decisions"]
    print(f"expected decisions  {dec['total']}  by N {dec['by_n']}")
    print(f"                    by strategy {dec['by_strategy']}  "
          f"site-N cells {dec['site_n_cells']}")
    print(f"zero-label refs     {report['zero_label_decisions_expected']}")
    ev = report["expected_evaluations"]
    print(f"expected evals      {ev['total']}  "
          f"(label-consuming {ev['label_consuming']} + zero-label "
          f"{ev['zero_label']})  by N {ev['by_n']}")
    print(f"                    by strategy {ev['by_strategy']}")
    ident = report["site_identities"]
    print(f"site identities     per-N {ident['per_n_site_counts']}  "
          f"all sites {ident['n_all_sites']}  repeats "
          f"0..{ident['n_expected_repeats'] - 1}")
    print(f"CI registry         {report['ci_schema']['n_slots']} slots  "
          f"{report['ci_schema']['composition']}")
    ps = report["point_section_schema"]
    print(f"point sections      {ps['n_required']}/{ps['n_registered']} required "
          f"· {ps['by_phase']} · derived internally, no caller injection")
    print(f"constants           {report['scientific_constants']}")
    gl = report["global_ledger"]
    print(f"global ledger       one allocator · {len(gl['critical_events'])} "
          f"critical events · barrier {gl['barrier_seqs']['all_decisions_frozen_seq']}"
          f" < {gl['barrier_seqs']['phase2_authority_verified_seq']}"
          f" < {gl['barrier_seqs']['first_target_outcome_read_seq']}")
    print(f"                    separate-counter stream rejected: "
          f"{gl['separate_counter_stream_rejected']}")
    arch = report["phase2_architecture"]
    print(f"phase2 gating      {arch['phase_states']} · no standalone eval CLI "
          f"· worker outcome access: none")
    print(f"report inputs      context + ledger only "
          f"(no SiteLevelResults, no CI values, no manifest strings)")
    er = report["evaluation_root"]
    print(f"evaluation root    {er['evaluation_root']}")
    print(f"                   identity {er['evaluation_root_identity']}...  "
          f"derived from context, no store/root/path parameter anywhere")
    print(f"ledger binding     {er['ledger_binding']}")
    print(f"                   authoritative id "
          f"{er['authoritative_ledger_id'][:16]}... · repeated resolver same id "
          f"{er['repeated_for_context_same_id']} · complete second id rejected "
          f"{er['complete_second_ledger_rejected']}")
    print(f"mandatory ledger   aggregate{tuple(er['aggregation_signature'])} · "
          f"report{tuple(er['report_signature'][:2])} · "
          f"None/empty/foreign refused: "
          f"{er['none_empty_and_foreign_ledgers_refused']}")
    print(f"outcome reads      {report['target_outcome_rows_read']} rows "
          f"(sole path: s2_outcomes.load_authoritative_target_outcomes_for_phase2)")
    for audit in report["static_audits"] + report["wiring_audits"]:
        print(f"audit  {audit['status']:<5} {audit['audit']}")
    print("=" * 66)
    print(f"FORMAL RESULTS DIR  {FORMAL_RESULTS} "
          f"({'absent' if not FORMAL_RESULTS.exists() else 'PRESENT'})")
    print("PREFLIGHT: PASS — no target outcome read, no science value produced")
    return 0


def _run_formal_cli() -> int:
    """Dispatch the authoritative run and print what it committed to disk."""
    try:
        outcome = execute_formal_run()
    except FormalRunRefused as exc:
        print(f"FORMAL S2 REFUSED: {exc}", file=sys.stderr)
        return 3

    provenance = outcome["provenance"]
    completion = outcome["persisted"]["completion"]
    repository = provenance["repository"]
    barrier = completion["barrier_seqs"]
    print("RQ2/S2 formal execution (--formal)")
    print("=" * 66)
    print(f"run_id              {provenance['run_id']}")
    print(f"protocol            {provenance['protocol_version']}  "
          f"{repository['protocol_tag']} {repository['protocol_tag_sha'][:12]}")
    print(f"head                {repository['head_sha'][:12]}  "
          f"dirty={repository['working_tree_dirty']}")
    print(f"results root        {provenance['authoritative_roots']['workdir']}")
    print(f"ledger id           {barrier['ledger_id'][:16]}...")
    print(f"decisions           {provenance['expected_decisions']['total']} "
          f"+ {provenance['expected_zero_label_decisions']} zero-label")
    print(f"evaluations         {completion['n_evaluations']}")
    print(f"intervals           {completion['report_completeness']['n_intervals']}")
    print(f"point sections      "
          f"{completion['report_completeness']['n_point_sections']}")
    print(f"barrier             frozen {barrier['all_decisions_frozen_seq']}"
          f" < authority {barrier['phase2_authority_verified_seq']}"
          f" < open {barrier['phase2_open_seq']}"
          f" < first read {barrier['first_target_outcome_read_seq']}")
    print(f"outcome rows read   {completion['target_outcome_rows_read']}")
    for name in sorted(outcome["persisted"]["files"]):
        print(f"file   {name}")
    print("=" * 66)
    print("FORMAL S2: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
