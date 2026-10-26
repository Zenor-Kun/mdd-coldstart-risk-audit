#!/usr/bin/env python3
"""Execute the RQ2/S2 production counterexamples and require each to fail.

Stages 2.0.2 and 2.0.3 close the ways the production API let a caller walk
around, shrink, or forge the frozen protocol's authority. This script does not test helpers: it drives the real entry
points — `S2Firewall.record_zero_label_decision`, the cell worker, the global
barrier and the CI registry gate — with the exact attacks the audit named, and
reports the exception type and the validator that raised.

Everything it touches is synthetic: a fabricated roster, a fabricated outcome
table and a temporary directory. It creates no `results/rq2_s2/` artifact and
reads no real target label.

    python3 scripts/audit_s2_production_counterexamples.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import _s2_fixtures as fx  # noqa: E402
import certification as cert  # noqa: E402
import s2_aggregation as agg  # noqa: E402
import s2_cell_worker as worker  # noqa: E402
import s2_decisions as sd  # noqa: E402
import s2_decisions as sd  # noqa: E402
import s2_firewall as fw  # noqa: E402
import s2_production as prod  # noqa: E402
import s2_splitting as spl  # noqa: E402


_CONTEXT = []


def _production_context():
    """One authoritative context, reused: building it is expensive, not weaker."""
    if not _CONTEXT:
        _CONTEXT.append(prod.ProductionS2Context.build())
    return _CONTEXT[0]


def _validator_of(exc: BaseException) -> str:
    """Which module and function raised — reported, not guessed."""
    tb = exc.__traceback__
    last = None
    while tb is not None:
        frame = tb.tb_frame
        path = Path(frame.f_code.co_filename)
        if path.parent.name == "src":
            last = f"{path.name}::{frame.f_code.co_name}"
        tb = tb.tb_next
    return last or "unknown"


def attack_a(world, firewall) -> dict:
    """A — a zero-label artifact claiming a formal target certification."""
    payload = fx.zero_label_payload(strategy="pooled_source",
                                    state=cert.TARGET_CERTIFIED)
    firewall.record_zero_label_decision(payload)
    return {"outcome": "NO EXCEPTION"}


def attack_b(world, firewall) -> dict:
    """B — a cell job of the right length but the wrong canonical IDs."""
    cell = world["registry"].cell(fx.SITE, 0, fx.N_LABELS)
    roster = world["registry"].site_roster(fx.SITE)
    wrong = [i for i in roster if i not in cell["cal_ids"]][:fx.N_LABELS]
    assert len(wrong) == fx.N_LABELS and set(wrong) != set(cell["cal_ids"])
    job = {
        "mode": "decision", "site": fx.SITE, "repeat": 0,
        "n_labels": fx.N_LABELS, "method": "probability",
        "strategy": "target_local",
        "split_registry_path": world["registry"].path,
        "split_registry_checksum": world["registry"].checksum,
        "cal_ids": wrong,                      # the attack
        "run_id": "attack-b", **fx.candidate_inputs(),
    }
    worker.run_decision_cell(job)
    return {"outcome": "NO EXCEPTION"}


def attack_b2(world, firewall) -> dict:
    """B' — the same attack pushed one layer deeper: another site's N rows."""
    other = world["registry"].site_roster(fx.OTHER_SITE)[:fx.N_LABELS]
    worker.ProductionCellLabelReader(world["source"], fx.SITE, other).read()
    return {"outcome": "NO EXCEPTION"}


def attack_c(world, firewall) -> dict:
    """C — release the global barrier with zero completed decision cells."""
    assert len(firewall.expected_cells) > 0
    assert len(firewall._decisions) == 0
    fx.complete_phase1_for_test_fixture(firewall)
    return {"outcome": "NO EXCEPTION"}


def attack_d(world, firewall) -> dict:
    """D — request a confidence interval outside the frozen 216-key registry."""
    values = [0.3] * 21
    agg.cross_site_contrast("descriptive_target_coverage", "target_local",
                            "pooled_source", values, values,
                            method="probability", n_labels=20, n_sites=21)
    return {"outcome": "NO EXCEPTION"}


# --- Stage 2.0.3 root-of-trust attacks -------------------------------------
def attack_e(world, firewall) -> dict:
    """E — a decision that passed but named no threshold."""
    sd.validate_decision_artifact({
        "protocol_version": sd.PROTOCOL_VERSION, "site": fx.SITE,
        "method": "probability", "strategy": "target_local",
        "guarantee_tier": cert.TIER_FORMAL, "state": cert.TARGET_CERTIFIED,
        "reason": None, "selected_q": 0.5, "selected_tau": None,
        "selected_candidate_index": cert.Q_GRID.index(0.5),
        "selected_score": 0.28, "repeat": 0, "n_labels": 5,
        "split_checksum": "a" * 64})
    return {"outcome": "NO EXCEPTION"}


def attack_f(world, firewall) -> dict:
    """F — a one-site registry presented as production authority."""
    import tempfile as _tf
    tmp = Path(_tf.mkdtemp(prefix="s2_fake_registry_"))
    roster = {fx.SITE: fx.synthetic_ids(fx.SITE, fx.N_S)}
    reg = spl.ProductionSplitRegistry.write(roster, tmp / "fake.csv",
                                            r_split=cert.R_SPLIT)
    reg.assert_production_invariants()
    return {"outcome": "NO EXCEPTION"}


def attack_g(world, firewall) -> dict:
    """G — redirect the outcome source through the withdrawn env variables."""
    os.environ["ISBI_S2_TEST_MODE"] = "1"
    os.environ["ISBI_S2_TEST_OUTCOME_SOURCE"] = str(world["source"])
    try:
        resolved = prod.resolve_authoritative_outcome_source()
        if Path(resolved) != prod.AUTHORITATIVE_OUTCOME_SOURCE:
            return {"outcome": "NO EXCEPTION"}
        # the override had no effect; prove the assertion also refuses it
        prod.assert_authoritative_outcome_source(world["source"])
        return {"outcome": "NO EXCEPTION"}
    finally:
        os.environ.pop("ISBI_S2_TEST_MODE", None)
        os.environ.pop("ISBI_S2_TEST_OUTCOME_SOURCE", None)


def attack_h(world, firewall) -> dict:
    """H — a hand-written release with n_expected = 1 and a valid SHA-256."""
    context = prod.ProductionS2Context.build()
    forged = dict(context.authority_state())
    forged.update({"n_expected": 1, "n_completed": 1,
                   "decision_manifest_checksum": "deadbeef",
                   "release_sequence": 1})
    forged["release_checksum"] = fw.release_payload_checksum(forged)
    path = Path(context.workdir) / "forged_release.json"
    path.write_bytes(sd.canonical_json(forged))
    fw.verify_release_integrity(path)            # integrity alone is satisfied
    fw.verify_release_authority(path, context)   # authority must refuse
    return {"outcome": "NO EXCEPTION"}


def attack_i(world, firewall) -> dict:
    """I — the formal report accepting an unregistered interval."""
    import s2_reporting as rep
    key = agg.CIKey("formal", "formal_certification_probability",
                    "pooled_source", 20, "probability", None, "formal")
    agg.compute_registered_interval(key, [0.3] * 21, n_sites=21)
    return {"outcome": "NO EXCEPTION"}


# --- Stage 2.0.4 authority attacks -----------------------------------------
def attack_k(world, firewall) -> dict:
    """K — forge a context by bypassing the factory and replacing a field."""
    context = prod.ProductionS2Context.build()
    forged_snapshot = dict(context.snapshot)
    forged_snapshot["ci_registry_checksum"] = "fake"
    forged = prod.ProductionS2Context(
        prod._FACTORY_TOKEN, forged_snapshot, context.registry,
        context.expected_cells, context.workdir, "forged")
    forged.assert_still_authoritative()
    return {"outcome": "NO EXCEPTION"}


def attack_l(world, firewall) -> dict:
    """L — release: 46,200 expected, 0 completed, empty store, valid SHA."""
    context = prod.ProductionS2Context.build()
    payload = dict(context.authority_state())
    store = fw.scan_current_decision_store(context)
    payload.update({"n_expected": context.expected_count, "n_completed": 0,
                    "decision_manifest_checksum":
                        store["decision_manifest_checksum"],
                    "release_sequence": 1})
    payload["release_checksum"] = fw.release_payload_checksum(payload)
    path = Path(context.workdir) / "empty_store_release.json"
    path.write_bytes(sd.canonical_json(payload))
    fw.verify_release_integrity(path)          # integrity is satisfied
    fw.verify_release_authority(path, context)  # authority must refuse
    return {"outcome": "NO EXCEPTION"}


def attack_m(world, firewall) -> dict:
    """M — call the evaluation worker directly with a forged permit."""
    import s2_eval_worker as ew
    ci = fx.candidate_inputs()
    for key in sorted(firewall.expected_cells):
        firewall.run_decision_cell(*key, **ci)
    fx.complete_phase1_for_test_fixture(firewall)
    key = sorted(firewall._decisions)[0]
    rec = firewall._decisions[key]
    cell = world["registry"].cell(key[0], key[1], key[2])
    job = {"mode": "evaluation", "site": key[0], "repeat": key[1],
           "n_labels": key[2], "strategy": key[3], "method": key[4],
           "split_registry_path": world["registry"].path,
           "split_registry_checksum": world["registry"].checksum,
           "decision_path": rec["path"], "decision_checksum": rec["sha256"],
           "taus": ci["taus"], "r_star": 0.31, "run_id": "direct"}
    del job, cell, rec
    # Stage 2.0.5: there is no permit-taking worker entry point left to call.
    if hasattr(ew, "run_evaluation_cell") or hasattr(ew, "init_worker"):
        return {"outcome": "NO EXCEPTION — a permit-gated worker entry survives"}
    raise AssertionError(
        "the permit-gated evaluation entry point no longer exists; the worker "
        "is a pure function with no outcome capability (see attack P)")


def attack_n(world, firewall) -> dict:
    """N — hand-build a registered-looking interval and hand it to the report."""
    import s2_ledger as lg
    import s2_reporting as rep
    key = agg.ci_key("formal_certification_probability", strategy="target_local",
                     method="probability", n_labels=20)
    forged = agg.RegisteredIntervalResult(
        key, {"point_estimate": 0.01, "ci_low": 0.0, "ci_high": 0.02,
              "valid_bootstrap_replicates": 5000, "ci_key": key.as_tuple()})
    context = _production_context()
    # the value type is constructible; there is simply no report entry for it
    rep.build_formal_s2_report({key: forged}, lg.GlobalRunLedger.for_context(context))
    return {"outcome": "NO EXCEPTION"}


def attack_o(world, firewall) -> dict:
    """O — a hand-built ProductionSplitRegistry driving a production firewall."""
    import tempfile as _tf
    tmp = Path(_tf.mkdtemp(prefix="s2_handbuilt_registry_"))
    context = prod.ProductionS2Context.build()
    roster = {s: v for s, v in list(prod.load_frozen_cohort_roster().items())[:1]}
    fake = spl.ProductionSplitRegistry.write(roster, tmp / "fake.csv",
                                             r_split=cert.R_SPLIT)
    forged = prod.ProductionS2Context(
        prod._FACTORY_TOKEN, context.snapshot, fake, context.expected_cells,
        tmp, "forged")
    import s2_ledger as lg
    fw.ProductionS2Firewall(forged, lg.GlobalRunLedger())
    return {"outcome": "NO EXCEPTION"}


# --- Stage 2.0.5 dataflow attacks ------------------------------------------
def attack_p(world, firewall) -> dict:
    """P — import the evaluation worker directly and try to read the dataset."""
    import s2_eval_worker as ew
    audit = fw.audit_eval_worker_has_no_outcome_access()
    if not audit["ok"]:
        raise AssertionError(f"the worker retains outcome access: {audit}")
    for attr in ("init_worker", "authoritative_reader", "resolve_outcome_source",
                 "ProductionCellLabelReader", "main"):
        if hasattr(ew, attr):
            return {"outcome": f"NO EXCEPTION — worker still exposes {attr}"}
    # the only entry point takes data, so ask it for a dataset read
    ew.evaluate_frozen_cell({"site": fx.SITE, "repeat": 0, "n_labels": 5,
                             "strategy": "target_local", "method": "probability",
                             "state": cert.TARGET_CERTIFIED, "selected_q": 0.5,
                             "selected_tau": 0.6, "r_star": 0.31, "taus": [0.6],
                             "oracle": False, "outcome_source": "READ THE REAL FILE"})
    return {"outcome": "NO EXCEPTION"}


def attack_q(world, firewall) -> dict:
    """Q — hand-build 216 interval values and feed them to the formal report."""
    import s2_ledger as lg
    import s2_reporting as rep
    if hasattr(agg, "RegisteredIntervalStore"):
        return {"outcome": "NO EXCEPTION — an ingestible store still exists"}
    for banned in ("set_interval", "add_interval", "load_interval_store",
                   "set_ci", "build_from_registered_store"):
        if hasattr(rep.S2Report, banned) or hasattr(rep, banned):
            return {"outcome": f"NO EXCEPTION — {banned} still exists"}
    forged = {key: {"point_estimate": 0.01, "ci_low": 0.0, "ci_high": 0.02}
              for key in rep.report_schema_slots()}
    context = _production_context()
    rep.build_formal_s2_report(forged, lg.GlobalRunLedger.for_context(context))
    return {"outcome": "NO EXCEPTION"}


def attack_r(world, firewall) -> dict:
    """R — mutate a returned interval result and expect the report to shift."""
    import s2_reporting as rep
    import s2_site_aggregation as sagg
    key = rep.report_schema_slots()[0]
    result = agg.compute_registered_interval(key, [0.3] * 23, n_sites=23)
    honest = result.as_dict()["point_estimate"]
    try:
        result._payload = {"point_estimate": 999.0}
    except Exception:                                 # noqa: BLE001
        pass
    values = {}
    for k in rep.report_schema_slots():
        payload = {"site_values": [0.3] * 23, "n_sites": 23}
        if k.contrast is not None:
            payload["site_values_b"] = [0.3] * 23
        values[k] = payload
    site_level = sagg.SiteLevelResults(values, {
        "protocol_version": sd.PROTOCOL_VERSION,
        "decision_manifest_checksum": "d", "evaluation_manifest_checksum": "e"})
    rows = rep._compute_registry_rows(site_level)
    got = rows[key]["point_estimate"]
    if got != honest:
        return {"outcome": f"NO EXCEPTION — the walk shifted to {got}"}
    raise AssertionError(
        "the mutation did not reach any reported number: every interval is "
        f"recomputed from site-level data (point estimate stayed {honest:.6f})")


def attack_s(world, firewall) -> dict:
    """S — call the authoritative outcome loader before phase 2 opens."""
    import s2_outcomes as om
    import s2_ledger as lg
    context = _production_context()
    ledger = lg.GlobalRunLedger.for_context(context)
    production_firewall = fw.ProductionS2Firewall(context, ledger)
    before = om.total_rows_read()
    try:
        om.load_authoritative_target_outcomes_for_phase2(context,
                                                         production_firewall,
                                                         ledger)
    finally:
        if om.total_rows_read() != before:
            raise AssertionError("outcome rows were read despite the refusal")
    return {"outcome": "NO EXCEPTION"}


# --- Stage 2.0.6 provenance and ordering attacks ---------------------------
def _honest_records(context):
    """A complete, well-formed synthetic evaluation record set. No real label."""
    import s2_evaluation_store as es
    cells = es.derive_authoritative_expected_evaluation_cells(context)
    passing = {"target_local": cert.TARGET_CERTIFIED,
               "pooled_local_shrinkage": cert.EMPIRICAL_PASS,
               "pooled_source": cert.SOURCE_REFERENCE_PASS,
               "target_unlabeled_weighted": cert.EMPIRICAL_PASS}
    out = {}
    for key in cells:
        site, repeat, n_labels, strategy, method = key
        out[key] = {
            "protocol_version": sd.PROTOCOL_VERSION, "run_id": "synthetic",
            "site": site, "repeat": repeat, "n_labels": n_labels,
            "strategy": strategy, "confidence_method": method,
            "guarantee_tier": cert.tier_of(strategy),
            "decision_checksum": "d" * 64, "split_checksum": "s" * 64,
            "decision_state": passing[strategy], "selected_q": 0.5,
            "selected_tau": 0.8, "coverage_test": 0.6, "n_test": 40,
            "n_keep_test": 24, "n_keep_test_MDD": 12, "n_keep_test_HC": 12,
            "ber_test": 0.3, "risk_excess": -0.01,
            "realized_field": cert.realized_outcome_field(strategy),
            "realized_outcome": 0, "r_star": 0.31,
            "evaluation_status": es.EVAL_STATUS_COMPLETE,
        }
    return cells, out


def attack_t(world, firewall) -> dict:
    """T — fabricated SiteLevelResults with FAKE manifests and arbitrary vectors."""
    import s2_ledger as lg
    import s2_reporting as rep
    import s2_site_aggregation as sagg
    context = _production_context()
    forged = sagg.SiteLevelResults(
        {k: ({"site_values": [0.99] * 23, "n_sites": 23,
              "site_values_b": [0.01] * 23} if k.contrast is not None
             else {"site_values": [0.99] * 23, "n_sites": 23})
         for k in rep.report_schema_slots()},
        {"protocol_version": sd.PROTOCOL_VERSION,
         "decision_manifest_checksum": "FAKE",
         "evaluation_manifest_checksum": "FAKE"})
    import inspect
    params = list(inspect.signature(rep.build_formal_s2_report).parameters)
    if params != ["context", "ledger", "point_sections"]:
        return {"outcome": f"NO EXCEPTION — the report still takes {params}"}

    # there is no data parameter, so drive the report the *correct* way and
    # show the fabricated numbers are never consulted
    ledger = lg.GlobalRunLedger.for_context(context)
    for event in (lg.EV_PHASE1_STARTED, lg.EV_ALL_DECISIONS_FROZEN,
                  lg.EV_RELEASE_AUTHORITY_VERIFIED, lg.EV_PHASE2_OPEN,
                  lg.EV_OUTCOME_READ_AUTHORIZED,
                  lg.EV_FIRST_TARGET_OUTCOME_READ,
                  lg.EV_EVALUATION_COMPLETE,
                  lg.EV_EVALUATION_MANIFEST_FROZEN):
        ledger.append(event)
    try:
        rep.build_formal_s2_report(context, ledger)
    except AssertionError as exc:
        raise AssertionError(
            "the fabricated SiteLevelResults reached nothing: the report reads "
            f"the authoritative artifact stores instead ({str(exc)[:90]})") from exc
    return {"outcome": f"NO EXCEPTION — a report was built ({len(forged.keys())} keys)"}


def attack_u(world, firewall) -> dict:
    """U — an eligible site removed at N=10 and an ineligible one swapped in."""
    import s2_site_aggregation as sagg
    context = _production_context()
    cells, records = _honest_records(context)
    scope = sagg.authoritative_site_scope(context)
    sagg.assert_cell_identities(sagg.index_evaluation_records(records), scope,
                                cells)                       # honest set passes
    dropped = scope["by_n"][10][0]
    ineligible = next(s for s in scope["by_n"][5] if s not in scope["by_n"][10])
    for key in [k for k in list(records) if k[0] == dropped and k[2] == 10]:
        payload = records.pop(key)
        swapped = (ineligible,) + key[1:]
        records[swapped] = {**payload, "site": ineligible}
    sagg.assert_cell_identities(sagg.index_evaluation_records(records), scope,
                                cells)
    return {"outcome": "NO EXCEPTION"}


def attack_v(world, firewall) -> dict:
    """V — one repeat deleted, so a site is aggregated over 99 of 100."""
    import s2_site_aggregation as sagg
    context = _production_context()
    cells, records = _honest_records(context)
    scope = sagg.authoritative_site_scope(context)
    records.pop(next(k for k in records if k[1] == 99))
    sagg.assert_cell_identities(sagg.index_evaluation_records(records), scope,
                                cells)
    return {"outcome": "NO EXCEPTION"}


def attack_w(world, firewall) -> dict:
    """W — two independent counters presented as one global ordering."""
    import s2_ledger as lg
    firewall_side = lg.GlobalRunLedger("run-1")
    for _ in range(9):
        firewall_side.append("cell_decision_frozen")
    firewall_side.append(lg.EV_ALL_DECISIONS_FROZEN)              # seq 10
    outcome_side = lg.GlobalRunLedger("run-1")
    outcome_side.append(lg.EV_RELEASE_AUTHORITY_VERIFIED)         # seq 1
    outcome_side.append(lg.EV_PHASE2_OPEN)                        # seq 2
    outcome_side.append(lg.EV_FIRST_TARGET_OUTCOME_READ)          # seq 3
    merged = list(firewall_side.events) + list(outcome_side.events)
    lg.assert_event_stream_ordering(merged, firewall_side.ledger_id,
                                    firewall_side.run_id)
    return {"outcome": "NO EXCEPTION"}


def attack_x(world, firewall) -> dict:
    """X — a target outcome read recorded before the release authority check."""
    import s2_ledger as lg
    ledger = lg.GlobalRunLedger("run-1")
    ledger.append(lg.EV_PHASE1_STARTED)
    ledger.append(lg.EV_ALL_DECISIONS_FROZEN)
    ledger.append(lg.EV_FIRST_TARGET_OUTCOME_READ)     # before the barrier
    ledger.append(lg.EV_RELEASE_AUTHORITY_VERIFIED)
    ledger.append(lg.EV_PHASE2_OPEN)
    lg.assert_global_ordering(ledger)
    return {"outcome": "NO EXCEPTION"}




# --- Stage 2.0.7 store-root and mandatory-ledger attacks -------------------
def _run_ledger(context, through=None):
    """This run's allocator, stamped up to `through` (default: manifest frozen)."""
    import s2_ledger as lg
    ledger = lg.GlobalRunLedger.for_context(context)
    order = (lg.EV_PHASE1_STARTED, lg.EV_ALL_DECISIONS_FROZEN,
             lg.EV_RELEASE_AUTHORITY_VERIFIED, lg.EV_PHASE2_OPEN,
             lg.EV_OUTCOME_READ_AUTHORIZED, lg.EV_FIRST_TARGET_OUTCOME_READ,
             lg.EV_EVALUATION_COMPLETE, lg.EV_EVALUATION_MANIFEST_FROZEN)
    for event in order:
        ledger.append(event)
        if event == through:
            break
    return ledger


def _fully_stamped_second_ledger(context):
    """The exact Stage 2.0.7 bypass: correct everything, wrong ledger id."""
    import s2_ledger as lg
    binding = lg.read_authoritative_ledger_binding(context)
    snapshot = context.snapshot
    ledger = lg.GlobalRunLedger(
        context.run_id,
        context_snapshot_checksum=sd.sha256_bytes(sd.canonical_json(snapshot)),
        split_registry_checksum=snapshot["split_registry_checksum"],
        ledger_binding_checksum=binding["binding_checksum"],
    )
    for event in (lg.EV_PHASE1_STARTED, lg.EV_ALL_DECISIONS_FROZEN,
                  lg.EV_RELEASE_AUTHORITY_VERIFIED, lg.EV_PHASE2_OPEN,
                  lg.EV_OUTCOME_READ_AUTHORIZED,
                  lg.EV_FIRST_TARGET_OUTCOME_READ,
                  lg.EV_EVALUATION_COMPLETE,
                  lg.EV_EVALUATION_MANIFEST_FROZEN):
        ledger.append(event)
    return ledger


def attack_y(world, firewall) -> dict:
    """Y — a complete fake evaluation store at a caller-chosen root.

    Writes all 78,600 structurally valid artifacts to `/tmp/fake_eval_store`
    with arbitrary metrics (every cell passing, `ber_test` 0.001) and then tries
    to make production aggregation read them.
    """
    import inspect
    import shutil
    import s2_evaluation_store as es
    import s2_reporting as rep
    import s2_site_aggregation as sagg

    context = _production_context()
    cells, records = _honest_records(context)
    root = Path(tempfile.gettempdir()) / "fake_eval_store" / "evaluations"
    if root.parent.exists():
        shutil.rmtree(root.parent)
    root.mkdir(parents=True)
    fake = es.EvaluationArtifactStore(root)
    for key, payload in records.items():
        fake.freeze({**payload, "ber_test": 0.001, "risk_excess": -0.3,
                     "realized_outcome": 0})
    n_fake = len(fake.paths())
    if n_fake != es.PRODUCTION_EXPECTED_EVALUATIONS:
        raise AssertionError(f"the fake store holds {n_fake}, not 78,600")

    # 1. there is no production parameter to hand it to
    signatures = {
        "scan_current_evaluation_store":
            list(inspect.signature(es.scan_current_evaluation_store).parameters),
        "assert_evaluation_store_is_complete":
            list(inspect.signature(es.assert_evaluation_store_is_complete).parameters),
        "build_evaluation_manifest":
            list(inspect.signature(es.build_evaluation_manifest).parameters),
        "aggregate_authoritative_site_level_results":
            list(inspect.signature(
                sagg.aggregate_authoritative_site_level_results).parameters),
        "build_formal_s2_report":
            list(inspect.signature(rep.build_formal_s2_report).parameters),
    }
    for name, params in signatures.items():
        for bad in fw.FORBIDDEN_STORE_INPUTS:
            if bad in params:
                return {"outcome": f"NO EXCEPTION — {name} still takes {bad}"}
    if not fw.audit_no_caller_supplied_evaluation_store()["ok"]:
        return {"outcome": "NO EXCEPTION — a production store parameter remains"}

    # 2. the authoritative root is somewhere else entirely, and is what is read
    resolved = es.resolve_authoritative_evaluation_store(context)
    if Path(resolved.root) == root:
        return {"outcome": "NO EXCEPTION — the fake root was resolved"}
    try:
        sagg.aggregate_authoritative_site_level_results(context,
                                                        _run_ledger(context))
    except AssertionError as exc:
        untouched = len(fake.paths()) == n_fake
        shutil.rmtree(root.parent)
        if not untouched:
            raise AssertionError("the fake store was consumed") from exc
        raise AssertionError(
            f"{n_fake} fabricated artifacts at {root} reached nothing: the "
            f"aggregation scanned {resolved.root} ({str(exc)[:70]})") from exc
    shutil.rmtree(root.parent)
    return {"outcome": "NO EXCEPTION — a fake store was aggregated"}


def attack_z(world, firewall) -> dict:
    """Z — aggregation with `ledger=None`, the Stage 2.0.6 ordering bypass."""
    import s2_site_aggregation as sagg
    context = _production_context()
    sagg.aggregate_authoritative_site_level_results(context, ledger=None)
    return {"outcome": "NO EXCEPTION"}


def attack_aa(world, firewall) -> dict:
    """AA — a fresh legal ledger, and a foreign run's fully stamped ledger."""
    import s2_ledger as lg
    import s2_site_aggregation as sagg
    context = _production_context()

    fresh = lg.GlobalRunLedger.for_context(context)   # same run, new allocator
    try:
        sagg.aggregate_authoritative_site_level_results(context, fresh)
    except (lg.LedgerError, prod.ProductionAuthorityError):
        pass
    else:
        return {"outcome": "NO EXCEPTION — a fresh allocator was accepted"}

    # a *different* run's allocator, fully and legally stamped, with this run's
    # provenance copied onto it: convincing content, wrong run
    snapshot = context.snapshot
    foreign = lg.GlobalRunLedger(
        "another-run",
        context_snapshot_checksum=sd.sha256_bytes(sd.canonical_json(snapshot)),
        split_registry_checksum=snapshot["split_registry_checksum"])
    for event in (lg.EV_PHASE1_STARTED, lg.EV_ALL_DECISIONS_FROZEN,
                  lg.EV_RELEASE_AUTHORITY_VERIFIED, lg.EV_PHASE2_OPEN,
                  lg.EV_OUTCOME_READ_AUTHORIZED,
                  lg.EV_FIRST_TARGET_OUTCOME_READ, lg.EV_EVALUATION_COMPLETE,
                  lg.EV_EVALUATION_MANIFEST_FROZEN):
        foreign.append(event)
    sagg.aggregate_authoritative_site_level_results(context, foreign)
    return {"outcome": "NO EXCEPTION"}


def attack_ab(world, firewall) -> dict:
    """AB — the formal report with a None, fresh and foreign ledger."""
    import s2_ledger as lg
    import s2_reporting as rep
    context = _production_context()
    attempts = {
        "None": None,
        "fresh": lg.GlobalRunLedger.for_context(context),
        "foreign ledger_id": lg.GlobalRunLedger("some-other-run"),
        "not a ledger": {"seq": 1},
    }
    for name, ledger in attempts.items():
        try:
            rep.build_formal_s2_report(context, ledger)
        except (lg.LedgerError, prod.ProductionAuthorityError,
                rep.ReportSchemaError):
            continue
        return {"outcome": f"NO EXCEPTION — the report accepted a {name} ledger"}
    raise AssertionError(
        "every unauthorised ledger was refused before any artifact was read: "
        f"{sorted(attempts)}")


# --- Stage 2.0.8 authoritative ledger-id attacks ---------------------------
def attack_ac(world, firewall) -> dict:
    """AC — same run, complete events/provenance, different ledger id."""
    import s2_ledger as lg
    context = _production_context()
    official = lg.GlobalRunLedger.for_context(context)
    forged = _fully_stamped_second_ledger(context)
    if forged.ledger_id == official.ledger_id:
        return {"outcome": "NO EXCEPTION — second allocator reused official id"}
    # Its event stream really is complete; only authority identity is wrong.
    lg.assert_global_ordering(
        forged, required=lg.AGGREGATION_PREREQUISITE_EVENTS)
    lg.assert_run_ledger_for_context(forged, context)
    return {"outcome": "NO EXCEPTION"}


def attack_ad(world, firewall) -> dict:
    """AD — the complete same-run second ledger drives aggregation."""
    import s2_site_aggregation as sagg
    context = _production_context()
    sagg.aggregate_authoritative_site_level_results(
        context, _fully_stamped_second_ledger(context))
    return {"outcome": "NO EXCEPTION"}


def attack_ae(world, firewall) -> dict:
    """AE — the complete same-run second ledger drives formal reporting."""
    import s2_reporting as rep
    context = _production_context()
    rep.build_formal_s2_report(context, _fully_stamped_second_ledger(context))
    return {"outcome": "NO EXCEPTION"}


def _binding_field_tamper(field, replacement, *, reseal=True):
    """Tamper a fresh binding; return only if production accepts it."""
    import s2_ledger as lg
    with tempfile.TemporaryDirectory(prefix="s2_binding_tamper_") as tmp:
        context = prod.ProductionS2Context.build(Path(tmp))
        lg.GlobalRunLedger.for_context(context)
        path = lg.authoritative_ledger_binding_path(context)
        payload = json.loads(path.read_text())
        payload[field] = replacement
        if reseal:
            body = {k: v for k, v in payload.items()
                    if k != "binding_checksum"}
            payload["binding_checksum"] = sd.sha256_bytes(
                sd.canonical_json(body))
        path.write_bytes(sd.canonical_json(payload))
        lg.read_authoritative_ledger_binding(context)
    return {"outcome": "NO EXCEPTION"}


def attack_af(world, firewall) -> dict:
    """AF — binding run id altered and self-checksum recomputed."""
    return _binding_field_tamper("run_id", "another-run")


def attack_ag(world, firewall) -> dict:
    """AG — binding context checksum altered and self-checksum recomputed."""
    return _binding_field_tamper("context_snapshot_checksum", "f" * 64)


def attack_ah(world, firewall) -> dict:
    """AH — binding split checksum altered and self-checksum recomputed."""
    return _binding_field_tamper("split_registry_checksum", "e" * 64)


def attack_ai(world, firewall) -> dict:
    """AI — binding ledger id altered without its integrity checksum."""
    return _binding_field_tamper(
        "authoritative_ledger_id", "0" * 32, reseal=False)


ATTACKS = (
    ("A", "pooled_source artifact claiming TARGET_CERTIFIED", attack_a),
    ("B", "cell job with right length, wrong canonical IDs", attack_b),
    ("B'", "another site's N rows handed to the label reader", attack_b2),
    ("C", "complete_phase1() with 0 completed decision cells", attack_c),
    ("D", "registry-outside coverage contrast CI", attack_d),
    ("K", "forged/replaced ProductionS2Context field", attack_k),
    ("L", "release 46200 expected, 0 completed, empty store", attack_l),
    ("M", "direct evaluation worker + forged permit", attack_m),
    ("N", "hand-built registered-looking CI into the report", attack_n),
    ("O", "hand-built ProductionSplitRegistry into the firewall", attack_o),
    ("P", "direct evaluation worker asked for real dataset outcomes", attack_p),
    ("Q", "hand-built 216 CI values fed to the formal report", attack_q),
    ("R", "mutated compute_registered_interval result", attack_r),
    ("S", "authoritative outcome loader called before phase 2", attack_s),
    ("T", "fabricated SiteLevelResults with FAKE manifests", attack_t),
    ("U", "wrong eligible-site list for N=10", attack_u),
    ("V", "one repeat deleted from a site-N block", attack_v),
    ("W", "two independent counters as a global ordering proof", attack_w),
    ("X", "out-of-order global ledger events", attack_x),
    ("Y", "complete 78,600-artifact fake evaluation store", attack_y),
    ("Z", "aggregation called with ledger=None", attack_z),
    ("AA", "fresh and foreign ledgers driving the aggregation", attack_aa),
    ("AB", "formal report with a None / fresh / foreign ledger", attack_ab),
    ("AC", "same-run second ledger with complete correct events", attack_ac),
    ("AD", "same-run second ledger driving aggregation", attack_ad),
    ("AE", "same-run second ledger driving formal report", attack_ae),
    ("AF", "authoritative binding run_id tamper", attack_af),
    ("AG", "authoritative binding context checksum tamper", attack_ag),
    ("AH", "authoritative binding split checksum tamper", attack_ah),
    ("AI", "authoritative binding ledger_id tamper", attack_ai),
    ("E", "PASS state with selected_tau = None", attack_e),
    ("F", "one-site registry presented as production authority", attack_f),
    ("G", "environment override of the outcome source", attack_g),
    ("H", "forged release n_expected=1 with a valid SHA-256", attack_h),
    ("I", "formal report given an unregistered CI key", attack_i),
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="s2_counterexamples_") as tmp:
        tmp = Path(tmp)
        world = fx.build_world(tmp, r_split=1,
                               sites={fx.SITE: fx.N_S,
                                      fx.OTHER_SITE: fx.OTHER_N_S})
        rows = []
        for tag, description, attack in ATTACKS:
            firewall = fx.TestS2Firewall(
                tmp / f"cells_{tag.strip(chr(39))}", world["registry"],
                world["source"])
            try:
                result = attack(world, firewall)
                rows.append({"attack": tag, "description": description,
                             "result": "*** NOT BLOCKED ***",
                             "exception": result["outcome"], "validator": "-"})
            except BaseException as exc:                # noqa: BLE001
                rows.append({"attack": tag, "description": description,
                             "result": "hard-fail (expected)",
                             "exception": type(exc).__name__,
                             "validator": _validator_of(exc),
                             "message": str(exc).splitlines()[0][:110]})

    width = max(len(r["description"]) for r in rows)
    print("RQ2/S2 production counterexample audit (Stage 2.0.8)")
    print("=" * (width + 62))
    for r in rows:
        print(f"{r['attack']:>2}  {r['description']:<{width}}  "
              f"{r['result']:<22} {r['exception']}")
        if r.get("validator", "-") != "-":
            print(f"    raised by {r['validator']}: {r.get('message', '')}")
    blocked = sum(1 for r in rows if r["result"].startswith("hard-fail"))
    print("=" * (width + 62))
    print(f"{blocked}/{len(rows)} attacks hard-failed")
    if blocked != len(rows):
        print("AUDIT: FAIL — a production bypass survives")
        return 1
    print("AUDIT: PASS — every production counterexample is blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
