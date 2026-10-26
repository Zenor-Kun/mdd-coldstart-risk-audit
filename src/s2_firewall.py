"""Cell-isolated firewall and global phase barrier (RQ2/S2, Stage 2.0.8).

Stage 2.0.6 changes two things here and nothing else. The firewall no longer
owns a sequence counter: every event it records is allocated by the run's single
`s2_ledger.GlobalRunLedger`, so its numbers are comparable with the outcome
loader's, the evaluation store's and the report's. And a phase-2 evaluation now
produces a **frozen evaluation artifact** through `s2_evaluation_store`, which
is what the formal aggregation later reads — the parent no longer hands the
report numbers it is holding in memory.

The original Stage 2.0.2 header follows, unchanged.

Stage 2.0.1 made the isolation unit a `(site, repeat, N)` cell behind a process
boundary. Stage 2.0.2 closes the production paths that could still walk around
the frozen invariants:

* `record_zero_label_decision` hashed whatever JSON it was given, so a
  `pooled_source` artifact carrying `TARGET_CERTIFIED` could be frozen. Every
  decision now goes through `s2_decisions.freeze_decision_artifact`:
  **validate -> canonical serialize -> write -> checksum -> freeze**.
* A cell job carried its own `cal_rows` and could redirect `label_source`. Jobs
  now name only `(site, repeat, N)` and a registered split manifest; the worker
  resolves membership itself and the outcome source is an internal constant.
* `complete_phase1(expected_cells=None)` accepted an empty decision set and
  still released. The expected set is now derived mechanically from the split
  registry and the release requires **exact set equality**.
* Evaluation trusted an in-process flag. It now requires an unforgeable global
  **release artifact**, checksummed over the protocol version, the split
  manifest, the expected-cell registry and the decision manifest.

The three global phases are unchanged:

    PHASE 0  zero-label decisions   — no target label is opened at all
    PHASE 1  cell decisions         — that cell's y_cal only
    ======== GLOBAL RELEASE: exact expected-cell set, all artifacts valid ======
    PHASE 2  cell evaluations       — that cell's y_test only
"""

from __future__ import annotations

import ast
import hashlib
import os
import hmac
import json
import secrets
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import certification as cert
import s2_decisions as sd
import s2_evaluation_store as evalstore
import s2_ledger as ledgermod

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WORKER = SRC / "s2_cell_worker.py"
PROTOCOL_DOC = ROOT / "docs" / "RQ2_S2_PROTOCOL.md"

#: Modules that run in the parent process and must never touch a target label.
PARENT_MODULES = ("src/s2_firewall.py", "src/s2_splitting.py",
                  "src/s2_aggregation.py", "src/certification.py",
                  "src/shrinkage.py", "src/s2_decisions.py",
                  "src/s2_ledger.py", "src/s2_evaluation_store.py",
                  "src/s2_site_aggregation.py")
LABEL_READING_MODULE = "src/s2_cell_worker.py"
PARENT_FORBIDDEN_CALLS = ("CellLabelReader", "build_xy", "load_target_y",
                          "load_cohort_labels")

#: Production modules whose formal intervals must pass the registry gate.
CI_PRODUCTION_MODULES = ("src/s2_firewall.py", "src/s2_reporting.py",
                         "src/s2_site_aggregation.py", "scripts/run_rq2_s2.py")
#: The single public gate through which every reported interval must pass.
CI_GATE = "compute_registered_interval"
#: Low-level primitives that a production path may not call directly.
CI_RAW_PRIMITIVES = ("bootstrap_summary", "paired_contrast",
                     "replicate_statistics", "percentile_ci")

RELEASE_FILENAME = "global_release.json"
#: Every frozen decision artifact lives here and nowhere else.
DECISION_STORE = "decisions"


def sha256_file(path) -> str:
    return sd.sha256_file(path)


def cell_key(site: str, repeat: int, n_labels: int, strategy: str,
             method: str) -> tuple:
    return (str(site), int(repeat), int(n_labels), str(strategy), str(method))


def _checksum_of(obj) -> str:
    return sd.sha256_bytes(sd.canonical_json(obj))


# ---------------------------------------------------------------------------
# the global release artifact — integrity and authority, kept apart
# ---------------------------------------------------------------------------
def release_payload_checksum(payload: dict) -> str:
    """Checksum over everything the release binds, excluding itself."""
    body = {k: v for k, v in payload.items() if k != "release_checksum"}
    return _checksum_of(body)


RELEASE_REQUIRED_FIELDS = (
    "protocol_version", "protocol_doc_checksum", "cohort_checksum",
    "roster_checksum", "site_roster_checksum", "outcome_source",
    "split_registry_checksum", "split_registry_canonical_checksum",
    "expected_cell_registry_checksum", "expected_count",
    "ci_registry_checksum", "ci_registry_count",
    "decision_manifest_checksum", "n_expected", "n_completed",
    "release_sequence", "release_checksum",
)


def verify_release_integrity(path, expected_checksum: str = None) -> dict:
    """**Integrity only**: the file is well formed and has not changed.

    It proves nothing about authority. A hand-written release with a
    recomputed self-checksum passes here and must still fail
    `verify_release_authority`. The two are separate functions with separate
    names precisely so a reader cannot mistake one for the other.
    """
    p = Path(path)
    if not p.exists():
        raise AssertionError(f"global release artifact {p} is not on disk")
    payload = json.loads(p.read_bytes())
    missing = [f for f in RELEASE_REQUIRED_FIELDS if f not in payload]
    if missing:
        raise AssertionError(f"release artifact is missing {missing}")
    if release_payload_checksum(payload) != payload["release_checksum"]:
        raise AssertionError("release artifact checksum does not match its body")
    if expected_checksum is not None and expected_checksum != payload["release_checksum"]:
        raise AssertionError("release checksum does not match the one presented")
    if payload["protocol_version"] != sd.PROTOCOL_VERSION:
        raise AssertionError("release artifact was issued under another protocol")
    return payload


def scan_current_decision_store(context) -> dict:
    """Re-scan the decision store and re-validate **every** artifact on disk.

    Returns the *actual* decision-cell set. Nothing is taken from a manifest, a
    counter or a recorded field: the set is what the filesystem currently holds,
    and each artifact is re-validated (schema, tier, state, threshold
    consistency, protocol version, cell key, split checksum, artifact checksum)
    on the way.
    """
    from s2_production import ProductionAuthorityError
    store = Path(context.workdir) / DECISION_STORE
    paths = sorted(store.glob("decision__*.json")) if store.exists() else []
    manifest, cells, records = {}, set(), {}
    for p in paths:
        frozen = sd.load_and_validate(p, zero_label=False)
        payload = frozen["payload"]
        key = cell_key(payload["site"], payload["repeat"], payload["n_labels"],
                       payload["strategy"], payload["method"])
        if key in cells:
            raise ProductionAuthorityError(f"duplicate decision artifact for {key}")
        registered = context.registry.cell(key[0], key[1], key[2])
        if payload["split_checksum"] != registered["cell_checksum"]:
            raise ProductionAuthorityError(
                f"decision {key} was taken on a split the registry does not hold")
        cells.add(key)
        records[key] = payload
        manifest["|".join(map(str, key))] = frozen["sha256"]
    # `records` carries the re-validated payloads so the §24.3 candidate ledger,
    # the §24.10 shrinkage diagnostics and the NOT_TESTABLE reason split are
    # derived from the artifacts this scan just verified rather than from a
    # second, unvalidated read. It takes no part in the manifest checksum.
    return {"n_completed": len(cells), "cells": frozenset(cells),
            "records": records,
            "decision_manifest_checksum": _checksum_of(manifest),
            "decision_manifest": manifest}


def assert_decision_store_is_complete(context) -> dict:
    """The actual decision set must **be** the authoritative expected set.

    Not `n_expected == 46200` on a field, and not a count: exact set equality
    between what the frozen contract expects and what the store actually holds.
    One missing cell, one extra, or an empty store each fail.
    """
    from s2_production import (PRODUCTION_EXPECTED_DECISIONS,
                               ProductionAuthorityError,
                               derive_authoritative_expected_decision_cells)
    expected = derive_authoritative_expected_decision_cells(context.registry)
    store = scan_current_decision_store(context)
    actual = store["cells"]
    if len(expected) != PRODUCTION_EXPECTED_DECISIONS:
        raise ProductionAuthorityError(
            f"expected {len(expected)} decisions, not {PRODUCTION_EXPECTED_DECISIONS}")
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ProductionAuthorityError(
            f"the decision store is not the authoritative expected set: "
            f"{len(missing)} missing, {len(extra)} unexpected "
            f"(store holds {len(actual)} of {len(expected)})")
    if len(actual) != PRODUCTION_EXPECTED_DECISIONS:
        raise ProductionAuthorityError(
            f"the decision store holds {len(actual)} cells, not "
            f"{PRODUCTION_EXPECTED_DECISIONS}")
    store["expected"] = expected
    return store


def verify_release_authority(path, context) -> dict:
    """**Authority**: re-derive the current state and require the release to match.

    Nothing the artifact asserts is believed. The context re-derives every
    authority field from the frozen sources; the decision store is re-scanned,
    every artifact re-validated, and its cell set required to equal the
    authoritative expected set exactly. Only then is the artifact compared,
    field by field.

    A hand-written release carrying `n_expected = 46200`, `n_completed = 0`, an
    empty store, a matching empty manifest and a valid self-checksum therefore
    fails: the store is not the expected set.
    """
    from s2_production import ProductionAuthorityError, ProductionS2Context
    if not isinstance(context, ProductionS2Context):
        raise ProductionAuthorityError(
            "release authority requires a ProductionS2Context; and the type is "
            "provenance, not proof — the fields below are re-derived")
    state = context.authority_state()          # re-derives everything
    store = assert_decision_store_is_complete(context)
    payload = verify_release_integrity(path)

    state = dict(state)
    state["decision_manifest_checksum"] = store["decision_manifest_checksum"]
    state["n_expected"] = context.expected_count
    state["n_completed"] = store["n_completed"]
    for field, current in state.items():
        if payload.get(field) != current:
            raise ProductionAuthorityError(
                f"release artifact {field}={payload.get(field)!r} does not match "
                f"the current authoritative value {current!r}")
    return payload


def verify_release_binding(path, expected_checksum: str, registry_checksum: str,
                           decision_checksum: str = None) -> dict:
    """Integrity plus binding to one split manifest. Still **not** authority."""
    payload = verify_release_integrity(path, expected_checksum)
    if payload["split_registry_checksum"] != registry_checksum:
        raise AssertionError(
            "the release was issued against a different split manifest")
    return payload


# ---------------------------------------------------------------------------
# phase-2 capability — evaluation cannot start from an artifact alone
# ---------------------------------------------------------------------------
PHASE_INIT = "INIT"
PHASE_1 = "PHASE1"
PHASE_1_COMPLETE = "PHASE1_COMPLETE"
PHASE_2_OPEN = "PHASE2_OPEN"

_CAPABILITY_TOKEN = object()


class Phase2Capability:
    """The dispatch token that opening phase 2 produces (Stage 2.0.5).

    **Demoted from a trust boundary to provenance and dispatch binding.** Stage
    2.0.4 claimed an in-process secret made a permit unforgeable by an arbitrary
    Python caller; that is not a property this repository can have, and it is no
    longer claimed. What actually prevents a direct worker call from touching
    target outcomes is architectural: the evaluation worker is a pure function
    with **no outcome-reading capability at all**, and the parent's
    `s2_outcomes.load_authoritative_target_outcomes_for_phase2()` is the only
    production path that materializes labels.

    This object still binds a run, an authority digest and a cell to each
    dispatch, which is useful provenance in the ledger.
    """

    __slots__ = ("_run_id", "_authority_digest", "_secret", "_registry_checksum",
                 "_protocol_checksum")

    def __init__(self, token, run_id, authority_digest, secret,
                 registry_checksum, protocol_checksum):
        if token is not _CAPABILITY_TOKEN:
            raise PermissionError(
                "a Phase2Capability is issued by open_phase2() alone")
        object.__setattr__(self, "_run_id", run_id)
        object.__setattr__(self, "_authority_digest", authority_digest)
        object.__setattr__(self, "_secret", secret)
        object.__setattr__(self, "_registry_checksum", registry_checksum)
        object.__setattr__(self, "_protocol_checksum", protocol_checksum)

    def __setattr__(self, name, value):
        raise PermissionError("a Phase2Capability is immutable")

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def authority_digest(self) -> str:
        return self._authority_digest

    def _mac(self, body: dict) -> str:
        return hmac.new(self._secret, sd.canonical_json(body),
                        hashlib.sha256).hexdigest()

    def issue_permit(self, key: tuple, decision_checksum: str,
                     split_checksum: str) -> dict:
        """One permit per evaluation cell, bound to that cell's frozen decision."""
        body = {"run_id": self._run_id, "cell": list(map(str, key)),
                "decision_checksum": decision_checksum,
                "split_checksum": split_checksum,
                "authority_digest": self._authority_digest,
                "protocol_checksum": self._protocol_checksum,
                "registry_checksum": self._registry_checksum}
        return {**body, "mac": self._mac(body)}

    def verifier_bundle(self) -> tuple:
        """What a worker process needs to *verify* a permit, and nothing else.

        Passed to a pool initializer over the process pipe — never through a
        file, an environment variable or a command line, so a standalone caller
        cannot obtain it.
        """
        return (self._run_id, self._secret, self._authority_digest)


def verify_permit(permit: dict, run_id: str, secret: bytes,
                  authority_digest: str, key: tuple,
                  decision_checksum: str) -> None:
    """Provenance check on a dispatch permit (Stage 2.0.5: **not** a trust boundary).

    Retained so a dispatch can be traced to the run that issued it. It is no
    longer claimed to stop an arbitrary caller, and nothing depends on it: the
    evaluation worker cannot read a label under any permit.
    """
    if not isinstance(permit, dict) or "mac" not in permit:
        raise PermissionError("no Phase2CellPermit was supplied")
    body = {k: v for k, v in permit.items() if k != "mac"}
    expected = hmac.new(secret, sd.canonical_json(body), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, str(permit["mac"])):
        raise PermissionError("the phase-2 permit is not authentic")
    if permit.get("run_id") != run_id:
        raise PermissionError("the permit belongs to another run")
    if permit.get("authority_digest") != authority_digest:
        raise PermissionError("the permit was issued under another authority")
    if tuple(permit.get("cell", ())) != tuple(map(str, key)):
        raise PermissionError("the permit names another cell")
    if permit.get("decision_checksum") != decision_checksum:
        raise PermissionError("the permit names another decision")


def _phase2_pool_call(inputs) -> dict:
    import s2_eval_worker
    return s2_eval_worker._pool_entry(inputs)


def _jsonable_inputs(inputs: dict) -> dict:
    """Arrays -> lists, so the pure worker's inputs pickle cleanly."""
    import numpy as np
    out = {}
    for key, value in inputs.items():
        out[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return out


# ---------------------------------------------------------------------------
# the orchestrator
# ---------------------------------------------------------------------------
class _S2FirewallBase:
    """Phase-ordered, cell-isolated execution mechanics.

    Deliberately not the production entry point: it takes a registry and a
    worker script, so a test fixture can drive the same mechanics at fixture
    scale. **The production runner imports `ProductionS2Firewall` only**, which
    accepts nothing but a `ProductionS2Context` and completes phase 1 at the
    frozen 46,200-cell scale.
    """

    def __init__(self, workdir, split_registry, ledger, python=None) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / DECISION_STORE).mkdir(exist_ok=True)
        (self.workdir / evalstore.EVALUATION_STORE).mkdir(exist_ok=True)
        self.registry = split_registry
        self._python = python or sys.executable
        #: Derived mechanically. Not an argument, not extensible, not trimmable.
        self.expected_cells = self.registry.expected_decision_cells()
        self._zero_label: dict[tuple, dict] = {}
        self._decisions: dict[tuple, dict] = {}
        self._evaluations: dict[tuple, dict] = {}
        self._release = None
        self._phase = PHASE_INIT
        self._capability = None
        #: The run's single sequence allocator. The firewall keeps **no**
        #: counter of its own and creates **no** ledger of its own: an ordering
        #: number it records must be comparable with the outcome loader's and
        #: the report's, which is only true if one allocator issued both
        #: (§17.21). A ledger minted here on a missing argument would be exactly
        #: the second allocator §17.25 forbids, so the argument is required.
        if not isinstance(ledger, ledgermod.GlobalRunLedger):
            raise ledgermod.LedgerError(
                "the firewall records into the run's GlobalRunLedger, which is "
                "supplied by the parent; a private counter — or a ledger minted "
                "here — cannot support a global ordering claim")
        self.ledger = ledger
        self.evaluation_store = evalstore.EvaluationArtifactStore(
            self.workdir / evalstore.EVALUATION_STORE)

    # -- the worker script is fixed by the class, never by a caller ---------
    def _worker_script(self) -> Path:
        return WORKER

    @property
    def phase(self) -> str:
        return self._phase

    # -- ledger -------------------------------------------------------------
    def _stamp(self, event: str, **detail) -> dict:
        """Record one event **through the run's single allocator**."""
        return self.ledger.append(event, **detail)

    # -- phase 0: zero-label decisions --------------------------------------
    def record_zero_label_decision(self, payload: dict) -> dict:
        """Freeze a decision taken with **no** target label at all.

        Takes the artifact *contents*, never a path: the firewall validates the
        object, serializes it canonically, writes those bytes and checksums
        them.
        """
        strategy = payload.get("strategy")
        if strategy not in cert.ZERO_LABEL_STRATEGIES:
            raise sd.DecisionArtifactError(
                f"{strategy!r} consumes target labels and may not be decided in "
                "the zero-label phase")
        key = (str(payload.get("site")), str(payload.get("method")), strategy)
        if key in self._zero_label:
            raise RuntimeError(f"zero-label decision {key} already frozen")
        path = self.workdir / f"zero__{key[0]}__{key[1]}__{key[2]}.json"
        frozen = sd.freeze_decision_artifact(payload, path, zero_label=True)
        self._zero_label[key] = frozen
        self._stamp("zero_label_decision_frozen", site=key[0], method=key[1],
                    strategy=key[2], sha256=frozen["sha256"],
                    state=frozen["payload"]["state"])
        return frozen

    # -- phase 1: one isolated decision worker per cell ---------------------
    def run_decision_cell(self, site: str, repeat: int, n_labels: int,
                          strategy: str, method: str, *, q_values, taus,
                          source_class_counts, r_star: float,
                          n0: float = 0.0) -> dict:
        """Launch an isolated worker for exactly one registered cell."""
        if self._release is not None:
            raise RuntimeError(
                "phase 1 is closed; a decision may not be taken after the "
                "global release")
        key = cell_key(site, repeat, n_labels, strategy, method)
        if key not in self.expected_cells:
            raise RuntimeError(
                f"cell {key} is not in the expected-cell registry derived from "
                "the split manifest; an unregistered cell may not be decided")
        if key in self._decisions:
            raise RuntimeError(f"cell {key} already decided")
        cell = self.registry.cell(site, repeat, n_labels)
        if self._phase != PHASE_1:
            self._stamp(ledgermod.EV_PHASE1_STARTED)
        self._phase = PHASE_1

        job = {
            "mode": "decision", "protocol_version": sd.PROTOCOL_VERSION,
            "site": str(site), "repeat": int(repeat), "n_labels": int(n_labels),
            "strategy": str(strategy), "method": str(method),
            "split_registry_path": self.registry.path,
            "split_registry_checksum": self.registry.checksum,
            "cell_checksum": cell["cell_checksum"],
            "q_values": [float(q) for q in q_values],
            "taus": [float(t) for t in taus],
            "source_class_counts": [[int(v) for v in row]
                                    for row in source_class_counts],
            "r_star": float(r_star), "n0": float(n0),
            "run_id": uuid.uuid4().hex,
        }
        out_path = (self.workdir / DECISION_STORE
                    / ("decision__" + "__".join(str(k) for k in key) + ".json"))
        payload = self._launch(job, key, "decision", out_path)
        sd.validate_decision_artifact(payload, zero_label=False)
        if payload["split_checksum"] != cell["cell_checksum"]:
            raise AssertionError(f"cell {key}: worker used a different split")
        digest = sha256_file(out_path)
        self._decisions[key] = {"path": str(out_path), "sha256": digest,
                                "payload": payload,
                                "split_checksum": cell["cell_checksum"]}
        self._stamp("cell_decision_frozen", cell=key, sha256=digest,
                    worker_pid=payload["worker_pid"],
                    cal_rows=payload["cal_label_rows_read"],
                    test_rows_during_decision=payload[
                        "test_label_rows_read_during_decision"])
        return payload

    # -- the global barrier -------------------------------------------------
    def _complete_phase1(self, require_expected_count: int) -> dict:
        """Exact-set completion, then the release artifact.

        `require_expected_count` is an **explicit integer**, never `None` and
        never optional: the production wrapper passes the frozen 46,200 and a
        tests-only helper passes the fixture's own size. There is no path on
        which the required count is inferred from what happened to be decided.
        """
        if self._release is not None:
            raise RuntimeError("phase 1 has already been released")
        if not isinstance(require_expected_count, int) or require_expected_count < 1:
            raise RuntimeError("the required expected-cell count must be a positive integer")
        expected = set(self.expected_cells)
        if len(expected) != require_expected_count:
            raise RuntimeError(
                f"this registry implies {len(expected)} decision cells, but "
                f"{require_expected_count} are required")
        completed = set(self._decisions)
        missing = sorted(expected - completed)
        extra = sorted(completed - expected)
        if missing or extra:
            raise RuntimeError(
                f"phase 1 cannot close: {len(missing)} missing and {len(extra)} "
                f"unexpected cells (e.g. missing {missing[:2]}, extra {extra[:2]})")

        for key, rec in self._decisions.items():
            frozen = sd.load_and_validate(rec["path"], rec["sha256"],
                                          zero_label=False)
            if frozen["payload"]["split_checksum"] != rec["split_checksum"]:
                raise RuntimeError(f"cell {key}: split checksum changed")
            registered = self.registry.cell(key[0], key[1], key[2])
            if registered["cell_checksum"] != rec["split_checksum"]:
                raise RuntimeError(
                    f"cell {key}: decision split does not match the registry")

        frozen_event = self._stamp(ledgermod.EV_ALL_DECISIONS_FROZEN,
                                   n_cells=len(completed))
        payload = self._release_payload(expected, frozen_event["seq"])
        path = self.workdir / RELEASE_FILENAME
        path.write_bytes(sd.canonical_json(payload))
        self._release = {"path": str(path),
                         "checksum": payload["release_checksum"],
                         "payload": payload}
        self.ledger.bind(release_checksum=payload["release_checksum"])
        return self._release

    def _release_payload(self, expected, release_sequence: int) -> dict:
        raise NotImplementedError

    @property
    def released(self) -> bool:
        return self._release is not None

    # -- phase 2: capability-gated, pool-isolated evaluation ----------------
    def _open_phase2(self, authority_digest: str) -> "Phase2Capability":
        """Mint the capability. Callers reach this through `open_phase2()`."""
        if self._release is None:
            raise RuntimeError("phase 2 cannot open before the global release")
        cap = Phase2Capability(_CAPABILITY_TOKEN, self.ledger.run_id,
                               authority_digest, secrets.token_bytes(32),
                               self.registry.checksum,
                               self._release["payload"]["protocol_doc_checksum"])
        self._capability = cap
        self._stamp(ledgermod.EV_RELEASE_AUTHORITY_VERIFIED,
                    authority_digest=authority_digest[:16])
        self._phase = PHASE_2_OPEN
        self._stamp(ledgermod.EV_PHASE2_OPEN,
                    release_checksum=self._release["checksum"][:16])
        return cap

    def run_evaluation_cell(self, capability, site: str, repeat: int,
                            n_labels: int, strategy: str, method: str, *, taus,
                            r_star: float, outcomes, oracle: bool = False
                            ) -> dict:
        """Evaluate one cell on outcomes **the parent already materialized**.

        Stage 2.0.5: the worker receives arrays, never a source. The parent
        obtains `outcomes` from
        `s2_outcomes.load_authoritative_target_outcomes_for_phase2()`, which
        cannot run before the global barrier — so the barrier, not a token,
        is what orders label access.
        """
        if self._phase != PHASE_2_OPEN or self._capability is None:
            raise RuntimeError(
                "phase 2 is not open; only open_phase2() may open it, and only "
                "after a full release authority verification")
        if capability is not self._capability:
            raise PermissionError(
                "the capability presented is not this run's phase-2 capability")
        key = cell_key(site, repeat, n_labels, strategy, method)
        rec = self._decisions.get(key) or self._zero_label_record(key)
        if rec is None:
            raise RuntimeError(
                f"cell {key} has no frozen decision artifact; evaluation is "
                "blocked")
        frozen = sd.load_and_validate(rec["path"], rec["sha256"])
        decision = frozen["payload"]
        cell = self.registry.cell(site, repeat, n_labels)
        if decision.get("split_checksum") not in (None, cell["cell_checksum"]):
            raise AssertionError(f"cell {key}: decision split does not match")

        partition = outcomes.for_cell(cell["site"], cell["test_ids"], method)
        inputs = {
            "site": cell["site"], "repeat": cell["repeat"],
            "n_labels": cell["n_labels"], "strategy": strategy,
            "method": method, "state": decision["state"],
            "selected_q": decision["selected_q"],
            "selected_tau": decision["selected_tau"],
            "r_star": float(r_star), "taus": [float(t) for t in taus],
            "oracle": bool(oracle),
            "decision_checksum": frozen["sha256"],
            "split_checksum": cell["cell_checksum"],
            **partition,
        }
        payload = self._run_in_phase2_pool(inputs)
        # the decision is read-only to evaluation: re-verify it afterwards
        sd.load_and_validate(rec["path"], rec["sha256"])
        frozen = self.evaluation_store.freeze(
            self._evaluation_artifact(key, cell, decision, frozen_sha=rec["sha256"],
                                      payload=payload))
        self._evaluations[key] = {"path": frozen["path"],
                                  "sha256": frozen["sha256"],
                                  "payload": frozen["payload"]}
        self._stamp("cell_evaluated", cell=key, worker_pid=payload["worker_pid"],
                    n_test=payload["n_test"],
                    evaluation_checksum=frozen["sha256"])
        return frozen["payload"]

    def _evaluation_artifact(self, key, cell, decision, frozen_sha, payload
                             ) -> dict:
        """Shape one frozen evaluation artifact. Counts, rates and states only.

        The status is **derived** from the evaluated cell rather than accepted
        as an input, and the decision fields are copied from the frozen decision
        artifact — never from the worker — so an artifact cannot claim a state
        or a threshold the decision store does not hold.
        """
        artifact = {
            "protocol_version": sd.PROTOCOL_VERSION,
            "run_id": self.ledger.run_id,
            "site": cell["site"], "repeat": cell["repeat"],
            "n_labels": cell["n_labels"],
            "strategy": key[3], "confidence_method": key[4],
            "guarantee_tier": cert.tier_of(key[3]),
            "decision_checksum": frozen_sha,
            "split_checksum": cell["cell_checksum"],
            "decision_state": decision["state"],
            "selected_q": decision["selected_q"],
            "selected_tau": decision["selected_tau"],
            "coverage_test": payload.get("coverage_test"),
            "n_test": payload["n_test"],
            "n_keep_test": payload.get("n_keep_test"),
            "n_keep_test_MDD": payload.get("n_keep_test_MDD"),
            "n_keep_test_HC": payload.get("n_keep_test_HC"),
            "ber_test": payload.get("ber_test"),
            "risk_excess": payload.get("risk_excess"),
            "realized_field": cert.realized_outcome_field(key[3]),
            "realized_outcome": payload.get("realized_outcome"),
            "r_star": payload["r_star"],
            "jsd_keep_vs_all": payload.get("jsd_keep_vs_all"),
            "eval_worker_pid": payload.get("worker_pid"),
        }
        if "oracle_infeasible" in payload:
            artifact["oracle_infeasible"] = payload["oracle_infeasible"]
            artifact["n_oracle_feasible_candidates"] = payload[
                "n_oracle_feasible_candidates"]
        artifact["evaluation_status"] = evalstore.evaluation_status_of(
            {"selected_tau": decision["selected_tau"],
             "ber_test": payload.get("ber_test")})
        return artifact

    def _run_in_phase2_pool(self, inputs: dict) -> dict:
        """One evaluation, in a worker process. Parallelism, not a barrier.

        The process boundary is retained for isolation of the *computation*;
        it is no longer load-bearing for label access, because the worker has
        no way to obtain a label it was not handed.
        """
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=1) as pool:
            return pool.apply(_phase2_pool_call, (_jsonable_inputs(inputs),))

    def _zero_label_record(self, key):
        site, _repeat, _n, strategy, method = key
        if strategy not in cert.ZERO_LABEL_STRATEGIES:
            return None
        return self._zero_label.get((site, method, strategy))

    # -- worker launch ------------------------------------------------------
    def _launch(self, job: dict, key: tuple, mode: str, out_path: Path):
        import os
        stem = "__".join(str(k) for k in key) + f"__{mode}"
        job_path = self.workdir / f"job_{stem}.json"
        job_path.write_bytes(sd.canonical_json(job))
        env = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin",
               "HOME": str(Path.home())}
        proc = subprocess.run(
            [self._python, str(self._worker_script()), str(job_path),
             str(out_path)],
            capture_output=True, text=True, env=env, cwd=str(ROOT))
        if proc.returncode != 0:
            raise RuntimeError(
                f"cell {key} {mode} worker failed:\n{proc.stderr.strip()[-2000:]}")
        payload = json.loads(Path(out_path).read_bytes())
        from s2_cell_worker import assert_payload_clean
        assert_payload_clean(payload, mode)
        if payload["worker_pid"] == os.getpid():
            raise AssertionError(
                f"cell {key} ran inside the parent process; the isolation unit "
                "is a process boundary, not a function call")
        return payload

    # -- ledgers ------------------------------------------------------------
    def cell_ledger(self) -> pd.DataFrame:
        rows = []
        rel_seq = (self._seq_of(ledgermod.EV_ALL_DECISIONS_FROZEN, None)
                   if self._release else None)
        for key, rec in self._decisions.items():
            site, repeat, n_labels, strategy, method = key
            d = rec["payload"]
            e = self._evaluations.get(key, {}).get("payload")
            dec_seq = self._seq_of("cell_decision_frozen", key)
            eval_seq = self._seq_of("cell_evaluated", key)
            rows.append({
                "site": site, "repeat": repeat, "n_labels": n_labels,
                "strategy": strategy, "method": method,
                "split_checksum": d["split_checksum"],
                "registry_checksum": d["registry_checksum"],
                "cal_id_set_checksum": d["cal_id_set_checksum"],
                "outcome_source_authoritative": d["outcome_source_authoritative"],
                "decision_worker_run_id": d["worker_run_id"],
                "decision_worker_pid": d["worker_pid"],
                "cal_label_rows_read": d["cal_label_rows_read"],
                "test_label_rows_read_during_decision":
                    d["test_label_rows_read_during_decision"],
                "decision_checksum": rec["sha256"],
                "decision_frozen_seq": dec_seq,
                "global_release_seq": rel_seq,
                "eval_worker_pid": (e or {}).get("worker_pid"),
                "n_test_evaluated": (e or {}).get("n_test"),
                "eval_decision_checksum": (e or {}).get("decision_checksum"),
                "test_read_after_release": bool(
                    eval_seq is not None and rel_seq is not None
                    and eval_seq > rel_seq),
                "evaluation_seq": eval_seq,
                "ordering_ok": bool(
                    d["test_label_rows_read_during_decision"] == 0
                    and dec_seq is not None and rel_seq is not None
                    and dec_seq < rel_seq
                    and (eval_seq is None or eval_seq > rel_seq)
                    and (e is None or e.get("decision_checksum") == rec["sha256"])
                    and (e is None or e.get("worker_pid") != d["worker_pid"])),
            })
        return pd.DataFrame(rows)

    def global_ledger(self) -> dict:
        rel = self._release["payload"] if self._release else {}
        seqs = self.ledger.barrier_seqs()
        rel_seq = seqs["all_decisions_frozen_seq"]
        return {
            "ledger_id": self.ledger.ledger_id, "run_id": self.ledger.run_id,
            "split_manifest_checksum": self.registry.checksum,
            "n_expected_decision_cells": len(self.expected_cells),
            "n_completed_decision_cells": len(self._decisions),
            "n_zero_label_decisions": len(self._zero_label),
            "n_evaluation_cells": len(self._evaluations),
            "release_sequence": rel.get("release_sequence"),
            "release_checksum": rel.get("release_checksum"),
            "phase2_authority_seq": seqs["phase2_authority_verified_seq"],
            "target_outcomes_materialized_pre_phase2": False,
            "barrier_ok": bool(
                self._release is not None
                and set(self._decisions) == set(self.expected_cells)
                and all(self._seq_of("cell_decision_frozen", k) < rel_seq
                        for k in self._decisions)
                and all(self._seq_of("cell_evaluated", k) > rel_seq
                        for k in self._evaluations)),
            "distinct_decision_worker_pids": len(
                {r["payload"]["worker_pid"] for r in self._decisions.values()}),
            **seqs,
        }

    def _seq_of(self, event: str, key):
        """Read a sequence number **from the run's ledger**. Allocates nothing."""
        return self.ledger.seq_of(event, cell=key)

    def events(self) -> pd.DataFrame:
        return self.ledger.frame()


class ProductionS2Firewall(_S2FirewallBase):
    """The only firewall a production run may use (Stage 2.0.3).

    Takes a `ProductionS2Context` and nothing else — no registry argument, no
    worker override, no scale flag. `complete_production_phase1()` is inherently
    a production completion: it requires the frozen 46,200 expected cells, the
    exact decomposition, a re-scan of the decision store and a re-derivation of
    every authority field before it will issue a release.
    """

    def __init__(self, context, ledger, python=None) -> None:
        from s2_production import ProductionAuthorityError, ProductionS2Context
        if not isinstance(context, ProductionS2Context):
            raise ProductionAuthorityError(
                "the production firewall accepts only a ProductionS2Context "
                "built by the authoritative constructor")
        from s2_production import (assert_expected_decomposition,
                                   assert_registry_matches_authority,
                                   derive_authoritative_expected_decision_cells)
        context.assert_still_authoritative()          # re-derives everything
        assert_registry_matches_authority(context.registry)
        expected = derive_authoritative_expected_decision_cells(context.registry)
        assert_expected_decomposition(expected)
        # No production component may start recording into a merely
        # same-run-looking allocator. Re-read the context-bound binding before
        # any decision/evaluation directory is touched.
        ledgermod.assert_ledger_binding_for_context(ledger, context)
        self.context = context
        super().__init__(context.workdir, context.registry, ledger, python)
        #: Writes go to the **authoritative** evaluation root, re-derived from
        #: the context — the same root the aggregation will later scan, and not
        #: a directory anyone passed in.
        self.evaluation_store = \
            evalstore.resolve_authoritative_evaluation_store(context)
        self.evaluation_store.root.mkdir(parents=True, exist_ok=True)
        if self.ledger.run_id != context.run_id:
            raise ProductionAuthorityError(
                "the production firewall and its run ledger belong to different "
                "runs; a global ordering claim is made within one run")
        if set(self.expected_cells) != set(expected) != set(context.expected_cells):
            raise ProductionAuthorityError(
                "the registry's expected cells disagree with the authoritative "
                "derivation")

    def complete_production_phase1(self) -> dict:
        """Authoritative completion. Takes no arguments and no options."""
        from s2_production import (PRODUCTION_EXPECTED_DECISIONS,
                                   assert_expected_decomposition)
        self.context.assert_still_authoritative()
        assert_expected_decomposition(self.expected_cells)
        release = self._complete_phase1(PRODUCTION_EXPECTED_DECISIONS)
        # the release must survive an independent authority re-derivation
        verify_release_authority(release["path"], self.context)
        self._phase = PHASE_1_COMPLETE
        return release

    def _release_payload(self, expected, release_sequence: int) -> dict:
        """Every field is a **consequence** of the current state, never an input."""
        state = self.context.authority_state()
        store = assert_decision_store_is_complete(self.context)
        payload = dict(state)
        payload.update({
            "decision_manifest_checksum": store["decision_manifest_checksum"],
            "n_expected": self.context.expected_count,
            "n_completed": store["n_completed"],
            "n_zero_label_decisions": len(self._zero_label),
            "release_sequence": int(release_sequence),
        })
        if payload["n_expected"] != payload["n_completed"]:
            raise RuntimeError(
                "a production release requires a complete decision store")
        payload["release_checksum"] = release_payload_checksum(payload)
        return payload

    def open_phase2(self):
        """The **only** production gate that opens target-test evaluation.

        Runs a full release *authority* verification — the context re-derives
        every field from the frozen sources, the decision store is re-scanned
        and required to equal the authoritative expected set exactly — and only
        then mints the `Phase2Capability` that `run_evaluation_cell` demands.
        """
        if self._phase != PHASE_1_COMPLETE:
            raise RuntimeError(
                f"phase 2 opens from {PHASE_1_COMPLETE}, not {self._phase}")
        payload = verify_release_authority(self._release["path"], self.context)
        return self._open_phase2(payload["release_checksum"])

    def complete_production_phase2(self) -> dict:
        """Close phase 2: the evaluation store must **be** the expected set.

        Exact set equality against the authoritative expected evaluation cells,
        every artifact re-bound to the decision the store actually holds, then
        the evaluation manifest recomputed from the files on disk. Both events
        are recorded through the run's single allocator, so the aggregation that
        follows can be ordered against the barrier.
        """
        if self._phase != PHASE_2_OPEN:
            raise RuntimeError(
                f"phase 2 closes from {PHASE_2_OPEN}, not {self._phase}")
        self.context.assert_still_authoritative()
        scan = evalstore.assert_evaluation_store_is_complete(self.context)
        if str(self.evaluation_store.root) != scan["evaluation_root"]:
            raise RuntimeError(
                "phase 2 wrote to a different root than the authoritative one")
        self._stamp(ledgermod.EV_EVALUATION_COMPLETE,
                    n_evaluated=scan["n_evaluated"])
        manifest = evalstore.build_evaluation_manifest(self.context)
        self.ledger.bind(evaluation_manifest_checksum=manifest[
            "evaluation_manifest_checksum"])
        self._stamp(ledgermod.EV_EVALUATION_MANIFEST_FROZEN,
                    evaluation_manifest_checksum=manifest[
                        "evaluation_manifest_checksum"][:16])
        return manifest


# ---------------------------------------------------------------------------
# static audits — the second of the two enforcement layers
# ---------------------------------------------------------------------------
def audit_parent_modules_label_free(root=ROOT) -> dict:
    findings = []
    for rel in PARENT_MODULES:
        tree = ast.parse((Path(root) / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                for bad in PARENT_FORBIDDEN_CALLS:
                    if bad in name:
                        findings.append((rel, name))
    return {"ok": not findings, "findings": findings}


def audit_no_module_level_label_cache(root=ROOT) -> dict:
    findings = []
    for rel in PARENT_MODULES + (LABEL_READING_MODULE,):
        tree = ast.parse((Path(root) / rel).read_text())
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = ([node.target] if isinstance(node, ast.AnnAssign)
                       else node.targets)
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if not names or node.value is None:
                continue
            value = ast.unparse(node.value)
            looks_cachey = any("cache" in n.lower() or n.lower().endswith("_y")
                               for n in names)
            if looks_cachey and (value.startswith(("{", "[")) or value in ("{}", "[]")):
                findings.append((rel, names))
    return {"ok": not findings, "findings": findings}


def audit_decision_path_has_no_test_reader(root=ROOT) -> dict:
    tree = ast.parse((Path(root) / LABEL_READING_MODULE).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_decision_cell")
    body = "\n".join(ast.unparse(s) for s in fn.body)
    hits = [tok for tok in ("test_rows", "test_confidence", "test_pred",
                            "D_test", "y_test") if tok in body]
    if "_resolve_cell(job, 'cal')" not in body and '_resolve_cell(job, "cal")' not in body:
        hits.append("decision path does not bind to the cal partition")
    return {"ok": not hits, "findings": hits}


def audit_no_caller_label_source(root=ROOT) -> dict:
    """No production entry point may accept an outcome-source argument."""
    findings = []
    for rel in (LABEL_READING_MODULE, "src/s2_firewall.py"):
        tree = ast.parse((Path(root) / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = [a.arg for a in node.args.args + node.args.kwonlyargs]
                bad = [a for a in args
                       if a in ("label_source", "outcome_source", "labels_path",
                                "target_confidence_path")]
                if bad:
                    findings.append((rel, node.name, bad))
    return {"ok": not findings, "findings": findings}


def audit_ci_gate(root=ROOT) -> dict:
    """The real runner and report builder may not create an interval themselves.

    Stage 2.0.2 audited `s2_firewall.py`, which creates no interval at all, so
    the audit proved nothing. It now scans the **formal runner**, the report
    builder and the aggregation module, and requires that every interval in the
    production call graph is produced by `compute_registered_interval`.
    """
    findings = []
    for rel in CI_PRODUCTION_MODULES:
        path = Path(root) / rel
        if not path.exists():
            findings.append((rel, "production module missing"))
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if any(p in name for p in CI_RAW_PRIMITIVES):
                    findings.append((rel, name))
                if name in ("np.quantile", "numpy.quantile", "np.percentile"):
                    findings.append((rel, f"direct quantile interval: {name}"))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", "") or ""
                names = {a.name for a in node.names}
                if "site_bootstrap" in mod or "site_bootstrap" in names:
                    findings.append((rel, "imports the bootstrap primitive module"))
                if names & set(CI_RAW_PRIMITIVES):
                    findings.append((rel, f"imports {sorted(names & set(CI_RAW_PRIMITIVES))}"))

    agg_tree = ast.parse((Path(root) / "src" / "s2_aggregation.py").read_text())
    gate = next((n for n in ast.walk(agg_tree)
                 if isinstance(n, ast.FunctionDef) and n.name == CI_GATE), None)
    if gate is None:
        findings.append(("src/s2_aggregation.py", f"missing {CI_GATE}"))
    else:
        # inside aggregation, only the gate may touch a primitive
        gate_lines = set(range(gate.lineno, (gate.end_lineno or gate.lineno) + 1))
        for node in ast.walk(agg_tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if any(p in name for p in CI_RAW_PRIMITIVES) and \
                        node.lineno not in gate_lines:
                    findings.append(("src/s2_aggregation.py",
                                     f"{name} outside {CI_GATE}"))
    return {"ok": not findings, "findings": findings}


def audit_no_environment_outcome_override(root=ROOT) -> dict:
    """No production module may read the withdrawn test-mode variables."""
    findings = []
    # spelled indirectly so the audit does not flag its own token list
    tokens = tuple("ISBI_S2_TEST_" + suffix for suffix in ("MODE", "OUTCOME_SOURCE"))
    for rel in PARENT_MODULES + (LABEL_READING_MODULE, "src/s2_production.py",
                                 "src/s2_reporting.py", "scripts/run_rq2_s2.py"):
        path = Path(root) / rel
        if not path.exists():
            continue
        text = path.read_text()
        for tok in tokens:
            if tok in text:
                findings.append((rel, tok))
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and ast.unparse(node).startswith(
                    ("os.environ", "os.getenv")):
                findings.append((rel, ast.unparse(node)))
    return {"ok": not findings, "findings": findings}


def audit_production_runner_root_of_trust(root=ROOT) -> dict:
    """The formal runner must build the context and use the production firewall."""
    path = Path(root) / "scripts" / "run_rq2_s2.py"
    if not path.exists():
        return {"ok": False, "findings": ["scripts/run_rq2_s2.py is missing"]}
    text = path.read_text()
    findings = []
    for required in ("ProductionS2Context.build", "ProductionS2Firewall",
                     "complete_production_phase1", "open_phase2",
                     "compute_registered_interval"):
        if required not in text:
            findings.append(f"runner never calls {required}")
    for forbidden in ("_complete_phase1(", "SplitRegistry(", "_S2FirewallBase"):
        if forbidden in text:
            findings.append(f"runner reaches past the production API: {forbidden}")
    return {"ok": not findings, "findings": findings}


def audit_decision_writes_are_validated(root=ROOT) -> dict:
    """Every decision artifact write in production goes through the validator."""
    findings = []
    for rel in ("src/s2_firewall.py", LABEL_READING_MODULE):
        tree = ast.parse((Path(root) / rel).read_text())
        calls = {ast.unparse(n.func) for n in ast.walk(tree)
                 if isinstance(n, ast.Call)}
        writes_json = {c for c in calls if "write_bytes" in c or "write_text" in c}
        validated = any("freeze_decision_artifact" in c for c in calls)
        if writes_json and not validated:
            findings.append((rel, sorted(writes_json)))
    return {"ok": not findings, "findings": findings}


def audit_eval_worker_has_no_outcome_access(root=ROOT) -> dict:
    """The evaluation worker must have no way to obtain target outcomes.

    Stage 2.0.5's actual guarantee: no outcome-source resolver, no dataset
    path, no label reader, no phenotype loader, no file access. A direct import
    and call therefore cannot produce a protocol label access.
    """
    path = Path(root) / "src" / "s2_eval_worker.py"
    if not path.exists():
        return {"ok": False, "findings": ["src/s2_eval_worker.py is missing"]}
    text = path.read_text()
    tree = ast.parse(text)
    findings = []
    forbidden_calls = ("read_csv", "open", "ProductionCellLabelReader",
                       "resolve_authoritative_outcome_source",
                       "resolve_outcome_source", "load_frozen_cohort_roster",
                       "_materialize_outcome_table", "read_bytes", "read_text")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            for bad in forbidden_calls:
                if bad in name:
                    findings.append(("call", name))
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            names = {a.name for a in node.names}
            for bad in ("s2_outcomes", "s2_production", "s1_data_access",
                        "fc_features", "pathlib", "s2_cell_worker"):
                if bad == module or bad in names:
                    findings.append(("import", bad))
    for token in ("results/rq1_s0", "confidence_target.csv",
                  "analysis_manifest", "AUTHORITATIVE"):
        if token in text:
            findings.append(("path token", token))
    return {"ok": not findings, "findings": findings}


def audit_single_outcome_read_path(root=ROOT) -> dict:
    """Exactly one production module may materialize target outcomes."""
    findings = []
    for rel in PARENT_MODULES + (LABEL_READING_MODULE, "src/s2_eval_worker.py",
                                 "src/s2_reporting.py", "scripts/run_rq2_s2.py"):
        path = Path(root) / rel
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if "_materialize_outcome_table" in name:
                    findings.append((rel, name))
    outcomes = Path(root) / "src" / "s2_outcomes.py"
    if not outcomes.exists():
        findings.append(("src/s2_outcomes.py", "missing"))
    else:
        text = outcomes.read_text()
        for required in ("assert_outcome_read_allowed",
                         "load_authoritative_target_outcomes_for_phase2",
                         "PHASE_2_OPEN"):
            if required not in text:
                findings.append(("src/s2_outcomes.py", f"missing {required}"))
    return {"ok": not findings, "findings": findings}


def audit_report_accepts_no_external_ci(root=ROOT) -> dict:
    """The formal report may not ingest caller-supplied interval values."""
    path = Path(root) / "src" / "s2_reporting.py"
    if not path.exists():
        return {"ok": False, "findings": ["src/s2_reporting.py is missing"]}
    text = path.read_text()
    tree = ast.parse(text)
    findings = []
    for banned in ("set_interval", "add_interval", "load_interval_store",
                   "set_ci", "build_from_registered_store"):
        if any(isinstance(n, ast.FunctionDef) and n.name == banned
               for n in ast.walk(tree)):
            findings.append(("function", banned))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and n_is_public(node.name):
            args = [a.arg for a in node.args.args + node.args.kwonlyargs]
            for bad in ("intervals", "ci_values", "store", "interval_store",
                        "ci_low", "ci_high", "point_estimate"):
                if bad in args:
                    findings.append((node.name, bad))
    if "build_formal_s2_report" not in text:
        findings.append(("src/s2_reporting.py", "missing build_formal_s2_report"))
    return {"ok": not findings, "findings": findings}


def n_is_public(name: str) -> bool:
    return not name.startswith("_")


#: The one module permitted to allocate an ordering number.
SEQUENCE_ALLOCATOR_MODULE = "src/s2_ledger.py"
#: Every other S2 production module. A worker-local diagnostic counter is
#: allowed to exist, but nothing here may mint a number used as global order.
SEQUENCE_SCOPED_MODULES = (
    "src/s2_firewall.py", "src/s2_outcomes.py", "src/s2_evaluation_store.py",
    "src/s2_site_aggregation.py", "src/s2_reporting.py", "src/s2_production.py",
    "src/s2_aggregation.py", "scripts/run_rq2_s2.py",
)


def audit_single_global_sequence_allocator(root=ROOT) -> dict:
    """No S2 production module but the ledger may allocate a sequence number.

    Stage 2.0.5's ordering claim compared a firewall counter with an
    outcome-read counter — two allocators, two scales, one meaningless
    inequality. This audit makes the shape unrepresentable: an `x_seq += 1`, an
    `x_seq = 0`, a `seq = len(events) + 1` or an event dict literal minting its
    own `seq` outside `src/s2_ledger.py` is a finding.
    """
    findings = []
    for rel in SEQUENCE_SCOPED_MODULES:
        path = Path(root) / rel
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.AugAssign):
                target = ast.unparse(node.target)
                if "seq" in target.lower():
                    findings.append((rel, f"local sequence counter {target}"))
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, (ast.Name, ast.Attribute)):
                        continue
                    name = ast.unparse(target)
                    if "seq" not in name.lower():
                        continue
                    value = ast.unparse(node.value)
                    if value.strip().isdigit() or "len(" in value:
                        findings.append((rel, f"{name} = {value}"))
            if isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys
                        if isinstance(k, ast.Constant)]
                if "seq" in keys:
                    findings.append(
                        (rel, "an event literal mints its own 'seq'"))
    allocator = Path(root) / SEQUENCE_ALLOCATOR_MODULE
    if not allocator.exists():
        findings.append((SEQUENCE_ALLOCATOR_MODULE, "missing"))
    else:
        text = allocator.read_text()
        for required in ("class GlobalRunLedger", "def append",
                         "def assert_global_ordering", "_next_seq",
                         "def assert_event_stream_ordering"):
            if required not in text:
                findings.append((SEQUENCE_ALLOCATOR_MODULE, f"missing {required}"))
    return {"ok": not findings, "findings": findings}


#: Names through which a caller could once feed the report its own numbers.
FORBIDDEN_REPORT_INPUTS = (
    "site_level_results", "site_level", "site_values", "site_vectors",
    "site_means", "p_cert", "ber_values", "coverage_values", "records",
    "decision_manifest_checksum", "evaluation_manifest_checksum", "intervals",
    "ci_values", "store", "interval_store", "ci_low", "ci_high",
    "point_estimate", "dataframe", "frame",
)


def audit_report_takes_no_site_level_input(root=ROOT) -> dict:
    """The formal report may not accept caller-supplied site-level results.

    Stage 2.0.6's blocker: `SiteLevelResults` provenance was two strings the
    caller typed, so fabricated site vectors reached a formal report. The remedy
    is the absence of the parameter — checked here on the real signature, not
    asserted in prose.
    """
    findings = []
    reporting = Path(root) / "src" / "s2_reporting.py"
    if not reporting.exists():
        return {"ok": False, "findings": ["src/s2_reporting.py is missing"]}
    tree = ast.parse(reporting.read_text())
    builder = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == "build_formal_s2_report"), None)
    if builder is None:
        findings.append(("src/s2_reporting.py", "no build_formal_s2_report"))
    else:
        args = [a.arg for a in builder.args.args + builder.args.kwonlyargs]
        if args[:2] != ["context", "ledger"]:
            findings.append(("build_formal_s2_report",
                             f"signature is {args}, not (context, ledger, ...)"))
        for bad in FORBIDDEN_REPORT_INPUTS:
            if bad in args:
                findings.append(("build_formal_s2_report", bad))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and n_is_public(node.name):
            args = [a.arg for a in node.args.args + node.args.kwonlyargs]
            for bad in FORBIDDEN_REPORT_INPUTS:
                if bad in args:
                    findings.append((node.name, bad))

    aggregation = Path(root) / "src" / "s2_site_aggregation.py"
    if not aggregation.exists():
        findings.append(("src/s2_site_aggregation.py", "missing"))
    else:
        tree = ast.parse(aggregation.read_text())
        fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "aggregate_authoritative_site_level_results"),
                  None)
        if fn is None:
            findings.append(("src/s2_site_aggregation.py",
                             "no aggregate_authoritative_site_level_results"))
        else:
            args = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
            if args != ["context", "ledger"]:
                findings.append(("aggregate_authoritative_site_level_results",
                                 f"signature is {args}"))
    return {"ok": not findings, "findings": findings}


#: Parameter names through which a caller could once decide **where** evaluation
#: artifacts are read from. A production entry point that takes a `context` may
#: not also take one of these: the root is a derivation, not an argument.
FORBIDDEN_STORE_INPUTS = (
    "store", "evaluation_store", "eval_store", "evaluation_root", "eval_root",
    "evaluation_dir", "eval_dir", "root", "path", "store_path", "workdir",
    "work_dir", "result_dir", "results_dir", "output_dir", "config",
)

#: Production-reachable modules the store-input audit covers.
STORE_SCOPED_MODULES = (
    "scripts/run_rq2_s2.py", "src/s2_evaluation_store.py",
    "src/s2_site_aggregation.py", "src/s2_reporting.py",
)

#: Per-artifact primitives that legitimately take a single file path. They read
#: or write *one* artifact and decide nothing about which store is authoritative;
#: none of them takes a context.
STORE_PATH_PRIMITIVES = ("freeze_evaluation_artifact",
                         "load_and_validate_evaluation")


def audit_no_caller_supplied_evaluation_store(root=ROOT) -> dict:
    """No production entry point may be told where the evaluation store is.

    Stage 2.0.7's first blocker: `scan_current_evaluation_store(context, store)`,
    `assert_evaluation_store_is_complete(context, store)` and
    `aggregate_authoritative_site_level_results(context, evaluation_store, ...)`
    all took a caller-supplied store, so a complete fake store — 78,600
    structurally valid artifacts with correctly copied decision bindings and
    arbitrary metrics — could be aggregated into a formal report.

    The rule checked here is mechanical: **a function that takes `context` may
    not also take a store, a root or a path.** The authoritative root comes from
    `resolve_authoritative_evaluation_store(context)` and from nowhere else.
    """
    findings = []
    for rel in STORE_SCOPED_MODULES:
        path = Path(root) / rel
        if not path.exists():
            findings.append((rel, "missing"))
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not n_is_public(node.name):
                continue
            args = [a.arg for a in node.args.args + node.args.kwonlyargs]
            if "context" not in args:
                continue
            if node.name in STORE_PATH_PRIMITIVES:
                continue
            for bad in FORBIDDEN_STORE_INPUTS:
                if bad in args:
                    findings.append((f"{rel}:{node.name}", f"accepts {bad}"))

    evalmod = Path(root) / "src" / "s2_evaluation_store.py"
    if evalmod.exists():
        text = evalmod.read_text()
        if "def resolve_authoritative_evaluation_store" not in text:
            findings.append(("src/s2_evaluation_store.py",
                             "no resolve_authoritative_evaluation_store"))
        tree = ast.parse(text)
        for name in ("scan_current_evaluation_store",):
            fn = next((n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == name), None)
            if fn is None:
                findings.append(("src/s2_evaluation_store.py", f"missing {name}"))
            elif "resolve_authoritative_evaluation_store" not in ast.unparse(fn):
                findings.append((name, "does not resolve its own root"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and \
                    node.name == "resolve_authoritative_evaluation_store":
                args = [a.arg for a in node.args.args + node.args.kwonlyargs]
                if args != ["context"]:
                    findings.append(("resolve_authoritative_evaluation_store",
                                     f"signature is {args}, not (context,)"))

    # no environment or CLI route to a different store either
    for rel in STORE_SCOPED_MODULES:
        path = Path(root) / rel
        if not path.exists():
            continue
        text = path.read_text()
        for bad in ("--evaluation-root", "--eval-store", "--evaluation-dir",
                    "--results-dir", "S2_EVALUATION_ROOT", "S2_EVAL_STORE"):
            if bad in text:
                findings.append((rel, f"offers {bad}"))
    return {"ok": not findings, "findings": findings}


#: Production entry points whose `ledger` argument carries an ordering claim.
#: `(relative module, function or class.method)`.
MANDATORY_LEDGER_APIS = (
    ("src/s2_site_aggregation.py", "aggregate_authoritative_site_level_results"),
    ("src/s2_reporting.py", "build_formal_s2_report"),
    ("src/s2_outcomes.py", "assert_outcome_read_allowed"),
    ("src/s2_outcomes.py", "load_authoritative_target_outcomes_for_phase2"),
    ("src/s2_firewall.py", "ProductionS2Firewall.__init__"),
    ("src/s2_firewall.py", "_S2FirewallBase.__init__"),
)


def _function_named(tree, dotted: str):
    """Find `f` or `Class.method` in a parsed module."""
    if "." not in dotted:
        return next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                     and n.name == dotted), None)
    cls_name, meth = dotted.split(".", 1)
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                and n.name == cls_name), None)
    if cls is None:
        return None
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef)
                 and n.name == meth), None)


def audit_no_optional_production_ledger(root=ROOT) -> dict:
    """A production ordering API may not default, fall back or mint a ledger.

    Stage 2.0.7's second blocker: `aggregate_authoritative_site_level_results`
    had `ledger=None`, and `None` skipped `assert_global_ordering` entirely — a
    formal aggregation with no ordering proof at all. `ledger` therefore has no
    default on any of these entry points, and no S2 production module may
    construct a `GlobalRunLedger`: a ledger created inside a component is by
    definition a second allocator.
    """
    findings = []
    for rel, dotted in MANDATORY_LEDGER_APIS:
        path = Path(root) / rel
        if not path.exists():
            findings.append((rel, "missing"))
            continue
        fn = _function_named(ast.parse(path.read_text()), dotted)
        if fn is None:
            findings.append((rel, f"no {dotted}"))
            continue
        args = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        if "ledger" not in args:
            findings.append((dotted, f"takes no ledger: {args}"))
            continue
        positional = [a.arg for a in fn.args.args]
        n_defaults = len(fn.args.defaults)
        defaulted = set(positional[len(positional) - n_defaults:]) if n_defaults \
            else set()
        defaulted |= {a.arg for a, d in zip(fn.args.kwonlyargs,
                                            fn.args.kw_defaults) if d is not None}
        if "ledger" in defaulted:
            findings.append((dotted, "ledger has a default; `ledger=None` "
                                     "silently skips the ordering assertion"))

    for rel in set(m for m, _ in MANDATORY_LEDGER_APIS) | {"scripts/run_rq2_s2.py"}:
        path = Path(root) / rel
        if not path.exists() or rel == "scripts/run_rq2_s2.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name.endswith("GlobalRunLedger") or \
                        name.endswith("GlobalRunLedger.for_context"):
                    findings.append((rel, f"constructs a ledger: {name}"))
    return {"ok": not findings, "findings": findings}


def audit_evaluation_store_is_derived(root=ROOT) -> dict:
    """Expected evaluation cells and manifests are derived, never supplied."""
    path = Path(root) / "src" / "s2_evaluation_store.py"
    if not path.exists():
        return {"ok": False, "findings": ["src/s2_evaluation_store.py is missing"]}
    tree = ast.parse(path.read_text())
    findings = []
    required = ("derive_authoritative_expected_evaluation_cells",
                "scan_current_evaluation_store",
                "assert_evaluation_store_is_complete",
                "build_evaluation_manifest", "freeze_evaluation_artifact")
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in required:
        if name not in defined:
            findings.append(("src/s2_evaluation_store.py", f"missing {name}"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and n_is_public(node.name):
            args = [a.arg for a in node.args.args + node.args.kwonlyargs]
            for bad in ("expected", "expected_cells", "expected_evaluation_cells",
                        "evaluation_manifest_checksum", "manifest"):
                if bad in args and node.name not in (
                        "expected_evaluation_decomposition",
                        "assert_expected_evaluation_decomposition"):
                    findings.append((node.name, f"accepts {bad}"))
    return {"ok": not findings, "findings": findings}


def run_static_audits(root=ROOT) -> pd.DataFrame:
    checks = {
        "parent_modules_never_decode_labels": audit_parent_modules_label_free(root),
        "no_module_level_label_cache": audit_no_module_level_label_cache(root),
        "decision_path_has_no_test_reader":
            audit_decision_path_has_no_test_reader(root),
        "no_caller_supplied_label_source": audit_no_caller_label_source(root),
        "ci_intervals_pass_the_registry_gate": audit_ci_gate(root),
        "decision_writes_are_validated": audit_decision_writes_are_validated(root),
        "no_environment_outcome_override":
            audit_no_environment_outcome_override(root),
        "production_runner_root_of_trust":
            audit_production_runner_root_of_trust(root),
        "eval_worker_has_no_outcome_access":
            audit_eval_worker_has_no_outcome_access(root),
        "single_production_outcome_read_path":
            audit_single_outcome_read_path(root),
        "report_accepts_no_external_ci":
            audit_report_accepts_no_external_ci(root),
        "single_global_sequence_allocator":
            audit_single_global_sequence_allocator(root),
        "report_takes_no_site_level_input":
            audit_report_takes_no_site_level_input(root),
        "evaluation_store_is_derived": audit_evaluation_store_is_derived(root),
        "no_caller_supplied_evaluation_store":
            audit_no_caller_supplied_evaluation_store(root),
        "no_optional_production_ledger":
            audit_no_optional_production_ledger(root),
    }
    return pd.DataFrame([{"audit": k, "status": "PASS" if v["ok"] else "FAIL",
                          "detail": str(v["findings"])[:300]}
                         for k, v in checks.items()])
