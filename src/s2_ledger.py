"""The single global run ledger (RQ2/S2, Stage 2.0.8).

Stage 2.0.5 claimed a global ordering:

    all_decisions_frozen_seq < phase2_authority_verified_seq
                             < first_target_outcome_read_seq

but the three numbers came from **two independent counters** — the firewall's
`self._seq` and `s2_outcomes`'s `len(_OUTCOME_READ_EVENTS) + 1`. A firewall
counter of 10 and an outcome counter of 2 are not comparable, so the assertion
compared numbers that were never on the same scale. Whether it passed or failed
carried no information.

This module makes the ordering mechanical. **One parent-owned allocator issues
every sequence number in a run**, so `seq` is a total order over the whole run
by construction rather than by coincidence. Every event carries the allocator's
`ledger_id` and the run's `run_id`, and `assert_global_ordering` refuses an
event stream that mixes two of either — which is exactly the Stage 2.0.5
failure, now unrepresentable rather than merely discouraged.

    PHASE1_STARTED
      -> ALL_DECISIONS_FROZEN
      -> RELEASE_AUTHORITY_VERIFIED
      -> PHASE2_OPEN
      -> OUTCOME_READ_AUTHORIZED
      -> FIRST_TARGET_OUTCOME_READ        <- first row actually materialized
      -> EVALUATION_COMPLETE
      -> EVALUATION_MANIFEST_FROZEN
      -> AGGREGATION_STARTED
      -> REPORT_BUILT

Timestamps are recorded but are **never** the order authority: wall clock can
move backwards, and `perf_counter` is per-process. Only `seq` orders events.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import s2_decisions as sd

# ---------------------------------------------------------------------------
# the critical global events, in their one legal order
# ---------------------------------------------------------------------------
EV_PHASE1_STARTED = "PHASE1_STARTED"
EV_ALL_DECISIONS_FROZEN = "ALL_DECISIONS_FROZEN"
EV_RELEASE_AUTHORITY_VERIFIED = "RELEASE_AUTHORITY_VERIFIED"
EV_PHASE2_OPEN = "PHASE2_OPEN"
EV_OUTCOME_READ_AUTHORIZED = "OUTCOME_READ_AUTHORIZED"
EV_FIRST_TARGET_OUTCOME_READ = "FIRST_TARGET_OUTCOME_READ"
EV_EVALUATION_COMPLETE = "EVALUATION_COMPLETE"
EV_EVALUATION_MANIFEST_FROZEN = "EVALUATION_MANIFEST_FROZEN"
EV_AGGREGATION_STARTED = "AGGREGATION_STARTED"
EV_REPORT_BUILT = "REPORT_BUILT"

#: The required order. A critical event may occur **at most once** in a run,
#: and never before any event listed above it.
GLOBAL_EVENT_ORDER = (
    EV_PHASE1_STARTED,
    EV_ALL_DECISIONS_FROZEN,
    EV_RELEASE_AUTHORITY_VERIFIED,
    EV_PHASE2_OPEN,
    EV_OUTCOME_READ_AUTHORIZED,
    EV_FIRST_TARGET_OUTCOME_READ,
    EV_EVALUATION_COMPLETE,
    EV_EVALUATION_MANIFEST_FROZEN,
    EV_AGGREGATION_STARTED,
    EV_REPORT_BUILT,
)
CRITICAL_GLOBAL_EVENTS = frozenset(GLOBAL_EVENT_ORDER)

#: The four the barrier claim rests on. `assert_global_ordering` requires these
#: to be present; the rest are ordered if present.
REQUIRED_BARRIER_EVENTS = (
    EV_ALL_DECISIONS_FROZEN,
    EV_RELEASE_AUTHORITY_VERIFIED,
    EV_PHASE2_OPEN,
    EV_FIRST_TARGET_OUTCOME_READ,
)

#: Provenance a ledger may be bound to. Write-once: a second, different value
#: for the same field is a hard failure, so a run cannot re-point its ledger at
#: another decision store half way through.
LEDGER_PROVENANCE_FIELDS = (
    "protocol_version", "context_snapshot_checksum", "split_registry_checksum",
    "decision_manifest_checksum", "release_checksum",
    "evaluation_manifest_checksum", "ledger_binding_checksum",
)

#: The context-bound run-control artifact that names the one authoritative
#: ledger id. Its path is never accepted from a caller; it is always resolved
#: as ``context.workdir / LEDGER_BINDING_FILENAME``.
LEDGER_BINDING_VERSION = "S2_LEDGER_BINDING/1"
LEDGER_BINDING_FILENAME = "authoritative_ledger_binding.json"
LEDGER_BINDING_LOCK_FILENAME = ".authoritative_ledger_binding.lock"
LEDGER_BINDING_FIELDS = (
    "binding_version", "protocol_version", "run_id",
    "context_snapshot_checksum", "split_registry_checksum",
    "authoritative_ledger_id", "binding_checksum",
)

#: Event fields every row carries.
LEDGER_EVENT_FIELDS = ("seq", "event", "run_id", "ledger_id",
                       "protocol_version", "utc", "monotonic_seconds")


class LedgerError(AssertionError):
    """Raised when the global event stream cannot support an ordering claim."""


def _binding_path_for_context(context) -> Path:
    """Resolve the ledger binding path from context authority, never input."""
    from s2_production import ProductionAuthorityError, ProductionS2Context

    if not isinstance(context, ProductionS2Context):
        raise ProductionAuthorityError(
            "an authoritative ledger binding requires a ProductionS2Context")
    context.assert_still_authoritative()
    root = Path(context.workdir)
    if context.assert_evaluation_root_authoritative().parent != root:
        raise ProductionAuthorityError(
            "the ledger binding root is not the context's authoritative run root")
    return root / LEDGER_BINDING_FILENAME


def authoritative_ledger_binding_path(context) -> Path:
    """Public read-only resolver for the context-bound binding location."""
    return _binding_path_for_context(context)


def _binding_body(context, ledger_id: str) -> dict:
    """Re-derive every contextual field recorded by the binding."""
    snapshot = context.snapshot
    return {
        "binding_version": LEDGER_BINDING_VERSION,
        "protocol_version": sd.PROTOCOL_VERSION,
        "run_id": context.run_id,
        "context_snapshot_checksum": sd.sha256_bytes(
            sd.canonical_json(snapshot)),
        "split_registry_checksum": snapshot["split_registry_checksum"],
        "authoritative_ledger_id": str(ledger_id),
    }


def _seal_binding(body: dict) -> dict:
    payload = dict(body)
    payload["binding_checksum"] = sd.sha256_bytes(sd.canonical_json(payload))
    return payload


def validate_authoritative_ledger_binding(payload: dict, context) -> dict:
    """Validate integrity and re-derive all non-ledger fields from context."""
    if not isinstance(payload, dict):
        raise LedgerError("the authoritative ledger binding is not an object")
    missing = set(LEDGER_BINDING_FIELDS) - set(payload)
    extra = set(payload) - set(LEDGER_BINDING_FIELDS)
    if missing or extra:
        raise LedgerError(
            f"the authoritative ledger binding has {sorted(missing)} missing "
            f"and {sorted(extra)} unexpected fields")

    body = {k: payload[k] for k in LEDGER_BINDING_FIELDS
            if k != "binding_checksum"}
    checksum = sd.sha256_bytes(sd.canonical_json(body))
    if payload["binding_checksum"] != checksum:
        raise LedgerError(
            "the authoritative ledger binding checksum does not match its content")

    expected = _binding_body(context, payload["authoritative_ledger_id"])
    for field in ("binding_version", "protocol_version", "run_id",
                  "context_snapshot_checksum", "split_registry_checksum"):
        if payload[field] != expected[field]:
            raise LedgerError(
                f"authoritative ledger binding {field!r} is "
                f"{payload[field]!r}; current context re-derives to "
                f"{expected[field]!r}")
    try:
        parsed = uuid.UUID(hex=str(payload["authoritative_ledger_id"]))
    except (ValueError, AttributeError, TypeError) as exc:
        raise LedgerError(
            "the authoritative ledger id is not a UUID hex identifier") from exc
    if parsed.hex != payload["authoritative_ledger_id"]:
        raise LedgerError(
            "the authoritative ledger id is not canonical lowercase UUID hex")
    return dict(payload)


def read_authoritative_ledger_binding(context) -> dict:
    """Read and revalidate the binding at the context-derived run root."""
    path = _binding_path_for_context(context)
    if not path.is_file():
        raise LedgerError(
            f"the authoritative ledger binding is absent at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerError(
            f"the authoritative ledger binding at {path} cannot be read") from exc
    return validate_authoritative_ledger_binding(payload, context)


def resolve_or_create_authoritative_ledger_binding(context) -> dict:
    """Atomically create once, otherwise resolve, the run's ledger binding.

    An advisory exclusive lock serialises supported production creators. The
    winner writes a complete temporary artifact, fsyncs it and atomically
    replaces the final path; every loser then reads and validates the winner.
    No caller supplies either path or expected id.
    """
    path = _binding_path_for_context(context)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / LEDGER_BINDING_LOCK_FILENAME
    with open(lock_path, "a+b") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                return read_authoritative_ledger_binding(context)

            payload = _seal_binding(_binding_body(context, uuid.uuid4().hex))
            temp_path = path.with_name(
                f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            fd = os.open(temp_path, flags, 0o600)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(sd.canonical_json(payload))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(temp_path, path)
                dir_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
            return read_authoritative_ledger_binding(context)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


class GlobalRunLedger:
    """The one sequence allocator of a run. Parent-owned, never per-component.

    Nothing else in S2 may allocate an ordering number. The firewall, the
    outcome loader, the evaluation store, the aggregation and the report all
    append to **this** object, so a comparison between any two of their sequence
    numbers is meaningful.
    """

    __slots__ = ("_ledger_id", "_run_id", "_protocol_version", "_provenance",
                 "_events", "_next_seq", "_t0")

    def __init__(self, run_id: str = None, **provenance) -> None:
        object.__setattr__(self, "_ledger_id", uuid.uuid4().hex)
        object.__setattr__(self, "_run_id", str(run_id or uuid.uuid4().hex))
        object.__setattr__(self, "_protocol_version", sd.PROTOCOL_VERSION)
        object.__setattr__(self, "_provenance",
                           {"protocol_version": sd.PROTOCOL_VERSION})
        object.__setattr__(self, "_events", [])
        object.__setattr__(self, "_next_seq", 1)
        object.__setattr__(self, "_t0", time.perf_counter())
        if provenance:
            self.bind(**provenance)

    def __setattr__(self, name, value):
        raise LedgerError("the global run ledger's identity is immutable; "
                          "append events, do not reassign its fields")

    # -- identity -----------------------------------------------------------
    @property
    def ledger_id(self) -> str:
        return self._ledger_id

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def protocol_version(self) -> str:
        return self._protocol_version

    @property
    def provenance(self) -> dict:
        return dict(self._provenance)

    @classmethod
    def for_context(cls, context) -> "GlobalRunLedger":
        """Resolve the production ledger of one authoritative run.

        The first call atomically creates the context-bound ledger binding. A
        later call re-opens that binding and therefore returns a ledger with
        the *same* id; it never mints a second authoritative id. Manifest
        checksums are bound later, as they come into existence.
        """
        binding = resolve_or_create_authoritative_ledger_binding(context)
        snapshot = context.snapshot
        ledger = cls(
            context.run_id,
            context_snapshot_checksum=sd.sha256_bytes(sd.canonical_json(snapshot)),
            split_registry_checksum=snapshot["split_registry_checksum"],
            ledger_binding_checksum=binding["binding_checksum"],
        )
        # Only this resolver adopts an already-authoritative id. The public
        # constructor always mints a distinct id, which is how the regression
        # counterexample constructs its fully stamped second allocator.
        object.__setattr__(ledger, "_ledger_id",
                           binding["authoritative_ledger_id"])
        return ledger

    # -- provenance ---------------------------------------------------------
    def bind(self, **fields) -> dict:
        """Bind provenance write-once. Re-binding the same value is a no-op."""
        for name, value in fields.items():
            if name not in LEDGER_PROVENANCE_FIELDS:
                raise LedgerError(
                    f"{name!r} is not a ledger provenance field "
                    f"(frozen set: {LEDGER_PROVENANCE_FIELDS})")
            current = self._provenance.get(name)
            if current is not None and current != value:
                raise LedgerError(
                    f"ledger provenance {name!r} is already bound to "
                    f"{current!r}; a run may not re-point it to {value!r}")
            self._provenance[name] = value
        return dict(self._provenance)

    # -- the allocator ------------------------------------------------------
    def append(self, event: str, **detail) -> dict:
        """Allocate the next sequence number and record one event.

        This is the **only** place in S2 where an ordering number is created.
        A critical global event may be appended once per run; a second attempt
        raises rather than overwriting, so a duplicate `PHASE2_OPEN` cannot be
        used to re-order the stream.
        """
        event = str(event)
        if event in CRITICAL_GLOBAL_EVENTS and self.seq_of(event) is not None:
            raise LedgerError(
                f"critical global event {event!r} has already been recorded at "
                f"seq {self.seq_of(event)}; it occurs at most once per run")
        forbidden = set(detail) & set(LEDGER_EVENT_FIELDS)
        if forbidden:
            raise LedgerError(
                f"an event may not overwrite its own identity fields "
                f"{sorted(forbidden)}")
        row = {
            "seq": self._next_seq, "event": event, "run_id": self._run_id,
            "ledger_id": self._ledger_id,
            "protocol_version": self._protocol_version,
            "utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_seconds": time.perf_counter() - self._t0,
            **detail,
        }
        self._events.append(row)
        self._advance()
        return dict(row)

    def _advance(self) -> None:
        """Move the allocator on by one. The only mutation of the counter."""
        object.__setattr__(self, "_next_seq", self._next_seq + 1)

    # -- reading ------------------------------------------------------------
    @property
    def events(self) -> tuple:
        return tuple(dict(e) for e in self._events)

    def __len__(self) -> int:
        return len(self._events)

    def seq_of(self, event: str, cell=None):
        """The sequence number of `event` (its first occurrence), or `None`."""
        hits = [e["seq"] for e in self._events
                if e["event"] == event
                and (cell is None or tuple(e.get("cell", ())) == tuple(cell))]
        return min(hits) if hits else None

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._events)

    def barrier_seqs(self) -> dict:
        """The named sequence numbers the protocol's ordering claim is made on.

        All six come from this allocator, so they are on one scale. §21 keeps
        the authorization and the first materialization apart: the gate can pass
        and the read still fail, and the ordering claim is made on the read.
        """
        return {
            "ledger_id": self._ledger_id, "run_id": self._run_id,
            "phase1_started_seq": self.seq_of(EV_PHASE1_STARTED),
            "all_decisions_frozen_seq": self.seq_of(EV_ALL_DECISIONS_FROZEN),
            "phase2_authority_verified_seq": self.seq_of(
                EV_RELEASE_AUTHORITY_VERIFIED),
            "phase2_open_seq": self.seq_of(EV_PHASE2_OPEN),
            "outcome_read_authorized_seq": self.seq_of(
                EV_OUTCOME_READ_AUTHORIZED),
            "first_outcome_row_materialized_seq": self.seq_of(
                EV_FIRST_TARGET_OUTCOME_READ),
            "first_target_outcome_read_seq": self.seq_of(
                EV_FIRST_TARGET_OUTCOME_READ),
            "evaluation_complete_seq": self.seq_of(EV_EVALUATION_COMPLETE),
            "evaluation_manifest_frozen_seq": self.seq_of(
                EV_EVALUATION_MANIFEST_FROZEN),
            "aggregation_started_seq": self.seq_of(EV_AGGREGATION_STARTED),
            "report_built_seq": self.seq_of(EV_REPORT_BUILT),
        }


# ---------------------------------------------------------------------------
# the ordering assertion
# ---------------------------------------------------------------------------
def assert_event_stream_ordering(events, ledger_id: str, run_id: str,
                                 required=REQUIRED_BARRIER_EVENTS) -> dict:
    """Validate a raw event stream. **One ledger, one run, one sequence.**

    Every event must carry the given `ledger_id` and `run_id`, the sequence
    numbers must be exactly `1..n` with no gap and no repetition, each critical
    event must occur at most once, and the critical events present must appear
    in `GLOBAL_EVENT_ORDER`.

    Merging two allocators' streams — the Stage 2.0.5 shape — fails on the
    identity check before the ordering is even looked at, which is the point:
    two counters cannot be made comparable by placing their numbers side by
    side.
    """
    rows = [dict(e) for e in events]
    if not rows:
        raise LedgerError("an empty event stream supports no ordering claim")

    ledgers = {str(e.get("ledger_id")) for e in rows}
    runs = {str(e.get("run_id")) for e in rows}
    if ledgers != {str(ledger_id)}:
        raise LedgerError(
            f"the event stream mixes {len(ledgers)} ledgers {sorted(ledgers)}; "
            "sequence numbers from different allocators are not comparable and "
            "may not be used as an ordering proof")
    if runs != {str(run_id)}:
        raise LedgerError(
            f"the event stream mixes {len(runs)} run ids {sorted(runs)}; a "
            "global ordering claim is made within one run")

    seqs = [e.get("seq") for e in rows]
    if any(not isinstance(s, int) or isinstance(s, bool) for s in seqs):
        raise LedgerError("every ledger event must carry an integer seq")
    if sorted(seqs) != list(range(1, len(seqs) + 1)):
        raise LedgerError(
            f"ledger sequence numbers are not exactly 1..{len(seqs)}: a gap, a "
            "duplicate or an externally injected number is present")
    if seqs != sorted(seqs):
        raise LedgerError("ledger events are not stored in allocation order")

    positions = {}
    for row in rows:
        event = str(row["event"])
        if event not in CRITICAL_GLOBAL_EVENTS:
            continue
        if event in positions:
            raise LedgerError(
                f"critical global event {event!r} occurs more than once "
                f"(seq {positions[event]} and {row['seq']})")
        positions[event] = int(row["seq"])

    missing = [e for e in required if e not in positions]
    if missing:
        raise LedgerError(
            f"the global ordering claim requires {list(required)}; missing "
            f"{missing}")

    present = [e for e in GLOBAL_EVENT_ORDER if e in positions]
    for earlier, later in zip(present, present[1:]):
        if positions[earlier] >= positions[later]:
            raise LedgerError(
                f"{earlier} (seq {positions[earlier]}) must precede {later} "
                f"(seq {positions[later]}); the global run order is "
                f"{list(GLOBAL_EVENT_ORDER)}")
    return {"n_events": len(rows), "ledger_id": str(ledger_id),
            "run_id": str(run_id), "critical_event_seqs": dict(positions)}


def assert_global_ordering(ledger, required=REQUIRED_BARRIER_EVENTS) -> dict:
    """The production assertion: one `GlobalRunLedger`, correctly ordered.

    Takes the ledger **object**, not a list of rows, so the identity the stream
    is checked against is the allocator's own — a caller cannot present a
    hand-assembled stream together with a matching pair of identifiers.
    """
    if not isinstance(ledger, GlobalRunLedger):
        raise LedgerError(
            "a global ordering claim is made on a GlobalRunLedger; a list of "
            "rows, a dict of counters or a second allocator's numbers will not "
            "do")
    out = assert_event_stream_ordering(ledger.events, ledger.ledger_id,
                                       ledger.run_id, required)
    seqs = ledger.barrier_seqs()
    ordered = (seqs["all_decisions_frozen_seq"]
               < seqs["phase2_authority_verified_seq"]
               < seqs["first_target_outcome_read_seq"])
    if not ordered:
        raise LedgerError(
            "all_decisions_frozen_seq < phase2_authority_verified_seq < "
            f"first_target_outcome_read_seq does not hold: {seqs}")
    out["barrier_seqs"] = seqs
    return out


def assert_full_run_ordering(ledger) -> dict:
    """Every one of the ten critical events, present and correctly ordered."""
    return assert_global_ordering(ledger, required=GLOBAL_EVENT_ORDER)


#: What must already have happened, in this run, before any formal quantity is
#: aggregated. Phase 2 must have opened, the first (and only) target-outcome
#: read must have occurred, and the evaluation store must have been closed and
#: its manifest frozen — otherwise "aggregation" would be reading a store that
#: was still being written (§17.25).
AGGREGATION_PREREQUISITE_EVENTS = (
    EV_ALL_DECISIONS_FROZEN,
    EV_RELEASE_AUTHORITY_VERIFIED,
    EV_PHASE2_OPEN,
    EV_FIRST_TARGET_OUTCOME_READ,
    EV_EVALUATION_COMPLETE,
    EV_EVALUATION_MANIFEST_FROZEN,
)


def assert_ledger_binding_for_context(ledger, context) -> dict:
    """Require the ledger id named by this context's run-control binding.

    This check intentionally performs no event-order assertion, so production
    components can call it before phase 1 starts. It does, however, re-read the
    binding from the context-derived run root and revalidate every binding and
    ledger provenance field. A same-run, fully stamped second allocator fails
    specifically because its id is not the authoritative bound id.
    """
    from s2_production import ProductionAuthorityError, ProductionS2Context

    if ledger is None:
        raise LedgerError(
            "production requires this run's authoritative GlobalRunLedger; "
            "`ledger=None` is not a permitted mode")
    if not isinstance(ledger, GlobalRunLedger):
        raise LedgerError(
            f"{type(ledger).__name__} is not a GlobalRunLedger; production "
            "requires the context-bound authoritative allocator")
    if not isinstance(context, ProductionS2Context):
        raise ProductionAuthorityError(
            "a run ledger is validated against a ProductionS2Context")

    binding = read_authoritative_ledger_binding(context)
    if ledger.run_id != context.run_id:
        raise LedgerError(
            f"the ledger belongs to run {ledger.run_id!r}, the context to "
            f"{context.run_id!r}; a global ordering claim is made within one run")
    if ledger.protocol_version != sd.PROTOCOL_VERSION:
        raise LedgerError(
            f"the ledger was allocated under protocol "
            f"{ledger.protocol_version!r}, not {sd.PROTOCOL_VERSION!r}")

    snapshot = context.snapshot
    provenance = ledger.provenance
    expected_provenance = {
        "context_snapshot_checksum": sd.sha256_bytes(
            sd.canonical_json(snapshot)),
        "split_registry_checksum": snapshot["split_registry_checksum"],
        "ledger_binding_checksum": binding["binding_checksum"],
    }
    for field, value in expected_provenance.items():
        if provenance.get(field) != value:
            raise LedgerError(
                f"ledger provenance {field!r} is {provenance.get(field)!r}; "
                f"this run's authority re-derives to {value!r}")

    if ledger.ledger_id != binding["authoritative_ledger_id"]:
        raise LedgerError(
            "authoritative ledger-id mismatch: presented "
            f"{ledger.ledger_id!r}, context-bound authority is "
            f"{binding['authoritative_ledger_id']!r}")
    return {"binding": binding, "ledger_id": ledger.ledger_id,
            "run_id": ledger.run_id, "ledger_provenance": provenance}


def assert_run_ledger_for_context(ledger, context,
                                  required=AGGREGATION_PREREQUISITE_EVENTS
                                  ) -> dict:
    """The ledger presented **is** this run's allocator, and it has got this far.

    Stage 2.0.6 let aggregation run with `ledger=None`, which skipped the
    ordering assertion entirely; a fresh ledger would have skipped it just as
    effectively, since an empty stream has nothing out of order in it. Both are
    refused here:

    * `None`, a list of rows or anything that is not a `GlobalRunLedger` fails;
    * a ledger whose `run_id` is not the context's fails — a second allocator
      is not made authoritative by carrying convincing events;
    * a ledger whose bound provenance does not match the context's freshly
      hashed snapshot and split registry fails;
    * a ledger missing any prerequisite event fails, which is what makes a
      *fresh* ledger — legal, correctly built, empty — unusable.
    """
    binding_check = assert_ledger_binding_for_context(ledger, context)
    out = assert_global_ordering(ledger, required=required)
    out["ledger_provenance"] = binding_check["ledger_provenance"]
    out["ledger_binding"] = binding_check["binding"]
    return out
