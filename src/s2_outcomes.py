"""The sole production path to target outcomes (RQ2/S2, Stage 2.0.8).

Stage 2.0.6 removes this module's private sequence counter. It records into the
run's single `s2_ledger.GlobalRunLedger` — the same instance the firewall uses —
so `first_target_outcome_read_seq` is on the same scale as
`all_decisions_frozen_seq` and the comparison between them means something. §21
keeps two moments apart, both allocated here:

    OUTCOME_READ_AUTHORIZED     every gate passed; nothing opened yet
    FIRST_TARGET_OUTCOME_READ   the first rows actually materialized

The ordering claim is made on the second. A gate that passes and a read that
then fails leaves the first event and not the second, which is the honest
record.

The original Stage 2.0.5 header follows, unchanged.

Every earlier stage tried to keep the *worker* from reading labels. Stage 2.0.5
inverts it: the worker has no such capability at all (`src/s2_eval_worker.py` is
a pure function), and materializing the authoritative target outcomes is the
parent's job, through this module, **once**, and only after the global barrier:

    all 46,200 decisions frozen
      -> complete_production_phase1()      exact decision-store equality
      -> open_phase2()                     full release authority re-derivation
      -> load_authoritative_target_outcomes_for_phase2()   <- the only reader

Reading the outcome table at that point is not decision leakage: every S2
decision already exists on disk, checksummed and re-validated. The ledger
records the ordering so the claim is auditable rather than asserted:

    all_decisions_frozen_seq < phase2_authority_verified_seq
                             < first_target_outcome_read_seq

During phase 1 the parent holds **no** target label, no diagnosis frame and no
outcome cache; `target_outcomes_materialized_pre_phase2` is recorded `False`.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import certification as cert
import s2_ledger as ledgermod
import s2_production as prod

LABEL_COLUMN = "y_true"
ID_COLUMN = "canonical_id"
SITE_COLUMN = "held_out_site"
PRED_COLUMN = "pred"

#: Append-only provenance of every outcome materialization in this process.
#: Not a data cache: it holds counts, sequence numbers and phase states only.
_OUTCOME_READ_EVENTS: list = []


class OutcomeAccessError(AssertionError):
    """Raised before a single outcome row is read when the gate is not open."""


def outcome_read_ledger() -> pd.DataFrame:
    return pd.DataFrame(_OUTCOME_READ_EVENTS)


def total_rows_read() -> int:
    return int(sum(e["rows_read"] for e in _OUTCOME_READ_EVENTS))


def _stamp(event: str, ledger=None, **detail) -> dict:
    """Record one outcome-access event. **Allocates no ordering number.**

    When a run ledger is available the sequence number comes from it — this
    module has no counter of its own, which is what makes
    `first_target_outcome_read_seq` comparable with the barrier's numbers. A
    refusal before a ledger exists is still recorded, with `ledger_seq = None`
    and `rows_read = 0`.
    """
    row = {"event": event, "utc": datetime.now(timezone.utc).isoformat(),
           "monotonic_seconds": time.perf_counter(), "rows_read": 0,
           "ledger_seq": None, "ledger_id": None, **detail}
    if ledger is not None:
        allocated = ledger.append(event, **{k: v for k, v in detail.items()
                                            if k != "reason"})
        row["ledger_seq"] = allocated["seq"]
        row["ledger_id"] = allocated["ledger_id"]
    _OUTCOME_READ_EVENTS.append(row)
    return row


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def _resolve_run_ledger(firewall, ledger):
    """The run's single allocator — the firewall's own, never a second one.

    The ledger is passed explicitly, as the official runner does, and must be
    the very object the firewall holds. A *different* one is refused, because
    two allocators cannot produce a comparable ordering (§22); `None` is refused
    too, since an omitted ledger is how Stage 2.0.6 let an ordering claim go
    unmade (§17.25).
    """
    owned = getattr(firewall, "ledger", None)
    if not isinstance(owned, ledgermod.GlobalRunLedger):
        raise OutcomeAccessError(
            "the firewall carries no global run ledger; there is nothing to "
            "order this read against. rows_read=0")
    if not isinstance(ledger, ledgermod.GlobalRunLedger):
        _stamp("outcome_read_refused", reason="no ledger presented")
        raise OutcomeAccessError(
            "this run's GlobalRunLedger must be presented with the read; there "
            "is no unordered mode. rows_read=0")
    if ledger is not owned:
        _stamp("outcome_read_refused", reason="foreign ledger")
        raise OutcomeAccessError(
            "the ledger presented is not this run's allocator; sequence numbers "
            "from two allocators are not comparable. rows_read=0")
    return owned


def assert_outcome_read_allowed(context, firewall, ledger) -> dict:
    """Every condition that must hold before one outcome row may be read.

    Raises `OutcomeAccessError` — with `rows_read = 0` recorded — unless the
    context re-derives as authoritative, the release passes a **full authority**
    verification (which itself requires the decision store to equal the
    authoritative 46,200 exactly), and the firewall is in `PHASE2_OPEN`.
    """
    import s2_firewall as fw

    if not isinstance(context, prod.ProductionS2Context):
        _stamp("outcome_read_refused", reason="not a ProductionS2Context")
        raise OutcomeAccessError(
            "target outcomes are materialized only under a ProductionS2Context")
    if not isinstance(firewall, fw.ProductionS2Firewall):
        _stamp("outcome_read_refused", reason="not a ProductionS2Firewall")
        raise OutcomeAccessError(
            "target outcomes are materialized only under the production firewall")
    run_ledger = _resolve_run_ledger(firewall, ledger)
    ledgermod.assert_ledger_binding_for_context(run_ledger, context)
    if firewall.phase != fw.PHASE_2_OPEN:
        _stamp("outcome_read_refused", phase=firewall.phase,
               reason="phase 2 is not open")
        raise OutcomeAccessError(
            f"phase 2 is not open (state {firewall.phase}); no target outcome "
            "may be materialized before the global barrier. rows_read=0")
    release = getattr(firewall, "_release", None)
    if release is None:
        _stamp("outcome_read_refused", reason="no release")
        raise OutcomeAccessError("no global release exists. rows_read=0")
    try:
        context.assert_still_authoritative()
        fw.verify_release_authority(release["path"], context)
    except AssertionError as exc:
        _stamp("outcome_read_refused", reason=str(exc)[:120])
        raise OutcomeAccessError(
            f"release authority does not re-derive: {exc}. rows_read=0") from exc

    return _stamp(ledgermod.EV_OUTCOME_READ_AUTHORIZED, ledger=run_ledger,
                  phase=firewall.phase,
                  release_checksum=release["checksum"][:16])


# ---------------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------------
class TargetOutcomeTable:
    """The materialized outcome table, addressable by `canonical_id`.

    Held by the parent **after** phase 2 opens, and handed to the pure
    evaluation worker one cell at a time. It exposes no path and no reader:
    once built, it is data.
    """

    __slots__ = ("_by_id", "_n_rows", "_source")

    def __init__(self, frame: pd.DataFrame, source) -> None:
        object.__setattr__(self, "_source", str(source))
        object.__setattr__(self, "_n_rows", int(len(frame)))
        by_id = {}
        for row in frame.itertuples(index=False):
            by_id[str(getattr(row, ID_COLUMN))] = row
        if len(by_id) != len(frame):
            raise OutcomeAccessError("duplicate canonical_id in the outcome table")
        object.__setattr__(self, "_by_id", by_id)

    def __setattr__(self, name, value):
        raise OutcomeAccessError("the materialized outcome table is immutable")

    @property
    def n_rows(self) -> int:
        return self._n_rows

    def for_cell(self, site: str, canonical_ids, method: str) -> dict:
        """The aligned `(y, pred, confidence)` arrays for one cell's partition."""
        ids = [str(i) for i in canonical_ids]
        missing = [i for i in ids if i not in self._by_id]
        if missing:
            raise OutcomeAccessError(
                f"{len(missing)} requested ids are not in the outcome table")
        rows = [self._by_id[i] for i in ids]
        foreign = [r for r in rows if str(getattr(r, SITE_COLUMN)) != str(site)]
        if foreign:
            raise OutcomeAccessError(
                f"{len(foreign)} requested ids belong to another site")
        column = f"confidence_{method}"
        return {
            "test_canonical_ids": ids,
            "y_test": np.array([int(getattr(r, LABEL_COLUMN)) for r in rows]),
            "pred_test": np.array([int(getattr(r, PRED_COLUMN)) for r in rows]),
            "confidence_test": np.array([float(getattr(r, column)) for r in rows]),
        }


def _materialize_outcome_table(path) -> TargetOutcomeTable:
    """Read an outcome table and count the rows. Performs no gating itself."""
    frame = pd.read_csv(Path(path), float_precision="round_trip")
    for column in (SITE_COLUMN, ID_COLUMN, LABEL_COLUMN, PRED_COLUMN):
        if column not in frame.columns:
            raise OutcomeAccessError(f"the outcome source lacks {column!r}")
    for method in cert.CONFIDENCE_METHODS:
        if f"confidence_{method}" not in frame.columns:
            raise OutcomeAccessError(f"the outcome source lacks confidence_{method}")
    return TargetOutcomeTable(frame, path)


def load_authoritative_target_outcomes_for_phase2(context, firewall, ledger
                                                  ) -> TargetOutcomeTable:
    """**The only production path that reads target outcomes.**

    Gate first, read second: a refusal records `rows_read = 0` and never opens
    the file. On success the ledger records the first read, its row count and
    the phase state, so the ordering claim of §17.19 is evidence rather than
    prose.
    """
    assert_outcome_read_allowed(context, firewall, ledger)
    run_ledger = _resolve_run_ledger(firewall, ledger)
    source = prod.resolve_authoritative_outcome_source()
    prod.assert_authoritative_outcome_source(source)
    table = _materialize_outcome_table(source)
    # the rows exist now; the ordering claim is made on this event, not on the
    # authorization that preceded it
    _stamp(ledgermod.EV_FIRST_TARGET_OUTCOME_READ, ledger=run_ledger,
           rows_read=table.n_rows, phase="PHASE2_OPEN", source=str(source))
    return table


def phase_ordering_ledger(firewall) -> dict:
    """The ordering §17.19 requires — every number from **one** allocator."""
    seqs = firewall.ledger.barrier_seqs()
    return {
        **seqs,
        "target_outcomes_materialized_pre_phase2": False,
        "n_outcome_rows_read": total_rows_read(),
        "ordering_ok": bool(
            seqs["all_decisions_frozen_seq"] is not None
            and seqs["phase2_authority_verified_seq"] is not None
            and seqs["first_target_outcome_read_seq"] is not None
            and seqs["all_decisions_frozen_seq"]
            < seqs["phase2_authority_verified_seq"]
            < seqs["first_target_outcome_read_seq"]),
    }
