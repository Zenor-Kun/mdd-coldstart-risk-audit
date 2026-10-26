"""The authoritative S2 decision-artifact validator (Stage 2.0.2).

Stage 2.0.1 froze the guarantee tiers but left a production hole: the firewall's
`record_zero_label_decision()` hashed whatever JSON it was handed. Nothing
checked the artifact's `strategy`, `state` or `guarantee_tier`, so a
`pooled_source` artifact carrying `TARGET_CERTIFIED` could be written, frozen
and trusted — a type system that only the *helpers* obeyed.

This module closes that. **Every** decision artifact in S2 — zero-label or
few-label, parent-written or worker-written — passes through
`freeze_decision_artifact`, whose order is fixed and not negotiable:

    validate  ->  canonical serialize  ->  write  ->  checksum  ->  freeze

Hashing before validating is what made "tamper after validation, before
checksum" possible; serializing the *validated object itself* removes the window
rather than narrowing it.

The scientific semantics are unchanged: this module adds no rule, it makes the
Stage 2.0.1 rules unbypassable from the production path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import certification as cert
import contracts

#: Bumped only when the frozen protocol document changes. Every decision
#: artifact carries it, and the global release artifact binds it.
PROTOCOL_VERSION = "RQ2_S2_PROTOCOL/2.0.8"

#: Fields every decision artifact must carry. A missing `guarantee_tier` is a
#: hard failure, not a defaulted value.
REQUIRED_DECISION_FIELDS = (
    "protocol_version", "site", "method", "strategy", "guarantee_tier",
    "state", "reason", "selected_q", "selected_tau",
    "selected_candidate_index", "selected_score",
)
#: Additional fields a per-cell (label-consuming) decision must carry.
REQUIRED_CELL_FIELDS = ("repeat", "n_labels", "split_checksum")

#: The states each zero-label strategy is permitted to reach. Narrower than the
#: tier's state set only in that it is spelled out at the production boundary.
ZERO_LABEL_EXPECTED_STATES = {
    "pooled_source": (cert.SOURCE_REFERENCE_PASS, cert.SOURCE_REFERENCE_FAIL,
                      cert.NOT_TESTABLE),
    "target_unlabeled_weighted": (cert.EMPIRICAL_PASS, cert.EMPIRICAL_NO_PASS,
                                  cert.NOT_TESTABLE),
}


class DecisionArtifactError(AssertionError):
    """Raised by the production validator. Never downgraded to a warning."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(payload: dict) -> bytes:
    """One byte-string per artifact: sorted keys, no whitespace drift."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


# ---------------------------------------------------------------------------
# the single authoritative validator
# ---------------------------------------------------------------------------
def validate_decision_artifact(payload: dict, *, zero_label: bool = None) -> dict:
    """Type-check one decision artifact against the frozen guarantee tiers.

    Mechanically calls `certification.assert_state_allowed` and the tier map,
    so the production path cannot disagree with the helpers. Raises
    `DecisionArtifactError` on any mismatch; there is no permissive branch and
    no default.
    """
    if not isinstance(payload, dict):
        raise DecisionArtifactError(f"decision artifact is {type(payload)}, not a dict")

    missing = [f for f in REQUIRED_DECISION_FIELDS if f not in payload]
    if missing:
        raise DecisionArtifactError(f"decision artifact is missing {missing}")

    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise DecisionArtifactError(
            f"protocol_version {payload['protocol_version']!r} != "
            f"{PROTOCOL_VERSION!r}")

    strategy = payload["strategy"]
    if strategy not in cert.STRATEGIES:
        raise DecisionArtifactError(f"{strategy!r} is not a frozen S2 strategy")

    site = str(payload["site"])
    if site not in contracts.FROZEN_SITES:
        raise DecisionArtifactError(f"{site!r} is not a frozen eligible site")
    if payload["method"] not in cert.CONFIDENCE_METHODS:
        raise DecisionArtifactError(f"{payload['method']!r} is not a frozen method")

    # --- the guarantee-tier gate ------------------------------------------
    expected_tier = cert.tier_of(strategy)
    if payload["guarantee_tier"] != expected_tier:
        raise DecisionArtifactError(
            f"{strategy!r} declares guarantee_tier {payload['guarantee_tier']!r}, "
            f"but its frozen tier is {expected_tier!r}")
    state = payload["state"]
    try:
        cert.assert_state_allowed(strategy, state)
    except (AssertionError, KeyError) as exc:
        raise DecisionArtifactError(
            f"{strategy!r} may not emit state {state!r}: {exc}") from exc

    # --- zero-label vs label-consuming ------------------------------------
    is_zero = strategy in cert.ZERO_LABEL_STRATEGIES
    if zero_label is not None and bool(zero_label) != is_zero:
        raise DecisionArtifactError(
            f"{strategy!r} is {'a' if is_zero else 'not a'} zero-label strategy; "
            f"it was recorded on the {'zero-label' if zero_label else 'cell'} path")
    if is_zero:
        if state not in ZERO_LABEL_EXPECTED_STATES[strategy]:
            raise DecisionArtifactError(
                f"zero-label {strategy!r} produced {state!r}; permitted states "
                f"are {ZERO_LABEL_EXPECTED_STATES[strategy]}")
        for field in ("repeat", "n_labels"):
            if payload.get(field) is not None:
                raise DecisionArtifactError(
                    f"zero-label {strategy!r} carries {field}={payload[field]!r}; "
                    "a zero-label decision has no label-budget axis")
    else:
        missing = [f for f in REQUIRED_CELL_FIELDS if payload.get(f) is None]
        if missing:
            raise DecisionArtifactError(f"cell decision is missing {missing}")
        for field in ("repeat", "n_labels"):
            v = payload[field]
            if isinstance(v, bool) or not isinstance(v, int):
                raise DecisionArtifactError(
                    f"cell decision field {field!r} must be a single integer")
        if payload["n_labels"] not in cert.N_GRID:
            raise DecisionArtifactError(
                f"n_labels {payload['n_labels']!r} is off the frozen N grid")
        if not 0 <= payload["repeat"] < cert.R_SPLIT:
            raise DecisionArtifactError(f"repeat {payload['repeat']!r} out of range")

    assert_decision_field_consistency(payload)
    return payload


# ---------------------------------------------------------------------------
# the complete state-dependent field schema (Stage 2.0.3)
# ---------------------------------------------------------------------------
#: Reason codes. `NOT_TESTABLE` must name *which* constraint bit; a generic
#: empty string or `None` is refused.
REASON_NO_ELIGIBLE = cert.REASON_NO_ELIGIBLE            # coverage / class filter
REASON_NO_PASS = cert.REASON_NO_PASS
REASON_CLASS_COUNT = "NO_ELIGIBLE_CANDIDATE__CLASS_COUNT"
REASON_COVERAGE = "NO_ELIGIBLE_CANDIDATE__COVERAGE"
NOT_TESTABLE_REASONS = (REASON_NO_ELIGIBLE, REASON_CLASS_COUNT, REASON_COVERAGE)
VALID_REASONS = (None, REASON_NO_PASS) + NOT_TESTABLE_REASONS

#: Fields that must be jointly present on a pass, and jointly absent otherwise.
SELECTION_FIELDS = ("selected_q", "selected_tau", "selected_candidate_index",
                    "selected_score")


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def assert_decision_field_consistency(payload: dict) -> None:
    """One complete state-dependent schema, applied to every tier.

    Stage 2.0.2 checked `selected_q` alone, so a pass state with
    `selected_tau = None`, or a fail state carrying a threshold, could be
    frozen. A decision that "failed but still selected a threshold" is not a
    decision, and it may not reach disk.

    * **pass states** (`TARGET_CERTIFIED`, `SOURCE_REFERENCE_PASS`,
      `EMPIRICAL_PASS`) require a finite `selected_q` **on the frozen grid**, a
      finite `selected_tau`, a valid `selected_candidate_index` that **points at
      that same `q`**, and a finite `selected_score` (the certifying `U_BER` for
      the formal tier, the screening score otherwise).
    * **fail states** (`TARGET_UNCERTIFIED`, `SOURCE_REFERENCE_FAIL`,
      `EMPIRICAL_NO_PASS`) require **all four selection fields to be `None`**,
      and reason `NO_PASSING_CANDIDATE`.
    * **`NOT_TESTABLE`** requires all four `None` and an explicit reason code
      naming the constraint that bit.
    """
    state = payload["state"]
    present = {f: payload.get(f) for f in SELECTION_FIELDS}

    if state == cert.NOT_TESTABLE:
        offenders = {f: v for f, v in present.items() if v is not None}
        if offenders:
            raise DecisionArtifactError(
                f"NOT_TESTABLE carries selection fields {offenders}; nothing "
                "was tested, so nothing may be selected")
        if payload["reason"] not in NOT_TESTABLE_REASONS:
            raise DecisionArtifactError(
                f"NOT_TESTABLE must name the constraint that bit; got "
                f"{payload['reason']!r}, expected one of {NOT_TESTABLE_REASONS}")
        return

    if cert.is_pass(state):
        missing = [f for f, v in present.items() if v is None]
        if missing:
            raise DecisionArtifactError(
                f"pass state {state!r} is missing {missing}; a passing decision "
                "must name the threshold it selected")
        if payload["selected_q"] not in cert.Q_GRID:
            raise DecisionArtifactError(
                f"selected_q {payload['selected_q']!r} is off the frozen grid")
        idx = payload["selected_candidate_index"]
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise DecisionArtifactError(
                f"selected_candidate_index {idx!r} must be an integer")
        if not 0 <= idx < len(cert.Q_GRID):
            raise DecisionArtifactError(
                f"selected_candidate_index {idx} is outside the frozen grid")
        if cert.Q_GRID[idx] != payload["selected_q"]:
            raise DecisionArtifactError(
                f"selected_candidate_index {idx} points at q="
                f"{cert.Q_GRID[idx]}, but selected_q is {payload['selected_q']}")
        for field in ("selected_tau", "selected_score"):
            if not _finite(payload[field]):
                raise DecisionArtifactError(
                    f"pass state {state!r} carries non-finite {field}="
                    f"{payload[field]!r}")
        if payload["reason"] is not None:
            raise DecisionArtifactError(
                f"pass state {state!r} must carry no failure reason "
                f"(got {payload['reason']!r})")
        return

    # remaining case: the tier's fail state
    offenders = {f: v for f, v in present.items() if v is not None}
    if offenders:
        raise DecisionArtifactError(
            f"fail state {state!r} carries selection fields {offenders}; a "
            "decision that failed may not also select a threshold")
    if payload["reason"] != REASON_NO_PASS:
        raise DecisionArtifactError(
            f"fail state {state!r} must carry reason {REASON_NO_PASS!r}, got "
            f"{payload['reason']!r}")


def freeze_decision_artifact(payload: dict, path, *,
                             zero_label: bool = None) -> dict:
    """validate -> canonical serialize -> write -> checksum -> freeze.

    The order is the point. The bytes that are hashed are the bytes produced
    from the **validated** object, so there is no window in which a state can be
    edited after validation and before checksumming.
    """
    validated = validate_decision_artifact(payload, zero_label=zero_label)
    blob = canonical_json(validated)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(blob)
    digest = sha256_bytes(blob)
    on_disk = sha256_file(p)
    if on_disk != digest:
        raise DecisionArtifactError(
            f"{p}: bytes on disk do not match the validated payload")
    return {"path": str(p), "sha256": digest, "bytes": len(blob),
            "payload": validated}


def load_and_validate(path, expected_checksum: str = None, *,
                      zero_label: bool = None) -> dict:
    """Re-read a frozen artifact, re-verify its checksum, re-run the validator.

    Used by the global barrier and by every evaluation worker: an artifact that
    was valid when written must still be valid, and still be the same bytes,
    when it is consumed.
    """
    p = Path(path)
    if not p.exists():
        raise DecisionArtifactError(f"decision artifact {p} is not on disk")
    blob = p.read_bytes()
    digest = sha256_bytes(blob)
    if expected_checksum is not None and digest != expected_checksum:
        raise DecisionArtifactError(
            f"{p}: checksum {digest[:12]}... != expected {expected_checksum[:12]}...")
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise DecisionArtifactError(f"{p}: malformed JSON ({exc})") from exc
    validate_decision_artifact(payload, zero_label=zero_label)
    return {"path": str(p), "sha256": digest, "payload": payload}
