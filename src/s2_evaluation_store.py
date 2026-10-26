"""The authoritative phase-2 evaluation artifact store (RQ2/S2, Stage 2.0.8).

Stage 2.0.5 removed the caller's ability to hand the report *intervals*, but
left it able to hand the report **numbers**: `build_formal_s2_report` accepted a
`SiteLevelResults` whose provenance was two strings the caller typed. A stream
of fabricated site vectors with `decision_manifest_checksum = "FAKE"` produced a
formal report. Provenance the caller supplies is not provenance.

The fix is the same shape as Stage 2.0.5's: remove the input, not guard it. A
formal report is now computed from **artifacts on disk**, and this module is
where those artifacts live and are validated.

    one phase-2 evaluation cell  ->  one frozen evaluation artifact
        validate -> canonical serialize -> write -> checksum -> freeze

Three things are re-derived rather than believed:

* the **expected** evaluation-cell set, mechanically, from the authoritative
  production context and the frozen 216-key CI registry — never supplied;
* each artifact's **decision binding**: the decision checksum it names must be
  the checksum the decision store actually holds for that cell, re-read at scan
  time, and the frozen state and threshold must be unchanged;
* the **evaluation manifest**, recomputed from the files that are there.

An evaluation artifact carries counts, rates and states. It never carries a raw
label, a prediction vector, a confidence vector or a list of identifiers.

Stage 2.0.7 closes the last input that was left: **where** the artifacts are
read from. Stage 2.0.6's `scan_current_evaluation_store(context, store)` took a
store object, so a caller could populate `/tmp/fake_eval_store` with 78,600
structurally valid artifacts carrying correctly copied decision bindings and
arbitrary metrics, and aggregate that. The store parameter is gone from every
production entry point; the root comes from `resolve_authoritative_evaluation_store`
alone, which derives it from the context and accepts no path, no root, no store,
no environment override and no configuration key.
"""

from __future__ import annotations

import json
from pathlib import Path

import certification as cert
import contracts
import s2_decisions as sd

#: Where a run's frozen evaluation artifacts live, relative to its workdir.
#: Mirrored from `s2_production.EVALUATION_STORE_DIRNAME` — the protocol has one
#: evaluation directory name and it is a constant, not a setting.
EVALUATION_STORE = "evaluations"

#: Prefix of one frozen evaluation artifact.
EVALUATION_PREFIX = "eval__"


class EvaluationArtifactError(AssertionError):
    """Raised by the authoritative evaluation-artifact validator."""


# ---------------------------------------------------------------------------
# the artifact schema
# ---------------------------------------------------------------------------
EVAL_STATUS_COMPLETE = "EVALUATED"
EVAL_STATUS_NOT_ESTIMABLE = "NOT_ESTIMABLE_RETAINED_CLASS"
EVAL_STATUS_NO_THRESHOLD = "NO_THRESHOLD_ADOPTED"
EVALUATION_STATUSES = (EVAL_STATUS_COMPLETE, EVAL_STATUS_NOT_ESTIMABLE,
                       EVAL_STATUS_NO_THRESHOLD)

#: Every field an evaluation artifact must carry.
REQUIRED_EVALUATION_FIELDS = (
    "protocol_version", "run_id", "site", "repeat", "n_labels", "strategy",
    "confidence_method", "guarantee_tier", "decision_checksum",
    "split_checksum", "decision_state", "selected_q", "selected_tau",
    "evaluation_status", "coverage_test", "n_test", "n_keep_test",
    "n_keep_test_MDD", "n_keep_test_HC", "ber_test", "risk_excess",
    "realized_field", "realized_outcome", "r_star",
)
#: Fields that may be present in addition (diagnostics permitted by §24.5).
OPTIONAL_EVALUATION_FIELDS = (
    "jsd_keep_vs_all", "oracle_infeasible", "n_oracle_feasible_candidates",
    "eval_worker_pid",
)
EVALUATION_ARTIFACT_FIELDS = frozenset(REQUIRED_EVALUATION_FIELDS
                                       + OPTIONAL_EVALUATION_FIELDS)

#: Substrings that mark a field as carrying raw target data rather than a
#: summary. An evaluation artifact is a summary; it is not a copy of `D_test`.
#: Deliberately specific: `n_labels` is the label *budget* `N`, not a label, so
#: a bare `labels` token would flag the artifact's own budget field.
RAW_OUTCOME_TOKENS = ("y_true", "y_test", "y_cal", "label_vector", "raw_label",
                      "class_label", "y_label", "canonical_id", "test_ids",
                      "cal_ids", "pred_test", "confidence_test", "prob_",
                      "_rows")

#: Fields that contain an outcome-looking substring but are structural.
NON_OUTCOME_FIELDS = frozenset({"n_labels"})


def evaluation_cell_key(site, repeat, n_labels, strategy, method) -> tuple:
    return (str(site), int(repeat), int(n_labels), str(strategy), str(method))


def _artifact_filename(key: tuple) -> str:
    return EVALUATION_PREFIX + "__".join(str(k) for k in key) + ".json"


def validate_evaluation_artifact(payload: dict) -> dict:
    """Type-check one evaluation artifact. No permissive branch, no default."""
    if not isinstance(payload, dict):
        raise EvaluationArtifactError(
            f"evaluation artifact is {type(payload)}, not a dict")

    missing = [f for f in REQUIRED_EVALUATION_FIELDS if f not in payload]
    if missing:
        raise EvaluationArtifactError(
            f"evaluation artifact is missing {missing}")
    for field in payload:
        low = str(field).lower()
        if field in NON_OUTCOME_FIELDS:
            continue
        if any(token in low for token in RAW_OUTCOME_TOKENS):
            raise EvaluationArtifactError(
                f"evaluation artifact field {field!r} looks like raw target "
                "data; an artifact carries counts, rates and states only")
    extra = sorted(set(payload) - EVALUATION_ARTIFACT_FIELDS)
    if extra:
        raise EvaluationArtifactError(
            f"evaluation artifact carries unregistered fields {extra}")
    for field, value in payload.items():
        if isinstance(value, (list, tuple, dict)):
            raise EvaluationArtifactError(
                f"evaluation artifact field {field!r} is a container; only "
                "scalars may be frozen")

    if payload["protocol_version"] != sd.PROTOCOL_VERSION:
        raise EvaluationArtifactError(
            f"evaluation artifact protocol_version "
            f"{payload['protocol_version']!r} != {sd.PROTOCOL_VERSION!r}")
    site = str(payload["site"])
    if site not in contracts.FROZEN_SITES:
        raise EvaluationArtifactError(f"{site!r} is not a frozen eligible site")
    strategy = payload["strategy"]
    if strategy not in cert.STRATEGIES:
        raise EvaluationArtifactError(f"{strategy!r} is not a frozen strategy")
    if payload["confidence_method"] not in cert.CONFIDENCE_METHODS:
        raise EvaluationArtifactError(
            f"{payload['confidence_method']!r} is not a frozen method")
    for field in ("repeat", "n_labels"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise EvaluationArtifactError(
                f"evaluation artifact field {field!r} must be a single integer")
    if payload["n_labels"] not in cert.N_GRID:
        raise EvaluationArtifactError(
            f"n_labels {payload['n_labels']!r} is off the frozen N grid")
    if not 0 <= payload["repeat"] < cert.R_SPLIT:
        raise EvaluationArtifactError(
            f"repeat {payload['repeat']!r} is outside 0..{cert.R_SPLIT - 1}")

    if payload["guarantee_tier"] != cert.tier_of(strategy):
        raise EvaluationArtifactError(
            f"{strategy!r} declares guarantee_tier "
            f"{payload['guarantee_tier']!r}, not {cert.tier_of(strategy)!r}")
    if payload["realized_field"] != cert.realized_outcome_field(strategy):
        raise EvaluationArtifactError(
            f"{strategy!r} must report its realized outcome under "
            f"{cert.realized_outcome_field(strategy)!r}, not "
            f"{payload['realized_field']!r}")
    cert.assert_state_allowed(strategy, payload["decision_state"])

    _assert_evaluation_status_consistency(payload)
    return payload


def _assert_evaluation_status_consistency(payload: dict) -> None:
    """The status, the adopted threshold and the outcome fields must agree."""
    status = payload["evaluation_status"]
    if status not in EVALUATION_STATUSES:
        raise EvaluationArtifactError(
            f"{status!r} is not a frozen evaluation status "
            f"{EVALUATION_STATUSES}")
    passed = cert.is_pass(payload["decision_state"])
    tau = payload["selected_tau"]

    if status == EVAL_STATUS_NO_THRESHOLD:
        if passed or tau is not None:
            raise EvaluationArtifactError(
                f"{status} requires a non-passing decision with no adopted "
                f"threshold; state={payload['decision_state']!r}, tau={tau!r}")
        for field in ("coverage_test", "ber_test", "risk_excess",
                      "realized_outcome"):
            if payload[field] is not None:
                raise EvaluationArtifactError(
                    f"{status} carries {field}={payload[field]!r}; no threshold "
                    "was adopted, so nothing was retained")
        return

    if not passed or tau is None:
        raise EvaluationArtifactError(
            f"{status} requires a passing decision with an adopted threshold; "
            f"state={payload['decision_state']!r}, tau={tau!r}")
    if payload["coverage_test"] is None:
        raise EvaluationArtifactError(
            f"{status} must record the coverage of the adopted threshold")
    if status == EVAL_STATUS_NOT_ESTIMABLE:
        if payload["ber_test"] is not None or payload["realized_outcome"] is not None:
            raise EvaluationArtifactError(
                "a retained-class-empty evaluation reports no BER and no "
                "realized outcome; it is never imputed")
    else:
        if payload["ber_test"] is None or payload["realized_outcome"] is None:
            raise EvaluationArtifactError(
                f"{EVAL_STATUS_COMPLETE} must carry a finite ber_test and a "
                "realized outcome")
        if payload["realized_outcome"] not in (0, 1):
            raise EvaluationArtifactError(
                f"realized_outcome {payload['realized_outcome']!r} is not 0/1")


def evaluation_status_of(payload: dict) -> str:
    """Derive the status from the evaluated cell — never taken as an input."""
    if payload.get("selected_tau") is None:
        return EVAL_STATUS_NO_THRESHOLD
    if payload.get("ber_test") is None:
        return EVAL_STATUS_NOT_ESTIMABLE
    return EVAL_STATUS_COMPLETE


def freeze_evaluation_artifact(payload: dict, path) -> dict:
    """validate -> canonical serialize -> write -> checksum -> freeze."""
    validated = validate_evaluation_artifact(payload)
    blob = sd.canonical_json(validated)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(blob)
    digest = sd.sha256_bytes(blob)
    if sd.sha256_file(p) != digest:
        raise EvaluationArtifactError(
            f"{p}: bytes on disk do not match the validated payload")
    return {"path": str(p), "sha256": digest, "payload": validated}


def load_and_validate_evaluation(path, expected_checksum: str = None) -> dict:
    p = Path(path)
    if not p.exists():
        raise EvaluationArtifactError(f"evaluation artifact {p} is not on disk")
    blob = p.read_bytes()
    digest = sd.sha256_bytes(blob)
    if expected_checksum is not None and digest != expected_checksum:
        raise EvaluationArtifactError(
            f"{p}: checksum {digest[:12]}... != expected "
            f"{expected_checksum[:12]}...")
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise EvaluationArtifactError(f"{p}: malformed JSON ({exc})") from exc
    validate_evaluation_artifact(payload)
    return {"path": str(p), "sha256": digest, "payload": payload}


# ---------------------------------------------------------------------------
# the authoritative expected evaluation-cell set — derived, never supplied
# ---------------------------------------------------------------------------
def descriptive_scope_of(strategy: str) -> dict:
    """The `N` and methods at which a strategy is evaluated on the target side.

    Derived from the **frozen 216-key CI registry**, not from a new constant: a
    strategy needs a target evaluation at exactly those `(N, method)` at which
    the registry requires a descriptive transported quantity for it. For
    `target_local` and `pooled_local_shrinkage` the formal and empirical blocks
    additionally require every `N` on the grid, so the union is taken.
    """
    import s2_aggregation as agg
    n_values, methods = set(), set()
    for key in agg.S2_REQUIRED_CI_KEYS:
        strategies = ({key.strategy} if key.contrast is None
                      else set(key.contrast))
        if strategy not in strategies:
            continue
        if key.n_labels is not None:
            n_values.add(int(key.n_labels))
        if key.method is not None:
            methods.add(key.method)
    if not n_values or not methods:
        raise EvaluationArtifactError(
            f"{strategy!r} has no registered target-side quantity; it needs no "
            "evaluation cell, which the registry must state explicitly")
    return {"n_values": tuple(sorted(n_values)),
            "methods": tuple(m for m in cert.CONFIDENCE_METHODS if m in methods)}


def derive_authoritative_expected_evaluation_cells(context) -> frozenset:
    """Every `(site, repeat, N, strategy, method)` a formal run must evaluate.

    Mechanical, from the authoritative production context and the frozen
    registry — **the caller supplies nothing**:

    * every label-consuming decision cell is evaluated exactly once, so the
      46,200 frozen decisions map one-to-one onto 46,200 evaluations;
    * each zero-label strategy is evaluated on the same `D_test(N, r)` at every
      `N` at which the registry requires a descriptive transported quantity for
      it, so the four strategies are compared on one common observable.
    """
    from s2_production import (ProductionAuthorityError, ProductionS2Context,
                               derive_authoritative_expected_decision_cells)
    if not isinstance(context, ProductionS2Context):
        raise ProductionAuthorityError(
            "the expected evaluation set is derived from a ProductionS2Context")
    registry = context.registry
    cells = set(derive_authoritative_expected_decision_cells(registry))
    for strategy in cert.ZERO_LABEL_STRATEGIES:
        scope = descriptive_scope_of(strategy)
        for (site, repeat, n_labels) in registry.cells:
            if n_labels not in scope["n_values"]:
                continue
            for method in scope["methods"]:
                cells.add(evaluation_cell_key(site, repeat, n_labels, strategy,
                                              method))
    return frozenset(cells)


def expected_evaluation_decomposition(expected) -> dict:
    """Counts by `N`, by strategy and by information budget. Reported, not read."""
    by_n, by_strategy, site_n = {}, {}, set()
    for (site, repeat, n_labels, strategy, method) in expected:
        by_n[n_labels] = by_n.get(n_labels, 0) + 1
        by_strategy[strategy] = by_strategy.get(strategy, 0) + 1
        site_n.add((site, n_labels))
        if method not in cert.CONFIDENCE_METHODS:
            raise EvaluationArtifactError(f"unknown method {method!r}")
        if strategy not in cert.STRATEGIES:
            raise EvaluationArtifactError(f"unknown strategy {strategy!r}")
    label_consuming = sum(by_strategy.get(s, 0)
                          for s in cert.LABEL_CONSUMING_STRATEGIES)
    zero_label = sum(by_strategy.get(s, 0) for s in cert.ZERO_LABEL_STRATEGIES)
    return {"total": len(expected), "by_n": dict(sorted(by_n.items())),
            "by_strategy": dict(sorted(by_strategy.items())),
            "site_n_cells": len(site_n),
            "label_consuming": label_consuming, "zero_label": zero_label}


def assert_expected_evaluation_decomposition(expected) -> dict:
    """The evaluation set must decompose exactly as the frozen rules imply."""
    from s2_production import (PRODUCTION_EXPECTED_DECISIONS,
                               ProductionAuthorityError)
    out = expected_evaluation_decomposition(expected)
    if out["label_consuming"] != PRODUCTION_EXPECTED_DECISIONS:
        raise ProductionAuthorityError(
            f"{out['label_consuming']} label-consuming evaluation cells, not "
            f"{PRODUCTION_EXPECTED_DECISIONS}; every frozen decision is "
            "evaluated exactly once")
    if out["total"] != PRODUCTION_EXPECTED_EVALUATIONS:
        raise ProductionAuthorityError(
            f"{out['total']} expected evaluation cells, not "
            f"{PRODUCTION_EXPECTED_EVALUATIONS}")
    if out["by_n"] != PRODUCTION_EVALUATIONS_BY_N:
        raise ProductionAuthorityError(
            f"per-N evaluation counts are {out['by_n']}, not "
            f"{PRODUCTION_EVALUATIONS_BY_N}")
    if out["by_strategy"] != PRODUCTION_EVALUATIONS_BY_STRATEGY:
        raise ProductionAuthorityError(
            f"per-strategy evaluation counts are {out['by_strategy']}, not "
            f"{PRODUCTION_EVALUATIONS_BY_STRATEGY}")
    return out


# ---------------------------------------------------------------------------
# exact production invariants (derived from the frozen rules, then hard-asserted)
# ---------------------------------------------------------------------------
#: 77 site-N cells x 100 repeats x 3 methods x 2 label-consuming strategies.
_LABEL_CONSUMING_EVALUATIONS = 46_200
#: (22 + 21 + 11) site-N cells at N >= 10 x 100 x 3 methods x 2 zero-label
#: strategies — the descriptive transported block of the frozen 216 registry.
_ZERO_LABEL_EVALUATIONS = 32_400

PRODUCTION_EXPECTED_EVALUATIONS = 78_600
PRODUCTION_EVALUATIONS_BY_N = {5: 13_800, 10: 26_400, 20: 25_200, 40: 13_200}
PRODUCTION_EVALUATIONS_BY_STRATEGY = {
    "pooled_local_shrinkage": 23_100,
    "pooled_source": 16_200,
    "target_local": 23_100,
    "target_unlabeled_weighted": 16_200,
}

assert _LABEL_CONSUMING_EVALUATIONS + _ZERO_LABEL_EVALUATIONS == \
    PRODUCTION_EXPECTED_EVALUATIONS
assert sum(PRODUCTION_EVALUATIONS_BY_N.values()) == PRODUCTION_EXPECTED_EVALUATIONS
assert sum(PRODUCTION_EVALUATIONS_BY_STRATEGY.values()) == \
    PRODUCTION_EXPECTED_EVALUATIONS


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------
class EvaluationArtifactStore:
    """A filesystem-backed store of frozen phase-2 evaluation artifacts.

    Not a container the caller fills: it is a **view of a directory**. Its
    constructor takes a path, its scan re-reads and re-validates every file, and
    it holds no evaluation value of its own.
    """

    __slots__ = ("_root",)

    def __init__(self, root) -> None:
        object.__setattr__(self, "_root", Path(root))

    def __setattr__(self, name, value):
        raise EvaluationArtifactError("the evaluation store is immutable")

    @classmethod
    def for_context(cls, context) -> "EvaluationArtifactStore":
        """The one store of a production run, resolved from its context."""
        return resolve_authoritative_evaluation_store(context)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, key: tuple) -> Path:
        return self._root / _artifact_filename(key)

    def freeze(self, payload: dict) -> dict:
        key = evaluation_cell_key(payload["site"], payload["repeat"],
                                  payload["n_labels"], payload["strategy"],
                                  payload["confidence_method"])
        path = self.path_for(key)
        if path.exists():
            raise EvaluationArtifactError(
                f"evaluation cell {key} is already frozen; a cell is evaluated "
                "once")
        return freeze_evaluation_artifact(payload, path)

    def paths(self) -> list:
        if not self._root.exists():
            return []
        return sorted(self._root.glob(f"{EVALUATION_PREFIX}*.json"))


def resolve_authoritative_evaluation_store(context) -> EvaluationArtifactStore:
    """**The only way production code learns where evaluation artifacts live.**

    Takes a `ProductionS2Context` and nothing else. There is no `path`, no
    `root`, no `store`, no `evaluation_dir`, no environment override and no
    configuration key: the root is re-derived from the context's authoritative
    registry and its identity re-checked on every call, so
    `EvaluationArtifactStore("/tmp/fake_eval_store")` has nowhere to be passed
    in — the production entry points do not take a store at all.
    """
    from s2_production import ProductionAuthorityError, ProductionS2Context
    if not isinstance(context, ProductionS2Context):
        raise ProductionAuthorityError(
            "the production evaluation store is resolved from a "
            "ProductionS2Context, never from a caller-supplied path")
    root = context.assert_evaluation_root_authoritative()
    if root.name != EVALUATION_STORE:
        raise ProductionAuthorityError(
            f"the authoritative evaluation root is {root}, which is not the "
            f"protocol's {EVALUATION_STORE!r} directory")
    return EvaluationArtifactStore(root)


def scan_current_evaluation_store(context) -> dict:
    """Re-scan and re-validate **every** evaluation artifact on disk.

    Each artifact is bound back to the authoritative decision store: the cell's
    frozen decision is re-read from disk, its checksum recomputed, and the
    artifact required to name that checksum, that split checksum, that protocol
    version, that guarantee tier, that state and that threshold. An artifact
    that declares a decision checksum of its own is never trusted on that
    declaration.

    The directory scanned is resolved here, from the context. A caller cannot
    name it (§17.24).
    """
    import s2_firewall as fw
    from s2_production import ProductionAuthorityError

    store = resolve_authoritative_evaluation_store(context)
    decisions = _authoritative_decision_index(context)

    manifest, cells, records = {}, set(), {}
    for path in store.paths():
        frozen = load_and_validate_evaluation(path)
        payload = frozen["payload"]
        key = evaluation_cell_key(payload["site"], payload["repeat"],
                                  payload["n_labels"], payload["strategy"],
                                  payload["confidence_method"])
        if path.name != _artifact_filename(key):
            raise EvaluationArtifactError(
                f"{path.name} does not name the cell it contains ({key})")
        if key in cells:
            raise EvaluationArtifactError(
                f"duplicate evaluation artifact for {key}")

        decision = decisions.get(_decision_key_for(key))
        if decision is None:
            raise ProductionAuthorityError(
                f"evaluation {key} has no frozen decision in the authoritative "
                "decision store")
        _assert_binds_decision(key, payload, decision, context)

        cells.add(key)
        records[key] = payload
        manifest["|".join(map(str, key))] = frozen["sha256"]

    return {"n_evaluated": len(cells), "cells": frozenset(cells),
            "records": records, "evaluation_manifest": manifest,
            "evaluation_manifest_checksum": sd.sha256_bytes(
                sd.canonical_json(manifest)),
            "n_decisions_bound": len(decisions),
            "evaluation_root": str(store.root),
            "evaluation_root_identity": context.snapshot[
                "evaluation_root_identity"]}


def _decision_key_for(eval_key: tuple) -> tuple:
    """The decision a given evaluation cell must bind to.

    A zero-label strategy decides once per `(site, method)` and is then
    evaluated on many `(repeat, N)` splits, so its evaluation cells all bind to
    the same decision artifact — which is exactly the §3.1 invariant that its
    decision may not vary with the split.
    """
    site, repeat, n_labels, strategy, method = eval_key
    if strategy in cert.ZERO_LABEL_STRATEGIES:
        return ("ZERO", site, strategy, method)
    return ("CELL", site, repeat, n_labels, strategy, method)


def _authoritative_decision_index(context) -> dict:
    """Re-read the decision store — both phases — and index it by decision key.

    The checksums come from the files that are on disk right now, recomputed
    here; nothing is taken from a release artifact, a manifest field or an
    evaluation artifact's own claim.
    """
    import s2_firewall as fw

    index = {}
    fw.scan_current_decision_store(context)      # re-validates every artifact
    for path in sorted((Path(context.workdir) / fw.DECISION_STORE)
                       .glob("decision__*.json")):
        frozen = sd.load_and_validate(path, zero_label=False)
        payload = frozen["payload"]
        key = ("CELL", str(payload["site"]), int(payload["repeat"]),
               int(payload["n_labels"]), str(payload["strategy"]),
               str(payload["method"]))
        index[key] = {"sha256": frozen["sha256"], "payload": payload}
    for path in sorted(Path(context.workdir).glob("zero__*.json")):
        frozen = sd.load_and_validate(path, zero_label=True)
        payload = frozen["payload"]
        key = ("ZERO", str(payload["site"]), str(payload["strategy"]),
               str(payload["method"]))
        index[key] = {"sha256": frozen["sha256"], "payload": payload}
    return index


def _assert_binds_decision(key: tuple, payload: dict, decision: dict,
                           context) -> None:
    """Cell key, decision checksum, split checksum, tier and threshold, exactly."""
    from s2_production import ProductionAuthorityError

    site, repeat, n_labels, strategy, method = key
    frozen = decision["payload"]
    if payload["decision_checksum"] != decision["sha256"]:
        raise ProductionAuthorityError(
            f"evaluation {key} names decision checksum "
            f"{str(payload['decision_checksum'])[:12]}..., but the store holds "
            f"{decision['sha256'][:12]}...")
    registered = context.registry.cell(site, repeat, n_labels)
    if payload["split_checksum"] != registered["cell_checksum"]:
        raise ProductionAuthorityError(
            f"evaluation {key} was taken on a split the registry does not hold")
    if frozen.get("split_checksum") not in (None, registered["cell_checksum"]):
        raise ProductionAuthorityError(
            f"evaluation {key}: its decision was taken on another split")
    if payload["protocol_version"] != frozen["protocol_version"]:
        raise ProductionAuthorityError(
            f"evaluation {key} and its decision were produced under different "
            "protocol versions")
    if payload["strategy"] != frozen["strategy"] or \
            payload["confidence_method"] != frozen["method"] or \
            str(payload["site"]) != str(frozen["site"]):
        raise ProductionAuthorityError(
            f"evaluation {key} does not name its decision's cell")
    if payload["guarantee_tier"] != frozen["guarantee_tier"]:
        raise ProductionAuthorityError(
            f"evaluation {key} declares tier {payload['guarantee_tier']!r}, its "
            f"decision {frozen['guarantee_tier']!r}")
    if payload["decision_state"] != frozen["state"]:
        raise ProductionAuthorityError(
            f"evaluation {key} reports state {payload['decision_state']!r}; the "
            f"frozen decision is {frozen['state']!r}")
    for field, decided in (("selected_q", frozen["selected_q"]),
                           ("selected_tau", frozen["selected_tau"])):
        if payload[field] != decided:
            raise ProductionAuthorityError(
                f"evaluation {key} used {field}={payload[field]!r}; the frozen "
                f"decision selected {decided!r}. An evaluation reads the "
                "threshold, it never changes it")


def assert_evaluation_store_is_complete(context) -> dict:
    """The actual evaluation set must **be** the authoritative expected set.

    Exact set equality. One missing cell, one extra, one duplicate, one
    malformed artifact, one wrong decision checksum, one wrong split checksum
    or one wrong protocol version each fail, and none of them produces a
    partial aggregation. "The actual set" means the one in the authoritative
    root — there is no parameter naming a different directory.
    """
    from s2_production import ProductionAuthorityError

    expected = derive_authoritative_expected_evaluation_cells(context)
    assert_expected_evaluation_decomposition(expected)
    scan = scan_current_evaluation_store(context)
    actual = scan["cells"]
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ProductionAuthorityError(
            f"the evaluation store is not the authoritative expected set: "
            f"{len(missing)} missing, {len(extra)} unexpected (store holds "
            f"{len(actual)} of {len(expected)}); e.g. missing "
            f"{sorted(missing)[:2]}, extra {sorted(extra)[:2]}")
    scan["expected"] = expected
    return scan


def build_evaluation_manifest(context) -> dict:
    """Recompute the evaluation manifest from the artifacts that are on disk.

    There is no parameter through which a manifest checksum can be supplied and
    none through which a different *store* can be: the value is a function of
    the files in the authoritative root, and a caller who wants a different
    checksum has to produce different artifacts there, which then fail their
    decision binding.
    """
    scan = assert_evaluation_store_is_complete(context)
    return {
        "evaluation_manifest": dict(scan["evaluation_manifest"]),
        "evaluation_manifest_checksum": scan["evaluation_manifest_checksum"],
        "n_evaluated": scan["n_evaluated"],
        "composition": expected_evaluation_decomposition(scan["cells"]),
        "evaluation_root": scan["evaluation_root"],
        "evaluation_root_identity": scan["evaluation_root_identity"],
    }


def assert_zero_label_store_is_complete(context) -> dict:
    """The 138 zero-label reference decisions, re-read and re-validated."""
    from s2_production import (PRODUCTION_ZERO_LABEL_DECISIONS,
                               ProductionAuthorityError)
    seen, manifest = set(), {}
    for path in sorted(Path(context.workdir).glob("zero__*.json")):
        frozen = sd.load_and_validate(path, zero_label=True)
        payload = frozen["payload"]
        key = (str(payload["site"]), str(payload["strategy"]),
               str(payload["method"]))
        if key in seen:
            raise ProductionAuthorityError(
                f"duplicate zero-label decision artifact for {key}")
        seen.add(key)
        manifest["|".join(key)] = frozen["sha256"]
    expected = {(site, strategy, method)
                for site in contracts.FROZEN_SITES
                for strategy in cert.ZERO_LABEL_STRATEGIES
                for method in cert.CONFIDENCE_METHODS}
    if seen != expected:
        raise ProductionAuthorityError(
            f"the zero-label decision store is not the authoritative set: "
            f"{len(expected - seen)} missing, {len(seen - expected)} unexpected")
    if len(seen) != PRODUCTION_ZERO_LABEL_DECISIONS:
        raise ProductionAuthorityError(
            f"{len(seen)} zero-label decisions, not "
            f"{PRODUCTION_ZERO_LABEL_DECISIONS}")
    return {"n_zero_label": len(seen), "zero_label_manifest": manifest,
            "zero_label_manifest_checksum": sd.sha256_bytes(
                sd.canonical_json(manifest))}


def load_zero_label_decisions(context) -> dict:
    """`{(site, strategy, method): payload}` for the 138 frozen references."""
    assert_zero_label_store_is_complete(context)
    out = {}
    for path in sorted(Path(context.workdir).glob("zero__*.json")):
        payload = sd.load_and_validate(path, zero_label=True)["payload"]
        out[(str(payload["site"]), str(payload["strategy"]),
             str(payload["method"]))] = payload
    return out
