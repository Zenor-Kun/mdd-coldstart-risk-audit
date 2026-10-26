"""The deterministic RQ3 result store — §16.1 / §16.2, frozen at Stage E.2a.

Six record kinds, each in its own directory, each one canonical-JSON file per
key, each checksummed:

    fits/         one model fit, one cell, one node count
    cv/           one model's LOSO-site fold records, one cell, one node count
    validation/   one (population, cell, block) unit's 40-vs-60 verdict (§9.6.5)
    cells/        one cell's §15 registered outputs
    families/     one Holm family of six pre-registered slots
    PROVENANCE.json / RESULT_CHECKSUMS.sha256 / COMPLETION.json

Determinism is a property of the bytes, not of a convention: every payload is
serialized with sorted keys, fixed separators and round-tripping float `repr`,
so re-running an identical computation produces byte-identical files and an
identical checksum ledger.

### Two stores, not one store with a flag (§16.1.2)

`FormalRQ3Store` is **hard-bound to `results/rq3/`**. It takes no root from any
caller — no argument, no environment variable, no configuration key — because
the property being protected is that *no argument value* can turn a test run
into an authoritative one (H23). It refuses to start at all if `results/rq3/`
already exists: stale roots fail closed, and archiving a previous attempt is a
separate operator action the runner may never perform for itself (H24).

`NonFormalRQ3Store` is a **different API**, used by smoke, unit tests and
synthetic execution. It refuses `results/rq3/` and every path inside it, and it
is *structurally* incapable of writing a formal completion marker: the barrier
that writes `COMPLETION.json` exists only on the formal class, and the
non-formal writer additionally refuses the marker's filename outright.

### Completion is not a claim a runner may make (§16.2)

`FormalRQ3Store.complete_run` runs the seven frozen steps in order, gated by the
frozen `completion_marker_is_permitted`. In particular it **re-opens and
re-reads every artifact and the ledger from disk through fresh handles**,
recomputes each digest from the re-read bytes, re-derives the expected key set
from the frozen registries, and re-verifies the recorded input and code
provenance — before writing `COMPLETION.json` atomically as the terminal
artifact. In-memory state is never accepted as evidence, and any failure in
steps 4-6 leaves the marker absent (H25).

### Prior stages are unwritable

`assert_writable_root` refuses any store root that is, contains or is contained
by a frozen artifact directory, and every write is resolved and required to stay
inside the store root, so no RQ3 execution path can overwrite
`results/rq1_s0/`, `results/rq1_s1/`, `results/rq2_s2/`, `results/rq2_s2_freeze/`,
`results/data_contract/`, `results/gate1*/`, `docs/` or `data/` (H13).
"""

from __future__ import annotations

import hashlib
import json
import os
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import rq3_data as data
import rq3_models as models
import rq3_protocol as proto

STORE_VERSION = "RQ3_STORE/2"

#: §16.1 — the authoritative root is a **constant of the protocol**, not a
#: parameter, not a default and not a suggestion. It is read from the freeze.
AUTHORITATIVE_STORE = proto.AUTHORITATIVE_RESULT_ROOT

FITS_DIR = "fits"
CV_DIR = "cv"
VALIDATION_DIR = "validation"
CELLS_DIR = "cells"
FAMILIES_DIR = "families"
RECORD_DIRS = (FITS_DIR, CV_DIR, VALIDATION_DIR, CELLS_DIR, FAMILIES_DIR)

PROVENANCE_FILE = "PROVENANCE.json"
CHECKSUM_FILE = "RESULT_CHECKSUMS.sha256"
COMPLETION_FILE = proto.COMPLETION_MARKER

#: Frozen upstream that RQ3 may read and may never write (H13).
PROTECTED_DIRS = (
    "results/rq1_s0", "results/rq1_s1", "results/rq2_s2", "results/rq2_s2_freeze",
    "results/data_contract", "results/gate1", "results/gate1_formal_permutation",
    "docs", "data", "scripts", "src", "tests",
)


class RQ3StoreError(AssertionError):
    """Structural corruption of the result store, or an illegal write target."""


class StaleResultRootError(RQ3StoreError):
    """§16.1.3 — a formal run may never begin with `results/rq3/` present."""


class CompletionBarrierError(RQ3StoreError):
    """§16.2 — a step of the completion barrier failed; the marker stays absent."""


class StoreConstructionError(RQ3StoreError):
    """§16.1.2 — a store was built outside its approved factory."""


# ---------------------------------------------------------------------------
# §9.6 / §16.2 — the node registry is frozen, not a parameter
# ---------------------------------------------------------------------------

#: Exactly two node counts exist, in exactly these roles: **40 authoritative**
#: (the science) and **60 numerical validation**. There is no registered K20 and
#: no registered K80.
#:
#: These are written as **literals inside the function below**, not as a module
#: tuple. A module-level `FROZEN_NODE_REGISTRY` is rebindable, and rebinding it
#: to `(20, 40, 60)`, `(40, 60, 80)`, `(60, 40)`, `(40, 40)` or `(40,)` changed
#: execution, the expected artifact set and the persisted provenance at once.
#: Every formal boundary now re-derives the roles here *and* re-asserts the
#: literals locally, so there is no single place to rebind.

def frozen_node_roles() -> tuple:
    """`(40, 60)` — authoritative first, validation second, asserted from literals."""
    authoritative, validation = 40, 60
    if (int(proto.AUTHORITATIVE_NODES) != authoritative
            or int(proto.GH_NODES_PRIMARY) != authoritative
            or int(proto.GH_NODES_VALIDATION) != validation):
        raise RQ3StoreError(
            f"§9.6: the frozen node roles are {proto.AUTHORITATIVE_NODES} "
            f"authoritative / {proto.GH_NODES_VALIDATION} validation, not 40 / 60"
        )
    roles = (authoritative, validation)
    if roles != (40, 60) or len(roles) != 2 or roles[0] == roles[1]:
        raise RQ3StoreError(f"§9.6: the node role order is {roles}, not (40, 60)")
    return roles


def assert_frozen_node_roles(roles) -> tuple:
    """Re-assert `(40, 60)` against the literals at a formal boundary."""
    checked = tuple(int(k) for k in roles)
    if checked != (40, 60):
        raise RQ3StoreError(
            f"§9.6: node roles {checked} are not the frozen (40, 60); 40 is "
            f"authoritative and 60 is the independent numerical validation"
        )
    return checked


def node_registry() -> tuple:
    """Read-only convenience. **Execution does not trust this name** — every
    formal boundary calls `frozen_node_roles()` and re-asserts the literals."""
    return assert_frozen_node_roles(frozen_node_roles())


# ---------------------------------------------------------------------------
# canonical serialization
# ---------------------------------------------------------------------------

def canonical_bytes(payload) -> bytes:
    """Deterministic JSON: sorted keys, fixed separators, round-tripping floats."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=_encode)
    return text.encode("utf-8")


def _encode(value):
    import numpy as np
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return [_encode(v) for v in value.tolist()]
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not serializable in an RQ3 record")


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fsync_directory(directory: Path) -> None:
    """§16.2 step 3 — a rename is only durable once its directory is synced."""
    handle = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _atomic_write(target: Path, blob: bytes) -> str:
    """Write, `fsync`, then `rename` into place, and sync the directory too."""
    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    _fsync_directory(target.parent)
    return sha256_bytes(blob)


# ---------------------------------------------------------------------------
# §16.1 — provenance the store computes for itself
# ---------------------------------------------------------------------------

def input_provenance() -> dict:
    """Re-hash the §1 consumed artifacts at run time (H1)."""
    digests = {}
    for relative, expected in sorted(proto.FROZEN_INPUTS.items()):
        path = proto.ROOT / relative
        if not path.exists():
            raise RQ3StoreError(f"H1: missing consumed artifact {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise RQ3StoreError(f"H1: checksum mismatch for {relative}: {digest}")
        digests[relative] = digest
    return digests


def code_provenance() -> dict:
    """Hash the implementation itself, so a record names the code that made it."""
    modules = (
        "src/rq3_protocol.py", "src/rq3_data.py", "src/rq3_glmm.py",
        "src/rq3_models.py", "src/rq3_inference.py", "src/rq3_store.py",
        "src/rq3_execution.py",
        "scripts/validate_rq3_protocol_freeze.py", "docs/RQ3_PROTOCOL.md",
    )
    return {name: sha256_file(proto.ROOT / name) for name in modules}


# ---------------------------------------------------------------------------
# §16.1 — provenance fields the store owns and a caller may never supply
# ---------------------------------------------------------------------------

#: Frozen identity. A caller passing any of these is refused outright, and the
#: store writes its own values *after* the caller's metadata is merged, so
#: neither ordering nor a bypassed check can let run metadata redefine what a
#: result claims to be. `inputs` and `code` are included: provenance a runner
#: hands in is a claim, provenance the store computes is evidence.
RESERVED_PROVENANCE_FIELDS = (
    "store_version",
    "protocol_version", "protocol_tag", "protocol_commit",
    "protocol_doc_sha256", "protocol_validator_sha256",
    "mechanism_closure_sha256",
    "authoritative",
    "quadrature_rule", "mode_solver", "gradient_rule",
    "authoritative_nodes", "validation_nodes", "node_registry",
    "n_validation_units",
    "inputs", "code",
)


def _reject_reserved(provenance: dict) -> dict:
    if not isinstance(provenance, dict):
        raise RQ3StoreError("provenance must be a mapping")
    offending = sorted(set(provenance) & set(RESERVED_PROVENANCE_FIELDS))
    if offending:
        raise RQ3StoreError(
            f"§16.1: caller provenance may not supply frozen identity fields "
            f"{offending}; they are written by the store from the freeze"
        )
    return dict(provenance)


def frozen_identity(authority, *, authoritative: bool,
                    _assert_roles=assert_frozen_node_roles) -> dict:
    """Every frozen identity field, **derived from a freshly verified authority**.

    `authority` is the mapping returned by the sealed fresh-authority verifier
    `rq3_protocol.sealed_verified_authority()`,
    which has just re-read and re-hashed the protocol document and the mechanism
    source, rebuilt and re-bound the mechanism dependency closure, and (for a
    formal run) resolved the literal tag ref and the freeze commit through git.
    Nothing here is copied out of a mutable runtime constant, so what gets
    persisted is what was checked.

    Called when the store is created and again, independently, at §16.2 step 5
    from a *second* verification — so the bytes on disk are compared against an
    authority established after they were written, not against the object that
    wrote them.

    Every *frozen identity* field is read out of `authority` — the literal tag
    ref, the freeze commit, the two file hashes, the mechanism closure digest,
    the quadrature/solver/gradient identities and the node roles — so no runtime
    constant of this module can change what a persisted record claims. `inputs`
    and `code` are different in kind: they are file digests recomputed here, and
    the §16.2 barrier's job is to recompute them again and compare.
    """
    roles = _assert_roles(authority["node_registry"])
    if (int(authority["authoritative_nodes"]) != roles[0]
            or int(authority["validation_nodes"]) != roles[1]):
        raise RQ3StoreError("§9.6: the verified authority's node roles are inconsistent")

    identity = {
        "store_version": STORE_VERSION,
        "protocol_version": authority["protocol_version"],
        "protocol_tag": authority["protocol_tag"],
        "protocol_commit": authority["protocol_commit"],
        "protocol_doc_sha256": authority["protocol_doc_sha256"],
        "protocol_validator_sha256": authority["protocol_validator_sha256"],
        "mechanism_closure_sha256": authority["mechanism_closure_sha256"],
        "authoritative": bool(authoritative),
        "quadrature_rule": authority["quadrature_rule"],
        "mode_solver": authority["mode_solver"],
        "gradient_rule": authority["gradient_rule"],
        "authoritative_nodes": int(roles[0]),
        "validation_nodes": int(roles[1]),
        "node_registry": [int(roles[0]), int(roles[1])],
        "n_validation_units": int(authority["n_validation_units"]),
        "code": code_provenance(),
    }
    if authoritative:
        # H1 — the consumed artifacts, re-hashed by the store itself. A
        # non-formal run reads no frozen input and therefore records none.
        identity["inputs"] = input_provenance()
    return identity


def authority_digest(authority, _sha256=sha256_bytes, _canonical=canonical_bytes) -> str:
    """A stable digest of a verified authority, minus its git-coverage flag."""
    payload = {k: v for k, v in dict(authority).items() if k != "git_verified"}
    return _sha256(_canonical(payload))


# ---------------------------------------------------------------------------
# record keys — derived from the frozen registries, never supplied
# ---------------------------------------------------------------------------

def _slug(*parts) -> str:
    for part in parts:
        text = str(part)
        if not text or any(c in text for c in "/\\.: "):
            raise RQ3StoreError(f"illegal key component {part!r}")
    return "__".join(str(p) for p in parts)


def fit_key(population, operating_point, method, model, n_nodes) -> str:
    return _slug(population, operating_point, method, model, f"K{int(n_nodes)}")


def cv_key(population, operating_point, method, model, n_nodes) -> str:
    return _slug(population, operating_point, method, model, f"K{int(n_nodes)}")


def validation_key(population, operating_point, method, block) -> str:
    return _slug(population, operating_point, method, block)


def cell_key(population, operating_point, method) -> str:
    return _slug(population, operating_point, method)


def family_key(population, block) -> str:
    return _slug(population, block)


def expected_keys() -> dict:
    """Re-derive the complete expected key set from the frozen registries.

    **It takes no arguments.** Populations, cells, blocks, models and node
    counts all come from the freeze, so a runner can neither declare a smaller
    expectation and then satisfy it, nor declare a larger one — passing
    `(40, 60, 80)` used to enlarge this set and make an unregistered K80 record
    "required"; there is now nothing to pass.
    """
    nodes = assert_frozen_node_roles(frozen_node_roles())
    if nodes != (40, 60):                      # literal, local, re-asserted here
        raise RQ3StoreError(f"§9.6: the expected key set requires (40, 60), got {nodes}")

    fits, cvs, validations, cells, families = set(), set(), set(), set(), set()
    for population in data.POPULATIONS:
        fitted = models.models_to_fit(population)
        cross_validated = models.models_to_cross_validate(population)
        for operating_point, method in proto.CELL_REGISTRY:
            cells.add(cell_key(population, operating_point, method))
            for n_nodes in nodes:
                for spec in fitted:
                    fits.add(fit_key(population, operating_point, method,
                                     spec.name, n_nodes))
                for spec in cross_validated:
                    cvs.add(cv_key(population, operating_point, method,
                                   spec.name, n_nodes))
            for block in models.blocks_for(population):
                validations.add(validation_key(population, operating_point,
                                               method, block.block))
        for block in models.blocks_for(population):
            families.add(family_key(population, block.block))

    if len(validations) != proto.N_VALIDATION_UNITS:
        raise RQ3StoreError(
            f"H27: {len(validations)} validation units, §9.6.5 registers "
            f"{proto.N_VALIDATION_UNITS}"
        )
    return {
        FITS_DIR: frozenset(fits),
        CV_DIR: frozenset(cvs),
        VALIDATION_DIR: frozenset(validations),
        CELLS_DIR: frozenset(cells),
        FAMILIES_DIR: frozenset(families),
    }


# ---------------------------------------------------------------------------
# write guard
# ---------------------------------------------------------------------------

def authoritative_root() -> Path:
    """§16.1 — the one formal root, resolved from the frozen constant."""
    return (proto.ROOT / AUTHORITATIVE_STORE).resolve()


def assert_writable_root(root) -> Path:
    """Refuse a store root that could touch a frozen artifact directory (H13)."""
    resolved = Path(root).resolve()
    for protected in PROTECTED_DIRS:
        guarded = (proto.ROOT / protected).resolve()
        if resolved == guarded:
            raise RQ3StoreError(f"H13: {protected} is frozen and may never be a store root")
        if guarded in resolved.parents:
            raise RQ3StoreError(f"H13: a store root may not live inside {protected}")
        if resolved in guarded.parents:
            raise RQ3StoreError(f"H13: a store root may not contain {protected}")
    if resolved == proto.ROOT.resolve():
        raise RQ3StoreError("H13: the repository root may never be a store root")
    return resolved


# ---------------------------------------------------------------------------
# construction capability — a store exists only if a factory made it
# ---------------------------------------------------------------------------

#: The two approved store types, and nothing else.
APPROVED_STORE_TYPES = ("FormalRQ3Store", "NonFormalRQ3Store")

#: **Inert labels** (Stage E.2b.5). These two names exist so that a caller, a log
#: line or a test can *spell* the two store kinds. Nothing on a formal path reads
#: them, and nothing on a formal path compares against them.
#:
#: They used to be the values `store_kind` returned, resolved as module globals
#: at call time, and every consumer then tested *positively* for formality
#: (`kind == FORMAL`, `kind == "formal"`). That structure was fail-open: setting
#: `rq3_store.FORMAL = float("nan")` made `store_kind` return a value equal to
#: nothing at all, so `execute` skipped the pre-fit authorization, `_write`
#: skipped the per-write gate and `write_checksums` skipped its gate — a
#: scientific record and a checksum ledger reached disk under an unverified
#: authority, and only `complete_run` failed, far too late.
#:
#: The authority is now `store_is_formal`, a **bool** decided inside the factory
#: closure from the sealed class objects and the factory attestation, and every
#: gate is entered unless the store is *proven* non-formal. Rebinding either name
#: below changes no control flow anywhere.
FORMAL = "formal"
NON_FORMAL = "non_formal"


def _build_store_authority(
        *,
        _verified_authority=proto.sealed_verified_authority,
        _assert_roles=assert_frozen_node_roles,
        _node_roles=frozen_node_roles,
        _authority_digest=authority_digest,
        _frozen_identity=frozen_identity,
        _reject=_reject_reserved,
        _writable_root=assert_writable_root,
):
    """Factory-owned authority. Nothing that *grants* it is a module attribute.

    The previous design exposed `_issue_construction_token` and two mutable
    module registries. Both were mintable: calling the issuer produced a valid
    capability, and inserting an `object.__new__` forgery into the live registry
    produced formal dispatch — neither went anywhere near `create_authoritative`,
    so neither was ever checked against the freeze.

    Here the capability (`_token`) and the attestation table (`_attested`) are
    **closure-local and never returned**. What is exported is a checker, a
    dispatcher, the authorization gate and the real factories — every one of
    which verifies rather than grants. There is no exported call that mints
    authority, and no exported mutable object whose mutation confers it.

    Every helper the closure depends on is captured **at definition time** as a
    keyword default (Stage E.2b.3 §5). Resolving them as module globals at call
    time made the last gate before a formal write a rebindable name: replacing
    `rq3_store.store_kind` with a permissive function let a forged exact-type
    object reach `_atomic_write`, and replacing
    `rq3_store.assert_formal_execution_authorized` with a no-op let a drifted
    formal run reach outcome construction. Neither substitution is visible from
    inside this closure any more.

    `_verified_authority` is the **sealed** fresh-authority verifier
    (Stage E.2b.4). Calling `proto.verified_authority()` here left the last
    verification before a formal write behind one more rebindable name — and
    behind five more inside it: replacing that alias, or
    `proto.assert_protocol_binding`, with a callable returning the *previously
    attested* authority made a stale answer indistinguishable from a fresh
    verification, and a scientific record reached disk after real mechanism
    drift. The sealed verifier resolves its whole dependency closure privately
    and refuses a substituted one outright.

    The **kind decision** is closure-local too (Stage E.2b.5). It used to be
    reported as the value of the module global `FORMAL`, which made the formal
    branch of every consumer skippable by ordinary rebinding; see the note on
    `FORMAL` above. `store_is_formal` returns a bool decided here, from the
    sealed class objects and the attestation table, and a store that is not
    provably one of the two approved kinds raises rather than returning either
    answer.
    """
    _token = object()                      # never escapes this closure
    _attested: dict = {}                   # id(store) -> (weakref, cls, digest)
    _sealed_classes: dict = {}

    #: The two labels, as **closure locals**. `store_kind` reports these; no
    #: control flow compares against them, and no module attribute defines them.
    _FORMAL_LABEL = "formal"
    _NON_FORMAL_LABEL = "non_formal"

    def _seal_store_classes(formal_cls, non_formal_cls) -> None:
        """Bind the two approved classes once, so dispatch cannot be re-pointed."""
        if _sealed_classes:
            raise StoreConstructionError("§16.1.2: the store classes are already sealed")
        _sealed_classes["formal"] = formal_cls
        _sealed_classes["non_formal"] = non_formal_cls

    def _authorize(candidate, cls) -> None:
        """Called by `__post_init__`. Checks a capability; cannot produce one."""
        if candidate is not _token:
            raise StoreConstructionError(
                f"§16.1.2: {cls.__name__} is constructible only through its "
                f"approved factory. Direct construction, subclassing, forged "
                f"constructor state and caller-minted capabilities are refused."
            )

    def _attest(store, digest: str) -> None:
        _attested[id(store)] = (weakref.ref(store), type(store), digest)

    def _attestation_of(store):
        record = _attested.get(id(store))
        if record is None:
            return None
        reference, cls, digest = record
        if reference() is not store or type(store) is not cls:
            return None
        return (cls, digest)

    def _authoritative_root() -> Path:
        root = (proto.ROOT / "results/rq3").resolve()
        if proto.AUTHORITATIVE_RESULT_ROOT != "results/rq3" \
                or not proto.is_authoritative_root("results/rq3"):
            raise StoreConstructionError(
                "§16.1: the frozen authoritative root is not 'results/rq3'")
        return root

    # -- dispatch -----------------------------------------------------------

    def store_is_formal(store) -> bool:
        """§16.1.2 — **the** kind decision: is this the authoritative store?

        Authority comes from the *sealed class object plus factory attestation*.
        Never from `store.is_formal`, never from a module-level registry a
        caller can write to, and — since Stage E.2b.5 — never expressed as a
        module-level string a caller can rebind. An exact-type instance built by
        `object.__new__`, or one inserted into any table by hand, has no
        attestation and fails here: before dispatch, before a fit, before a
        formal write, before the ledger and before completion.

        It answers `True` or `False` and nothing else. There is no third value
        and no "unknown" return that a caller could arrange in order to fall out
        of a formal branch: anything that is not provably one of the two sealed,
        attested, correctly-rooted store types raises.
        """
        formal_cls = _sealed_classes.get("formal")
        non_formal_cls = _sealed_classes.get("non_formal")
        if formal_cls is None or non_formal_cls is None:       # pragma: no cover
            raise StoreConstructionError("§16.1.2: the store classes are not sealed")
        cls = type(store)
        if cls is formal_cls:
            formal = True
        elif cls is non_formal_cls:
            formal = False
        else:
            raise StoreConstructionError(
                f"§16.1.2: {cls.__qualname__} is not an approved RQ3 "
                f"store ({formal_cls.__name__}, {non_formal_cls.__name__}); it "
                f"may not be executed against"
            )
        attestation = _attestation_of(store)
        if attestation is None:
            raise StoreConstructionError(
                f"§16.1.2: this {cls.__qualname__} carries no factory "
                f"attestation; it was never built by an approved factory"
            )
        if not isinstance(getattr(store, "root", None), Path):
            raise StoreConstructionError("§16.1.2: the store carries no resolved root")
        root = _authoritative_root()
        if formal and store.root != root:
            raise StoreConstructionError(
                f"§16.1.1: a formal store at {store.root} is not the "
                f"authoritative root {root}"
            )
        if not formal and (store.root == root or root in store.root.parents):
            raise StoreConstructionError(
                f"§16.1.2: a non-formal store at {store.root} claims the "
                f"authoritative root"
            )
        return formal

    def store_kind(store) -> str:
        """The same decision, spelled as a label, for logs, reports and tests.

        The label comes from a **closure local**, so `rq3_store.FORMAL = <x>`
        cannot change what this returns — and no formal boundary consults it in
        the first place; they all take the bool from `store_is_formal`.
        """
        return _FORMAL_LABEL if store_is_formal(store) else _NON_FORMAL_LABEL

    def attested_authority_digest(store) -> str:
        """The verified-authority digest recorded when this store was created."""
        store_is_formal(store)
        return _attestation_of(store)[1]

    # -- the authorization gate ---------------------------------------------

    def assert_formal_execution_authorized(store):
        """§16.2 / §9 — the gate every formal fit and formal write passes through.

        Re-establishes the complete frozen authority (literal tag ref, freeze
        commit, protocol document bytes, validator bytes, mechanism dependency
        closure including every helper's own namespace and every third-party
        module alias, and the K40/K60 roles) and requires it to be *the same
        authority* the store was created under. A run whose protocol identity
        changed between store creation and execution stops here, before a single
        formal fit and before a single authoritative scientific write.

        It lives inside the closure so that the store's own call sites can
        capture *this* function at class-creation time. A module-level alias of
        the same name is exported for callers, but nothing on a critical path
        resolves the gate through that alias — and the verification itself is
        the sealed one captured above, not `proto.verified_authority`.
        """
        if not store_is_formal(store):
            raise StoreConstructionError(
                "§16.1.2: formal execution requires the formal store API")
        authority = _verified_authority()
        _assert_roles(authority["node_registry"])
        if _authority_digest(authority) != attested_authority_digest(store):
            raise StoreConstructionError(
                "§16.1: the frozen authority changed between store creation and "
                "formal execution; the run stops rather than continuing under a "
                "different identity"
            )
        return authority

    # -- the factories ------------------------------------------------------

    def create_authoritative(provenance: dict):
        """§16.1.1 / §16.1.3 — the only way a formal store comes into existence."""
        formal_cls = _sealed_classes["formal"]
        metadata = _reject(provenance)

        # the complete frozen binding — document bytes, mechanism source bytes,
        # mechanism dependency closure, literal tag ref, freeze commit —
        # re-established *now* through the sealed verifier, and before an
        # authoritative tree is allowed to exist
        authority = _verified_authority()
        roles = _assert_roles(authority["node_registry"])
        if roles != (40, 60):                       # literal, local, re-asserted
            raise StoreConstructionError("§9.6: formal execution requires (40, 60)")

        root = _authoritative_root()
        if root.exists():
            # §16.1.3 — empty, partial, complete or abandoned, it all fails the
            # same way, before anything is written (H24)
            raise StaleResultRootError(
                f"H24: {root} already exists. A formal RQ3 run never merges into, "
                f"reuses, resumes, repairs or overwrites a prior result tree. "
                f"Archive or remove it explicitly before running."
            )
        _writable_root(root)
        root.mkdir(parents=True, exist_ok=False)
        for name in RECORD_DIRS:
            (root / name).mkdir(exist_ok=False)

        store = formal_cls(root=root, construction_token=_token)
        _attest(store, _authority_digest(authority))
        # §16.1 — caller metadata first, verified frozen identity second, so run
        # metadata can never redefine what the result claims to be
        store._write(root / PROVENANCE_FILE, {
            **metadata,
            **_frozen_identity(authority, authoritative=True),
        })
        return store

    def create_non_formal(root, provenance: dict):
        non_formal_cls = _sealed_classes["non_formal"]
        metadata = _reject(provenance)
        resolved = _writable_root(root)
        formal = _authoritative_root()
        if resolved == formal or formal in resolved.parents or resolved in formal.parents:
            raise StoreConstructionError(
                f"§16.1.2: {resolved} is (or contains) the authoritative root "
                f"{formal}; a non-formal store never writes there — not through "
                f"this factory and not through any caller-minted capability"
            )
        # the same binding, minus the git round trip: a non-formal store writes
        # no authoritative artifact, but it still may not run on drifted bytes
        authority = _verified_authority(include_git=False)
        _assert_roles(authority["node_registry"])

        resolved.mkdir(parents=True, exist_ok=True)
        for name in RECORD_DIRS:
            (resolved / name).mkdir(exist_ok=True)
        store = non_formal_cls(root=resolved, construction_token=_token)
        _attest(store, _authority_digest(authority))
        store._write(resolved / PROVENANCE_FILE, {
            **metadata,
            **_frozen_identity(authority, authoritative=False),
        })
        return store

    def open_non_formal(root):
        non_formal_cls = _sealed_classes["non_formal"]
        resolved = _writable_root(root)
        formal = _authoritative_root()
        if resolved == formal or formal in resolved.parents:
            raise StoreConstructionError(
                f"§16.1.2: {resolved} is inside the authoritative root {formal}"
            )
        if not (resolved / PROVENANCE_FILE).exists():
            raise RQ3StoreError(f"{root} is not an RQ3 store: no {PROVENANCE_FILE}")
        authority = _verified_authority(include_git=False)
        store = non_formal_cls(root=resolved, construction_token=_token)
        _attest(store, _authority_digest(authority))
        return store

    return (_authorize, _seal_store_classes, store_is_formal, store_kind,
            attested_authority_digest, assert_formal_execution_authorized,
            create_authoritative, create_non_formal, open_non_formal)


(_authorize_construction, _seal_store_classes, store_is_formal, store_kind,
 attested_authority_digest, assert_formal_execution_authorized,
 _create_authoritative_store, _create_non_formal_store,
 _open_non_formal_store) = _build_store_authority()


# ---------------------------------------------------------------------------
# Stage E.2b.5 — factory bindings captured in closure cells
# ---------------------------------------------------------------------------
#
# `FormalRQ3Store.create_authoritative` used to be
#
#     return _create_authoritative_store(provenance)
#
# which resolves a **module global at call time**. `rq3_store.
# _create_authoritative_store = <substitute>` therefore replaced the entire
# claimed authoritative-creation path — the sealed verification, the stale-root
# refusal, the attestation and the provenance write — with whatever the caller
# supplied, through the public classmethod the runner actually calls.
#
# The binders below capture the genuine factory in a **closure cell** at class
# body execution time. The exported aliases stay (they are the names an auditor
# will try to rebind, and rebinding them must be visibly inert), but no class
# method resolves them again. Closure-cell surgery is outside the threat model
# these stages are audited under; ordinary module-name rebinding is inside it,
# and is what this closes.

def _bind_formal_factory(_genuine):
    """Capture the authoritative factory; return the bound classmethod."""

    def create_authoritative(cls, provenance: dict):
        """§16.1.1 / §16.1.3 — the only way a formal store comes into existence.

        The body lives in the factory closure (§16.1.2) so that the construction
        capability and the attestation table it writes are not module
        attributes, and the *reference* to that body is a closure cell of this
        function (E.2b.5) so that the module alias naming it is not an authority
        either. It verifies the whole frozen binding through the sealed verifier
        before an authoritative tree is allowed to exist.
        """
        return _genuine(provenance)

    return classmethod(create_authoritative)


def _bind_non_formal_factories(_genuine_create, _genuine_open):
    """The same capture for the two non-formal entry points."""

    def create(cls, root, provenance: dict):
        """Build a non-formal store; refuses the authoritative root (§16.1.2)."""
        return _genuine_create(root, provenance)

    def open_existing(cls, root):
        """Re-open a non-formal store; refuses the authoritative root (§16.1.2)."""
        return _genuine_open(root)

    open_existing.__name__ = "open"
    return classmethod(create), classmethod(open_existing)


# ---------------------------------------------------------------------------
# the record layer, shared by both stores
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _RQ3RecordStore:
    """Deterministic, checksummed, append-only-by-key RQ3 records.

    This layer knows nothing about completion. Neither the formal barrier nor a
    completion marker can be reached from here, which is what makes the
    non-formal store *structurally* unable to declare a formal run finished.
    """

    root: Path

    #: A one-shot capability minted by the approved factory (never a default).
    construction_token: object = None

    #: Digest of every file this store wrote, recorded **at write time**.
    #:
    #: The §16.2 ledger is written *by* the barrier, so a record corrupted
    #: before that step would otherwise be absorbed into the ledger and become
    #: the recorded truth. This journal closes that window: step 1 re-hashes
    #: every artifact from disk and compares it against what was written. It is
    #: not a substitute for the independent re-read of steps 4-6 — those still
    #: discard every in-memory copy and verify the ledger against fresh bytes.
    written_digests: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self, _authorize=_authorize_construction) -> None:
        # `_authorize` is bound as a default at class-creation time, from the
        # factory closure, so rebinding the module attribute cannot disable it.
        # It only ever *checks* a capability it has no way to hand out.
        _authorize(self.construction_token, type(self))

    def __init_subclass__(cls, **kwargs) -> None:
        """Only this module may define a store class (§16.1.2).

        Subclassing is how a duck-typed "formal" store would otherwise be built,
        and dispatch below matches on the exact type — so a subclass could not
        gain authority anyway. This closes the door one step earlier.
        """
        super().__init_subclass__(**kwargs)
        if cls.__module__ != __name__:
            raise StoreConstructionError(
                f"§16.1.2: {cls.__qualname__} subclasses an RQ3 store from "
                f"{cls.__module__}; store classes are defined only in {__name__}"
            )

    @property
    def is_formal(self) -> bool:
        """Derived from the exact class, never from stored state.

        This is a read-only view for logging. **Dispatch does not consult it** —
        see `store_kind` — because an attribute is something any object can
        carry, and authority is not.
        """
        return type(self) is FormalRQ3Store

    # -- sealed guards -------------------------------------------------------

    def authorized_is_formal(self, _is_formal=store_is_formal) -> bool:
        """**The** sealed kind decision, as a bool (Stage E.2b.5).

        The genuine closure predicate is captured here as a default at
        class-creation time. Callers that reach for `rq3_store.store_kind` are
        reading a module attribute, and a module attribute can be replaced with
        a permissive function; this cannot. Nor can the *answer* be neutralized
        by rebinding `rq3_store.FORMAL`: there is no string in this decision.
        """
        return _is_formal(self)

    def authorized_kind(self, _store_kind=store_kind) -> str:
        """The same decision as a label, for logs and reports.

        Read-only: no formal boundary branches on this. `authorized_is_formal`
        is what the write path, the ledger, the barrier and the orchestrator
        use, so a label that somehow compared unequal to everything could not
        turn a gate off.
        """
        return _store_kind(self)

    # -- writing ------------------------------------------------------------

    def _path(self, kind: str, key: str) -> Path:
        if kind not in RECORD_DIRS:
            raise RQ3StoreError(f"unknown record kind {kind!r}")
        target = (self.root / kind / f"{key}.json").resolve()
        if self.root not in target.parents:
            raise RQ3StoreError(f"H13: {target} escapes the store root")
        return target

    def _write(self, path: Path, payload,
               _is_formal=store_is_formal,
               _formal_gate=assert_formal_execution_authorized) -> str:
        # Every write — scientific record, provenance, ledger, marker — passes
        # the sealed authority check first: exact sealed type, factory
        # attestation, and root discipline (§16.1.2). Both guards are captured
        # from the factory closure at class-creation time, so rebinding
        # `rq3_store.store_kind` cannot make a forged object writable, and the
        # decision below is the closure's **bool** (E.2b.5), not a comparison
        # against `rq3_store.FORMAL` — a module global whose rebinding used to
        # make this branch, and every other formal branch, skippable.
        formal = _is_formal(self)
        if formal:
            # §16.2 / E.2b.3 §6 — an authoritative write is authorized *now*,
            # not by an attestation taken when the store was created. Mechanism
            # drift, a swapped helper namespace, a forged module alias or a
            # changed protocol byte between creation and this write stops the
            # write itself; "completion will catch it later" is not the
            # requirement, and by then the record would already be on disk.
            _formal_gate(self)
        target = Path(path).resolve()
        if target != (self.root / PROVENANCE_FILE).resolve() \
                and self.root not in target.parents:
            raise RQ3StoreError(f"H13: refusing to write outside the store root: {target}")
        if target.name == COMPLETION_FILE and not formal:
            raise RQ3StoreError(
                f"§16.1.2: a non-formal store may never write {COMPLETION_FILE}"
            )
        digest = _atomic_write(target, canonical_bytes(payload))
        if target.name != COMPLETION_FILE:
            self.written_digests[str(target.relative_to(self.root))] = digest
        return digest

    def put(self, kind: str, key: str, payload: dict) -> str:
        """Write one record. Re-writing a key with different bytes is a failure.

        A run that produces two different answers for the same registered key
        has a determinism defect, and silently keeping the second would hide it.
        """
        path = self._path(kind, key)
        blob = canonical_bytes(payload)
        if path.exists() and sha256_file(path) != sha256_bytes(blob):
            raise RQ3StoreError(
                f"{kind}/{key} already holds different bytes — non-deterministic run"
            )
        return self._write(path, payload)

    def get(self, kind: str, key: str) -> dict:
        path = self._path(kind, key)
        if not path.exists():
            raise RQ3StoreError(f"{kind}/{key} is absent from the store")
        return json.loads(path.read_bytes().decode("utf-8"))

    def keys(self, kind: str) -> frozenset:
        if kind not in RECORD_DIRS:
            raise RQ3StoreError(f"unknown record kind {kind!r}")
        directory = self.root / kind
        if not directory.exists():
            return frozenset()
        return frozenset(p.stem for p in sorted(directory.glob("*.json")))

    # -- completeness of the record set -------------------------------------

    def missing(self) -> dict:
        expected = expected_keys()
        return {kind: tuple(sorted(keys - self.keys(kind)))
                for kind, keys in expected.items()}

    def unexpected(self) -> dict:
        expected = expected_keys()
        return {kind: tuple(sorted(self.keys(kind) - keys))
                for kind, keys in expected.items()}

    def assert_complete(self) -> dict:
        """Every registered key present, and every 40-vs-60 check actually run."""
        missing = {k: v for k, v in self.missing().items() if v}
        if missing:
            raise RQ3StoreError(f"the RQ3 store is incomplete: {missing}")
        extra = {k: v for k, v in self.unexpected().items() if v}
        if extra:
            raise RQ3StoreError(f"the RQ3 store holds unregistered records: {extra}")

        evaluated = 0
        for key in sorted(self.keys(VALIDATION_DIR)):
            record = self.get(VALIDATION_DIR, key)
            verdict = record.get("numerical_validation")
            if not isinstance(verdict, dict) or "n_evaluated" not in verdict:
                raise RQ3StoreError(
                    f"H18: validation/{key} carries no seven-of-seven verdict"
                )
            required = int(verdict.get("n_required", -1))
            if required != proto.N_REQUIRED_AGREEMENT_CRITERIA:
                raise RQ3StoreError(
                    f"H18: validation/{key} declares {required} required criteria"
                )
            if bool(verdict.get("passed")) and not (
                    int(verdict.get("n_evaluated", -1)) == proto.N_REQUIRED_AGREEMENT_CRITERIA
                    and int(verdict.get("n_passed", -1)) == proto.N_REQUIRED_AGREEMENT_CRITERIA):
                raise RQ3StoreError(
                    f"H18: validation/{key} reports PASS without seven of seven"
                )
            status = record.get("status")
            if status != proto.STATUS_OK and record.get("failing_condition") \
                    not in proto.STATUS_CODES:
                raise RQ3StoreError(
                    f"§9.5.1: validation/{key} carries unregistered code "
                    f"{record.get('failing_condition')!r}"
                )
            evaluated += 1
        if evaluated != proto.N_VALIDATION_UNITS:
            raise RQ3StoreError(
                f"H27: {evaluated} validation units evaluated, §9.6.5 registers "
                f"{proto.N_VALIDATION_UNITS}"
            )

        for key in sorted(self.keys(FAMILIES_DIR)):
            family = self.get(FAMILIES_DIR, key)
            if int(family.get("m", -1)) != proto.HOLM_FAMILY_SIZE:
                raise RQ3StoreError(f"H12: families/{key} declares m={family.get('m')}")
            slots = family.get("slots", [])
            if len(slots) != proto.HOLM_FAMILY_SIZE:
                raise RQ3StoreError(f"H12: families/{key} holds {len(slots)} slots")
            for slot in slots:
                unestimable = (slot.get("p_raw") == proto.NOT_ESTIMABLE
                               or slot.get("p_holm") == proto.NOT_ESTIMABLE)
                if unestimable and slot.get("significant"):
                    raise RQ3StoreError(
                        f"H15: families/{key} declares an unestimable slot significant"
                    )
                if slot.get("p_raw") == proto.NOT_ESTIMABLE \
                        and slot.get("p_holm") != proto.NOT_ESTIMABLE:
                    raise RQ3StoreError(
                        f"H15: families/{key} serialized a numeric adjusted p for an "
                        f"unestimable slot"
                    )
        return {"n_validation_records": evaluated}

    # -- the checksum ledger ------------------------------------------------

    def ledger_entries(self) -> tuple:
        """Every record plus the provenance file, in a fixed order."""
        entries = []
        for kind in RECORD_DIRS:
            for key in sorted(self.keys(kind)):
                entries.append(f"{kind}/{key}.json")
        entries.append(PROVENANCE_FILE)
        return tuple(entries)

    def write_checksums(self, _is_formal=store_is_formal,
                        _formal_gate=assert_formal_execution_authorized) -> str:
        """Freeze a sorted, relative-path checksum ledger over every record."""
        if _is_formal(self):
            # the ledger is an authoritative artifact too, and it is written
            # through `_atomic_write` rather than `_write`
            _formal_gate(self)
        lines = [f"{sha256_file(self.root / relative)}  {relative}"
                 for relative in self.ledger_entries()]
        blob = ("\n".join(lines) + "\n").encode("utf-8")
        return _atomic_write(self.root / CHECKSUM_FILE, blob)

    def verify_checksums(self) -> int:
        """Re-read the ledger and re-hash every file it names."""
        target = self.root / CHECKSUM_FILE
        if not target.exists():
            raise RQ3StoreError(f"no {CHECKSUM_FILE} in the store")
        checked = 0
        for line in target.read_text().splitlines():
            if not line.strip():
                continue
            digest, relative = line.split("  ", 1)
            path = (self.root / relative).resolve()
            if not path.exists():
                raise RQ3StoreError(f"ledger names a missing file: {relative}")
            if sha256_file(path) != digest:
                raise RQ3StoreError(f"checksum mismatch for {relative}")
            checked += 1
        return checked

    @property
    def complete(self) -> bool:
        return (self.root / COMPLETION_FILE).exists()


# ---------------------------------------------------------------------------
# the non-formal store — a different API, not a flag (§16.1.2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NonFormalRQ3Store(_RQ3RecordStore):
    """Smoke, unit-test and synthetic execution.

    It **refuses** `results/rq3/` and every path inside it, and it has no
    completion barrier at all: `complete_run` exists only on `FormalRQ3Store`,
    so no argument value and no call sequence can make a run through this class
    authoritative (H23).
    """

    # Bound in closure cells at class body execution time (E.2b.5), not
    # resolved as module aliases at call time.
    create, open = _bind_non_formal_factories(_create_non_formal_store,
                                              _open_non_formal_store)

    def seal_non_formal(self) -> dict:
        """Check the record set and freeze a ledger — and write **no** marker.

        This is deliberately not the §16.2 barrier. It proves the persistence
        wiring works without producing the one artifact that means "a formal RQ3
        run completed" (§16.2), which this class cannot write at all.
        """
        summary = self.assert_complete()
        ledger = self.write_checksums()
        verified = self.verify_checksums()
        return {
            "authoritative": False,
            "completion_marker": None,
            "checksum_ledger_sha256": ledger,
            "n_files_verified": int(verified),
            "n_records": {kind: len(self.keys(kind)) for kind in RECORD_DIRS},
            **summary,
        }


# ---------------------------------------------------------------------------
# the formal store — hard-bound root, stale-root fail-closed, §16.2 barrier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormalRQ3Store(_RQ3RecordStore):
    """The one authoritative store. Its root is `results/rq3/` and nothing else.

    `create_authoritative` takes **no root argument** (§16.1.1) and hard-fails
    if the root already exists (§16.1.3). There is no merge, no resume, no
    repair, no back-fill and no overwrite; removing a previous attempt is an
    operator action taken before the run, never something this class does.
    """

    # The whole body lives in the factory closure (§16.1.2) — there is no
    # exported call that mints a construction capability — and the reference to
    # it is a **closure cell** of the binder (E.2b.5), so the module alias
    # `_create_authoritative_store` is no longer consulted when a runner calls
    # `FormalRQ3Store.create_authoritative(...)`.
    create_authoritative = _bind_formal_factory(_create_authoritative_store)

    # -- the sealed execution gate ------------------------------------------

    def authorize_formal_execution(
            self, _gate=assert_formal_execution_authorized):
        """Re-verify the whole frozen authority before the first formal fit.

        The genuine gate is captured from the factory closure at class-creation
        time. Rebinding `rq3_store.assert_formal_execution_authorized` to a
        no-op no longer lets a drifted formal run reach outcome construction,
        because the orchestrator asks the *store* to authorize itself.
        """
        return _gate(self)

    # -- §16.2 the completion barrier ---------------------------------------

    def complete_run(self, _gate=assert_formal_execution_authorized) -> dict:
        """The seven frozen steps of §16.2, in order, gated by the frozen rule.

        `COMPLETION.json` is the **terminal** artifact and attests to a ledger
        that was independently re-read from disk. If any check in steps 4-6
        fails the marker is not written, the run is `FAILED` / `INCOMPLETE`, and
        no part of the tree may be treated as authoritative output (H25).
        """
        # §16.2 — the authority is re-established here, independently of the
        # verification that ran when this store was created. The literal tag
        # ref, the freeze commit, the document bytes, the validator bytes, the
        # mechanism dependency closure (every helper through its own namespace,
        # every third-party alias by identity) and the node roles must all still
        # be the frozen ones, and must be the *same* ones this store was created
        # under, or the marker is never written. The gate is the sealed one,
        # bound at class-creation time, not a module attribute.
        authority = _gate(self)

        order = proto.completion_barrier_order()
        completed: list = []

        def done(step: str) -> None:
            expected = order[len(completed)]
            if step != expected:
                raise CompletionBarrierError(
                    f"§16.2: step {step!r} out of order, expected {expected!r}"
                )
            completed.append(step)

        if (self.root / COMPLETION_FILE).exists():
            raise CompletionBarrierError(
                f"H25: {COMPLETION_FILE} already exists; it is written once, terminally"
            )

        # 1 — every registered scientific and persistence artifact is present,
        #     and still byte-identical to what this run wrote. The ledger is
        #     written *next*, so a record corrupted before that step would
        #     otherwise be signed rather than caught.
        summary = self.assert_complete()
        self._verify_written_digests()
        done("write_artifacts")

        # 2 — the authoritative checksum ledger over those artifacts
        ledger_digest = self.write_checksums()
        done("write_checksum_ledger")

        # 3 — flush, fsync and close (each write already fsync'd; sync the tree)
        for kind in RECORD_DIRS:
            _fsync_directory(self.root / kind)
        _fsync_directory(self.root)
        done("flush_and_fsync")

        # 4 — re-open and re-read from disk through fresh handles, discarding
        #     every in-memory copy. Nothing below reuses a value computed above.
        reread = self._reread_from_disk()
        done("reread_from_disk")

        # 5 — recompute digests from the re-read bytes and compare against the
        #     re-read ledger; re-verify counts and every frozen identity field
        self._verify_reread(reread, ledger_digest, authority)
        done("recompute_and_compare_digests")

        # 6 — the artifact set is exactly the registered set
        self._verify_artifact_set(reread)
        done("verify_artifact_set_exact")

        # 7 — only now, and atomically
        if not proto.completion_marker_is_permitted(completed):
            raise CompletionBarrierError(
                f"§16.2: the barrier is incomplete ({completed}); the marker is refused"
            )
        marker = {
            # every identity in the marker comes from the authority verified at
            # the top of this barrier, never from a runtime constant
            "store_version": STORE_VERSION,
            "protocol_version": authority["protocol_version"],
            "protocol_tag": authority["protocol_tag"],
            "protocol_commit": authority["protocol_commit"],
            "protocol_doc_sha256": authority["protocol_doc_sha256"],
            "protocol_validator_sha256": authority["protocol_validator_sha256"],
            "mechanism_closure_sha256": authority["mechanism_closure_sha256"],
            "authoritative_nodes": int(authority["authoritative_nodes"]),
            "validation_nodes": int(authority["validation_nodes"]),
            "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checksum_ledger_sha256": reread["ledger_sha256"],
            "provenance_sha256": reread["provenance_sha256"],
            "n_files_verified": int(reread["n_files"]),
            "barrier_steps": list(completed) + [order[-1]],
            "n_records": {kind: len(reread["keys"][kind]) for kind in RECORD_DIRS},
            **summary,
        }
        self._write(self.root / COMPLETION_FILE, marker)
        return marker

    def _verify_written_digests(self) -> None:
        """§16.2 step 1 — every artifact is still what this run wrote."""
        if not self.written_digests:
            raise CompletionBarrierError(
                "§16.2: this store wrote nothing; a completion marker would attest "
                "to artifacts it cannot vouch for"
            )
        for relative, digest in sorted(self.written_digests.items()):
            path = self.root / relative
            if not path.exists():
                raise CompletionBarrierError(f"§16.2: {relative} was written but is gone")
            if sha256_file(path) != digest:
                raise CompletionBarrierError(
                    f"§16.2: {relative} changed on disk after it was written"
                )

    # -- barrier steps 4-6 ---------------------------------------------------

    def _reread_from_disk(self) -> dict:
        """§16.2 step 4 — fresh handles, fresh bytes, no in-memory reuse."""
        ledger_path = self.root / CHECKSUM_FILE
        if not ledger_path.exists():
            raise CompletionBarrierError(f"§16.2: {CHECKSUM_FILE} is absent from disk")
        with open(ledger_path, "rb") as handle:
            ledger_blob = handle.read()

        entries = {}
        for line in ledger_blob.decode("utf-8").splitlines():
            if not line.strip():
                continue
            if "  " not in line:
                raise CompletionBarrierError(f"§16.2: malformed ledger line {line!r}")
            digest, relative = line.split("  ", 1)
            if relative in entries:
                raise CompletionBarrierError(f"§16.2: {relative} is listed twice")
            entries[relative] = digest

        keys = {kind: sorted(self.keys(kind)) for kind in RECORD_DIRS}
        return {
            "ledger_sha256": sha256_bytes(ledger_blob),
            "entries": entries,
            "keys": keys,
            "n_files": len(entries),
            "provenance_sha256": None,
        }

    def _verify_reread(self, reread: dict, ledger_digest: str, authority) -> None:
        """§16.2 step 5 — digests, counts and provenance, all from the re-read bytes."""
        if reread["ledger_sha256"] != ledger_digest:
            raise CompletionBarrierError(
                "§16.2: the ledger on disk is not the ledger that was written"
            )
        expected_entries = set(self.ledger_entries())
        if set(reread["entries"]) != expected_entries:
            missing = sorted(expected_entries - set(reread["entries"]))
            extra = sorted(set(reread["entries"]) - expected_entries)
            raise CompletionBarrierError(
                f"§16.2: ledger coverage mismatch — missing {missing}, extra {extra}"
            )

        for relative, recorded in sorted(reread["entries"].items()):
            path = (self.root / relative).resolve()
            if self.root not in path.parents:
                raise CompletionBarrierError(f"H13: {relative} escapes the store root")
            if not path.exists():
                raise CompletionBarrierError(f"§16.2: {relative} named but absent")
            with open(path, "rb") as handle:
                blob = handle.read()
            if sha256_bytes(blob) != recorded:
                raise CompletionBarrierError(
                    f"§16.2: {relative} does not match its ledger digest"
                )
            if relative == PROVENANCE_FILE:
                reread["provenance_sha256"] = recorded
                self._verify_provenance(json.loads(blob.decode("utf-8")), authority)

        if reread["provenance_sha256"] is None:
            raise CompletionBarrierError(f"§16.2: {PROVENANCE_FILE} is not in the ledger")

        # the registered record counts, recomputed from what is on disk
        expected = expected_keys()
        for kind in RECORD_DIRS:
            if len(reread["keys"][kind]) != len(expected[kind]):
                raise CompletionBarrierError(
                    f"§16.2: {kind} holds {len(reread['keys'][kind])} records, "
                    f"the frozen registry declares {len(expected[kind])}"
                )

    def _verify_provenance(self, provenance: dict, authority=None, *,
                           _verify=proto.sealed_verified_authority) -> None:
        """§16.2 step 5 — **every** frozen identity field, re-derived now.

        The in-memory provenance object that wrote the file is not consulted:
        `frozen_identity` is rebuilt from the bound protocol and from the
        artifacts on disk, and the re-read record must match it field for field.
        A subset check — version, authoritative, inputs, code — left the protocol
        tag, the freeze commit, the document and validator hashes, the
        quadrature identity and the node identities unverified, so a store
        carrying a fake tag, a zero commit and node counts 999/998 could still
        complete.
        """
        # a freshly verified authority, established *after* the bytes on disk
        # were written — never the one the store was created with, and never
        # through a module alias (the sealed verifier is captured at
        # class-creation time, like every other guard on this path)
        verified = _verify() if authority is None else authority
        expected_identity = frozen_identity(verified, authoritative=True)

        if provenance.get("protocol_version") != expected_identity["protocol_version"]:
            raise CompletionBarrierError(
                f"§16.2: the store records protocol "
                f"{provenance.get('protocol_version')!r}, the verified authority "
                f"is {expected_identity['protocol_version']!r}"
            )
        if not provenance.get("authoritative"):
            raise CompletionBarrierError(
                "§16.2: a formal completion marker over a non-authoritative store"
            )

        authority = expected_identity

        for label in ("inputs", "code"):
            recorded = provenance.get(label)
            if not isinstance(recorded, dict) or not recorded:
                raise CompletionBarrierError(
                    f"§16.2: the store records no {label} provenance"
                )
            current = authority[label]
            if recorded != current:
                changed = sorted(k for k in set(recorded) | set(current)
                                 if recorded.get(k) != current.get(k))
                raise CompletionBarrierError(
                    f"§16.2: recorded {label} provenance no longer matches: {changed}"
                )

        for field_name in RESERVED_PROVENANCE_FIELDS:
            if field_name in ("inputs", "code"):
                continue
            expected = authority[field_name]
            if provenance.get(field_name) != expected:
                raise CompletionBarrierError(
                    f"§16.2: the store records {field_name}="
                    f"{provenance.get(field_name)!r}, the frozen authority is "
                    f"{expected!r}"
                )

    def _verify_artifact_set(self, reread: dict) -> None:
        """§16.2 step 6 — nothing missing, nothing extra, no unexpected key."""
        expected = expected_keys()
        for kind in RECORD_DIRS:
            on_disk = set(reread["keys"][kind])
            if on_disk != set(expected[kind]):
                raise CompletionBarrierError(
                    f"§16.2: {kind} missing {sorted(expected[kind] - on_disk)}, "
                    f"extra {sorted(on_disk - set(expected[kind]))}"
                )
        allowed_files = {PROVENANCE_FILE, CHECKSUM_FILE, COMPLETION_FILE}
        for entry in sorted(self.root.iterdir()):
            if entry.is_dir():
                if entry.name not in RECORD_DIRS:
                    raise CompletionBarrierError(
                        f"§16.2: unregistered directory {entry.name!r} in the store"
                    )
                unexpected = [p.name for p in sorted(entry.iterdir())
                              if p.suffix != ".json"]
                if unexpected:
                    raise CompletionBarrierError(
                        f"§16.2: unregistered files in {entry.name}/: {unexpected}"
                    )
            elif entry.name not in allowed_files:
                raise CompletionBarrierError(
                    f"§16.2: unregistered file {entry.name!r} in the store root"
                )


# The dispatcher resolves against these two class objects, captured in the
# factory closure — not against module attributes, which could be re-pointed.
# Sealing is one-shot, and the sealer is dropped from the module namespace.
_seal_store_classes(FormalRQ3Store, NonFormalRQ3Store)
del _seal_store_classes

# The factory binders have done their one job — the closure cells are written.
# Dropping them keeps the module free of any exported call that could re-bind a
# factory onto a class after import.
del _bind_formal_factory, _bind_non_formal_factories


