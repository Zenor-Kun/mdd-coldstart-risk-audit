"""The RQ2/S2 production root of trust (Stage 2.0.3).

Stage 2.0.2's gates were internally correct and still bypassable, because
*authority* was whatever the caller handed in. A `SplitRegistry` holding one
site, one repeat and fabricated identifiers satisfied every internal invariant
and was then treated as the manifest; a release artifact could be hand-written
with `n_expected = 1` and a self-consistent SHA-256; and two environment
variables could redirect the outcome source of a production run.

This module makes authority something the code **recomputes**, never something
it is told:

* `resolve_authoritative_outcome_source()` — one immutable path, derived from
  the frozen data contract. No environment variable, no CLI flag, no config
  entry and no job payload can move it.
* `build_production_split_registry_from_frozen_contract()` — the registry is
  rebuilt from the frozen cohort's canonical identifiers, the frozen site
  roster, the frozen `N` grid, the frozen eligibility rule, `R_SPLIT = 100` and
  the frozen split RNG. A registry is *derived*, never accepted.
* `ProductionS2Context.build()` — the only object the production Phase 1 / 2
  APIs accept. Immutable, strict, and carrying no `test_mode`, `allow_partial`,
  `production_scale`, `skip_validation`, `force` or `unsafe` field.

The exact production decision count is a hard invariant, not a claim:

    77 site-N cells x 100 repeats x 3 methods x 2 label-consuming strategies
        = 46,200
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import certification as cert
import contracts
import s2_decisions as sd
import s2_splitting as spl

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# the authoritative sources — repository-defined, not configurable
# ---------------------------------------------------------------------------
#: The frozen cohort (`DATA_CONTRACT_v1.md` §3.12), tag `data-contract-v1`.
AUTHORITATIVE_COHORT = ROOT / "results" / "data_contract" / "analysis_manifest_v1.csv"
#: The frozen row-addressable outcome source, tag `rq1-s0-results-v1`.
AUTHORITATIVE_OUTCOME_SOURCE = ROOT / "results" / "rq1_s0" / "confidence_target.csv"
#: The frozen protocol document this run executes.
PROTOCOL_DOC = ROOT / "docs" / "RQ2_S2_PROTOCOL.md"

#: Columns of the outcome source the root-of-trust check may read. Both are
#: identity columns; no outcome column is decoded when the context is built.
COHORT_BINDING_COLUMNS = ("held_out_site", "canonical_id")


# ---------------------------------------------------------------------------
# exact production invariants (frozen; hard-asserted, never inferred)
# ---------------------------------------------------------------------------
PRODUCTION_SITE_COUNTS_BY_N = {5: 23, 10: 22, 20: 21, 40: 11}
PRODUCTION_SITE_N_CELLS = 77
PRODUCTION_N_METHODS = 3
PRODUCTION_LABEL_CONSUMING_STRATEGIES = ("target_local", "pooled_local_shrinkage")
PRODUCTION_DECISIONS_BY_N = {5: 13_800, 10: 13_200, 20: 12_600, 40: 6_600}
PRODUCTION_DECISIONS_BY_STRATEGY = {"target_local": 23_100,
                                    "pooled_local_shrinkage": 23_100}
PRODUCTION_EXPECTED_DECISIONS = 46_200
#: Zero-label reference decisions: 23 sites x 3 methods x 2 zero-label strategies.
PRODUCTION_ZERO_LABEL_DECISIONS = 138

assert sum(PRODUCTION_SITE_COUNTS_BY_N.values()) == PRODUCTION_SITE_N_CELLS
assert set(PRODUCTION_LABEL_CONSUMING_STRATEGIES) == set(cert.LABEL_CONSUMING_STRATEGIES)
for _n, _s in PRODUCTION_SITE_COUNTS_BY_N.items():
    assert PRODUCTION_DECISIONS_BY_N[_n] == _s * cert.R_SPLIT * PRODUCTION_N_METHODS * 2
assert sum(PRODUCTION_DECISIONS_BY_N.values()) == PRODUCTION_EXPECTED_DECISIONS
assert sum(PRODUCTION_DECISIONS_BY_STRATEGY.values()) == PRODUCTION_EXPECTED_DECISIONS
assert PRODUCTION_N_METHODS == len(cert.CONFIDENCE_METHODS)


class ProductionAuthorityError(AssertionError):
    """Raised when production authority cannot be established or re-derived."""


# ---------------------------------------------------------------------------
# the immutable outcome-source resolver
# ---------------------------------------------------------------------------
def resolve_authoritative_outcome_source() -> Path:
    """The single production outcome source. Takes no argument, reads no env.

    There is deliberately no parameter, no environment lookup, no config entry
    and no override branch: the path is a repository constant bound to the
    frozen data contract, and a production run that cannot find it fails rather
    than falling back to something else.
    """
    if not AUTHORITATIVE_OUTCOME_SOURCE.exists():
        raise ProductionAuthorityError(
            f"the authoritative outcome source {AUTHORITATIVE_OUTCOME_SOURCE} is "
            "not present; a production run may not substitute another file")
    return AUTHORITATIVE_OUTCOME_SOURCE


def assert_authoritative_outcome_source(path) -> Path:
    """Re-resolve and compare. Never trust a recorded boolean.

    An artifact claiming `outcome_source_authoritative = True` proves nothing;
    the production path re-resolves the authoritative source and compares the
    resolved path itself.
    """
    resolved = resolve_authoritative_outcome_source()
    if Path(path).resolve() != resolved.resolve():
        raise ProductionAuthorityError(
            f"outcome source {path} is not the authoritative source {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# the frozen cohort roster
# ---------------------------------------------------------------------------
def load_frozen_cohort_roster() -> dict:
    """`{site: (canonical_id, ...)}` in the frozen canonical order.

    Read from the frozen manifest with `usecols`, so no diagnosis column is
    decoded, and asserted against the frozen cohort constants before it is
    allowed to seed anything.
    """
    if not AUTHORITATIVE_COHORT.exists():
        raise ProductionAuthorityError(
            f"the frozen cohort {AUTHORITATIVE_COHORT} is not present")
    frame = pd.read_csv(AUTHORITATIVE_COHORT,
                        usecols=["canonical_id", "site", "inclusion_status"])
    frame = frame[frame["inclusion_status"] == "INCLUDED"].copy()
    spl.assert_label_free(frame, "frozen cohort roster")
    if len(frame) != contracts.FROZEN_N_ACQUISITIONS:
        raise ProductionAuthorityError(
            f"cohort has {len(frame)} acquisitions, not "
            f"{contracts.FROZEN_N_ACQUISITIONS}")
    sites = sorted(set(frame["site"].astype(str)))
    if set(sites) != set(contracts.FROZEN_SITES):
        raise ProductionAuthorityError("the eligible-site set moved")
    if not frame["canonical_id"].is_unique:
        raise ProductionAuthorityError("duplicate canonical_id in the cohort")

    roster = {}
    for site in contracts.FROZEN_SITES:
        ids = sorted(frame.loc[frame["site"].astype(str) == site,
                               "canonical_id"].astype(str))
        roster[site] = tuple(ids)
    return roster


def assert_roster_binds_outcome_source(roster: dict) -> dict:
    """Every cohort acquisition must exist in the outcome source, at its site.

    A label-free pass: only `held_out_site` and `canonical_id` are decoded.
    This is what stops a registry of fabricated identifiers from being usable —
    the identifiers must be the cohort's, and the cohort's must be resolvable in
    the authoritative source.
    """
    source = resolve_authoritative_outcome_source()
    key = pd.read_csv(source, usecols=list(COHORT_BINDING_COLUMNS))
    site_of = dict(zip(key["canonical_id"].astype(str),
                       key["held_out_site"].astype(str)))
    missing, misplaced = [], []
    for site, ids in roster.items():
        for cid in ids:
            if cid not in site_of:
                missing.append(cid)
            elif site_of[cid] != site:
                misplaced.append(cid)
    if missing or misplaced:
        raise ProductionAuthorityError(
            f"{len(missing)} cohort ids are absent from the authoritative "
            f"outcome source and {len(misplaced)} sit at another site "
            f"(e.g. {(missing or misplaced)[:3]})")
    return {"n_bound": sum(len(v) for v in roster.values()),
            "outcome_source": str(source)}


# ---------------------------------------------------------------------------
# the authoritative production split registry
# ---------------------------------------------------------------------------
def build_production_split_registry_from_frozen_contract(
        workdir=None) -> spl.ProductionSplitRegistry:
    """Derive the one production split manifest. The only production path.

    Everything that decides its contents is frozen upstream: the cohort's
    canonical identifiers and site membership, the frozen site roster, the `N`
    grid, the eligibility rule, `R_SPLIT = 100`, the split RNG namespace and the
    nested-split algorithm. A caller supplies **nothing** but a directory to
    write it to.
    """
    roster = load_frozen_cohort_roster()
    binding = assert_roster_binds_outcome_source(roster)
    sizes = {s: len(v) for s, v in roster.items()}
    by_n = spl.eligible_sites_by_n(sizes)
    observed = {n: len(v) for n, v in by_n.items()}
    if observed != PRODUCTION_SITE_COUNTS_BY_N:
        raise ProductionAuthorityError(
            f"per-N eligibility is {observed}, not the frozen "
            f"{PRODUCTION_SITE_COUNTS_BY_N}")

    workdir = Path(workdir) if workdir is not None else Path(
        tempfile.mkdtemp(prefix="s2_production_registry_"))
    workdir.mkdir(parents=True, exist_ok=True)
    registry = spl.ProductionSplitRegistry.write(
        roster, workdir / "production_split_registry.csv", r_split=cert.R_SPLIT)
    registry.assert_production_invariants()
    registry.binding = binding
    return registry


# ---------------------------------------------------------------------------
# the authoritative snapshot — current facts, recomputed from frozen sources
# ---------------------------------------------------------------------------
SNAPSHOT_FIELDS = (
    "protocol_version", "protocol_doc_checksum", "cohort_checksum",
    "roster_checksum", "site_roster_checksum", "outcome_source",
    "split_registry_checksum", "split_registry_canonical_checksum",
    "expected_cell_registry_checksum", "expected_count",
    "ci_registry_checksum", "ci_registry_count",
    "evaluation_root", "evaluation_root_identity",
)

#: The one directory name a run's frozen evaluation artifacts live in. It is a
#: constant of the protocol, not a setting: there is no flag, no environment
#: variable and no argument that renames it (§17.24).
EVALUATION_STORE_DIRNAME = "evaluations"


def derive_authoritative_evaluation_root(registry=None):
    """The evaluation root of a run, **derived** from its authoritative registry.

    Stage 2.0.6 let a caller hand `aggregate_authoritative_site_level_results` an
    `EvaluationArtifactStore("/tmp/fake")`, so a complete fake store with
    correctly copied decision bindings and arbitrary metrics could be aggregated
    into a formal report. The remedy is the same one Stage 2.0.5 and 2.0.6
    applied a layer up: the root is not a parameter anywhere, it is a function of
    the frozen split registry's own location.
    """
    reg = registry if registry is not None else \
        build_production_split_registry_from_frozen_contract()
    return Path(reg.path).parent / EVALUATION_STORE_DIRNAME


def evaluation_root_identity(root, registry_checksum: str) -> str:
    """The canonical identity of an evaluation root, bound to the run."""
    return sd.sha256_bytes(sd.canonical_json({
        "protocol_version": sd.PROTOCOL_VERSION,
        "store_dirname": EVALUATION_STORE_DIRNAME,
        "evaluation_root": str(root),
        "split_registry_checksum": str(registry_checksum),
    }))


def derive_authoritative_snapshot(registry=None) -> dict:
    """The current authoritative facts, derived from the frozen sources.

    Nothing here reads a context, an artifact or a recorded claim. A registry
    may be supplied to avoid rebuilding a 7 700-cell manifest twice in one
    call; when it is, it is the *caller's* registry that gets measured, which is
    exactly what the equality check downstream needs.
    """
    import s2_aggregation as agg

    roster = load_frozen_cohort_roster()
    reg = registry if registry is not None else \
        build_production_split_registry_from_frozen_contract()
    expected = reg.expected_decision_cells()
    evaluation_root = derive_authoritative_evaluation_root(reg)
    return {
        "protocol_version": sd.PROTOCOL_VERSION,
        "protocol_doc_checksum": (sd.sha256_file(PROTOCOL_DOC)
                                  if PROTOCOL_DOC.exists() else "ABSENT"),
        "cohort_checksum": sd.sha256_file(AUTHORITATIVE_COHORT),
        "roster_checksum": sd.sha256_bytes(sd.canonical_json(
            {s: list(v) for s, v in roster.items()})),
        "site_roster_checksum": sd.sha256_bytes(sd.canonical_json(
            list(contracts.FROZEN_SITES))),
        "outcome_source": str(resolve_authoritative_outcome_source()),
        "split_registry_checksum": reg.checksum,
        "split_registry_canonical_checksum": reg.canonical_checksum(),
        "expected_cell_registry_checksum": expected_cells_checksum(expected),
        "expected_count": len(expected),
        "ci_registry_checksum": sd.sha256_bytes(sd.canonical_json(
            sorted(str(k.as_tuple()) for k in agg.S2_REQUIRED_CI_KEYS))),
        "ci_registry_count": agg.N_REQUIRED_INTERVALS,
        "evaluation_root": str(evaluation_root),
        "evaluation_root_identity": evaluation_root_identity(
            evaluation_root, reg.checksum),
    }


def expected_cells_checksum(expected) -> str:
    return sd.sha256_bytes(sd.canonical_json(
        sorted("|".join(map(str, k)) for k in expected)))


def assert_expected_decomposition(expected) -> dict:
    """The 46,200 must decompose exactly as the protocol states."""
    by_n, by_strategy, site_n = {}, {}, set()
    for (site, repeat, n, strategy, method) in expected:
        by_n[n] = by_n.get(n, 0) + 1
        by_strategy[strategy] = by_strategy.get(strategy, 0) + 1
        site_n.add((site, n))
        if method not in cert.CONFIDENCE_METHODS:
            raise ProductionAuthorityError(f"unknown method {method!r}")
        if strategy not in PRODUCTION_LABEL_CONSUMING_STRATEGIES:
            raise ProductionAuthorityError(
                f"{strategy!r} is not a label-consuming strategy")
    if by_n != PRODUCTION_DECISIONS_BY_N:
        raise ProductionAuthorityError(
            f"per-N decision counts are {by_n}, not {PRODUCTION_DECISIONS_BY_N}")
    if by_strategy != PRODUCTION_DECISIONS_BY_STRATEGY:
        raise ProductionAuthorityError(
            f"per-strategy counts are {by_strategy}, not "
            f"{PRODUCTION_DECISIONS_BY_STRATEGY}")
    if len(site_n) != PRODUCTION_SITE_N_CELLS:
        raise ProductionAuthorityError(
            f"{len(site_n)} site-N cells, not {PRODUCTION_SITE_N_CELLS}")
    if len(expected) != PRODUCTION_EXPECTED_DECISIONS:
        raise ProductionAuthorityError(
            f"{len(expected)} expected decisions, not "
            f"{PRODUCTION_EXPECTED_DECISIONS}")
    return {"by_n": by_n, "by_strategy": by_strategy,
            "site_n_cells": len(site_n), "total": len(expected)}


def derive_authoritative_expected_decision_cells(registry=None) -> frozenset:
    """The expected decision-cell set, derived rather than read."""
    reg = registry if registry is not None else \
        build_production_split_registry_from_frozen_contract()
    reg.assert_production_invariants()
    expected = reg.expected_decision_cells()
    assert_expected_decomposition(expected)
    return expected


def assert_registry_matches_authority(registry) -> dict:
    """Rebuild the authoritative registry and require exact content equality.

    `isinstance(registry, ProductionSplitRegistry)` is provenance, not proof —
    the class is publicly constructible. The registry a production run uses must
    reproduce, cell for cell and identifier for identifier, the one the frozen
    contract derives.
    """
    if not isinstance(registry, spl.ProductionSplitRegistry):
        raise ProductionAuthorityError(
            "a production run requires a ProductionSplitRegistry derived from "
            "the frozen contract")
    registry.assert_production_invariants()
    authoritative = build_production_split_registry_from_frozen_contract()
    mine, theirs = registry.canonical_representation(), \
        authoritative.canonical_representation()
    if mine != theirs:
        bad_cells = sorted(set(theirs["cells"]) ^ set(mine["cells"]))[:3]
        differing = [k for k in theirs["cells"]
                     if k in mine["cells"] and mine["cells"][k] != theirs["cells"][k]]
        raise ProductionAuthorityError(
            f"the registry does not match the authoritative derivation "
            f"(cell-set difference e.g. {bad_cells}, "
            f"{len(differing)} cells differ in membership)")
    if registry.checksum != authoritative.checksum:
        # content matches but the serialisation does not: still not the same file
        raise ProductionAuthorityError(
            "the registry file checksum does not match the authoritative one")
    return {"n_cells": len(mine["cells"]), "canonical_checksum":
            registry.canonical_checksum()}


# ---------------------------------------------------------------------------
# the production context — an authority *cache*, never the authority itself
# ---------------------------------------------------------------------------
_FACTORY_TOKEN = object()


class ProductionS2Context:
    """Immutable, opaque root-of-trust cache for one production S2 run.

    Deliberately **not** a dataclass: `dataclasses.replace()` on a frozen
    dataclass is a public forgery API, and Stage 2.0.3 relied on the type alone.
    Construction goes through `build()`, the attributes are read-only slots, and
    — the part that actually matters — **every production boundary re-derives
    the authority and compares it to what this object caches**. A forged or
    replaced context therefore fails at use, not merely at construction.

    Carries no `test_mode`, `allow_partial`, `production_scale`,
    `custom_source`, `skip_validation`, `force` or `unsafe` field.
    """

    __slots__ = ("_snapshot", "_registry", "_expected_cells", "_workdir",
                 "_run_id")

    def __init__(self, token, snapshot, registry, expected_cells, workdir,
                 run_id):
        if token is not _FACTORY_TOKEN:
            raise ProductionAuthorityError(
                "ProductionS2Context is built by ProductionS2Context.build() "
                "alone; and construction provenance is not the guarantee — "
                "every boundary re-derives the authority")
        object.__setattr__(self, "_snapshot", dict(snapshot))
        object.__setattr__(self, "_registry", registry)
        object.__setattr__(self, "_expected_cells", frozenset(expected_cells))
        object.__setattr__(self, "_workdir", Path(workdir))
        object.__setattr__(self, "_run_id", str(run_id))

    def __setattr__(self, name, value):
        raise ProductionAuthorityError("the production context is immutable")

    def __delattr__(self, name):
        raise ProductionAuthorityError("the production context is immutable")

    # -- read-only view -----------------------------------------------------
    @property
    def snapshot(self) -> dict:
        return dict(self._snapshot)

    @property
    def registry(self):
        return self._registry

    @property
    def expected_cells(self) -> frozenset:
        return self._expected_cells

    @property
    def expected_count(self) -> int:
        return len(self._expected_cells)

    @property
    def workdir(self) -> Path:
        return self._workdir

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def protocol_version(self) -> str:
        return self._snapshot["protocol_version"]

    @property
    def outcome_source(self) -> Path:
        return Path(self._snapshot["outcome_source"])

    @property
    def evaluation_root(self) -> Path:
        """The run's one evaluation root — cached here, re-derived on use."""
        return Path(self._snapshot["evaluation_root"])

    def assert_evaluation_root_authoritative(self) -> Path:
        """Re-derive the evaluation root and its identity, then return it.

        The only way a production run learns where its evaluation artifacts
        live. It is deliberately a *derivation*: a caller has no parameter to
        point it somewhere else, and the cached value is not believed — it is
        recomputed from the registry and compared, exactly as every other
        authority field is.
        """
        root = derive_authoritative_evaluation_root(self._registry)
        identity = evaluation_root_identity(root, self._registry.checksum)
        if str(root) != self._snapshot.get("evaluation_root"):
            raise ProductionAuthorityError(
                f"the evaluation root does not re-derive: cached "
                f"{self._snapshot.get('evaluation_root')!r} vs current {str(root)!r}")
        if identity != self._snapshot.get("evaluation_root_identity"):
            raise ProductionAuthorityError(
                "the evaluation-root identity does not re-derive")
        if root.parent != self._workdir:
            raise ProductionAuthorityError(
                f"the evaluation root {root} does not lie in the run's "
                f"authoritative work directory {self._workdir}")
        return root

    # -- the only constructor ----------------------------------------------
    @classmethod
    def build(cls, workdir=None) -> "ProductionS2Context":
        """Derive every element of authority from the frozen upstream state."""
        import uuid

        registry = build_production_split_registry_from_frozen_contract(workdir)
        registry.assert_production_invariants()
        expected = derive_authoritative_expected_decision_cells(registry)
        snapshot = derive_authoritative_snapshot(registry)
        if snapshot["expected_count"] != PRODUCTION_EXPECTED_DECISIONS:
            raise ProductionAuthorityError(
                f"the production registry implies {snapshot['expected_count']} "
                f"decision cells, not {PRODUCTION_EXPECTED_DECISIONS}")
        if snapshot["ci_registry_count"] != 216:
            raise ProductionAuthorityError(
                f"the CI registry holds {snapshot['ci_registry_count']} keys, not 216")
        return cls(_FACTORY_TOKEN, snapshot, registry, expected,
                   Path(registry.path).parent, uuid.uuid4().hex)

    # -- the check every boundary runs --------------------------------------
    def assert_still_authoritative(self) -> dict:
        """Re-derive **everything** and compare, field by field.

        This does not check the context against itself. It rebuilds the
        authoritative registry from the frozen contract, re-reads the cohort,
        re-resolves the outcome source, re-derives the expected-cell set and
        re-measures the CI registry — then requires the cached snapshot to
        agree. `replace(context, ci_registry_checksum="fake")` or any other
        substitution fails here.
        """
        assert_registry_matches_authority(self._registry)
        current = derive_authoritative_snapshot(self._registry)
        cached = self._snapshot
        for field in SNAPSHOT_FIELDS:
            if cached.get(field) != current[field]:
                raise ProductionAuthorityError(
                    f"authority field {field!r} does not re-derive: cached "
                    f"{cached.get(field)!r} vs current {current[field]!r}")
        expected = derive_authoritative_expected_decision_cells(self._registry)
        if set(expected) != set(self._expected_cells):
            raise ProductionAuthorityError(
                "the cached expected-cell set does not re-derive")
        if len(expected) != PRODUCTION_EXPECTED_DECISIONS:
            raise ProductionAuthorityError(
                f"expected count {len(expected)} != {PRODUCTION_EXPECTED_DECISIONS}")
        assert_authoritative_outcome_source(self.outcome_source)
        self.assert_evaluation_root_authoritative()
        return current

    def authority_state(self) -> dict:
        """The release-bound authority fields, freshly re-derived."""
        current = self.assert_still_authoritative()
        return {k: current[k] for k in SNAPSHOT_FIELDS}


#: Field names a production context may never carry (§28 of Stage 2.0.2).
FORBIDDEN_CONTEXT_FIELDS = ("test_mode", "allow_partial", "production_scale",
                            "custom_source", "skip_validation", "force",
                            "unsafe", "allow_nonproduction_registry")
assert not set(ProductionS2Context.__slots__) & set(FORBIDDEN_CONTEXT_FIELDS)
assert not hasattr(ProductionS2Context, "__dataclass_fields__")
