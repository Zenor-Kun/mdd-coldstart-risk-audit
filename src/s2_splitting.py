"""Label-free target-site splits (RQ2/S2, `docs/RQ2_S2_PROTOCOL.md` §5).

**Stage 2.0.1 correction.** Stage 2.0's `TwoStageTargetGuard` opened the
calibration labels of a `(site, N)` for **all 100 repeats at once**. Because the
repeats are independent uniform permutations, the union of 100 calibration sets
covers essentially the whole site — so the "two-stage" firewall materialized the
entire target label vector before most of that budget's decisions were frozen.
The class is **removed**, not patched, and nothing in this module reads a label
of any kind.

What survives here is the part that was always label-free: the split RNG, the
nesting rule, the per-`N` site eligibility, and a split manifest that carries
indices and identifiers only. Label access now lives behind a per-cell process
boundary in `src/s2_firewall.py` and `src/s2_cell_worker.py`, whose isolation
unit is one `(site, repeat, N)` cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import contracts
from certification import N_GRID, R_SPLIT

# ---------------------------------------------------------------------------
# frozen constants (§5.1)
# ---------------------------------------------------------------------------
#: RNG namespace of the target-site label splits. Distinct from the shrinkage
#: meta namespace [20260807, 2001] and the site-bootstrap namespace
#: [20260807, 1102]; none of the three may be reused for another purpose.
SPLIT_ENTROPY = [20260807, 2000]

#: Columns a split manifest may carry. No outcome, and no phenotype.
SPLIT_MANIFEST_COLUMNS = ("site", "repeat", "position", "canonical_id",
                          "n_acquisitions")


def site_eligible_at_n(n_s: int, n_labels: int) -> bool:
    """`N <= floor(n_s / 2)` and `n_s - N >= 15` (spec §4.2.10)."""
    n_s, n_labels = int(n_s), int(n_labels)
    return n_labels <= n_s // 2 and n_s - n_labels >= 15


def feasible_n_grid(n_s: int) -> tuple:
    """The label budgets a site can host, ascending."""
    return tuple(n for n in N_GRID if site_eligible_at_n(n_s, n))


def eligible_sites_by_n(site_sizes: dict) -> dict:
    """`{N: (sites eligible at N, in frozen site order)}` (§4.2)."""
    order = [s for s in contracts.FROZEN_SITES if s in site_sizes]
    out = {n: tuple(s for s in order if site_eligible_at_n(site_sizes[s], n))
           for n in N_GRID}
    for a, b in zip(N_GRID, N_GRID[1:]):
        if not set(out[b]) <= set(out[a]):
            raise AssertionError(
                f"eligibility is not nested between N={a} and N={b}; the "
                "threshold max(2N, N+15) is increasing in N, so this cannot "
                "happen with the frozen rule (§4.2)")
    return out


# ---------------------------------------------------------------------------
# §5.2 — one permutation per (site, repeat), nested across N
# ---------------------------------------------------------------------------
def split_permutation(site: str, repeat: int, n_s: int) -> np.ndarray:
    """The frozen permutation of a site's acquisitions for one repeat.

    Deterministic in `(site, repeat, n_s)` alone: same result on any machine,
    no dependence on call order, no `spawn()`, no global RNG. **Nothing
    outcome-derived reaches this function**, so the split cannot depend on
    diagnosis, prediction, confidence or any S0/S1 result.
    """
    site = str(site)
    if site not in contracts.FROZEN_SITES:
        raise KeyError(f"{site!r} is not a frozen eligible site")
    repeat = int(repeat)
    if not 0 <= repeat < R_SPLIT:
        raise ValueError(f"repeat must lie in [0, {R_SPLIT}), got {repeat}")
    o = contracts.FROZEN_SITES.index(site)
    seq = np.random.SeedSequence(SPLIT_ENTROPY + [o, repeat])
    rng = np.random.Generator(np.random.PCG64(seq))
    return rng.permutation(int(n_s))


def nested_splits(site: str, repeat: int, n_s: int) -> dict:
    """`{N: (D_cal indices, D_test indices)}` for every feasible `N`.

    `D_cal(N)` is the first `N` entries of the *same* permutation, so the
    calibration sets are nested and one repeat is a single onboarding
    trajectory under a growing label budget.

    Nesting is a property of the **index sets**, not of any shared label
    object: each `(site, repeat, N)` cell later re-reads its own rows inside
    its own process, and no label read by one cell is ever handed to another.
    """
    perm = split_permutation(site, repeat, n_s)
    return {n: (perm[:n].copy(), perm[n:].copy()) for n in feasible_n_grid(n_s)}


def cal_indices(site: str, repeat: int, n_s: int, n_labels: int) -> np.ndarray:
    """The `D_cal(N)` positions of exactly one cell — never a repeat range."""
    if not site_eligible_at_n(n_s, n_labels):
        raise ValueError(f"site {site} is not eligible at N={n_labels}")
    return split_permutation(site, repeat, n_s)[:int(n_labels)].copy()


def test_indices(site: str, repeat: int, n_s: int, n_labels: int) -> np.ndarray:
    """The `D_test(N)` positions of exactly one cell — never a repeat range."""
    if not site_eligible_at_n(n_s, n_labels):
        raise ValueError(f"site {site} is not eligible at N={n_labels}")
    return split_permutation(site, repeat, n_s)[int(n_labels):].copy()


def assert_nested(splits: dict) -> None:
    """Nesting, disjointness and exhaustiveness of one repeat's splits."""
    budgets = sorted(splits)
    for n in budgets:
        cal, test = splits[n]
        if len(cal) != n:
            raise AssertionError(f"|D_cal({n})| == {len(cal)} != {n}")
        if set(cal) & set(test):
            raise AssertionError(f"D_cal({n}) and D_test({n}) overlap")
        if len(set(cal) | set(test)) != len(cal) + len(test):
            raise AssertionError(f"N={n}: the split is not a partition")
    for a, b in zip(budgets, budgets[1:]):
        if not set(splits[a][0]) <= set(splits[b][0]):
            raise AssertionError(f"D_cal({a}) is not a subset of D_cal({b})")


# ---------------------------------------------------------------------------
# the label-free split manifest — the only split artifact
# ---------------------------------------------------------------------------
def build_split_manifest(site: str, canonical_ids, r_split: int = R_SPLIT
                         ) -> pd.DataFrame:
    """One row per `(repeat, position)`: which acquisition sits where.

    Carries identifiers and integer positions only. A decision cell derives its
    own `D_cal` from this manifest and reads nothing else; the manifest never
    carries a diagnosis, and `assert_label_free` refuses to write one.
    """
    ids = np.asarray(canonical_ids).astype(str)
    rows = []
    for r in range(int(r_split)):
        perm = split_permutation(site, r, ids.size)
        for pos, idx in enumerate(perm):
            rows.append({"site": str(site), "repeat": r, "position": pos,
                         "canonical_id": ids[idx], "n_acquisitions": ids.size})
    out = pd.DataFrame(rows, columns=list(SPLIT_MANIFEST_COLUMNS))
    assert_label_free(out, f"split manifest for {site}")
    return out


#: Substrings marking a column as carrying, or derived from, an outcome.
#: Deliberately specific: `n_labels` is the label *budget* `N`, not a label, so
#: a bare `label` token would flag the split registry's own budget column.
OUTCOME_TOKENS = ("diagnosis", "y_true", "y_cal", "y_test", "prevalence",
                  "n_mdd", "n_hc", "bacc", "auroc", "ber", "violation", "err",
                  "label_value", "raw_label", "y_label", "class_label")

#: Column names that contain an outcome-looking substring but are structural.
NON_OUTCOME_COLUMNS = frozenset({"n_labels"})


def assert_label_free(frame, context: str) -> None:
    """Raise if any column of `frame` is, or is derived from, an outcome."""
    bad = sorted(c for c in map(str, frame.columns)
                 if c not in NON_OUTCOME_COLUMNS
                 and any(t in c.lower() for t in OUTCOME_TOKENS))
    if bad:
        raise AssertionError(f"{context}: outcome-derived columns present: {bad}")


def write_split_manifest(site: str, canonical_ids, path,
                         r_split: int = R_SPLIT) -> Path:
    frame = build_split_manifest(site, canonical_ids, r_split)
    p = Path(path)
    frame.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# the registered split manifest — the SOLE authority for cell membership
# ---------------------------------------------------------------------------
#: Columns of the registered split manifest. Identifiers and integers only.
SPLIT_REGISTRY_COLUMNS = ("site", "repeat", "n_labels", "n_acquisitions",
                          "cal_ids", "test_ids", "cell_checksum")


def _cell_checksum(site: str, repeat: int, n_labels: int, cal_ids, test_ids) -> str:
    from s2_decisions import PROTOCOL_VERSION, sha256_bytes
    blob = "|".join([PROTOCOL_VERSION, str(site), str(int(repeat)),
                     str(int(n_labels)), ";".join(cal_ids), ";".join(test_ids)])
    return sha256_bytes(blob.encode("utf-8"))


def build_split_registry(site_ids: dict, r_split: int = R_SPLIT) -> pd.DataFrame:
    """Every feasible `(site, repeat, N)` cell with its exact membership.

    `site_ids` maps a site to its acquisitions' `canonical_id`s in the frozen
    canonical order. The registry is derived mechanically from that roster, the
    frozen `N` grid, the frozen eligibility rule and the frozen split RNG — a
    caller supplies the roster, never a cell.

    **The registered manifest is the sole authority for cell membership.** A
    worker resolves its rows from here; it may not be handed rows, and it may
    not re-draw a split of its own.
    """
    rows = []
    for site in [s for s in contracts.FROZEN_SITES if s in site_ids]:
        ids = [str(i) for i in site_ids[site]]
        if len(set(ids)) != len(ids):
            raise AssertionError(f"site {site}: duplicate canonical_id in the roster")
        n_s = len(ids)
        for repeat in range(int(r_split)):
            perm = split_permutation(site, repeat, n_s)
            for n in feasible_n_grid(n_s):
                cal = [ids[i] for i in perm[:n]]
                test = [ids[i] for i in perm[n:]]
                rows.append({
                    "site": site, "repeat": repeat, "n_labels": n,
                    "n_acquisitions": n_s,
                    "cal_ids": ";".join(cal), "test_ids": ";".join(test),
                    "cell_checksum": _cell_checksum(site, repeat, n, cal, test),
                })
    frame = pd.DataFrame(rows, columns=list(SPLIT_REGISTRY_COLUMNS))
    assert_label_free(frame, "split registry")
    return frame


class SplitRegistry:
    """Immutable, checksummed cell-membership authority.

    Loaded once, verified once, and consulted by every worker. `cell()` raises
    unless the cell is registered exactly once and its membership satisfies
    every structural invariant — length, disjointness, exhaustiveness,
    uniqueness and its own frozen checksum. Checking `len(cal_rows) == N` is
    what Stage 2.0.1 did and it was not enough.
    """

    def __init__(self, frame: pd.DataFrame, checksum: str, path=None) -> None:
        assert_label_free(frame, "split registry")
        missing = sorted(set(SPLIT_REGISTRY_COLUMNS) - set(frame.columns))
        if missing:
            raise AssertionError(f"split registry is missing columns {missing}")
        self._frame = frame.reset_index(drop=True)
        self.checksum = str(checksum)
        self.path = None if path is None else str(path)
        key = list(zip(self._frame["site"].astype(str),
                       self._frame["repeat"].astype(int),
                       self._frame["n_labels"].astype(int)))
        if len(set(key)) != len(key):
            raise AssertionError("split registry contains a duplicated cell")
        self._index = {k: i for i, k in enumerate(key)}

    # -- construction -------------------------------------------------------
    @classmethod
    def write(cls, site_ids: dict, path, r_split: int = R_SPLIT) -> "SplitRegistry":
        from s2_decisions import sha256_file
        frame = build_split_registry(site_ids, r_split)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(p, index=False)
        return cls(frame, sha256_file(p), p)

    @classmethod
    def load(cls, path, expected_checksum: str = None) -> "SplitRegistry":
        from s2_decisions import sha256_file
        p = Path(path)
        digest = sha256_file(p)
        if expected_checksum is not None and digest != expected_checksum:
            raise AssertionError(
                f"split registry {p}: checksum {digest[:12]}... != expected "
                f"{expected_checksum[:12]}...")
        frame = pd.read_csv(p, dtype={"cal_ids": str, "test_ids": str})
        return cls(frame, digest, p)

    # -- the authority ------------------------------------------------------
    def cell(self, site: str, repeat: int, n_labels: int) -> dict:
        """The exact membership of one registered cell, fully re-verified."""
        key = (str(site), int(repeat), int(n_labels))
        if key not in self._index:
            raise AssertionError(
                f"cell {key} is not registered in the split manifest; the "
                "registry is the sole authority for cell membership")
        row = self._frame.iloc[self._index[key]]
        cal = tuple(str(row["cal_ids"]).split(";")) if str(row["cal_ids"]) else ()
        test = tuple(str(row["test_ids"]).split(";")) if str(row["test_ids"]) else ()

        if len(cal) != int(n_labels):
            raise AssertionError(f"cell {key}: |cal| == {len(cal)} != {n_labels}")
        if len(set(cal)) != len(cal) or len(set(test)) != len(test):
            raise AssertionError(f"cell {key}: duplicate canonical_id in the split")
        if set(cal) & set(test):
            raise AssertionError(f"cell {key}: cal and test overlap")
        if len(cal) + len(test) != int(row["n_acquisitions"]):
            raise AssertionError(
                f"cell {key}: cal + test != the site's acquisition count")
        expected = _cell_checksum(key[0], key[1], key[2], cal, test)
        if expected != str(row["cell_checksum"]):
            raise AssertionError(f"cell {key}: split checksum mismatch")
        return {"site": key[0], "repeat": key[1], "n_labels": key[2],
                "cal_ids": cal, "test_ids": test,
                "n_acquisitions": int(row["n_acquisitions"]),
                "cell_checksum": str(row["cell_checksum"]),
                "registry_checksum": self.checksum}

    def site_roster(self, site: str) -> tuple:
        """The site's full acquisition set, as the registry knows it."""
        sub = self._frame[self._frame["site"].astype(str) == str(site)]
        if sub.empty:
            raise AssertionError(f"site {site} is not in the split registry")
        row = sub.iloc[0]
        cal = str(row["cal_ids"]).split(";") if str(row["cal_ids"]) else []
        test = str(row["test_ids"]).split(";") if str(row["test_ids"]) else []
        return tuple(sorted(set(cal) | set(test)))

    @property
    def cells(self) -> tuple:
        return tuple(sorted(self._index))

    @property
    def sites(self) -> tuple:
        return tuple(s for s in contracts.FROZEN_SITES
                     if s in set(self._frame["site"].astype(str)))

    def expected_decision_cells(self, strategies=None, methods=None) -> frozenset:
        """`(site, repeat, N, strategy, method)` for every label-consuming cell.

        Derived mechanically from the registry, the frozen label-consuming
        strategy set and the frozen confidence-method set. **The caller cannot
        supply, extend or trim this set** — that is what made an empty phase 1
        able to satisfy the Stage 2.0.1 barrier.
        """
        from certification import CONFIDENCE_METHODS, LABEL_CONSUMING_STRATEGIES
        strategies = tuple(strategies or LABEL_CONSUMING_STRATEGIES)
        methods = tuple(methods or CONFIDENCE_METHODS)
        unknown = set(strategies) - set(LABEL_CONSUMING_STRATEGIES)
        if unknown:
            raise AssertionError(
                f"{sorted(unknown)} consume no target label and are decided in "
                "the zero-label phase, not in a decision cell")
        return frozenset((s, r, n, strat, m)
                         for (s, r, n) in self.cells
                         for strat in strategies for m in methods)

    def canonical_representation(self) -> dict:
        """The registry's full content, in a form two registries can be compared on.

        Equality of *this* — every cell key, every calibration and test id, every
        per-cell checksum and every site roster — is what a production run
        requires, not `isinstance`. A hand-built registry that merely satisfies
        the internal invariants will differ here from the one the frozen
        contract derives.
        """
        cells = {}
        for (site, repeat, n) in self.cells:
            row = self.cell(site, repeat, n)
            cells["|".join((site, str(repeat), str(n)))] = {
                "cal": list(row["cal_ids"]), "test": list(row["test_ids"]),
                "checksum": row["cell_checksum"]}
        return {"rosters": {s: list(self.site_roster(s)) for s in self.sites},
                "cells": cells}

    def canonical_checksum(self) -> str:
        from s2_decisions import canonical_json, sha256_bytes
        return sha256_bytes(canonical_json(self.canonical_representation()))

    @property
    def roster_checksum(self) -> str:
        from s2_decisions import canonical_json, sha256_bytes
        rosters = {s: list(self.site_roster(s)) for s in self.sites}
        return sha256_bytes(canonical_json(rosters))

    def is_production_scale(self, site_sizes: dict = None) -> dict:
        """Is this the frozen 23-site, `R_SPLIT`-repeat registry, or a fixture?

        Recorded on the release artifact so a synthetic-scale run can never be
        mistaken for a production one.
        """
        sites = self.sites
        repeats = sorted({r for (_s, r, _n) in self.cells})
        ok = (len(sites) == len(contracts.FROZEN_SITES)
              and set(sites) == set(contracts.FROZEN_SITES)
              and repeats == list(range(R_SPLIT)))
        return {"production_scale": bool(ok), "n_sites": len(sites),
                "n_repeats": len(repeats), "n_cells": len(self.cells)}


class ProductionSplitRegistry(SplitRegistry):
    """A split registry that is **derived**, never accepted (Stage 2.0.3).

    Stage 2.0.2 let a caller build a `SplitRegistry` holding one site, one
    repeat and fabricated identifiers; it satisfied every internal invariant and
    the firewall treated it as the manifest. Internal consistency is not
    authority.

    A `ProductionSplitRegistry` may only be produced by
    `s2_production.build_production_split_registry_from_frozen_contract()`, and
    `assert_production_invariants()` re-derives the whole thing rather than
    reading a `production_scale` flag: the frozen 23-site roster, `R_SPLIT` full
    repeats, the frozen per-`N` eligibility, and — for a sample of cells and for
    every site — the split the frozen RNG actually produces.

    The type itself is part of the gate: the production APIs check
    `isinstance(..., ProductionSplitRegistry)`, so a fixture registry cannot
    stand in for one.
    """

    #: How many cells have their membership re-derived from the RNG. The check
    #: is exhaustive per site and per `N`; a full 7 700-cell re-derivation costs
    #: nothing scientifically but doubles preflight time, so a deterministic
    #: stratified sample is used and the sampling rule is fixed here.
    RNG_RECHECK_REPEATS = (0, 1, R_SPLIT - 1)

    def assert_production_invariants(self) -> dict:
        """Re-derive authority. Never reads a self-declared flag."""
        sites = self.sites
        if set(sites) != set(contracts.FROZEN_SITES):
            raise AssertionError(
                f"a production registry must hold the frozen {len(contracts.FROZEN_SITES)}"
                f"-site roster; got {len(sites)} sites")
        repeats = sorted({r for (_s, r, _n) in self.cells})
        if repeats != list(range(R_SPLIT)):
            raise AssertionError(
                f"a production registry must hold repeats 0..{R_SPLIT - 1}; got "
                f"{len(repeats)}")

        sizes = {}
        for site in sites:
            roster = self.site_roster(site)
            sizes[site] = len(roster)
        by_n = eligible_sites_by_n(sizes)
        expected_cells = {(s, r, n) for n in N_GRID for s in by_n[n]
                          for r in range(R_SPLIT)}
        if set(self.cells) != expected_cells:
            missing = sorted(expected_cells - set(self.cells))[:3]
            extra = sorted(set(self.cells) - expected_cells)[:3]
            raise AssertionError(
                f"registered cells are not the frozen eligibility closure "
                f"(missing e.g. {missing}, extra e.g. {extra})")

        # re-derive membership from the frozen RNG, per site and per N
        checked = 0
        for site in sites:
            roster = self.site_roster(site)
            ordered = tuple(sorted(roster))
            for n in feasible_n_grid(len(ordered)):
                for repeat in self.RNG_RECHECK_REPEATS:
                    perm = split_permutation(site, repeat, len(ordered))
                    cal = tuple(ordered[i] for i in perm[:n])
                    cell = self.cell(site, repeat, n)
                    if tuple(cell["cal_ids"]) != cal:
                        raise AssertionError(
                            f"cell ({site}, {repeat}, {n}) does not match the "
                            "frozen split RNG")
                    checked += 1
        return {"n_sites": len(sites), "n_repeats": len(repeats),
                "n_cells": len(self.cells), "n_rng_rechecked": checked,
                "per_n_site_counts": {n: len(v) for n, v in by_n.items()}}
