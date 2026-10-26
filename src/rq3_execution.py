"""RQ3 orchestration — one execution path, used by both entry points.

`scripts/run_rq3.py` (the authoritative Stage E.3 execution) and
`scripts/smoke_rq3.py` (the non-authoritative mechanical smoke) both
call `execute` below. They differ only in the populations and outcomes handed
in: real frozen ones, or synthetic ones. Keeping the orchestration in one place
is the point — a smoke test that exercises a *different* path proves nothing
about the path that will produce the result.

The order is fixed and carries no discretion:

    for each population, for each of the six registered cells in §2.4.1 order
        fit and cross-validate at K = 40   (authoritative)
        fit and cross-validate at K = 60   (numerical validation only)
        validate each registered block 40-vs-60, seven of seven
        assemble the cell's §15 outputs
    then the four Holm families of six slots
    then the §16.2 completion barrier — for a formal run only, and only after
    every artifact and the ledger have been independently re-read from disk

Nothing is skipped on the basis of a result, and no cell is selected, dropped or
reordered by anything but the frozen registry.

The two stores are **different APIs, not one API with a flag** (§16.1.2): the
formal store runs the barrier and writes `COMPLETION.json`; the non-formal store
has no barrier and cannot write that marker at all. `execute` dispatches on the
store's **exact type**, decided inside the store's own factory closure and
delivered as a bool, so no argument value passed to a runner, no attribute set on
a hand-made object and no rebound module global can turn a synthetic run into an
authoritative one — or, in the other direction, turn the authorization of a
formal run off (H23).
"""

from __future__ import annotations

import time

import rq3_data as data
import rq3_inference as inference
import rq3_models as models
import rq3_protocol as proto
import rq3_store as store_module

NOT_ESTIMABLE = proto.NOT_ESTIMABLE


def default_outcome_builder(population: data.Population, operating_point: str,
                            method: str) -> data.CellOutcome:
    """The frozen §2.1 outcome for a real population."""
    return data.build_cell_outcome(population, operating_point, method)


def execute(store, populations: dict,
            outcome_builder=default_outcome_builder, progress=None,
            *,
            _is_formal=store_module._RQ3RecordStore.authorized_is_formal,
            _formal_gate=store_module.FormalRQ3Store.authorize_formal_execution,
            ) -> dict:
    """Run every registered cell of every population and persist the results.

    `populations` maps population name -> `Population`. Both frozen populations
    must be present: the registered output set spans them, and a partial run may
    not be completed (the §16.2 barrier enforces this independently, from the
    bytes on disk rather than from anything this function believes).

    **There is no node-count parameter.** The registry is exactly `(40, 60)` with
    40 authoritative and 60 the independent numerical validation; a caller that
    could pass `(40, 60, 80)` could enlarge the expected artifact set and have an
    unregistered K80 fit accepted as required output.
    """
    # The dispatcher and the authorization gate are the **sealed store methods**,
    # captured unbound at definition time and applied to whatever was handed in.
    # Resolving them here as `store_module.store_kind` /
    # `store_module.assert_formal_execution_authorized` made the last check
    # before the first fit a rebindable module attribute (E.2b.3 §5, §7); looking
    # them up on the *instance* would let a duck-typed object supply its own.
    #
    # The decision is a **bool** decided inside the store's factory closure
    # (E.2b.5). It used to be a string compared against a literal here, and the
    # string came from `store_module.FORMAL`: rebinding that module global made
    # `kind == "formal"` false for a genuine formal store, so this pre-fit
    # authorization — and the barrier dispatch at the end of the run — were
    # simply skipped. There is no string left to poison; anything that is not a
    # sealed, attested, correctly-rooted store raises inside `_is_formal`.
    formal = _is_formal(store)                 # exact sealed type, never `is_formal`
    if formal:
        # §16.1 — the complete frozen binding (literal tag ref, freeze commit,
        # protocol and validator bytes, mechanism dependency closure, module
        # alias identities, K40/K60), re-established before the first formal fit
        # and before the first authoritative scientific write
        _formal_gate(store)
    nodes = store_module.assert_frozen_node_roles(store_module.frozen_node_roles())
    if nodes != (40, 60):                      # literal, local, re-asserted here
        raise inference.RQ3InferenceError(
            f"§9.6: execution requires the frozen node roles (40, 60), got {nodes}")
    if set(populations) != set(data.POPULATIONS):
        raise inference.RQ3InferenceError(
            f"expected populations {sorted(data.POPULATIONS)}, got {sorted(populations)}"
        )

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    started = time.perf_counter()
    summaries, timings = [], {}

    for population_name in data.POPULATIONS:
        population = populations[population_name]
        if population.name != population_name:
            raise inference.RQ3InferenceError(
                f"population {population_name!r} carries the name {population.name!r}"
            )
        for operating_point, method in proto.CELL_REGISTRY:
            cell_started = time.perf_counter()
            say(f"{population_name} {operating_point}/{method}")
            outcome = outcome_builder(population, operating_point, method)

            per_nodes = {}
            for n_nodes in nodes:
                cell_fits = inference.run_cell(population, outcome, n_nodes)
                per_nodes[n_nodes] = cell_fits
                _persist_cell_fits(store, cell_fits)

            summary = inference.summarize_cell(
                population, outcome,
                per_nodes[proto.GH_NODES_PRIMARY],
                per_nodes[proto.GH_NODES_VALIDATION],
            )
            _persist_cell_summary(store, summary)
            summaries.append(summary)
            timings[f"{population_name}|{operating_point}|{method}"] = \
                round(time.perf_counter() - cell_started, 3)

    say("holm families")
    families = inference.holm_families(summaries)
    for family in families:
        store.put(
            store_module.FAMILIES_DIR,
            store_module.family_key(family["population"], family["block"]),
            family,
        )

    # §16.2 — the barrier belongs to the formal store alone. A non-formal store
    # verifies its records and its ledger and stops there: it has no
    # `complete_run`, and its writer refuses the marker's filename outright.
    if formal:
        say("completion barrier")
        completion = store.complete_run()
    else:
        say("non-formal seal (no completion marker)")
        completion = store.seal_non_formal()

    return {
        "n_cells": len(summaries),
        "n_families": len(families),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "per_cell_seconds": timings,
        "authoritative": bool(formal),
        "completion": completion,
    }


def _persist_cell_fits(store, cell: inference.CellFits) -> None:
    """One record per fit and per cross-validation, keyed by the frozen registry."""
    for name, fit in sorted(cell.fits.items()):
        payload = fit.to_payload()
        payload.update({
            "population": cell.population,
            "operating_point": cell.operating_point,
            "method": cell.method,
            "model": name,
            "formula": models.model_by_name(cell.population, name).formula,
            "terms": list(models.model_by_name(cell.population, name).terms),
            "design_columns": list(data.design_columns(
                models.model_by_name(cell.population, name).terms)),
            "transform": cell.transform.to_payload(),
        })
        store.put(
            store_module.FITS_DIR,
            store_module.fit_key(cell.population, cell.operating_point,
                                 cell.method, name, cell.n_nodes),
            payload,
        )
    for name, cv in sorted(cell.cross_validations.items()):
        payload = cv.to_payload()
        payload.update({
            "operating_point": cell.operating_point,
            "method": cell.method,
            "scheme": "leave-one-site-out",
            "aggregation": "site-macro-average",
        })
        store.put(
            store_module.CV_DIR,
            store_module.cv_key(cell.population, cell.operating_point,
                                cell.method, name, cell.n_nodes),
            payload,
        )


def _persist_cell_summary(store, summary: inference.CellSummary) -> None:
    """The cell's §15 outputs, plus one numerical-validation record per block."""
    for block_name, result in sorted(summary.blocks.items()):
        store.put(
            store_module.VALIDATION_DIR,
            store_module.validation_key(summary.population, summary.operating_point,
                                        summary.method, block_name),
            {
                "population": summary.population,
                "operating_point": summary.operating_point,
                "method": summary.method,
                "block": block_name,
                "status": result.status,
                "failing_condition": result.failing_condition,
                "authoritative_nodes": int(proto.AUTHORITATIVE_NODES),
                "validation_nodes": int(proto.GH_NODES_VALIDATION),
                "numerical_validation": result.numerical_validation or {
                    "passed": False,
                    "n_evaluated": 0,
                    "n_required": int(proto.N_REQUIRED_AGREEMENT_CRITERIA),
                    "n_passed": 0,
                    "reason": result.failing_condition,
                    "criteria": {},
                },
            },
        )
    store.put(
        store_module.CELLS_DIR,
        store_module.cell_key(summary.population, summary.operating_point,
                              summary.method),
        summary.to_payload(),
    )


def build_frozen_populations(progress=None) -> dict:
    """The two frozen complete-case subsets, with their counts re-asserted (H2-H4)."""
    if progress is not None:
        progress("binding frozen populations")
    return {name: data.build_population(name) for name in data.POPULATIONS}
