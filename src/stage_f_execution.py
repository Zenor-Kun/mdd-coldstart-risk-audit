"""Stage-F orchestration — the three modules, then the persistence barrier.

The three scientific modules are executed through their own namespaces and never
through each other: `stage_f_conformal` (F-RQ4), `stage_f_sensitivities` (F-SENS)
and `stage_f_svm` (F-SVM). No result of one qualifies, rescues or reinterprets
another; they share one freeze, one store and one barrier, not one question.

There is **no automatic rerun**: no failed cell is retried, no partial store is
resumed and none is cleaned. If a Formal run fails, all evidence is preserved,
the partial store is moved to `results/failed_runs/stage_f_<timestamp>/`
untouched, and independent diagnosis is required before any rerun (§J.2).
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import stage_f_conformal as conformal
import stage_f_inputs as inputs
import stage_f_registry as reg
import stage_f_sensitivities as sens
import stage_f_store as store_module
import stage_f_svm as svm

ROOT = reg.ROOT


class FormalRunRefused(RuntimeError):
    """A Formal Stage-F run refused to start. Nothing was written."""


def _say(progress):
    def emit(message):
        if progress is not None:
            progress(message)
    return emit


def run_all(store, *, work_root: Path, progress=None, n_jobs: int = 1) -> dict:
    """Execute F-RQ4, F-SENS and F-SVM and persist every registered artifact.

    The barrier is **not** run here: `complete_run()` belongs to the formal
    store alone, and a smoke store has no such method at all.
    """
    say = _say(progress)
    sens.assert_closed_set()
    reachability = sens.reachability_report()

    say("F-RQ4 conformal robustness")
    rq4 = conformal.run(progress=progress)
    store_module.persist_rq4(store, rq4)

    say("F-SENS sensitivities")
    target = inputs.load_target_confidence()
    registry = inputs.load_split_registry()
    blocks = None
    for label in (sens.SENS1, sens.SENS2, sens.SENS3, sens.SENS4):
        say(f"F-SENS {label}")
        if blocks is None:
            import s2_inputs as si
            blocks = si.SourceOofBlocks()
        tables = sens.run_s2_sensitivity(label, blocks=blocks, registry=registry,
                                         target=target, progress=progress)
        store_module.persist_sensitivity(store, label, tables)

    import rq3_execution as rq3_execution

    populations = rq3_execution.build_frozen_populations(progress=progress)
    for label in (sens.SENS5, sens.SENS6):
        say(f"F-SENS {label}")
        scratch = Path(work_root) / label
        tables = sens.run_rq3_sensitivity(label, scratch, populations=populations,
                                          target=target, progress=progress)
        store_module.persist_sensitivity(store, label, tables)

    say("F-SVM robustness comparator")
    result = svm.run(n_jobs=n_jobs, progress=progress)
    store_module.persist_svm(store, result)

    return {"reachability": reachability,
            "n_registered_artifacts": store.n_artifacts_written,
            "rq4_cells": reg.N_REGISTERED_CELLS,
            "rq4_ci_slots": reg.N_CI_SLOTS,
            "sensitivities": list(reg.SENSITIVITY_LABELS)}


def execute_formal(*, progress=None, n_jobs: int = 1) -> dict:
    """The authoritative Stage-F run. The store root is not a parameter.

    Preflight (§J.1) runs first and any failure aborts before a single artifact
    is written. On failure after writing began, the partial store is preserved
    under `results/failed_runs/` and never becomes authoritative.
    """
    say = _say(progress)
    say("preflight")
    preflight = inputs.preflight(require_empty_store=True)

    store = store_module.FormalStageFStore.create_authoritative(
        store_module.build_provenance(reg.RUN_KIND_FORMAL,
                                      {"preflight": preflight}))
    work_root = Path(tempfile.mkdtemp(prefix="stage_f_work_"))
    try:
        summary = run_all(store, work_root=work_root, progress=progress,
                          n_jobs=n_jobs)
        say("completion barrier")
        completion = store.complete_run()
    except BaseException:
        _preserve_failed_store(store.root)
        raise
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    return {"authoritative": True, "completion": completion, **summary}


def _preserve_failed_store(root: Path) -> Path | None:
    """§J.2 — preserve all evidence, never delete, never make it authoritative."""
    if not Path(root).exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "results" / "failed_runs" / f"stage_f_{stamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(root), str(destination))
    return destination


def execute_smoke(store_root, *, progress=None, n_jobs: int = 1) -> dict:
    """A non-authoritative rehearsal. Writes no `COMPLETION.json`, ever.

    `SmokeStageFStore` refuses the authoritative root and has no `complete_run`,
    so this path is structurally incapable of producing an authoritative store.
    """
    say = _say(progress)
    say("preflight (smoke)")
    inputs.preflight(require_empty_store=False, store_root=Path(store_root))
    store = store_module.SmokeStageFStore.create(store_root)
    work_root = Path(tempfile.mkdtemp(prefix="stage_f_smoke_"))
    try:
        summary = run_all(store, work_root=work_root, progress=progress,
                          n_jobs=n_jobs)
        seal = store.seal_smoke()
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    return {"authoritative": False, "seal": seal, **summary}
