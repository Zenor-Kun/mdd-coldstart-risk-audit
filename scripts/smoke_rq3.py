#!/usr/bin/env python3
"""NON-AUTHORITATIVE RQ3 smoke benchmark.

**This produces no RQ3 scientific result and no RQ3 conclusion.** It runs the
production orchestration path on **synthetic** populations and a **synthetic**
outcome drawn independently of every covariate, so nothing it computes is an
association between anything real. Its purposes are:

    numerical execution works    every registered model fits at both node counts
    runtime is practical         a measured per-fit and per-cell cost
    persistence wiring works     records and the checksum ledger
    the 40/60 path works         both node counts fitted, seven-of-seven verdict

Three guards keep it from being mistaken for the real thing:

* it writes through `rq3_store.NonFormalRQ3Store`, a **different API** from the
  formal store (§16.1.2) — it refuses `results/rq3/` and every path inside it,
  and it is structurally incapable of writing `COMPLETION.json`, which is the
  only admissible evidence that a formal RQ3 run completed (§16.2);
* it takes no argument that could redirect output into the authoritative root,
  and `scripts/validate_rq3_protocol_freeze.py` fails while that root exists;
* every record it writes is stamped `run_kind = SMOKE_SYNTHETIC`.

It also times one real-sized synthetic fit (2222 rows / 23 sites and 484 rows /
9 sites) so the formal run's cost can be projected without running it.

    python3 scripts/smoke_rq3.py
    python3 scripts/smoke_rq3.py --store-root /tmp/rq3_smoke --keep
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rq3_data as data              # noqa: E402
import rq3_execution as execution    # noqa: E402
import rq3_glmm as glmm              # noqa: E402
import rq3_models as models          # noqa: E402
import rq3_protocol as proto         # noqa: E402
import rq3_store as store_module     # noqa: E402
import rq3_synthetic as synthetic    # noqa: E402

RUN_KIND = "SMOKE_SYNTHETIC"


def assert_not_the_authoritative_store(root: Path) -> None:
    """The smoke may never write where the formal execution will.

    `NonFormalRQ3Store.create` refuses the authoritative root independently;
    this is the earlier, friendlier message for a command-line mistake.
    """
    authoritative = store_module.authoritative_root()
    resolved = Path(root).resolve()
    if resolved == authoritative or authoritative in resolved.parents \
            or resolved in authoritative.parents:
        raise SystemExit(
            f"REFUSED  {resolved} is (or contains) the authoritative store "
            f"{authoritative}; the smoke is non-authoritative and never writes there."
        )


def benchmark_real_sized_fit(quiet: bool) -> dict:
    """Time one fit at each frozen population's real shape, on synthetic data.

    Row counts and site counts match the frozen subsets so the projection is
    honest; the values are random, so nothing scientific is produced.
    """
    timings = {}
    shapes = (
        ("full_sample_shape", 23, proto.FULL_CC_N // 23, 4),
        ("mdd_only_shape", 9, proto.MDD_CC_N // 9, 5),
    )
    for label, n_sites, n_per_site, n_slopes in shapes:
        drawn = synthetic.simulate_glmm(
            n_sites=n_sites, n_per_site=n_per_site,
            beta=tuple([-0.3] + [0.2] * n_slopes), sigma_site=0.5, seed=11,
        )
        for n_nodes in (proto.GH_NODES_PRIMARY, proto.GH_NODES_VALIDATION):
            started = time.perf_counter()
            fit = glmm.fit_glmm(drawn["X"], drawn["R"], drawn["sites"], n_nodes)
            elapsed = time.perf_counter() - started
            timings[f"{label}|K{n_nodes}"] = round(elapsed, 3)
            if not quiet:
                print(f"  ..  {label} n={len(drawn['R'])} sites={n_sites} "
                      f"K={n_nodes}: {elapsed:.2f}s ({fit.status})")
    return timings


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", default=None,
                        help="defaults to a temporary directory that is removed")
    parser.add_argument("--n-per-site", type=int, default=6)
    parser.add_argument("--keep", action="store_true",
                        help="keep the temporary store instead of removing it")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    temporary = None
    if args.store_root is None:
        temporary = tempfile.mkdtemp(prefix="rq3_smoke_")
        root = Path(temporary)
    else:
        root = Path(args.store_root)
    assert_not_the_authoritative_store(root)

    def say(message: str) -> None:
        if not args.quiet:
            print(f"  ..  {message}", flush=True)

    print(f"RQ3 smoke — {RUN_KIND} — {proto.PROTOCOL_VERSION} "
          f"({proto.EXPECTED_PROTOCOL_TAG})")
    print(f"  quadrature: {proto.QUADRATURE_RULE}")
    print("  no real R_i is read; no RQ3 association is produced\n")

    try:
        say("timing one real-sized fit per population shape (synthetic values)")
        fit_timings = benchmark_real_sized_fit(args.quiet)

        populations = synthetic.synthetic_populations(n_per_site=args.n_per_site)
        for name, population in sorted(populations.items()):
            say(f"synthetic {name}: n={population.n}, sites={population.n_sites}")

        store = store_module.NonFormalRQ3Store.create(root, {
            "stage": "E.2b",
            "run_kind": RUN_KIND,
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": ("synthetic populations and a covariate-independent synthetic "
                     "outcome; no frozen artifact is read and no RQ3 result is produced"),
            "real_sized_fit_seconds": fit_timings,
        })

        started = time.perf_counter()
        result = execution.execute(
            store, populations,
            outcome_builder=synthetic.synthetic_outcome_builder(seed=3),
            progress=say,
        )
        elapsed = time.perf_counter() - started
        n_verified = result["completion"]["n_files_verified"]

        expected = store_module.expected_keys()
        print("\n  persistence")
        for kind in store_module.RECORD_DIRS:
            print(f"    {kind:<12} {len(store.keys(kind)):>3} records "
                  f"(expected {len(expected[kind])})")
        print(f"    checksums    {n_verified:>3} files verified")
        print(f"    completion   ABSENT by construction — a non-formal store "
              f"cannot write {store_module.COMPLETION_FILE}")

        print("\n  execution")
        print(f"    synthetic cells      {result['n_cells']}")
        print(f"    Holm families        {result['n_families']}")
        print(f"    synthetic wall clock {elapsed:.1f}s")

        print("\n  projected formal cost, from the real-sized fit timings")
        projection = _project(fit_timings)
        for label, seconds in projection.items():
            print(f"    {label:<28} {seconds / 60.0:8.1f} min")

        print("\nPASS  mechanical execution, 40/60 fitting and persistence all wired.")
        print("      Nothing above is an RQ3 result.")
        return 0
    finally:
        if temporary is not None and not args.keep:
            shutil.rmtree(temporary, ignore_errors=True)


def _project(fit_timings: dict) -> dict:
    """Project the formal run's cost from measured per-fit times.

    Counts come from the frozen registries: per cell and node count, one
    full-data fit per registered model plus one LOSO fold fit per
    cross-validated model per site.
    """
    projection = {}
    for population, label in ((data.FULL_SAMPLE, "full_sample_shape"),
                              (data.MDD_ONLY, "mdd_only_shape")):
        n_sites = synthetic.SYNTHETIC_SITE_COUNT[population]
        n_full_data = len(models.models_to_fit(population))
        n_cv_models = len(models.models_to_cross_validate(population))
        total = 0.0
        for n_nodes in (proto.GH_NODES_PRIMARY, proto.GH_NODES_VALIDATION):
            per_fit = fit_timings[f"{label}|K{n_nodes}"]
            fits_per_cell = n_full_data + n_cv_models * n_sites
            total += per_fit * fits_per_cell * len(proto.CELL_REGISTRY)
        projection[population] = total
    projection["TOTAL"] = float(np.sum(list(projection.values())))
    return projection


if __name__ == "__main__":
    sys.exit(main())
