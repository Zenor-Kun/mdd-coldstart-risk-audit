#!/usr/bin/env python3
"""The authoritative RQ3 execution — Stage E.3, NOT run at Stage E.2b.

This script produces the formal RQ3 result store. Stage E.2b is implementation
alignment and mechanical validation only, so it is written, tested and left
unexecuted:

* it refuses to run without `--confirm-formal-execution`, so an accidental
  invocation cannot create `results/rq3/`;
* `scripts/validate_rq3_protocol_freeze.py` — part of the green suite — fails
  while `results/rq3/` exists, which is itself a standing guard that no
  implementation-stage command created it.

**The output root is not a parameter (§16.1.1).** There is no `--store-root`, no
environment-variable override and no configuration key by which the
authoritative location can be redirected: formal results are written to
`results/rq3/` or the run aborts (H13, H23). A runner that would accept an
alternative root "for testing" has already violated that clause, which is why
smoke and unit-test execution go through `rq3_store.NonFormalRQ3Store` — a
different API that refuses `results/rq3/` and cannot write a completion marker
at all — rather than through a flag on this script.

**A stale root fails closed (§16.1.3).** If `results/rq3/` already exists —
empty, partial, complete, or left behind by an aborted attempt — the run
hard-fails before writing anything (H24). It never merges into, reuses,
resumes, repairs, back-fills or overwrites a prior result tree, and it never
inspects one to decide what to skip. Archiving or removing a previous attempt is
a separate, explicit, operator-initiated action taken before the run.

What it does, once authorised:

    verify the §1 consumed artifacts against their frozen SHA256 and the S0
    ledger (H1); re-derive the 2232/23 base cohort (H2) and the frozen 2222/23
    and 484/9 complete-case subsets (H3, H4); confirm the reconstructed `R_i`
    reproduces the published S0 operating-point coverage in all six cells (H5);
    then run every registered cell at K = 40 (authoritative) and K = 60
    (numerical validation) under the frozen mode/curvature-adaptive quadrature
    of §9.2, persist every record, and run the §16.2 completion barrier — which
    re-reads every artifact and the ledger from disk before `COMPLETION.json`,
    the terminal artifact, is written atomically.

    python3 scripts/run_rq3.py --confirm-formal-execution
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rq3_data as data              # noqa: E402
import rq3_execution as execution    # noqa: E402
import rq3_protocol as proto         # noqa: E402
import rq3_store as store_module     # noqa: E402


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # §16.1.1 — deliberately no --store-root and no output-location argument of
    # any other name. The authoritative root is a constant of the protocol.
    parser.add_argument(
        "--confirm-formal-execution", action="store_true",
        help="required; without it this script refuses to run",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.confirm_formal_execution:
        print(
            "REFUSED  the authoritative RQ3 execution was not confirmed.\n"
            "         Stage E.2b is implementation alignment and mechanical\n"
            "         validation only; pass --confirm-formal-execution to run\n"
            "         the formal stage.",
            file=sys.stderr,
        )
        return 2

    def say(message: str) -> None:
        if not args.quiet:
            print(f"  ..  {message}", flush=True)

    say(f"protocol {proto.PROTOCOL_VERSION} ({proto.EXPECTED_PROTOCOL_TAG})")
    say(f"quadrature {proto.QUADRATURE_RULE}, K={proto.AUTHORITATIVE_NODES} "
        f"authoritative / K={proto.GH_NODES_VALIDATION} validation")

    # H1 — the consumed artifacts, re-hashed now rather than trusted
    inputs = store_module.input_provenance()
    say(f"inputs: {len(inputs)} frozen artifacts verified")

    # H5 — the reconstructed outcome reproduces the published S0 coverage
    coverage = data.assert_outcome_reproduces_s0()
    say(f"outcome: {len(coverage)} cells reproduce the frozen S0 coverage")

    populations = execution.build_frozen_populations(progress=say)
    for name, population in sorted(populations.items()):
        say(f"population {name}: n={population.n}, sites={population.n_sites}")

    # §16.1.1 / §16.1.3 — the root is taken from the freeze, and a pre-existing
    # tree aborts the run here, before anything is written
    # the frozen identity fields — protocol tag, freeze commit, document and
    # validator hashes, node identities, the authoritative flag, and the input
    # and code provenance — are **not** passed in: the store derives them and
    # refuses a caller that tries to supply them (§16.1)
    store = store_module.FormalRQ3Store.create_authoritative({
        "stage": "E.3",
        "run_kind": "AUTHORITATIVE",
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outcome_coverage_discrepancy": coverage,
        "populations": {
            name: {"n": population.n, "n_sites": population.n_sites,
                   "sites": list(population.sites)}
            for name, population in sorted(populations.items())
        },
    })

    result = execution.execute(store, populations, progress=say)

    print(f"\nRQ3 authoritative execution — {proto.PROTOCOL_VERSION}")
    print(f"  cells      {result['n_cells']}")
    print(f"  families   {result['n_families']}")
    print(f"  elapsed    {result['elapsed_seconds']:.1f}s")
    print(f"  store      {store.root}")
    print(f"  completion {store_module.COMPLETION_FILE} "
          f"({'written' if store.complete else 'ABSENT'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
