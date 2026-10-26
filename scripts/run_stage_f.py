#!/usr/bin/env python3
"""The authoritative Stage-F execution — NOT run at Stage F.2.

Stage F.2 is implementation and mechanical validation only, so this script is
written, tested and left unexecuted. Three guards keep an accidental invocation
from producing an authoritative store:

* it refuses to run without **both** `--formal` and
  `--i-understand-this-is-authoritative` (§J.2), so no default action can begin
  a Formal run;
* **the output root is not a parameter** (§I.2). There is no `--out`, no
  `--root`, no `--store-root` and no environment override by which the
  authoritative location could be redirected: Formal results go to
  `results/stage_f_v2/` or the run aborts. Smoke execution goes through
  `stage_f_store.SmokeStageFStore`, a different API that refuses that root and
  is structurally incapable of writing `COMPLETION.json`;
* **a stale root fails closed** (§I.2). If `results/stage_f_v2/` exists and is
  non-empty — empty-but-present, partial, complete, or left by an aborted
  attempt — the run hard-fails before writing anything. It never merges into,
  resumes, repairs, back-fills or overwrites a prior tree, and it never inspects
  one to decide what to skip. Archiving a previous attempt is a separate,
  explicit operator action taken before the run.

What it does, once authorised: run the §J.1 preflight over the 17 frozen inputs
(including the genuine Gate-1 content binding and the FC cache identity key),
then execute F-RQ4, the six sensitivities and F-SVM into `results/stage_f_v2/`,
and run the §I.1 seven-step barrier — which re-reads every artifact from disk and
recomputes its digest before `COMPLETION.json`, the terminal artifact, is
written atomically.

    python3 scripts/run_stage_f.py --preflight          # read-only, writes nothing
    python3 scripts/run_stage_f.py --smoke /tmp/stage_f_smoke
    python3 scripts/run_stage_f.py --formal --i-understand-this-is-authoritative
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import stage_f_execution as execution   # noqa: E402
import stage_f_inputs as inputs         # noqa: E402
import stage_f_registry as reg          # noqa: E402

CONFIRMATION_FLAG = "--i-understand-this-is-authoritative"


def _progress(message: str) -> None:
    print(f"[stage-f] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_stage_f.py",
        description="Stage F — conformal robustness, the six registered "
                    "sensitivities and the linear-SVM comparator.")
    parser.add_argument("--preflight", action="store_true",
                        help="run the §J.1 input preflight and stop. Read-only.")
    parser.add_argument("--smoke", metavar="ROOT", default=None,
                        help="non-authoritative rehearsal into ROOT. Refuses "
                             "results/stage_f_v2/ and writes no COMPLETION.json.")
    parser.add_argument("--formal", action="store_true",
                        help="authoritative execution into results/stage_f_v2/. "
                             f"Requires {CONFIRMATION_FLAG}.")
    parser.add_argument(CONFIRMATION_FLAG, dest="confirmed", action="store_true",
                        help="explicit confirmation that a Formal run is "
                             "authoritative and irreversible.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="parallel outer folds for the SVM comparator.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.formal and args.smoke is not None:
        print("REFUSED  --formal and --smoke are mutually exclusive.",
              file=sys.stderr)
        return 2

    if args.preflight:
        report = inputs.preflight(require_empty_store=False)
        print(json.dumps({k: v for k, v in report.items()
                          if k != "consumed_artifacts"},
                         indent=2, sort_keys=True, default=str))
        print("PREFLIGHT: PASS — nothing was written, no Stage-F quantity "
              "was computed.")
        return 0

    if args.smoke is not None:
        result = execution.execute_smoke(args.smoke, progress=_progress,
                                         n_jobs=args.n_jobs)
        marker = Path(args.smoke) / reg.COMPLETION_FILE
        if marker.exists():                      # defence in depth, not a hope
            print(f"REFUSED  a smoke run produced {marker}", file=sys.stderr)
            return 1
        print(json.dumps(result["seal"], indent=2, sort_keys=True, default=str))
        print("SMOKE: complete — non-authoritative, no COMPLETION.json written.")
        return 0

    if not args.formal:
        print(__doc__)
        print("No action taken. A Formal run requires both --formal and "
              f"{CONFIRMATION_FLAG}.")
        return 0

    if not args.confirmed:
        print(f"REFUSED  --formal requires {CONFIRMATION_FLAG}. A Formal Stage-F "
              f"run writes the authoritative {reg.STORE_ROOT} store and is not "
              "repeatable: it never overwrites, merges, resumes or cleans.",
              file=sys.stderr)
        return 2

    result = execution.execute_formal(progress=_progress, n_jobs=args.n_jobs)
    print(json.dumps(result["completion"], indent=2, sort_keys=True, default=str))
    print(f"FORMAL: complete — {reg.STORE_ROOT} is authoritative.")
    return 0


if __name__ == "__main__":                            # pragma: no cover
    raise SystemExit(main())
