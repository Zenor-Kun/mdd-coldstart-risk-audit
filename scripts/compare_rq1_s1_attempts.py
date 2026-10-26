#!/usr/bin/env python3
"""Compare the authoritative Stage 1.3a S1 rerun against superseded attempt 1.

Provenance comparison **only**. This script decides nothing: whatever it finds,
no protocol rule, threshold, grid or definition may be changed on the strength
of it. Its single job is to state, artifact by artifact, whether repairing the
target-diagnosis access protocol moved any number.

The expectation is identity. The repair changed *who may look at what and
when*, not a single scientific definition, so every decision quantity is a
function of exactly the same frozen inputs as before. A difference would mean
the old run's dataflow was not as clean as the review concluded, and is a
reason to stop and audit rather than to explain.

Usage:
    python3 scripts/compare_rq1_s1_attempts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NEW = ROOT / "results" / "rq1_s1"
OLD = ROOT / "results" / "rq1_s1_attempt1_superseded"
REPORT = NEW / "ATTEMPT_COMPARISON.md"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def compare_frame(name: str, keys: list[str], value_cols: list[str] | None = None
                  ) -> dict:
    """Align two versions of one artifact on `keys` and compare every value.

    Bitwise identity is checked first with `==` on the raw values, so a
    difference of one ULP is reported as a difference rather than rounded away.
    """
    a, b = _read(OLD / name), _read(NEW / name)
    out = {"artifact": name, "rows_old": len(a), "rows_new": len(b)}
    if len(a) != len(b):
        return {**out, "verdict": "ROW COUNT DIFFERS", "n_value_cells": 0,
                "n_cells_differing": None, "max_abs_diff": None}
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    if not all(a[k].astype(str).equals(b[k].astype(str)) for k in keys):
        return {**out, "verdict": "KEYS DIFFER", "n_value_cells": 0,
                "n_cells_differing": None, "max_abs_diff": None}

    cols = value_cols or [c for c in a.columns
                          if c in b.columns and c not in keys]
    n_cells = n_diff = 0
    worst = 0.0
    worst_col = ""
    for c in cols:
        va, vb = a[c], b[c]
        na = pd.to_numeric(va, errors="coerce")
        nb = pd.to_numeric(vb, errors="coerce")
        numeric = na.notna() & nb.notna()
        n_cells += len(va)
        # numeric cells: exact equality, then magnitude of any difference
        d = (na[numeric] != nb[numeric])
        if d.any():
            n_diff += int(d.sum())
            m = float(np.max(np.abs((na[numeric] - nb[numeric])[d])))
            if m > worst:
                worst, worst_col = m, c
        # non-numeric cells (NOT_ESTIMABLE, site names, flags): string equality
        s = (va[~numeric].astype(str) != vb[~numeric].astype(str))
        if s.any():
            n_diff += int(s.sum())
            worst_col = worst_col or c
    return {**out, "n_value_cells": n_cells, "n_cells_differing": n_diff,
            "max_abs_diff": worst, "worst_column": worst_col,
            "verdict": "IDENTICAL" if n_diff == 0 else "DIFFERS"}


def main() -> int:
    print("=" * 78)
    print("RQ1/S1 — authoritative Stage 1.3a rerun vs superseded attempt 1")
    print("=" * 78)

    comparisons = [
        ("selected_domain_C.csv", ["held_out_site"]),
        ("domain_meta_objective.csv", ["held_out_site", "C_domain"]),
        ("domain_fit_diagnostics.csv", ["held_out_site"]),
        ("density_weights.csv", ["held_out_site", "canonical_id"]),
        ("ess_by_site.csv", ["held_out_site"]),
        ("thresholds_weighted_source.csv", ["held_out_site", "method", "q"]),
        ("weighted_source_risk.csv", ["held_out_site", "method", "q"]),
        ("risk_coverage_all_q.csv", ["held_out_site", "method", "q"]),
        ("op50_summary.csv", ["method"]),
        ("op30_summary.csv", ["method"]),
        ("s1_vs_s0_contrasts.csv", ["operating_point", "quantity", "method"]),
        ("site_bootstrap_intervals.csv", ["quantity", "method_or_contrast"]),
        ("jsd_summary.csv", ["operating_point", "method"]),
        ("worst_site_summary.csv", ["operating_point", "method"]),
    ]
    rows = [compare_frame(name, keys) for name, keys in comparisons]
    table = pd.DataFrame(rows)

    for r in rows:
        flag = "==" if r["verdict"] == "IDENTICAL" else "!!"
        print(f"  [{flag}] {r['artifact']:44s} {r['verdict']:12s} "
              f"cells={r['n_value_cells']:>7} differing={r['n_cells_differing']}")

    n_diff_artifacts = int((table["verdict"] != "IDENTICAL").sum())
    total_cells = int(table["n_value_cells"].sum())
    total_diff = int(table["n_cells_differing"].fillna(-1).sum())

    L = ["# Stage 1.3a rerun vs Stage 1.3 attempt 1 — provenance comparison\n",
         "**Purpose**: provenance only. Nothing in the protocol may be changed "
         "on the strength of this table, whichever way it comes out.\n",
         "**What changed between the two runs**: the target-diagnosis *access "
         "protocol*, and nothing else. Attempt 1 materialized the full "
         "diagnosis vector pre-decision, counted the cohort composition from "
         "it, and held one source-OOF frame carrying `y_true` for all 23 sites; "
         "the rerun enforces an outer-fold-scoped firewall. No scientific "
         "definition — cohort, FC representation, disease classifier, S0 "
         "confidence scores, `r*_s`, `q` grid, domain PCA dimension, "
         "`C_domain` grid, meta-LOSO objective, weight formula, "
         "normalize→clip→STOP, weighted quantile, ESS, CI scope, the three "
         "confidence methods — was touched.\n",
         "\n| artifact | rows | value cells | cells differing | max |diff| | verdict |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| `{r['artifact']}` | {r['rows_new']} | "
                 f"{r['n_value_cells']} | {r['n_cells_differing']} | "
                 f"{'0' if not r['max_abs_diff'] else format(r['max_abs_diff'], '.3e')} | "
                 f"**{r['verdict']}** |")
    L.append(f"\n**{len(rows) - n_diff_artifacts}/{len(rows)} artifacts "
             f"identical**, {total_diff} differing cells out of {total_cells} "
             "compared.\n")
    if n_diff_artifacts == 0:
        L.append("Every decision quantity — the 23 selected `C_domain`, the "
                 "115 `J(C)` values, all 49 104 density weights, the 23 ESS "
                 "values, all 1173 weighted thresholds, the weighted source "
                 "risk, the 17-`q` target curve, both operating-point "
                 "summaries, the 36 intervals, the paired contrasts, the JSD "
                 "summary and the worst-site table — is **bitwise identical** "
                 "to the superseded attempt.\n")
        L.append("This is the expected result and is the strongest available "
                 "confirmation of the closure review's finding: attempt 1's "
                 "failure was one of access discipline and provenance, not of "
                 "dataflow. No target label had in fact entered any S1 "
                 "decision — had one done so, sealing that door would have "
                 "moved a number here.\n")
    else:
        L.append("**A difference was found. Stop and audit.** The repair "
                 "changed only who may read what and when, so any moved number "
                 "means a target quantity was reaching a decision in attempt 1. "
                 "Do not reconcile the two runs by adjusting the protocol.\n")
    L.append("\n## Status of the two runs\n")
    L.append("- `results/rq1_s1/` — **authoritative**, Stage 1.3a.\n")
    L.append("- `results/rq1_s1_attempt1_superseded/` — **NON-COMPLIANT / "
             "SUPERSEDED**, preserved as provenance, never citable as a result; "
             "see its `SUPERSEDED.md` and `INVALID_ASSERTIONS.csv`.\n")
    REPORT.write_text("\n".join(L) + "\n")
    table.to_csv(NEW / "attempt_comparison.csv", index=False)

    print(f"\n[compare] {len(rows) - n_diff_artifacts}/{len(rows)} artifacts "
          f"identical; {total_diff}/{total_cells} value cells differ")
    print(f"[compare] {REPORT}")
    return 1 if n_diff_artifacts else 0


if __name__ == "__main__":
    raise SystemExit(main())
