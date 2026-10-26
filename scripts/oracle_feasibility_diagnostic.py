"""Post-hoc oracle feasibility diagnostic (spec: 'oracle-infeasible' is a
post-hoc diagnostic only, never a headline certification metric).

For every held-out site and confidence score, using ALL target labels and the
frozen 17 source-fixed S0 candidate thresholds, ask whether any candidate has
full-site selective balanced error <= r*_s. This is an optimistic upper bound
on feasibility (point estimates, best of 17 candidates); it uses no
calibration split and certifies nothing.

Input (frozen, read-only): results/rq1_s0/risk_coverage_all_q.csv
Outputs: paper/ISBI_template-master/revision_support/source_data/
           oracle_feasibility_per_site.csv, oracle_feasibility_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/rq1_s0/risk_coverage_all_q.csv"
OUT = ROOT / "paper/ISBI_template-master/revision_support/source_data"
M_MIN_PLAN = 9  # zero-error CP floor with equal class counts at r* ~ 0.31


def main() -> None:
    d = pd.read_csv(SRC)
    d["ber"] = pd.to_numeric(d["BER_sel_target"], errors="coerce")  # NaN = a class absent
    d["feasible"] = d["ber"] <= d["r_star_s"]
    rows = []
    for (m, s), g in d.groupby(["method", "held_out_site"]):
        f = g[g["feasible"]]
        n_min = f[["n_keep_MDD", "n_keep_HC"]].min(axis=1)
        rows.append(dict(
            method=m, site=s, n_s=int(g["n_s"].iloc[0]),
            r_star=float(g["r_star_s"].iloc[0]),
            best_ber=float(g["ber"].min()),
            oracle_feasible=bool(len(f)),
            n_feasible_candidates=int(len(f)),
            max_feasible_coverage=float(f["coverage_target"].max()) if len(f) else np.nan,
            q_at_max_feasible_coverage=float(f.loc[f["coverage_target"].idxmax(), "q"]) if len(f) else np.nan,
            feasible_with_min_class_ge9=bool((n_min >= M_MIN_PLAN).any()),
        ))
    per = pd.DataFrame(rows).sort_values(["method", "site"])
    per.to_csv(OUT / "oracle_feasibility_per_site.csv", index=False)
    summ = per.groupby("method").agg(
        sites=("site", "size"),
        oracle_feasible_sites=("oracle_feasible", "sum"),
        median_max_feasible_coverage=("max_feasible_coverage", "median"),
        min_max_feasible_coverage=("max_feasible_coverage", "min"),
        max_max_feasible_coverage=("max_feasible_coverage", "max"),
        feasible_with_min_class_ge9=("feasible_with_min_class_ge9", "sum"),
    )
    summ.to_csv(OUT / "oracle_feasibility_summary.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
