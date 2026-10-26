"""Post-hoc first-test decomposition of the frozen S2 target_local decisions
(source data for Fig. 2 panel (e); certifies nothing).

Every TARGET_UNCERTIFIED cell stopped at its first eligible candidate. For that
candidate we reconstruct the retained class counts (n_MDD, n_HC) on D_s^cal and
the zero-error floor U_min = 1 - [0.05^(1/n_MDD) + 0.05^(1/n_HC)]/2, i.e. the
smallest U_BER attainable with no observed errors. A TU cell is "zero-error
infeasible" if U_min > r*_s, otherwise "feasible but failed on observed errors".
The reconstruction must reproduce the frozen first eligible index and U_BER of
every decision record, or the script fails without writing.

Inputs (frozen, read-only): results/rq2_s2/{decisions/,production_split_registry.csv}
                            results/rq1_s0/{confidence_target,thresholds_source_oof}.csv
Output: paper/ISBI_template-master/revision_support/source_data/posthoc_summary.csv
"""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper/ISBI_template-master/revision_support/source_data/posthoc_summary.csv"
Q_GRID = [round(0.80 - 0.05 * i, 2) for i in range(17)]
METHODS = ("probability", "bootstrap", "mahalanobis")
C_MIN, M_MIN, ALPHA_CLASS = 0.20, 3, 0.05


def cp_upper(x: int, n: int) -> float:
    return 1.0 if x == n else float(beta.ppf(1 - ALPHA_CLASS, x + 1, n - x))


def load_decisions() -> dict:
    out = {}
    for f in (RES / "rq2_s2/decisions").glob("decision__*__target_local__*.json"):
        d = json.loads(f.read_text())
        out[(d["site"], d["repeat"], d["n_labels"], d["method"])] = d
    return out


def main() -> None:
    reg = pd.read_csv(RES / "rq2_s2/production_split_registry.csv")
    conf = pd.read_csv(RES / "rq1_s0/confidence_target.csv").set_index("canonical_id")
    th = pd.read_csv(RES / "rq1_s0/thresholds_source_oof.csv")
    th = th[th["q"] != "NO_ABSTENTION"].astype({"q": float})
    tau = {k: g.set_index("q")["tau"] for k, g in th.groupby(["held_out_site", "method"])}
    dec = load_decisions()
    assert len(dec) == 3 * len(reg), (len(dec), len(reg))

    rows = []
    for _, r in reg.iterrows():
        c = conf.loc[r["cal_ids"].split(";")]
        y, err, n = c["y_true"].to_numpy(), (c["pred"] != c["y_true"]).to_numpy(), int(r["n_labels"])
        for m in METHODS:
            d = dec[(r["site"], int(r["repeat"]), n, m)]
            t = tau[(r["site"], m)]
            fe = None
            for i, q in enumerate(Q_GRID):
                k = c[f"confidence_{m}"].to_numpy() >= t[q]
                nM, nH = int((k & (y == 1)).sum()), int((k & (y == 0)).sum())
                if k.sum() / n >= C_MIN and nM >= M_MIN and nH >= M_MIN:
                    fe = i
                    break
            assert fe == d["first_eligible_index"], (d["site"], d["repeat"], n, m, fe)
            row = dict(N=n, state=d["state"])
            if fe is not None:
                xM, xH = int((k & (y == 1) & err).sum()), int((k & (y == 0) & err).sum())
                u = 0.5 * (cp_upper(xM, nM) + cp_upper(xH, nH))
                assert abs(u - d["score_by_q"][str(Q_GRID[fe])]) < 1e-12, (d["site"], n, m)
                row["zero_error_infeasible"] = 1 - 0.5 * (0.05 ** (1 / nM) + 0.05 ** (1 / nH)) > d["r_star"]
            rows.append(row)

    df = pd.DataFrame(rows)
    assert set(df["state"]) <= {"NOT_TESTABLE", "TARGET_UNCERTIFIED"}, set(df["state"])
    tu = df[df["state"] == "TARGET_UNCERTIFIED"]
    summary = pd.DataFrame({
        "Not testable": df[df["state"] == "NOT_TESTABLE"].groupby("N").size(),
        "Zero-error infeasible": tu[tu["zero_error_infeasible"] == True].groupby("N").size(),
        "Observed-error failure": tu[tu["zero_error_infeasible"] == False].groupby("N").size(),
    }).reindex(sorted(df["N"].unique())).fillna(0).astype(int).rename_axis("N").reset_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)
    print(summary.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
