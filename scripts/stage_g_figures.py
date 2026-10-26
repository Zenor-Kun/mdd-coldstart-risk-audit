#!/usr/bin/env python3
"""Stage G — render Fig. 1 and Fig. 2 for the 4-page ISBI manuscript.

Consumes **only** the Stage-G source-data CSVs written by
``scripts/stage_g_build_source_data.py``. It performs no analysis, no
smoothing, no interpolation and no confidence-interval computation: every
plotted value is a row of a source-data file, and every source-data row is
traceable to a frozen CLOSED-stage authority.

Outputs vector PDF (primary) plus a 600-dpi PNG companion under
``results/stage_g/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "stage_g" / "source_data"
FIG = ROOT / "results" / "stage_g" / "figures"

# --- validated categorical palette (dataviz six-checks: ALL PASS) ----------
METHOD_COLOR = {
    "probability": "#0072B2",
    "bootstrap": "#D55E00",
    "mahalanobis": "#009E73",
}
METHOD_MARKER = {"probability": "o", "bootstrap": "s", "mahalanobis": "^"}
METHOD_LABEL = {
    "probability": "max probability",
    "bootstrap": "bootstrap disagreement",
    "mahalanobis": "Mahalanobis distance",
}
# information budget is ORDERED -> sequential single-hue ramp, light to dark
REGIME_COLOR = {"S0": "#c6dbef", "S1": "#6baed6", "S2": "#08519c"}
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dddddd"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 6.0,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "grid.color": GRID,
    "grid.linewidth": 0.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def tidy(ax, grid_axis="both"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, length=2.2)


def save(fig, stem):
    # CreationDate is dropped so a re-render is byte-reproducible and an
    # auditor's regeneration produces no spurious diff.
    fig.savefig(FIG / f"{stem}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIG / f"{stem}.png", dpi=600)
    plt.close(fig)
    print(f"  wrote {stem}.pdf / {stem}.png")


# ==========================================================================
# Figure 1
# ==========================================================================
def figure1():
    aurc = pd.read_csv(SRC / "fig1_aurc_aggregation.csv")
    per_site = pd.read_csv(SRC / "fig1_per_site_aurc.csv")
    rc = pd.read_csv(SRC / "fig1_risk_coverage_site_balanced.csv")
    acc = pd.read_csv(SRC / "fig1_right_site_acceptance.csv")
    sites = pd.read_csv(SRC / "table1_sites.csv")

    # Guard: no acquisition-count-weighted selective-risk quantity may exist.
    assert not [c for c in rc.columns if "patient" in c.lower()]
    assert not [c for c in rc.columns if "weight" in c.lower()
                and c != "aggregation_rule"]

    r_star = sites["r_star_s"].to_numpy(float)
    ORDER = ["probability", "bootstrap", "mahalanobis"]

    fig = plt.figure(figsize=(7.16, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.02, 1.30], wspace=0.46)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ---- panel a: FROZEN patient-weighted vs site-balanced AURC ----------
    # Discrete estimate-and-interval plot. Deliberately not a curve: the two
    # aggregations of one method are never joined, and nothing is plotted
    # against coverage.
    for i, method in enumerate(ORDER):
        c = METHOD_COLOR[method]
        ps = per_site[per_site.method == method]["AURC_s"].to_numpy(float)
        axA.plot(ps, np.full(ps.shape, -i - 0.30), ls="none", marker="|",
                 ms=4.2, mew=0.6, color="#9a9a9a", zorder=2)
        for j, (agg, mfc) in enumerate((("patient_weighted", c),
                                        ("site_balanced", "white"))):
            r = aurc[(aurc.method == method) & (aurc.aggregation == agg)].iloc[0]
            y = -i + (0.22 if j == 0 else 0.00)
            axA.errorbar(r.aurc, y,
                         xerr=[[r.aurc - r.ci_low], [r.ci_high - r.aurc]],
                         fmt=METHOD_MARKER[method], ms=4.0, mfc=mfc, mec=c,
                         mew=0.9, ecolor=c, elinewidth=1.0, capsize=2.0,
                         zorder=3)
    axA.set_yticks([-i for i in range(3)])
    axA.set_yticklabels([METHOD_LABEL[m].replace(" ", "\n", 1) for m in ORDER],
                        fontsize=5.6)
    axA.set_ylim(-2.55, 1.35)
    axA.set_xlim(0.15, 0.60)
    axA.set_xlabel("AURC  (0/1-loss ranking quality)\n"
                   "bars: registered 95% site-cluster-bootstrap CI")
    axA.set_title("aggregation: patient-weighted vs site-balanced",
                  color=INK, pad=3, fontsize=6.8)
    tidy(axA, grid_axis="x")
    axA.legend(handles=[
        Line2D([], [], color=INK, marker="o", ms=4.0, mfc=INK, mew=0.9, ls="none",
               label="patient-weighted"),
        Line2D([], [], color=INK, marker="o", ms=4.0, mfc="white", mec=INK,
               mew=0.9, ls="none", label="site-balanced"),
        Line2D([], [], color="#9a9a9a", marker="|", ms=4.2, mew=0.6, ls="none",
               label="23 per-site $AURC_s$"),
    ], loc="upper left", frameon=False, handlelength=0.9, borderpad=0.1,
        labelspacing=0.3, handletextpad=0.4, fontsize=5.5)

    # ---- panel b: FROZEN site-balanced selective risk-coverage -----------
    axB.axhspan(r_star.min(), r_star.max(), color="#f0f0f0", zorder=0.5)
    axB.axhline(r_star.mean(), color=MUTED, lw=0.6, ls=(0, (4, 2)), zorder=0.6)
    for method in ORDER:
        c = METHOD_COLOR[method]
        for regime, ls, mfc in (("S0", "-", c), ("S1", (0, (3, 1.6)), "white")):
            g = rc[(rc.regime == regime) & (rc.method == method)].sort_values("q")
            axB.plot(g.coverage_site_balanced, g.ber_site_balanced, color=c,
                     ls=ls, lw=1.1, marker=METHOD_MARKER[method], ms=2.5,
                     mew=0.5, mfc=mfc, mec=c, zorder=3)
            red = g[g.ber_denominator_sites < g.n_sites_in_fold_set]
            axB.plot(red.coverage_site_balanced, red.ber_site_balanced, ls="none",
                     marker="o", ms=4.3, mfc="none", mec=MUTED, mew=0.45,
                     zorder=2.5)
    axB.set_xlim(0.10, 1.02)
    axB.set_ylim(0.255, 0.44)
    axB.set_xlabel("actual target coverage")
    axB.set_ylabel("selective balanced error (site-balanced)")
    axB.set_title("selective risk–coverage", color=INK, pad=3, fontsize=6.8)
    tidy(axB)
    axB.text(1.01, r_star.max() + 0.0025, r"$r^*_s$ range (23 folds)",
             ha="right", va="bottom", fontsize=5.5, color=MUTED)
    axB.legend(handles=[
        Line2D([], [], color=INK, ls="-", lw=1.1, label="S0 source-only"),
        Line2D([], [], color=INK, ls=(0, (3, 1.6)), lw=1.1, label="S1 zero-label"),
        Line2D([], [], color="none", marker="o", ms=4.3, mfc="none", mec=MUTED,
               mew=0.45, label="< 23 estimable sites"),
    ], loc="lower right", frameon=False, handlelength=1.5, borderpad=0.1,
        labelspacing=0.22, handletextpad=0.4, fontsize=5.5)

    # ---- panel c: per-site acceptance, ordered by site sample size -------
    a = acc[(acc.regime == "S0") & (acc.operating_point == "OP50")].copy()
    order = (a[["site", "n_s"]].drop_duplicates()
             .sort_values(["n_s", "site"], ascending=[False, True])["site"].tolist())
    x = np.arange(len(order))
    wide = a.pivot(index="site", columns="method", values="acceptance_rate").loc[order]
    axC.vlines(x, wide.min(axis=1).to_numpy(), wide.max(axis=1).to_numpy(),
               color=GRID, lw=1.6, zorder=1.5)
    for method in ORDER:
        c = METHOD_COLOR[method]
        g = a[a.method == method].set_index("site").loc[order]
        axC.plot(x, g.acceptance_rate.to_numpy(), ls="none",
                 marker=METHOD_MARKER[method], ms=3.2, mew=0.35,
                 mfc=c, mec="white", color=c, zorder=3)
    axC.axhline(0.50, color=INK, lw=0.7, ls=(0, (4, 2)), zorder=2)
    axC.set_xticks(x)
    axC.set_xticklabels(order, rotation=90, fontsize=4.9)
    axC.set_xlim(-0.7, len(order) - 0.3)
    axC.set_ylim(0.0, 1.12)
    axC.set_ylabel("acceptance rate at OP50")
    axC.set_xlabel("site, ordered by acquisitions")
    axC.set_title("S0 per-site acceptance (23 sites)", color=INK, pad=3,
                  fontsize=6.8)
    tidy(axC, grid_axis="y")
    # annotation placed in the empty band above the data (max observed 0.80)
    axC.text(len(order) - 0.4, 1.075, "source-intended coverage 0.50",
             ha="right", va="top", fontsize=5.6, color=INK)

    method_handles = [
        Line2D([], [], color=METHOD_COLOR[m], marker=METHOD_MARKER[m], ms=3.0,
               mew=0, lw=1.1, label=METHOD_LABEL[m]) for m in ORDER
    ]
    axC.legend(handles=method_handles, loc="lower left", frameon=False,
               handlelength=1.8, borderpad=0.1, labelspacing=0.25)

    for ax, tag in ((axA, "a"), (axB, "b"), (axC, "c")):
        ax.text(-0.02, 1.16, tag, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top", ha="right", color=INK)

    save(fig, "fig1_risk_coverage_and_site_acceptance")


# ==========================================================================
# Figure 2
# ==========================================================================
def figure2():
    burden = pd.read_csv(SRC / "fig2_uncertified_burden.csv")
    iv = pd.read_csv(SRC / "fig2_certification_intervals.csv")

    fig = plt.figure(figsize=(7.16, 2.70))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.80, 1.30, 1.00], wspace=0.95)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ---- panel A: uncertified sites vs N ---------------------------------
    b = (burden.groupby("n_labels")
         .agg(n_eligible=("n_eligible_sites", "max"),
              burden=("expected_uncertified_site_burden", "max"),
              certified=("n_sites_certified_at_least_once", "max"),
              exploratory=("exploratory", "max"))
         .reset_index().sort_values("n_labels").reset_index(drop=True))
    assert (burden.groupby("n_labels")["expected_uncertified_site_burden"].nunique() == 1).all()
    assert int(b.certified.max()) == 0

    for i, r in b.iterrows():
        kw = dict(hatch="///") if r.exploratory else {}
        axA.bar(i, r.burden, width=0.62, color=REGIME_COLOR["S2"],
                edgecolor=INK, lw=0.6, zorder=3, **kw)
        axA.text(i, r.burden + 0.5, f"{int(r.burden)}/{int(r.n_eligible)}",
                 ha="center", va="bottom", fontsize=5.9, color=INK)
    axA.set_xticks(range(len(b)))
    axA.set_xticklabels([("5\n(expl.)" if int(n) == 5 else str(int(n))) for n in b.n_labels])
    axA.set_xlabel("N target labels")
    axA.set_ylabel("uncertified sites / eligible sites")
    axA.set_ylim(0, 31)
    axA.set_title("S2  target-local certification", color=INK, pad=3)
    tidy(axA, grid_axis="y")
    axA.text(0.97, 0.905, "certified sites: 0 at every N;\nidentical for all 3\nconfidence methods",
             transform=axA.transAxes, ha="right", va="top", fontsize=5.2, color=INK)

    # ---- panel B: information-budget grouped pass / certification --------
    rows = [
        ("S0", "pooled_source", "source_reference_pass_rate", None,
         "pooled-source", "source reference"),
        ("S1", "target_unlabeled_weighted", "zero_label_empirical_pass_rate", None,
         "weighted", "empirical screening"),
        ("S2", "pooled_local_shrinkage", "empirical_pass_probability", 5, "shrinkage  N=5", ""),
        ("S2", "pooled_local_shrinkage", "empirical_pass_probability", 10, "shrinkage  N=10", ""),
        ("S2", "pooled_local_shrinkage", "empirical_pass_probability", 20, "shrinkage  N=20", ""),
        ("S2", "pooled_local_shrinkage", "empirical_pass_probability", 40, "shrinkage  N=40",
         "empirical screening"),
        ("S2", "target_local", "formal_certification_probability", 5, "target-local  N=5", ""),
        ("S2", "target_local", "formal_certification_probability", 10, "target-local  N=10", ""),
        ("S2", "target_local", "formal_certification_probability", 20, "target-local  N=20", ""),
        ("S2", "target_local", "formal_certification_probability", 40, "target-local  N=40",
         "formal certification"),
    ]
    n_rows = len(rows)
    ypos = np.arange(n_rows)[::-1]          # first row at the top

    bounds = {}
    for i, (reg, *_r) in enumerate(rows):
        bounds.setdefault(reg, [i, i])[1] = i
    for reg, (i0, i1) in bounds.items():
        y1, y0 = ypos[i0] + 0.5, ypos[i1] - 0.5
        axB.add_patch(Rectangle((-0.012, y0), 0.222, y1 - y0,
                                facecolor=REGIME_COLOR[reg], alpha=0.32,
                                edgecolor="white", lw=0.8, zorder=0.5))
        del reg

    labels = []
    for (reg, strat, quant, n, lab, tier), y in zip(rows, ypos):
        sel = iv[(iv.quantity == quant) & (iv.strategy == strat)]
        if n is not None:
            sel = sel[sel.n_labels == n]
        assert len(sel) == 3, (quant, strat, n, len(sel))
        v = sel["point_estimate"].astype(float)
        lo = sel["ci_low"].astype(float)
        hi = sel["ci_high"].astype(float)
        assert v.nunique() == 1 and lo.nunique() == 1 and hi.nunique() == 1
        axB.errorbar(float(v.iloc[0]), y,
                     xerr=[[float(v.iloc[0] - lo.iloc[0])], [float(hi.iloc[0] - v.iloc[0])]],
                     fmt="o", ms=3.8, mfc=REGIME_COLOR[reg], mec=INK, mew=0.5,
                     ecolor=INK, elinewidth=0.8, capsize=1.8, zorder=3)
        axB.text(0.203, y, f"{int(sel['n_sites'].iloc[0])}", ha="right", va="center",
                 fontsize=5.4, color=MUTED)
        labels.append(f"{reg}  {lab}")

    axB.set_yticks(ypos)
    axB.set_yticklabels(labels, fontsize=5.4)
    axB.set_ylim(ypos.min() - 0.6, ypos.max() + 1.45)
    axB.set_xlim(-0.012, 0.210)
    axB.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
    axB.set_xlabel("certification / pass probability")
    axB.set_title("information budget — not peer algorithms", color=INK, pad=3)
    tidy(axB, grid_axis="x")
    axB.text(0.203, ypos.max() + 0.95,
             "all 30 estimates 0.000, 95% CI [0.000, 0.000]",
             ha="right", va="center", fontsize=5.4, color=INK)
    axB.text(0.203, ypos.min() - 0.75, "eligible sites", ha="right", va="top",
             fontsize=5.2, color=MUTED)
    fig.text(0.5, -0.045,
             "tiers:  S0 source-reference  ·  S1 and S2-shrinkage empirical screening  ·  "
             "S2 target-local formal certification",
             ha="center", va="top", fontsize=5.2, color=MUTED, style="italic")

    # ---- panel C: quantities that are NOT_ESTIMABLE ----------------------
    ne = iv[iv.quantity.isin(["formal_attainable_coverage",
                              "formal_realized_violation_rate"])]
    assert (ne["status"] == "NOT_ESTIMABLE").all()
    assert (ne["n_estimable_sites"].astype(float) == 0).all()

    axC.add_patch(Rectangle((-0.55, 0.0), 4.1, 1.0, facecolor="none",
                            edgecolor="#c4c4c4", lw=0.5, hatch="\\\\", zorder=1))
    axC.axhline(0.10, color="#B3452B", lw=1.1, ls=(0, (4, 2)), zorder=4)
    axC.text(3.5, 0.125, r"nominal $\alpha_{\rm UCB} = 0.10$", ha="right",
             va="bottom", fontsize=5.8, color="#B3452B", zorder=5)
    axC.text(1.5, 0.615, "NOT ESTIMABLE", ha="center", va="center", fontsize=7.2,
             fontweight="bold", color=INK, zorder=5,
             bbox=dict(fc="white", ec="none", pad=1.6))
    axC.text(1.5, 0.395,
             "attainable coverage and realized\n"
             "violation rate have 0 estimable\n"
             "sites at every N and every method\n"
             "— no threshold was ever adopted,\n"
             "so the reference line below has\n"
             "nothing to be compared against",
             ha="center", va="center", fontsize=5.2, color=INK, zorder=5,
             linespacing=1.4, bbox=dict(fc="white", ec="none", pad=1.6))
    axC.text(1.5, 0.875,
             "minimum certifiable $N$: censored at\n"
             r"every site $\times$ method," "\n"
             r"$P(\mathrm{censored}) = 1.000$ [1.000, 1.000]",
             ha="center", va="center", fontsize=5.2, color=INK, zorder=5,
             linespacing=1.4, bbox=dict(fc="white", ec="none", pad=1.6))
    axC.set_xticks([0, 1, 2, 3])
    axC.set_xticklabels(["5", "10", "20", "40"])
    axC.set_xlim(-0.55, 3.55)
    axC.set_ylim(0.0, 1.0)
    axC.set_xlabel("N target labels")
    axC.set_ylabel("coverage / violation rate")
    axC.set_title("S2  realized behaviour", color=INK, pad=3)
    tidy(axC, grid_axis="y")

    for ax, tag in ((axA, "a"), (axB, "b"), (axC, "c")):
        ax.text(-0.02, 1.16, tag, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top", ha="right", color=INK)

    save(fig, "fig2_certification_label_cost")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    print("Stage G — rendering figures from Stage-G source data")
    figure1()
    figure2()


if __name__ == "__main__":
    main()
