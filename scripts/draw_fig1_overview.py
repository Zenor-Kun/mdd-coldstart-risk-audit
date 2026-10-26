"""Vector study overview, sized for a single manuscript column.

Contract: distinguish information available for threshold transfer from the
target labels needed for local certification. This is a workflow schematic;
no synthetic observations or illustrative result curves are used.
"""
from pathlib import Path
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/isbi-figure-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/isbi-figure-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

sys.path.insert(0, "/root/.codex/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT = Path(__file__).resolve().parents[1] / "paper/ISBI_template-master/figures"
STEM = OUT / "fig1_redesigned"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 6.6, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.6, "savefig.dpi": 600,
})
INK = "#25313B"
MUTED = "#5B6670"
LINE = "#CCD2D7"
COLORS = ["#35678C", "#417F78", "#836299"]


def text(ax, x, y, value, size=6.6, color=INK, weight="normal", ha="left"):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", linespacing=1.35)


def line(ax, x1, y1, x2, y2, color=LINE, lw=0.6):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, clip_on=False)


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=6, color=color, lw=0.65, shrinkA=0, shrinkB=0))


def box(ax, x, y, w, h, fill="#F5F7F8", edge=LINE):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                           edgecolor=edge, linewidth=0.6))


def heading(ax, letter, title, height):
    text(ax, 0, height-2, letter, size=8, weight="bold")
    text(ax, 5, height-2, title, size=7.5, weight="bold")


width_in, height_in = 89/25.4, 120/25.4
fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
gs = fig.add_gridspec(3, 1, height_ratios=[30, 36, 42],
                      left=0.035, right=0.965, bottom=0.025, top=0.98,
                      hspace=0.13)
axes = [fig.add_subplot(gs[i]) for i in range(3)]
for ax, height in zip(axes, [30, 36, 42]):
    ax.set(xlim=(0, 82), ylim=(0, height))
    ax.axis("off")

# a: cohort and held-out deployment, not decorative medical imagery.
ax = axes[0]
heading(ax, "a", "Leave-one-site-out deployment", 30)
text(ax, 0, 22.5, "REST-meta-MDD", weight="bold")
text(ax, 82, 22.5, "23 sites · 2,232 rs-fMRI acquisitions", size=6.1, ha="right")
text(ax, 0, 17.8, "AAL116 FC · 6,670 Fisher-z connections", color=MUTED)
box(ax, 0, 3.5, 34, 9)
text(ax, 17, 9.3, "22 source sites", weight="bold", ha="center")
text(ax, 17, 5.8, "Train classifier", size=6.1, color=MUTED, ha="center")
arrow(ax, 36, 8, 46, 8)
box(ax, 48, 3.5, 34, 9)
text(ax, 65, 9.3, "1 held-out target site", weight="bold", ha="center")
text(ax, 65, 5.8, "Fixed predictions", size=6.1, color=MUTED, ha="center")

# b: a restrained comparison table; S2 tests S0 candidates, not S1 refits.
ax = axes[1]
heading(ax, "b", "Target-site information budgets", 36)
text(ax, 0, 28.6, "Same classifier and predictions across S0–S2", size=6.2, color=MUTED)
line(ax, 0, 25, 82, 25)
rows = [
    (20.8, "S0", "Source only", "Source-derived thresholds"),
    (12.4, "S1", "+ Unlabeled target data", "Reweighted source thresholds"),
    (4.0, "S2", "+ Labeled target data", "Local test of S0 thresholds · N = 5, 10, 20, 40"),
]
for (y, label, title, detail), color in zip(rows, COLORS):
    ax.add_patch(Rectangle((0, y-3.1), 0.85, 6.2, facecolor=color, edgecolor="none"))
    text(ax, 3, y, label, size=8, color=color, weight="bold")
    text(ax, 14, y+1.7, title, size=6.7, weight="bold")
    text(ax, 14, y-1.9, detail, size=5.7, color=MUTED)
    if label != "S2":
        line(ax, 14, y-4.2, 82, y-4.2, lw=0.4)

# c: prediction selection and two distinct evidence questions.
ax = axes[2]
heading(ax, "c", "Selective prediction and risk audit", 42)
text(ax, 41, 33.7, "Confidence score c(x) and threshold τ", ha="center")
box(ax, 5, 25.0, 31, 5.8, fill="#EFF3F6")
box(ax, 46, 25.0, 31, 5.8, fill="#F7F7F7")
text(ax, 20.5, 27.9, "c(x) ≥ τ: retain", ha="center", size=6.3)
text(ax, 61.5, 27.9, "c(x) < τ: abstain", ha="center", size=6.3, color=MUTED)
arrow(ax, 28, 32, 20.5, 31)
arrow(ax, 54, 32, 61.5, 31)
line(ax, 20.5, 24.5, 20.5, 22.7, color=MUTED)
line(ax, 20.5, 22.7, 62, 22.7, color=MUTED)
arrow(ax, 20.5, 22.7, 20.5, 21)
arrow(ax, 62, 22.7, 62, 21)
box(ax, 0, 2.0, 39, 18.4, fill="white")
box(ax, 43, 2.0, 39, 18.4, fill="white")
text(ax, 3, 17.5, "Empirical evaluation", size=6.6, weight="bold")
text(ax, 3, 10.7, "Risk–coverage\nPer-site acceptance\nAbstention associations", size=6.0)
text(ax, 46, 17.5, "S2 local certification", size=6.6, color=COLORS[2], weight="bold")
text(ax, 46, 13.2, "Target-calibration risk bound", size=5.8)
text(ax, 62.5, 8.6, r"$U(\tau)\leq r^{*}$", size=8, ha="center")
text(ax, 62.5, 4.4, "Eligible candidates only", size=5.6, color=MUTED, ha="center")

fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=str(STEM)+".alignment.json",
    overlay_svg=str(STEM)+".alignment.svg", tolerance_pt=1.5,
    gutter_tolerance_pt=1.5, strict=True)
fig.savefig(STEM.with_suffix(".pdf"), facecolor="white")
fig.savefig(STEM.with_suffix(".svg"), facecolor="white")
fig.savefig(STEM.with_suffix(".png"), dpi=600, facecolor="white")
plt.close(fig)
print(STEM.with_suffix(".pdf"))
