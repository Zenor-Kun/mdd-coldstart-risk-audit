"""Replace panel (e) of the frozen combined results figure (v6) with the
post-hoc first-test decomposition, leaving panels (a)-(d) and the shared
legend bit-identical (PDF overlay, no redraw).

Inputs (read-only):
  paper/ISBI_template-master/figures/fig_combined_results_v6.pdf
  paper/ISBI_template-master/revision_support/source_data/posthoc_summary.csv
    (from scripts/build_posthoc_summary.py)
Output:
  paper/ISBI_template-master/figures/fig_combined_results_v8.pdf (+ .png)

The old panel (e) is redacted out of the v6 page before compositing, so its
text and vector art are removed from the file rather than painted over.

The counts are the frozen S2 decision cells (site x ordering x budget x score):
NT = not testable (no eligible candidate); TU cells are split post hoc into
"infeasible even with zero errors" (U_min > r*_s at the first-tested
candidate) and "feasible but failed on observed errors".
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper/ISBI_template-master/figures/fig_combined_results_v6.pdf"
DATA = ROOT / "paper/ISBI_template-master/revision_support/source_data/posthoc_summary.csv"
FIGDIR = ROOT / "paper/ISBI_template-master/figures"
PANEL = FIGDIR / "fig2_panel_e_posthoc.pdf"
OUT = FIGDIR / "fig_combined_results_v8.pdf"

# Region of panel (e) in v6 (inches); the shared legend below y=3.80 is kept.
CLIP_IN = (3.45, 1.95, 7.10, 3.80)

C_NT, C_ST, C_EL = "#dfe6ec", "#5f7a8c", "#d9822b"


def draw_panel(df: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444", "axes.linewidth": 0.6,
    })
    n = df["N"].tolist()
    nt = df["Not testable"].tolist()
    st = df["Zero-error infeasible"].tolist()
    el = df["Observed-error failure"].tolist()
    x = range(len(n))
    fig, ax = plt.subplots(figsize=(3.65, 1.85))
    ax.bar(x, nt, 0.6, color=C_NT, label="Not testable (NT)")
    ax.bar(x, st, 0.6, bottom=nt, color=C_ST,
           label="TU, infeasible even with zero errors")
    ax.bar(x, el, 0.6, bottom=[a + b for a, b in zip(nt, st)], color=C_EL,
           label="TU, feasible but failed on observed errors")
    for i in x:
        if nt[i]:
            ax.text(i, nt[i] / 2, f"{nt[i]:,} NT", ha="center", va="center",
                    fontsize=6.3, color="#333")
        if st[i]:
            ax.text(i, nt[i] + st[i] / 2, f"{st[i]:,}", ha="center",
                    va="center", fontsize=6.3, color="w")
        if el[i]:
            ax.text(i, nt[i] + st[i] + el[i] + 150, f"+{el[i]:,}", ha="center",
                    va="bottom", fontsize=6.3, color=C_EL)
    ax.set_xticks(list(x))
    ax.set_xticklabels(n)
    ax.set_xlabel("Target labels, N")
    ax.set_ylabel("Number of formal decision cells")
    ax.set_ylim(0, 10500)
    ax.set_yticks([0, 2000, 4000, 6000])
    ax.set_title("(e) First-test outcome by budget (post hoc)")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(length=2, width=0.6)
    ax.legend(fontsize=6, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.02), handlelength=1.2, labelspacing=0.3)
    fig.tight_layout(pad=0.15)
    fig.savefig(PANEL, metadata={"CreationDate": None})
    plt.close(fig)


def composite() -> None:
    src = pymupdf.open(SRC)  # edited in memory only; v6 on disk is untouched
    x0, y0, x1, y1 = (v * 72 for v in CLIP_IN)
    clip = pymupdf.Rect(x0, y0, x1, y1)
    # Remove only line art fully inside the clip, so the panel (d) frame survives.
    src[0].add_redact_annot(clip, fill=False, cross_out=False)
    src[0].apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                            graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)
    out = pymupdf.open()
    page = out.new_page(width=src[0].rect.width, height=src[0].rect.height)
    page.show_pdf_page(page.rect, src, 0)
    panel = pymupdf.open(PANEL)
    pw, ph = panel[0].rect.width, panel[0].rect.height
    s = min(clip.width / pw, clip.height / ph)
    target = pymupdf.Rect(clip.x1 - pw * s, clip.y0, clip.x1, clip.y0 + ph * s)
    page.show_pdf_page(target, panel, 0)
    out.save(OUT, garbage=3, deflate=True)
    out[0].get_pixmap(dpi=600).save(OUT.with_suffix(".png"))


if __name__ == "__main__":
    df = pd.read_csv(DATA).sort_values("N")
    draw_panel(df)
    composite()
    print("wrote", OUT)
