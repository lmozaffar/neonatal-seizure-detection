#!/usr/bin/env python3
"""Bar charts for the two AUROC results: the paper's SPH3/SOP7 alarm-based
headline (~0.8) and our raw per-epoch/per-fold AUROC (~0.6-0.7)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np

OUT_DIR = "figures"

BLUE = "#2a78d6"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "text.color": TEXT_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def bar_chart(df, value_col, title, subtitle, out_path, reference_line=None, reference_label=None):
    df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    models = df["model"].tolist()
    values = df[value_col].values

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)

    x = np.arange(len(models))
    bar_width = 0.55
    bars = ax.bar(x, values, width=bar_width, color=BLUE, zorder=3,
                   edgecolor="none")

    # 4px-equivalent rounded cap: emulate with a small rounding patch isn't native in mpl bar;
    # approximate with slightly rounded look via capstyle on a thin outline is unnecessary here —
    # keep clean flat-top bars, square at baseline (per spec: rounded data-end is a nice-to-have,
    # flat top remains correct and legible for a static PNG).

    for xi, v in zip(x, values):
        ax.text(xi, v + 0.025, f"{v:.3f}", ha="center", va="bottom",
                fontsize=10.5, color=TEXT_PRIMARY, fontweight="medium", zorder=4)

    if reference_line is not None:
        ax.axhline(reference_line, color=BASELINE, linestyle=(0, (4, 3)), linewidth=1.5, zorder=2)
        ax.text(len(models) - 0.5 + bar_width/2 + 0.05, reference_line, reference_label,
                ha="left", va="center", fontsize=9.5, color=TEXT_MUTED, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    ax.set_title(title, fontsize=13.5, fontweight="bold", color=TEXT_PRIMARY, pad=28, loc="left")
    fig.text(0.015, 0.915, subtitle, fontsize=9.5, color=TEXT_SECONDARY, ha="left")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    sph_sop = pd.read_csv("results_sph_sop/sph_sop_summary.csv")
    raw = pd.read_csv("full_reproduction_summary.csv")

    bar_chart(
        sph_sop, "headline_SPH3_SOP7",
        title="SPH3/SOP7 alarm-based AUROC",
        subtitle="The paper's own evaluation method (3-min SPH, 7-min SOP) — their reported headline is ~0.80 AUROC",
        out_path=f"{OUT_DIR}/sph_sop_auroc.png",
        reference_line=0.80, reference_label="paper reported ≈0.80",
    )

    bar_chart(
        raw, "mean_auroc",
        title="Raw per-epoch / per-fold AUROC",
        subtitle="Mean test AUROC across all 10 kfolds, classifying each 20-second epoch independently",
        out_path=f"{OUT_DIR}/raw_epoch_auroc.png",
    )
