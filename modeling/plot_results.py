"""Grouped bar chart of cross-validated AUROC per model and feature set.

    .venv/bin/python -m modeling.plot_results

Reads results/cv_auroc.csv, prints the table behind the chart and saves
results/auroc_bar_chart.png. Bar height is the mean AUROC across folds; error bars
are the standard deviation across folds (pandas default, ddof=1). Models appear in
MODEL_ORDER, whichever are present.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling.common import RESULTS_DIR

MODEL_ORDER = ["LogReg", "RF", "KNN", "SVC", "ConvLSTM"]
SET_LABELS = {"set1": "Feature set 1: Temko (55 features)",
              "set2": "Feature set 2: Temko + rhythmicity (58)"}
SET_COLORS = {"set1": "#2a78d6", "set2": "#eb6834"}   # categorical slots 1 and 2, fixed order
TEXT, TEXT_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def summary_table(cv: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of AUROC across folds, one row per model and feature set."""
    models = [m for m in MODEL_ORDER if m in set(cv.model)] + sorted(set(cv.model) - set(MODEL_ORDER))
    table = (cv.groupby(["model", "feature_set"]).auroc.agg(mean="mean", std="std", n_folds="size")
             .reset_index())
    table["model"] = pd.Categorical(table.model, models, ordered=True)
    return table.sort_values(["model", "feature_set"]).reset_index(drop=True)


def plot(table: pd.DataFrame, out_path) -> None:
    models = list(table.model.cat.categories)
    sets = sorted(table.feature_set.unique())
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for i, s in enumerate(sets):
        rows = table[table.feature_set == s].set_index("model").reindex(models)
        offset = (i - (len(sets) - 1) / 2) * (width + 0.02)   # small surface gap between paired bars
        ax.bar(x + offset, rows["mean"], width, color=SET_COLORS.get(s), label=SET_LABELS.get(s, s), zorder=2)
        ax.errorbar(x + offset, rows["mean"], yerr=rows["std"], fmt="none", ecolor=TEXT_MUTED,
                    elinewidth=1.2, capsize=3, zorder=3)
        for xi, (m, sd) in zip(x + offset, zip(rows["mean"], rows["std"])):
            if pd.notna(m):
                ax.text(xi, m + sd + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_MUTED)

    ax.axhline(0.5, color=TEXT_MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate("chance\n(0.5)", xy=(1.0, 0.5), xycoords=("axes fraction", "data"), xytext=(4, 0),
                textcoords="offset points", ha="left", va="center", fontsize=8, color=TEXT_MUTED)

    ax.set_xticks(x, models)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylabel("AUROC (mean across 5 patient-level folds)", color=TEXT)
    ax.set_xlabel("Model", color=TEXT)
    ax.set_title("Window-level seizure detection: cross-validated AUROC", color=TEXT, loc="left", fontsize=12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=TEXT_MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_MUTED)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT, ncols=2, bbox_to_anchor=(0, 1.0))
    fig.text(0.01, 0.01, "Error bars: standard deviation across folds. Held-out eligible windows with label 0/1.",
             fontsize=7.5, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    cv = pd.read_csv(RESULTS_DIR / "cv_auroc.csv", float_precision="round_trip")
    table = summary_table(cv)
    print(table.assign(mean=table["mean"].round(3), std=table["std"].round(3)).to_string(index=False))
    out = RESULTS_DIR / "auroc_bar_chart.png"
    plot(table, out)
    print(f"\nsaved {out.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
