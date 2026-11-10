"""Seven-method integration benchmark of the Supplement, redrawn at print size from the stored score table.

    supp_integration_seven_methods/composite.pdf    composite integration score, 7 cohorts x 7 methods, row maximum marked
    supp_integration_seven_methods/mean_score.pdf   mean score per method, with the median rank and the cohorts won
    supp_integration_metrics/<metric>.pdf           six raw metrics, one heat map each (7 cohorts x 7 methods)

Input: ``notebooks/cache/baselines_master/master_table.csv`` (one row per cohort and method: the six raw metrics and
the composite score). Nothing is rescored: the summary of ``mean_score`` (mean, median rank within cohort, number of
cohorts with the highest score) is derived from the composite column as in the benchmark notebook.

Rows are cohorts and columns are methods, in the order of the first-version panels. All heat maps share one cell size
and are saved at their exact size with the axes at fixed places, so that they line up when composed with
``Panel(..., ink=False)``.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m figures.redraw.integration_scores [panel ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from figures.redraw import _common as C

TABLE = C.NB_CACHE / "baselines_master" / "master_table.csv"
COHORTS = ["panc_hmonkey", "panc_hmouse", "panc_hzeb", "panc_4sp", "test_hmac", "test_hmouse", "test_3sp"]  # table keys
METHODS = ["PCA", "scVI-sp", "scVI-donor", "scANVI", "Harmony", "sysVI", "Chimera"]
HIGHLIGHT = "Chimera"   # column with an outline and a bold name
# metric column -> (heading with the better direction, color map with dark = better, file name)
METRICS = {
    "knn_species_purity": ("kNN species purity ↓", "Blues_r", "knn_species_purity"),
    "knn_mixing_entropy": ("kNN mixing entropy ↑", "Blues", "knn_mixing_entropy"),
    "sil_species": ("silhouette by species ↓", "Blues_r", "silhouette_species"),
    "knn_celltype_purity": ("kNN cell-type purity ↑", "YlOrBr", "knn_celltype_purity"),
    "sil_celltype": ("silhouette by cell type ↑", "YlOrBr", "silhouette_celltype"),
    "label_transfer_acc": ("label-transfer accuracy ↑", "YlOrBr", "label_transfer_accuracy"),
}
WITH_COHORT_NAMES = ("knn_species_purity", "sil_species", "sil_celltype")   # left panel of each row of two

BP = 1 / 72.0
# One geometry for all heat maps (bp at print): cohort names, seven columns of CELL_W, then RIGHT for the color bar of
# the composite or PAD beside a raw-metric map. Two raw-metric maps and one 7 bp gap fill the text width:
# (84 + 192.5 + 3.33) + 7 + (3.33 + 192.5 + 3.33) = 486.
LEFT, CELL_W, CELL_H, RIGHT, PAD = 84.0, 27.5, 13.75, 53.5, 10.0 / 3
MAP_W, MAP_H = 7 * CELL_W, 7 * CELL_H
VALUE_SIZE, METHOD_SIZE = 5.8, 6.0


def save_exact(fig, family: str, name: str):
    """Save at the drawn size: the notebooks' design crops every saved figure, which would move the fixed axes."""
    import matplotlib as mpl

    with mpl.rc_context({"savefig.bbox": "standard"}):
        return C.save_panel(fig, family, name, exact=True)


def table() -> pd.DataFrame:
    C.need(TABLE)
    T = pd.read_csv(TABLE)
    if sorted(T["dataset"].unique()) != sorted(COHORTS) or sorted(T["model"].unique()) != sorted(METHODS) or len(T) != 49:
        raise SystemExit(f"{TABLE} does not hold the 7 cohorts x 7 methods")
    return T


def matrix(T: pd.DataFrame, column: str) -> pd.DataFrame:
    """Cohorts x methods, in the fixed order of the figure."""
    return T.pivot(index="dataset", columns="model", values=column).loc[COHORTS, METHODS]


def summary(M: pd.DataFrame) -> pd.DataFrame:
    """Per method: mean score, median rank within cohort (1 = best) and the number of cohorts with the highest score."""
    ranks = M.rank(axis=1, ascending=False, method="min")
    won = M.idxmax(axis=1).value_counts().reindex(M.columns).fillna(0).astype(int)
    return pd.DataFrame({"mean": M.mean(axis=0), "median_rank": ranks.median(axis=0), "won": won})


def draw_cells(ax, M: pd.DataFrame, cmap: str, vmin: float, vmax: float, fmt: str, mark_best: bool = False):
    """Heat map with the value in every cell (light text on the dark end of the map) and an outline on one column."""
    from matplotlib.patches import Rectangle

    im = ax.imshow(M.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    best = np.nanargmax(M.to_numpy(), axis=1) if mark_best else np.full(len(M), -1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            rel = (v - vmin) / max(vmax - vmin, 1e-9)
            dark_cell = rel < 0.4 if cmap.endswith("_r") else rel > 0.6
            text = fmt.format(v).replace("-", "−") + ("★" if j == best[i] else "")
            ax.text(j, i, text, ha="center", va="center", fontsize=VALUE_SIZE, color="white" if dark_cell else C.INK,
                    fontweight="bold" if j == best[i] else "normal")
    jx = list(M.columns).index(HIGHLIGHT)
    ax.add_patch(Rectangle((jx - 0.5, -0.5), 1, M.shape[0], fill=False, edgecolor=C.INK, linewidth=1.1, zorder=10,
                           clip_on=False))
    ax.tick_params(length=0, pad=2.5)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def cohort_ticks(ax, show: bool = True) -> None:
    ax.set_yticks(range(len(COHORTS)))
    ax.set_yticklabels([C.COHORT[C.NOTEBOOK_KEY[k]] for k in COHORTS] if show else [], fontsize=C.SIZE["tick"])


def method_ticks(ax) -> None:
    """Method names under the columns; 'scVI (species batch)' on three short lines to fit one column."""
    names = [C.METHOD[m].replace(" (", "\n(").replace(" batch", "\nbatch") for m in METHODS]
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(names, fontsize=METHOD_SIZE, linespacing=1.05)
    ax.get_xticklabels()[METHODS.index(HIGHLIGHT)].set_fontweight("bold")


def composite() -> list:
    """(a) Composite score per cohort and method; star and bold type on the highest score of each cohort."""
    import matplotlib.pyplot as plt

    C.design()
    M = matrix(table(), "total_score")
    top, bottom = 4.0, 27.0
    w, h = LEFT + MAP_W + RIGHT, top + MAP_H + bottom
    fig = plt.figure(figsize=(w * BP, h * BP))
    ax = fig.add_axes([LEFT / w, bottom / h, MAP_W / w, MAP_H / h])
    im = draw_cells(ax, M, "Blues", 0.0, 1.0, "{:.2f}", mark_best=True)
    cohort_ticks(ax)
    method_ticks(ax)
    cax = fig.add_axes([(LEFT + MAP_W + 8) / w, bottom / h, 5.0 / w, MAP_H / h])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_label("composite integration score\n(scaled within cohort)", fontsize=6.2, linespacing=1.1)
    cb.ax.tick_params(labelsize=6.0, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    return [save_exact(fig, "supp_integration_seven_methods", "composite")]


def mean_score() -> list:
    """(b) Mean composite score per method under the columns of (a); median rank and cohorts won over each bar."""
    import matplotlib.pyplot as plt

    C.design()
    S = summary(matrix(table(), "total_score"))
    print(S.round(3).to_string())
    top, bars, bottom = 26.0, 60.0, 27.0
    w, h = LEFT + MAP_W + RIGHT, top + bars + bottom
    fig = plt.figure(figsize=(w * BP, h * BP))
    ax = fig.add_axes([LEFT / w, bottom / h, MAP_W / w, bars / h])
    rects = ax.bar(np.arange(len(METHODS)), S["mean"].to_numpy(), width=0.8, color=C.GREY_BAR, edgecolor=C.INK,
                   linewidth=0.5)
    rects[METHODS.index(HIGHLIGHT)].set_linewidth(1.1)
    over = ax.get_xaxis_transform()   # x in columns, y in axes heights: two lines of numbers above the axes
    lines = {"median rank": 1.0 + 15.5 / bars, "cohorts with best score": 1.0 + 6.5 / bars}
    for i, m in enumerate(METHODS):
        ax.text(i, S.at[m, "mean"] + 0.025, f"{S.at[m, 'mean']:.2f}", ha="center", va="bottom", fontsize=VALUE_SIZE)
        ax.text(i, lines["median rank"], f"{S.at[m, 'median_rank']:.1f}", ha="center", fontsize=VALUE_SIZE, transform=over)
        ax.text(i, lines["cohorts with best score"], f"{int(S.at[m, 'won'])} of {len(COHORTS)}", ha="center",
                fontsize=VALUE_SIZE, transform=over)
    for label, y in lines.items():
        ax.text(-0.5 - 6.0 / CELL_W, y, label, ha="right", fontsize=6.0, transform=over)
    ax.set_xlim(-0.5, len(METHODS) - 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("mean score")
    method_ticks(ax)
    ax.tick_params(axis="x", length=0, pad=2.5)
    return [save_exact(fig, "supp_integration_seven_methods", "mean_score")]


def raw_metric(T: pd.DataFrame, metric: str):
    """One raw metric, cohorts x methods, colored over its own range (dark = better), value in every cell."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    heading, cmap, name = METRICS[metric]
    M = matrix(T, f"raw__{metric}")
    rows = metric in WITH_COHORT_NAMES
    left = LEFT if rows else PAD
    top, labels, bar = 13.0, 25.0, 17.0
    w, h = left + MAP_W + PAD, top + MAP_H + labels + bar
    fig = plt.figure(figsize=(w * BP, h * BP))
    ax = fig.add_axes([left / w, (labels + bar) / h, MAP_W / w, MAP_H / h])
    im = draw_cells(ax, M, cmap, float(M.to_numpy().min()), float(M.to_numpy().max()), "{:.3f}")
    ax.set_title(heading, fontsize=C.SIZE["title"], pad=3.5)
    cohort_ticks(ax, show=rows)
    method_ticks(ax)
    cax = fig.add_axes([(left + CELL_W) / w, (bar - 5.0) / h, 5 * CELL_W / w, 4.0 / h])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=MaxNLocator(5))
    cb.ax.tick_params(labelsize=5.5, length=1.8, width=0.4, pad=1.5)
    cb.outline.set_linewidth(0.4)
    return save_exact(fig, "supp_integration_metrics", name)


def raw_metrics() -> list:
    """(a) to (f): the three species-mixing metrics (blue) and the three cell-type conservation metrics (orange)."""
    C.design()
    T = table()
    return [raw_metric(T, m) for m in METRICS]


PANELS = {"composite": composite, "mean_score": mean_score, "raw_metrics": raw_metrics}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(PANELS):
        for p in PANELS[name]():
            print("wrote", p)
