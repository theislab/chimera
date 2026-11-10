"""Ten-method integration benchmark of the Supplement, redrawn at print size from the stored score table.

    supp_integration_ten_methods/purity.pdf         kNN species purity before scaling, 10 methods x 7 cohorts
    supp_integration_ten_methods/mixing.pdf         species mixing score
    supp_integration_ten_methods/conservation.pdf   cell-type conservation score
    supp_integration_ten_methods/composite.pdf      composite score
    supp_integration_raw_metrics/<metric>.pdf       the six metrics before scaling, one heat map each

Input: ``notebooks/cache/baselines_master_ext/master_table_ext.csv`` (one row per cohort and method: the six raw
metrics, the two scores and the composite). Nothing is rescored.

Rows are methods, grouped by their use of cell-type labels in training, and columns are cohorts, as in the main
integration figure (``analysis/tasks/T8_4_figures/fig3_integration.py``), which shows the three scores of the same table:
same names, same groups, same cell width, the dashed outline on the testis cells of Chimera (labels in training), bold
type on the best method of a cohort. Every heat map is dark where a method does better. The panels are saved at their
exact size with the axes at fixed places; a row of the figure is one panel with the method names and one or two
without, and the cohort names stand under the lower row only. They are composed with ``X(...)``.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m figures.redraw.integration_ten_methods [panel ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from figures.redraw import _common as C
from figures.redraw.integration_scores import METRICS, save_exact

TABLE = C.NB_CACHE / "baselines_master_ext" / "master_table_ext.csv"
COHORTS = ["panc_hmonkey", "panc_hmouse", "panc_hzeb", "panc_4sp", "test_hmac", "test_hmouse", "test_3sp"]  # table keys
GROUPS = [("no cell-type labels", ["PCA", "Harmony", "scVI-sp", "scVI-donor", "sysVI", "CAMEX"]),
          ("within-species clusters", ["SATURN-Leiden"]),
          ("cell-type labels", ["scANVI", "SATURN"]),
          ("no labels on liver; labels on testis (dashed)", ["Chimera"])]
METHODS = [m for _, ms in GROUPS for m in ms]
METHOD = {**C.METHOD, "CAMEX": "CAMEX", "SATURN": "SATURN", "SATURN-Leiden": "SATURN (Leiden clusters)"}
LABELS_ON_TESTIS = "Chimera"   # its testis cells carry a dashed outline
# the four heat maps of supp_integration_ten_methods: file name -> column, color-bar label, color map, range, best
SCORES = {
    "purity": ("raw__knn_species_purity", "kNN species purity ↓\n(before scaling; lower = more mixing)", "Blues_r", (0.5, 1.0), "min"),
    "mixing": ("mixing", "species mixing score\n(scaled within cohort, 1 = best)", "Blues", (0.0, 1.0), "max"),
    "conservation": ("biology", "cell-type conservation score\n(scaled within cohort, 1 = best)", "Blues", (0.0, 1.0), "max"),
    "composite": ("total_score", "composite score\n(scaled within cohort, 1 = best)", "Blues", (0.0, 1.0), "max"),
}
WITH_METHOD_NAMES = ("purity", "conservation", "knn_species_purity", "knn_celltype_purity")   # first panel of each row
WITH_COHORT_NAMES = ("conservation", "composite", "knn_celltype_purity", "sil_celltype", "label_transfer_acc")   # lower row

BP = 1 / 72.0
# One geometry for all heat maps (bp at print), with the map of the main figure. A panel with the method names and two
# without, with two gaps of 5.85 bp, fill the text width: (89.3 + 126 + 1.4) + 5.85 + (1.4 + 126 + 1.4) + 5.85 + 128.8 = 486.
LEFT, MAP_W, RIGHT = 89.3, 126.0, 1.4   # method names (RIGHT where there are none), the map, right margin
PITCH = 10.5                       # one line of a heat map: a method, or the heading of a group
TOP, NAMES, BAR = 3.0, 72.0, 33.0  # above the map; cohort names under it (4 bp where there are none); color bar and its label
VALUE_SIZE, METHOD_SIZE, COHORT_SIZE, GROUP_SIZE = 5.5, 6.0, 5.5, 5.5


def table() -> pd.DataFrame:
    C.need(TABLE)
    T = pd.read_csv(TABLE)
    if sorted(T["dataset"].unique()) != sorted(COHORTS) or sorted(T["model"].unique()) != sorted(METHODS) or len(T) != 70:
        raise SystemExit(f"{TABLE} does not hold the 7 cohorts x 10 methods")
    return T


def matrix(T: pd.DataFrame, column: str) -> pd.DataFrame:
    """Methods x cohorts, in the fixed order of the figure."""
    return T.pivot(index="model", columns="dataset", values=column).loc[METHODS, COHORTS]


def heat(M: pd.DataFrame, label: str, cmap: str, limits: tuple, best: str | None, method_names: bool, cohort_names: bool):
    """One heat map, methods (rows, under the heading of their group) by cohorts: the value in every cell, light on the
    dark end of the map; ``best``: "max" or "min" sets the best method of each cohort in bold."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    lines = [m for _, ms in GROUPS for m in [None, *ms]]                    # None: the line of a group heading
    A = np.full((len(lines), len(COHORTS)), np.nan)
    for i, m in enumerate(lines):
        if m is not None:
            A[i] = M.loc[m].to_numpy()
    vmin, vmax = limits
    left = LEFT if method_names else RIGHT
    under = NAMES if cohort_names else 4.0
    map_h = PITCH * len(lines)
    w, h = left + MAP_W + RIGHT, TOP + map_h + under + BAR
    fig = plt.figure(figsize=(w * BP, h * BP))
    ax = fig.add_axes([left / w, (under + BAR) / h, MAP_W / w, map_h / h])
    colors = plt.get_cmap(cmap).copy()
    colors.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(A), cmap=colors, vmin=vmin, vmax=vmax, aspect="auto")
    pick = {"max": np.nanargmax, "min": np.nanargmin}.get(best)
    top = {j: int(pick(A[:, j])) for j in range(A.shape[1])} if pick else {}
    for i, m in enumerate(lines):
        if m is None:
            continue
        for j, cohort in enumerate(COHORTS):
            v = A[i, j]
            rel = (v - vmin) / max(vmax - vmin, 1e-9)
            dark = rel < 0.4 if cmap.endswith("_r") else rel > 0.6
            ax.text(j, i, f"{v:.2f}".replace("-", "−"), ha="center", va="center", fontsize=VALUE_SIZE,
                    color="white" if dark else C.INK, fontweight="bold" if top.get(j) == i else "normal")
            if m == LABELS_ON_TESTIS and cohort.startswith("test"):
                ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False, edgecolor="white" if dark else C.INK,
                                       linewidth=0.6, linestyle=(0, (1.5, 1))))
    if method_names:
        k = 0
        for group, ms in GROUPS:
            ax.text(-0.45, k + 0.1, group, ha="left", va="center", fontsize=GROUP_SIZE, style="italic", color=C.MUTED)
            k += len(ms) + 1
    ax.axvline(3.5, color="white", linewidth=2.0)                            # liver | testis
    ax.set_xticks(range(len(COHORTS)))
    ax.set_xticklabels([C.COHORT[C.NOTEBOOK_KEY[c]] for c in COHORTS] if cohort_names else [], rotation=90, ha="center", va="top",
                       fontsize=COHORT_SIZE)
    rows = [i for i, m in enumerate(lines) if m is not None]
    ax.set_yticks(rows)
    ax.set_yticklabels([METHOD[lines[i]] for i in rows] if method_names else [], fontsize=METHOD_SIZE)
    ax.tick_params(length=0, pad=2.5)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([(left + 9.0) / w, (BAR - 6.0) / h, (MAP_W - 18.0) / w, 4.0 / h])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=5.5, labelpad=2.0, linespacing=1.1)
    cb.ax.tick_params(labelsize=5.5, length=1.8, width=0.4, pad=1.5)
    cb.outline.set_linewidth(0.4)
    return fig


def scores() -> list:
    """The raw species purity and the three scores of supp_integration_ten_methods."""
    C.design()
    T = table()
    out = []
    for name, (column, label, cmap, limits, best) in SCORES.items():
        M = matrix(T, column)
        pick = M.idxmin() if best == "min" else M.idxmax()
        print(f"{name}: best method per cohort: " + ", ".join(f"{C.COHORT[C.NOTEBOOK_KEY[c]]} {pick[c]} {M.at[pick[c], c]:.2f}" for c in COHORTS))
        fig = heat(M, label, cmap, limits, best, name in WITH_METHOD_NAMES, name in WITH_COHORT_NAMES)
        out.append(save_exact(fig, "supp_integration_ten_methods", name))
    return out


def raw_metrics() -> list:
    """The six metrics before scaling: three of species mixing (blue) and three of cell-type conservation (orange),
    each colored over its own range."""
    C.design()
    T = table()
    out = []
    for metric, (heading, cmap, name) in METRICS.items():
        M = matrix(T, f"raw__{metric}")
        limits = (float(M.to_numpy().min()), float(M.to_numpy().max()))
        fig = heat(M, f"{heading}\n(before scaling)", cmap, limits, None, metric in WITH_METHOD_NAMES, metric in WITH_COHORT_NAMES)
        out.append(save_exact(fig, "supp_integration_raw_metrics", name))
    return out


PANELS = {"scores": scores, "raw_metrics": raw_metrics}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(PANELS):
        for p in PANELS[name]():
            print("wrote", p)
