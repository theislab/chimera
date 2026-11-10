"""UMAP grids of the cell embeddings of five integration methods, redrawn at print size from stored coordinates.

    supp_integration_umap_testis/grid.pdf   three-species testis: 5 methods (rows) x 3 colorings (columns)
    supp_integration_umap_liver/grid.pdf    liver Human+Monkey, same grid

Inputs per cohort: ``notebooks/cache/chimera_umap/<cohort key>__<method>__umap.npy`` (the coordinates the notebook
stored for Chimera, PCA, scANVI, scVI with donor batch and sysVI) and the cell annotations (species, cell type, donor)
of the cohort's data object. No UMAP is computed: the script stops when a coordinate file is missing.

Every cell is drawn, in the order of the notebook panels: one color after the other, donors and cell types from the
largest group to the smallest, species as Human, Macaque or Monkey, Mouse; a later group lies over an earlier one. Colors
are those of the notebook panels. Each column has one legend with the cell counts. The point clouds are images at
600 dpi; everything else is vector.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m figures.redraw.integration_umap [testis|liver]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from figures.redraw import _common as C

# cohort -> key of the cached coordinate files, donor column of the cell annotations, figure key
COHORTS = {
    "testis_3sp": dict(key="test_3sp", donor="sample_batch", figure="supp_integration_umap_testis"),
    "liver_hmonkey": dict(key="panc_hmonkey", donor="batch", figure="supp_integration_umap_liver"),
}
METHODS = {"Chimera": "chimera", "PCA": "pca", "scANVI": "scanvi", "scVI-donor": "scvi_donor", "sysVI": "sysvi"}  # rows
COLUMNS = ("by donor", "by species", "by cell type")
# Okabe-Ito and four more hues, as in the notebook panels: the n-th largest donor or cell type takes the n-th color.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#999999",
           "#7B0099", "#0F8000", "#C45A00", "#5C3317"]
MAX_CELL_TYPES = 12   # further cell types would be pooled as "other" (none in these two cohorts)

# Page geometry in bp at print size.
WIDTH, NAME_W, RIGHT = 486.0, 14.0, 2.0          # page, strip for the method names, right margin
TILE_W, TILE_H, GAP_X, GAP_Y = 152.5, 98.0, 6.25, 3.0
HEAD_H, MARK_H, LEGEND_LINE = 12.0, 12.0, 7.7    # column headings; band of the axis mark; one legend line
# The notebook drew every cell with s = 2.5 on axes of about 215 x 244 bp; same share of the map here.
POINT = dict(s=2.5 * TILE_W * TILE_H / (215.0 * 244.0), alpha=0.85, linewidths=0, rasterized=True)
RASTER_DPI = 600


def annotations(cohort: str) -> pd.DataFrame:
    """Species, cell type and donor of every cell, in the cell order of the stored coordinates."""
    import anndata as ad

    path = C.DATA / C.ADATA[cohort]
    C.need(path)
    a = ad.read_h5ad(path, backed="r")
    cell_type = "cell_ontology_class" if "cell_ontology_class" in a.obs.columns else "cell_type"
    obs = a.obs[[COHORTS[cohort]["donor"], "species", cell_type]].copy()
    a.file.close()
    obs.columns = ["donor", "species", "cell_type"]
    return obs.astype(str)


def colorings(obs: pd.DataFrame, fd) -> list:
    """Per column: (label of every cell, groups in drawing order, color of each group)."""
    donors = list(obs["donor"].value_counts().index)
    species = sorted(obs["species"].unique(), key=fd.species_sort_key)
    counts = obs["cell_type"].value_counts()
    top = list(counts.index[:MAX_CELL_TYPES])
    cell_type = np.where(obs["cell_type"].isin(top), obs["cell_type"], "other")
    types = top + (["other"] if len(counts) > MAX_CELL_TYPES else [])
    type_color = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(top)} | {"other": "#bbbbbb"}
    return [(obs["donor"].to_numpy(), donors, {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(donors)}),
            (obs["species"].to_numpy(), species, {s: fd.species_color(s) for s in species}),
            (cell_type, types, type_color)]


def grid(cohort: str) -> list:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fd = C.design()
    info = COHORTS[cohort]
    files = {m: C.NB_CACHE / "chimera_umap" / f"{info['key']}__{slug}__umap.npy" for m, slug in METHODS.items()}
    C.need(*files.values())
    obs = annotations(cohort)
    columns = colorings(obs, fd)
    legend_lines = max(-(-len(groups) // (2 if len(groups) > 8 else 1)) for _, groups, _ in columns)
    grid_h = len(METHODS) * TILE_H + (len(METHODS) - 1) * GAP_Y
    height = HEAD_H + grid_h + MARK_H + legend_lines * LEGEND_LINE + 3.0
    fig = plt.figure(figsize=(WIDTH / 72, height / 72))
    page = fig.add_axes([0, 0, 1, 1])   # the whole page in bp, origin at the top left, for names and marks
    page.set_xlim(0, WIDTH)
    page.set_ylim(height, 0)
    page.axis("off")
    left = [NAME_W + j * (TILE_W + GAP_X) for j in range(3)]
    top = [HEAD_H + i * (TILE_H + GAP_Y) for i in range(len(METHODS))]

    for i, (method, path) in enumerate(files.items()):
        xy = np.load(path)
        if xy.shape != (len(obs), 2):
            raise SystemExit(f"{path.name}: {xy.shape[0]} points for {len(obs)} cells")
        for j, (labels, groups, color) in enumerate(columns):
            ax = fig.add_axes([left[j] / WIDTH, 1 - (top[i] + TILE_H) / height, TILE_W / WIDTH, TILE_H / height])
            for g in groups:
                ax.scatter(xy[labels == g, 0], xy[labels == g, 1], color=color[g], **POINT)
            ax.axis("off")
        page.text(NAME_W / 2 - 0.5, top[i] + TILE_H / 2, C.METHOD[method], rotation=90, ha="center", va="center",
                  fontsize=C.SIZE["title"], fontweight="bold" if method == "Chimera" else "normal")

    for j, (labels, groups, color) in enumerate(columns):
        page.text(left[j] + TILE_W / 2, HEAD_H - 3.5, COLUMNS[j], ha="center", va="baseline", fontsize=C.SIZE["title"])
        n = pd.Series(labels).value_counts()
        handles = [Line2D([], [], linestyle="none", marker="o", markersize=3.6, markeredgewidth=0, color=color[g],
                          label=f"{g} (n = {n[g]:,})") for g in groups]
        fig.legend(handles=handles, loc="upper center", ncol=2 if len(groups) > 8 else 1, frameon=False, fontsize=6.0,
                   bbox_to_anchor=((left[j] + TILE_W / 2) / WIDTH, 1 - (HEAD_H + grid_h + MARK_H) / height),
                   borderpad=0, borderaxespad=0, handletextpad=0.2, handlelength=1.0, labelspacing=0.28,
                   columnspacing=1.2)
        print(COLUMNS[j], {g: int(n[g]) for g in groups})

    # One axis mark for all maps, outside the lower left corner of the grid.
    x0, y0, arm = NAME_W - 2.0, HEAD_H + grid_h + 2.0, 22.0
    arrow = dict(arrowstyle="-|>", linewidth=0.6, color=C.INK, mutation_scale=4.5, shrinkA=0, shrinkB=0)
    page.annotate("", xy=(x0 + arm, y0), xytext=(x0, y0), arrowprops=arrow)
    page.annotate("", xy=(x0, y0 - arm), xytext=(x0, y0), arrowprops=arrow)
    page.text(x0 + 1.5, y0 + 2.0, "UMAP 1", ha="left", va="top", fontsize=5.5)
    page.text(x0 - 2.0, y0 - 1.5, "UMAP 2", rotation=90, ha="right", va="bottom", fontsize=5.5)

    with mpl.rc_context({"savefig.dpi": RASTER_DPI, "savefig.bbox": "standard"}):   # the page as drawn, not cropped
        return [C.save_panel(fig, info["figure"], "grid", exact=True)]


def testis() -> list:
    return grid("testis_3sp")


def liver() -> list:
    return grid("liver_hmonkey")


PANELS = {"testis": testis, "liver": liver}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(PANELS):
        for p in PANELS[name]():
            print("wrote", p)
