"""Δα by gene category for one cohort, redrawn at print size: box plots and the effect-size heat map on the same rows.

    supp_categories_liver/boxes.pdf           Δα of the genes of every gene category (box plots), with the number of
    supp_categories_testis/boxes.pdf          genes, Cohen's d and a star for a median different from zero
    supp_categories_liver/effect_sizes.pdf    Cohen's d of Δα ("shift") and of |Δα| ("amplitude") for the same rows
    supp_categories_testis/effect_sizes.pdf   (heat map)

Cohorts: the four-species liver model and the testis Human+Macaque model. Inputs, per cohort: ``final_state.npz`` of the
stored run (α and prior); the gene names of the cohort's data object; the cohort's ortholog table (through
``notebooks/_paper_utils.py``) and the category vocabulary ``notebooks/_gene_categories.py``. The gene table, the
categories kept (five genes or more), their order (median Δα) and all statistics follow sections A.1 and A.3 of
``notebooks/_alpha_biology_builder.py``. Both panels of a cohort have the same page height and the same row positions,
so they are composed with ``X(...)``. The liver box plot has the Δα axis of every Δα panel; the testis box plot has a
narrower axis (``AXIS``), because the values of that model span a small range. A row with genes beyond the axis carries
a mark at its edge.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m figures.redraw.categories [cohort ...]
"""
from __future__ import annotations

import sys
from functools import lru_cache

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from figures.redraw import _common as C

FIGURE = {"liver_4sp": "supp_categories_liver", "testis_hmac": "supp_categories_testis"}   # cohort -> figure key (panel folder)
AXIS = {"testis_hmac": ((-0.10, 0.20), (-0.1, 0.0, 0.1, 0.2))}   # cohort -> limits and ticks of its Δα axis, where not the house axis
MIN_GENES = 5            # a category is shown when it has at least this many genes in the cohort
Q = 0.05                 # Benjamini-Hochberg level of the stars
STAR = "★"
PITCH = 11.5             # bp per row
ROW_PITCH = {"testis_hmac": 11.0}   # cohort -> bp per row where not PITCH: 41 rows of testis on the height of its page in the paper
TOP, BOTTOM = 13.0, 31.0     # bp above the first row (column headings) and below the last (axis label)
TOTAL, GAP = 486.0, 7.0      # bp: the two panels and the gap between them fill the text width
CELL = 33.0              # bp, width of one heat-map cell
ROW_SIZE = 5.8           # pt, row names, values in cells and at the boxes
D_OFFSET = 5.0           # bp between the box-plot axes and the Cohen's d of the row


def minus(text: str) -> str:
    return text.replace("-", "−")


def text_width(strings, size: float) -> float:
    """Width in bp of the longest of some strings in the figure font."""
    prop = FontProperties(family=mpl.rcParams["font.family"])
    return max(TextPath((0, 0), s, size=size, prop=prop).get_extents().width for s in strings)


@lru_cache(maxsize=None)
def category_table(cohort: str) -> tuple:
    """Δα and category of every gene, and the statistics of the categories shown (rows from the largest median Δα down)."""
    import anndata as ad
    from scipy.stats import mannwhitneyu, wilcoxon

    C.design()                                   # puts the notebooks folder on the path
    import _gene_categories as gc                # the category vocabulary
    import _paper_utils as pu                    # gene names of the cohort's ortholog table; Benjamini-Hochberg

    path = C.DATA / C.ADATA[cohort]
    C.need(path)
    final = np.load(C.run_dir(cohort) / "final_state.npz")
    a = ad.read_h5ad(path, backed="r")
    var_names = np.asarray(a.var_names)
    a.file.close()
    if len(var_names) != len(final["alpha"]):
        raise SystemExit(f"{path.name} has {len(var_names)} genes, the run has {len(final['alpha'])}")
    symbols = pu.annotate_genes_with_names(var_names, cohort)
    genes = pd.DataFrame({"symbol": symbols.to_numpy(), "category": gc.annotate_extended(var_names, symbols)["category"].to_numpy(),
                          "delta": final["alpha"] - final["prior"]})

    size = genes.groupby("category", observed=True).size()
    shown = [c for c in gc.CATEGORY_ORDER_EXTENDED if c != gc.OTHER and size.get(c, 0) >= MIN_GENES]
    order = genes[genes["category"].isin(shown)].groupby("category", observed=True)["delta"].median().sort_values().index.tolist()

    v = genes["delta"].to_numpy()
    absolute = np.abs(v)
    sd, sd_abs, mean_abs = float(np.std(v, ddof=1)), float(np.std(absolute, ddof=1)), float(np.mean(absolute))
    rows = []
    for c in order:
        m = (genes["category"] == c).to_numpy()
        shift = float(np.mean(v[m]) / sd)                                   # Cohen's d of Δα against all genes of the cohort
        amplitude = float((np.mean(absolute[m]) - mean_abs) / sd_abs) * (1.0 if shift >= 0 else -1.0)   # of |Δα|, with the sign of the shift
        try:
            p_shift = float(wilcoxon(v[m], alternative="two-sided", zero_method="wilcox")[1])           # median Δα different from 0
        except ValueError:
            p_shift = 1.0
        try:
            p_amplitude = float(mannwhitneyu(absolute[m], absolute[~m], alternative="greater")[1])      # |Δα| larger than in the other genes
        except ValueError:
            p_amplitude = 1.0
        rows.append(dict(category=c, n=int(m.sum()), median=float(np.median(v[m])), shift=shift, amplitude=amplitude,
                         p_shift=p_shift, p_amplitude=p_amplitude))
    t = pd.DataFrame(rows).set_index("category")
    t["q_box"] = pu.bh_fdr(t["p_shift"].to_numpy())                         # stars at the boxes: over the categories
    q = pu.bh_fdr(np.r_[t["p_shift"].to_numpy(), t["p_amplitude"].to_numpy()])   # stars in the heat map: both columns pooled
    t["q_shift"], t["q_amplitude"] = q[: len(t)], q[len(t):]
    return genes, t.iloc[::-1]


def layout(cohort: str) -> dict:
    """Widths (bp) of the two panels and of their parts, from the longest row name of the cohort."""
    _, t = category_table(cohort)
    names = C.category_names()
    label = [C.category(c, names) for c in t.index]
    with_n = [f"{s} (n = {n:,})" for s, n in zip(label, t["n"])]
    b_left = 2.0 + text_width(label, ROW_SIZE) + 3.0
    b_width = b_left + 2 * CELL + 38.0                                       # color bar and its labels: 38 bp
    a_width = TOTAL - GAP - b_width
    a_left = 13.5 + text_width(with_n, ROW_SIZE) + 6.0                       # axis label, row names, ticks
    a_right = D_OFFSET + text_width([f"d = −0.00  {STAR}"], ROW_SIZE) + 3.0  # Cohen's d and star at the right of every row
    pitch = ROW_PITCH.get(cohort, PITCH)
    height = TOP + pitch * len(t) + BOTTOM
    return dict(label=label, with_n=with_n, a=(a_width, a_left, a_right), b=(b_width, b_left), height=height, rows=len(t), pitch=pitch)


def panel(width: float, height: float, left: float, axes_width: float, rows: int, pitch: float = PITCH):
    mpl.rcParams.update({"savefig.bbox": None})                              # exact page size
    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes([left / width, BOTTOM / height, axes_width / width, pitch * rows / height])
    return fig, ax


def cohort_phrase(cohort: str) -> str:
    name = C.COHORT[cohort]
    return name[0].lower() + name[1:]


def boxes(cohort: str = "liver_4sp") -> list:
    genes, t = category_table(cohort)
    fd = C.design()
    g = layout(cohort)
    width, left, right = g["a"]
    n = g["rows"]
    fig, ax = panel(width, g["height"], left, width - left - right, n, g["pitch"])
    y = np.arange(n)[::-1]                                                   # first row of the table at the top
    data = [genes.loc[genes["category"] == c, "delta"].to_numpy() for c in t.index]
    ax.boxplot(data, orientation="horizontal", positions=y, widths=0.62, patch_artist=True, manage_ticks=False,
               medianprops=dict(color=C.DOWN, linewidth=1.0),
               boxprops=dict(facecolor="#dddddd", edgecolor=C.INK, linewidth=0.5),
               whiskerprops=dict(color=C.INK, linewidth=0.5), capprops=dict(color=C.INK, linewidth=0.5),
               flierprops=dict(marker="o", markersize=1.0, markerfacecolor=C.INK, markeredgecolor="none", alpha=0.4))
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--", zorder=0)
    (lo, hi), ticks = AXIS.get(cohort, (fd.DELTA_ALPHA_RANGE, fd.DELTA_ALPHA_TICKS))
    ax.set_xlim(lo, hi)
    ax.set_xticks(list(ticks))
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(y, g["with_n"], fontsize=ROW_SIZE)
    ax.tick_params(axis="y", pad=2.0)
    ax.set_xlabel(f"Δα ({cohort_phrase(cohort)})")
    ax.set_ylabel("Gene category", labelpad=4.5)
    off = []
    for yy, c, d in zip(y, t.index, data):
        r = t.loc[c]
        ax.annotate(minus(f"$d$ = {r['shift']:+.2f}") + (f"  {STAR}" if r["q_box"] < Q else ""), xy=(1.0, yy),
                    xycoords=ax.get_yaxis_transform(), xytext=(D_OFFSET, 0), textcoords="offset points", ha="left", va="center_baseline",
                    fontsize=ROW_SIZE, color="#444444", annotation_clip=False)
        below, above = int((d < lo).sum()), int((d > hi).sum())
        if below:                                                            # genes beyond the axis: a mark at its edge
            ax.plot([0.02], [yy], "<", transform=ax.get_yaxis_transform(), markersize=2.6, color=C.MUTED, markeredgewidth=0, clip_on=False, zorder=6)
        if above:
            ax.plot([0.98], [yy], ">", transform=ax.get_yaxis_transform(), markersize=2.6, color=C.MUTED, markeredgewidth=0, clip_on=False, zorder=6)
        if below or above:
            q1, q3 = np.percentile(d, [25, 75])
            off.append(f"{c}: {below} below {lo}, {above} above {hi} of {len(d)} genes; median {np.median(d):+.3f}, quartiles {q1:+.3f} to {q3:+.3f}")
    print(f"{cohort}: {len(genes):,} genes, {n} categories; stars at the boxes: {int((t['q_box'] < Q).sum())}; "
          f"Δα axis {lo} to {hi}, all genes of the cohort from {genes['delta'].min():+.3f} to {genes['delta'].max():+.3f}")
    print("rows with genes beyond the Δα axis (marked at its edge):\n   " + "\n   ".join(off or ["none"]))
    return [C.save_panel(fig, FIGURE.get(cohort, f"supp_categories_{cohort}"), "boxes", exact=True)]


def effect_sizes(cohort: str = "liver_4sp") -> list:
    _, t = category_table(cohort)
    C.design()
    g = layout(cohort)
    width, left = g["b"]
    n = g["rows"]
    fig, ax = panel(width, g["height"], left, 2 * CELL, n, g["pitch"])
    values = t[["shift", "amplitude"]].to_numpy()
    stars = (t[["q_shift", "q_amplitude"]].to_numpy() < Q)
    vmax = float(max(np.abs(values).max(), 1.0))
    im = ax.imshow(values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    for i in range(n):
        for j in range(2):
            ink = "white" if abs(values[i, j]) > 0.6 * vmax else C.INK
            ax.text(j, i, minus(f"{values[i, j]:+.2f}"), ha="center", va="center_baseline", fontsize=ROW_SIZE, color=ink)
            if stars[i, j]:
                ax.text(j + 0.38, i - 0.20, STAR, ha="center", va="center", fontsize=4.6, color=ink)
    ax.xaxis.tick_top()
    ax.set_xticks([0, 1], ["shift", "amplitude"], fontsize=C.SIZE["title"])
    ax.set_yticks(range(n), g["label"], fontsize=ROW_SIZE)
    ax.tick_params(length=0, pad=2.5)
    ax.tick_params(axis="x", pad=2.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Cohen's $d$ on Δα / signed |Δα|\n" + f"({cohort_phrase(cohort)})", fontsize=C.SIZE["tick"], labelpad=4.0, linespacing=1.15)
    h = g["pitch"] * n / g["height"]
    cax = fig.add_axes([(left + 2 * CELL + 10.0) / width, BOTTOM / g["height"] + 0.33 * h, 6.0 / width, 0.34 * h])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("effect size", fontsize=C.SIZE["tick"], labelpad=2.0)
    cb.ax.tick_params(labelsize=6.0, length=2.0, pad=1.5)
    cb.outline.set_linewidth(0.4)
    print(f"{cohort}: color range ±{vmax:.2f}; stars in the heat map: shift {int(stars[:, 0].sum())}, amplitude {int(stars[:, 1].sum())}; "
          f"cells beyond ±1.6: {int((np.abs(values) > 1.6).sum())} of {values.size}")
    return [C.save_panel(fig, FIGURE.get(cohort, f"supp_categories_{cohort}"), "effect_sizes", exact=True)]


PANELS = {"boxes": boxes, "effect_sizes": effect_sizes}

if __name__ == "__main__":
    for cohort in sys.argv[1:] or list(FIGURE):
        for name, draw in PANELS.items():
            for p in draw(cohort):
                print("wrote", p)
