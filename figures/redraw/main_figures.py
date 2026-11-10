"""Panels of the main figures that were first assembled by hand, redrawn at print size from stored outputs.

    main_alpha_vs_prior/scatter.pdf                 α of every gene against its protein prior (liver Human+Monkey)
    main_alpha_vs_prior/shift.pdf                   histogram of Δα
    main_alpha_vs_prior/distribution.pdf            density of α over that of the prior

    main_delta_alpha_liver/category_boxes.pdf       Δα of the genes of the twelve largest gene categories (liver 4 species)
    main_delta_alpha_liver/category_effect_sizes.pdf   Cohen's d of Δα and of |Δα| for the same rows

    main_protein/attribution_shares.pdf             share of each input group under four attribution methods
    main_protein/atlas_alpha.pdf                    protein map (UMAP of the ESM-2 components) colored by α
    main_protein/atlas_delta_alpha.pdf              the map colored by Δα
    main_protein/atlas_correction.pdf               the map colored by the absolute correction of the gate network
    main_protein/atlas_categories.pdf               twelve gene categories on the map (at most 30 genes each)
    main_protein/atlas_categories_key.pdf           the key of the twelve categories
    main_protein/atlas_clusters.pdf                 median Δα of twelve protein clusters in four liver cohorts
    main_protein/archetype_number.pdf               silhouette score and agreement between seeds for 3 to 12 archetypes
    main_protein/archetype_delta_alpha.pdf          Δα of the genes of each archetype
    main_protein/archetype_protein_map.pdf          the archetypes on the protein map
    main_protein/archetype_neighbors.pdf            same-archetype share among protein neighbors, minus chance
    main_protein/archetype_prediction.pdf           accuracy of predicting the archetype from expression and from all inputs

Inputs. ``final_state.npz`` of the liver Human+Monkey run (α and prior) and of the four liver runs (α, prior,
correction); of the four-species liver run also ``cache/esm2_umap.npy`` (the stored map) and
``tables/gene_archetypes.csv`` (the archetype of every gene); the gene names and the 64 ESM-2 components stored with
the data objects; the category vocabulary ``notebooks/_gene_categories.py``. No model is loaded and nothing is trained.
Four sets of values need the trained gate network and are copied here from the stored outputs of the executed notebooks,
with the cell named at each table (``STORED_...``). The twelve clusters are four k-means fits with a fixed seed; their
medians are compared with the values printed on the first version of the panel, as are the category statistics, and the
script stops when one differs.

Panels in one row have the same page height and the same axes box, so they are composed with ``X(...)``.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m figures.redraw.main_figures [panel ...]
"""
from __future__ import annotations

import sys
from functools import lru_cache

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from figures.redraw import _common as C
from figures.redraw import atlas, categories

TEXT_WIDTH, GAP = 486.0, 7.0            # bp: text width of the paper, gap between panels in a row
THIRD = 1311 * 72 / 600                 # bp (157.32): three panels and two gaps fill the text width
NOTE = C.SIZE["text"]                   # pt, notes and values on a panel
ABOVE = 6.0                             # pt, a note or a key in the band above the axes of a narrow panel


def minus(text: str) -> str:
    return text.replace("-", "−")


def page(width: float, height: float, left: float, right: float, bottom: float, top: float):
    """A figure of width x height bp with one axes box at fixed margins (bp); point clouds are stored at 600 dpi."""
    fd = C.design()
    mpl.rcParams.update({"savefig.bbox": None, "savefig.dpi": 600})
    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes([left / width, bottom / height, 1 - (left + right) / width, 1 - (bottom + top) / height])
    return fd, fig, ax


def above(ax, text: str, ha: str = "right", **kw) -> None:
    """A note in the band above the axes: never over data."""
    ax.annotate(text, xy=(1.0 if ha == "right" else 0.0, 1.0), xycoords="axes fraction", xytext=(0, 3.5), textcoords="offset points",
                ha=ha, va="bottom", fontsize=kw.pop("fontsize", ABOVE), color=C.INK, annotation_clip=False, **kw)


# ---------------------------------------------------------------------------------------------------------------------
# Figure 2: the gate value against the protein prior (liver Human+Monkey)
# ---------------------------------------------------------------------------------------------------------------------
GATE = "main_alpha_vs_prior"
GATE_COHORT = "liver_hmonkey"
PRIOR_LABEL = "protein prior π"


def gate_values() -> tuple:
    f = np.load(C.run_dir(GATE_COHORT) / "final_state.npz")
    return f["alpha"].astype(np.float64), f["prior"].astype(np.float64)


def gate_frame():
    return page(THIRD, 138.0, left=31.0, right=9.5, bottom=25.0, top=13.0)   # right: half of the last tick label


def scatter() -> list:
    from scipy import stats

    alpha, prior = gate_values()
    fd, fig, ax = gate_frame()
    fd.density_scatter(ax, prior, alpha, s=0.7, alpha=0.18)       # dot area in proportion to the panel
    fd.add_identity_line(ax, lim=(0, 1))
    fd.style_alpha_axis(ax, "x")
    fd.style_alpha_axis(ax, "y")
    ax.set_xlabel(PRIOR_LABEL)
    ax.set_ylabel("α")
    rho = stats.spearmanr(prior, alpha).correlation
    above(ax, f"Spearman ρ = {rho:.3f}")
    print(f"scatter: {len(alpha):,} genes, Spearman {rho:.3f}")
    return [C.save_panel(fig, GATE, "scatter", exact=True)]


def shift() -> list:
    alpha, prior = gate_values()
    delta = alpha - prior
    fd, fig, ax = gate_frame()
    ax.hist(delta, bins=fd.DELTA_ALPHA_BINS, color="#bdbdbd", edgecolor="#888888", linewidth=0.3)
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--")
    fd.style_delta_alpha_axis(ax, "x")
    ax.set_xlabel("Δα")
    ax.set_ylabel("number of genes")
    up, down = (delta > 0).mean(), (delta < 0).mean()
    above(ax, f"Δα > 0: {100 * up:.0f}% of genes;  < 0: {100 * down:.0f}%")
    lo, hi = fd.DELTA_ALPHA_RANGE
    below, over = int((delta < lo).sum()), int((delta > hi).sum())
    if below:                                                     # genes beyond the axis: counted at its edge
        ax.text(0.02, 0.13, f"← {below} genes", transform=ax.transAxes, ha="left", va="bottom", fontsize=NOTE, color=C.MUTED)
    if over:
        ax.text(0.98, 0.13, f"{over} genes →", transform=ax.transAxes, ha="right", va="bottom", fontsize=NOTE, color=C.MUTED)
    print(f"shift: above 0 {up:.4f}, below 0 {down:.4f}; beyond the axis: {below} below {lo}, {over} above {hi}")
    return [C.save_panel(fig, GATE, "shift", exact=True)]


def distribution() -> list:
    alpha, prior = gate_values()
    fd, fig, ax = gate_frame()
    ax.hist(prior, bins=fd.ALPHA_BINS, color="#bbbbbb", edgecolor="none", density=True)
    ax.hist(alpha, bins=fd.ALPHA_BINS, color=C.INK, histtype="step", linewidth=0.9, density=True)
    fd.style_alpha_axis(ax, "x")
    ax.set_xlabel(f"α, {PRIOR_LABEL}")
    ax.set_ylabel("density")
    ax.legend(handles=[Patch(facecolor="#bbbbbb", edgecolor="none", label=PRIOR_LABEL),
                       Line2D([], [], color=C.INK, linewidth=0.9, label="α")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=ABOVE, handlelength=1.3,
              handletextpad=0.4, columnspacing=0.9, borderpad=0, borderaxespad=0.45)
    ax.text(0.03, 0.97, f"n = {len(alpha):,} genes\nmean α = {alpha.mean():.3f}\nSD = {alpha.std():.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=NOTE, color=C.INK, linespacing=1.25)
    print(f"distribution: n {len(alpha):,}, mean α {alpha.mean():.3f}, SD {alpha.std():.3f}")
    return [C.save_panel(fig, GATE, "distribution", exact=True)]


# ---------------------------------------------------------------------------------------------------------------------
# Figure 4 a, b: Δα of the twelve largest gene categories (liver 4 species)
# ---------------------------------------------------------------------------------------------------------------------
DEPLOYMENT = "main_delta_alpha_liver"
CATEGORY_COHORT = "liver_4sp"
N_CATEGORIES = 12
ROW_HEIGHT = 271.3            # bp: height of panel (c) of the figure, so that the three panels of the row are at scale 1
ROW_TOP, ROW_BOTTOM = 13.0, 31.0
BOX_RANGE, BOX_TICKS = (-0.25, 0.15), (-0.2, -0.1, 0.0, 0.1)   # holds every whisker; genes beyond it are marked at its edge
EFFECT_WIDTH, CELL = 93.0, 26.0
# Printed on the first version of the two panels (category, genes, shift, amplitude, star at shift, star at amplitude).
FIRST_VERSION_CATEGORIES = (
    ("ecm_other", 55, +0.51, -0.20, True, False), ("tf_homeobox", 58, +0.19, -0.52, True, False),
    ("kinases_signaling", 105, -0.23, +0.18, False, False), ("tf_forkhead_etc", 112, -0.15, +0.14, False, False),
    ("cytoskeleton_microtubule", 56, -0.49, +0.13, True, False), ("dna_replication_repair", 112, -0.53, +0.18, True, False),
    ("cell_cycle", 100, -0.40, +0.17, True, False), ("tf_zinc_finger", 114, -0.53, +0.06, True, False),
    ("nuclear_receptors", 121, -0.49, -0.16, True, True), ("splicing_factors", 70, -0.60, +0.14, True, False),
    ("oxphos_nuclear_encoded", 66, -0.78, -0.06, True, True), ("ribosomal_mt", 58, -1.38, -0.76, True, True),
)


@lru_cache(maxsize=1)
def largest_categories() -> tuple:
    """Δα of every gene, and the statistics of the twelve categories with the most genes (largest median Δα first).

    The effect sizes and tests are those of ``categories.category_table``; the stars follow the first version of the
    panels: Benjamini-Hochberg over the 24 tests of the twelve rows."""
    genes, t = categories.category_table(CATEGORY_COHORT)
    import _paper_utils as pu                                   # on the path after category_table

    by_size = t["n"].sort_values(ascending=False, kind="stable")
    if by_size.iloc[N_CATEGORIES - 1] == by_size.iloc[N_CATEGORIES]:
        raise SystemExit("the twelfth and the thirteenth category have the same number of genes")
    rows = t.loc[[c for c in t.index if c in set(by_size.index[:N_CATEGORIES])]].copy()
    q = pu.bh_fdr(np.r_[rows["p_shift"].to_numpy(), rows["p_amplitude"].to_numpy()])
    rows["star_shift"], rows["star_amplitude"] = q[:len(rows)] < categories.Q, q[len(rows):] < categories.Q
    got = tuple((c, int(r["n"]), round(float(r["shift"]), 2), round(float(r["amplitude"]), 2), bool(r["star_shift"]),
                 bool(r["star_amplitude"])) for c, r in rows.iterrows())
    if got != FIRST_VERSION_CATEGORIES:
        raise SystemExit("category statistics differ from the first version of the panels:\n" +
                         "\n".join(f"  {a}  |  {b}" for a, b in zip(got, FIRST_VERSION_CATEGORIES) if a != b))
    print(f"categories: the {len(rows)} rows, their gene counts, effect sizes and stars equal the first version")
    return genes, rows


def two_lines(name: str, count: str, limit: float, size: float) -> str:
    """A row name with its gene count, on two lines when one line is wider than ``limit`` bp.

    The count goes to the second line; a name that is itself too wide is broken at the space or hyphen that leaves
    the two lines most alike in width."""
    width = lambda t: categories.text_width([t], size)   # noqa: E731
    if width(f"{name} {count}") <= limit:
        return f"{name} {count}"
    if width(name) <= limit:
        return f"{name}\n{count}"
    text = f"{name} {count}"
    cuts = [i for i, ch in enumerate(name) if ch == " "] + [i + 1 for i, ch in enumerate(name) if ch == "-"]
    best = min(cuts, key=lambda i: max(width(text[:i].rstrip()), width(text[i:].lstrip())))
    return text[:best].rstrip() + "\n" + text[best:].lstrip()


def category_rows() -> tuple:
    """Row names of the two panels (name and number of genes) and the width of the box-plot panel."""
    _, rows = largest_categories()
    names = [two_lines(C.category(c), f"(n = {int(n):,})", 68.0, categories.ROW_SIZE) for c, n in zip(rows.index, rows["n"])]
    left = 2.0 + max(categories.text_width(s.split("\n"), categories.ROW_SIZE) for s in names) + 6.0
    return names, TEXT_WIDTH - 2 * GAP - 199.5 - EFFECT_WIDTH, left      # 199.5 bp: width of panel (c) as composed


def category_boxes() -> list:
    genes, rows = largest_categories()
    names, width, left = category_rows()
    n = len(rows)
    _, fig, ax = page(width, ROW_HEIGHT, left=left, right=4.0, bottom=ROW_BOTTOM, top=ROW_TOP)
    y = np.arange(n)[::-1]
    data = [genes.loc[genes["category"] == c, "delta"].to_numpy() for c in rows.index]
    ax.boxplot(data, orientation="horizontal", positions=y, widths=0.6, patch_artist=True, manage_ticks=False,
               medianprops=dict(color=C.DOWN, linewidth=1.0), boxprops=dict(facecolor="#dddddd", edgecolor=C.INK, linewidth=0.5),
               whiskerprops=dict(color=C.INK, linewidth=0.5), capprops=dict(color=C.INK, linewidth=0.5),
               flierprops=dict(marker="o", markersize=1.2, markerfacecolor=C.INK, markeredgecolor="none", alpha=0.45))
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--", zorder=0)
    lo, hi = BOX_RANGE
    ax.set_xlim(lo, hi)
    ax.set_xticks(list(BOX_TICKS), [minus(f"{v:+.1f}") if v else "0" for v in BOX_TICKS])
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(y, names, fontsize=categories.ROW_SIZE, linespacing=1.1)
    ax.tick_params(axis="y", pad=2.0)
    ax.set_xlabel(f"Δα ({categories.cohort_phrase(CATEGORY_COHORT)})")
    off = []
    for yy, c, d in zip(y, rows.index, data):
        q1, q3 = np.percentile(d, [25, 75])
        if d[d >= q1 - 1.5 * (q3 - q1)].min() < lo or d[d <= q3 + 1.5 * (q3 - q1)].max() > hi:
            raise SystemExit(f"{c}: a whisker leaves the axis")
        below = int((d < lo).sum())
        if (d > hi).any():
            raise SystemExit(f"{c}: genes above the axis")
        if below:                                                  # single genes beyond the axis: a mark and their number at its edge
            ax.plot([0.022], [yy], "<", transform=ax.get_yaxis_transform(), markersize=2.8, color=C.MUTED, markeredgewidth=0, clip_on=False,
                    zorder=6)
            ax.text(0.05, yy + 0.33, f"{below}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=NOTE - 0.3,
                    color=C.MUTED)
            off.append(f"{c}: {below} of {len(d)} genes below {lo}")
    print("category_boxes: genes beyond the Δα axis (marked at its edge with their number):\n   " + "\n   ".join(off or ["none"]))
    return [C.save_panel(fig, DEPLOYMENT, "category_boxes", exact=True)]


def category_effect_sizes() -> list:
    _, rows = largest_categories()
    n = len(rows)
    width, left = EFFECT_WIDTH, 3.0
    _, fig, ax = page(width, ROW_HEIGHT, left=left, right=width - left - 2 * CELL, bottom=ROW_BOTTOM, top=ROW_TOP)
    values = rows[["shift", "amplitude"]].to_numpy()
    stars = rows[["star_shift", "star_amplitude"]].to_numpy()
    vmax = float(max(np.abs(values).max(), 1.0))
    im = ax.imshow(values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    for i in range(n):
        for j in range(2):
            ink = "white" if abs(values[i, j]) > 0.6 * vmax else C.INK
            ax.text(j, i, minus(f"{values[i, j]:+.2f}"), ha="center", va="center_baseline", fontsize=categories.ROW_SIZE, color=ink)
            if stars[i, j]:
                ax.text(j + 0.36, i - 0.27, categories.STAR, ha="center", va="center", fontsize=4.6, color=ink)
    ax.xaxis.tick_top()
    ax.set_xticks([0, 1], ["shift", "amplitude"], fontsize=NOTE)
    ax.set_yticks([])
    ax.tick_params(length=0, pad=2.0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.5, (ROW_BOTTOM - 5.0) / ROW_HEIGHT, "Cohen's $d$ of Δα and\nof |Δα| (signed)", ha="center", va="top",
             fontsize=C.SIZE["tick"], linespacing=1.15)                # centered on the page: wider than the two columns
    h = 1 - (ROW_BOTTOM + ROW_TOP) / ROW_HEIGHT
    cax = fig.add_axes([(left + 2 * CELL + 6.0) / width, ROW_BOTTOM / ROW_HEIGHT + 0.30 * h, 5.5 / width, 0.40 * h])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("effect size", fontsize=C.SIZE["tick"], labelpad=1.5)
    cb.ax.tick_params(labelsize=NOTE, length=2.0, pad=1.5)
    cb.ax.set_yticks([-1, -0.5, 0, 0.5, 1], ["−1", "−0.5", "0", "0.5", "1"])
    cb.outline.set_linewidth(0.4)
    print(f"category_effect_sizes: color range ±{vmax:.2f}; stars: shift {int(stars[:, 0].sum())}, amplitude {int(stars[:, 1].sum())}")
    return [C.save_panel(fig, DEPLOYMENT, "category_effect_sizes", exact=True)]


# ---------------------------------------------------------------------------------------------------------------------
# Figure 8 a: share of each input group under four attribution methods (liver 4 species)
# ---------------------------------------------------------------------------------------------------------------------
PROTEIN = "main_protein"
# Percent of the total importance per input group, as printed by cell 29 of notebooks/09_gin_attention_overall.ipynb
# (the attribution needs the trained gate network). Columns in the order of INPUT_GROUP.
INPUT_GROUP = ("ESM-2 PCs", "esm2_intersp_var", "mean_identity", "expression", "identity_spread")
STORED_ATTRIBUTION = {
    "weight path": (88.2, 1.6, 1.4, 4.8, 4.0),
    "gradient × input": (76.7, 2.9, 0.6, 16.9, 3.0),
    "integrated gradients": (75.8, 2.9, 0.5, 18.0, 2.9),
    "ablation": (38.1, 8.6, 1.6, 47.1, 4.5),
}
GROUP_COLOR = {"ESM-2 PCs": "#cccccc", "esm2_intersp_var": "#56B4E9", "mean_identity": "#0072B2", "expression": "#D55E00",
               "identity_spread": "#888888"}                       # as in the first version of the panel
DARK_GROUP = ("expression", "mean_identity", "identity_spread")   # a percentage on these is written in white
PRINTED_FROM = 6.0                                                 # percent: narrower segments carry no number
SHARES_SIZE = (267.4, 163.9)   # bp: with panel (b) at 0.92 of its drawn size the two fill the text width at one height


def attribution_shares() -> list:
    width, height = SHARES_SIZE
    _, fig, ax = page(width, height, left=74.0, right=7.0, bottom=58.0, top=4.0)
    methods = list(STORED_ATTRIBUTION)
    y = np.arange(len(methods))
    left = np.zeros(len(methods))
    table = np.array([STORED_ATTRIBUTION[m] for m in methods])
    drawn = 100.0 * table / table.sum(axis=1, keepdims=True)       # the stored values have one decimal: rows sum to 99.9 to 100.1
    for j, g in enumerate(INPUT_GROUP):
        ax.barh(y, drawn[:, j], left=left, height=0.74, color=GROUP_COLOR[g], edgecolor="white", linewidth=0.5, label=C.GATE_GROUP[g])
        for yi in y:
            if table[yi, j] >= PRINTED_FROM:
                ax.text(left[yi] + drawn[yi, j] / 2, yi, f"{table[yi, j]:.0f}%", ha="center", va="center_baseline", fontsize=C.SIZE["tick"],
                        color="white" if g in DARK_GROUP else C.INK)
        left = left + drawn[:, j]
    ax.set_yticks(y, methods)
    ax.tick_params(axis="y", length=0, pad=3.0)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of total importance")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(6.0 / width, 3.0 / height),   # in the order of the segments
               ncol=2, frameon=False, fontsize=C.SIZE["legend"], handlelength=0.9, handleheight=0.8, handletextpad=0.5, columnspacing=1.4,
               labelspacing=0.35, borderpad=0, borderaxespad=0)
    print("attribution_shares: printed percentages", {m: [f"{v:.0f}" for v in STORED_ATTRIBUTION[m] if v >= PRINTED_FROM] for m in methods})
    return [C.save_panel(fig, PROTEIN, "attribution_shares", exact=True)]


# ---------------------------------------------------------------------------------------------------------------------
# Figure 8 d to h: the protein map of the four-species liver run
# ---------------------------------------------------------------------------------------------------------------------
ATLAS_COHORT = "liver_4sp"
MAP_HEIGHT = 125.0
MAP_BOX = dict(left=10.0, bottom=12.0, width=113.0, height=110.0)   # bp: the map has the same box on every panel
# The view of the map: the main cloud. Five small groups of genes lie far from it (left, below, right); with them in view
# the cloud would fill a quarter of the box. Their number is printed by ``protein_map``.
MAP_VIEW = ((-6.6, 9.1), (4.0, 18.9))
DOT = dict(s=1.1, alpha=0.85, linewidths=0, rasterized=True)
# The twelve largest categories in the order of the first version (by number of genes): order of the colors and of the draws.
ATLAS_CATEGORIES = ("nuclear_receptors", "tf_zinc_finger", "dna_replication_repair", "tf_forkhead_etc", "kinases_signaling",
                    "cell_cycle", "splicing_factors", "oxphos_nuclear_encoded", "tf_homeobox", "ribosomal_mt",
                    "cytoskeleton_microtubule", "ecm_other")
CATEGORY_COLOR = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#5b3e80", "#0F8000", "#7B0099", "#A52A2A",
                  "#F0E442", "#444444")
PER_CATEGORY, DRAW_SEED = 30, 0
KEY_SIZE = (TEXT_WIDTH - GAP - THIRD, 52.0)                      # bp: the key and the cluster panel stand beside the category map
CLUSTER_HEIGHT = MAP_HEIGHT - 6.0 - KEY_SIZE[1]                  # 6 bp: gap between the two in the figure
N_CLUSTERS, CLUSTER_SEED, CLUSTER_STARTS = 12, 0, 10
# Median Δα of the twelve clusters, sorted within the cohort, as printed on the first version of the panel.
FIRST_VERSION_CLUSTERS = {
    "liver_hmonkey": (-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.00, -0.00, +0.00, +0.00, +0.01, +0.01),
    "liver_hmouse": (-0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.02, -0.02, -0.02, -0.01, +0.00, +0.01),
    "liver_hzeb": (-0.00, +0.00, +0.00, +0.00, +0.00, +0.00, +0.01, +0.01, +0.01, +0.02, +0.02, +0.02),
    "liver_4sp": (-0.06, -0.05, -0.05, -0.03, -0.02, -0.01, -0.01, -0.01, -0.01, +0.01, +0.01, +0.02),
}


@lru_cache(maxsize=1)
def protein_map() -> dict:
    """The stored map of the genes (UMAP of the 64 ESM-2 components), α, the prior, Δα and the correction of the gate network."""
    d = C.run_dir(ATLAS_COHORT)
    C.need(d / "cache" / "esm2_umap.npy")
    xy = np.load(d / "cache" / "esm2_umap.npy").astype(np.float32)
    f = np.load(d / "final_state.npz")
    if len(xy) != len(f["alpha"]):
        raise SystemExit(f"the stored map has {len(xy)} genes, the run has {len(f['alpha'])}")
    (x0, x1), (y0, y1) = MAP_VIEW
    outside = (xy[:, 0] < x0) | (xy[:, 0] > x1) | (xy[:, 1] < y0) | (xy[:, 1] > y1)
    print(f"protein map: {len(xy):,} genes; {int(outside.sum())} genes ({100 * outside.mean():.1f}%) in small distant groups are outside the view")
    return dict(xy=xy, outside=outside, alpha=f["alpha"].astype(np.float32), prior=f["prior"].astype(np.float32),
                delta_alpha=(f["alpha"] - f["prior"]).astype(np.float32), correction=f["correction"].astype(np.float32))


def map_page(width: float = THIRD):
    """A page with the map box at its fixed place; no ticks: the two map axes have no unit."""
    b = MAP_BOX
    fd, fig, ax = page(width, MAP_HEIGHT, left=b["left"], right=width - b["left"] - b["width"], bottom=b["bottom"],
                       top=MAP_HEIGHT - b["bottom"] - b["height"])
    protein_map()
    ax.set_xlim(*MAP_VIEW[0])
    ax.set_ylim(*MAP_VIEW[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("ESM-2 UMAP 1", labelpad=2.5)
    ax.set_ylabel("ESM-2 UMAP 2", labelpad=2.0)
    return fig, ax


def colored_map(name: str, values: np.ndarray, cmap: str, vmin: float, vmax: float, label: str, ticks, extend: str = "neither") -> list:
    fig, ax = map_page()
    xy = protein_map()["xy"]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, vmin=vmin, vmax=vmax, **DOT)
    b = MAP_BOX
    cax = fig.add_axes([(b["left"] + b["width"] + 4.0) / THIRD, (b["bottom"] + 0.2 * b["height"]) / MAP_HEIGHT, 4.5 / THIRD,
                        0.6 * b["height"] / MAP_HEIGHT])
    cb = fig.colorbar(sc, cax=cax, ticks=ticks, extend=extend, extendfrac=0.04)
    cb.ax.set_yticklabels([minus(f"{t:g}") for t in ticks])
    cb.set_label(label, fontsize=C.SIZE["label"], labelpad=1.5)
    cb.ax.tick_params(labelsize=NOTE, length=2.0, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cb.solids.set_rasterized(True)
    return [C.save_panel(fig, PROTEIN, name, exact=True)]


def atlas_alpha() -> list:
    return colored_map("atlas_alpha", protein_map()["alpha"], "viridis", 0.0, 1.0, "α", [0, 0.25, 0.5, 0.75, 1])


def atlas_delta_alpha() -> list:
    delta = protein_map()["delta_alpha"]
    limit = float(np.percentile(np.abs(delta), 99))               # as in the first version: 99th percentile of |Δα|
    print(f"atlas_delta_alpha: color range ±{limit:.3f}; {int((np.abs(delta) > limit).sum())} genes beyond it take the end colors")
    return colored_map("atlas_delta_alpha", delta, "RdBu_r", -limit, limit, "Δα", [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], extend="both")


def atlas_correction() -> list:
    size = np.abs(protein_map()["correction"])
    limit = float(np.percentile(size, 99))                        # as in the first version: 99th percentile of |correction|
    print(f"atlas_correction: absolute correction, color range 0 to {limit:.2f}; {int((size > limit).sum())} genes above it take the "
          f"end color; {100 * float((protein_map()['correction'] < 0).mean()):.1f}% of the corrections are negative")
    return colored_map("atlas_correction", size, "magma", 0.0, limit, "|correction $c_g$|", [0, 0.5, 1.0, 1.5], extend="max")


@lru_cache(maxsize=1)
def category_draws() -> list:
    """For the twelve categories: key, number of genes, color, and the genes drawn (at most PER_CATEGORY, seeded)."""
    genes, rows = largest_categories()
    if set(rows.index) != set(ATLAS_CATEGORIES):
        raise SystemExit("the twelve largest categories are not those of the first version")
    category = genes["category"].to_numpy()
    rng = np.random.default_rng(DRAW_SEED)                        # one generator for the twelve draws, in this order
    out = []
    for c, color in zip(ATLAS_CATEGORIES, CATEGORY_COLOR):
        index = np.flatnonzero(category == c)
        n = len(index)
        if n > PER_CATEGORY:
            index = rng.choice(index, size=PER_CATEGORY, replace=False)
        out.append((c, n, color, index))
    if [n for _, n, _, _ in out] != sorted((n for _, n, _, _ in out), reverse=True):
        raise SystemExit("the categories are not in the order of their number of genes")
    return out


def atlas_categories() -> list:
    fig, ax = map_page()
    xy = protein_map()["xy"]
    ax.scatter(xy[:, 0], xy[:, 1], s=0.9, color="#cccccc", alpha=0.4, linewidths=0, rasterized=True, zorder=1)
    for _, _, color, index in category_draws():
        ax.scatter(xy[index, 0], xy[index, 1], s=6.5, color=color, edgecolor="white", linewidth=0.3, alpha=0.9, zorder=4, rasterized=True)
    return [C.save_panel(fig, PROTEIN, "atlas_categories", exact=True)]


def atlas_categories_key() -> list:
    C.design()
    width, height = KEY_SIZE
    fig = plt.figure(figsize=(width / 72, height / 72))
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=3.6, markerfacecolor=color, markeredgecolor="white",
                      markeredgewidth=0.3, label=f"{C.category(c)} (n = {n:,})") for c, n, color, _ in category_draws()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0 / width, 1 - 1.0 / height), ncol=2, frameon=False, fontsize=NOTE,
               handlelength=0.9, handletextpad=0.4, columnspacing=1.6, labelspacing=0.36, borderpad=0, borderaxespad=0)
    return [C.save_panel(fig, PROTEIN, "atlas_categories_key", exact=True)]


@lru_cache(maxsize=1)
def cluster_medians() -> dict:
    """Median Δα of the twelve k-means clusters of the 64 ESM-2 components, sorted, for the four liver cohorts.

    Section B.3 of ``notebooks/05_esm_alpha_geometry.ipynb``: one fit per cohort with a fixed seed. The values are
    compared with those printed on the first version of the panel (two decimals)."""
    from sklearn.cluster import KMeans

    out = {}
    for cohort in atlas.LIVER:
        d = atlas.protein_space(cohort)
        labels = KMeans(n_clusters=N_CLUSTERS, random_state=CLUSTER_SEED, n_init=CLUSTER_STARTS).fit(d["esm"]).labels_
        out[cohort] = np.sort([float(np.median(d["delta_alpha"][labels == c])) for c in range(N_CLUSTERS)])
        printed = tuple(float(f"{v:+.2f}") for v in out[cohort])
        if printed != tuple(float(f"{v:+.2f}") for v in FIRST_VERSION_CLUSTERS[cohort]):
            raise SystemExit(f"{cohort}: cluster medians differ from the first version of the panel:\n  {printed}\n  "
                             f"{FIRST_VERSION_CLUSTERS[cohort]}")
        print(f"atlas_clusters: {cohort}: the twelve medians equal the first version ({out[cohort][0]:+.4f} to {out[cohort][-1]:+.4f})",
              flush=True)
    return out


def atlas_clusters() -> list:
    C.design()
    medians = cluster_medians()
    table = np.vstack([medians[c] for c in atlas.LIVER])
    width = KEY_SIZE[0]
    labels = [C.COHORT_SHORT[c] for c in atlas.LIVER]
    left = 2.0 + categories.text_width(labels, C.SIZE["tick"]) + 4.0
    _, fig, ax = page(width, CLUSTER_HEIGHT, left=left, right=2.0, bottom=25.5, top=1.5)
    limit = float(np.abs(table).max())
    ax.imshow(table, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, minus(f"{table[i, j]:+.2f}"), ha="center", va="center_baseline", fontsize=NOTE,
                    color="white" if abs(table[i, j]) > 0.6 * limit else C.INK)
    ax.set_xticks(range(N_CLUSTERS), [str(i + 1) for i in range(N_CLUSTERS)], fontsize=NOTE)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0, pad=2.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="x", pad=1.0)
    ax.set_xlabel("median Δα of 12 protein clusters, sorted within the cohort\n($k$-means on the ESM-2 components, one fit per cohort)",
                  fontsize=NOTE + 0.4, labelpad=1.5, linespacing=1.1)
    print(f"atlas_clusters: color range ±{limit:.3f}")
    return [C.save_panel(fig, PROTEIN, "atlas_clusters", exact=True)]


# ---------------------------------------------------------------------------------------------------------------------
# Figure 8 i, k to n: the three gene archetypes of the gate network (liver 4 species)
# ---------------------------------------------------------------------------------------------------------------------
# Printed by cell 11 of notebooks/11_gene_archetypes.ipynb: k-means on the hidden layer of the gate network for 3 to 12
# archetypes, 30 seeds each (silhouette score; mean adjusted Rand index over the pairs of seeds).
STORED_NUMBER = {3: (0.127, 0.989), 4: (0.125, 0.967), 5: (0.123, 0.981), 6: (0.121, 0.988), 7: (0.094, 0.872), 8: (0.094, 0.916),
                 9: (0.093, 0.915), 10: (0.085, 0.862), 11: (0.089, 0.675), 12: (0.076, 0.665)}
CHOSEN = 3
# Printed by cell 41 of the same notebook: genes of each archetype, and the mean share of same-archetype genes among
# the 15 nearest protein neighbors minus the share of the archetype among all genes.
STORED_NEIGHBORS = {0: (3202, 0.212), 1: (3264, 0.070), 2: (2413, 0.248)}
# Printed by cell 49: accuracy of a multinomial logistic regression predicting the archetype, 5-fold cross-validation
# (mean, SD over the folds).
STORED_PREDICTION = (("chance\n(1 in 3)", 0.333, None), ("log mean\nexpression only", 0.569, 0.002), ("all 72\ngate inputs", 0.940, 0.006))
J_SCALE = 0.775                # the stored panel (j) is placed at this scale: its key prints at 5.0 pt
J_SIZE = (324.1, 243.2)        # bp: the stored panel without its heading line
ARCHETYPE_ROW = J_SIZE[1] * J_SCALE
NUMBER_WIDTH = 136.0
NEIGHBOR_WIDTH = 128.0
ARCHETYPE_MAP_WIDTH = 150.0


@lru_cache(maxsize=1)
def archetype_table() -> tuple:
    arch = atlas.archetypes()
    sizes = {a: int((arch == a).sum()) for a in range(3)}
    if sizes != {a: n for a, (n, _) in STORED_NEIGHBORS.items()}:
        raise SystemExit(f"archetype sizes {sizes} differ from the stored output")
    return arch, sizes


def archetype_number() -> list:
    C.design()
    mpl.rcParams.update({"savefig.bbox": None})
    width, height = NUMBER_WIDTH, ARCHETYPE_ROW
    left, right, bottom, top, gap = 33.0, 5.0, 24.0, 4.0, 11.0
    h = (height - bottom - top - gap) / 2
    fig = plt.figure(figsize=(width / 72, height / 72))
    upper = fig.add_axes([left / width, (bottom + h + gap) / height, 1 - (left + right) / width, h / height])
    lower = fig.add_axes([left / width, bottom / height, 1 - (left + right) / width, h / height])
    k = np.array(sorted(STORED_NUMBER))
    for ax, column, label, color, marker in ((upper, 0, "silhouette score", C.INK, "o"), (lower, 1, "ARI between seeds", "#777777", "s")):
        v = np.array([STORED_NUMBER[i][column] for i in k])
        ax.plot(k, v, color=color, marker=marker, markersize=2.6, linewidth=1.0)
        ax.plot([CHOSEN], [STORED_NUMBER[CHOSEN][column]], marker=marker, markersize=4.6, markerfacecolor="none", markeredgecolor=C.UP,
                markeredgewidth=0.8, linestyle="none", zorder=5)
        ax.set_xlim(2.5, 12.5)
        ax.set_xticks(k)
        ax.tick_params(axis="x", labelsize=NOTE)
        ax.set_ylabel(label, linespacing=1.05)
        ax.yaxis.set_label_coords(-(left - 7.5) / (width - left - right), 0.5)
    upper.set_xticklabels([])
    upper.set_ylim(0.07, 0.135)
    upper.yaxis.set_major_locator(MultipleLocator(0.02))
    lower.set_ylim(0.62, 1.02)
    lower.yaxis.set_major_locator(MultipleLocator(0.1))
    lower.set_xlabel("number of archetypes $k$")
    upper.text(0.98, 0.97, f"ring: $k$ = {CHOSEN}, chosen", transform=upper.transAxes, ha="right", va="top", fontsize=NOTE, color=C.UP)
    return [C.save_panel(fig, PROTEIN, "archetype_number", exact=True)]


def archetype_delta_alpha() -> list:
    arch, sizes = archetype_table()
    delta = protein_map()["delta_alpha"]
    width = TEXT_WIDTH - 2 * GAP - J_SIZE[0] * J_SCALE - NUMBER_WIDTH
    fd, fig, ax = page(width, ARCHETYPE_ROW, left=31.0, right=3.0, bottom=24.0, top=4.0)
    data = [delta[arch == a] for a in range(3)]
    bp = ax.boxplot(data, widths=0.56, patch_artist=True, showfliers=False, medianprops=dict(color=C.INK, linewidth=1.0),
                    whiskerprops=dict(color="#444444", linewidth=0.7), capprops=dict(color="#444444", linewidth=0.7),
                    boxprops=dict(edgecolor="#444444", linewidth=0.6))
    for patch, color in zip(bp["boxes"], atlas.ARCHETYPE_COLOR):
        patch.set_facecolor(mpl.colors.to_rgba(color, 0.55))
    ax.axhline(0, color="#888888", linewidth=0.5, linestyle="--", zorder=0)
    ax.set_xticks([1, 2, 3], [f"A{a}" for a in range(3)])
    ax.set_xlim(0.45, 3.55)
    ax.set_ylim(*fd.DELTA_ALPHA_RANGE)
    ax.set_yticks(list(fd.DELTA_ALPHA_TICKS), [minus(f"{v:+.1f}") if v else "0" for v in fd.DELTA_ALPHA_TICKS])
    ax.set_xlabel("archetype")
    ax.set_ylabel("Δα", labelpad=1.0)
    print("archetype_delta_alpha: medians", [round(float(np.median(d)), 4) for d in data], "means", [round(float(np.mean(d)), 4) for d in data])
    return [C.save_panel(fig, PROTEIN, "archetype_delta_alpha", exact=True)]


def archetype_protein_map() -> list:
    arch, sizes = archetype_table()
    fig, ax = map_page(ARCHETYPE_MAP_WIDTH)
    xy = protein_map()["xy"]
    for a in range(3):                                            # in this order, as in the first version
        m = arch == a
        ax.scatter(xy[m, 0], xy[m, 1], color=atlas.ARCHETYPE_COLOR[a], **DOT)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=3.4, markerfacecolor=atlas.ARCHETYPE_COLOR[a], markeredgecolor="none",
                      label=f"A{a}") for a in range(3)]                    # their sizes are in the key of the hidden-layer map
    b = MAP_BOX
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=((b["left"] + b["width"] + 4.0) / ARCHETYPE_MAP_WIDTH,
                                                                  (b["bottom"] + 0.5 * b["height"]) / MAP_HEIGHT),
               frameon=False, fontsize=C.SIZE["legend"], handlelength=0.9, handletextpad=0.3, labelspacing=0.5, borderpad=0, borderaxespad=0)
    return [C.save_panel(fig, PROTEIN, "archetype_protein_map", exact=True)]


def archetype_neighbors() -> list:
    _, sizes = archetype_table()
    _, fig, ax = page(NEIGHBOR_WIDTH, MAP_HEIGHT, left=19.0, right=5.0, bottom=33.0, top=4.0)
    y = np.arange(3)
    v = np.array([STORED_NEIGHBORS[a][1] for a in range(3)])
    ax.barh(y, v, height=0.72, color=list(atlas.ARCHETYPE_COLOR), edgecolor="none")
    for yi, value in zip(y, v):
        ax.text(value + 0.006, yi, minus(f"{value:+.3f}"), ha="left", va="center_baseline", fontsize=C.SIZE["tick"], color=C.INK)
    ax.set_yticks(y, [f"A{a}" for a in range(3)])
    ax.tick_params(axis="y", length=0, pad=2.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.34)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("same-archetype share of protein\nneighbors above chance, ΔP", linespacing=1.1)
    ax.xaxis.set_label_coords((NEIGHBOR_WIDTH / 2 - 19.0) / (NEIGHBOR_WIDTH - 24.0), -15.0 / (MAP_HEIGHT - 37.0))
    return [C.save_panel(fig, PROTEIN, "archetype_neighbors", exact=True)]


def archetype_prediction() -> list:
    C.design()
    width = TEXT_WIDTH - 2 * GAP - ARCHETYPE_MAP_WIDTH - NEIGHBOR_WIDTH
    names = [n for n, _, _ in STORED_PREDICTION]
    left = 2.0 + categories.text_width([s for n in names for s in n.split("\n")], C.SIZE["tick"]) + 5.0
    _, fig, ax = page(width, MAP_HEIGHT, left=left, right=5.0, bottom=33.0, top=4.0)
    y = np.arange(len(names))
    mean = np.array([m for _, m, _ in STORED_PREDICTION])
    sd = np.array([s or 0.0 for _, _, s in STORED_PREDICTION])
    ax.barh(y, mean, xerr=sd, height=0.72, color=["#dddddd", "#aaaaaa", "#666666"], edgecolor="none",
            error_kw=dict(ecolor=C.INK, capsize=1.5, capthick=0.5, linewidth=0.5))
    for yi, (_, m, s) in zip(y, STORED_PREDICTION):
        text = f"{m:.3f}" + (f" ± {s:.3f}" if s else "")
        inside = m > 0.7
        ax.text(m - 0.03 if inside else m + 0.03, yi, text, ha="right" if inside else "left", va="center_baseline", fontsize=NOTE,
                color="white" if inside else C.INK)
    ax.set_yticks(y, names, linespacing=1.1)
    ax.tick_params(axis="y", length=0, pad=2.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("accuracy of predicting the archetype\n(5-fold cross-validation)", linespacing=1.1)
    ax.xaxis.set_label_coords((width / 2 - left) / (width - left - 5.0), -15.0 / (MAP_HEIGHT - 37.0))
    return [C.save_panel(fig, PROTEIN, "archetype_prediction", exact=True)]


PANELS = {"scatter": scatter, "shift": shift, "distribution": distribution,
          "category_boxes": category_boxes, "category_effect_sizes": category_effect_sizes,
          "attribution_shares": attribution_shares,
          "atlas_alpha": atlas_alpha, "atlas_delta_alpha": atlas_delta_alpha, "atlas_correction": atlas_correction,
          "atlas_categories": atlas_categories, "atlas_categories_key": atlas_categories_key, "atlas_clusters": atlas_clusters,
          "archetype_number": archetype_number, "archetype_delta_alpha": archetype_delta_alpha,
          "archetype_protein_map": archetype_protein_map, "archetype_neighbors": archetype_neighbors,
          "archetype_prediction": archetype_prediction}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(PANELS):
        for p in PANELS[name]():
            print("wrote", p)
