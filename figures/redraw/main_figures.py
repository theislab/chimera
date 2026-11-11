"""Panels of the main figures that were first assembled by hand, redrawn at print size from stored outputs.

    main_alpha_vs_prior/scatter.pdf                 α of every gene against its protein prior (liver Human+Monkey)
    main_alpha_vs_prior/shift.pdf                   histogram of Δα
    main_alpha_vs_prior/distribution.pdf            density of α over that of the prior
    main_alpha_vs_prior/training.pdf                Δα of 120 genes over the stored epochs of training

    main_delta_alpha_liver/category_boxes.pdf       Δα of the genes of the twelve largest gene categories (liver 4 species)
    main_delta_alpha_liver/category_effect_sizes.pdf   Cohen's d of Δα and of |Δα| for the same rows
                                                    (panel c of that figure: analysis/tasks/T8_4_figures/fig4_additions.py)

    main_protein/attribution_shares.pdf             share of each input group under four attribution methods
    main_protein/atlas_alpha.pdf                    protein map (UMAP of the ESM-2 components) colored by α
    main_protein/atlas_delta_alpha.pdf              the map colored by Δα
    main_protein/atlas_correction.pdf               the map colored by the absolute correction of the gate network
    main_protein/atlas_categories.pdf               twelve gene categories on the map (at most 30 genes each)
    main_protein/atlas_categories_key.pdf           the key of the twelve categories, in the order of ``category_boxes``
    main_protein/atlas_clusters.pdf                 median Δα of twelve protein clusters in four liver cohorts
    main_protein/archetype_number.pdf               silhouette score and agreement between seeds for 3 to 12 archetypes
    main_protein/archetype_delta_alpha.pdf          Δα of the genes of each archetype
    main_protein/archetype_protein_map.pdf          the archetypes on the protein map
    main_protein/archetype_neighbors.pdf            same-archetype share among protein neighbors, minus chance
    main_protein/archetype_prediction.pdf           accuracy of predicting the archetype from expression and from all inputs

Inputs. ``final_state.npz`` of the liver Human+Monkey run (α and prior) and its ``alpha_snapshots/`` (α and prior at
every fifth epoch); ``final_state.npz`` of the four liver runs (α, prior, correction); of the four-species liver run
also ``cache/esm2_umap.npy`` (the stored map) and ``tables/gene_archetypes.csv`` (the archetype of every gene); the
gene names and the 64 ESM-2 components stored with the data objects; the category vocabulary
``notebooks/_gene_categories.py``. No model is loaded and nothing is trained.
Four sets of values need the trained gate network and are copied here from the stored outputs of the executed notebooks,
with the cell named at each table (``STORED_...``). The twelve clusters are four k-means fits with a fixed seed; their
medians are compared with the values printed on the first version of the panel, as are the category statistics, and the
script stops when one differs.

Panels in one row have the same page height and the same axes box, so they are composed with ``X(...)``. The panels of
the main figures (``main_alpha_vs_prior``, ``main_delta_alpha_liver``, the atlas panels of ``main_protein``) are set in
the type and line weights of the main figures (``main_axes``, ``heading``; pages on the grid of the layout tool,
``on_grid``); the supplementary panels (``attribution_shares``, the archetype panels) keep those of ``_common``.

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
from matplotlib.ticker import MultipleLocator

from figures.redraw import _common as C
from figures.redraw import atlas, categories

TEXT_WIDTH, GAP = 486.0, 7.0            # bp: text width of the paper, gap between panels in a row
THIRD = 1311 * 72 / 600                 # bp (157.32): three panels and two gaps fill the text width
PIXEL = 72 / 300                        # bp (0.24): the layout tool measures a page on this grid
NOTE = C.SIZE["text"]                   # pt, notes and values on a panel of a supplementary figure
# Type and lines of the panels of the main figures: one scale for the ten figures of the main text. The supplementary
# panels of this script keep the sizes of ``_common.SIZE``.
HEAD, TITLE, TICK, SMALL = 6.5, 6.5, 6.0, 5.5   # pt: panel heading (bold); axis title; tick labels, row names, keys; notes inside a plot
AXIS_LINE, TICK_LINE, THIN = 0.6, 0.5, 0.5      # pt: axes; ticks; the thinnest line of a main figure
LIGHT = "#BBBBBB"                               # genes that are not the subject of a panel; zero lines


def minus(text: str) -> str:
    return text.replace("-", "−")


def signed(value: float, digits: int = 1) -> str:
    """A tick label on an axis of Δα: the sign in front of every value but zero ("+0.2", "−0.2", "0")."""
    return "0" if round(value, digits) == 0 else minus(f"{value:+.{digits}f}")


def on_grid(bp: float) -> float:
    """A page size on the grid of the layout tool, so that the panel is placed at the size it is drawn at."""
    return round(bp / PIXEL) * PIXEL


def plain(value: float) -> str:
    """A tick label without trailing zeros ("0", "0.5", "1")."""
    return f"{value:g}"


def page(width: float, height: float, left: float, right: float, bottom: float, top: float):
    """A figure of width x height bp with one axes box at fixed margins (bp); point clouds are stored at 600 dpi."""
    fd = C.design()
    mpl.rcParams.update({"savefig.bbox": None, "savefig.dpi": 600})
    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes([left / width, bottom / height, 1 - (left + right) / width, 1 - (bottom + top) / height])
    return fd, fig, ax


def main_axes(*axes) -> None:
    """The type and the line weights of a main figure on axes that ``page`` made: tick labels 6.0 pt, axis titles
    6.5 pt, axes 0.6 pt, ticks 0.5 pt and 2.5 bp long. Only panels of the main figures call it."""
    for ax in axes:
        ax.tick_params(which="both", labelsize=TICK, width=TICK_LINE, pad=1.5)
        ax.tick_params(which="major", length=2.5)
        ax.tick_params(which="minor", length=1.5)
        ax.xaxis.label.set_size(TITLE)
        ax.yaxis.label.set_size(TITLE)
        for s in ax.spines.values():
            s.set_linewidth(AXIS_LINE)


def heading(ax, text: str, rise: float = 3.5) -> None:
    """The heading of a main-figure panel: bold, in sentence case, its left edge on the left edge of the axes."""
    ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, rise), textcoords="offset points", ha="left", va="baseline",
                fontsize=HEAD, fontweight="bold", color=C.INK, annotation_clip=False)


# ---------------------------------------------------------------------------------------------------------------------
# main_alpha_vs_prior: the gate value against the protein prior (liver Human+Monkey)
# ---------------------------------------------------------------------------------------------------------------------
GATE = "main_alpha_vs_prior"
GATE_COHORT = "liver_hmonkey"
PRIOR_LABEL = "protein prior π"
TAIL = 0.05                       # a gene is in a tail of Δα beyond this distance from the prior
TAIL_SHARES = (16, 22)            # percent of the genes above +TAIL and below −TAIL, as the paper states them
SIGN_SHARES = (48, 52)            # percent of the genes above and below the prior
GATE_RHO = "0.77"                 # Spearman correlation of α with the prior, as the paper states it
# bp. The four panels have one axes box (87.0 x 84.9) and one page height. Their axes follow each other at one step across
# the text width (37.25 bp between two axes), the first y title 1.5 bp from the left edge of the figure and the last tick
# label 1.4 bp from its right edge. The pages differ in width, because the y axes need bands of different width and the
# last tick label of panel (b), "+0.4", reaches 7 bp beyond its axes: name -> (page width, on the grid of the layout
# tool; left margin of the axes). ``gate_frame`` checks that pages and gaps fill the text width at one step.
GATE_AXES, GATE_HEIGHT, GATE_BOTTOM, GATE_TOP = 87.0, on_grid(116.0), 20.5, 10.5
GATE_PAGES = {"scatter": (112.8, 22.48), "shift": (123.6, 26.93), "distribution": (110.4, 20.59), "training": (118.08, 27.44)}
TAIL_COLOR = {1: C.UP, 0: LIGHT, -1: C.DOWN}   # a gene by the tail it ends in: above +TAIL, between, below −TAIL


def gate_values() -> tuple:
    f = np.load(C.run_dir(GATE_COHORT) / "final_state.npz")
    return f["alpha"].astype(np.float64), f["prior"].astype(np.float64)


def tail_class(delta: np.ndarray) -> np.ndarray:
    """+1 for a gene more than ``TAIL`` above its prior, −1 for one more than ``TAIL`` below, 0 otherwise."""
    return np.where(delta > TAIL, 1, np.where(delta < -TAIL, -1, 0))


def gate_frame(name: str):
    """The page of one of the four panels (``GATE_PAGES``), with the axes box that the four share."""
    start, at = 0.0, []                                              # where each axes box begins in the figure
    for width, left in GATE_PAGES.values():
        at.append(start + left)
        start += width + GAP
    if abs(start - GAP - TEXT_WIDTH) > 0.25 or np.ptp(np.diff(at)) > 0.02:
        raise SystemExit(f"the four pages do not fill the text width at one step: {start - GAP:.2f} bp, steps {np.round(np.diff(at), 2)}")
    width, left = GATE_PAGES[name]
    return page(width, GATE_HEIGHT, left=left, right=width - left - GATE_AXES, bottom=GATE_BOTTOM, top=GATE_TOP)


def unit_axis(ax, which: str) -> None:
    """An axis from 0 to 1 (α, the prior): labels at 0, 0.5 and 1, short ticks at the quarters."""
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xlim if which == "x" else ax.set_ylim)(0.0, 1.0)
    axis.set_ticks([0.0, 0.5, 1.0], [plain(v) for v in (0.0, 0.5, 1.0)])
    axis.set_ticks([0.25, 0.75], minor=True)


def scatter() -> list:
    """Panel ``main_alpha_vs_prior/scatter`` (a): the gate value α of every gene against its protein prior, liver
    Human+Monkey run.

    What it draws. One dot per gene; the line is the diagonal α = prior. A gene is vermilion when it ends more than
    0.05 above its prior, blue when more than 0.05 below, light gray otherwise: the three classes of panel (b),
    with the colored dots on top. In the upper left corner, which holds no gene, the Spearman correlation of α with the prior over all genes.

    Reads ``final_state.npz`` of the run (arrays ``alpha`` and ``prior``). The correlation is computed from the two
    arrays and must print as the paper states it (``GATE_RHO``); the function stops otherwise.

        python -m figures.redraw.main_figures scatter     # in chimera_env, from the repository root
    """
    from scipy import stats

    alpha, prior = gate_values()
    kind = tail_class(alpha - prior)
    _, fig, ax = gate_frame("scatter")
    ax.plot([0, 1], [0, 1], color=C.MUTED, linewidth=AXIS_LINE, zorder=1)
    for k, shade in ((0, 0.5), (-1, 0.7), (1, 0.7)):                # the two tails on top
        m = kind == k
        ax.scatter(prior[m], alpha[m], s=1.2 ** 2, color="#8C8C8C" if k == 0 else TAIL_COLOR[k], alpha=shade, linewidths=0, rasterized=True,
                   zorder=2 if k == 0 else 3)
    unit_axis(ax, "x")
    unit_axis(ax, "y")
    ax.set_xlabel(PRIOR_LABEL, labelpad=2.0)
    ax.set_ylabel("α", labelpad=2.0)
    main_axes(ax)
    heading(ax, "Ordering kept")
    rho = stats.spearmanr(prior, alpha).correlation
    if f"{rho:.2f}" != GATE_RHO:
        raise SystemExit(f"scatter: Spearman correlation {rho:.4f}; the paper states {GATE_RHO}")
    corner = (prior < 0.42) & (alpha > 0.78)                         # the corner that holds the correlation holds no gene
    if corner.any():
        raise SystemExit(f"scatter: {int(corner.sum())} genes lie under the correlation printed in the upper left corner")
    ax.text(0.04, 0.97, f"Spearman\nρ = {rho:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=SMALL, color=C.INK, linespacing=1.2, zorder=6)
    print(f"scatter: {len(alpha):,} genes, Spearman {rho:.4f}; above +{TAIL}: {int((kind == 1).sum())}, below −{TAIL}: {int((kind == -1).sum())}")
    return [C.save_panel(fig, GATE, "scatter", exact=True)]


def shift() -> list:
    """Panel ``main_alpha_vs_prior/shift`` (b): the histogram of Δα = α − prior over the genes of the liver Human+Monkey run.

    What it draws. Bars 0.01 wide. The bars beyond +0.05 are vermilion, those beyond −0.05 blue, those between light
    gray; the dotted lines stand at ±0.05 (``TAIL``) and the solid line at Δα = 0, the protein prior. Beside each tail
    its share of the genes, in the color of the tail; at the top the shares of the genes below and above zero. Genes
    beyond the axis are counted at its edge ("← 115 genes").

    Reads ``final_state.npz`` of the run (arrays ``alpha`` and ``prior``). The two tail shares and the two shares of
    the signs are counted from these arrays and must equal the shares the paper states (``TAIL_SHARES``,
    ``SIGN_SHARES``); the function stops otherwise. It also stops if the colored bars do not hold exactly the genes of
    the two tails that lie on the axis.

        python -m figures.redraw.main_figures shift     # in chimera_env, from the repository root
    """
    alpha, prior = gate_values()
    delta = alpha - prior
    fd, fig, ax = gate_frame("shift")
    lo, hi = fd.DELTA_ALPHA_RANGE
    counts, edges = np.histogram(delta, bins=fd.DELTA_ALPHA_BINS)
    middle = (edges[:-1] + edges[1:]) / 2
    kind = tail_class(middle)
    if counts[kind == 1].sum() != ((delta > TAIL) & (delta <= hi)).sum() or counts[kind == -1].sum() != ((delta < -TAIL) & (delta >= lo)).sum():
        raise SystemExit("shift: the colored bars do not hold the genes of the two tails")
    ax.bar(middle, counts, width=np.diff(edges), color=[TAIL_COLOR[k] for k in kind], linewidth=0, zorder=2)
    tails = (100 * (delta > TAIL).mean(), 100 * (delta < -TAIL).mean())
    if (round(tails[0]), round(tails[1])) != TAIL_SHARES:
        raise SystemExit(f"shift: {tails[0]:.2f}% of the genes above +{TAIL} and {tails[1]:.2f}% below −{TAIL}; the paper states {TAIL_SHARES}")
    up, down = 100 * (delta > 0).mean(), 100 * (delta < 0).mean()
    if (round(up), round(down)) != SIGN_SHARES:
        raise SystemExit(f"shift: {up:.2f}% of the genes above the prior and {down:.2f}% below; expected {SIGN_SHARES}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 800)
    ax.axvline(0, color=C.MUTED, linewidth=AXIS_LINE, zorder=3)
    across = ax.get_xaxis_transform()
    for x, share, word, ha, k in ((TAIL, tails[0], "above", "left", 1), (-TAIL, tails[1], "below", "right", -1)):
        ax.plot([x, x], [0, 0.89], transform=across, color=C.MUTED, linewidth=AXIS_LINE, linestyle=(0, (1, 1.5)), zorder=3)
        # beside the line, in the color of the tail: the share in bold, then the side and the threshold
        for text, y, size, weight in ((f"{share:.0f}%", 0.665, TICK, "bold"), (word, 0.585, SMALL, "normal"), (signed(k * TAIL, 2), 0.505, SMALL, "normal")):
            ax.annotate(text, (x, y), xycoords=across, xytext=(2.5 * k, 0), textcoords="offset points", ha=ha, va="baseline", fontsize=size,
                        fontweight=weight, color=TAIL_COLOR[k])
    ax.text(0.02, 0.985, f"{down:.0f}% below 0", transform=ax.transAxes, ha="left", va="top", fontsize=SMALL, color=C.INK)
    ax.text(0.98, 0.985, f"{up:.0f}% above 0", transform=ax.transAxes, ha="right", va="top", fontsize=SMALL, color=C.INK)
    ax.set_xticks(list(fd.DELTA_ALPHA_TICKS), [signed(v) for v in fd.DELTA_ALPHA_TICKS])
    ax.set_yticks([0, 200, 400, 600, 800])
    ax.set_xlabel("Δα", labelpad=2.0)
    ax.set_ylabel("number of genes", labelpad=2.0)
    main_axes(ax)
    heading(ax, "Two tails")
    below, over = int((delta < lo).sum()), int((delta > hi).sum())
    if below:                                                     # genes beyond the axis: counted at its edge
        ax.text(0.025, 0.10, f"← {below}\ngenes", transform=ax.transAxes, ha="left", va="bottom", fontsize=SMALL, color=C.MUTED, linespacing=1.1)
    if over:
        ax.text(0.975, 0.10, f"{over} →\ngenes", transform=ax.transAxes, ha="right", va="bottom", fontsize=SMALL, color=C.MUTED, linespacing=1.1)
    print(f"shift: above 0 {up:.2f}%, below 0 {down:.2f}%; above +{TAIL} {tails[0]:.2f}%, below −{TAIL} {tails[1]:.2f}%; "
          f"beyond the axis: {below} below {lo}, {over} above {hi}; highest bar {counts.max()}")
    return [C.save_panel(fig, GATE, "shift", exact=True)]


def distribution() -> list:
    """Panel ``main_alpha_vs_prior/distribution`` (c): the distribution of the gate value α over that of the protein
    prior, liver Human+Monkey run.

    What it draws. Two histograms as densities, 50 bins from 0 to 1: the prior as a gray area, α as a black outline,
    each named beside it. In the upper left corner the number of genes, the mean of α and its standard deviation.

    Reads ``final_state.npz`` of the run (arrays ``alpha`` and ``prior``); the three numbers are computed from ``alpha``.

        python -m figures.redraw.main_figures distribution     # in chimera_env, from the repository root
    """
    alpha, prior = gate_values()
    fd, fig, ax = gate_frame("distribution")
    height, _, _ = ax.hist(prior, bins=fd.ALPHA_BINS, color=LIGHT, edgecolor="none", density=True)
    outline, _, _ = ax.hist(alpha, bins=fd.ALPHA_BINS, color=C.INK, histtype="step", linewidth=1.0, density=True)
    unit_axis(ax, "x")
    ax.set_ylim(0, 4.75)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel(f"α, {PRIOR_LABEL}", labelpad=2.0)
    ax.set_ylabel("density", labelpad=2.0)
    main_axes(ax)
    heading(ax, "Same shape, shifted")
    middle = (fd.ALPHA_BINS[:-1] + fd.ALPHA_BINS[1:]) / 2
    peak = int(np.argmax(height))                                   # the two names stand where each distribution lies outside the other
    ax.text(middle[peak], height[peak] + 0.1, "prior π", ha="center", va="bottom", fontsize=SMALL, color=C.MUTED)
    flank = int(np.argmax(np.where(middle < middle[peak] - 0.1, outline - height, -np.inf)))
    ax.text(middle[flank] - 0.035, outline[flank] + 0.05, "α", ha="right", va="bottom", fontsize=TICK, color=C.INK)
    ax.text(0.03, 0.985, f"n = {len(alpha):,} genes\nmean α = {alpha.mean():.3f}\nSD = {alpha.std():.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=SMALL, color=C.INK, linespacing=1.2)
    print(f"distribution: n {len(alpha):,}, mean α {alpha.mean():.3f}, SD {alpha.std():.3f}; highest bar of the prior {height.max():.2f}")
    return [C.save_panel(fig, GATE, "distribution", exact=True)]


def training() -> list:
    """Panel ``main_alpha_vs_prior/training`` (d): Δα of single genes over the epochs of training, liver Human+Monkey run.

    What it draws. One line per gene: Δα = α − prior at every stored epoch (every fifth epoch, 0 to 145) for 120 of
    the genes, the same 120 as in the supplementary panel ``supp_alpha_training/trajectories``. Every line starts at
    zero, the protein prior (solid line), because the gate starts at the prior. A line is vermilion when its gene ends
    more than 0.05 above the prior, blue when more than 0.05 below, gray otherwise: the classes of panels (a) and (b),
    from the final state. The dotted lines stand at ±0.05. The y axis is the x axis of panel (b); a line that leaves it
    is cut at its edge.

    Reads ``alpha_snapshots/alpha_epoch_*.npz`` of the run (arrays ``alpha``, ``prior``, ``epoch``) through
    ``figures.redraw.alpha.snapshots`` and ``final_state.npz`` (arrays ``alpha`` and ``prior``, for the class of a
    gene). Nothing is computed but the difference of the two stored arrays. The function stops unless the genes are
    those of the supplementary panel, the gate of every gene equals its prior at the first stored epoch, and the prior
    is the same array at every epoch.

        python -m figures.redraw.main_figures training     # in chimera_env, from the repository root
    """
    from figures.redraw import alpha as training_panels

    epochs, alpha, prior = training_panels.snapshots()
    final_alpha, final_prior = gate_values()
    pick = np.random.default_rng(training_panels.TRACE_SEED).choice(alpha.shape[1], size=training_panels.N_TRACES, replace=False)
    if not np.array_equal(alpha[:, pick], training_panels.trajectory_series()[1]):
        raise SystemExit("training: the genes drawn are not those of the supplementary panel")
    if np.abs(alpha[0] - prior[0]).max() > 1e-6 or np.abs(prior - final_prior).max() > 0:
        raise SystemExit("training: the gate does not start at the prior, or the prior changed during training")
    kind = tail_class(final_alpha - final_prior)[pick]
    delta = alpha[:, pick] - prior[:, pick]
    fd, fig, ax = gate_frame("training")
    lo, hi = fd.DELTA_ALPHA_RANGE
    ax.axhline(0, color=C.MUTED, linewidth=AXIS_LINE, zorder=1)
    for k in (0, -1, 1):                                             # the two tails on top
        ax.plot(epochs, delta[:, kind == k], color=TAIL_COLOR[k] if k else "#A8A8A8", linewidth=THIN + 0.1, alpha=0.9, zorder=2 if k == 0 else 3,
                solid_joinstyle="round")
    for x in (TAIL, -TAIL):
        ax.axhline(x, color=C.MUTED, linewidth=AXIS_LINE, linestyle=(0, (1, 1.5)), zorder=4)
    ax.set_ylim(lo, hi)
    ax.set_yticks(list(fd.DELTA_ALPHA_TICKS), [signed(v) for v in fd.DELTA_ALPHA_TICKS])
    ax.set_xlim(-6, 156)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlabel("epoch", labelpad=2.0)
    ax.set_ylabel("Δα", labelpad=0.5)
    main_axes(ax)
    heading(ax, "Moved during training")
    ax.text(0.04, 0.97, f"{len(pick)} of {alpha.shape[1]:,} genes", transform=ax.transAxes, ha="left", va="top", fontsize=SMALL, color=C.INK)
    left = int((delta.min(axis=0) < lo).sum())
    print(f"training: {len(pick)} of {alpha.shape[1]:,} genes at {len(epochs)} stored epochs ({epochs[0]} to {epochs[-1]}); above {int((kind == 1).sum())}, "
          f"between {int((kind == 0).sum())}, below {int((kind == -1).sum())}; {left} lines leave the axis at {lo} (lowest {delta.min():.3f}); "
          f"class at the last stored epoch differs from the final class for {int((tail_class(delta[-1]) != kind).sum())} genes")
    return [C.save_panel(fig, GATE, "training", exact=True)]


# ---------------------------------------------------------------------------------------------------------------------
# main_delta_alpha_liver, panels a, b: Δα of the twelve largest gene categories (liver 4 species)
# ---------------------------------------------------------------------------------------------------------------------
DEPLOYMENT = "main_delta_alpha_liver"
CATEGORY_COHORT = "liver_4sp"
N_CATEGORIES = 12
# bp: the three panels of the figure are pages of one height with one band for the axes (twelve rows of 13.3 bp), so that
# they are placed at the size they are drawn at and their x axes lie on one line. Panel (c), the plane of prior and Δα, is
# drawn by analysis/tasks/T8_4_figures/fig4_additions.py with the same three numbers and takes the rest of the text width.
ROW_HEIGHT, ROW_BOTTOM, ROW_TOP = on_grid(190.5), 19.56, 11.0    # page; under the axes (tick labels, axis title); above them (column headings)
BOX_WIDTH, EFFECT_WIDTH = on_grid(196.0), on_grid(60.0)          # pages of panels (a) and (b)
NAME_COLUMN, COUNT_COLUMN, CELL = 93.0, 15.0, 28.0               # bp: names of the rows; their numbers of genes; a cell of panel (b)
BOX_RANGE, BOX_TICKS = (-0.25, 0.15), (-0.2, -0.1, 0.0, 0.1)     # holds every whisker; genes beyond it are marked at its edge
FLOOR = 5.0                                                      # pt: the smallest type (numbers at the edge of an axis)
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


def tint(color: str, share: float = 0.25) -> tuple:
    """A color at ``share`` on white, as an opaque color."""
    return tuple(share * c + (1 - share) for c in mpl.colors.to_rgb(color))


def category_boxes() -> list:
    """Panel ``main_delta_alpha_liver/category_boxes`` (a): Δα of the genes of the twelve gene categories with the most
    genes, four-species liver, as box plots.

    What it draws. One row per category, from the largest median Δα down; beside the name, in a column headed "n", the
    number of genes. Box: the quartiles of Δα over the genes of the category; black line: the median; whiskers: the
    last gene within 1.5 interquartile ranges of the box; dots: the genes beyond the whiskers. The face of a box says
    what panel (b) tests: vermilion when the category is shifted above the prior (q < 0.05), blue when below, gray
    when the shift is not significant. The gray line is Δα = 0, the protein prior. The axis holds every whisker (the
    function stops otherwise); where single genes lie beyond its left end, the row carries an arrowhead and their
    number, as the note under the names says.

    Reads the gene table and the rows of ``largest_categories``, which come from ``categories.category_table``:
    ``final_state.npz`` of the stored four-species liver run (arrays ``alpha`` and ``prior``; Δα is their difference),
    the gene names of the data object of the cohort, its ortholog table (gene symbols) and the category vocabulary
    ``notebooks/_gene_categories.py``. Columns used here: ``category`` and ``delta`` of the gene table; ``n``,
    ``shift`` and ``star_shift`` of the rows. Nothing is computed here but the quartiles that the box plot draws.

    The page has the height and the rows of ``category_effect_sizes`` and of panel (c) of the figure; the two panels
    of this script are drawn together and composed with ``X(...)``:

        python -m figures.redraw.main_figures category_boxes category_effect_sizes     # in chimera_env, from the repository root
    """
    genes, rows = largest_categories()
    n = len(rows)
    names = [C.category(c) for c in rows.index]
    if 1.5 + categories.text_width(names, TICK) + 3.5 > NAME_COLUMN:
        raise SystemExit(f"category_boxes: a row name is wider than its column ({categories.text_width(names, TICK):.1f} bp)")
    _, fig, ax = page(BOX_WIDTH, ROW_HEIGHT, left=NAME_COLUMN + COUNT_COLUMN, right=3.5, bottom=ROW_BOTTOM, top=ROW_TOP)
    y = np.arange(n)[::-1]
    data = [genes.loc[genes["category"] == c, "delta"].to_numpy() for c in rows.index]
    thin = dict(color=C.INK, linewidth=THIN)
    drawn = ax.boxplot(data, orientation="horizontal", positions=y, widths=0.62, patch_artist=True, manage_ticks=False,
                       medianprops=dict(color=C.INK, linewidth=1.0, solid_capstyle="butt"), boxprops=dict(edgecolor=C.INK, linewidth=THIN),
                       whiskerprops=thin, capprops=thin,
                       flierprops=dict(marker="o", markersize=1.1, markerfacecolor=C.INK, markeredgecolor="none", alpha=0.45))
    side = [int(np.sign(r.shift)) if r.star_shift else 0 for r in rows.itertuples()]    # the sign of a shift that panel (b) marks
    for box, k in zip(drawn["boxes"], side):
        box.set_facecolor({1: tint(C.UP), -1: tint(C.DOWN), 0: "#E6E6E6"}[k])
    ax.axvline(0, color=LIGHT, linewidth=AXIS_LINE, zorder=0)
    lo, hi = BOX_RANGE
    ax.set_xlim(lo, hi)
    ax.set_xticks(list(BOX_TICKS), [signed(v) for v in BOX_TICKS])
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks([])
    ax.set_xlabel(f"Δα ({categories.cohort_phrase(CATEGORY_COHORT)})", labelpad=2.0)
    main_axes(ax)
    beside = dict(xycoords=("axes fraction", "data"), textcoords="offset points", ha="right", va="center_baseline", annotation_clip=False)
    for yy, name, count in zip(y, names, rows["n"]):
        ax.annotate(name, (0.0, yy), xytext=(-COUNT_COLUMN - 3.5, 0), fontsize=TICK, color=C.INK, **beside)
        ax.annotate(f"{int(count):,}", (0.0, yy), xytext=(-4.0, 0), fontsize=SMALL, color=C.MUTED, **beside)
    ax.annotate("n", (0.0, 1.0), xycoords="axes fraction", xytext=(-4.0 - categories.text_width(["100"], SMALL) / 2, 3.0), textcoords="offset points",
                ha="center", va="baseline", fontsize=SMALL, color=C.MUTED, annotation_clip=False)
    off = []
    for yy, c, d in zip(y, rows.index, data):
        q1, q3 = np.percentile(d, [25, 75])
        if d[d >= q1 - 1.5 * (q3 - q1)].min() < lo or d[d <= q3 + 1.5 * (q3 - q1)].max() > hi:
            raise SystemExit(f"{c}: a whisker leaves the axis")
        below = int((d < lo).sum())
        if (d > hi).any():
            raise SystemExit(f"{c}: genes above the axis")
        if below:                                                  # single genes beyond the axis: a mark and their number at its edge
            ax.plot([0.03], [yy], "<", transform=ax.get_yaxis_transform(), markersize=2.4, color=C.MUTED, markeredgewidth=0, clip_on=False,
                    zorder=6)
            ax.text(0.055, yy + 0.33, f"{below}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=FLOOR,
                    color=C.MUTED)
            off.append(f"{c}: {below} of {len(d)} genes below {lo}")
    if off:                                                        # what the arrowheads are, under the names
        at = (-NAME_COLUMN - COUNT_COLUMN + 3.2, -ROW_BOTTOM + 3.2)   # bp from the lower left corner of the axes
        ax.annotate("number of genes beyond the axis", (0.0, 0.0), xycoords="axes fraction", xytext=(at[0] + 3.4, at[1]), textcoords="offset points",
                    ha="left", va="baseline", fontsize=FLOOR, color=C.MUTED, annotation_clip=False)
        fig.add_artist(Line2D([(NAME_COLUMN + COUNT_COLUMN + at[0]) / BOX_WIDTH], [(ROW_BOTTOM + at[1] + 1.75) / ROW_HEIGHT], transform=fig.transFigure,
                              linestyle="none", marker="<", markersize=2.4, color=C.MUTED, markeredgewidth=0))
    print("category_boxes: genes beyond the Δα axis (marked at its edge with their number):\n   " + "\n   ".join(off or ["none"]))
    print("category_boxes: boxes above the prior:", [c for c, k in zip(rows.index, side) if k > 0], "; not significant:", [c for c, k in zip(rows.index, side) if k == 0])
    return [C.save_panel(fig, DEPLOYMENT, "category_boxes", exact=True)]


def category_effect_sizes() -> list:
    """Panel ``main_delta_alpha_liver/category_effect_sizes`` (b): two effect sizes of Δα for the rows of
    ``category_boxes``, as a heat map with the value in every cell.

    What it draws. One row per gene category, at the height of its box plot, and two columns. "shift": Cohen's d of Δα,
    the mean Δα of the genes of the category over the standard deviation of Δα of all genes of the cohort. "amplitude":
    Cohen's d of |Δα| against all genes, written with the sign of the shift. Both columns share one diverging color
    scale centered on zero; the value stands in every cell, so the panel has no color bar. An asterisk after a value
    marks q < 0.05, Benjamini-Hochberg over the 24 tests of the two columns (shift: Wilcoxon signed-rank test of Δα
    against zero; amplitude: one-sided Mann-Whitney test of |Δα| against the other genes).

    Reads the rows of ``largest_categories`` and nothing else, columns ``shift``, ``amplitude``, ``star_shift`` and
    ``star_amplitude``. They come from ``categories.category_table``, which reads ``final_state.npz`` of the stored
    four-species liver run (arrays ``alpha`` and ``prior``), the gene names of the data object of the cohort, its
    ortholog table and the category vocabulary ``notebooks/_gene_categories.py``. No value is computed here.

    Drawn together with ``category_boxes`` (same page height, same rows) and composed with ``X(...)``:

        python -m figures.redraw.main_figures category_boxes category_effect_sizes     # in chimera_env, from the repository root
    """
    from figures import style

    _, rows = largest_categories()
    n = len(rows)
    width, left = EFFECT_WIDTH, 1.0
    _, fig, ax = page(width, ROW_HEIGHT, left=left, right=width - left - 2 * CELL, bottom=ROW_BOTTOM, top=ROW_TOP)
    values = rows[["shift", "amplitude"]].to_numpy()
    stars = rows[["star_shift", "star_amplitude"]].to_numpy()
    vmax = float(max(np.abs(values).max(), 1.0))
    ax.imshow(values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    number = categories.text_width(["+0.00"], SMALL)
    edge = 0.5 + (number - categories.text_width([style.SIGNIFICANT], SMALL)) / 2 / CELL    # value and asterisk together are centered in the cell
    for i in range(n):
        for j in range(2):
            ink = "white" if abs(values[i, j]) > 0.6 * vmax else C.INK
            ax.text(j - 0.5 + edge, i, minus(f"{values[i, j]:+.2f}"), ha="right", va="center_baseline", fontsize=SMALL, color=ink)
            if stars[i, j]:                                         # the one mark for q < 0.05, directly after the value
                ax.text(j - 0.5 + edge, i, style.SIGNIFICANT, ha="left", va="center_baseline", fontsize=SMALL, color=ink)
    ax.xaxis.tick_top()
    ax.set_xticks([0, 1], ["shift", "amplitude"])
    ax.set_yticks([])
    main_axes(ax)
    ax.tick_params(length=0, pad=2.0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text((left + CELL) / width, (ROW_BOTTOM - 3.0) / ROW_HEIGHT, "Cohen's $d$\n(signed)", ha="center", va="top", fontsize=TITLE, linespacing=1.05)
    print(f"category_effect_sizes: color range ±{vmax:.2f}; stars: shift {int(stars[:, 0].sum())}, amplitude {int(stars[:, 1].sum())}")
    return [C.save_panel(fig, DEPLOYMENT, "category_effect_sizes", exact=True)]


# ---------------------------------------------------------------------------------------------------------------------
# supp_gin_inputs_controls, panel a: share of each input group under four attribution methods (liver 4 species)
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
GROUP_COLOR = C.GATE_GROUP_COLOR                                   # one color per input group, shared with the residue cards
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
# main_protein: the protein map of the four-species liver run
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
N_CLUSTERS, CLUSTER_SEED, CLUSTER_STARTS = 12, 0, 10
# Median Δα of the twelve clusters, sorted within the cohort, as printed on the first version of the panel.
FIRST_VERSION_CLUSTERS = {
    "liver_hmonkey": (-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.00, -0.00, +0.00, +0.00, +0.01, +0.01),
    "liver_hmouse": (-0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.02, -0.02, -0.02, -0.01, +0.00, +0.01),
    "liver_hzeb": (-0.00, +0.00, +0.00, +0.00, +0.00, +0.00, +0.01, +0.01, +0.01, +0.02, +0.02, +0.02),
    "liver_4sp": (-0.06, -0.05, -0.05, -0.03, -0.02, -0.01, -0.01, -0.01, -0.01, +0.01, +0.01, +0.02),
}
# The main figure, in bp on the grid of the layout tool. Upper row: four square pages, each a map without axes that fills
# its page (four pages and three gaps fill the text width). Lower row: the cluster panel and, under the category map, its key.
ATLAS_PAGE = on_grid((TEXT_WIDTH - 3 * GAP) / 4)                 # 116.16
ATLAS_EDGE = 1.0                                                 # white around a map
STRIP_SIZE = (on_grid(283.2), on_grid(52.0))                     # the cluster panel
ATLAS_KEY_WIDTH = on_grid(TEXT_WIDTH - GAP - STRIP_SIZE[0])      # 195.84
# Where the color bar of a map stands: in the upper right corner of the map, which holds no gene (bp from the lower left
# corner of the map: left end, lower edge, length, height).
ATLAS_BAR = (60.0, 101.8, 40.0, 3.5)
# The bar of the Δα map is longer, so that its three labels stand apart: as long as the corner allows between the genes
# at its left and the edge of the map, with the two end labels centered on the ends of the bar.
ATLAS_BAR_DELTA = (53.0, 101.8, 51.0, 3.5)
ATLAS_DOT = 1.3                                                  # pt: a gene on a map; larger on the Δα map, the map of the title
DELTA_CLIP = 0.15                                                # the Δα map takes its end colors at ±0.15, as its color bar says


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


# Categories named on the Δα map (panel b) and, at the same places, on the category map (panel d), so that the eye ties
# the two maps: those of the twelve whose genes lie together and that have free room beside them (the zinc-finger factors
# lie together too, with no free room). key -> where the name stands, in fractions of the map (``at``, ``ha``), in how many
# lines, and where its leader stops, in pt from the middle of the category (None: the name stands beside its category).
LANDMARKS = {
    "ribosomal_mt": dict(at=(0.635, 0.715), ha="left", lines=2, leader=5.0),
    "tf_homeobox": dict(at=(0.29, 0.915), ha="right", lines=2, leader=None),
    "ecm_other": dict(at=(0.80, 0.075), ha="center", lines=1, leader=9.0),
}
LANDMARK_RADIUS, LANDMARK_SHARE = 1.0, 0.6   # a named category has at least this share of its genes within this distance (map units) of their middle


def landmarks() -> list:
    """Name and middle (median place on the stored map) of the categories of ``LANDMARKS``. The panel stops if the genes of
    one of them do not lie together (``LANDMARK_SHARE`` of them within ``LANDMARK_RADIUS`` of the middle): a name on the
    map must stand for a place."""
    genes, _ = largest_categories()
    xy = protein_map()["xy"]
    category = genes["category"].to_numpy()
    out = []
    for key in LANDMARKS:
        place = xy[category == key]
        middle = np.median(place, axis=0)
        near = float((np.hypot(*(place - middle).T) < LANDMARK_RADIUS).mean())
        if near < LANDMARK_SHARE:
            raise SystemExit(f"landmark {key}: only {100 * near:.0f}% of its genes lie within {LANDMARK_RADIUS} of their middle")
        print(f"landmark {C.category(key)}: {len(place)} genes, {100 * near:.0f}% within {LANDMARK_RADIUS} of ({middle[0]:+.2f}, {middle[1]:+.2f})")
        out.append((key, C.category(key), middle))
    return out


def atlas_page():
    """A square page that the map fills (``ATLAS_PAGE``), without axes: the two directions of the map have no unit. Only
    the maps of the main figure use it; the supplementary archetype map keeps ``map_page``."""
    fd, fig, ax = page(ATLAS_PAGE, ATLAS_PAGE, left=ATLAS_EDGE, right=ATLAS_EDGE, bottom=ATLAS_EDGE, top=ATLAS_EDGE)
    protein_map()
    ax.set_xlim(*MAP_VIEW[0])
    ax.set_ylim(*MAP_VIEW[1])
    ax.axis("off")
    return fig, ax


def name_landmarks(ax, named) -> None:
    """The names of ``LANDMARKS`` on a map, each at its fixed place, with a line to the middle of its category where it stands apart."""
    for key, text, middle in named:
        mark = LANDMARKS[key]
        leader = None if mark["leader"] is None else dict(arrowstyle="-", color=C.MUTED, linewidth=THIN, shrinkA=1.0, shrinkB=mark["leader"])
        ax.annotate(text.replace(" ", "\n", mark["lines"] - 1), xy=tuple(middle), xycoords="data", xytext=mark["at"], textcoords="axes fraction",
                    ha=mark["ha"], va="center", fontsize=SMALL, color=C.INK, linespacing=1.0, zorder=6, arrowprops=leader)


def colored_map(name: str, values: np.ndarray, cmap, vmin: float, vmax: float, label: str, ticks: dict, extend: str = "neither",
                named: tuple = (), dot: float = ATLAS_DOT, directions: bool = False, bar: tuple = ATLAS_BAR) -> list:
    """One map of the protein-space atlas, colored by a value of every gene (panels a to c of the main figure
    ``main_protein``: ``main_protein/atlas_alpha``, ``atlas_delta_alpha``, ``atlas_correction``).

    Every dot is a gene at its stored place on the map (``protein_map``: ``cache/esm2_umap.npy`` of the four-species
    liver run), colored by ``values`` (α, Δα = α − prior, or the absolute correction of the gate network, from
    ``final_state.npz`` of the same run: arrays ``alpha``, ``prior``, ``correction``). The map has no axes; with
    ``directions`` two short arrows in the lower left corner name its two directions (panel a). The color bar stands in
    the upper right corner of the map, which holds no gene (``bar``), with its title above it; ``ticks`` maps the
    values that are marked to their labels, and a value beyond ``vmin`` or ``vmax`` takes the end color (``extend``).
    The genes are drawn in their stored order, which does not depend on the value, so that no class of genes hides
    another. ``named``: categories to name on the map (``landmarks``; the Δα map names three). Nothing is fitted.

        python -m figures.redraw.main_figures atlas_alpha atlas_delta_alpha atlas_correction     # in chimera_env, from the repository root
    """
    fig, ax = atlas_page()
    xy = protein_map()["xy"]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=dot ** 2, alpha=0.9, linewidths=0, rasterized=True)
    name_landmarks(ax, named)
    if directions:                                                # the two directions of the map, once for the four maps
        corner = 2.5                                              # bp from the lower left corner of the map
        for end, word, turn, at in (((corner + 12.0, corner), "UMAP 1", 0, (corner + 14.0, corner)), ((corner, corner + 12.0), "UMAP 2", 90, (corner, corner + 14.0))):
            ax.annotate("", xy=end, xytext=(corner, corner), xycoords="axes points", textcoords="axes points",
                        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.16", color=C.MUTED, linewidth=AXIS_LINE, shrinkA=0, shrinkB=0, mutation_scale=7))
            ax.annotate(word, xy=at, xycoords="axes points", ha="left", va="center", rotation=turn, rotation_mode="anchor", fontsize=SMALL, color=C.MUTED)
    x0, y0, length, height = bar
    cax = fig.add_axes([(ATLAS_EDGE + x0) / ATLAS_PAGE, (ATLAS_EDGE + y0) / ATLAS_PAGE, length / ATLAS_PAGE, height / ATLAS_PAGE])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=list(ticks), extend=extend, extendfrac=0.07)
    cb.ax.set_xticklabels(list(ticks.values()))
    cb.ax.tick_params(labelsize=TICK, length=2.0, width=TICK_LINE, pad=1.3)
    cb.ax.set_title(label, fontsize=TITLE, pad=3.0)
    cb.outline.set_linewidth(THIN)
    cb.solids.set_rasterized(True)
    cb.solids.set_alpha(1.0)
    return [C.save_panel(fig, PROTEIN, name, exact=True)]


def atlas_alpha() -> list:
    return colored_map("atlas_alpha", protein_map()["alpha"], "viridis", 0.0, 1.0, "α", {0: "0", 0.5: "0.5", 1: "1"}, directions=True)


def atlas_delta_alpha() -> list:
    named = tuple(landmarks())  # before the page is set up: reading the categories resets the drawing defaults (600 dpi for the clouds)
    delta = protein_map()["delta_alpha"]
    beyond = np.abs(delta) > DELTA_CLIP
    print(f"atlas_delta_alpha: color range ±{DELTA_CLIP}; {int(beyond.sum())} genes ({100 * beyond.mean():.1f}%) beyond it take the end colors, "
          f"as the color bar says; largest |Δα| {np.abs(delta).max():.3f}")
    return colored_map("atlas_delta_alpha", delta, "RdBu_r", -DELTA_CLIP, DELTA_CLIP, "Δα",
                       {-DELTA_CLIP: f"≤\u2009{signed(-DELTA_CLIP, 2)}", 0: "0", DELTA_CLIP: f"≥\u2009{signed(DELTA_CLIP, 2)}"}, extend="both", named=named,
                       dot=1.6, bar=ATLAS_BAR_DELTA)


def atlas_correction() -> list:
    size = np.abs(protein_map()["correction"])
    limit = float(np.percentile(size, 99))                        # as in the first version: 99th percentile of |correction|
    print(f"atlas_correction: absolute correction, color range 0 to {limit:.2f}; {int((size > limit).sum())} genes above it take the "
          f"end color; {100 * float((protein_map()['correction'] < 0).mean()):.1f}% of the corrections are negative")
    light_to_dark = mpl.colors.LinearSegmentedColormap.from_list("correction", mpl.colormaps["magma_r"](np.linspace(0.05, 0.90, 256)))
    return colored_map("atlas_correction", size, light_to_dark, 0.0, limit, "|$c_g$|", {0: "0", 1: "1", limit: f"≥\u2009{limit:.1f}"}, extend="max")


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
    """Panel d of the main figure ``main_protein`` (``main_protein/atlas_categories``): the twelve gene categories with
    the most genes on the map, at most 30 genes each (``category_draws``; the same draw on every run), over all genes in
    light gray. The three categories that the Δα map names are named here at the same places (``landmarks``). Reads the
    stored map and the category vocabulary ``notebooks/_gene_categories.py``; the key is the separate panel
    ``atlas_categories_key``.

        python -m figures.redraw.main_figures atlas_categories atlas_categories_key     # in chimera_env, from the repository root
    """
    draws = category_draws()  # before the page is set up: reading the categories resets the drawing defaults (600 dpi for the clouds)
    named = tuple(landmarks())
    fig, ax = atlas_page()
    xy = protein_map()["xy"]
    ax.scatter(xy[:, 0], xy[:, 1], s=ATLAS_DOT ** 2, color="#DDDDDD", linewidths=0, rasterized=True, zorder=1)
    for _, _, color, index in draws:
        ax.scatter(xy[index, 0], xy[index, 1], s=2.4 ** 2, color=color, edgecolor="white", linewidth=0.3, zorder=4, rasterized=True)
    name_landmarks(ax, named)
    return [C.save_panel(fig, PROTEIN, "atlas_categories", exact=True)]


def atlas_categories_key() -> list:
    """The key of ``atlas_categories`` as a panel of its own (``main_protein/atlas_categories_key``): the twelve
    categories with their colors, in two columns of six, in the order of the rows of ``category_boxes`` (panel a of the
    figure ``main_delta_alpha_liver``: from the largest median Δα down). It stands under the category map, beside
    the cluster panel, flush with the right edge of the figure."""
    C.design()
    _, rows = largest_categories()
    color = {c: col for c, _, col, _ in category_draws()}
    C.design()                                                    # reading the categories resets the drawing defaults
    width, height = ATLAS_KEY_WIDTH, STRIP_SIZE[1]
    fig = plt.figure(figsize=(width / 72, height / 72))
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=3.4, markerfacecolor=color[c], markeredgecolor="white",
                      markeredgewidth=0.3, label=C.category(c)) for c in rows.index]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1 - 1.5 / width, 1 - 1.5 / height), ncol=2, frameon=False, fontsize=TICK,
               handlelength=0.9, handletextpad=0.4, columnspacing=1.8, labelspacing=0.32, borderpad=0, borderaxespad=0)
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
    """Panel e of the main figure ``main_protein`` (``main_protein/atlas_clusters``): for each of the four liver cohorts
    the median Δα of twelve protein clusters, sorted from lowest to highest within the cohort, with the value in every
    cell. The values are printed in units of 0.01 with their sign and one decimal ("−6.2" is a median Δα of −0.062), as
    the axis title says, so that the spread along a row can be read and no value shows as a signed zero. The medians
    are those of ``cluster_medians`` (``final_state.npz`` of the four liver runs and the 64 ESM-2 components stored
    with the data objects); the script stops if they differ, at two decimals, from the values of the first version of
    the panel, or if a value would print as zero.

        python -m figures.redraw.main_figures atlas_clusters     # in chimera_env, from the repository root
    """
    C.design()
    medians = cluster_medians()
    table = np.vstack([medians[c] for c in atlas.LIVER])
    labels = [C.COHORT[c] for c in atlas.LIVER]                    # "Liver Human+Monkey", ...
    cell, row = 17.0, 8.8                                          # bp
    width, height = STRIP_SIZE
    left = width - 3.0 - N_CLUSTERS * cell
    if 1.5 + categories.text_width(labels, TICK) + 3.0 > left:
        raise SystemExit("atlas_clusters: a cohort name is wider than its column")
    _, fig, ax = page(width, height, left=left, right=3.0, bottom=height - 1.0 - len(labels) * row, top=1.0)
    limit = float(np.abs(table).max())
    ax.imshow(table, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    if (np.round(100 * table, 1) == 0).any():
        raise SystemExit("atlas_clusters: a median prints as zero at one decimal of the unit 0.01; print it with more")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, minus(f"{100 * table[i, j]:+.1f}"), ha="center", va="center_baseline", fontsize=SMALL,
                    color="white" if abs(table[i, j]) > 0.6 * limit else C.INK)
    ax.set_xticks(range(N_CLUSTERS), [str(i + 1) for i in range(N_CLUSTERS)])
    ax.set_yticks(range(len(labels)), labels)
    main_axes(ax)
    ax.tick_params(length=0, pad=2.0)
    ax.tick_params(axis="x", pad=0.8, labelsize=SMALL)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("median Δα (×10⁻²) of 12 protein clusters, sorted within cohort", fontsize=TITLE, labelpad=1.0)
    print(f"atlas_clusters: color range ±{limit:.3f}; printed from {minus(f'{100 * table.min():+.1f}')} to {minus(f'{100 * table.max():+.1f}')} (units of 0.01)")
    return [C.save_panel(fig, PROTEIN, "atlas_clusters", exact=True)]


# ---------------------------------------------------------------------------------------------------------------------
# supp_atlas_archetypes: the three gene archetypes of the gate network (liver 4 species)
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


PANELS = {"scatter": scatter, "shift": shift, "distribution": distribution, "training": training,
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
