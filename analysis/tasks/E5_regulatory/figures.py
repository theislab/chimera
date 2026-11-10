"""E5 panels (exploratory): regulatory and gene-architecture features, targets of transcription factors, regulatory input
by quintile. Three panels of the figure supp_regulatory_architecture, saved through figures/style.py to
figures/panels/supp_regulatory_architecture/; reads the result tables of results/analysis/E5 only. The panel properties is
also a panel of the main figure on tissues, beside supp_protein_architecture/domains: both are drawn by the same helpers
(analysis/tasks/E2_protein/figures.py) with the same height, margins and key.

Every panel is drawn at its size on the page and saved at that exact size, so the figure composes at scale 1.

    PYTHONPATH=$PWD python -m analysis.tasks.E5_regulatory.figures [panel ...]     # every panel, or the named ones, from the cached tables (draw_panels)
    CHIMERA_FIGURES_OUT=<folder> ... the same command                               # dry run: panels under <folder>/panels
"""
from __future__ import annotations

import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.common import gene_ids  # noqa: E402
from analysis.tasks.E2_protein.figures import GATE, PER_GENE, PROFILE, ROW_H, SERIES, W_DOMAINS, panel_axes, readout_key, rows_axis  # noqa: E402
from figures import style  # noqa: E402

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this task
O = gene_ids.REPO_ROOT / "results" / "analysis" / "E5"
FIGURE = "supp_regulatory_architecture"  # the panels go to figures/panels/<FIGURE>/
# Sizes in bp at print size. properties and regulons share a row (height ROW_H) and fill the text width. In the main figure
# on tissues, properties and supp_protein_architecture/domains fill the text width with a 10 bp gap.
W_PROPERTIES = style.TEXTWIDTH - 10 - W_DOMAINS
W_REGULONS = style.TEXTWIDTH - style.GAP - W_PROPERTIES
SIZE_DOSE = (style.TEXTWIDTH, 116)  # regulatory input by quintile, in a row of its own
S_TICK, S_TEXT = style.SIZE_TICK, style.SIZE_TEXT  # axis labels take SIZE_LABEL from style.apply()
INK, MUTED, RULE = style.INK, style.MUTED, style.RULE
OFFSETS = (.22, 0, -.22)  # place of the three readouts of SERIES within a row, top to bottom (in rows)

PROPS = [  # (key, label, group break after)
    ("n_dels_domain", "Distal enhancers in domain (ENCODE)"),
    ("log_great_domain", "Regulatory domain size (GREAT)"),
    ("tf_indegree", "Regulating TFs (DoRothEA)"),
    ("abc_links_mean", "Enhancer-gene links (ABC)"),
    ("n_ccre_50kb", "cCREs within 50 kb"),
    ("es_h3k27me3", "H3K27me3 in ES cells"),
    ("bivalent_npc", "Bivalent promoter (NPC)"),
    ("prc2_targets", "PRC2 target (ES)"),
    ("phylop_promoter", "Promoter phyloP"),
    ("mir_density", "Conserved miRNA sites per kb 3'UTR"),
    ("log_length_utr3", "3'UTR length"),
    ("n_tss", "Alternative promoters"),
    ("n_transcripts", "Transcript isoforms"),
    ("phylop_cds", "Coding phyloP"),
    ("promoter_cgi", "CpG-island promoter"),
    ("bidirectional", "Bidirectional promoter"),
    ("gene_density", "Gene density (±500 kb)"),
    ("housekeeping", "Housekeeping gene"),
]
BREAKS = {4, 7, 13}

TFS_UP = ["ZEB2", "NR2F2", "RBPJ", "RUNX2", "GATA6", "IKZF1", "AHR", "KLF9", "TCF7L2"]
TFS_DN = ["ZFX", "ZBTB33", "NFYA", "MBD1", "FOXK2", "NRF1"]


def save(fig, name: str):
    """One panel at its exact size: figures/panels/<FIGURE>/<name>.pdf."""
    return style.save_panel(fig, f"{FIGURE}/{name}", exact=True)


def panel_properties():
    """Partial Spearman of 18 regulatory and gene-architecture features with each readout: a small dot per tissue, the
    symbol at the median over tissues; at the right, the number of tissues in which the per-gene value has the sign of its median."""
    tv = pd.read_csv(O / "scan_tissue_values.csv")
    sm = pd.read_csv(O / "scan_summary.csv", index_col=0)
    right = 22
    fig, ax = panel_axes(W_PROPERTIES, ROW_H, left=125, right=right)
    n = len(PROPS)
    for i, (k, lab) in enumerate(PROPS):
        y = n - 1 - i
        for (ro, m, area), off in zip(SERIES, OFFSETS):
            v = tv[(tv.property == k) & (tv.readout == ro)].rho_p.to_numpy()
            ax.scatter(v, np.full(len(v), y + off), s=4, color=m.color, alpha=.35, lw=0)
            ax.scatter([np.median(v)], [y + off], s=area, marker=m.marker, color=m.color, edgecolor="white", linewidth=.4, zorder=3)
        ax.annotate(f"{int(sm.loc[k, 'D_free_same_sign'])}/10", (1, y), xycoords=ax.get_yaxis_transform(), xytext=(right - 2, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=S_TEXT, color=INK)
        if i in BREAKS:
            ax.axhline(y - .5, color=RULE, lw=.5)
    ax.set_xlim(-.28, .22)  # holds every tissue (lowest -0.254, highest 0.195)
    ax.set_xticks([-.2, -.1, 0, .1, .2])
    rows_axis(ax, [lab for _, lab in PROPS], "Regulatory and gene-architecture features", "partial Spearman with the readout")
    ax.annotate("tissues with the sign of\nthe median, per-gene gate", (1, 1), xycoords="axes fraction", xytext=(right - 2, 2.5), textcoords="offset points",
                ha="right", va="bottom", fontsize=S_TEXT, color=INK, linespacing=1.1)
    readout_key(fig)
    return save(fig, "properties")


def panel_dose():
    """Residual readout by quintile of the number of distal enhancers and of regulating transcription factors: median and
    interquartile range over tissues; dotted, the per-gene gate with cell-type specificity held equal."""
    q = pd.read_csv(O / "quintiles.csv")
    (w, h), left, bottom, top, aw, gap = SIZE_DOSE, 37, 33, 5, 150, 16
    fig = plt.figure(figsize=(style.inch(w), style.inch(h)))
    axs = [fig.add_axes([(left + j * (aw + gap)) / w, bottom / h, aw / w, (h - top - bottom) / h]) for j in range(2)]
    axs[1].sharey(axs[0])
    lines = (("D_can", GATE, "-", GATE.name), ("D_free", PER_GENE, "-", PER_GENE.name),
             ("rt_D_free", PER_GENE, ":", f"{PER_GENE.name},\nat equal cell-type specificity"), ("B1", PROFILE, "-", PROFILE.name))
    for ax, (p, lab) in zip(axs, (("n_dels_domain", "distal enhancers in domain"), ("tf_indegree", "regulating TFs"))):
        d = q[q.property == p].groupby(["tissue", "quintile"])[["D_can", "D_free", "B1", "rt_D_free"]].median().reset_index()
        for y, m, ls, nm in lines:
            s = d.groupby("quintile")[y].agg(med="median", lo=lambda v: np.quantile(v, .25), hi=lambda v: np.quantile(v, .75))
            x = s.index.to_numpy()
            if ls == "-":
                ax.fill_between(x, s.lo, s.hi, color=m.color, alpha=.12, lw=0)
            ax.plot(x, s.med, color=m.color, ls=ls, lw=1.1, marker=m.marker, ms=3.4 if ls == "-" else 2.6, mec="white", mew=.4, label=nm)
        ax.axhline(0, color=RULE, lw=style.LW_REF, zorder=0)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel(f"{lab}\n(quintile, fewest → most)", labelpad=2)
    axs[1].tick_params(labelleft=False)
    axs[0].set_ylabel("residual readout (SD)")
    axs[1].legend(loc="center left", bbox_to_anchor=(1.05, .5), fontsize=S_TICK, handlelength=2.6, handletextpad=.5, borderaxespad=0, labelspacing=.7)
    return save(fig, "dose")


def panel_regulons():
    """Residual readout of the targets of 15 transcription factors, with the number of regulating factors held equal:
    median and interquartile range over tissues."""
    r = pd.read_csv(O / "regulon_indeg.csv")
    tv = r.groupby(["tf", "tissue"])[["D_can", "D_free", "B1"]].median().reset_index()
    tfs = TFS_UP + TFS_DN
    fig, ax = panel_axes(W_REGULONS, ROW_H, left=31, right=10)
    n = len(tfs)
    for i, tf in enumerate(tfs):
        y = n - 1 - i
        for (ro, m, area), off in zip(SERIES, OFFSETS):
            v = tv.loc[tv.tf == tf, ro].dropna().to_numpy()
            lo, hi = np.quantile(v, [.25, .75])
            ax.plot([lo, hi], [y + off] * 2, color=m.color, lw=.8, alpha=.6)
            ax.scatter([np.median(v)], [y + off], s=area, marker=m.marker, color=m.color, edgecolor="white", linewidth=.4, zorder=3)
        if i == len(TFS_UP) - 1:
            ax.axhline(y - .5, color=RULE, lw=.5)
    ax.set_xlim(-.65, .75)  # holds every interquartile range (lowest quartile -0.58, highest 0.54)
    ax.set_xticks([-.5, -.25, 0, .25, .5, .75])
    rows_axis(ax, tfs, "Targets of TFs, at equal number of regulating TFs", "residual readout of targets (SD)", italic=True)
    ax.text(-.62, n - 1.6, "lineage and\nsignaling TFs", ha="left", fontsize=S_TEXT, color=MUTED, linespacing=1.1)
    ax.text(.73, 1.6, "CpG-promoter\nTFs", ha="right", fontsize=S_TEXT, color=MUTED, linespacing=1.1)
    readout_key(fig)
    return save(fig, "regulons")


# panel name in figures/panels/<FIGURE>/ -> the function that draws it
DRAW = {"properties": panel_properties, "regulons": panel_regulons, "dose": panel_dose}


def draw_panels(names=None) -> list:
    """Draw the panels of this task from the cached tables of results/analysis/E5 and nothing else. ``names``: panel names
    as in ``DRAW`` (default: all three). Returns the files written."""
    style.apply()
    return [DRAW[n]() for n in (DRAW if names is None else names)]


if __name__ == "__main__":
    for path in draw_panels(sys.argv[1:] or None):
        print(path)
