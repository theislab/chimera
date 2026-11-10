"""E2 panels (exploratory): protein and gene properties, protein-domain families, gene age by duplication class. Three
panels of the figure supp_protein_architecture, saved through figures/style.py to figures/panels/supp_protein_architecture/;
reads the result tables of results/analysis/E2 only. The panel domains is also a panel of the main figure on tissues.

Every panel is drawn at its size on the page and saved at that exact size, so the figure composes at scale 1.

    PYTHONPATH=$PWD python -m analysis.tasks.E2_protein.figures [panel ...]     # every panel, or the named ones, from the cached tables (draw_panels)
    CHIMERA_FIGURES_OUT=<folder> ... the same command                            # dry run: panels under <folder>/panels
"""
from __future__ import annotations

import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analysis.common import gene_ids  # noqa: E402
from figures import style  # noqa: E402

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this task
O = gene_ids.REPO_ROOT / "results" / "analysis" / "E2"
FIGURE = "supp_protein_architecture"  # the panels go to figures/panels/<FIGURE>/
# Sizes in bp at print size. properties and domains share a row and fill the text width. In the main figure on tissues,
# domains stands beside supp_regulatory_architecture/properties, drawn with the same height and margins (220 + 256 bp and a
# 10 bp gap fill the text width there).
ROW_H = 244
W_DOMAINS = 220
W_PROPERTIES = style.TEXTWIDTH - style.GAP - W_DOMAINS
TOP, BOTTOM = 17, 34  # bp from the panel edge to the axes: heading above; tick labels, axis label and key below
SIZE_AGE = (style.TEXTWIDTH, 116)  # gene age by duplication class, in a row of its own
S_TITLE, S_TICK, S_TEXT = style.SIZE_TITLE, style.SIZE_TICK, style.SIZE_TEXT  # axis labels take SIZE_LABEL from style.apply()
INK, MUTED, RULE = style.INK, style.MUTED, style.RULE
GATE, PER_GENE, PROFILE = style.READOUT["delta_alpha"], style.READOUT["delta_alpha_per_gene"], style.READOUT["profile_correlation"]
# the three readouts of every row, top to bottom: column of the tables, mark (color, marker, name), area of the symbol
SERIES = (("D_can", GATE, 16), ("D_free", PER_GENE, 19), ("B1", PROFILE, 15))
# duplication classes of the gene-age panel: key, mark. Not in the house table; hues that no readout of this figure has
DUP = (("ohnolog", style.Mark("#009E73", "^", "2R ohnologs")), ("ssd", style.Mark("#E69F00", "o", "small-scale duplicates")),
       ("single", style.Mark("#7A7A7A", "s", "singletons")))

PROPS = [  # (key, label), top to bottom
    ("gtex_tau", "Tissue specificity (GTEx τ)"),
    ("hpa_loc_plasma_membrane", "Plasma membrane (HPA)"),
    ("ohnolog", "2R ohnolog"),
    ("signal_peptide", "Signal peptide"),
    ("paralog_count", "Paralog count"),
    ("transmembrane", "Transmembrane"),
    ("hpa_loc_junctions", "Cell junctions (HPA)"),
    ("hpa_secreted", "Secreted (HPA)"),
    ("phaplo", "Haploinsufficiency (pHaplo)"),
    ("cd_marker", "CD marker"),
    ("immune", "Immunity keyword"),
    ("loeuf", "LoF tolerance (LOEUF)"),
    ("ptriplo", "Triplosensitivity (pTriplo)"),
    ("gene_age", "Gene age (older)"),
    ("complex_member", "Complex member"),
    ("common_essential", "Common essential (DepMap)"),
    ("hpa_loc_nucleus", "Nuclear (HPA)"),
    ("depmap_essentiality", "Essentiality (DepMap)"),
]

FAMS = [  # (family key, short label)
    ("pfam:PF00307", "Calponin homology"), ("pfam:PF00041", "Fibronectin III"), ("pfam:PF00373", "FERM"),
    ("pfam:PF00018", "SH3"), ("pfam:PF00001", "7TM receptor"), ("tf_dbd:Homeodomain", "Homeodomain TF"),
    ("pfam:PF00169", "PH"), ("pfam:PF07714", "Tyrosine/S-T kinase"), ("pfam:PF00010", "bHLH"),
    ("pfam:PF00017", "SH2"), ("pfam:PF00620", "RhoGAP"), ("pfam:PF00168", "C2"),
    ("pfam:PF00226", "DnaJ"), ("pfam:PF12937", "F-box"), ("pfam:PF00443", "USP deubiquitinase"),
    ("pfam:PF01352", "KRAB"), ("pfam:PF00271", "Helicase C"), ("pfam:PF00400", "WD40"),
    ("pfam:PF00628", "PHD finger"), ("pfam:PF13181", "TPR"), ("pfam:PF00632", "HECT ligase"),
    ("pfam:PF00076", "RNA recognition motif"), ("pfam:PF00439", "Bromodomain"),
]


def save(fig, name: str):
    """One panel at its exact size: figures/panels/<FIGURE>/<name>.pdf."""
    return style.save_panel(fig, f"{FIGURE}/{name}", exact=True)


def panel_axes(width: float, height: float, left: float, right: float, top: float = TOP, bottom: float = BOTTOM):
    """A figure of width x height bp with one axes inside the given margins (bp)."""
    fig = plt.figure(figsize=(style.inch(width), style.inch(height)))
    ax = fig.add_axes([left / width, bottom / height, (width - left - right) / width, (height - top - bottom) / height])
    return fig, ax


def rows_axis(ax, labels: list, title: str, xlabel: str, italic: bool = False) -> None:
    """Rows from top to bottom, a zero line, the heading at the left edge of the panel and the axis label."""
    n = len(labels)
    ax.axvline(0, color=RULE, lw=style.LW_REF, zorder=0)
    ax.set_yticks(range(n), labels[::-1], fontsize=S_TICK, style="italic" if italic else "normal")
    ax.set_ylim(-.7, n - .3)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.figure.text(1.5 / (ax.figure.get_figwidth() * 72), 1 - 2 / (ax.figure.get_figheight() * 72), title, ha="left", va="top", fontsize=S_TITLE)


def readout_key(fig) -> None:
    """The key of the three readouts, in one line at the bottom of the panel."""
    h = [Line2D([], [], ls="", marker=m.marker, color=m.color, ms=np.sqrt(area), mec="white", mew=.4, label=m.name) for _, m, area in SERIES]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=len(h), fontsize=S_TICK, handlelength=1.0, handletextpad=.2,
               columnspacing=.9, borderpad=0, borderaxespad=.25)


def panel_properties():
    """Partial Spearman of 18 protein and gene properties with each readout: a small dot per tissue, the symbol at the
    median over tissues; at the right, the number of tissues in which the per-gene value has the sign of its median."""
    tv = pd.read_csv(O / "scan_tissue_values.csv")
    sm = pd.read_csv(O / "scan_summary.csv", index_col=0)
    right = 22
    fig, ax = panel_axes(W_PROPERTIES, ROW_H, left=96, right=right)
    n = len(PROPS)
    for i, (k, lab) in enumerate(PROPS):
        y = n - 1 - i
        for (ro, m, area), off in zip(SERIES, (.22, 0, -.22)):
            v = tv[(tv.property == k) & (tv.readout == ro)].rho_p.to_numpy()
            ax.scatter(v, np.full(len(v), y + off), s=4, color=m.color, alpha=.35, lw=0)
            ax.scatter([np.median(v)], [y + off], s=area, marker=m.marker, color=m.color, edgecolor="white", linewidth=.4, zorder=3)
        ax.annotate(f"{int(sm.loc[k, 'D_free_same_sign'])}/10", (1, y), xycoords=ax.get_yaxis_transform(), xytext=(right - 2, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=S_TEXT, color=INK)
        if i in (9, 11):
            ax.axhline(y - .5, color=RULE, lw=.5)
    ax.set_xlim(-.26, .34)  # holds every tissue (lowest -0.234, highest 0.322)
    ax.set_xticks([-.2, -.1, 0, .1, .2, .3])
    rows_axis(ax, [lab for _, lab in PROPS], "Protein and gene properties", "partial Spearman with the readout")
    ax.annotate("tissues with the sign of\nthe median, per-gene gate", (1, 1), xycoords="axes fraction", xytext=(right - 2, 2.5), textcoords="offset points",
                ha="right", va="bottom", fontsize=S_TEXT, color=INK, linespacing=1.1)
    readout_key(fig)
    return save(fig, "properties")


def panel_families():
    """Residual readout of the members of 23 protein-domain and transcription-factor families: median and interquartile
    range over tissues."""
    tv = pd.read_csv(O / "family_tissue_values.csv")
    fig, ax = panel_axes(W_DOMAINS, ROW_H, left=73, right=9)
    n = len(FAMS)
    for i, (k, lab) in enumerate(FAMS):
        y = n - 1 - i
        for (ro, m, area), off in zip(SERIES, (.2, 0, -.2)):
            v = tv[(tv.family == k) & (tv.readout == ro)].mean_resid.dropna().to_numpy()
            if not len(v):
                continue
            lo, hi = np.quantile(v, [.25, .75])
            ax.plot([lo, hi], [y + off] * 2, color=m.color, lw=.8, alpha=.6)
            ax.scatter([np.median(v)], [y + off], s=area, marker=m.marker, color=m.color, edgecolor="white", linewidth=.4, zorder=3)
        if i == 11:
            ax.axhline(y - .5, color=RULE, lw=.5)
    ax.set_xlim(-.75, 1.0)  # holds every interquartile range (lowest quartile -0.71, highest 0.70)
    ax.set_xticks([-.5, 0, .5, 1])
    rows_axis(ax, [lab for _, lab in FAMS], "Protein-domain families", "residual readout of family members (SD)")
    ax.text(.98, 11.25, "signaling,\ncell surface", ha="right", fontsize=S_TEXT, color=MUTED, linespacing=1.1)
    ax.text(.98, 1.0, "RNA, chromatin,\nproteostasis", ha="right", fontsize=S_TEXT, color=MUTED, linespacing=1.1)
    readout_key(fig)
    return save(fig, "domains")


def panel_age_dup():
    """Residual Δα with the per-gene gate by the clade in which the gene arose, for the three duplication classes: median
    and interquartile range over tissues."""
    G = pd.read_parquet(O / "gene_resid.parquet", columns=["cohort", "tissue", "hsapiens_gene_id", "r_D_free"])
    Pp = pd.read_parquet(O / "props.parquet", columns=["hsapiens_gene_id", "gene_age", "ohnolog", "paralog_count"])
    G = G.merge(Pp, on="hsapiens_gene_id")
    G["dup"] = np.where(G.ohnolog > 0, "ohnolog", np.where(G.paralog_count > 0, "ssd", "single"))
    bins = [(-1, 4, "Tetrapods\nand younger"), (4, 5, "Bony\nvertebrates"), (5, 6, "Jawed\nvertebrates"), (6, 7, "Vertebrates"),
            (7, 8, "Chordates"), (8, 9, "Bilaterians"), (9, 10, "Opisthokonts")]
    G["b"] = pd.cut(G.gene_age, [b[0] for b in bins] + [10], labels=False)
    fig, ax = panel_axes(*SIZE_AGE, left=44, right=118, top=5, bottom=32)
    for d, m in DUP:
        a = G[G.dup == d].groupby(["tissue", "cohort", "b"]).r_D_free.agg(["mean", "size"]).reset_index()
        a = a[a["size"] >= 10].groupby(["tissue", "b"])["mean"].median().reset_index()
        s = a.groupby("b")["mean"].agg(med="median", lo=lambda v: np.quantile(v, .25), hi=lambda v: np.quantile(v, .75),
                                       nt="size")
        s = s[s.nt >= 7]
        x = s.index.to_numpy()
        ax.fill_between(x, s.lo, s.hi, color=m.color, alpha=.15, lw=0)
        ax.plot(x, s.med, color=m.color, lw=1.1, marker=m.marker, ms=3.5, mec="white", mew=.4, label=m.name)
    ax.axhline(0, color=RULE, lw=style.LW_REF, zorder=0)
    ax.set_xticks(range(len(bins)), [b[2] for b in bins], fontsize=S_TICK, linespacing=1.05)
    ax.set_yticks([-.3, -.2, -.1, 0, .1, .2])
    ax.set_xlabel("clade in which the gene arose (younger to older)", labelpad=2)
    ax.set_ylabel("residual Δα,\nper-gene gate (SD)", linespacing=1.1)
    ax.legend(loc="center left", bbox_to_anchor=(1.03, .5), fontsize=S_TICK, handletextpad=.4, borderaxespad=0, labelspacing=.6)
    return save(fig, "age_duplication")


# panel name in figures/panels/<FIGURE>/ -> the function that draws it
DRAW = {"properties": panel_properties, "domains": panel_families, "age_duplication": panel_age_dup}


def draw_panels(names=None) -> list:
    """Draw the panels of this task from the cached tables of results/analysis/E2 and nothing else. ``names``: panel names
    as in ``DRAW`` (default: all three). Returns the files written."""
    style.apply()
    return [DRAW[n]() for n in (DRAW if names is None else names)]


if __name__ == "__main__":
    for path in draw_panels(sys.argv[1:] or None):
        print(path)
