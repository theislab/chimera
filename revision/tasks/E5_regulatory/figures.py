"""E5 candidate panels (exploratory). Usage: python -m revision.tasks.E5_regulatory.figures

Panels in results/revision/E5/panels/; composed with F1_figures/compose.py --spec revision/tasks/E5_regulatory/specs_e5.py
(CHIMERA_MANUSCRIPT pointed at results/revision/E5/ms_stage so nothing is written into the manuscript).
House style of E2 (Okabe-Ito hues with a distinct marker per series; palette validator needs node, not installed here).
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from revision.common import gene_ids  # noqa: E402

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
                     "xtick.major.width": .6, "ytick.major.width": .6, "legend.frameon": False, "legend.fontsize": 6.2,
                     "axes.titlesize": 7.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})
O = gene_ids.REPO_ROOT / "results" / "revision" / "E5"
P = O / "panels"
COL = dict(free="#D55E00", can="#0072B2", b1="#7A7A7A")
GRID = "#E6E6E6"
SERIES = (("D_can", .22, "o", COL["can"]), ("D_free", 0, "^", COL["free"]), ("B1", -.22, "D", COL["b1"]))

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


def save(fig, name):
    P.mkdir(parents=True, exist_ok=True)
    fig.savefig(P / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(P / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def panel_properties():
    tv = pd.read_csv(O / "scan_tissue_values.csv")
    sm = pd.read_csv(O / "scan_summary.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(3.3, 3.9))
    n = len(PROPS)
    for i, (k, lab) in enumerate(PROPS):
        y = n - 1 - i
        for ro, off, mk, c in SERIES:
            v = tv[(tv.property == k) & (tv.readout == ro)].rho_p.to_numpy()
            ax.scatter(v, np.full(len(v), y + off), s=4, color=c, alpha=.35, lw=0)
            ax.scatter([np.median(v)], [y + off], s=18 if ro != "B1" else 12, marker=mk, color=c, edgecolor="white",
                       linewidth=.4, zorder=3)
        ax.text(0.295, y, f"{int(sm.loc[k, 'D_free_same_sign'])}/10", va="center", ha="right", fontsize=5.6, color="#555555")
        if i in BREAKS:
            ax.axhline(y - .5, color=GRID, lw=.6)
    ax.axvline(0, color="#999999", lw=.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in PROPS][::-1], fontsize=6.2)
    ax.set_xlim(-.3, .3)
    ax.set_ylim(-.7, n - .3)
    ax.set_xlabel("Partial Spearman with Δα (median over tissues)")
    ax.set_title("Regulatory and gene architecture", pad=9)
    h = [plt.Line2D([], [], marker=m, ls="", color=c, ms=4) for _, _, m, c in SERIES]
    ax.legend(h, ["Δα, canonical gate", "Δα, per-gene gate", "Pseudobulk profile corr."], loc="lower left", fontsize=5.6,
              bbox_to_anchor=(0, .36),
              handletextpad=.2, borderaxespad=.1)
    ax.text(0.295, n - .45, "same sign", ha="right", va="bottom", fontsize=5.2, color="#555555")
    save(fig, "e5_properties")


def panel_dose():
    q = pd.read_csv(O / "quintiles.csv")
    fig, axs = plt.subplots(1, 2, figsize=(3.2, 2.0), sharey=True)
    for ax, (p, lab) in zip(axs, (("n_dels_domain", "Distal enhancers in domain"), ("tf_indegree", "Regulating TFs"))):
        d = q[q.property == p].groupby(["tissue", "quintile"])[["D_can", "D_free", "B1", "rt_D_free"]].median().reset_index()
        for y, mk, c, ls, nm in (("D_can", "o", COL["can"], "-", "Δα, canonical"), ("D_free", "^", COL["free"], "-", "Δα, per-gene"),
                                 ("rt_D_free", "^", COL["free"], ":", "per-gene, equal cell-type τ"),
                                 ("B1", "D", COL["b1"], "-", "Pseudobulk profile corr.")):
            s = d.groupby("quintile")[y].agg(med="median", lo=lambda v: np.quantile(v, .25), hi=lambda v: np.quantile(v, .75))
            x = s.index.to_numpy()
            if ls == "-":
                ax.fill_between(x, s.lo, s.hi, color=c, alpha=.12, lw=0)
            ax.plot(x, s.med, color=c, ls=ls, lw=1.1, marker=mk, ms=3.2 if ls == "-" else 2.4, mec="white", mew=.4,
                    label=nm)
        ax.axhline(0, color="#999999", lw=.6)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel(f"{lab}\n(quintile, fewest → most)", fontsize=6.2)
    axs[0].set_ylabel("Residual Δα (SD)")
    axs[0].set_title("Regulatory input and deployment")
    h, lb = axs[0].get_legend_handles_labels()
    axs[0].legend(h, lb, loc="upper left", fontsize=5.0, handletextpad=.3, borderaxespad=.1)
    save(fig, "e5_dose")


def panel_regulons():
    r = pd.read_csv(O / "regulon_indeg.csv")
    tv = r.groupby(["tf", "tissue"])[["D_can", "D_free", "B1"]].median().reset_index()
    tfs = TFS_UP + TFS_DN
    fig, ax = plt.subplots(figsize=(2.3, 3.0))
    n = len(tfs)
    for i, tf in enumerate(tfs):
        y = n - 1 - i
        for ro, off, mk, c in SERIES:
            v = tv.loc[tv.tf == tf, ro].dropna().to_numpy()
            lo, hi = np.quantile(v, [.25, .75])
            ax.plot([lo, hi], [y + off] * 2, color=c, lw=.8, alpha=.6)
            ax.scatter([np.median(v)], [y + off], s=16 if ro != "B1" else 10, marker=mk, color=c, edgecolor="white",
                       linewidth=.4, zorder=3)
        if i == len(TFS_UP) - 1:
            ax.axhline(y - .5, color=GRID, lw=.6)
    ax.axvline(0, color="#999999", lw=.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tfs[::-1], fontsize=6.2, style="italic")
    ax.set_ylim(-.7, n - .3)
    ax.set_xlim(-.65, .75)
    ax.set_xlabel("Residual Δα of targets (SD), at equal TF in-degree")
    ax.set_title("TF regulons")
    ax.text(-.62, n - 1.6, "lineage and\nsignalling TFs", ha="left", fontsize=5.6, color="#555555")
    ax.text(.73, 1.6, "CpG-promoter\nTFs", ha="right", fontsize=5.6, color="#555555")
    save(fig, "e5_regulons")


def main():
    panel_properties()
    panel_dose()
    panel_regulons()


if __name__ == "__main__":
    main()
