"""E2 candidate panels (exploratory). Usage: python -m revision.tasks.E2_protein.figures

Panels in results/revision/E2/panels/; composed with F1_figures/compose.py --spec revision/tasks/E2_protein/specs_e2.py
(CHIMERA_MANUSCRIPT pointed at results/revision/E2/ms_stage so nothing is written into the manuscript).
Palette: Okabe-Ito hues with a distinct marker per series (validator needs node, not installed here; as Y5, X4, X7).
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
O = gene_ids.REPO_ROOT / "results" / "revision" / "E2"
P = O / "panels"
COL = dict(free="#D55E00", can="#0072B2", b1="#7A7A7A", ohnolog="#D55E00", ssd="#0072B2", single="#7A7A7A")
GRID = "#E6E6E6"

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
        for ro, off, mk, c in (("D_can", .22, "o", COL["can"]), ("D_free", 0, "^", COL["free"]), ("B1", -.22, "D", COL["b1"])):
            v = tv[(tv.property == k) & (tv.readout == ro)].rho_p.to_numpy()
            ax.scatter(v, np.full(len(v), y + off), s=4, color=c, alpha=.35, lw=0)
            ax.scatter([np.median(v)], [y + off], s=18 if ro != "B1" else 12, marker=mk, color=c, edgecolor="white",
                       linewidth=.4, zorder=3)
        same = int(sm.loc[k, "D_free_same_sign"])
        ax.text(0.335, y, f"{same}/10", va="center", ha="right", fontsize=5.6, color="#555555")
        if i in (9, 11):
            ax.axhline(y - .5, color=GRID, lw=.6)
    ax.axvline(0, color="#999999", lw=.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in PROPS][::-1], fontsize=6.2)
    ax.set_xlim(-.2, .34)
    ax.set_ylim(-.7, n - .3)
    ax.set_xlabel("Partial Spearman with Δα (median over tissues)")
    ax.set_title("Protein and gene properties", pad=9)
    h = [plt.Line2D([], [], marker=m, ls="", color=c, ms=4) for m, c in (("o", COL["can"]), ("^", COL["free"]), ("D", COL["b1"]))]
    ax.legend(h, ["Δα, canonical gate", "Δα, per-gene gate", "Pseudobulk profile corr."], loc="upper left", fontsize=5.6,
              handletextpad=.2, borderaxespad=.1)
    ax.text(0.335, n - .45, "same sign", ha="right", va="bottom", fontsize=5.2, color="#555555")
    save(fig, "e2_properties")


def panel_families():
    tv = pd.read_csv(O / "family_tissue_values.csv")
    fig, ax = plt.subplots(figsize=(2.6, 3.9))
    n = len(FAMS)
    for i, (k, lab) in enumerate(FAMS):
        y = n - 1 - i
        for ro, off, mk, c in (("D_can", .2, "o", COL["can"]), ("D_free", 0, "^", COL["free"]), ("B1", -.2, "D", COL["b1"])):
            v = tv[(tv.family == k) & (tv.readout == ro)].mean_resid.dropna().to_numpy()
            if not len(v):
                continue
            lo, hi = np.quantile(v, [.25, .75])
            ax.plot([lo, hi], [y + off] * 2, color=c, lw=.8, alpha=.6)
            ax.scatter([np.median(v)], [y + off], s=16 if ro != "B1" else 10, marker=mk, color=c, edgecolor="white",
                       linewidth=.4, zorder=3)
        if i == 11:
            ax.axhline(y - .5, color=GRID, lw=.6)
    ax.axvline(0, color="#999999", lw=.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in FAMS][::-1], fontsize=6.2)
    ax.set_ylim(-.7, n - .3)
    ax.set_xlabel("Residual Δα of family members (SD)")
    ax.set_title("Domain architecture")
    ax.text(.98, 11.25, "signalling,\ncell surface", transform=ax.transData, ha="right", fontsize=5.6, color="#555555")
    ax.text(.98, 1.0, "RNA, chromatin,\nproteostasis", transform=ax.transData, ha="right", fontsize=5.6, color="#555555")
    ax.set_xlim(-.6, 1.0)
    save(fig, "e2_domains")


def panel_age_dup():
    G = pd.read_parquet(O / "gene_resid.parquet", columns=["cohort", "tissue", "hsapiens_gene_id", "r_D_free"])
    Pp = pd.read_parquet(O / "props.parquet", columns=["hsapiens_gene_id", "gene_age", "ohnolog", "paralog_count"])
    G = G.merge(Pp, on="hsapiens_gene_id")
    G["dup"] = np.where(G.ohnolog > 0, "ohnolog", np.where(G.paralog_count > 0, "ssd", "single"))
    bins = [(-1, 4, "Tetra-\npods+"), (4, 5, "Bony\nvert."), (5, 6, "Jawed\nvert."), (6, 7, "Verte-\nbrates"),
            (7, 8, "Chor-\ndates"), (8, 9, "Bila-\nterians"), (9, 10, "Opistho-\nkonts")]
    G["b"] = pd.cut(G.gene_age, [b[0] for b in bins] + [10], labels=False)
    fig, ax = plt.subplots(figsize=(2.9, 2.2))
    for d, lab, mk in (("ohnolog", "2R ohnologs", "^"), ("ssd", "Small-scale duplicates", "o"), ("single", "Singletons", "s")):
        a = G[G.dup == d].groupby(["tissue", "cohort", "b"]).r_D_free.agg(["mean", "size"]).reset_index()
        a = a[a["size"] >= 10].groupby(["tissue", "b"])["mean"].median().reset_index()
        s = a.groupby("b")["mean"].agg(med="median", lo=lambda v: np.quantile(v, .25), hi=lambda v: np.quantile(v, .75),
                                       nt="size")
        s = s[s.nt >= 7]
        x = s.index.to_numpy()
        ax.fill_between(x, s.lo, s.hi, color=COL[d], alpha=.15, lw=0)
        ax.plot(x, s.med, color=COL[d], lw=1.2, marker=mk, ms=3.5, mec="white", mew=.4, label=lab)
    ax.axhline(0, color="#999999", lw=.6)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([b[2] for b in bins], fontsize=5.6)
    ax.set_xlabel("Origin of the gene (phylostratum, older →)")
    ax.set_ylabel("Residual Δα, per-gene gate (SD)")
    ax.set_title("Duplication history and gene age")
    ax.legend(loc="lower left", fontsize=5.6, handletextpad=.3, borderaxespad=.1)
    save(fig, "e2_age_dup")


def main():
    panel_properties()
    panel_families()
    panel_age_dup()


if __name__ == "__main__":
    main()
