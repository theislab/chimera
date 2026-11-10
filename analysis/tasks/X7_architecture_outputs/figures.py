"""X7 (EXPLORATORY): figures. PDF with TrueType fonts (no Type 3), via notebooks/_figure_design.py.

Writes results/analysis/X7/figures/fig_main_architecture_outputs.pdf and fig_supp_residual_share_cells.pdf.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from analysis.common import gene_ids
from analysis.tasks.X7_architecture_outputs.forward import OUT
from analysis.tasks.X7_architecture_outputs.baselines import BD

FIG = OUT / "figures"
# Okabe-Ito hues, fixed order, each with its own marker (secondary encoding)
COL = {"chimera": "#D55E00", "scvi": "#0072B2", "chimera_pur": "#CC79A7", "direct": "#222222", "direct2": "#999999"}
LAB_COH = {"liv_hmouse": "Liver H+Mouse", "liv_4sp": "Liver 4-sp", "test_hmac": "Testis H+Mac",
           "test_hmouse": "Testis H+Mouse", "test_3sp": "Testis 3-sp", "liv_hmonkey": "Liver H+Monkey",
           "liv_hzeb": "Liver H+Zeb"}


def design():
    sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
    import _figure_design as fd
    fd.apply_design()
    import matplotlib as mpl
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6, "axes.titlesize": 6.5,
                         "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5})
    import matplotlib.pyplot as plt
    return plt, fd


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2)


def panel_types(ax):
    r = pd.read_csv(OUT / "types_rho.csv")
    r = r[r.gold == "div"]
    tests = [("test_hmac", True), ("test_3sp", True), ("test_hmac", False), ("test_hmouse", False),
             ("test_3sp", False), ("liv_hmouse", False), ("liv_4sp", False)]
    scores = [("tv", "residual share (Chimera)", COL["chimera"], "o"),
              ("npur_chimera", "species purity, Chimera z", COL["chimera_pur"], "D"),
              ("npur_scvi", "species purity, scVI z", COL["scvi"], "s"),
              ("direct_div", "pseudobulk divergence", COL["direct"], "^")]
    yl = []
    for i, (c, fine) in enumerate(tests):
        d = r[(r.cohort == c) & (r.fine == fine)]
        n = int(d.n_types.iloc[0])
        yl.append(f"{LAB_COH[c]}, {n} {'stages' if fine else 'types'}")
        for j, (s, lab, col, mk) in enumerate(scores):
            e = d[d.score == s]
            if e.empty:
                continue
            y = i + (j - 1.5) * 0.17
            if len(e) > 1:
                ax.hlines(y, e.rho.min(), e.rho.max(), color=col, linewidth=0.8)
            ax.scatter(e[e.seed == 42].rho, [y], color=col, marker=mk, s=12, zorder=3, edgecolor="white",
                       linewidth=0.4, label=lab if i == 0 else None)
    ax.axvline(0, color="#bbbbbb", linewidth=0.5)
    ax.set_yticks(range(len(tests)))
    ax.set_yticklabels(yl)
    ax.invert_yaxis()
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Spearman across types with independent divergence")
    ax.legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2,
              handletextpad=0.2, columnspacing=0.8)
    ax.set_title("a  Type level: which states diverge", loc="left")
    clean(ax)


def panel_cellvar(ax):
    d = pd.read_csv(OUT / "posthoc_cellvar.csv")
    d = d[d.seed == 42].set_index("cohort").loc[list(LAB_COH)]
    x = np.arange(len(d))
    w = 0.38
    ax.bar(x - w / 2, d.tv_eta2_species, w, color=COL["chimera"], label="species")
    ax.bar(x + w / 2, d.tv_eta2_type, w, color=COL["direct2"], label="cell type")
    ax.set_xticks(x)
    ax.set_xticklabels([LAB_COH[c] for c in d.index], rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel("share of per-cell variance\nof residual share")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    ax.set_title("b  Residual share is mostly species", loc="left")
    clean(ax)


def panel_genetype(ax):
    r = pd.read_csv(OUT / "genetype_rho.csv")
    c42 = r[r.seed == 42]
    order = [(c, t) for c in ("liv_hmouse", "liv_4sp", "test_hmouse", "test_3sp")
             for t in c42[c42.cohort == c].type.unique()]
    for j, (s, col, mk, lab) in enumerate((("R", COL["chimera"], "o", "residual, per gene and type"),
                                           ("Ddir", COL["direct"], "^", "pseudobulk level gap"))):
        for i, (c, t) in enumerate(order):
            e = c42[(c42.cohort == c) & (c42.type == t) & (c42.score == s)].iloc[0]
            x = i + (j - 0.5) * 0.3
            ax.vlines(x, e.ci_lo, e.ci_hi, color=col, linewidth=0.7)
            ax.scatter([x], [e.rho_p], color=col, marker=mk, s=10, zorder=3, edgecolor="white", linewidth=0.4,
                       label=lab if i == 0 else None)
            sd = r[(r.cohort == c) & (r.type == t) & (r.score == s) & (r.seed != 42)]
            ax.scatter(np.full(len(sd), x + 0.08), sd.rho_p, color=col, marker=".", s=3, alpha=0.6, linewidths=0)
    ax.axhline(0, color="#bbbbbb", linewidth=0.5)
    short = {"Hepatocytes": "Hep", "Endothelial": "Endo", "fibroblasts": "Fib", "macrophage": "Mac",
             "Other_somatic": "Som", "Sertoli": "Ser"}
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short.get(t, t) for _, t in order], rotation=90)
    for k, (lo, hi, lab) in enumerate(((0, 3, "Liver\nH+Mouse"), (4, 7, "Liver\n4-sp"), (8, 13, "Testis H+Mouse"),
                                       (14, 19, "Testis 3-sp"))):
        if k:
            ax.axvline(lo - 0.5, color="#dddddd", linewidth=0.5)
        ax.text((lo + hi) / 2, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5)
    ax.set_ylabel("partial rank correlation with\nindependent gene divergence")
    ax.set_ylim(-0.08, 0.2)
    ax.legend(frameon=False, fontsize=5, loc="lower left", ncol=2)
    ax.set_title("c  Gene x type: same counts do as well", loc="left", pad=20)
    clean(ax)


def panel_species(ax):
    r = pd.read_csv(OUT / "species_rho.csv")
    r = r[(r.universe == "unmasked_counts_all")]
    items = [("test_3sp", "contrasts", "Bt", "Chimera species margins", COL["chimera"], "o"),
             ("test_3sp", "contrasts", "dir_contrast", "direct margin, no model", COL["direct2"], "v"),
             ("test_3sp", "contrasts", "E_contrast", "pairwise correlation contrast", COL["direct"], "^"),
             ("liv_4sp", "contrasts", "Bl", None, COL["chimera"], "o"),
             ("liv_4sp", "contrasts", "dir_contrast", None, COL["direct2"], "v"),
             ("liv_4sp", "contrasts", "E_contrast", None, COL["direct"], "^"),
             ("liv_4sp", "contrasts", "E_HMouse", "human-mouse correlation alone", COL["scvi"], "s"),
             ("liv_4sp", "species", "S_Mouse", "mouse margin alone", COL["chimera_pur"], "D")]
    xs = [0, 0.35, 0.7, 1.6, 1.95, 2.3, 2.65, 3.0]
    for x, (c, st, s, lab, col, mk) in zip(xs, items):
        e = r[(r.cohort == c) & (r.set == st) & (r.score == s)]
        e0 = e[e.seed == 42].iloc[0]
        ax.vlines(x, e0.ci_lo, e0.ci_hi, color=col, linewidth=0.7)
        ax.scatter([x], [e0.rho_p], color=col, marker=mk, s=12, zorder=3, edgecolor="white", linewidth=0.4, label=lab)
        sd = e[e.seed != 42]
        ax.scatter(np.full(len(sd), x + 0.1), sd.rho_p, color=col, marker=".", s=3, alpha=0.6, linewidths=0)
    ax.axhline(0, color="#bbbbbb", linewidth=0.5)
    ax.set_xticks([0.35, 2.3])
    ax.set_xticklabels(["Testis 3-sp\nprimates vs mouse\n(Murat PP minus PO)", "Liver 4-sp\nmouse vs monkey\n(TS vs TMS)"], fontsize=5)
    ax.set_ylabel("partial rank correlation with gold")
    ax.legend(frameon=False, fontsize=5, loc="upper left", ncol=1, handletextpad=0.2)
    ax.set_ylim(-0.02, 0.36)
    ax.set_title("d  Species-resolved margins in one joint fit", loc="left")
    clean(ax)


def fig_main():
    plt, fd = design()
    fig = plt.figure(figsize=(6.8, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], hspace=0.75, wspace=0.55)
    panel_types(fig.add_subplot(gs[0, 0]))
    panel_cellvar(fig.add_subplot(gs[0, 1]))
    panel_genetype(fig.add_subplot(gs[1, 0]))
    panel_species(fig.add_subplot(gs[1, 1]))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_main_architecture_outputs.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_main_architecture_outputs.png", dpi=200, bbox_inches="tight")


def fig_supp():
    plt, fd = design()
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2))
    for ax, c in zip(axes, ("test_3sp", "liv_hmouse", "liv_4sp")):
        t = pd.read_parquet(BD / f"{c}__cells.parquet")
        types = [k for k in t.coarse.value_counts().index if (t.coarse == k).sum() >= 100][:6]
        sps = sorted(t.species.unique(), key=fd.species_sort_key)
        pos = 0
        ticks = []
        for k in types:
            for s in sps:
                v = t[(t.coarse == k) & (t.species == s)].tv
                if len(v) >= 30:
                    q = np.percentile(v, [10, 25, 50, 75, 90])
                    ax.vlines(pos, q[0], q[4], color=fd.species_color(s), linewidth=0.6)
                    ax.vlines(pos, q[1], q[3], color=fd.species_color(s), linewidth=2.2)
                    ax.scatter([pos], [q[2]], color="white", s=3, zorder=3)
                pos += 1
            ticks.append((pos - len(sps) / 2 - 0.5, k))
            pos += 0.8
        ax.set_xticks([p for p, _ in ticks])
        ax.set_xticklabels([k[:10] for _, k in ticks], rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title(LAB_COH[c], loc="left")
        clean(ax)
        for s in sps:
            ax.plot([], [], color=fd.species_color(s), linewidth=2, label=s)
        ax.legend(frameon=False, fontsize=5, loc="upper right")
    axes[0].set_ylabel("per-cell residual share (TV)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_supp_residual_share_cells.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_supp_residual_share_cells.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    fig_main()
    fig_supp()
