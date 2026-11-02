"""Y5 compact figure. Usage: python -m revision.tasks.Y5_aspects.figures"""
from __future__ import annotations
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from revision.tasks.Y5_aspects import common as K

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
                     "legend.frameon": False, "legend.fontsize": 6.2, "axes.titlesize": 7.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left"})
O = K.OUT
OI = dict(chim="#0072B2", tool="#7A7A7A", direct="#D55E00", s="#009E73", y="#E69F00", p="#CC79A7", sky="#56B4E9")
GRID = "#E6E6E6"


def panel_latents(ax, c, title):
    t = pd.read_csv(O / "F3/b1z_rho.csv")
    t = t[t.cohort == c]
    a = pd.read_csv(O / "F3/anchor_rho.csv")
    a = a[a.cohort == c]
    tools = ["scvi", "scanvi", "sysvi", "harmony", "pca", "saturn", "camex"]
    lab = dict(chimera="Chimera", scvi="scVI", scanvi="scANVI", sysvi="sysVI", harmony="Harmony", pca="PCA", saturn="SATURN", camex="CAMEX")
    order = ["chimera"] + tools
    for i, n in enumerate(order):
        col = OI["chim"] if n == "chimera" else OI["tool"]
        def vals(tab, pre):
            if n == "chimera":
                q = tab[tab.y.str.startswith(pre + "chimera_s")]
            else:
                q = tab[tab.y.isin([pre + n, pre + n + "_s42"])]
            return q.rho_p.to_numpy()
        v1, v2 = vals(t, "B1z_"), vals(a, "")
        for v, off, m in ((v1, -.17, "o"), (v2, .17, "^")):
            if len(v):
                ax.plot([i + off] * 2, [v.min(), v.max()], color=col, lw=1)
                ax.scatter([i + off], [v.mean()], s=20, marker=m, color=col, edgecolor="white", linewidth=.5, zorder=3)
            else:
                ax.text(i + off, .004, "×", ha="center", va="bottom", color=col, fontsize=7)
    b = a[a.y == "B1b"].rho_p.mean()
    ax.axhline(b, color=OI["direct"], lw=1, ls="--")
    ax.text(len(order) - .55, b, "labels", color="#333333", fontsize=6, va="bottom", ha="right")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lab[o] for o in order], rotation=45, ha="right")
    ax.set_ylim(-.02, .36)
    if title.startswith("a"):
        ax.scatter([], [], marker="o", s=16, color="#555555", label="Leiden clusters")
        ax.scatter([], [], marker="^", s=16, color="#555555", label="centroid nearest cells")
        ax.legend(loc="lower left", bbox_to_anchor=(0, .1), handletextpad=.1, ncol=1)
    ax.set_ylabel("partial ρ with held-out pattern")
    ax.yaxis.grid(True, color=GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.set_title(title)


def panel_dose(ax):
    d = pd.read_csv(O / "F3/b1z_dose.csv")
    d = d.groupby(["cohort", "latent"]).agg(c=("cells_in_shared", "first"), r=("rho_p", "mean")).reset_index()
    for coh, col, m, lb in (("liv_hmouse", OI["chim"], "o", "Liver H+Mouse"), ("test_hmouse", OI["y"], "s", "Testis H+Mouse")):
        q = d[d.cohort == coh]
        ch = q.latent.str.startswith("chimera")
        ax.scatter(q.c[~ch], q.r[~ch], s=16, marker=m, facecolor="white", edgecolor=col, linewidth=.9, zorder=3)
        ax.scatter(q.c[ch], q.r[ch], s=16, marker=m, color=col, edgecolor="white", linewidth=.3, zorder=3, label=lb)
    ax.set_xlabel("share of cells in cross-species clusters")
    ax.set_ylabel("partial ρ, cluster profile correlation")
    ax.yaxis.grid(True, color=GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.02, 0), handletextpad=.1, title="filled: Chimera seeds", title_fontsize=6)
    ax.set_title("c  Clusters shared across species")


def panel_props(ax):
    d = pd.read_csv(O / "f1d_controls.csv")
    s = pd.read_csv(O / "f1_scan.csv")
    props = ["log_length", "signal_peptide", "loc_secreted", "paralog_count", "domain_dens", "loc_mitochondrion"]
    names = dict(log_length="protein length", signal_peptide="signal peptide", loc_secreted="secreted", paralog_count="paralog count", domain_dens="domain density", loc_mitochondrion="mitochondrial")
    q = d[(d.cohort == "liv_hmouse") & (d.arm == "m")]
    m = lambda y: q[q.y == y].groupby("property").rho_p.mean()
    sdir = s[(s.cohort == "liv_hmouse") & (s["cov"] == "C4") & (s.y == "Sdir")].groupby("property").rho_p.mean()
    series = [("Δα", m("D_raw"), OI["chim"], "o"), ("Δα, no features (Config M)", m("D_m"), OI["sky"], "D"),
              ("S_g", m("S"), OI["s"], "s"), ("direct margin", sdir, OI["direct"], "^")]
    for j, (lb, v, col, mk) in enumerate(series):
        y = np.arange(len(props)) + (j - 1.5) * .17
        ax.scatter([v[p] for p in props], y, s=14, marker=mk, color=col, edgecolor="white", linewidth=.3, label=lb, zorder=3)
    ax.axvline(0, color="#888888", lw=.6)
    ax.set_yticks(range(len(props)))
    ax.set_yticklabels([names[p] for p in props])
    ax.invert_yaxis()
    ax.set_xlabel("partial ρ with property, liver H+Mouse")
    ax.xaxis.grid(True, color=GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.45, -.2), ncol=2, handletextpad=.1, columnspacing=.6)
    ax.set_title("d  Protein properties, liver")


def panel_depth(ax):
    t = pd.read_csv(O / "F2/depth_rho.csv")
    t = t[t.cohort == "test_3sp"]
    x = dict(G_lt10=0, G_10_45=1, G_160_180=2, G_318=3)
    series = [("B1z", "cluster profile correlation", OI["chim"], "o"), ("B1b", "labelled profile correlation", OI["direct"], "^"), ("S", "S_g", OI["s"], "s")]
    for sc, lb, col, mk in series:
        q = t[t.score == sc].groupby("gold").agg(r=("rho_p", "mean"), lo=("lo", "min"), hi=("hi", "max"))
        xs = [x[g] for g in x]
        ys = [q.r[g] for g in x]
        ax.plot(xs, ys, color=col, lw=1.1, marker=mk, ms=3.6, mec="white", mew=.3, label=lb)
        ax.vlines(xs, [q.lo[g] for g in x], [q.hi[g] for g in x], color=col, lw=.6)
    c = pd.read_csv(O / "F2/depth_conditional.csv")
    c = c[(c.cohort == "test_3sp") & (c.with_identity == "identity_axis") & (c.y == "identity_chicken")]
    v = [c[c.deep == g].rho_p.mean() for g in x]
    ax.plot([x[g] for g in x], v, color=OI["p"], lw=1.1, marker="D", ms=3, mec="white", mew=.3, label="protein identity to chicken")
    ax.axhline(0, color="#888888", lw=.6)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["<10", "10-45", "160-180", "318"])
    ax.set_xlabel("million years between held-out species")
    ax.set_ylabel("partial ρ with profile agreement")
    ax.yaxis.grid(True, color=GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, handlelength=1.4, columnspacing=.8)
    ax.set_title("e  Phylogenetic depth, testis")


def panel_x(ax):
    t = pd.read_csv(O / "F5/xlinked_depth.csv")
    sp = ["chimpanzee", "bonobo", "gorilla", "gibbon", "marmoset", "opossum", "platypus", "chicken"]
    for r, off, col, mk, lb in (("XCR", -.15, OI["chim"], "o", "X conserved region"), ("XAR", .15, OI["y"], "s", "X added region")):
        q = t[t.region == r].set_index("species").loc[sp]
        xs = np.arange(len(sp)) + off
        ax.vlines(xs, q.lo, q.hi, color=col, lw=.8)
        ax.scatter(xs, q.rank_shift, s=16, marker=mk, color=col, edgecolor="white", linewidth=.3, label=lb, zorder=3)
    ax.axhline(0, color="#888888", lw=.6)
    ax.axvspan(4.5, 5.5, color="#F2F2F2", zorder=0)
    ax.set_xticks(range(len(sp)))
    ax.set_xticklabels(sp, rotation=45, ha="right")
    ax.set_ylabel("rank shift, spermatocyte minus\nspermatogonia, vs autosomes")
    ax.legend(loc="upper left", handletextpad=.1)
    ax.yaxis.grid(True, color=GRID, lw=.5)
    ax.set_axisbelow(True)
    ax.set_title("f  Human X genes in meiosis")


def main():
    fig = plt.figure(figsize=(7.4, 5.2))
    gs = fig.add_gridspec(2, 3, wspace=.42, hspace=.62, left=.08, right=.99, top=.95, bottom=.16)
    panel_latents(fig.add_subplot(gs[0, 0]), "test_3sp", "a  Testis three species")
    panel_latents(fig.add_subplot(gs[0, 1]), "liv_hmouse", "b  Liver human + mouse")
    panel_dose(fig.add_subplot(gs[0, 2]))
    panel_props(fig.add_subplot(gs[1, 0]))
    panel_depth(fig.add_subplot(gs[1, 1]))
    panel_x(fig.add_subplot(gs[1, 2]))
    d = O / "figures"
    d.mkdir(exist_ok=True)
    fig.savefig(d / "fig_y5_aspects.pdf", bbox_inches="tight")
    fig.savefig(d / "fig_y5_aspects.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
