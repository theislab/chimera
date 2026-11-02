"""Y1 compact figure. Static, for the ledger and a possible supplementary panel. Palette: dataviz reference slots (validator not run, no node here).

    python -I launch.py revision.tasks.Y1_gene_tissue.figure
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revision.tasks.Y1_gene_tissue import common as Y
from revision.tasks.Y1_gene_tissue import rules as RU

sys.path.insert(0, str(Y.ROOT / "notebooks"))
import _figure_design as fd  # noqa: E402

OUT = RU.OUT
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
CS, CB, CD, CDIR = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"  # S, B1, D, direct margin; each with its own marker
MK = {"S": "o", "B1": "s", "D": "^", "Sdir": "D"}
COL = {"S": CS, "B1": CB, "D": CD, "Sdir": CDIR}
NAME = {"S": r"$S_g$", "B1": "B1", "D": r"$\Delta\alpha$", "Sdir": "direct margin"}
TIS = ["liver", "testis", "kidney", "islet", "bone_marrow", "eye"]
TL = {"liver": "liver", "testis": "testis", "kidney": "kidney", "islet": "islet", "bone_marrow": "marrow", "eye": "eye"}


def lettered(ax, s):
    ax.text(-0.02, 1.05, s, transform=ax.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom", color=INK)


def panel_a(ax):
    v = pd.read_csv(OUT / "variance_split.csv")
    v = v[v.design == "ED"].set_index("score").loc[["S", "D", "B1"]]
    tones = {"gene": "#2a78d6", "gxt": "#9ec5f0", "noise": "#d8d7d2"}
    for i, (sc, r) in enumerate(v.iterrows()):
        left = 0
        for k, c, w in (("gene_share", tones["gene"], r.gene_share), ("gxt_share", tones["gxt"], r.gxt_share), ("noise_share", tones["noise"], r.noise_share)):
            ax.barh(i, w - 0.004, left=left, color=c, height=0.62)
            left += w
        ax.text(r.gene_share + 0.012, i, f"{r.gene_share:.2f}", va="center", ha="left", fontsize=6.5, color=INK)
    ax.set_yticks(range(3), [NAME[s] for s in v.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of variance of z, given expression and detection per tissue")
    h = [plt.Rectangle((0, 0), 1, 1, color=tones[k]) for k in tones]
    ax.legend(h, ["gene (shared across tissues)", "gene x tissue", "technical noise"], frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, handlelength=1, columnspacing=0.8)


def panel_b(axs):
    r = pd.read_csv(OUT / "contrast_rho.csv")
    golds = ["liver", "testis", "kidney", "pancreas"]
    match = {"liver": "liver", "testis": "testis", "kidney": "kidney", "pancreas": "islet"}
    vm = 0.24
    for ax, sc in zip(axs, ("S", "B1")):
        M = r[r.score == sc].pivot(index="source", columns="gold", values="est").reindex(TIS)[golds]
        im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
        ax.set_xticks(range(4), golds, rotation=30, ha="right")
        ax.set_yticks(range(6), [TL[t] for t in TIS] if sc == "S" else [""] * 6)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for i, t in enumerate(TIS):
            for j, g in enumerate(golds):
                v = M.iloc[i, j]
                ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=5.8, color="white" if abs(v) > 0.15 else INK)
                if match[g] == t:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=1.1))
        ax.set_xlabel(f"gold; readout {NAME[sc]}", fontsize=7)
    axs[0].set_ylabel("tissue of the readout")


def forest(ax, rows, labels, scores, xlabel, ref=0):
    n = len(scores)
    off = np.linspace(-0.26, 0.26, n) if n > 1 else [0]
    for j, sc in enumerate(scores):
        d = rows[rows.score == sc].set_index("key").reindex(labels)
        y = np.arange(len(labels)) + off[j]
        ax.hlines(y, d.lo, d.hi, color=COL[sc], lw=1.2)
        ax.plot(d.est, y, MK[sc], color=COL[sc], ms=4, mec="white", mew=0.5, label=NAME[sc], ls="")
    ax.axvline(ref, color=GRID, lw=0.8, zorder=0)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)


def panel_c(ax):
    d = pd.read_csv(OUT / "contrast_delta.csv")
    d = d[d.score.isin(["S", "B1", "D"])].assign(key=lambda x: x.gold)
    forest(ax, d, ["liver", "testis", "kidney", "pancreas"], ["S", "B1", "D"], "$\\Delta_T$: same tissue minus other tissues, $\\rho_p$")
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.1, columnspacing=1.0)


def panel_d(ax):
    w = pd.read_csv(OUT / "within_gene.csv")
    a = pd.read_csv(OUT / "posthoc" / "A_within_matched.csv")
    lab = {"liver|testis": "liver, testis", "liver|kidney": "liver, kidney", "liver|pancreas": "liver, pancreas", "testis|kidney": "testis, kidney", "testis|pancreas": "testis, pancreas", "kidney|pancreas": "kidney, pancreas"}
    w["key"], a["key"] = w.pair.map(lab), a.pair.map(lab)
    ls = list(lab.values())
    for sc, of in (("S", -0.2), ("B1", 0.2)):
        d = a[a.score == sc].set_index("key").reindex(ls)
        y = np.arange(6) + of
        ax.hlines(y, d.lo, d.hi, color=COL[sc], lw=1.2)
        ax.plot(d.est, y, MK[sc], color=COL[sc], ms=4, mec="white", mew=0.5, ls="", label=f"{NAME[sc]}, matched variability")
        e = w[w.score == sc].set_index("key").reindex(ls)
        ax.plot(e.est, y, MK[sc], mfc="white", mec=COL[sc], ms=3.5, mew=0.7, ls="", label=f"{NAME[sc]}, $C$ only")
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_yticks(range(6), ls)
    ax.invert_yaxis()
    ax.set_xlabel("tissue difference in readout vs in gold, within gene, $\\rho$")
    ax.legend(frameon=False, fontsize=5.6, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=2, handletextpad=0.1, columnspacing=0.8, labelspacing=0.15)


def panel_e(ax):
    d = pd.read_parquet(OUT / "posthoc" / "D_flip_cells.parquet")
    cats = ["level only", "both", "pattern only", "neither"]
    cols = {"level only": "#2a78d6", "both": "#8a7fd8", "pattern only": "#eb6834", "neither": "#d8d7d2"}
    tb = pd.crosstab(d.t_lo, d.cat)[cats].reindex(TIS)
    sh = tb.div(tb.sum(axis=1), axis=0)
    for i, t in enumerate(TIS):
        left = 0
        for c in cats:
            ax.barh(i, sh.at[t, c] - 0.006, left=left, color=cols[c], height=0.66)
            left += sh.at[t, c]
    ax.set_yticks(range(6), [f"{TL[t]} ({int(tb.loc[t].sum())})" for t in TIS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("flip genes by what is low in their low tissue")
    h = [plt.Rectangle((0, 0), 1, 1, color=cols[c]) for c in cats]
    ax.legend(h, ["level gap", "both", "pattern lost", "neither"], frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, handlelength=1, columnspacing=0.8, handletextpad=0.3)


def panel_f(ax):
    r = pd.read_csv(OUT / "worked_readouts.csv")
    genes = ["RUNX1", "GCK", "SLC2A2", "PLEKHG1", "MERTK", "B2M"]
    r = r[r.gene.isin(genes)]
    y = {g: i for i, g in enumerate(genes)}
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    for g in genes:
        d = r[(r.gene == g) & r.support]
        ax.hlines(y[g], d.z_S.min(), d.z_S.max(), color=GRID, lw=2, zorder=1)
        for _, a in d.iterrows():
            ax.plot(a.z_S, y[g], "o", color=CS, ms=4.5, mec="white", mew=0.5, zorder=3)
            ax.plot(a.z_B1, y[g] + 0.24, "s", color=CB, ms=3, mec="white", mew=0.3, zorder=2)
    # label the extreme tissues only
    for g in genes:
        d = r[(r.gene == g) & r.support]
        lo, hi = d.loc[d.z_S.idxmin()], d.loc[d.z_S.idxmax()]
        ax.annotate(TL[lo.tissue], (lo.z_S, y[g]), xytext=(0, -7), textcoords="offset points", ha="center", va="top", fontsize=5.6, color=MUTED)
        ax.annotate(TL[hi.tissue], (hi.z_S, y[g]), xytext=(0, -7), textcoords="offset points", ha="center", va="top", fontsize=5.6, color=MUTED)
    ax.set_yticks(range(len(genes)), genes)
    ax.set_ylim(len(genes) - 0.15, -0.6)
    ax.set_xlabel("z within tissue, count-supported tissues")
    h = [plt.Line2D([], [], marker="o", color=CS, ls="", ms=4), plt.Line2D([], [], marker="s", color=CB, ls="", ms=3)]
    ax.legend(h, [NAME["S"], "B1"], frameon=False, fontsize=6, loc="lower left", handletextpad=0.1, ncol=2, bbox_to_anchor=(0, 1.0), columnspacing=0.8)


def main():
    fd.apply_design()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.62, 1, 1], hspace=0.72, wspace=0.65, left=0.1, right=0.985, top=0.945, bottom=0.075)
    a = fig.add_subplot(gs[0, :2])
    b1, b2 = fig.add_subplot(gs[0:1, 2]), fig.add_subplot(gs[0:1, 3])
    c = fig.add_subplot(gs[1, :2])
    d = fig.add_subplot(gs[1, 2:])
    e = fig.add_subplot(gs[2, :2])
    f = fig.add_subplot(gs[2, 2:])
    panel_a(a)
    panel_b([b1, b2])
    panel_c(c)
    panel_d(d)
    panel_e(e)
    panel_f(f)
    for ax, s in zip((a, b1, c, d, e, f), "abcdef"):
        lettered(ax, s)
    d_ = OUT / "figures"
    d_.mkdir(exist_ok=True)
    fig.savefig(d_ / "fig_y1_gene_tissue.pdf")
    fig.savefig(d_ / "fig_y1_gene_tissue.png", dpi=220)


if __name__ == "__main__":
    main()
