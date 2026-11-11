"""Gene by tissue: six panels (variance split, tissue contrast as heat maps and as differences, within-gene tissue
differences, flip genes, worked genes), drawn from the result tables of this task.

The panels are saved as three pages of one size, figures/panels/main_tissues/gene_by_tissue_row{1,2,3}.pdf, each with one
row of panels drawn. The figure spec cuts the six panels out of these pages at fixed positions, so the page and the
place of every axes on it are fixed here (PAGE_W, PAGE_H, GRID) and do not depend on any label.

Inputs: ``results/analysis/Y1/`` (``variance_split.csv``, ``contrast_rho.csv``, ``contrast_delta.csv``,
``within_gene.csv``, ``worked_readouts.csv``, ``posthoc/A_within_matched.csv``, ``posthoc/D_flip_cells.parquet``),
bundle ``analysis_results``.

    python -I analysis/tasks/Y1_gene_tissue/launch.py analysis.tasks.Y1_gene_tissue.figure
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.tasks.Y1_gene_tissue import common as Y
from analysis.tasks.Y1_gene_tissue import rules as RU

sys.path.insert(0, str(Y.ROOT / "notebooks"))
import _figure_design as fd  # noqa: E402

from figures import style  # noqa: E402

OUT = RU.OUT

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this figure
INK, MUTED, RULE = style.INK, style.MUTED, style.RULE
READOUT = {"S": style.READOUT["count_evidence"], "B1": style.READOUT["profile_correlation"], "D": style.READOUT["delta_alpha"],
           "Sdir": style.READOUT["direct_margin"]}  # one color and one marker per readout
MK = {k: m.marker for k, m in READOUT.items()}
COL = {k: m.color for k, m in READOUT.items()}
NAME = {"S": r"$S_g$", "B1": "profile correlation", "D": r"$\Delta\alpha$", "Sdir": "direct margin"}  # reader names
BLUE, BLUE_LIGHT, ORANGE, PURPLE, PALE = "#0072B2", "#8CC0DC", "#E69F00", "#8A7FD8", "#D9D9D9"  # shares of a bar (panels a and e)
REFERENCE = "independent reference"  # the independent data a readout is compared with
TIS = ["liver", "testis", "kidney", "islet", "bone_marrow", "eye"]
TL = {"liver": "Liver", "testis": "Testis", "kidney": "Kidney", "islet": "Islet", "bone_marrow": "Bone marrow", "eye": "Eye"}
# Display name of each reference: the "pancreas" reference is the islet reference (keys stay as in the result files).
GL = {"liver": "Liver", "testis": "Testis", "kidney": "Kidney", "pancreas": "Islet"}

# Geometry in bp. A row page is PAGE_W x PAGE_H; GRID gives the outer edges of the grid of axes on it, from the lower
# left corner of the page.
PAGE_W, PAGE_H = 511.914, 429.135
GRID = dict(left=51.690, right=510.474, bottom=24.940, top=413.308)


def panel_a(ax):
    v = pd.read_csv(OUT / "variance_split.csv")
    v = v[v.design == "ED"].set_index("score").loc[["S", "D", "B1"]]
    tones = {"gene": BLUE, "gxt": BLUE_LIGHT, "noise": PALE}
    for i, (sc, r) in enumerate(v.iterrows()):
        left = 0
        for k, c, w in (("gene_share", tones["gene"], r.gene_share), ("gxt_share", tones["gxt"], r.gxt_share), ("noise_share", tones["noise"], r.noise_share)):
            ax.barh(i, w - 0.004, left=left, color=c, height=0.62)
            left += w
        ax.text(r.gene_share + 0.012, i, f"{r.gene_share:.2f}", va="center", ha="left", fontsize=6.5, color=INK)
    ax.set_yticks(range(3), [NAME[s].replace(" ", "\n") if s == "B1" else NAME[s] for s in v.index])  # two lines: the label stays on the page
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of variance of z, given expression and detection per tissue")
    h = [plt.Rectangle((0, 0), 1, 1, color=tones[k]) for k in tones]
    ax.legend(h, ["gene (shared across tissues)", "gene by tissue", "technical noise"], frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, handlelength=1, columnspacing=0.8)


def panel_b(axs):
    r = pd.read_csv(OUT / "contrast_rho.csv")
    golds = ["liver", "testis", "kidney", "pancreas"]
    match = {"liver": "liver", "testis": "testis", "kidney": "kidney", "pancreas": "islet"}
    vm = 0.24
    for ax, sc in zip(axs, ("S", "B1")):
        M = r[r.score == sc].pivot(index="source", columns="gold", values="est").reindex(TIS)[golds]
        im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
        ax.set_xticks(range(4), [GL[g] for g in golds], rotation=30, ha="right")
        ax.set_yticks(range(6), [TL[t] for t in TIS] if sc == "S" else [""] * 6)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for i, t in enumerate(TIS):
            for j, g in enumerate(golds):
                v = M.iloc[i, j]
                shown = ".00" if round(v, 2) == 0 else f"{v:.2f}".replace("0.", ".").replace("-", "−")  # a value that rounds to zero carries no sign
                ax.text(j, i, shown, ha="center", va="center", fontsize=5.8, color="white" if abs(v) > 0.15 else INK)
                if match[g] == t:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=1.1))
        ax.set_title(NAME[sc], fontsize=7, pad=3)  # the readout of this heat map
        ax.set_xlabel(REFERENCE, fontsize=7, loc="center" if sc == "S" else "right")  # right: the label ends with the heat map, on the page
    axs[0].yaxis.tick_right()  # the tissue names stand between the two heat maps and name the rows of both
    axs[0].set_ylabel("tissue of the readout")


def forest(ax, rows, labels, scores, xlabel, ref=0):
    n = len(scores)
    off = np.linspace(-0.26, 0.26, n) if n > 1 else [0]
    for j, sc in enumerate(scores):
        d = rows[rows.score == sc].set_index("key").reindex(labels)
        y = np.arange(len(labels)) + off[j]
        ax.hlines(y, d.lo, d.hi, color=COL[sc], lw=1.2)
        ax.plot(d.est, y, MK[sc], color=COL[sc], ms=4, mec="white", mew=0.5, label=NAME[sc], ls="")
    ax.axvline(ref, color=RULE, lw=0.8, zorder=0)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)


def panel_c(ax):
    d = pd.read_csv(OUT / "contrast_delta.csv")
    d = d[d.score.isin(["S", "B1", "D"])].assign(key=lambda x: x.gold)
    forest(ax, d, ["liver", "testis", "kidney", "pancreas"], ["S", "B1", "D"], "same tissue minus other tissues, $\\rho_p$")
    ax.set_yticks(range(4), [GL[g] for g in ["liver", "testis", "kidney", "pancreas"]])
    ax.legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.1, columnspacing=1.0)


def panel_d(ax):
    w = pd.read_csv(OUT / "within_gene.csv")
    a = pd.read_csv(OUT / "posthoc" / "A_within_matched.csv")
    lab = {"liver|testis": "Liver, testis", "liver|kidney": "Liver, kidney", "liver|pancreas": "Liver, islet", "testis|kidney": "Testis, kidney", "testis|pancreas": "Testis, islet", "kidney|pancreas": "Kidney, islet"}
    w["key"], a["key"] = w.pair.map(lab), a.pair.map(lab)
    ls = list(lab.values())
    for sc, of in (("S", -0.2), ("B1", 0.2)):
        d = a[a.score == sc].set_index("key").reindex(ls)
        y = np.arange(6) + of
        ax.hlines(y, d.lo, d.hi, color=COL[sc], lw=1.2)
        ax.plot(d.est, y, MK[sc], color=COL[sc], ms=4, mec="white", mew=0.5, ls="")
        e = w[w.score == sc].set_index("key").reindex(ls)
        ax.plot(e.est, y, MK[sc], mfc="white", mec=COL[sc], ms=3.5, mew=0.7, ls="")
    ax.axvline(0, color=RULE, lw=0.8, zorder=0)
    ax.set_yticks(range(6), ls)
    ax.invert_yaxis()
    ax.set_xlabel(f"tissue difference in readout vs in {REFERENCE}, within gene, $\\rho$", loc="right")  # right: it ends with the axis, on the page
    # readouts by color and marker, the two adjustments by fill (column-wise: readouts left, fill right)
    h = [plt.Line2D([], [], marker=MK[sc], color=COL[sc], ls="", ms=4, mec="white", mew=0.5) for sc in ("S", "B1")]
    h += [plt.Line2D([], [], marker="o", color=INK, ls="", ms=3.5), plt.Line2D([], [], marker="o", mfc="white", mec=INK, ls="", ms=3.5, mew=0.7)]
    ax.legend(h, [NAME["S"], NAME["B1"], "filled, matched variability", "open, covariates only"], frameon=False, fontsize=5.6, loc="lower center",
              bbox_to_anchor=(0.5, 0.98), ncol=2, handletextpad=0.1, columnspacing=0.8, labelspacing=0.15)


def panel_e(ax):
    d = pd.read_parquet(OUT / "posthoc" / "D_flip_cells.parquet")
    cats = ["level only", "both", "pattern only", "neither"]
    cols = {"level only": BLUE, "both": PURPLE, "pattern only": ORANGE, "neither": PALE}
    tb = pd.crosstab(d.t_lo, d.cat)[cats].reindex(TIS)
    sh = tb.div(tb.sum(axis=1), axis=0)
    for i, t in enumerate(TIS):
        left = 0
        for c in cats:
            ax.barh(i, sh.at[t, c] - 0.006, left=left, color=cols[c], height=0.66)
            left += sh.at[t, c]
    # a name of two words on two lines: the longest label ends where the page begins
    ax.set_yticks(range(6), [f"{TL[t]} ({int(tb.loc[t].sum())})".replace(" ", "\n", TL[t].count(" ")) for t in TIS])
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
    ax.axvline(0, color=RULE, lw=0.8, zorder=0)
    for g in genes:
        d = r[(r.gene == g) & r.support]
        ax.hlines(y[g], d.z_S.min(), d.z_S.max(), color=RULE, lw=2, zorder=1)
        for _, a in d.iterrows():
            ax.plot(a.z_S, y[g], MK["S"], color=COL["S"], ms=4.2, mec="white", mew=0.5, zorder=3)
            ax.plot(a.z_B1, y[g] + 0.24, MK["B1"], color=COL["B1"], ms=3.4, mec="white", mew=0.3, zorder=2)
    # label the extreme tissues only
    for g in genes:
        d = r[(r.gene == g) & r.support]
        lo, hi = d.loc[d.z_S.idxmin()], d.loc[d.z_S.idxmax()]
        ax.annotate(TL[lo.tissue], (lo.z_S, y[g]), xytext=(0, -7), textcoords="offset points", ha="center", va="top", fontsize=5.6, color=MUTED)
        ax.annotate(TL[hi.tissue], (hi.z_S, y[g]), xytext=(0, -7), textcoords="offset points", ha="center", va="top", fontsize=5.6, color=MUTED)
    ax.set_yticks(range(len(genes)), genes)
    ax.set_ylim(len(genes) - 0.15, -0.6)
    ax.set_xlabel("z within tissue, count-supported tissues")
    h = [plt.Line2D([], [], marker=MK["S"], color=COL["S"], ls="", ms=4), plt.Line2D([], [], marker=MK["B1"], color=COL["B1"], ls="", ms=3.4)]
    ax.legend(h, [NAME["S"], NAME["B1"]], frameon=False, fontsize=6, loc="lower left", handletextpad=0.1, ncol=2, bbox_to_anchor=(0, 1.0), columnspacing=0.8)


def draw_panels() -> None:
    """The six panels from the cached tables, as three pages of one size, each with one row of panels drawn (the other
    rows are not drawn on it, so a figure that takes a row from its page carries no hidden text of the others). The
    pages carry no panel letters."""
    fd.apply_design()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig = plt.figure(figsize=(style.inch(PAGE_W), style.inch(PAGE_H)))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.62, 1, 1], hspace=0.72, wspace=0.65, left=GRID["left"] / PAGE_W, right=GRID["right"] / PAGE_W,
                          top=GRID["top"] / PAGE_H, bottom=GRID["bottom"] / PAGE_H)
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
    rows = [[a, b1, b2], [c, d], [e, f]]
    for k, keep in enumerate(rows, 1):
        for ax in fig.axes:
            ax.set_visible(ax in keep)
        style.save_panel(fig, f"main_tissues/gene_by_tissue_row{k}", exact=True, close=False)
    plt.close(fig)


def main():
    draw_panels()


if __name__ == "__main__":
    main()
