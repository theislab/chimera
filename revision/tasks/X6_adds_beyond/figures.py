"""X6 (EXPLORATORY): one compact figure, incremental information (a) and label degradation (b).

Writes results/revision/X6/figures/fig_adds_beyond.{pdf,png}. PDF fonts TrueType (type 42).
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from revision.tasks.X4_where_differs import common as CM  # noqa: E402
from revision.tasks.X6_adds_beyond.degrade import OUT  # noqa: E402

sys.path.insert(0, str(CM.REPO / "notebooks"))
import _figure_design as FD  # noqa: E402

FIG = OUT / "figures"
# one colour and one marker per readout (X4 hues), so identity is never colour alone
STY = {"D": ("#D55E00", "o", r"$\Delta\alpha$"), "S": ("#0072B2", "s", r"margin $S_g$"),
       "Sdir": ("#009E73", "D", "direct margin (labels)"), "B1b_raw": ("#222222", "^", "B1b profile corr."),
       "B1c_raw": ("#666666", "v", "B1c level match"), "B2b": ("#999999", "<", "B2b specificity"),
       "B1z_raw": ("#CC79A7", "P", "B1z, Leiden on Chimera z"), "B3a": ("#E69F00", "X", "B3a features")}
ORDER = ["D", "S", "Sdir", "B1b_raw", "B1c_raw", "B2b", "B1z_raw", "B3a"]
GOLD = {"G_H3": "liver pattern\nTS vs TMS", "G_H1": "liver level\nEVEE", "G_H4": "testis pattern\nMurat 8 sp.",
        "G_H4_primates": "testis pattern\nprimates", "G_H2": "testis level\nEVEE"}
FOREST_GOLDS = ["G_H3", "G_H1", "G_H4", "G_H4_primates", "G_H2"]
DEG_GOLDS = [("liv_hmouse", "G_H3"), ("liv_hmouse", "G_H1"), ("test_hmouse", "G_H4")]
CONDS = [("full", "full"), ("noise10_mean", "10%\nnoise"), ("noise25_mean", "25%\nnoise"), ("coarse3", "3\nclasses"),
         ("leiden", "Leiden\nr 0.3"), ("leiden_r02", "Leiden\nr 0.2")]


def letter(ax, s):
    ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="right")


def forest(axs):
    d = pd.read_csv(OUT / "incremental_rho.csv")
    d = d[d.variant == "C*"]
    for ax, gold in zip(axs, FOREST_GOLDS):
        x = d[d.gold == gold]
        for i, sc in enumerate(ORDER):
            giv = "other simple baselines" if sc.startswith("B") else "all simple baselines"
            r = x[(x.score == sc) & (x.given == giv)].iloc[0]
            c, m, _ = STY[sc]
            ax.errorbar(r.rho_p, i, xerr=[[r.rho_p - r.ci_lo], [r.ci_hi - r.rho_p]], fmt=m, color=c, ms=3.8, lw=0.8,
                        capsize=0, mec="white", mew=0.3)
        ax.axvline(0, color="#999999", lw=0.5)
        ax.set_ylim(len(ORDER) - 0.4, -0.6)
        ax.set_yticks(range(len(ORDER)), [STY[s][2] for s in ORDER] if ax is axs[0] else [""] * len(ORDER), fontsize=6.5)
        ax.set_xlim(-0.25, 0.2)
        ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2], ["-0.2", "", "0", "", "0.2"])
        ax.set_title(f"{GOLD[gold]}\nn = {int(x.n.iloc[0]):,}", fontsize=6.5)
    axs[2].set_xlabel(r"$\rho_p$ with independent gold, given covariates and all other simple baselines", fontsize=7)


def degradation(axs):
    d = pd.read_csv(OUT / "degradation_rho.csv")
    for ax, (c, gold) in zip(axs, DEG_GOLDS):
        x = d[(d.cohort == c) & (d.gold == gold)]
        xs = np.arange(len(CONDS))
        for sc in ("B1b_raw", "B1c_raw", "B2b", "Sdir"):
            col, m, _ = STY[sc]
            v = [x[(x.score == sc) & (x.condition == cd)].iloc[0] for cd, _ in CONDS]
            off = {"B1b_raw": -0.15, "B1c_raw": -0.05, "B2b": 0.05, "Sdir": 0.15}[sc]
            ax.plot(xs + off, [r.rho_p for r in v], "-", color=col, lw=0.6, alpha=0.7)
            ax.errorbar(xs + off, [r.rho_p for r in v], yerr=[[r.rho_p - r.ci_lo for r in v], [r.ci_hi - r.rho_p for r in v]],
                        fmt=m, color=col, ms=3.4, lw=0.7, capsize=0, mec="white", mew=0.3)
        for sc, ls in (("S", "-"), ("D", "--"), ("B1z_raw", ":")):
            r = x[(x.score == sc) & (x.condition == "model")].iloc[0]
            col = STY[sc][0]
            ax.axhspan(r.ci_lo, r.ci_hi, color=col, alpha=0.10, lw=0)
            ax.axhline(r.rho_p, color=col, ls=ls, lw=0.9)
        ax.axhline(0, color="#999999", lw=0.5)
        ax.set_xticks(xs, [l for _, l in CONDS], fontsize=6)
        lab = "Chimera used labels" if c.startswith("test") else "Chimera without labels"
        ax.set_title(f"{GOLD[gold].replace(chr(10), ', ')}\n{lab}, n = {int(x.n.iloc[0]):,}", fontsize=6.5)
    axs[0].set_ylabel(r"$\rho_p$ given covariates", fontsize=7)


def main():
    FD.apply_design()
    FIG.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(FD.DOUBLE_W, 5.6))
    gs = fig.add_gridspec(2, 15, height_ratios=[1, 1.05], hspace=0.85, wspace=1.6, left=0.17, right=0.99, top=0.91, bottom=0.17)
    fa = [fig.add_subplot(gs[0, 3 * k:3 * k + 3]) for k in range(5)]
    forest(fa)
    letter(fa[0], "a")
    fb = [fig.add_subplot(gs[1, 5 * k:5 * k + 5]) for k in range(3)]
    for ax in fb[1:]:
        ax.sharey(fb[0])
    degradation(fb)
    letter(fb[0], "b")
    h = [plt.Line2D([], [], color=STY[s][0], marker=STY[s][1], ls="", ms=3.6) for s in ("B1b_raw", "B1c_raw", "B2b", "Sdir")]
    h += [plt.Line2D([], [], color=STY[s][0], ls=ls, lw=0.9) for s, ls in (("S", "-"), ("D", "--"), ("B1z_raw", ":"))]
    l = ["B1b, recomputed", "B1c, recomputed", "B2b, recomputed", "direct margin, recomputed",
         r"margin $S_g$ (no labels on liver)", r"$\Delta\alpha$", "B1z (Chimera latent)"]
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.3, handlelength=1.6, bbox_to_anchor=(0.55, 0.0))
    FD.save_figure(fig, "fig_adds_beyond", out_dir=FIG, formats=("pdf", "png"), close=True)
    print("figure written", flush=True)


if __name__ == "__main__":
    main()
