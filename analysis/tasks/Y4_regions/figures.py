"""Y4 (EXPLORATORY): one compact figure. Writes results/analysis/Y4/figures/fig_regions.{pdf,png}. PDF fonts TrueType."""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.tasks.T4_3_scoreboard import stats as ST  # noqa: E402
from analysis.tasks.Y4_regions import replicate as RP  # noqa: E402
from analysis.tasks.Y4_regions import rules as RU  # noqa: E402
from analysis.tasks.Y4_regions import screen as SC  # noqa: E402

sys.path.insert(0, str(SC.REPO / "notebooks"))
import _figure_design as FD  # noqa: E402

OUT = SC.OUT
FIG = OUT / "figures"
D_C, S_C, SDIR_C, BASE_C, GREY = "#D55E00", "#0072B2", "#009E73", "#222222", "#999999"
LABEL = {
    "expr_T2__tau_T3": "L level, S: mid expr., specific", "absshift_T3__spread_T3": "T pattern, S: large gap, high spread",
    "pihi__tau_T2": "L pattern, S: high π, mid specificity", "expr_T2__spread_T1": "L pattern, S: mid expr., flat",
    "absshift_T3": "L pattern, Δα: large gap", "pilo__spread_T3": "L pattern, Δα: low π, high spread",
    "spread_T1": "L pattern, S: flat", "tau_Q4": "T level, S: upper specificity",
    "peakmargin_T3": "T pattern, Δα: strong peak type", "pihi__expr_T2": "L pattern, Δα, 4-sp: high π, mid expr.",
}


def letter(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="right")


def forest(ax):
    u = pd.read_csv(OUT / "replication_units.csv")
    p = pd.read_csv(OUT / "replication_primary.csv").sort_values(["replicates", "rho"], ascending=[False, False]).reset_index(drop=True)
    for i, r in p.iterrows():
        col = S_C if r.readout == "S" else D_C
        mk = "s" if r.readout == "S" else "o"
        o = u[(u.section == "discovery_final") & (u.cohort == r.cohort) & (u.gold == r.gold) & (u.readout == r.readout) & (u.region == r.region)].iloc[0]
        ax.errorbar(o.rho, i - 0.2, xerr=[[o.rho - o.lo], [o.hi - o.rho]], fmt=mk, mfc="white", mec=col, color=col, ms=3.4, lw=0.6, mew=0.7)
        ax.errorbar(r.rho, i + 0.08, xerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt=mk, color=col, ms=3.8, lw=1.0, mec="white", mew=0.3)
        J = [b for b in str(r.J).split(",") if b]
        best = max(r[f"rho_{b}"] for b in J)
        ax.plot(best, i + 0.34, "^", color=BASE_C, ms=3.4, mec="white", mew=0.3)
        ax.plot(r.rho_Sdir, i + 0.34, "D", color=SDIR_C, ms=3.0, mec="white", mew=0.3)
    ax.axvline(0, color=GREY, lw=0.5)
    ax.set_ylim(len(p) - 0.4, -0.6)
    n_rep = int(p.replicates.sum())
    ax.axhline(n_rep - 0.48, color=GREY, lw=0.5, ls=":")
    ax.set_yticks(range(len(p)), [LABEL[x] for x in p.region], fontsize=6)
    ax.set_xlabel(r"$\rho_p$ inside the region, given covariates", fontsize=7)
    ax.set_title("6 of 10 carried regions replicate (above the line)", fontsize=6.8, loc="left")
    ax.set_xlim(-0.17, 0.95)
    h = [plt.Line2D([], [], color=GREY, marker="s", mfc="white", mec=GREY, ls="", ms=3.4),
         plt.Line2D([], [], color=GREY, marker="s", ls="", ms=3.8),
         plt.Line2D([], [], color=S_C, marker="s", ls="", ms=3.8), plt.Line2D([], [], color=D_C, marker="o", ls="", ms=3.8),
         plt.Line2D([], [], color=BASE_C, marker="^", ls="", ms=3.4), plt.Line2D([], [], color=SDIR_C, marker="D", ls="", ms=3.0)]
    ax.legend(h, ["odd genes", "even genes", r"$S_g$", r"$\Delta\alpha$", "best direct baseline, even", "direct margin, even"], fontsize=5.6,
              loc="lower right", frameon=False, handletextpad=0.2, labelspacing=0.25, borderaxespad=0.1)


def terms(ax):
    d = pd.read_csv(OUT / "posthoc_level_terms.csv")
    d = d[(d.region == "expr_T2__tau_T3") & (d.what == "EVEE")]
    order = [("S", r"$S_g$"), ("Sdir", "direct\nmargin"), ("Sdir_pattern", "pattern\ngain"), ("Sdir_level", "level\ncost"), ("B1c_raw", "B1c"), ("B1b_raw", "B1b")]
    off = {("region", "even"): 0.12, ("region", "odd"): -0.12, ("rest", "even"): 0.30, ("rest", "odd"): -0.30}
    for (part, sp), dx in off.items():
        x = d[(d.part == part) & (d.split == sp)].set_index("score")
        for i, (k, _) in enumerate(order):
            r = x.loc[k]
            filled = part == "region"
            c = S_C if filled else GREY
            ax.errorbar(i + dx, r.rho, yerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt="o" if sp == "even" else "s", color=c, ms=3.2, lw=0.7, mfc=c if sp == "even" else "white", mec=c, mew=0.6)
    ax.axhline(0, color=GREY, lw=0.5)
    ax.set_xticks(range(len(order)), [l for _, l in order], fontsize=6)
    ax.set_ylabel(r"$\rho_p$ with liver EVEE level", fontsize=7)
    ax.set_title("Inside, gain stops running against level", fontsize=6.8, loc="left")
    ax.set_ylim(-0.3, 0.78)
    h = [plt.Line2D([], [], color=S_C, marker="o", ls="", ms=3.2), plt.Line2D([], [], color=GREY, marker="o", ls="", ms=3.2),
         plt.Line2D([], [], color=GREY, marker="s", mfc="white", mec=GREY, ls="", ms=3.2), plt.Line2D([], [], color=GREY, marker="o", ls="", ms=3.2)]
    ax.legend(h[:3], ["inside the region", "outside", "open = odd, filled = even"], fontsize=5.6, frameon=False, loc="upper center", ncol=3, handletextpad=0.1, columnspacing=0.7)


def scatter(ax):
    c = "liv_hmouse"
    t = SC.table(c)
    m = RP.mask_of(c, "expr_T2__tau_T3") & t["split"].isin(["odd", "even"]).to_numpy() & t[["G_H1"] + RU.C].notna().all(1).to_numpy()
    T = t.loc[m].copy()
    Z = SC.zdesign(T, RU.C)
    for k in ("S", "G_H1"):
        e = ST.resid(ST.midrank(T[k].to_numpy(float)), Z)
        T["r_" + k] = e / e.std()
    ax.scatter(T.loc[T.split == "odd", "r_S"], T.loc[T.split == "odd", "r_G_H1"], s=6, facecolors="none", edgecolors=GREY, lw=0.4, label="odd")
    ax.scatter(T.loc[T.split == "even", "r_S"], T.loc[T.split == "even", "r_G_H1"], s=6, color=GREY, lw=0, label="even")
    names = {"HEG1": (0.1, 0.1), "PLXND1": (-0.55, 0.32), "PREX1": (0.12, -0.12), "TGFBI": (-0.82, -0.28), "VIPR1": (-0.55, 0.22), "SPEF2": (0.12, -0.12), "AK9": (-0.1, 0.16)}
    for n, (dx, dy) in names.items():
        r = T[T.symbol == n]
        if len(r):
            r = r.iloc[0]
            ax.plot(r.r_S, r.r_G_H1, "s", color=S_C, ms=3.4, mec="white", mew=0.3)
            ax.text(r.r_S + dx, r.r_G_H1 + dy, n, fontsize=5.6)
    ax.axhline(0, color=GREY, lw=0.4)
    ax.axvline(0, color=GREY, lw=0.4)
    ax.set_xlabel(r"$S_g$, residual rank given covariates (SD)", fontsize=7)
    ax.set_ylabel("EVEE level gold, residual rank (SD)", fontsize=7)
    ax.set_title("Mid expression, specific genes, liver human-mouse", fontsize=6.8, loc="left")
    ax.legend(fontsize=5.6, frameon=False, loc="lower right", handletextpad=0.1)


def gap(ax):
    o = pd.read_csv(OUT / "gap_map_odd.csv")
    e = pd.read_csv(OUT / "gap_map_even.csv")
    rows = [("liv_hmouse", "cls_secreted", "L secreted"), ("test_hmouse", "kw_Cilium", "T cilium"), ("test_hmouse", "age_young", "T young genes"),
            ("liv_hmouse", "kw_Ubl_conjugation_pathway", "L ubiquitin pathway"), ("test_hmouse", "kw_mRNA_processing", "T mRNA processing"),
            ("test_hmouse", "kw_Protein_transport", "T protein transport")]
    for i, (c, reg, lab) in enumerate(rows):
        for k, (d, dy) in enumerate(((o, -0.14), (e, 0.14))):
            r = d[(d.cohort == c) & (d.region == reg)].iloc[0]
            ax.plot([r.id_pct, r.gold_pct], [i + dy, i + dy], color="#CCCCCC", lw=0.6, zorder=1)
            mf = {"mfc": "white"} if k == 0 else {}
            ax.plot(r.id_pct, i + dy, "v", color=BASE_C, ms=3.4, mew=0.6, **mf)
            ax.plot(r.S_pct, i + dy, "s", color=S_C, ms=3.4, mew=0.6, **mf)
            ax.plot(r.gold_pct, i + dy, "D", color=SDIR_C, ms=3.0, mew=0.6, **mf)
    ax.axvline(0.5, color=GREY, lw=0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)), [x[2] for x in rows], fontsize=6)
    ax.set_xlim(0.15, 0.85)
    ax.set_xlabel("Mean percentile given expression and detection, unmasked genes", fontsize=7)
    ax.set_title("Largest gaps between protein identity and deployment", fontsize=6.8, loc="left")
    h = [plt.Line2D([], [], color=BASE_C, marker="v", ls="", ms=3.4), plt.Line2D([], [], color=S_C, marker="s", ls="", ms=3.4), plt.Line2D([], [], color=SDIR_C, marker="D", ls="", ms=3.0)]
    ax.legend(h, ["protein identity", r"$S_g$", "pattern gold", ], fontsize=5.6, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.56), handletextpad=0.2, labelspacing=0.25)


def main():
    FD.apply_design()
    FIG.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(FD.DOUBLE_W, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.42, wspace=0.28, left=0.235, right=0.985, top=0.95, bottom=0.085)
    a, b, c, d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    forest(a)
    terms(b)
    gap(c)
    scatter(d)
    for ax, s in zip((a, b, c, d), "abcd"):
        letter(ax, s)
    FD.save_figure(fig, "fig_regions", out_dir=FIG, formats=("pdf", "png"), close=True)
    print("figure written", flush=True)


if __name__ == "__main__":
    main()
