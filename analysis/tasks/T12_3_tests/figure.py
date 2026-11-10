"""T12.3 supplementary figure: bone-marrow D and S_g against Shay et al. 2013 COE, with the B1 baseline.

Panels (all from ``results/analysis/P12``; nothing is recomputed except the binned residuals of a):
a  mean C-residual rank of D, S_g and B1 (residualised on pi) per COE decile (E1 universe);
b  E1: rho_p(score, COE | C) with 95% ESM-block CI, all genes and top pi tertile;
c  E2: stratified AUROC, Shay divergent vs COE-conserved genes, 95% CI (0.5 = no separation);
d  E3: named-lineage hits of 10 genes, with the chance expectation;
e  E4: mouse-row minus lemur-row share of divergent S_g (Chimera);
f  E4 baseline: E(human, lemur) - E(human, mouse) pseudobulk correlation.

    PYTHONPATH=<chimera repo> python analysis/tasks/T12_3_tests/figure.py
"""

from __future__ import annotations

import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.common import controls as K  # noqa: E402
from analysis.common import gene_ids  # noqa: E402
from analysis.tasks.T4_3_scoreboard import stats as S  # noqa: E402
from analysis.tasks.T8_4_figures import common as cm  # noqa: E402

RES = gene_ids.REPO_ROOT / "results" / "analysis" / "P12"
T = RES / "t12_3"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COL = {"D": "#2a78d6", "S_ema": "#1baf7a", "B1_pi": "#eb6834", "B1_raw": "#8f8e89"}
NAME = {"D": "D (Chimera)", "S_ema": "S$_g$ (Chimera)", "B1_pi": "B1, resid. on π", "B1_raw": "B1, raw"}
ORDER = ["D", "S_ema", "B1_pi", "B1_raw"]

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 6, "axes.titlesize": 6.5,
                     "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.fontsize": 5.5,
                     "legend.frameon": False, "font.family": "sans-serif"})


def clean(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def label(ax, s):
    """Panel letters are added by F1_figures/compose.py under each panel."""


def panel_a(ax):
    tab = pd.read_parquet(RES / "readout" / "bm_genes.parquet")
    coe = pd.read_csv(gene_ids.REPO_ROOT / "data/analysis/bone_marrow/shay_coe.tsv", sep="\t")
    coe = coe.drop_duplicates("hsapiens_gene_id", keep=False).set_index("hsapiens_gene_id")["coe"]
    tab["COE"] = coe.reindex(tab["hsapiens_gene_id"]).to_numpy()
    u = tab.dropna(subset=["D", "COE", "B1_raw", "B1_pi"] + K.COVARIATE_COLUMNS).reset_index(drop=True)
    Z = S.design(u, K.COVARIATE_COLUMNS)
    dec = pd.qcut(S.midrank(u["COE"]), 10, labels=False)
    x = u.groupby(dec)["COE"].median().to_numpy()
    for s in ("D", "S_ema", "B1_pi"):
        e = S.resid(S.midrank(u[s]), Z) / len(u)
        m = pd.Series(e).groupby(dec).mean()
        se = pd.Series(e).groupby(dec).sem()
        ax.errorbar(x, m, yerr=1.96 * se, color=COL[s], lw=1.4, marker="o", ms=3.2, capsize=0,
                    elinewidth=0.8, label=NAME[s], mec="white", mew=0.5)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("Shay COE (decile median)")
    ax.set_ylabel("mean residual rank given C\n(fraction of genes)")
    ax.set_title(f"Score vs COE, n = {len(u):,} genes", loc="left")
    ax.legend(loc="upper left", handlelength=1.2)
    clean(ax, "y")


def forest(ax, df, est, lo, hi, groups, xref, xlabel, title, heads=None):
    y0 = 0
    yt, yl = [], []
    for gi, (g, mk) in enumerate(groups):
        if heads:
            ax.text(0.0, y0 - 0.75, heads[gi], transform=ax.get_yaxis_transform(), fontsize=5.5, color=INK2,
                    ha="left", va="center", style="italic")
        for i, s in enumerate(ORDER):
            r = df[(df["score"] == s) & (df["grp"] == g)]
            if not len(r):
                continue
            r = r.iloc[0]
            y = y0 + i
            ax.plot([r[lo], r[hi]], [y, y], color=COL[s], lw=1.2, solid_capstyle="round")
            ax.plot(r[est], y, marker=mk, ms=4, color=COL[s], mfc=COL[s] if mk == "o" else "white", mew=1.0)
            yt.append(y)
            yl.append(NAME[s] if (heads or g == groups[0][0]) else "")
        y0 += len(ORDER) + 1
    ax.axvline(xref, color=INK2, lw=0.6, ls="--")
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    ax.set_ylim(y0 - 1, -1.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    clean(ax, "x")


def main() -> int:
    e1 = pd.read_csv(T / "e1.tsv", sep="\t")
    e1 = e1[e1["score"].isin(ORDER)].rename(columns={"universe": "grp"})
    e2 = pd.read_csv(T / "e2.tsv", sep="\t")
    e2 = e2[(e2["strata"] == "pi_x_expr_deciles_5to1") & e2["score"].isin(ORDER)].assign(grp="main")
    ex = pd.read_csv(T / "explore_e2_range_matched.tsv", sep="\t")
    ex = ex[ex["comparison"].str.startswith("conserved with") & ex["score"].isin(ORDER)].assign(grp="range")
    e3 = pd.read_csv(T / "e3.tsv", sep="\t")
    e4 = pd.read_csv(T / "e4.tsv", sep="\t")

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), gridspec_kw=dict(wspace=0.95, hspace=0.6))
    a, b, c, d, e, f = axes.ravel()
    panel_a(a)
    label(a, "a")

    forest(b, e1, "rho_p", "ci_lo", "ci_hi", [("all", "o"), ("top_pi_tertile", "s")], 0.0,
           "ρ$_p$(score, COE | C), 95% CI", "E1: partial rank correlation",
           heads=["all genes (4,111)", f"top π tertile ({int(e1.loc[e1.grp == 'top_pi_tertile', 'n'].iloc[0]):,})"])
    label(b, "b")
    n_div = int(e2["n_divergent"].iloc[0])
    n_con = int(e2["n_conserved"].iloc[0])
    n_c2 = int(ex["n_conserved"].iloc[0])
    forest(c, pd.concat([e2, ex]), "auroc", "ci_lo", "ci_hi", [("main", "o"), ("range", "s")], 0.5,
           "stratified AUROC, 95% CI (< 0.5: divergent lower)", f"E2: Shay divergent ({n_div}) vs COE-conserved",
           heads=[f"pre-specified: all conserved ({n_con:,})",
                  f"exploratory: range-matched ({n_c2})"])
    label(c, "c")

    names = {"Chimera C_gk, all species rows": ("Chimera, all rows", COL["D"]),
             "Chimera mouse rows only": ("Chimera, mouse rows", COL["S_ema"]),
             "B1 leave-one-type-out": ("B1 leave-one-type-out", COL["B1_pi"])}
    for i, r in enumerate(e3.itertuples()):
        nm, cl = names[r.readout]
        d.barh(i, r.n_hit, color=cl, height=0.55)
        d.plot([r.expected_empirical] * 2, [i - 0.38, i + 0.38], color=INK, lw=1.2)
        d.text(r.n_hit + 0.2, i, f"{r.n_hit}/10, p = {r.p_empirical:.2f}", va="center", fontsize=5, color=INK)
    d.set_yticks(range(len(e3)))
    d.set_yticklabels([names[r][0].replace(", ", ",\n").replace(" leave", "\nleave") for r in e3["readout"]])
    d.invert_yaxis()
    d.set_xlim(0, 10)
    d.set_xticks([0, 2, 4, 6, 8, 10])
    d.set_xlabel("of 10 named genes: top divergent\ntype is a Shay-named lineage")
    d.set_title("E3: named lineages (tick: chance)", loc="left")
    clean(d, "x")
    label(d, "d")

    sets = ["Shay divergent", "COE conserved (above median)", "all vocabulary"]
    short = {"Shay divergent": "divergent", "COE conserved (above median)": "conserved",
             "all vocabulary": "all genes"}
    for ax, key, cl, xl, tt, lab in (
            (e, "Chimera", COL["D"], "mouse minus lemur share\nof divergent S$_g$, 95% CI", "E4: Chimera species rows", "e"),
            (f, "B1 pair", COL["B1_pi"], "E(human, lemur) − E(human, mouse)\n95% CI", "E4 baseline: pseudobulk", "f")):
        sub = e4[e4["readout"].str.startswith(key)].set_index("gene_set").loc[sets]
        for i, (gs, r) in enumerate(sub.iterrows()):
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=cl, lw=1.2, solid_capstyle="round")
            ax.plot(r["mean"], i, "o", ms=4, color=cl)
        ax.axvline(0, color=INK2, lw=0.6, ls="--")
        ax.set_yticks(range(len(sets)))
        ax.set_yticklabels([f"{short[s]} ({int(sub.loc[s, 'n']):,})" for s in sets])
        ax.invert_yaxis()
        ax.set_xlabel(xl)
        ax.set_title(tt, loc="left")
        clean(ax, "x")
        label(ax, lab)
    out = RES / "fig_t12_3_bone_marrow"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    # One PDF per panel for F1_figures/compose.py.
    for k, ax in zip("abcdef", axes.ravel()):
        cm.save_axes(fig, f"supp_bone_marrow_{k}", [ax])
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(out.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
