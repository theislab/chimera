"""Supplementary figure: bone-marrow D and S_g against Shay et al. 2013 COE, with the B1 baseline.

Panels (all from ``results/analysis/P12``; nothing is recomputed except the binned residuals of a):
a  mean C-residual rank of D, S_g and B1 (residualized on pi) per COE decile (E1 universe);
b  E1: rho_p(score, COE | C) with 95% ESM-block CI, all genes and top pi tertile;
c  E2: stratified AUROC, Shay divergent vs COE-conserved genes, 95% CI (0.5 = no separation);
d  E3: named-lineage hits of 10 genes, with the chance expectation;
e  E4: mouse-row minus lemur-row share of divergent S_g (Chimera);
f  E4 baseline: E(human, lemur) - E(human, mouse) pseudobulk correlation.

Each panel is one file, figures/panels/supp_bone_marrow_reference/<a..f>.pdf, drawn at its exact size on the page: three
panels of one width in a row, the axes of a row on one baseline (house style: figures/style.py).

    PYTHONPATH=<chimera repo> python -m analysis.tasks.T12_3_tests.figure
"""

from __future__ import annotations

import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from analysis.common import controls as K  # noqa: E402
from analysis.common import gene_ids  # noqa: E402
from analysis.tasks.T4_3_scoreboard import stats as S  # noqa: E402
from figures import style  # noqa: E402

RES = gene_ids.REPO_ROOT / "results" / "analysis" / "P12"
T = RES / "t12_3"

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this figure
FAMILY = "supp_bone_marrow_reference"
INK, REF_LINE = style.INK, "#888888"  # REF_LINE: zero and no-separation lines
DELTA_ALPHA, COUNT_EVIDENCE, PROFILE = (style.READOUT[k] for k in ("delta_alpha", "count_evidence", "profile_correlation"))
PROFILE_AS_IS = "#999999"  # profile correlation without the prior removed: the marker of profile correlation in a lighter gray
COL = {"D": DELTA_ALPHA.color, "S_ema": COUNT_EVIDENCE.color, "B1_pi": PROFILE.color, "B1_raw": PROFILE_AS_IS}
MARKER = {"D": DELTA_ALPHA.marker, "S_ema": COUNT_EVIDENCE.marker, "B1_pi": PROFILE.marker, "B1_raw": PROFILE.marker}
NAME = {"D": "Δα", "S_ema": COUNT_EVIDENCE.name, "B1_pi": PROFILE.name + ",\nprior removed", "B1_raw": PROFILE.name}
ORDER = ["D", "S_ema", "B1_pi", "B1_raw"]
COE = "COE"  # conservation of expression across sorted blood lineages; the caption names it and its source
HALO = [pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()]  # keeps the chance tick visible on a dark bar

# Sizes in bp. Three panels of one width fill a row; a row has one height and one set of margins above and below the
# axes, so the axes of a row share a baseline. LEFT is the room for the tick labels and the axis label of each panel.
PANEL_W = (style.TEXTWIDTH - 2 * style.GAP) / 3
PANEL_H = {"top": 149.0, "bottom": 103.0}
ABOVE_AXES, BELOW_AXES, RIGHT, EDGE = 13.0, 34.0, 6.0, 1.5
LEFT = {"a": 47.0, "b": 80.0, "c": 80.0, "d": 80.0, "e": 68.0, "f": 68.0}


def new_panel(name, row, title):
    """A figure of the exact size of a panel with one axes at the margins of its row, and its title at the left edge."""
    w, h = PANEL_W, PANEL_H[row]
    fig = plt.figure(figsize=(style.inch(w), style.inch(h)))
    ax = fig.add_axes([LEFT[name] / w, BELOW_AXES / h, (w - LEFT[name] - RIGHT) / w, (h - ABOVE_AXES - BELOW_AXES) / h])
    fig.text(EDGE / w, 1 - EDGE / h, title, ha="left", va="top", fontsize=style.SIZE_TITLE)
    return fig, ax


def save(fig, name):
    """Save a panel at its exact size (placed with X(...) in the figure spec); nothing may reach beyond the page."""
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    inside = box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.get_figwidth() and box.y1 <= fig.get_figheight()
    assert inside, f"panel {name}: content box {[round(72 * v, 1) for v in box.extents]} bp beyond the page {PANEL_W:.1f} x {72 * fig.get_figheight():.1f} bp"
    return style.save_panel(fig, f"{FAMILY}/{name}", exact=True)


def panel_a():
    tab = pd.read_parquet(RES / "readout" / "bm_genes.parquet")
    coe = pd.read_csv(gene_ids.REPO_ROOT / "data/analysis/bone_marrow/shay_coe.tsv", sep="\t")
    coe = coe.drop_duplicates("hsapiens_gene_id", keep=False).set_index("hsapiens_gene_id")["coe"]
    tab["COE"] = coe.reindex(tab["hsapiens_gene_id"]).to_numpy()
    u = tab.dropna(subset=["D", "COE", "B1_raw", "B1_pi"] + K.COVARIATE_COLUMNS).reset_index(drop=True)
    Z = S.design(u, K.COVARIATE_COLUMNS)
    dec = pd.qcut(S.midrank(u["COE"]), 10, labels=False)
    x = u.groupby(dec)["COE"].median().to_numpy()
    fig, ax = new_panel("a", "top", f"By decile of {COE}, n = {len(u):,} genes")
    for s in ("D", "S_ema", "B1_pi"):
        e = S.resid(S.midrank(u[s]), Z) / len(u)
        m = pd.Series(e).groupby(dec).mean()
        se = pd.Series(e).groupby(dec).sem()
        ax.errorbar(x, m, yerr=1.96 * se, color=COL[s], lw=style.LW_DATA, marker=MARKER[s], ms=3.6, capsize=0,
                    elinewidth=0.8, label=NAME[s], mec="white", mew=0.4)
    ax.axhline(0, color=REF_LINE, lw=style.LW_REF)
    ax.set_xlabel(f"conservation of expression, {COE}\n(decile median)", loc="right")
    ax.set_ylabel("mean covariate-adjusted rank\n(fraction of genes)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.03, 1.0), handlelength=1.6, labelspacing=0.3, borderaxespad=0)
    save(fig, "a")


def forest(name, df, est, lo, hi, groups, xref, xlabel, title, heads):
    """Blocks of rows (`groups`: key of the block and whether its markers are filled), each under its heading. A heading
    starts at the left edge of the panel; the axis line and the reference line are drawn block by block, so that no
    line runs through a heading."""
    fig, ax = new_panel(name, "top", title)
    from_edge = blended_transform_factory(fig.transFigure, ax.transData)
    y0 = 0
    yt, yl = [], []
    for (g, filled), head in zip(groups, heads):
        ax.text(EDGE / PANEL_W, y0 - 0.85, head, transform=from_edge, fontsize=style.SIZE_TEXT, ha="left", va="center",
                style="italic")
        ys = []
        for i, s in enumerate(ORDER):
            r = df[(df["score"] == s) & (df["grp"] == g)]
            if not len(r):
                continue
            r = r.iloc[0]
            y = y0 + i
            ax.plot([r[lo], r[hi]], [y, y], color=COL[s], lw=style.LW_CI, solid_capstyle="round")
            ax.plot(r[est], y, marker=MARKER[s], ms=4, color=COL[s], mfc=COL[s] if filled else "white", mew=0.8)
            ys.append(y)
            yl.append(NAME[s])
        span = [min(ys) - 0.5, max(ys) + 0.5]
        ax.plot([xref, xref], span, color=REF_LINE, lw=style.LW_REF, ls="--", zorder=1)
        ax.plot([0, 0], span, transform=ax.get_yaxis_transform(), color=INK, lw=style.LW_AXES, clip_on=False,
                solid_capstyle="butt")
        yt += ys
        y0 += len(ORDER) + 1
    ax.spines["left"].set_visible(False)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, linespacing=0.95)
    ax.set_ylim(y0 - 1.4, -1.4)
    ax.set_xlabel(xlabel, loc="right")
    save(fig, name)


def draw_panels() -> None:
    """The six panels from the cached tables (nothing is computed except the binned residuals of panel a)."""
    style.apply()
    e1 = pd.read_csv(T / "e1.tsv", sep="\t")
    e1 = e1[e1["score"].isin(ORDER)].rename(columns={"universe": "grp"})
    e2 = pd.read_csv(T / "e2.tsv", sep="\t")
    e2 = e2[(e2["strata"] == "pi_x_expr_deciles_5to1") & e2["score"].isin(ORDER)].assign(grp="main")
    ex = pd.read_csv(T / "explore_e2_range_matched.tsv", sep="\t")
    ex = ex[ex["comparison"].str.startswith("conserved with") & ex["score"].isin(ORDER)].assign(grp="range")
    e3 = pd.read_csv(T / "e3.tsv", sep="\t")
    e4 = pd.read_csv(T / "e4.tsv", sep="\t")

    panel_a()

    def n_of(grp):
        return int(e1.loc[e1.grp == grp, "n"].iloc[0])

    forest("b", e1, "rho_p", "ci_lo", "ci_hi", [("all", True), ("top_pi_tertile", False)], 0.0,
           f"partial rank correlation with {COE},\n$\\rho_p$ (95% CI)", f"Partial rank correlation with {COE}",
           heads=[f"all genes ({n_of('all'):,})", f"top π tertile ({n_of('top_pi_tertile'):,})"])
    n_div = int(e2["n_divergent"].iloc[0])
    n_con = int(e2["n_conserved"].iloc[0])
    n_d2 = int(ex["n_divergent"].iloc[0])
    n_c2 = int(ex["n_conserved"].iloc[0])
    forest("c", pd.concat([e2, ex]), "auroc", "ci_lo", "ci_hi", [("main", True), ("range", False)], 0.5,
           "stratified AUROC (95% CI);\nbelow 0.5: divergent genes score lower", "Divergent against conserved genes",
           heads=[f"pre-specified: {n_div:,} against {n_con:,}",
                  f"exploratory, range-matched: {n_d2:,} against {n_c2:,}"])

    # row of the table -> name, color, filled
    names = {"Chimera C_gk, all species rows": ("$S_g$ by cell type,\nall species", COUNT_EVIDENCE.color, True),
             "Chimera mouse rows only": ("$S_g$ by cell type,\nmouse cells only", COUNT_EVIDENCE.color, False),
             "B1 leave-one-type-out": (PROFILE.name + ",\none cell type left out", PROFILE.color, True)}
    fig, d = new_panel("d", "bottom", "Named lineages")
    for i, r in enumerate(e3.itertuples()):
        nm, cl, filled = names[r.readout]
        d.barh(i, r.n_hit, color=cl if filled else "white", edgecolor=cl, linewidth=0.8, height=0.55)
        d.plot([r.expected_empirical] * 2, [i - 0.4, i + 0.4], color=INK, lw=1.1, solid_capstyle="butt", path_effects=HALO)
        d.text(r.n_hit + 0.3, i, f"{r.n_hit}/10, p = {r.p_empirical:.2f}", va="center", fontsize=style.SIZE_TEXT, color=INK)
    d.set_yticks(range(len(e3)))
    d.set_yticklabels([names[r][0] for r in e3["readout"]], linespacing=0.95)
    d.invert_yaxis()
    d.set_xlim(0, 10)
    d.set_xticks([0, 2, 4, 6, 8, 10])
    d.set_xlabel("of 10 genes: top divergent cell type\nis a published lineage (tick: chance)", loc="right")
    save(fig, "d")

    sets = ["Shay divergent", "COE conserved (above median)", "all vocabulary"]
    short = {"Shay divergent": "divergent", "COE conserved (above median)": "conserved",
             "all vocabulary": "all genes"}
    for name, key, mark, xl, tt in (
            ("e", "Chimera", COUNT_EVIDENCE, "share of divergent $S_g$,\nmouse minus lemur cells (95% CI)", "Divergent $S_g$ by species"),
            ("f", "B1 pair", PROFILE, "profile correlation, human–lemur\nminus human–mouse (95% CI)", "Profile correlation by species pair")):
        fig, ax = new_panel(name, "bottom", tt)
        sub = e4[e4["readout"].str.startswith(key)].set_index("gene_set").loc[sets]
        for i, (gs, r) in enumerate(sub.iterrows()):
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=mark.color, lw=style.LW_CI, solid_capstyle="round")
            ax.plot(r["mean"], i, mark.marker, ms=4, color=mark.color)
        ax.axvline(0, color=REF_LINE, lw=style.LW_REF, ls="--")
        ax.set_yticks(range(len(sets)))
        ax.set_yticklabels([f"{short[s]} ({int(sub.loc[s, 'n']):,})" for s in sets])
        ax.set_ylim(len(sets) - 0.5, -0.5)
        ax.set_xlabel(xl, loc="right")
        save(fig, name)


def main() -> int:
    draw_panels()
    return 0


if __name__ == "__main__":
    sys.exit(main())
