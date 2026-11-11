"""Genes with few counts (exploratory): three panels (masked genes, thinned depth, held-out chromosomes) from the
result tables of this task. The figure is not in the paper; its layout is ``extra_sparse_genes`` in
``figures/specs/not_in_paper.py``.

Inputs: ``results/analysis/D1/`` (``a_rho.csv``, ``b_rho.csv``, ``c_rho.csv``, ``c_ceiling.csv``), bundle
``analysis_results``. Writes ``results/analysis/D1/panels/``.
    python -m analysis.tasks.D1_sparse_genes.figure
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.tasks.D1_sparse_genes import rules as RU  # noqa: E402
from analysis.tasks.T8_4_figures import common as cm  # noqa: E402

OUT = RU.RES / "panels"
D_C, SDIR_C, FWD_C, R_C, B1_C, GREY = "#D55E00", "#009E73", "#0072B2", "#CC79A7", "#666666", "#AAAAAA"
STYLE = {"D": (D_C, "o", "Δα, imputed"), "Sdir": (SDIR_C, "s", "Sdir, direct"), "S_fwd": (FWD_C, "^", "Forward margin"), "R_S_p": (R_C, "D", "Ridge on protein, target S"),
         "R_Sdir_p": (GREY, "v", "Ridge on protein, target Sdir"), "B1": (B1_C, "P", "B1, where defined")}
COH = {"liv_hmouse": "Liver H+Mouse", "liv_hmonkey": "Liver H+Monkey", "liv_hzeb": "Liver H+Zebrafish", "liv_4sp": "Liver 4 species", "test_hmouse": "Testis H+Mouse", "test_hmac": "Testis H+Macaque",
       "test_3sp": "Testis 3 species", "kidney": "Kidney", "islet": "Islet", "bone_marrow": "Bone marrow"}
KIND = {"pattern": "pattern", "level": "level", "pattern_coe": "pattern"}
SHOW = [("liv_hmouse", "pattern"), ("liv_hmonkey", "pattern"), ("liv_hzeb", "pattern"), ("liv_4sp", "pattern"), ("test_hmouse", "pattern"), ("test_hmac", "pattern"), ("test_3sp", "pattern"),
        ("kidney", "pattern"), ("islet", "pattern"), ("liv_hmouse", "level"), ("liv_4sp", "level"), ("test_hmouse", "level"), ("test_3sp", "level"), ("kidney", "level")]


def masked_panel(path_name: str):
    a = pd.read_csv(RU.RES / "a_rho.csv")
    a = a[(a.stratum == "masked") & (a.universe == "clean") & (a.split == "all") & (a.group == "always")]
    fig, ax = plt.subplots(figsize=(2.9, 3.4))
    ys = np.arange(len(SHOW))[::-1]
    offs = {"D": 0.26, "Sdir": 0.09, "S_fwd": -0.09, "R_S_p": -0.26}
    for y, (c, k) in zip(ys, SHOW):
        g = a[(a.cohort == c) & (a.kind == k)].set_index("score")
        for s, o in offs.items():
            if s not in g.index:
                continue
            col, mk, _ = STYLE[s]
            ax.plot([g.ci_lo[s], g.ci_hi[s]], [y + o, y + o], color=col, lw=0.9, solid_capstyle="round")
            ax.plot(g.rho_p[s], y + o, mk, color=col, ms=3.2, mew=0)
    ax.axvline(0, color="#444444", lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{COH[c]}, {KIND[k]}" for c, k in SHOW], fontsize=6)
    ax.set_xlabel("ρp with independent gold, masked genes", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=6)
    h = [plt.Line2D([], [], color=STYLE[s][0], marker=STYLE[s][1], ls="", ms=3.5, label=STYLE[s][2]) for s in offs]
    ax.legend(handles=h, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.09), ncol=2, handletextpad=0.1, columnspacing=0.8)
    return fig, ax


def depth_panel(kind_rows):
    b = pd.read_csv(RU.RES / "b_rho.csv", low_memory=False)
    b = b[(b.metric == "gold") & (b.set == "sparse_f") & (b.split == "all")]
    xs = {"depth25": 1, "depth10": 2, "depth5": 3}
    xl = {"depth25law": 1, "depth10law": 2}
    fig, axes = plt.subplots(1, len(kind_rows), figsize=(2.2 * len(kind_rows), 1.9), sharey=True)
    for ax, (c, k) in zip(np.atleast_1d(axes), kind_rows):
        g = b[(b.cohort == c) & (b.kind == k)]
        for s, lab in (("D_f", "D"), ("Sdir_f", "Sdir"), ("S_fwd_f", "S_fwd"), ("R_S_f", "R_S_p"), ("B1_f", "B1")):
            col, mk, _ = STYLE[lab]
            rows = g[(g.score == s) & (g.pair == ("R_S_f" if s == "D_f" else s))]
            if rows.empty:
                continue
            pts = sorted((xs[r.setting], r.rho_p, r.ci_lo, r.ci_hi) for r in rows.itertuples() if r.setting in xs)
            if pts:
                p = np.array(pts)
                ax.fill_between(p[:, 0], p[:, 2], p[:, 3], color=col, alpha=0.12, lw=0)
                ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=1.0, marker=mk, ms=3)
            if lab == "D":
                law = sorted((xl[r.setting] + 0.08, r.rho_p, r.ci_lo, r.ci_hi) for r in rows.itertuples() if r.setting in xl)
                for x, v, lo, hi in law:
                    ax.plot([x, x], [lo, hi], color=col, lw=0.7)
                    ax.plot(x, v, mk, mfc="white", mec=col, ms=3.2, mew=0.8)
        ax.axhline(0, color="#444444", lw=0.6)
        ax.set_xticks([1, 2, 3], ["25%", "10%", "5%"], fontsize=6)
        ax.set_title(f"{COH[c].replace(' species', '-sp')}", fontsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=6)
    np.atleast_1d(axes)[0].set_ylabel("ρp, pattern gold, newly sparse", fontsize=6.5)
    h = [plt.Line2D([], [], color=STYLE[s][0], marker=STYLE[s][1], ls="-", lw=1, ms=3.5, label=STYLE[s][2]) for s in ("D", "Sdir", "S_fwd", "R_S_p", "B1")]
    h.append(plt.Line2D([], [], color=D_C, marker="o", mfc="white", ls="", ms=3.5, label="Δα, anchor scaled by depth"))
    fig.legend(handles=h, fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=6, handletextpad=0.3, columnspacing=0.9)
    return fig, axes


def heldout_panel():
    """Held-out chromosomes, arm P (protein columns only): agreement with what was measured on those genes, given C; folds A and B as two marks."""
    c = pd.read_csv(RU.RES / "c_rho.csv")
    c = c[c.set == "held_unmasked"]
    tg = [("D_full", "measured Δα"), ("S_full", "measured S"), ("Sdir_full", "measured Sdir"), ("GOLD", "pattern gold")]
    sc = [("D_P", D_C, "o", "Chimera gate, protein only"), ("Rh_S", R_C, "D", "Ridge on protein, target S"), ("kNNh_D", GREY, "v", "Nearest proteins, their Δα"), ("pi", B1_C, "s", "π")]
    coh = [k for k in RU.DEPTH_COHORTS if k in set(c.cohort)]
    fig, axes = plt.subplots(1, len(coh), figsize=(1.1 * len(coh), 2.3), sharey=True)
    ceil = pd.read_csv(RU.RES / "c_ceiling.csv")
    for ax, k in zip(np.atleast_1d(axes), coh):
        for j, (t, _) in enumerate(tg):
            for i, (s, col, mk, _) in enumerate(sc):
                tt = ("G_H4" if k.startswith("test") else "G_H3") if t == "GOLD" else t
                g = c[(c.cohort == k) & (c.target == tt) & (c.score == s)]
                for f, dx in (("A", -0.05), ("B", 0.05)):
                    r = g[g.fold == f]
                    if len(r):
                        x = j + (i - 1.5) * 0.18 + dx
                        ax.plot([x, x], [r.ci_lo.iloc[0], r.ci_hi.iloc[0]], color=col, lw=0.7)
                        ax.plot(x, r.rho_p.iloc[0], mk, color=col, ms=2.8, mew=0, alpha=1.0 if f == "A" else 0.6)
        cl = ceil[(ceil.cohort == k) & (ceil.set == "held_unmasked")].ceil_partial.mean()
        ax.plot([-0.4, 0.4], [cl, cl], color="#222222", lw=0.6, ls=":")
        ax.axhline(0, color="#444444", lw=0.6)
        ax.set_xticks(range(len(tg)), [x[1] for x in tg], rotation=40, ha="right", fontsize=6)
        ax.set_title(COH[k].replace(" species", "-sp").replace("Liver ", "Liver\n").replace("Testis ", "Testis\n"), fontsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=6)
    np.atleast_1d(axes)[0].set_ylabel("ρp on held-out genes, given C", fontsize=6.5)
    h = [plt.Line2D([], [], color=col, marker=mk, ls="", ms=3.5, label=lab) for _, col, mk, lab in sc]
    h.append(plt.Line2D([], [], color="#222222", ls=":", lw=0.8, label="seed against other seeds"))
    fig.legend(handles=h, fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.1, columnspacing=0.8)
    return fig, axes


def main():
    cm.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, _ = masked_panel("d1_masked")
    fig.savefig(OUT / "d1_masked.pdf", bbox_inches="tight")
    fig.savefig(OUT / "d1_masked.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    if (RU.RES / "b_rho.csv").exists():
        fig, _ = depth_panel([("liv_hmouse", "pattern"), ("liv_4sp", "pattern"), ("test_hmouse", "pattern")])
        fig.savefig(OUT / "d1_depth.pdf", bbox_inches="tight")
        fig.savefig(OUT / "d1_depth.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    if (RU.RES / "c_rho.csv").exists():
        fig, _ = heldout_panel()
        fig.savefig(OUT / "d1_heldout.pdf", bbox_inches="tight")
        fig.savefig(OUT / "d1_heldout.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
