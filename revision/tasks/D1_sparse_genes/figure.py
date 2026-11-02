"""D1 (EXPLORATORY): panels of one figure from the D1 result tables; composed through F1_figures.compose with specs_d1.py.
    python -m revision.tasks.D1_sparse_genes.figure
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from revision.tasks.D1_sparse_genes import rules as RU  # noqa: E402
from revision.tasks.T8_4_figures import common as cm  # noqa: E402

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
    fig, ax = plt.subplots(figsize=(3.3, 3.5))
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
    ax.legend(handles=h, fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, 1.09), ncol=2, handletextpad=0.1, columnspacing=0.8)
    return fig, ax


def depth_panel(kind_rows):
    b = pd.read_csv(RU.RES / "b_rho.csv", low_memory=False)
    b = b[(b.metric == "gold") & (b.set == "sparse_f") & (b.split == "all")]
    xs = {"depth25": 1, "depth10": 2, "depth5": 3}
    fig, axes = plt.subplots(1, len(kind_rows), figsize=(1.55 * len(kind_rows), 1.9), sharey=True)
    for ax, (c, k) in zip(np.atleast_1d(axes), kind_rows):
        g = b[(b.cohort == c) & (b.kind == k)]
        for s, lab in (("D_f", "D"), ("Sdir_f", "Sdir"), ("S_fwd_f", "S_fwd"), ("R_S_f", "R_S_p"), ("B1_f", "B1")):
            col, mk, _ = STYLE[lab]
            rows = g[(g.score == s) & (g.pair == ("R_S_f" if s == "D_f" else s))]
            if rows.empty:
                continue
            pts = sorted((xs[r.setting], r.rho_p, r.ci_lo, r.ci_hi) for r in rows.itertuples())
            p = np.array(pts)
            ax.fill_between(p[:, 0], p[:, 2], p[:, 3], color=col, alpha=0.12, lw=0)
            ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=1.0, marker=mk, ms=3)
        ax.axhline(0, color="#444444", lw=0.6)
        ax.set_xticks([1, 2, 3], ["25%", "10%", "5%"], fontsize=6)
        ax.set_title(f"{COH[c]}, {KIND.get(k, k)}", fontsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=6)
    np.atleast_1d(axes)[0].set_ylabel("ρp with gold, newly sparse genes", fontsize=6.5)
    return fig, axes


def heldout_panel():
    c = pd.read_csv(RU.RES / "c_rho.csv")
    c = c[c.set == "held_unmasked"]
    rows = [("D_full", "measured Δα"), ("S_full", "measured S"), ("Sdir_full", "measured Sdir")]
    return c, rows


def main():
    cm.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, _ = masked_panel("d1_masked")
    fig.savefig(OUT / "d1_masked.pdf", bbox_inches="tight")
    fig.savefig(OUT / "d1_masked.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
