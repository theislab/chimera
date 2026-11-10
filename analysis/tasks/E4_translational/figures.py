"""E4 candidate panels (exploratory), house style of H1/E1. Panels to results/analysis/E4/panels/; composed with specs_e4.py.

    PYTHONPATH=$PWD mamba run -n chimera_env python -m analysis.tasks.E4_translational.figures
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from analysis.tasks.E4_translational import rules as RU  # noqa: E402
from analysis.tasks.H1_harmonise.figures import TISSUE_LABEL, panel_rc  # noqa: E402
from analysis.tasks.T8_4_figures import common as cm  # noqa: E402

PANELS = RU.OUT / "panels"
UP, DOWN, GREY = "#D55E00", "#0072B2", "#BBBBBB"
RCOL = {"D_can": "#000000", "D_free": "#E69F00", "S": "#56B4E9", "B1": "#999999", "pi_raw": "#CC79A7"}
RLAB = {"D_can": "Δα, canonical gate", "D_free": "Δα, per-gene gate", "S": "S_g", "B1": "pseudobulk profile corr.", "pi_raw": "protein identity π"}
TIS = RU.TISSUES + ["skeletal muscle"]


def save(fig, name):
    PANELS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PANELS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(PANELS / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def cohort_of(t):
    return RU.HM_THREAD.get(t) or RU.HM_EXTRA.get(t)


def p_heat():
    """Set z of each tissue's disease genes and indicated targets in every human-mouse tissue, three read-outs."""
    Z = pd.read_csv(RU.OUT / "q23_sets.csv")
    Z = Z[(Z.strata == "pi_expr") & Z.role.isin(["thread", "hm_extra"])]
    C = pd.read_csv(RU.OUT / "q23_contrast.csv")
    rows = [(t, f"own:{t}", f"{TISSUE_LABEL[t]}: disease genes") for t in TIS] + \
           [(t, f"target:ind_{t}", f"{TISSUE_LABEL[t]}: drug targets") for t in TIS if t != "testis"]
    rows = [r for r in rows if ((Z.set == r[1]) & (Z.tissue == r[0])).any()]
    nr, nc = len(rows), len(TIS)
    rh, wl, wc = 0.118, 1.45, 0.15
    blocks = (("D_can", "canonical gate"), ("D_free", "per-gene gate"), ("B1", "pseudobulk profile\ncorrelation"))
    W = wl + 3 * nc * wc + 2 * 0.08
    top, bot = 0.72, 0.05
    H = rh * nr + top + bot
    fig = plt.figure(figsize=(W, H))
    vmax = 6.0
    for b, (r_, title) in enumerate(blocks):
        ax = fig.add_axes([(wl + b * (nc * wc + 0.08)) / W, bot / H, nc * wc / W, rh * nr / H])
        M = np.full((nr, nc), np.nan)
        for i, (own, k, _) in enumerate(rows):
            for j, t in enumerate(TIS):
                v = Z[(Z.readout == r_) & (Z.set == k) & (Z.tissue == t)].z
                if len(v):
                    M[i, j] = v.iloc[0]
        ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
        for i, (own, k, _) in enumerate(rows):
            j = TIS.index(own)
            cc = C[(C.set == k) & (C.readout == r_)]
            hit = len(cc) and (cc.q.iloc[0] < 0.05) and (cc.contrast.iloc[0] > 0)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.0 if hit else 0.5, ec="#222222", ls="-" if hit else ":"))
        ax.set_xticks(range(nc))
        ax.set_xticklabels([TISSUE_LABEL[x] for x in TIS], fontsize=5.3, rotation=40, ha="left", rotation_mode="anchor")
        ax.xaxis.tick_top()
        ax.set_yticks(range(nr))
        ax.set_yticklabels([r[2] for r in rows] if b == 0 else [], fontsize=5.2)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(len([r for r in rows if r[1].startswith("own:")]) - 0.5, color="#ffffff", lw=2.0)
        ax.text((nc - 1) / 2, -4.4, title, ha="center", va="bottom", fontsize=5.8, color=cm.INK)
    cax = fig.add_axes([0.2 / W, (H - 0.2) / H, 0.95 / W, 0.05 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap="RdBu_r"), cax=cax, orientation="horizontal")
    cb.set_ticks([-vmax, 0, vmax])
    cb.ax.tick_params(labelsize=5, length=1.5)
    cb.outline.set_linewidth(0.4)
    cax.set_title("set z vs identity- and\nexpression-matched genes", fontsize=5.0, pad=1.5)
    save(fig, "e4_heat")


def p_partial():
    """Own-tissue partial effect (given identity, expression, detection, tau, PubMed) for disease genes and indicated targets."""
    P = pd.read_csv(RU.OUT / "q23_partial.csv")
    P = P[P.role.isin(["thread", "hm_extra"])]
    rows = []
    for t in TIS:
        for k, lab in ((f"own:{t}", "disease genes"), (f"target:ind_{t}", "drug targets")):
            x = P[(P.set == k) & (P.tissue == t)]
            if len(x) and x.n.iloc[0] >= 15:
                rows.append((t, k, f"{TISSUE_LABEL[t]}, {lab} ({x.n.iloc[0]})"))
    fig, ax = plt.subplots(figsize=(2.55, 0.155 * len(rows) + 0.45))
    rds = ["D_can", "D_free", "B1"]
    off = {"D_can": -0.22, "D_free": 0.0, "B1": 0.22}
    for i, (t, k, lab) in enumerate(rows):
        for r in rds:
            x = P[(P.set == k) & (P.tissue == t) & (P.readout == r)]
            if not len(x):
                continue
            x = x.iloc[0]
            sig = (x.lo > 0) or (x.hi < 0)
            ax.plot([x.lo, x.hi], [i + off[r]] * 2, color=RCOL[r], lw=0.7)
            ax.plot(x.beta, i + off[r], "o", ms=2.6, mfc=RCOL[r] if sig else "white", mec=RCOL[r], mew=0.6)
    ax.axvline(0, color=cm.RULE, lw=0.6)
    ax.set_xlim(-0.8, 1.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in rows], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel("effect in own tissue, SD of the read-out\n(given identity, expression, detection, τ, PubMed)", fontsize=6.8)
    ax.tick_params(labelsize=6.6, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h = [Line2D([], [], color=RCOL[r], marker="o", ms=2.6, lw=0.7, label=RLAB[r]) for r in rds]
    ax.legend(handles=h, fontsize=6.4, frameon=False, loc="lower right", bbox_to_anchor=(1.02, 1.0), ncol=1)
    save(fig, "e4_partial")


EXAMPLES = {  # tissue: {name: label offset in points}; chosen after reading, from robust calls (labelled post hoc)
    "bone marrow": {"IFNGR1": (6, 2), "IL6R": (8, -2), "IL4R": (8, -6), "CD86": (14, 4), "EGLN2": (4, -2), "PSMA2": (-22, -6)},
    "cortex": {"PDE10A": (3, -6), "FAAH": (3, -5), "KCNQ1": (3, -6), "SCN8A": (-18, 7), "GABRA1": (-26, -9), "OPRM1": (-10, 8)},
    "eye": {"ROCK2": (-22, -3), "ROCK1": (-20, 3), "CA12": (3, 3), "PLAT": (3, 3), "GSK3B": (-22, -4)},
    "kidney": {"SLC22A12": (4, 3), "SCNN1G": (4, 4), "SLC9A3": (4, -6), "SERPING1": (4, 1)},
}


def p_targets():
    T = pd.read_parquet(RU.OUT / "tables" / "cohorts.parquet")
    tg = pd.read_parquet(RU.OUT / "annot" / "targets.parquet").set_index("hsapiens_gene_id")
    fig, axs = plt.subplots(1, 4, figsize=(6.6, 1.75), gridspec_kw=dict(wspace=0.32))
    for ax, (t, names) in zip(axs, EXAMPLES.items()):
        u = T[T.cohort == cohort_of(t)].copy()
        u["pi_pct"] = u.pi_raw.rank(pct=True)
        a = u[u.hsapiens_gene_id.isin(tg.index[tg.approved])]
        ax.scatter(u.pi_pct, u.int_free, s=0.6, c="#E3E3E3", lw=0, rasterized=True)
        col = np.where(a.call == 1, UP, np.where(a.call == -1, DOWN, "#8C8C8C"))
        ind = a.hsapiens_gene_id.map(tg[f"ind_{t}"]).fillna(False).to_numpy(bool)
        ax.scatter(a.pi_pct[~ind], a.int_free[~ind], s=3.5, c=col[~ind], lw=0, alpha=0.8)
        ax.scatter(a.pi_pct[ind], a.int_free[ind], s=9, c=col[ind], marker="D", lw=0.3, edgecolors="k")
        for _, r in a[a.symbol.isin(list(names)) & (a.call != 0)].iterrows():
            ax.annotate(r.symbol, (r.pi_pct, r.int_free), fontsize=4.8, xytext=names[r.symbol], textcoords="offset points",
                        color=UP if r.call > 0 else DOWN, arrowprops=dict(arrowstyle="-", lw=0.3, color="#555555", shrinkA=0, shrinkB=1))
        ax.axhline(0, color=cm.RULE, lw=0.5)
        ax.set_title(f"{TISSUE_LABEL[t]} (human–mouse)", fontsize=5.8, pad=2)
        ax.set_xlabel("protein identity, percentile", fontsize=5.4)
        if ax is axs[0]:
            ax.set_ylabel("Δα, per-gene gate (rank-normal)", fontsize=5.4)
        ax.tick_params(labelsize=5.0, length=1.5)
        ax.set_ylim(-3.6, 3.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h = [Line2D([], [], ls="", marker="o", ms=2.5, color=UP, label="approved target, above the prior on both gates"),
         Line2D([], [], ls="", marker="o", ms=2.5, color=DOWN, label="below the prior on both gates"),
         Line2D([], [], ls="", marker="o", ms=2.5, color="#8C8C8C", label="no call"),
         Line2D([], [], ls="", marker="D", ms=3, mfc="white", mec="k", mew=0.4, label="indicated in this organ system"),
         Line2D([], [], ls="", marker="o", ms=2.5, color="#E3E3E3", label="all count-supported genes")]
    fig.legend(handles=h, fontsize=5.0, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=5)
    save(fig, "e4_targets")


def p_q1():
    Q = pd.read_csv(RU.OUT / "q1_concordance.csv")
    Q = Q[~Q.readout.str.startswith("covariate")]
    groups = [("Y_mgi", False, "all disease genes (MGI)"), ("Y_mgi", True, "organ-matched (MGI)"), ("Y_impc", False, "IMPC Phenodigm")]
    rds = ["D_can", "D_free", "S", "B1", "pi_raw"]
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    for gi, (o, org, lab) in enumerate(groups):
        sub = Q[(Q.outcome == o) & (Q.organ_matched == org)]
        for ri, r in enumerate(rds):
            x = sub[sub.readout == r]
            if not len(x):
                continue
            x = x.iloc[0]
            y = gi * (len(rds) + 1.6) + ri
            sig = x.OR_lo > 1 or x.OR_hi < 1
            ax.plot([x.OR_lo, x.OR_hi], [y, y], color=RCOL[r], lw=0.7)
            ax.plot(x.OR, y, "o", ms=2.6, mfc=RCOL[r] if sig else "white", mec=RCOL[r], mew=0.6)
        if len(sub):
            ax.text(0.5, gi * (len(rds) + 1.6) - 1.1, f"{lab}: {int(sub.n_pos.max())} of {int(sub.n_genes.max())} genes modelled",
                    fontsize=4.9, ha="center", color=cm.INK, transform=ax.get_yaxis_transform())
    ax.axvline(1, color=cm.RULE, lw=0.6)
    ax.set_xscale("log")
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("odds ratio per SD that the mouse mutant models\nthe same human disease (given covariates)", fontsize=5.4)
    ax.tick_params(labelsize=5.0, length=1.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    h = [Line2D([], [], color=RCOL[r], marker="o", ms=2.5, lw=0.7, label=RLAB[r]) for r in rds]
    ax.legend(handles=h, fontsize=4.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    save(fig, "e4_q1")


def main() -> int:
    cm.setup()
    panel_rc()
    p_heat()
    p_partial()
    p_targets()
    if (RU.OUT / "q1_concordance.csv").exists():
        p_q1()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
