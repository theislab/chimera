"""E4 panels (exploratory): disease genes and drug targets against the protein prior. Four panels of two figures, saved
through figures/style.py to figures/panels/<figure>/; reads the result tables of results/analysis/E4 only.

Every panel is drawn at its size on the page and saved at that exact size, so the figures compose at scale 1:
supp_disease_genes (heat, partial, q1) and supp_drug_targets (targets).

    PYTHONPATH=$PWD python -m analysis.tasks.E4_translational.figures [panel ...]     # every panel, or the named ones, from the cached tables (draw_panels)
    CHIMERA_FIGURES_OUT=<folder> ... the same command                                  # dry run: panels under <folder>/panels
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from analysis.tasks.E4_translational import rules as RU  # noqa: E402
from analysis.tasks.H1_harmonise import rules as H1  # noqa: E402
from figures import style  # noqa: E402

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this task
WIDTH = style.TEXTWIDTH  # bp; a full-width panel
ROW_H = 250  # bp; height of the two panels that share the lower row of supp_disease_genes
S_TITLE, S_LABEL, S_TICK, S_TEXT = style.SIZE_TITLE, style.SIZE_LABEL, style.SIZE_TICK, style.SIZE_TEXT
INK, MUTED, RULE = style.INK, style.MUTED, style.RULE
NA_CELL = "#F2F2F2"  # heat-map cell without a value
UP, DOWN = style.ABOVE, style.BELOW  # called above / below the prior on both gates
NO_CALL, CLOUD = "#9A9A9A", "#E3E3E3"  # approved target without a call; all genes with count evidence
READOUT = {"D_can": style.READOUT["delta_alpha"], "D_free": style.READOUT["delta_alpha_per_gene"], "S": style.READOUT["count_evidence"],
           "B1": style.READOUT["profile_correlation"], "pi_raw": style.READOUT["protein_identity"]}  # keys of the result tables
TIS = RU.TISSUES + ["skeletal muscle"]
TISSUE_NAME = {t: style.TISSUE["muscle" if t == "skeletal muscle" else t].name for t in TIS}
Z_MAX = 6.0  # color limit of the set z; cells beyond it take the end color (the color bar is pointed)


def save(fig, name: str):
    """One panel at its exact size: figures/panels/<figure>/<name>.pdf."""
    return style.save_panel(fig, name, exact=True)


def signed(v, _=None) -> str:
    """Tick label with one decimal and a true minus sign."""
    return f"{v:.1f}".replace("-", "−")


def cohort_of(t):
    return RU.HM_THREAD.get(t) or RU.HM_EXTRA.get(t)


def p_heat():
    """Set z of each tissue's disease genes and indicated targets in every human-mouse tissue, three readouts; the own
    tissue of a set is outlined (solid: own above the other tissues at q < 0.05)."""
    Z = pd.read_csv(RU.OUT / "q23_sets.csv")
    Z = Z[(Z.strata == "pi_expr") & Z.role.isin(["thread", "hm_extra"])]
    C = pd.read_csv(RU.OUT / "q23_contrast.csv")
    rows = [(t, f"own:{t}", f"{TISSUE_NAME[t]}, disease genes") for t in TIS] + \
           [(t, f"target:ind_{t}", f"{TISSUE_NAME[t]}, drug targets") for t in TIS if t != "testis"]
    rows = [r for r in rows if ((Z.set == r[1]) & (Z.tissue == r[0])).any()]
    nr, nc = len(rows), len(TIS)
    blocks = (("D_can", READOUT["D_can"].name), ("D_free", READOUT["D_free"].name), ("B1", READOUT["B1"].name))
    # geometry in bp: row labels, three blocks of nc columns with a gap between them, the tissue names above
    left, gap, right, top, bot, ch = 118.0, 6.0, 3.0, 78.0, 3.0, 10.5
    cw = (WIDTH - left - right - 2 * gap) / (3 * nc)
    H = ch * nr + top + bot
    fig = plt.figure(figsize=(style.inch(WIDTH), style.inch(H)))
    for b, (r_, title) in enumerate(blocks):
        ax = fig.add_axes([(left + b * (nc * cw + gap)) / WIDTH, bot / H, nc * cw / WIDTH, ch * nr / H])
        M = np.full((nr, nc), np.nan)
        for i, (own, k, _) in enumerate(rows):
            for j, t in enumerate(TIS):
                v = Z[(Z.readout == r_) & (Z.set == k) & (Z.tissue == t)].z
                if len(v):
                    M[i, j] = v.iloc[0]
        ax.set_facecolor(NA_CELL)
        ax.imshow(np.ma.masked_invalid(M), cmap=style.DIVERGING, vmin=-Z_MAX, vmax=Z_MAX, aspect="auto", interpolation="nearest")
        for i, (own, k, _) in enumerate(rows):
            j = TIS.index(own)
            cc = C[(C.set == k) & (C.readout == r_)]
            hit = len(cc) and (cc.q.iloc[0] < 0.05) and (cc.contrast.iloc[0] > 0)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.0 if hit else 0.5, ec=INK, ls="-" if hit else ":"))
        ax.set_xticks(range(nc))
        ax.set_xticklabels([TISSUE_NAME[x] for x in TIS], fontsize=S_TICK, rotation=90, ha="center", va="bottom")
        ax.xaxis.tick_top()
        ax.set_yticks(range(nr))
        ax.set_yticklabels([r[2] for r in rows] if b == 0 else [], fontsize=S_TICK)
        ax.tick_params(length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.axhline(len([r for r in rows if r[1].startswith("own:")]) - 0.5, color="#ffffff", lw=2.0)
        ax.set_title(title, fontsize=S_TITLE, pad=3)
    cax = fig.add_axes([16 / WIDTH, (H - 27) / H, 76 / WIDTH, 5 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-Z_MAX, Z_MAX), cmap=style.DIVERGING), cax=cax, orientation="horizontal", extend="both")
    cb.set_ticks([-Z_MAX, 0, Z_MAX])
    cb.ax.tick_params(labelsize=S_TICK, length=1.5)
    cb.outline.set_linewidth(0.4)
    cax.set_title("set z against identity- and\nexpression-matched genes", fontsize=S_TICK, pad=2)
    return save(fig, "supp_disease_genes/heat")


def p_partial():
    """Own-tissue partial effect (given identity, expression, detection, tau, PubMed) for disease genes and indicated targets."""
    P = pd.read_csv(RU.OUT / "q23_partial.csv")
    P = P[P.role.isin(["thread", "hm_extra"])]
    rows = []
    for t in TIS:
        for k, lab in ((f"own:{t}", "disease genes"), (f"target:ind_{t}", "drug targets")):
            x = P[(P.set == k) & (P.tissue == t)]
            if len(x) and x.n.iloc[0] >= 15:
                rows.append((t, k, f"{TISSUE_NAME[t]}, {lab} ({int(x.n.iloc[0]):,})"))
    fig, ax = plt.subplots(figsize=(style.inch(270), style.inch(ROW_H)), layout="constrained")
    rds = ["D_can", "D_free", "B1"]
    off = {"D_can": -0.24, "D_free": 0.0, "B1": 0.24}
    for i, (t, k, lab) in enumerate(rows):
        for r in rds:
            x = P[(P.set == k) & (P.tissue == t) & (P.readout == r)]
            if not len(x):
                continue
            x = x.iloc[0]
            sig = (x.lo > 0) or (x.hi < 0)
            ax.plot([x.lo, x.hi], [i + off[r]] * 2, color=READOUT[r].color, lw=0.8)
            ax.plot(x.beta, i + off[r], READOUT[r].marker, ms=3.2, mfc=READOUT[r].color if sig else "white", mec=READOUT[r].color, mew=0.6)
    ax.axvline(0, color=RULE, lw=0.6, zorder=0)
    ax.set_xlim(-0.8, 1.0)  # holds every interval (lowest -0.72, highest 0.92)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in rows], fontsize=S_TICK)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.xaxis.set_major_formatter(FuncFormatter(signed))
    ax.set_xlabel("effect in own tissue, SD of the readout\n(given identity, expression, detection,\ncell-type specificity τ, PubMed count)", fontsize=S_LABEL)
    ax.tick_params(labelsize=S_TICK, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h = [Line2D([], [], color=READOUT[r].color, marker=READOUT[r].marker, ms=3.2, lw=0.8, label=READOUT[r].name) for r in rds]
    fig.legend(handles=h, fontsize=S_TICK, frameon=False, loc="outside upper right", ncol=1, handletextpad=0.4, borderaxespad=0.2)
    return save(fig, "supp_disease_genes/partial")


# tissue: {gene: center of its label (percentile of identity, normal score)}; example genes with a call on both gates,
# chosen after the results were read. The labels stand in the bands above and below the data, each with a line to its point.
EXAMPLES = {
    "bone marrow": {"IFNGR1": (9, 4.5), "CD86": (31, 3.75), "IL4R": (49, 3.75), "IL6R": (66, 3.75), "EGLN2": (48, -3.75), "PSMA2": (82, -3.75)},
    "cortex": {"OPRM1": (45, 3.75), "GABRA1": (72, 4.5), "SCN8A": (91, 3.75), "PDE10A": (10, -3.75), "FAAH": (33, -3.75), "KCNQ1": (56, -3.75)},
    "eye": {"CA12": (10, 3.75), "PLAT": (30, 3.75), "ROCK1": (80, 3.75), "ROCK2": (58, -3.75), "GSK3B": (88, -3.75)},
    "kidney": {"SERPING1": (14, 4.5), "SLC22A12": (38, 3.75), "SLC9A3": (66, 4.5), "SCNN1G": (74, 3.75)},
}
Y_LIM = 4.95  # holds every gene (normal scores reach 3.92) and the two label bands


def p_targets():
    """Approved drug targets on the per-gene gate against the rank of protein identity, four tissues; the approved targets
    without a call are drawn first and light, the called ones on top."""
    T = pd.read_parquet(RU.OUT / "tables" / "cohorts.parquet")
    tg = pd.read_parquet(RU.OUT / "annot" / "targets.parquet").set_index("hsapiens_gene_id")
    reg = H1.cohorts()
    fig, axs = plt.subplots(1, 4, figsize=(style.inch(WIDTH), style.inch(200)), sharey=True, layout="constrained")
    for ax, (t, names) in zip(axs, EXAMPLES.items()):
        u = T[T.cohort == cohort_of(t)].copy()
        u["pi_pct"] = 100 * u.pi_raw.rank(pct=True)
        a = u[u.hsapiens_gene_id.isin(tg.index[tg.approved])]
        ax.scatter(u.pi_pct, u.int_free, s=0.6, c=CLOUD, lw=0, rasterized=True)
        col = np.where(a.call == 1, UP, np.where(a.call == -1, DOWN, NO_CALL))
        ind = a.hsapiens_gene_id.map(tg[f"ind_{t}"]).fillna(False).to_numpy(bool)
        called = (a.call != 0).to_numpy()
        for c_, z in ((False, 2), (True, 3)):
            m = (called == c_) & ~ind
            ax.scatter(a.pi_pct[m], a.int_free[m], s=3.5 if not c_ else 5, c=col[m], lw=0, alpha=0.8 if not c_ else 0.95, zorder=z)
            m = (called == c_) & ind
            ax.scatter(a.pi_pct[m], a.int_free[m], s=9, c=col[m], marker="D", lw=0.25 if not c_ else 0.4, edgecolors="#6F6F6F" if not c_ else INK, zorder=z + 0.5)
        for _, r in a[a.symbol.isin(list(names)) & (a.call != 0)].iterrows():
            ax.annotate(r.symbol, (r.pi_pct, r.int_free), fontsize=S_TEXT, xytext=names[r.symbol], textcoords="data", ha="center", va="center", zorder=5,
                        color=UP if r.call > 0 else DOWN, arrowprops=dict(arrowstyle="-", lw=0.35, color="#555555", shrinkA=0.5, shrinkB=1.5))
        ax.axhline(0, color=RULE, lw=0.5, zorder=1)
        ax.set_title(style.cohort_name(reg[cohort_of(t)]["label"]), fontsize=S_TITLE, pad=3)
        ax.set_xlabel("protein identity, percentile", fontsize=S_LABEL)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlim(-3, 103)
        ax.tick_params(labelsize=S_TICK, length=2)
        ax.set_ylim(-Y_LIM, Y_LIM)
        ax.set_yticks([-4, -2, 0, 2, 4])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel(f"{READOUT['D_free'].name} (normal score)", fontsize=S_LABEL)
    h = [Line2D([], [], ls="", marker="o", ms=3, color=UP, label="approved target, above the prior on both gates"),
         Line2D([], [], ls="", marker="o", ms=3, color=DOWN, label="approved target, below the prior on both gates"),
         Line2D([], [], ls="", marker="o", ms=3, color=NO_CALL, label="approved target, no call"),
         Line2D([], [], ls="", marker="D", ms=3.2, mfc="white", mec=INK, mew=0.4, label="indicated for this organ system"),
         Line2D([], [], ls="", marker="o", ms=3, color=CLOUD, label="all genes with count evidence")]
    fig.legend(handles=h, fontsize=S_TICK, frameon=False, loc="outside upper center", ncol=3, handletextpad=0.3, columnspacing=1.2)
    return save(fig, "supp_drug_targets/targets")


def p_q1():
    """Odds ratio per SD of each readout that the mouse mutant of a human disease gene models the same disease, three outcomes."""
    Q = pd.read_csv(RU.OUT / "q1_concordance.csv")
    Q = Q[~Q.readout.str.startswith("covariate")]
    groups = [("Y_mgi", False, "all disease genes (MGI)"), ("Y_mgi", True, "organ-matched (MGI)"), ("Y_impc", False, "IMPC Phenodigm")]
    rds = ["D_can", "D_free", "S", "B1", "pi_raw"]
    step = len(rds) + 1.9
    fig, ax = plt.subplots(figsize=(style.inch(209), style.inch(ROW_H)), layout="constrained")
    for gi, (o, org, lab) in enumerate(groups):
        sub = Q[(Q.outcome == o) & (Q.organ_matched == org)]
        for ri, r in enumerate(rds):
            x = sub[sub.readout == r]
            if not len(x):
                continue
            x = x.iloc[0]
            y = gi * step + ri
            sig = x.OR_lo > 1 or x.OR_hi < 1
            ax.plot([x.OR_lo, x.OR_hi], [y, y], color=READOUT[r].color, lw=0.8)
            ax.plot(x.OR, y, READOUT[r].marker, ms=3.2, mfc=READOUT[r].color if sig else "white", mec=READOUT[r].color, mew=0.6)
        if len(sub):
            ax.plot([1, 1], [gi * step - 0.55, gi * step + len(rds) - 0.45], color=RULE, lw=0.6, zorder=0)
            noun = "" if "genes" in lab else "genes "
            ax.text(0.0, gi * step - 1.2, f"{lab}: {int(sub.n_pos.max()):,} of {int(sub.n_genes.max()):,} {noun}modeled in mouse",
                    fontsize=S_TEXT, ha="left", va="center", color=INK, transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlim(0.74, 1.2)  # holds every interval (lowest 0.763, highest 1.168)
    ax.set_xticks([0.8, 0.9, 1.0, 1.1])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.minorticks_off()
    ax.set_yticks([])
    ax.set_ylim((len(groups) - 1) * step + len(rds) - 0.4, -1.9)
    ax.set_xlabel("odds ratio per SD that the mouse mutant models\nthe same human disease (given covariates)", fontsize=S_LABEL)
    ax.tick_params(labelsize=S_TICK, length=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    h = [Line2D([], [], color=READOUT[r].color, marker=READOUT[r].marker, ms=3.2, lw=0.8, label=READOUT[r].name) for r in rds]
    fig.legend(handles=h, fontsize=S_TICK, frameon=False, loc="outside lower center", ncol=2, handletextpad=0.4, columnspacing=1.0)
    return save(fig, "supp_disease_genes/q1")


# panel name in figures/panels -> the function that draws it
DRAW = {"supp_disease_genes/heat": p_heat, "supp_disease_genes/partial": p_partial, "supp_disease_genes/q1": p_q1,
        "supp_drug_targets/targets": p_targets}


def draw_panels(names=None) -> list:
    """Draw the panels of this task from the cached tables of results/analysis/E4 and nothing else. ``names``: panel names
    as in ``DRAW`` (default: all four). Returns the files written."""
    style.apply()
    return [DRAW[n]() for n in (DRAW if names is None else names)]


def main() -> int:
    for path in draw_panels(sys.argv[1:] or None):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
