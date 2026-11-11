"""E4 panels (exploratory): disease genes and drug targets against the protein prior. Four panels of two figures, saved
through figures/style.py to figures/panels/<figure>/; reads the result tables of results/analysis/E4 only.

Every panel is drawn at its size on the page and saved at that exact size, so the figures compose at scale 1:
supp_disease_genes (heat, partial, q1) and the main figure on drug targets, main_drug_targets (targets; the panel
file is figures/panels/supp_drug_targets/targets.pdf).

    PYTHONPATH=$PWD python -m analysis.tasks.E4_translational.figures [panel ...]     # every panel, or the named ones, from the cached tables (draw_panels)
    CHIMERA_FIGURES_OUT=<folder> ... the same command                                  # dry run: panels under <folder>/panels
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

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
            # as in the figure of programs by tissue: solid where the own tissue stands above the others, dashed otherwise,
            # on a white line that keeps the outline visible on dark cells
            lw = 1.1 if hit else 0.7
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=lw + 0.9, ec="white", clip_on=False))
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=lw, ec=INK, ls="-" if hit else (0, (2, 1.3)), clip_on=False))
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
# Approved targets above and below the prior under both gates, and the numbers expected from genes of equal identity and
# expression, over the nine models with human and mouse together, as the text gives them.
PRINTED_TOTALS = {"above": 284, "exp_above": 181, "below": 58, "exp_below": 90}


def target_counts() -> pd.DataFrame:
    """Per model: the number of approved targets (``n``), how many are called above and below the prior under both gates
    (``above``, ``below``) and how many are expected from the genes of their strata of identity and expression
    (``exp_above``, ``exp_below``). Rows of the set ``target:approved`` of ``results/analysis/E4/q23_callshare.csv``.

    Checked before anything is drawn: summed over the nine models with human and mouse, the stored counts are those the
    text prints (``PRINTED_TOTALS``) and those of the test over all models (``callshare_perm.csv``)."""
    q = pd.read_csv(RU.OUT / "q23_callshare.csv")
    q = q[q.set == "target:approved"].set_index("cohort")
    nine = q.loc[[*RU.HM_THREAD.values(), RU.HM_EXTRA["skeletal muscle"]]]
    total = {k: int(round(float(nine[k].sum()))) for k in PRINTED_TOTALS}
    assert total == PRINTED_TOTALS, f"approved targets over the nine models: {total}; the text prints {PRINTED_TOTALS}"
    pooled = pd.read_csv(RU.OUT / "callshare_perm.csv").set_index("group").loc["approved targets"]
    assert {k: int(round(float(pooled[k]))) for k in PRINTED_TOTALS} == PRINTED_TOTALS and int(pooled.n) == int(nine.n.sum()), "callshare_perm.csv differs"
    return q


# The figure on drug targets: sizes in bp and pt at print size, and its marks from back to front.
TG = dict(height=205.0, left=23.0, right=5.0, gap=8.0, bottom=20.0, top=42.5, key=17.0,
          s_head=6.5, s_title=6.5, s_tick=6.0, s_count=6.0, s_key=6.0, s_note=5.5,
          spread=3.0)                                                      # the tied genes are drawn over ± this many percentile units
NO_CALL_MARK = dict(marker="o", ms=2.2, mfc="#A0A0A0", mec="none", mew=0)
NO_CALL_INDICATED = dict(marker="D", ms=2.3, mfc="none", mec="#888888", mew=0.5)
CALLED_MARK = dict(marker="o", ms=3.4, mec="white", mew=0.4)               # in the color of the call
CALLED_INDICATED = dict(marker="D", ms=3.4, mec=INK, mew=0.5)              # in the color of the call
TIED_NOTE = "upper bound of the prior"


def p_targets():
    """The main figure on drug targets (``supp_drug_targets/targets``, 486 x 205 bp; figure key ``main_drug_targets``).

    What it draws. Four plots, one tissue each (bone marrow, cortex, eye, kidney; the models of ``EXAMPLES``). x: the
    percentile of protein identity among the genes of the model; y: Δα with the per-gene gate as a normal score. Light
    gray dots: every gene with count evidence. On top, the targets of approved drugs, the marks of the finding last and
    largest: small gray dots, targets without a call (an open diamond where the drug is indicated for a disease of
    that organ system); vermilion and blue, targets called above and below the prior under both gates (a diamond with
    a dark edge where indicated). The named genes and the places of their names are those of ``EXAMPLES``. Under the
    name of each tissue stand the numbers of approved targets called above and below the prior, each with the number
    expected from genes of equal identity and expression (``target_counts``): the comparison that the title of the
    figure states.

    The genes at the upper bound of the prior have one protein identity and so one percentile. They are drawn over
    ``TG["spread"]`` percentile units to either side of it, in the order of their gene identifiers (a place that says
    nothing about Δα), so that their marks do not print on each other; the first plot names the band (``TIED_NOTE``).

    Reads ``results/analysis/E4/tables/cohorts.parquet`` (columns ``cohort``, ``hsapiens_gene_id``, ``symbol``,
    ``pi_raw``, ``int_free``, ``call``), ``results/analysis/E4/annot/targets.parquet`` (``hsapiens_gene_id``,
    ``approved``, ``ind_<tissue>``) and, for the printed counts, ``results/analysis/E4/q23_callshare.csv`` and
    ``callshare_perm.csv``. The only arithmetic is the percentile rank of ``pi_raw`` within a model, for the x axis; the
    printed counts are stored values, and the panel stops if they are not the counts of the points it draws.

        python -m analysis.tasks.E4_translational.figures supp_drug_targets/targets          # in chimera_env, from the repository root
    """
    g = SimpleNamespace(**TG)
    T = pd.read_parquet(RU.OUT / "tables" / "cohorts.parquet")
    tg = pd.read_parquet(RU.OUT / "annot" / "targets.parquet").set_index("hsapiens_gene_id")
    reg = H1.cohorts()
    counts = target_counts()
    W, H = WIDTH, g.height
    w, h = (W - g.left - g.right - 3 * g.gap) / 4, g.height - g.bottom - g.top
    fig = plt.figure(figsize=(style.inch(W), style.inch(H)))
    axs = []
    for k, (t, names) in enumerate(EXAMPLES.items()):
        x0 = g.left + k * (w + g.gap)
        ax = fig.add_axes([x0 / W, g.bottom / H, w / W, h / H])
        axs.append(ax)
        u = T[T.cohort == cohort_of(t)].copy()
        u["pi_pct"] = 100 * u.pi_raw.rank(pct=True)
        tied = (u.pi_raw == u.pi_raw.max()).to_numpy()                     # the genes at the upper bound of the prior
        assert tied.sum() > 0.05 * len(u) and u.pi_pct[tied].nunique() == 1, f"{t}: no group of genes at the upper bound of the prior"
        at = float(u.pi_pct[tied].iloc[0])
        order = u.hsapiens_gene_id[tied].rank(method="first").to_numpy()   # a place within the band that says nothing about Δα
        u.loc[tied, "pi_pct"] = at + g.spread * (2 * (order - 1) / max(tied.sum() - 1, 1) - 1)
        a = u[u.hsapiens_gene_id.isin(tg.index[tg.approved])]
        q = counts.loc[cohort_of(t)]
        drawn = (len(a), int((a.call == 1).sum()), int((a.call == -1).sum()))
        assert drawn == (int(q.n), int(q.above), int(q.below)), f"{t}: {drawn} approved targets drawn (all, above, below); stored {int(q.n)}, {int(q.above)}, {int(q.below)}"
        ax.scatter(u.pi_pct, u.int_free, s=0.6, c=CLOUD, lw=0, rasterized=True, zorder=1)
        ind = a.hsapiens_gene_id.map(tg[f"ind_{t}"]).fillna(False).to_numpy(bool)
        called = (a.call != 0).to_numpy()
        color = np.where(a.call == 1, UP, DOWN)
        ax.plot(a.pi_pct[~called & ~ind], a.int_free[~called & ~ind], ls="none", zorder=2, **NO_CALL_MARK)
        ax.plot(a.pi_pct[~called & ind], a.int_free[~called & ind], ls="none", zorder=2.5, **NO_CALL_INDICATED)
        for c_ in (DOWN, UP):                                              # the marks of the finding last
            m = called & (color == c_)
            ax.plot(a.pi_pct[m & ~ind], a.int_free[m & ~ind], ls="none", mfc=c_, zorder=3, **CALLED_MARK)
            ax.plot(a.pi_pct[m & ind], a.int_free[m & ind], ls="none", mfc=c_, zorder=3.5, **CALLED_INDICATED)
        for _, r in a[a.symbol.isin(list(names)) & (a.call != 0)].iterrows():
            ax.annotate(r.symbol, (r.pi_pct, r.int_free), fontsize=g.s_note, xytext=names[r.symbol], textcoords="data", ha="center", va="center", zorder=5,
                        color=UP if r.call > 0 else DOWN, arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED, shrinkA=0.5, shrinkB=2.0))
        ax.axhline(0, color=RULE, lw=0.6, zorder=0.5)
        ax.set_xticks([0, 50, 100])
        ax.set_xlim(-3, 103)
        ax.set_ylim(-Y_LIM, Y_LIM)
        ax.set_yticks([-4, -2, 0, 2, 4], ["−4", "−2", "0", "+2", "+4"] if k == 0 else [])
        ax.tick_params(labelsize=g.s_tick, length=2.5, width=0.5, pad=1.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.6)
        # the tissue, and under it the two counts with what is expected: left edge on the axes
        y0 = g.bottom + h
        fig.text(x0 / W, (y0 + 18.2) / H, style.cohort_name(reg[cohort_of(t)]["label"]), ha="left", va="baseline", fontsize=g.s_head, fontweight="bold", color=INK)
        for line, (word, n_, e_, c_) in enumerate((("above", q.above, q.exp_above, UP), ("below", q.below, q.exp_below, DOWN))):
            fig.text(x0 / W, (y0 + 10.4 - 7.2 * line) / H, f"{word}: {int(n_)} ({e_:.0f} expected)", ha="left", va="baseline", fontsize=g.s_count, color=c_)
        if k == 0:  # the band of the tied genes, named once, beside it
            ax.text(at + g.spread + 2.0, 0.6, TIED_NOTE, rotation=90, ha="left", va="center", fontsize=g.s_note, color=MUTED, zorder=5)
    fig.text((g.left + (W - g.right - g.left) / 2) / W, 2.4 / H, "protein identity, percentile", ha="center", va="baseline", fontsize=g.s_title, color=INK)
    fig.text(5.0 / W, (g.bottom + h / 2) / H, f"{READOUT['D_free'].name} (normal score)", rotation=90, ha="center", va="center", fontsize=g.s_title, color=INK)
    hd = [Line2D([], [], ls="", mfc=UP, label="approved target, above the prior on both gates", **CALLED_MARK),
          Line2D([], [], ls="", mfc=DOWN, label="approved target, below the prior on both gates", **CALLED_MARK),
          Line2D([], [], ls="", label="approved target, no call", **NO_CALL_MARK),
          Line2D([], [], ls="", mfc="white", label="indicated for this organ system", **{**CALLED_INDICATED, "mew": 0.5}),
          Line2D([], [], ls="", marker="o", ms=1.6, mfc=CLOUD, mec="none", label="all genes with count evidence"),
          Line2D([], [], ls="", marker="", label="expected: from genes of equal identity and expression")]
    fig.legend(handles=hd, fontsize=g.s_key, frameon=False, loc="upper left", bbox_to_anchor=(g.left / W, 1 - 1.0 / H), ncol=3, handletextpad=0.4, handlelength=1.0,
               columnspacing=1.4, labelspacing=0.3, borderpad=0, borderaxespad=0)
    return save(fig, "supp_drug_targets/targets")


def p_q1():
    """Odds ratio per SD of each readout that the mouse mutant of a human disease gene models the same disease, three outcomes."""
    Q = pd.read_csv(RU.OUT / "q1_concordance.csv")
    Q = Q[~Q.readout.str.startswith("covariate")]
    groups = [("Y_mgi", False, "MGI, all disease genes"), ("Y_mgi", True, "MGI, organ-matched subset"), ("Y_impc", False, "IMPC Phenodigm")]
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
            ax.text(0.0, gi * step - 1.2, f"{lab}: {int(sub.n_pos.max()):,} of {int(sub.n_genes.max()):,} modeled in mouse",
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
