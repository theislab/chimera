"""Supplementary panels on the difference between the two gates (Δα of the per-gene gate minus Δα of the protein-feature gate), one PDF per panel in
figures/panels/<family>/, composed by figures/compose.py from figures/specs/. Reads results/analysis/X8/ only. House style: figures/style.py; every panel is
drawn at its size on the page and placed at scale 1.

    PYTHONPATH=$PWD python -m analysis.tasks.X8_gate_divergence.figures        # every panel, from the cached tables (draw_panels)
    CHIMERA_FIGURES_OUT=<folder> ... the same command                          # dry run: panels under <folder>/panels
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

from analysis.tasks.X8_gate_divergence import rules as RU  # noqa: E402
from figures import style  # noqa: E402

OUT = RU.OUT

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this task
INK, MUTED = style.INK, style.MUTED
S_TITLE, S_TICK, S_TEXT = style.SIZE_TITLE, style.SIZE_TICK, style.SIZE_TEXT  # axis labels take SIZE_LABEL from style.apply()
GATE, PER_GENE = style.READOUT["delta_alpha"], style.READOUT["delta_alpha_per_gene"]  # Δα of the protein-feature gate and of the per-gene gate
DNDS, LOEUF = style.READOUT["dnds"], style.READOUT["loeuf"]
DIFFERENCE = "difference"  # Δα of the per-gene gate minus Δα of the protein-feature gate, as rank z
DIFFERENCE_LONG = "difference of Δα, per-gene gate minus protein-feature gate"
# The side of the difference is shown by position (above or below the diagonal, right or left of zero) and in the diverging maps; the two gates alone carry
# the colors of style.READOUT.
CALLED, BACKGROUND = INK, style.BACKGROUND_GENES  # genes whose difference passes the seed-noise test, all other genes
BAR, OTHER, REF_LINE = "#555555", "#666666", "#888888"  # bars of shares; a series beside the two gates; zero and identity lines
GUIDE, GROUP_RULE = "#EEEEEE", "#D5D5D5"  # row guides of the cohort lists; rules between their tissues
NA_CELL = "#D9D9D9"  # heat-map cell without a value: darker than the center of the diverging map, so it cannot be read as a value near zero
RASTER_DPI = 400  # rasterized gene clouds: above 300 ppi on the page
W_FULL, GAP = style.TEXTWIDTH, style.GAP
PITCH = 7.2  # bp per row of a list (names at SIZE_TEXT)
LABEL_W = 148.0  # bp left of a cohort list for the cohort names
LIST_TOP = 5.0  # bp above the rows of a list
W_SCATTER = 169.0  # panel of the gene cloud; the cohort panel beside it takes the rest of the row
W_COVARIATES = 197.0  # covariate heat map without row names; the neighborhood panel beside it carries them
BOTTOM_ROW_1, BOTTOM_ROW_2 = 44.0, 70.0  # bp under the rows of the lists of the first and second row of the first figure
W_CROSS, H_ROW_CLASSES = 170.0, 176.0  # box plots across cohorts and the height it shares with the gene-class heat map beside it
W_NAMED = 330.0  # the named-gene figure, narrower than the page and centered

TISSUE_ORDER = ["liver", "testis", "bone marrow", "kidney", "islet", "eye", "cortex", "lung", "skeletal muscle", "tumour myeloid"]  # tissue names of the cohort registry
_TISSUE_KEY = {"skeletal muscle": "muscle", "tumour myeloid": "tumor"}  # tissue names of the registry that differ from the keys of style.TISSUE
TLAB = {t: style.TISSUE[_TISSUE_KEY.get(t, t)].name for t in TISSUE_ORDER}
# Cohorts inside a tissue in the order of the cohort table of the supplement (pairs first, then the larger species sets).
COHORT_ORDER = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp", "test_hmac", "test_hmouse", "test_3sp", "bone_marrow", "kidney", "islet",
                "vz_hmouse", "vz_hmmul", "vz_hmfas", "vz_hpig", "vz_primates", "vz_full",
                "brain_m1_hmouse", "brain_m1", "brain_dlpfc", "brain_dlpfc_hchimp", "brain_dlpfc_hmac", "brain_dlpfc_hmarm", "brain_dlpfc_hmac_own",
                "lung_healthy", "muscle_ageing", "tumour_zilionis"]
_NAME_START = {"Tumor myeloid ": "Lung-tumor myeloid ", "Muscle ": "Skeletal muscle "}  # cohort names whose tissue is written out
COV_NAME = {"expr": "expression level", "detection": "detection", "abs_S_can": "|$S_g$|, protein-feature gate", "abs_S_free": "|$S_g$|, per-gene gate",
            "pi_raw": "protein identity π", "tau": "cell-type specificity τ", "det_gap": "detection gap", "ALL_R2": "joint R²"}
SET_LABEL = {"class|mitochondrial": "Mitochondrial proteins", "class|TF": "Transcription factors", "class|TF_C2H2_ZF": "C2H2 zinc-finger TFs", "class|receptor": "Receptors",
             "broad|Translation & ribosome": "Translation & ribosome", "broad|Protein homeostasis": "Protein homeostasis", "hallmark|E2F Targets": "E2F targets",
             "hallmark|G2-M Checkpoint": "G2-M checkpoint", "hallmark|Myc Targets V1": "Myc targets V1", "hallmark|Myc Targets V2": "Myc targets V2"}
# Independent references: pattern = agreement of cell-type profiles between species in data the models never saw, level = agreement of expression level.
REF_LABEL = {"pattern": "pattern, two atlases", "pattern_primates": "pattern, held-out primates", "level": "level, EVEE", "level_atlas": "level per cell type, two atlases",
             "pattern_fine": "pattern, fine cell types", "pattern_nephron": "pattern, nephron cell types", "pattern_noacinar": "pattern, without acinar cells",
             "pattern_fine_noacinar": "pattern, fine, without acinar cells", "pattern_coe": "pattern, sorted lineages"}
REF_LABEL_TISSUE = {("testis", "pattern"): "pattern, 8 held-out species"}  # the testis pattern reference spans eight species absent from training
# Genes named on the gene cloud of kidney Human+Mouse, each with the place of its name (rank z of the two gates), clear of every point.
SCATTER_GENES = {"FKBP5": (2.55, -0.7), "ARF5": (2.55, -1.2), "PPP2CB": (2.55, -1.7), "MARK2": (2.55, -2.2), "TERF2IP": (2.55, -2.7)}
# Named genes of the last figure: 14 of the 358 genes of results/analysis/X8/named.csv with `clean` true (they pass every check in two or more tissues, with
# one sign wherever they are called), chosen to show the two patterns of the gene classes and one family split. In the order of the rows: six near-identical,
# broadly expressed core proteins that read lower with the per-gene gate than their paralog families; the FKBP family, which the protein embedding groups
# together and the counts separate (FKBP4 and FKBP5 lower, FKBP7 higher); five metabolic and nucleolar enzymes that read higher.
NAMED = ["MARK2", "PPP2CB", "ARF5", "GSK3B", "XRCC6", "TERF2IP", "FKBP4", "FKBP5", "FKBP7", "HADH", "ALDH6A1", "QDPR", "MSRB3", "FBL"]
NAMED_GROUPS = (6, 3, 5)  # rows per group, in the order above


def page(w: float, h: float):
    """A panel page of w x h bp, its size on the page of the paper."""
    return plt.figure(figsize=(style.inch(w), style.inch(h)))


def box(fig, x: float, y: float, w: float, h: float):
    """Axes at a place given in bp from the lower left corner of the page."""
    W, H = fig.get_size_inches() * 72
    return fig.add_axes([x / W, y / H, w / W, h / H])


def save(fig, name: str, raster: bool = False) -> None:
    """One panel at its exact size. ``raster``: the panel holds a rasterized cloud."""
    with plt.rc_context({"savefig.dpi": RASTER_DPI} if raster else {}):
        style.save_panel(fig, name, exact=True)


def signed(v: float, digits: int = 2) -> str:
    """A signed value with a true minus sign."""
    return f"{v:+.{digits}f}".replace("-", "−")


def cohort_name(label: str) -> str:
    """The name of a cohort as a row of a list: tissue first, species set written out."""
    s = style.cohort_name(label).replace(", own table", ", own ortholog table")
    for a, b in _NAME_START.items():
        if s.startswith(a):
            s = b + s[len(a):]
    return s


def cohort_order() -> list[tuple[str, str, str]]:
    """(cohort, tissue, name) of every cohort, in the order of the rows."""
    reg = RU.cohorts()
    rows = [(c, v["tissue"], cohort_name(v["label"])) for c, v in reg.items()]
    rank = {c: i for i, c in enumerate(COHORT_ORDER)}
    return sorted(rows, key=lambda r: (TISSUE_ORDER.index(r[1]), rank.get(r[0], len(rank)), list(reg).index(r[0])))


def list_height(n_rows: int, bottom: float) -> float:
    """Height in bp of a panel that holds a list of n_rows rows."""
    return LIST_TOP + n_rows * PITCH + bottom


def tissue_rules(ax, order: list, y, x0: float = 0.0, color: str = GROUP_RULE) -> None:
    """Thin rules between the tissues of a cohort list (y: row positions in the order of `order`)."""
    for i in range(1, len(order)):
        if order[i][1] != order[i - 1][1]:
            ax.axhline((y[i] + y[i - 1]) / 2, xmin=x0, color=color, lw=0.4, zorder=0.5)


def list_axis(ax, y, names: list | None) -> None:
    """The y axis of a list: one row per entry, names at the left or none."""
    ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names if names is not None else [""] * len(y), fontsize=S_TEXT)
    ax.tick_params(axis="y", length=0, pad=2)


# ---------------------------------------------------------------- supp_gate_gap: one cohort, gene by gene
def p_scatter(c: str = "kidney", label: dict | None = None) -> None:
    """One cohort: Δα of the two gates as rank z, one point per gene; genes whose difference passes the seed-noise test are dark. `label`: gene -> place of
    its name. Every gene lies inside the axes."""
    g = pd.read_parquet(OUT / "genes" / f"{c}.parquet")
    left, right = 33.0, 5.0
    side = W_SCATTER - left - right
    fig = page(W_SCATTER, list_height(len(cohort_order()), BOTTOM_ROW_1))
    ax = box(fig, left, BOTTOM_ROW_1, side, side)
    o, s = g[g.call == 0], g[g.call != 0]
    ax.scatter(o.z_can, o.z_free, s=1.0, c=BACKGROUND, lw=0, rasterized=True)
    ax.scatter(s.z_can, s.z_free, s=1.3, c=CALLED, lw=0, rasterized=True)
    m = float(np.ceil(max(g.z_can.abs().max(), g.z_free.abs().max()) * 2) / 2)
    ax.plot((-m, m), (-m, m), color=REF_LINE, lw=0.6, ls="--")
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_xticks(np.arange(-np.floor(m / 2) * 2, m + 0.1, 2))
    ax.set_yticks(np.arange(-np.floor(m / 2) * 2, m + 0.1, 2))
    ax.set_xlabel(f"{GATE.name} (rank z)")
    ax.set_ylabel(f"{PER_GENE.name} (rank z)")
    for sym, (lx, ly) in (label or {}).items():
        r = g[g.symbol == sym]
        if len(r):
            r = r.iloc[0]
            ax.plot([r.z_can, lx], [r.z_free, ly], color=INK, lw=0.4, path_effects=[withStroke(linewidth=1.2, foreground="white")], zorder=4)
            ax.plot(r.z_can, r.z_free, "o", mfc="none", mec=INK, ms=3.4, mew=0.6, path_effects=[withStroke(linewidth=1.8, foreground="white")], zorder=5)
            ax.text(lx + (0.06 if lx >= r.z_can else -0.06), ly, sym, fontsize=S_TEXT, style="italic", ha="left" if lx >= r.z_can else "right", va="center", zorder=6)
    rho = pd.Series(g.z_free).corr(pd.Series(g.z_can), method="spearman")
    n_up, n_down = int((g.call == 1).sum()), int((g.call == -1).sum())
    ax.text(0.03, 0.975, f"{n_up:,} higher", transform=ax.transAxes, ha="left", va="top", fontsize=S_TICK)
    ax.text(0.99, 0.095, f"{n_down:,} lower", transform=ax.transAxes, ha="right", va="center", fontsize=S_TICK)
    x0 = -(left - 2) / side
    ax.text(x0, 1.165, cohort_name(RU.cohorts()[c]["label"]), transform=ax.transAxes, ha="left", va="bottom", fontsize=S_TITLE)
    ax.text(x0, 1.03, f"ρ = {rho:.2f}, n = {len(g):,} genes\ndark: difference passes the seed-noise test", transform=ax.transAxes, ha="left", va="bottom", fontsize=S_TICK,
            linespacing=1.3)
    save(fig, "supp_gate_gap/scatter", raster=True)


# ---------------------------------------------------------------- supp_gate_gap: reliability and calls per cohort
def p_cohorts() -> None:
    s = pd.read_csv(OUT / "cohort_gap.csv").set_index("cohort")
    cal = pd.read_csv(OUT / "calibration.csv")
    order = cohort_order()
    y = np.arange(len(order))[::-1]
    w, gap_x, right = W_FULL - GAP - W_SCATTER, 8.0, 6.0
    aw = (w - LABEL_W - right - 2 * gap_x) / 3
    fig = page(w, list_height(len(order), BOTTOM_ROW_1))
    a1, a2, a3 = (box(fig, LABEL_W + k * (aw + gap_x), BOTTOM_ROW_1, aw, len(order) * PITCH) for k in range(3))
    nul = cal.groupby(["cohort", "gate"]).reliability_split.mean()  # null: mean over splits, per calibration cohort and gate
    a1.axvspan(nul.min(), nul.max(), color=style.NULL_BAND, lw=0)
    for yi, (c, t, lab) in zip(y, order):
        r = s.loc[c]
        if np.isfinite(r.reliability_split):
            a1.plot(r.reliability_split, yi, "o", color=INK, ms=2.6)
        else:
            a1.text(nul.max() + 0.06, yi, "one seed", va="center", fontsize=S_TEXT, color=MUTED)
        a2.plot(r.rho_zf_zc, yi, "o", color=INK, ms=2.6)
        a3.barh(yi, (r.n_up) / r.n_U * 100, color=BAR, height=0.65)
        a3.barh(yi, -(r.n_down) / r.n_U * 100, color=BAR, height=0.65)
    a1.set_xlim(-0.35, 1.02)
    a1.set_xticks([0, 0.5, 1])
    a1.set_xticklabels(["0", "0.5", "1"])
    a1.set_xlabel("reliability of\nthe difference\n(band: null)")
    a2.set_xlim(0.2, 0.85)
    a2.set_xticks([0.3, 0.5, 0.7])
    a2.set_xlabel("ρ between\nthe two gates")
    a3.axvline(0, color=INK, lw=0.5)
    a3.set_xlabel("genes called, %\nlower | higher")
    lim = max(abs(np.array(a3.get_xlim())))
    a3.set_xlim(-lim, lim)
    a3.set_xticks([-15, 0, 15])
    a3.set_xticklabels(["15", "0", "15"])
    for k, a in enumerate((a1, a2, a3)):
        list_axis(a, y, [lab for c, t, lab in order] if k == 0 else None)
        tissue_rules(a, order, y)
    save(fig, "supp_gate_gap/cohorts")


# ---------------------------------------------------------------- supp_gate_gap: protein neighborhoods
def p_neighbors() -> None:
    n = pd.read_csv(OUT / "nbr_resid.csv").set_index("cohort")
    order = cohort_order()
    y = np.arange(len(order))[::-1]
    w, right = W_FULL - GAP - W_COVARIATES, 6.0
    fig = page(w, list_height(len(order), BOTTOM_ROW_2))
    ax = box(fig, LABEL_W, BOTTOM_ROW_2, w - LABEL_W - right, len(order) * PITCH)
    for yi, (c, t, lab) in zip(y, order):
        r = n.loc[c]
        ax.plot([0, max(r.share_can, r.share_free, r.pull)], [yi, yi], color=GUIDE, lw=0.8, zorder=0)
        ax.plot(r.share_free_null_hi, yi, "|", color=MUTED, ms=4)
        ax.plot(r.pull, yi, "^", color=OTHER, ms=2.6)
        ax.plot(r.share_can, yi, GATE.marker, color=GATE.color, ms=2.8)
        ax.plot(r.share_free, yi, PER_GENE.marker, color=PER_GENE.color, ms=3.1)
    list_axis(ax, y, [lab for c, t, lab in order])
    tissue_rules(ax, order, y)
    ax.set_xlim(0, None)
    ax.set_xlabel("ρ with the mean of the 15 nearest ESM-2 neighbors\n(given expression, detection, π, τ, detection gap)", loc="right")
    h = [Line2D([], [], marker=GATE.marker, ls="", color=GATE.color, ms=3.2, label=GATE.name),
         Line2D([], [], marker=PER_GENE.marker, ls="", color=PER_GENE.color, ms=3.5, label=PER_GENE.name),
         Line2D([], [], marker="^", ls="", color=OTHER, ms=3, label=f"{GATE.name}, against the neighbors'\n{PER_GENE.name}, given the gene's own"),
         Line2D([], [], marker="|", ls="", color=MUTED, ms=5, label=f"random neighbors, 97.5th percentile\n({PER_GENE.name})")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(3 / w, 3 / (fig.get_size_inches()[1] * 72)), ncol=2, handletextpad=0.3, columnspacing=1.2, borderaxespad=0, labelspacing=0.35)
    save(fig, "supp_gate_gap/neighbors")


# ---------------------------------------------------------------- supp_gate_gap: covariates
def p_covariates() -> None:
    """Rows as in the neighborhood panel beside it (same page height and row positions); no row names here."""
    cv = pd.read_csv(OUT / "covariates.csv")
    d = cv[cv.readout == "delta"].pivot(index="cohort", columns="covariate", values="partial_rho")
    cols = ["expr", "detection", "abs_S_can", "abs_S_free", "pi_raw", "tau", "det_gap"]
    order = cohort_order()
    M = d.reindex([c for c, _, _ in order])[cols].to_numpy(float)
    R2 = d.reindex([c for c, _, _ in order])[["ALL_R2"]].to_numpy(float)
    lim, cell, gap_x = 0.6, 18.5, 3.0
    fig = page(W_COVARIATES, list_height(len(order), BOTTOM_ROW_2))
    ax = box(fig, 2, BOTTOM_ROW_2, len(cols) * cell, len(order) * PITCH)
    bx = box(fig, 2 + len(cols) * cell + gap_x, BOTTOM_ROW_2, cell, len(order) * PITCH)
    im = ax.imshow(M, cmap=style.DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    bx.imshow(R2, cmap="Greys", vmin=0, vmax=0.8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]) and abs(M[i, j]) >= 0.25:
                ax.text(j, i, signed(M[i, j]), ha="center", va="center", fontsize=S_TEXT, color="white" if abs(M[i, j]) > 0.45 else INK)
        if np.isfinite(R2[i, 0]):
            bx.text(0, i, f"{R2[i, 0]:.2f}", ha="center", va="center", fontsize=S_TEXT, color="white" if R2[i, 0] > 0.4 else INK)
    for a, names in ((ax, [COV_NAME[c] for c in cols]), (bx, [COV_NAME["ALL_R2"]])):
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=-45, ha="left", rotation_mode="anchor", fontsize=S_TICK)
        a.set_yticks([])
        a.tick_params(length=0, pad=3)
        for sp in a.spines.values():
            sp.set_visible(False)
        for i in range(1, len(order)):
            if order[i][1] != order[i - 1][1]:
                a.axhline(i - 0.5, color="white", lw=1.0)
    n_out = int((np.abs(M[np.isfinite(M)]) > lim).sum())
    cax = box(fig, 2 + (len(cols) + 1) * cell + gap_x + 5, BOTTOM_ROW_2 + len(order) * PITCH * 0.25, 5.5, len(order) * PITCH * 0.5)
    cb = fig.colorbar(im, cax=cax, extend="both" if n_out else "neither")
    cb.set_label(f"partial ρ ({DIFFERENCE}, covariate)" + (f"; {n_out} values beyond the scale" if n_out else ""), fontsize=S_TICK)
    cb.set_ticks([-0.6, -0.3, 0, 0.3, 0.6])
    cb.ax.tick_params(labelsize=S_TICK, length=2)
    cb.outline.set_linewidth(0.4)
    save(fig, "supp_gate_gap/covariates")


# ---------------------------------------------------------------- supp_gate_gap_classes: across cohorts
def p_cross() -> None:
    X = pd.read_csv(OUT / "cross_cohort_rho.csv")
    icc = pd.read_csv(OUT / "icc.csv").set_index("readout").icc
    w, h, left, bottom, top = W_CROSS, H_ROW_CLASSES, 42.0, 62.0, 12.0
    fig = page(w, h)
    ax = box(fig, left, bottom, w - left - 4, h - bottom - top)
    keys = [("rho_z_can", "z_can", GATE.name, GATE.color, GATE.marker), ("rho_z_free", "z_free", PER_GENE.name, PER_GENE.color, PER_GENE.marker), ("rho_delta", "delta", DIFFERENCE, INK, "D")]
    pos = 0
    rng = np.random.default_rng(0)
    groups = []
    for same, lab in ((True, "same tissue"), (False, "different tissues")):
        start = pos
        for k, _, name, col, mk in keys:
            v = X[X.same_tissue == same][k].dropna().to_numpy()
            ax.boxplot(v, positions=[pos], widths=0.6, showfliers=False, medianprops=dict(color=col, lw=1.2), boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
            ax.scatter(pos + rng.uniform(-0.18, 0.18, len(v)), v, s=0.8, color=col, alpha=0.35, lw=0)
            ax.plot(pos, -0.045, mk, color=col, ms=3.2, transform=ax.get_xaxis_transform(), clip_on=False)
            pos += 1
        groups.append((start + 1, f"{lab}\n{int((X.same_tissue == same).sum())} pairs of cohorts"))
        pos += 0.6
    ax.axhline(0, color=REF_LINE, lw=0.6)
    ax.set_xticks([])
    for x, lab in groups:
        ax.text(x, -0.10, lab, ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=S_TICK, linespacing=1.15)
    ax.set_ylabel("Spearman between cohorts\n(shared genes)")
    h_ = [Line2D([], [], marker=mk, ls="", color=col, ms=3.2, label=f"{name} ({icc[r]:.2f})") for _, r, name, col, mk in keys]
    fig.legend(handles=h_, loc="lower left", bbox_to_anchor=(2 / w, 9 / h), ncol=1, handletextpad=0.3, borderaxespad=0, labelspacing=0.3)
    fig.text(3 / w, 1.5 / h, "in parentheses: gene's share of variance, 8 tissues (ICC)", fontsize=S_TEXT, color=MUTED, ha="left", va="bottom")
    save(fig, "supp_gate_gap_classes/cross_tissue")


# ---------------------------------------------------------------- supp_gate_gap_classes: gene classes and Hallmark sets
def p_sets() -> None:
    r = pd.read_csv(OUT / "set_recurrence.csv")
    th = list(RU.thread())
    keep = r[(r.readout == "delta") & r.recurrent].sort_values("mean_z", ascending=False).key.tolist()
    sub = r[r.readout == "delta"].set_index("key").loc[keep]
    M = sub[[f"z_{t}" for t in th]].to_numpy(float)
    side = np.c_[r[r.readout == "z_can"].set_index("key").loc[keep].mean_z, r[r.readout == "z_free"].set_index("key").loc[keep].mean_z]
    lim = 6.0
    w, h, left, bottom, top, cell_w, side_w, gap_x = W_FULL - GAP - W_CROSS, H_ROW_CLASSES, 74.0, 62.0, 12.0, 18.0, 21.0, 5.0
    fig = page(w, h)
    ax = box(fig, left, bottom, len(th) * cell_w, h - bottom - top)
    bx = box(fig, left + len(th) * cell_w + gap_x, bottom, 2 * side_w, h - bottom - top)
    im = ax.imshow(M, cmap=style.DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) >= 1.96:
                ax.text(j, i, "•", ha="center", va="center", fontsize=S_TEXT, color="white" if abs(M[i, j]) > 4 else INK)
    ax.set_xticks(range(len(th)))
    ax.set_xticklabels([TLAB[t] for t in th], rotation=45, ha="right", rotation_mode="anchor", fontsize=S_TICK)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([SET_LABEL.get(k, k.split("|")[1]) for k in keep], fontsize=S_TICK)
    ax.set_title(f"{DIFFERENCE} (per-gene minus protein-feature)", fontsize=S_TICK, pad=3)
    bx.imshow(side, cmap=style.DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(side.shape[0]):
        for j in range(2):
            bx.text(j, i, signed(side[i, j], 1), ha="center", va="center", fontsize=S_TEXT, color="white" if abs(side[i, j]) > 4 else INK)
    bx.set_xticks([0, 1])
    bx.set_xticklabels([GATE.name, PER_GENE.name], rotation=45, ha="right", rotation_mode="anchor", fontsize=S_TICK)
    bx.set_yticks([])
    bx.set_title("each gate", fontsize=S_TICK, pad=3)
    for a in (ax, bx):
        a.tick_params(length=0, pad=3)
        for sp in a.spines.values():
            sp.set_visible(False)
    cax = box(fig, left + len(th) * cell_w + 2 * (gap_x + side_w), bottom + (h - bottom - top) * 0.1, 5.5, (h - bottom - top) * 0.8)
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label(f"set z (colors end at ±{lim:.0f})", fontsize=S_TICK)
    cb.ax.tick_params(labelsize=S_TICK, length=2)
    cb.outline.set_linewidth(0.4)
    fig.text(3 / w, (bottom - 4) / h, "• |z| ≥ 1.96", fontsize=S_TICK, ha="left", va="top")
    save(fig, "supp_gate_gap_classes/sets")


# ---------------------------------------------------------------- supp_gate_gap_classes: independent references
def p_references() -> None:
    """One row per cohort and independent reference, under tissue headings: partial ρ of the Δα of each gate with the reference, given the other gate's Δα."""
    T = pd.read_csv(OUT / "golds_trackc.csv")
    rows = []
    for (c, kind), g in T.groupby(["cohort", "kind"], sort=False):
        f = g[(g.group == "given_can") & (g.method == "z_free")]
        k = g[(g.group == "given_free") & (g.method == "z_can")]
        if len(f) and len(k):
            rows.append(dict(cohort=c, unit=kind, f=f.value.iloc[0], flo=f.ci_lo.iloc[0], fhi=f.ci_hi.iloc[0], k=k.value.iloc[0], klo=k.ci_lo.iloc[0], khi=k.ci_hi.iloc[0]))
    cf = OUT / "golds_cortex.csv"
    if cf.exists():
        C = pd.read_csv(cf)
        for (c, gname), g in C.groupby(["cohort", "gold"], sort=False):
            f = g[(g.group == "given_can") & (g.score == "z_free")]
            k = g[(g.group == "given_free") & (g.score == "z_can")]
            rows.append(dict(cohort=c, unit=f"held-out {gname.replace('_', ', ')}", f=f.rho_p.iloc[0], flo=f.ci_lo.iloc[0], fhi=f.ci_hi.iloc[0], k=k.rho_p.iloc[0], klo=k.ci_lo.iloc[0],
                             khi=k.ci_hi.iloc[0]))
    D = pd.DataFrame(rows)
    order = cohort_order()
    rank = {c: i for i, (c, _, _) in enumerate(order)}
    tissue = {c: t for c, t, _ in order}
    name = {c: lab[len(TLAB[t]) + 1:] if lab.startswith(TLAB[t] + " ") else lab for c, t, lab in order}  # the cohort without its tissue, under a tissue heading
    D["tissue"] = D.cohort.map(tissue)
    D["label"] = [f"{name[c]} · {REF_LABEL_TISSUE.get((tissue[c], u), REF_LABEL.get(u, u))}" for c, u in zip(D.cohort, D.unit)]
    D = D.assign(_r=D.cohort.map(rank)).sort_values("_r", kind="stable")
    tissues = [t for t in TISSUE_ORDER if (D.tissue == t).any()]
    cut = int(np.argmin([abs(sum((D.tissue == t).sum() + 1 for t in tissues[:k]) - (len(D) + len(tissues)) / 2) for k in range(1, len(tissues))])) + 1
    blocks = [tissues[:cut], tissues[cut:]]  # two blocks of tissues, side by side, as even as the tissues allow
    n_rows = max(sum((D.tissue == t).sum() + 1 for t in b) for b in blocks)
    lo, hi = min(D.flo.min(), D.klo.min()), max(D.fhi.max(), D.khi.max())
    lim = (np.floor(lo * 10 - 0.5) / 10, np.ceil(hi * 10 + 0.5) / 10)
    w, bottom, label_w, right, mid = W_FULL, 40.0, 150.0, 9.0, 6.0
    aw = (w - 2 * label_w - right - mid) / 2
    fig = page(w, list_height(n_rows, bottom))
    for b, ts in enumerate(blocks):
        ax = box(fig, label_w + b * (label_w + aw + mid), bottom, aw, n_rows * PITCH)
        ax.axvline(0, color=REF_LINE, lw=0.6)
        yy, names, pos = [], [], n_rows - 1
        for t in ts:
            ax.text(-(label_w - 2) / aw, pos, TLAB[t], transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=S_TICK, fontweight="bold", clip_on=False)
            if pos < n_rows - 1:
                ax.plot([-(label_w - 2) / aw, 1], [pos + 0.5, pos + 0.5], transform=ax.get_yaxis_transform(), color=GROUP_RULE, lw=0.4, clip_on=False)
            pos -= 1
            for r in D[D.tissue == t].itertuples():
                ax.errorbar(r.k, pos + 0.17, xerr=[[r.k - r.klo], [r.khi - r.k]], fmt=GATE.marker, color=GATE.color, ms=2.4, lw=0.7)
                ax.errorbar(r.f, pos - 0.17, xerr=[[r.f - r.flo], [r.fhi - r.f]], fmt=PER_GENE.marker, color=PER_GENE.color, ms=2.7, lw=0.7)
                yy.append(pos)
                names.append(r.label)
                pos -= 1
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.set_yticks(yy)
        ax.set_yticklabels(names, fontsize=S_TEXT)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_xlim(lim)
        ax.set_xlabel("partial ρ with independent reference", loc="right")
    h = [Line2D([], [], marker=GATE.marker, ls="-", lw=0.7, color=GATE.color, ms=3.2, label=f"{GATE.name}, given covariates and the per-gene gate"),
         Line2D([], [], marker=PER_GENE.marker, ls="-", lw=0.7, color=PER_GENE.color, ms=3.5, label=f"{PER_GENE.name}, given covariates and the protein-feature gate")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(3 / w, 2 / list_height(n_rows, bottom)), ncol=1, handletextpad=0.4, borderaxespad=0, labelspacing=0.3)
    save(fig, "supp_gate_gap_classes/references")


# ---------------------------------------------------------------- supp_gate_gap_classes: sequence constraint
def p_sequence() -> None:
    s = pd.read_csv(OUT / "sequence.csv")
    s = s[s.readout == "delta"]
    order = cohort_order()
    tissues = [t for t in TISSUE_ORDER if any(o[1] == t for o in order)]
    count = [sum(o[1] == t for o in order) for t in tissues]
    cut = int(np.argmin([abs(sum(count[:k]) - len(order) / 2) for k in range(1, len(tissues))])) + 1
    blocks = [[o for o in order if o[1] in tissues[:cut]], [o for o in order if o[1] in tissues[cut:]]]  # two blocks of tissues, side by side
    n_rows = max(len(b) for b in blocks)
    m = float(np.ceil(max(abs(s.p_ci_lo.min()), abs(s.p_ci_hi.max())) * 20) / 20)
    w, bottom, right, mid = W_FULL, 34.0, 9.0, 6.0
    aw = (w - 2 * LABEL_W - right - mid) / 2
    h = list_height(n_rows, bottom)
    fig = page(w, h)
    for b, rows in enumerate(blocks):
        ax = box(fig, LABEL_W + b * (LABEL_W + aw + mid), bottom + (n_rows - len(rows)) * PITCH, aw, len(rows) * PITCH)
        y = np.arange(len(rows))[::-1]
        ax.axvline(0, color=REF_LINE, lw=0.6)
        for yi, (c, t, lab) in zip(y, rows):
            for con, mark, off in (("log10_dNdS", DNDS, 0.17), ("LOEUF", LOEUF, -0.17)):
                r = s[(s.cohort == c) & (s.constraint == con)]
                if len(r):
                    r = r.iloc[0]
                    ax.errorbar(r.partial_rho, yi + off, xerr=[[r.partial_rho - r.p_ci_lo], [r.p_ci_hi - r.partial_rho]], fmt=mark.marker, color=mark.color, ms=2.5, lw=0.7)
        ax.set_xlim(-m, m)
        list_axis(ax, y, [lab for c, t, lab in rows])
        tissue_rules(ax, rows, y)
        ax.set_xlabel(f"partial ρ ({DIFFERENCE}, constraint)\ngiven π, expression, detection", loc="right")
    hd = [Line2D([], [], marker=DNDS.marker, ls="-", lw=0.7, color=DNDS.color, ms=3.2, label=f"{DNDS.name}, human–mouse"),
          Line2D([], [], marker=LOEUF.marker, ls="-", lw=0.7, color=LOEUF.color, ms=3.2, label=LOEUF.name)]
    short = int(np.argmin([len(b) for b in blocks]))
    fig.legend(handles=hd, loc="upper right", bbox_to_anchor=((short * (LABEL_W + aw + mid) + LABEL_W + aw) / w, ((n_rows - len(blocks[short])) * PITCH + 2) / h), ncol=1,
               handletextpad=0.4, borderaxespad=0, labelspacing=0.3)
    save(fig, "supp_gate_gap_classes/sequence")


# ---------------------------------------------------------------- supp_gate_gap_named: named examples
def named_matrix(genes: list[str]) -> tuple:
    """(M, C) for the named-gene panel, genes by tissues: M, the difference of a gene in a tissue, the mean over the
    tissue's cohorts (NaN where the gene has no value in any of them); C, non-zero where the difference is called in a cohort."""
    reg = RU.cohorts()
    tis = [t for t in TISSUE_ORDER]
    values = [[[] for _ in tis] for _ in genes]
    C = np.zeros((len(genes), len(tis)))
    for c in reg:
        g = pd.read_parquet(OUT / "genes" / f"{c}.parquet").set_index("symbol")
        j = tis.index(reg[c]["tissue"])
        for i, s in enumerate(genes):
            if s in g.index:
                r = g.loc[[s]].iloc[0]
                if r.call != 0:
                    C[i, j] = r.call
                if np.isfinite(r.delta):
                    values[i][j].append(float(r.delta))
    M = np.array([[np.mean(v) if v else np.nan for v in row] for row in values])  # a cell is the mean over the tissue's cohorts
    return M, C


def p_named(genes: list[str]) -> None:
    tis = [t for t in TISSUE_ORDER]
    M, C = named_matrix(genes)
    nb = pd.read_csv(OUT / "named.csv").set_index("symbol")
    lim, cell_w, cell_h = 3.0, 20.0, 10.0
    w, left, top, bottom = W_NAMED, 40.0, 12.0, 86.0
    h = top + len(genes) * cell_h + bottom
    fig = page(w, h)
    ax = box(fig, left, bottom, len(tis) * cell_w, len(genes) * cell_h)
    cmap = plt.get_cmap(style.DIVERGING).copy()
    cmap.set_bad(NA_CELL)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(len(genes)):
        for j in range(len(tis)):
            if C[i, j] != 0:
                ax.text(j, i, "•", ha="center", va="center", fontsize=S_TEXT, color="white" if abs(M[i, j]) > 2 else INK)
    ax.set_xticks(range(len(tis)))
    ax.set_xticklabels([TLAB[t] for t in tis], rotation=45, ha="right", rotation_mode="anchor", fontsize=S_TICK)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=S_TICK, style="italic")
    ax.tick_params(length=0, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if genes == NAMED:
        for k in np.cumsum(NAMED_GROUPS)[:-1]:
            ax.axhline(k - 0.5, color="white", lw=2.0)
    for i, s in enumerate(genes):
        if s in nb.index:
            ax.text(len(tis) - 0.25, i, ", ".join(nb.loc[s, "nbr_symbols"].split(";")[:3]), va="center", fontsize=S_TEXT, color=MUTED, style="italic")
    ax.text(len(tis) - 0.25, -0.5, "nearest ESM-2 neighbors", va="bottom", fontsize=S_TEXT, color=MUTED)
    n_out = int((np.abs(M[np.isfinite(M)]) > lim).sum())
    cax = box(fig, left + len(tis) * cell_w / 2 - 60, 33, 120, 5.5)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both" if n_out else "neither")
    cb.set_label(f"{DIFFERENCE_LONG} (rank z)" + (f"; {n_out} cells beyond ±{lim:.0f}" if n_out else ""), fontsize=S_TICK)
    cb.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cb.ax.tick_params(labelsize=S_TICK, length=2)
    cb.outline.set_linewidth(0.4)
    hd = [Line2D([], [], marker="o", ls="", color=INK, ms=1.8, label="difference called in a cohort of the tissue"),
          Patch(fc=NA_CELL, ec="none", label="gene without count evidence in both fits")]
    fig.legend(handles=hd, loc="lower left", bbox_to_anchor=(left / w, 1.5 / h), ncol=2, handletextpad=0.4, handlelength=1.0, columnspacing=1.5, borderaxespad=0)
    save(fig, "supp_gate_gap_named/named")


def draw_panels(genes: list[str] | None = None) -> None:
    """Every panel of the three figures, from the cached tables of results/analysis/X8/. `genes`: the rows of the named-gene panel (default: NAMED)."""
    style.apply()
    p_scatter("kidney", SCATTER_GENES)
    p_cohorts()
    p_neighbors()
    p_covariates()
    p_cross()
    p_sets()
    if (OUT / "golds_trackc.csv").exists():
        p_references()
    p_sequence()
    p_named(genes or NAMED)


def main() -> int:
    draw_panels(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
