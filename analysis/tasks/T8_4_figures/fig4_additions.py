"""Panels on the liver reading: prior landscape, cell-type drivers, margin support, tissue comparison.

  fig4c_prior_landscape    Prior landscape (fig:dalpha-biology c): protein-identity prior vs Delta-alpha on
                           liver 4 species, with five groups of named genes, each named on the plane in the color
                           of its markers. A dagger marks a named tail gene (|Delta-alpha|
                           > 0.05) that lacks margin support (T2_5_celltype_margin: count evidence S_g unmasked, of
                           the sign of Delta-alpha and at least the cohort median |S_g|), i.e. placed through its
                           protein features. Saved at its exact size as ``main_delta_alpha_liver/prior_landscape``.
  fig4_celltype_drivers    Where each named gene's count-evidence margin S_g comes from: the exact
                           per-cell-type split C_gk (posterior-mean z) of T2_5_celltype_margin, liver 4 species.
  fig4_margin_support      Fraction of tail genes (|Delta-alpha| > 0.05) with margin support, per cohort and
                           tail, one trained run per cohort: tables of T2_5_celltype_margin on the canonical s42
                           runs, and the same flag (``margin.margin_flag``) on the first kept run of every other
                           cohort in ``runs.tsv`` of H1_harmonise, so later cohorts join when H1_harmonise lists
                           them. Checked against T2_5_celltype_margin.
  fig4_cross_tissue        Tissue comparison (T2_3_tissue, primary version): Delta-alpha against
                           the raw prior, centered on the cohort median, same species pair, genes joined
                           on Ensembl ID.
The names on the left are those the script saves under; ``figures/panel_names.tsv`` gives the panel file of each
(``figures/panels/main_delta_alpha_liver/``).
Sources: ``results/analysis/T2_5/`` (canonical s42 runs), ``results/analysis/T2_3/tissue_comparison.csv``.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.text import Text
from scipy.stats import spearmanr

from analysis.tasks.T8_4_figures import common as cm

T25 = cm.RES / "T2_5"
T23 = cm.RES / "T2_3"
# Gene groups of the prior landscape panel (notebooks/_alpha_biology_builder.py, H_THEMES), same colors. The first group
# holds enzymes of hepatocyte metabolism: the urea cycle (CPS1, ARG1, ASL, OTC), gluconeogenesis (PCK1) and the breakdown
# of phenylalanine (PAH).
HEPATOCYTE_ENZYMES = "hepatocyte metabolic enzymes"
THEMES = {
    HEPATOCYTE_ENZYMES: (["CPS1", "ARG1", "ASL", "OTC", "PCK1", "PAH"], "#0072B2"),
    "mitochondrially encoded OXPHOS": (["MT-CO1", "MT-CO2", "MT-CO3", "MT-CYB", "MT-ND1", "MT-ND2", "MT-ND4",
                                        "MT-ATP6"], "#5B3E80"),
    "supporting tissue": (["SELP", "SELE", "VCAM1", "F3", "CR2", "CD40LG", "COL17A1", "PDGFRB", "KIT", "C1QC"],
                      "#D55E00"),
    "conserved controls": (["DDX4", "RPL3", "EEF1A1", "EEF2", "ACTB", "TUBA1A", "COX4I1", "NDUFA1", "HNF4A"], "#7F7F7F"),
}
DAGGER = "†"
# Not drawn (ortholog audit O1): OTC has no human counts in the source matrix (a source zero); COL17A1 and
# CD40LG have wrong zebrafish one-to-one targets (col19a1, identity 5; tnfsf10l4, identity 18); SELE's
# zebrafish target is selp and it has no mouse counts; SELP's zebrafish target is an unnamed low-confidence
# gene (identity 13). Applies to panels c and d alike.
O1_DROPPED = {"OTC", "COL17A1", "CD40LG", "SELE", "SELP"}


# Genes of the lower tail that pass every check of the named-gene table (same sign in all five seeds, margin support,
# counts in every species): named in bold in the landscape panel, which otherwise names genes that the text qualifies.
EVERY_CHECK = ["HAL", "ARG2", "CYP7A1"]
EVERY_CHECK_NAME, EVERY_CHECK_COLOR = "enzymes passing every check", cm.INK

# The landscape panel is the third panel of the main Δα figure. Its page has the height of the two panels beside it and its
# axes their band of rows (ROW_HEIGHT, ROW_BOTTOM and ROW_TOP of figures/redraw/main_figures.py), so that the three x axes
# lie on one line; its width is what the two panels leave of the text width. All in bp, on the 0.24 bp grid of the layout tool.
LANDSCAPE_PANEL = "main_delta_alpha_liver/prior_landscape"
LANDSCAPE_SIZE = (216.0, 190.56)
LANDSCAPE_BAND = (19.56, 11.0)            # under the axes (tick labels, axis title) and above them (the margin plot)
LANDSCAPE_VIEW = ((0.45, 1.0), (-0.60, 0.25))   # prior and Δα; the few genes left of it are marked at the frame
# The groups as they are named on the plane: text, place of its anchor (bp from the lower left corner of the axes; the
# baseline of the last line) and alignment. Each name stands in the color of its markers, beside them.
GROUP_LABEL = {
    "supporting tissue": ("supporting tissue", (74.0, 152.0), "center"),
    "mitochondrially encoded OXPHOS": ("mitochondrially encoded", (4.0, 3.0), "left"),
    HEPATOCYTE_ENZYMES: ("hepatocyte\nmetabolic enzymes", (174.5, 3.0), "right"),
    "conserved controls": ("conserved controls", (174.5, 127.5), "right"),
    "enzymes passing every check": ("enzymes passing\nevery check", (80.0, 75.5), "right"),
}
DAGGER_NOTE = (f"{DAGGER} placed through\nprotein features", (174.5, 147.5), "right")
# Where the symbol of a gene stands: offset of its anchor from the marker (bp), horizontal and vertical alignment. A symbol
# farther than LEADER bp from its marker is tied to it by a line.
SYMBOL_PLACE = {
    "F3": ((-3.8, 1.5), "right", "center"), "CR2": ((-3.4, -5.5), "right", "center"), "VCAM1": ((-3.0, -4.2), "center", "top"),
    "C1QC": ((-1.0, -10.5), "center", "top"), "KIT": ((2.0, -4.2), "center", "top"), "PDGFRB": ((3.8, 0.5), "left", "center"),
    "NDUFA1": ((-3.8, -0.5), "right", "center"), "COX4I1": ((-3.8, -1.0), "right", "center"), "DDX4": ((0.0, 4.0), "center", "bottom"),
    "HNF4A": ((-3.8, -1.5), "right", "center"), "RPL3": ((-3.8, 0.0), "right", "center"), "EEF2": ((-3.8, 0.0), "right", "center"),
    "CYP7A1": ((-3.6, 3.0), "right", "bottom"), "ARG2": ((-3.6, -3.0), "right", "top"), "HAL": ((3.6, -3.0), "left", "top"),
    "MT-ND2": ((-3.8, 0.0), "right", "center"), "MT-ND4": ((-3.8, 0.5), "right", "center"), "MT-ATP6": ((-3.8, -0.5), "right", "center"),
    "MT-CO2": ((2.0, -4.2), "center", "top"), "MT-CYB": ((0.0, 4.2), "center", "bottom"), "MT-CO3": ((6.0, -6.2), "left", "top"),
    "MT-CO1": ((3.8, 0.0), "left", "center"), "ARG1": ((-0.7, 12.5), "right", "bottom"), "PAH": ((-5.6, -6.0), "right", "top"),
    "ASL": ((-2.4, 5.5), "center", "bottom"), "PCK1": ((-1.0, 11.0), "center", "bottom"), "CPS1": ((3.8, 0.0), "left", "center"),
}
LEADER = 6.5
COINCIDE = 1.5  # bp: two named genes of different groups closer than this would hide each other; they share one marker


def signed(v: float, digits: int = 2) -> str:
    """A value with its sign and the minus sign (not a hyphen); a value that rounds to zero carries no sign."""
    return f"{0:.{digits}f}" if round(float(v), digits) == 0 else f"{v:+.{digits}f}".replace("-", "−")


def genes_4sp() -> pd.DataFrame:
    return pd.read_parquet(T25 / "liv_4sp__genes.parquet")


def every_check(g: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``g`` for the genes of ``EVERY_CHECK``, after confirming in the stored tables what the name of the group
    says: each gene is in the plotted table with margin support and hepatocytes as the cell type that contributes most
    to its count evidence (``T2_5/liv_4sp__genes.parquet``), and the named-gene table gives it the class "confident",
    the same sign in all five seeds and counts in every species (``X2/named_genes.csv``)."""
    s = g[g.symbol.isin(EVERY_CHECK)].copy()
    assert sorted(s.symbol) == sorted(EVERY_CHECK), f"not in the plotted table: {sorted(set(EVERY_CHECK) - set(s.symbol))}"
    assert s["tail"].all() and s.margin_support.all() and not s.lacks_margin_support.any(), "a gene of EVERY_CHECK lacks margin support"
    assert (s.dominant_cell_type == "Hepatocytes").all(), "a gene of EVERY_CHECK is not led by hepatocytes"
    t = pd.read_csv(cm.RES / "X2" / "named_genes.csv").set_index("symbol").loc[EVERY_CHECK]
    passed = (t["class"] == "confident") & (t.sign_consistency == 1.0) & (t.margin_support.astype(str) == "True") & (t.complete.astype(str) == "True")
    assert passed.all(), f"not passing every check in X2/named_genes.csv: {list(t.index[~passed])}"
    s["theme"], s["color"], s["label"] = EVERY_CHECK_NAME, EVERY_CHECK_COLOR, s.symbol
    return s


def named(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for theme, (genes, col) in THEMES.items():
        s = g[g.symbol.isin(set(genes) - O1_DROPPED)].copy()
        s["theme"], s["color"] = theme, col
        rows.append(s)
    out = pd.concat(rows)
    # Dagger: T2.5 flag ``lacks_margin_support`` (a tail gene, |Delta-alpha| > 0.05, whose count evidence
    # does not back its sign), as planned for the main panel. Near-zero genes outside the tail carry no
    # claim and no dagger.
    out["label"] = out.symbol + np.where(out.lacks_margin_support, DAGGER, "")
    return out


def panel_landscape() -> plt.Figure:
    """Main Δα figure, panel c (``main_delta_alpha_liver/prior_landscape``, 216 x 191 bp): every gene of the four-species
    liver model on the plane of protein prior and Δα, with five groups of named genes.

    What it draws. One gray dot per gene, protein prior π on the x axis (from 0.45; the 17 genes left of it are small
    triangles on the frame, at their Δα) and Δα on the y axis; the gray line is Δα = 0. The two margins hold the
    distribution of each quantity over all genes (60 bins). The genes of ``THEMES`` are colored markers with their
    symbols (those of ``O1_DROPPED`` are left out); a dagger after a symbol marks a tail gene (|Δα| > 0.05) whose Δα is
    not backed by count evidence of the same sign, that is, a gene placed through its protein features. The genes of
    ``EVERY_CHECK`` are larger black markers with bold symbols: three hepatocyte enzymes below the prior that pass
    every check (``every_check``). Each group is named on the plane in the color of its markers (``GROUP_LABEL``);
    there is no key. Two named genes of different groups that lie closer than ``COINCIDE`` (PAH and MT-CO3, 0.7 bp
    apart) would hide each other: they share one marker at the middle of the two, a little larger than a single one,
    its left half in the color of the gene with the lower prior and its right half in the color of the other, each
    with its symbol on its side.

    Reads ``results/analysis/T2_5/liv_4sp__genes.parquet``, columns ``symbol``, ``prior``, ``delta_alpha`` and
    ``lacks_margin_support`` (the dagger) and ``dominant_cell_type`` (the group named "hepatocyte metabolic enzymes"
    must have hepatocytes as the cell type that contributes most to the count evidence of each of its genes), and for
    the three genes of ``EVERY_CHECK`` their class in ``results/analysis/X2/named_genes.csv``. No value is computed.

    Every symbol stands at a fixed place beside its marker (``SYMBOL_PLACE``), tied to it by a line when it stands
    apart. The panel stops when a named gene lies outside the axes, or when a symbol or a group name lies on a marker
    or on another symbol or name. The page and the band of the axes are those of the two panels beside it
    (``LANDSCAPE_SIZE``, ``LANDSCAPE_BAND``).

        python -m analysis.tasks.T8_4_figures.fig4_additions --panel main_delta_alpha_liver/prior_landscape     # in chimera_env, from the repository root
    """
    g = genes_4sp()
    nm = pd.concat([named(g), every_check(g)])
    (W, H), (bottom, band) = LANDSCAPE_SIZE, LANDSCAPE_BAND
    left, gap, air = 28.0, 2.0, 0.8  # bp: y-axis band, distance of a margin plot from the plot, white kept at the page edge
    margin = band - gap - air  # depth of a margin plot
    w, h = W - left - gap - margin - air, H - bottom - band
    xlim, ylim = LANDSCAPE_VIEW
    assert g.delta_alpha.between(*ylim).all() and (g.prior <= xlim[1]).all(), "a gene lies above, below or right of the axes"
    assert nm.prior.between(*xlim).all(), "a named gene lies outside the axes"
    assert (nm[nm.theme == HEPATOCYTE_ENZYMES].dominant_cell_type == "Hepatocytes").all(), "a gene of the hepatocyte group is not led by hepatocytes"
    mark, strong, size, name_size, line = 3.5, 4.5, 5.5, 6.0, 0.5  # pt: marker of a named gene, of a gene of EVERY_CHECK; symbols; group names; thin lines
    double = 4.4  # pt: a marker that two genes share, so that each half can be seen
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([left / W, bottom / H, w / W, h / H])
    top = fig.add_axes([left / W, (bottom + h + gap) / H, w / W, margin / H], sharex=ax)
    right = fig.add_axes([(left + w + gap) / W, bottom / H, margin / W, h / H], sharey=ax)
    shown = g[g.prior >= xlim[0]]
    cm.fd.density_scatter(ax, shown.prior, shown.delta_alpha, s=1.2, alpha=0.25, color="#BBBBBB")
    edge = g[g.prior < xlim[0]]  # genes left of the axis: on the frame, at their Δα
    ax.scatter(np.full(len(edge), 2.6 / w), edge.delta_alpha, marker="<", s=5, color="#999999", linewidths=0, transform=ax.get_yaxis_transform(), zorder=3)
    ax.axhline(0, color=cm.MUTED, lw=0.6, zorder=2)
    texts, owners = [], []
    groups = [(theme, col) for theme, (_, col) in THEMES.items()] + [(EVERY_CHECK_NAME, EVERY_CHECK_COLOR)]
    # Named genes of different groups that would hide each other: one marker for the two, a half for each.
    at = np.c_[(nm.prior - xlim[0]) / (xlim[1] - xlim[0]) * w, (nm.delta_alpha - ylim[0]) / (ylim[1] - ylim[0]) * h]  # bp in the axes
    shared = [(i, j) for i in range(len(nm)) for j in range(i) if nm.theme.iloc[i] != nm.theme.iloc[j] and np.hypot(*(at[i] - at[j])) < COINCIDE]
    assert len({k for pair in shared for k in pair}) == 2 * len(shared), "more than two named genes at one place"
    halved = set()
    for pair in shared:
        a, b = sorted(pair, key=lambda k: nm.prior.iloc[k])  # the gene with the lower prior at the left
        ax.plot([nm.prior.iloc[[a, b]].mean()], [nm.delta_alpha.iloc[[a, b]].mean()], marker="o", fillstyle="left", ms=double, mfc=nm.color.iloc[a],
                markerfacecoloralt=nm.color.iloc[b], mec="white", mew=0.4, ls="none", zorder=4)
        halved |= {nm.symbol.iloc[a], nm.symbol.iloc[b]}
        print(f"one marker for {nm.symbol.iloc[a]} (left half) and {nm.symbol.iloc[b]} (right half), {np.hypot(*(at[a] - at[b])):.2f} bp apart")
    for theme, col in groups:
        s = nm[nm.theme == theme]
        bold = theme == EVERY_CHECK_NAME
        own = s[~s.symbol.isin(halved)]
        ax.scatter(own.prior, own.delta_alpha, s=(strong if bold else mark) ** 2, color=col, edgecolor="white", linewidth=0.6 if bold else 0.4, zorder=5 if bold else 4)
        for r in s.itertuples():
            (dx, dy), ha, va = SYMBOL_PLACE[r.symbol]
            apart = np.hypot(dx, dy) > LEADER
            tie = dict(arrowstyle="-", color=cm.MUTED, lw=line, shrinkA=0.6, shrinkB=(double / 2 + 1.0) if r.symbol in halved else (mark / 2 + 0.8),
                       relpos=({"left": 0.0, "center": 0.5, "right": 1.0}[ha], {"bottom": 0.0, "center": 0.5, "top": 1.0}[va])) if apart else None
            texts.append(ax.annotate(r.label, (r.prior, r.delta_alpha), xytext=(dx, dy), textcoords="offset points", fontsize=name_size if bold else size,
                                     ha=ha, va=va, color=cm.INK, zorder=6, fontweight="bold" if bold else "normal", arrowprops=tie))
            owners.append(r.symbol)
        text, at, ha = GROUP_LABEL[theme]
        texts.append(ax.annotate(text, at, xycoords="axes points", ha=ha, va="baseline", fontsize=name_size, color=col, zorder=6,
                                 fontweight="bold" if bold else "normal", linespacing=1.05))
        owners.append(None)
    text, at, ha = DAGGER_NOTE
    texts.append(ax.annotate(text, at, xycoords="axes points", ha=ha, va="baseline", fontsize=size, color=cm.MUTED, zorder=6, linespacing=1.1))
    owners.append(None)
    ax.set_xlabel("protein prior π", fontsize=6.5, labelpad=2.0)
    ax.set_ylabel("Δα (liver 4 species)", fontsize=6.5, labelpad=1.0)
    ax.tick_params(labelsize=6.0, length=2.5, width=0.5, pad=1.5)
    ax.set_xlim(*xlim)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.5", "0.6", "0.7", "0.8", "0.9", "1"])
    ax.set_ylim(*ylim)
    ticks = np.arange(-6, 3) / 10
    ax.set_yticks(ticks, ["0" if v == 0 else f"{v:+.1f}".replace("-", "−") for v in ticks])
    top.hist(g.prior, bins=60, color="#BBBBBB", edgecolor="none")
    right.hist(g.delta_alpha, bins=60, color="#BBBBBB", edgecolor="none", orientation="horizontal")
    for a in (top, right):  # margin plots: the bars alone
        a.axis("off")
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    # No symbol or group name on a marker (the marker, its edge and 0.5 bp of air), on another symbol or name, or outside the page.
    fig.canvas.draw()
    boxes = [Text.get_window_extent(t) for t in texts]  # the words alone, without the line to the marker
    points = ax.transData.transform(nm[["prior", "delta_alpha"]].to_numpy())
    reach = (strong / 2 + 0.6 + 0.5) * fig.dpi / 72
    clash = []
    for i, b in enumerate(boxes):
        on = [nm.symbol.iloc[j] for j, (x, y) in enumerate(points) if np.hypot(max(b.x0 - x, 0, x - b.x1), max(b.y0 - y, 0, y - b.y1)) < reach]
        on += [texts[j].get_text() for j in range(i) if b.overlaps(boxes[j])]
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > fig.bbox.y1:
            on.append("the edge of the page")
        if on:
            clash.append(f"{texts[i].get_text()!r} lies on {', '.join(on)}")
    assert not clash, "; ".join(clash)
    return fig


CT_GROUPS = [("hepatocytes", ["Hepatocytes"], "#E69F00"), ("endothelial", ["Endothelial"], "#56B4E9"),
             ("stellate / fibroblasts", ["fibroblasts"], "#009E73"),
             ("Kupffer cells and macrophages", ["Kupffer cells", "Macrophages"], "#CC79A7"),
             ("other cell types", None, "#BBBBBB")]


def panel_drivers() -> plt.Figure:
    """Cell-type sources of the count evidence (``main_delta_alpha_liver/celltype_drivers``): for every gene named on the
    landscape panel, the three bold ones included, the contribution of each cell type to its count evidence $S_g$
    (stacked, positive to the right, negative to the left; tick, the total), genes ordered by Δα (in parentheses).

    Reads ``results/analysis/T2_5/liv_4sp__genes.parquet`` and ``liv_4sp__C_gk.parquet`` (contribution of each cell type
    to the margin of each gene, and the weight of each cell type). The contributions of the cell types of a group are
    summed for the bar; nothing else is computed.

        python -m analysis.tasks.T8_4_figures.fig4_additions --panel main_delta_alpha_liver/celltype_drivers     # in chimera_env, from the repository root
    """
    g = genes_4sp()
    nm = pd.concat([named(g), every_check(g)]).sort_values("delta_alpha")
    C = pd.read_parquet(T25 / "liv_4sp__C_gk.parquet")
    C = C[C.var_name.isin(nm.var_name)]
    named_types = {t for _, ts, _ in CT_GROUPS if ts for t in ts}
    W = C.groupby("cell_type").weight_k.first()
    fig, ax = plt.subplots(figsize=(3.25, 5.05), layout="constrained")
    for y, r in enumerate(nm.itertuples()):
        c = C[C.var_name == r.var_name].set_index("cell_type").C_gk_meanz
        pos = neg = 0.0
        for name, ts, col in CT_GROUPS:
            v = c[[t for t in c.index if t not in named_types]].sum() if ts is None else c.reindex(ts).fillna(0).sum()
            if v >= 0:
                ax.barh(y, v, left=pos, color=col, height=0.72, lw=0)
                pos += v
            else:
                ax.barh(y, v, left=neg, color=col, height=0.72, lw=0)
                neg += v
        ax.plot([c.sum()], [y], marker="|", ms=5, color=cm.INK, mew=0.9, zorder=4)
    ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=0)
    ax.set_yticks(range(len(nm)))
    ax.set_yticklabels([f"{r.label}  ({signed(r.delta_alpha)})" for r in nm.itertuples()], fontsize=5.6)
    for tick, r in zip(ax.get_yticklabels(), nm.itertuples()):  # bold, as on the landscape panel
        tick.set_fontweight("bold" if r.symbol in EVERY_CHECK else "normal")
    ax.set_ylim(len(nm) - 0.4, -0.6)
    ax.set_xlabel("contribution to count evidence $S_g$, by cell type\n(liver 4 species; tick: total $S_g$)",
                  fontsize=6.5)
    hs = [Patch(color=col, label=f"{name} (weight {W.reindex(ts).sum():.2f})" if ts else name)
          for name, ts, col in CT_GROUPS]
    fig.legend(handles=hs, loc="outside lower center", ncol=2, fontsize=5.6, handlelength=1.0,
               columnspacing=0.8)
    ax.text(0.01, 0.01, f"(Δα in parentheses)   {DAGGER} no margin support", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=5.6, color=cm.MUTED)
    return fig


# The seven hepatocyte catabolic enzymes that the text names as conserved proteins used at different levels.
ENZYMES = ["ALAS1", "ACOX1", "PCK1", "PDK4", "HAL", "GOT1", "ALDOB"]
ENZYME_PANEL = "main_validation/celltype_drivers_enzymes"


def panel_drivers_enzymes() -> plt.Figure:
    """Main validation figure, panel c (``main_validation/celltype_drivers_enzymes``, 162 x 120 bp): the cell types
    behind the count evidence of seven hepatocyte enzymes. It is ``panel_drivers`` for the genes of ``ENZYMES``, drawn small.

    What it draws. One bar per gene on four-species liver: the count evidence S_g split exactly into the
    contributions of the cell types (posterior-mean z), stacked from zero, negative parts to the left and positive
    parts to the right, in the colors of the liver cell types of ``figures/style.py``; the black tick is the sum, the
    count evidence of the gene. For these seven genes nearly all of it comes from hepatocytes. The key names the
    groups of ``CT_GROUPS`` in the words of ``figures/style.py`` ("stellate and fibroblasts").

    Reads ``results/analysis/T2_5/liv_4sp__C_gk.parquet`` and nothing else (columns ``symbol``, ``cell_type``,
    ``C_gk_meanz``): the rows of the seven genes, summed within the cell-type groups of ``CT_GROUPS`` as
    ``panel_drivers`` does. No other value is computed. The panel stops unless hepatocytes carry the largest part
    of every gene, which is what the text says.

        python -m analysis.tasks.T8_4_figures.fig4_additions --panel main_validation/celltype_drivers_enzymes     # in chimera_env, from the repository root
    """
    C = pd.read_parquet(T25 / "liv_4sp__C_gk.parquet")
    C = C[C.symbol.isin(ENZYMES)]
    assert set(C.symbol) == set(ENZYMES), f"liv_4sp__C_gk.parquet lacks {sorted(set(ENZYMES) - set(C.symbol))}"
    named_types = {t for _, ts, _ in CT_GROUPS if ts for t in ts}
    W, H = 162.24, 120.0  # the width is on the 0.24 bp grid of the layout tool
    left, right, bottom, top = 29.0, 5.0, 44.0, 31.0
    fig = plt.figure(figsize=(cm.style.inch(W), cm.style.inch(H)))
    ax = fig.add_axes([left / W, bottom / H, (W - left - right) / W, (H - bottom - top) / H])
    for y, sym in enumerate(ENZYMES):
        c = C[C.symbol == sym].set_index("cell_type").C_gk_meanz
        parts = [(c[[t for t in c.index if t not in named_types]].sum() if ts is None else c.reindex(ts).fillna(0).sum(), col) for _, ts, col in CT_GROUPS]
        assert abs(parts[0][0]) == max(abs(v) for v, _ in parts), f"{sym}: hepatocytes do not carry the largest part of the count evidence"
        pos = neg = 0.0
        for v, col in parts:
            if v >= 0:
                ax.barh(y, v, left=pos, color=col, height=0.7, lw=0)
                pos += v
            else:
                ax.barh(y, v, left=neg, color=col, height=0.7, lw=0)
                neg += v
        ax.plot([c.sum()], [y], marker="|", ms=5.5, color=cm.INK, mew=0.9, zorder=4)
    ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=0)
    ax.set_yticks(range(len(ENZYMES)))
    ax.set_yticklabels(ENZYMES, fontsize=6.0)
    ax.set_ylim(len(ENZYMES) - 0.45, -0.55)
    ax.set_xlim(-0.42, 0.06)
    ax.set_xticks([-0.4, -0.3, -0.2, -0.1, 0])
    ax.set_xticklabels(["−0.4", "−0.3", "−0.2", "−0.1", "0"])
    ax.tick_params(labelsize=6.0, length=2.5, width=0.5, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.text(0.5, (bottom - 16.6) / H, "contribution to count evidence $S_g$ (tick: total)", ha="center", va="baseline", fontsize=6.5, color=cm.INK)
    fig.text(2.0 / W, (H - 1.2) / H, "Cell types behind\nthe count evidence", ha="left", va="top", fontsize=6.5, fontweight="bold", color=cm.INK, linespacing=1.12)
    fig.text(2.0 / W, (H - 22.6) / H, "seven hepatocyte enzymes, liver 4 species", ha="left", va="baseline", fontsize=5.5, color=cm.INK)
    # The key, in the order of the stack down the first column and on into the second; the two long names share the second column.
    names = [name.replace(" / ", " and ") for name, _, _ in CT_GROUPS]
    assert set(names[:4]) == set(cm.style.LIVER_CELL_TYPE) - {"other"}, "the key does not use the names of figures/style.py"
    hs = [Patch(facecolor=col, edgecolor="none", label=name) for name, (_, _, col) in zip(names, CT_GROUPS)]
    order = [0, 1, 4, 2, 3]  # hepatocytes, endothelial, other cell types | stellate and fibroblasts, Kupffer cells and macrophages
    fig.legend(handles=[hs[i] for i in order], loc="lower left", bbox_to_anchor=(2.0 / W, 1.2 / H), ncol=2, fontsize=6.0, handlelength=0.85, handleheight=0.7,
               columnspacing=0.9, handletextpad=0.35, labelspacing=0.16, frameon=False, borderpad=0, borderaxespad=0)
    return fig


MS_COHORTS = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp", "test_hmac", "test_hmouse", "test_3sp"]
TISSUES = ["liver", "testis", "bone marrow", "kidney", "islet", "eye"]


def cohort_rows() -> pd.DataFrame:
    """Every cohort of H1 in tissue order: cohort, tissue, label, first kept run."""
    runs = pd.read_csv(cm.RES / "H1" / "runs.tsv", sep="\t")
    runs = runs[runs.kept.astype(str) == "True"].sort_values("seed").groupby("cohort", sort=False).first().reset_index()
    table = pd.read_csv(cm.RES / "H1" / "cohort_table.csv").set_index("cohort")
    labels = table.label
    runs = runs.set_index("cohort").reindex(table.index).reset_index()
    runs["label"] = [cm.style.cohort_name(c if c in cm.COHORT else labels[c]) for c in runs.cohort]
    runs = runs[runs.tissue.isin(TISSUES)]  # H1 also lists the C1 cohorts (cortex, lung, muscle, tumor); the margin-support panel keeps the 16 until its layout is extended
    runs["order"] = [TISSUES.index(t) for t in runs.tissue]
    return runs.sort_values("order", kind="stable").reset_index(drop=True)


def _run_tail(c: str, run: str) -> pd.DataFrame:
    """One run: Delta-alpha against the prior it anchors on, S_g_ema, margin flag of T2.5."""
    sys.path.insert(0, str(cm.REPO / "analysis" / "tasks" / "T2_5_celltype_margin"))
    import margin as MG

    if c in MS_COHORTS:
        g = pd.read_parquet(T25 / f"{c}__genes.parquet")
        return g[g["tail"]][["delta_alpha", "margin_support"]]
    with np.load(cm.REPO / "training" / "runs" / run / "final_state.npz", allow_pickle=True) as z:
        da, S = z["alpha"] - z["prior"], z["S_g_ema"]
    t = pd.DataFrame(dict(delta_alpha=da, margin_support=MG.margin_flag(da, S)))
    return t[np.abs(da) > MG.TAIL]


def panel_margin_support() -> plt.Figure:
    h = pd.read_csv(T25 / "headline_margin_support.csv")
    h = h[(h.run_label == h.cohort) & (h.group_key == "cell_type")].set_index("cohort")
    rows = cohort_rows()
    # Exact size and axes, shared with the sequence-axis panel below it.
    pitch = 12.2 / 72
    H = pitch * len(rows) + 0.66
    fig = plt.figure(figsize=(3.4, H))
    ax = fig.add_axes([1.06 / 3.4, 0.48 / H, 2.3 / 3.4, pitch * len(rows) / H])
    source = []
    for i, r0 in enumerate(rows.itertuples()):
        c = r0.cohort
        t = _run_tail(c, r0.run)
        neg, pos = t[t.delta_alpha < 0], t[t.delta_alpha > 0]
        for k, (sub, col, name) in enumerate([(neg, "#0072B2", "negative tail (Δα < −0.05)"),
                                               (pos, "#D55E00", "positive tail (Δα > +0.05)")]):
            y = i + (k - 0.5) * 0.46
            source.append(dict(cohort=c, run=r0.run, tail=name[:8], n=len(sub),
                               frac=sub.margin_support.mean() if len(sub) else np.nan))
            if len(sub) == 0:
                ax.text(0.01, y, "no genes", va="center", fontsize=5.9, color=cm.MUTED)
                continue
            f = sub.margin_support.mean()
            ax.barh(y, f, height=0.42, color=col, lw=0, label=name if not ax.containers or len(ax.containers) < 2 else None)
            ax.text(f + 0.01, y, f"{f:.2f}  (n = {len(sub):,})", va="center", fontsize=5.9, color=cm.INK)
        if c in MS_COHORTS:
            assert abs(h.loc[c, "frac_tail_with_margin_support"] - t.margin_support.mean()) < 1e-9
    pd.DataFrame(source).to_csv(cm.RES / "T8_4" / "fig4_margin_support_source.csv", index=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows.label, fontsize=5.8)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i in np.where(np.diff(rows.order))[0]:
        ax.axhline(i + 0.5, color=cm.RULE, lw=0.4)
    ax.set_xlim(0, 1.5)  # room for the numbers beside the bars
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("fraction of tail genes whose Δα is backed by\ncount evidence of the same sign (margin support)",
                  fontsize=6, labelpad=1)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.9, handlelength=1.0, frameon=False,
               borderpad=0, columnspacing=0.8, handletextpad=0.3)
    return fig


PAIRS = [("liv_hmonkey", "test_hmac", "H+Monkey vs H+Macaque", "liver Human+Monkey vs testis Human+Macaque"),
         ("liv_hmouse", "test_hmouse", "H+Mouse vs H+Mouse", "liver Human+Mouse vs testis Human+Mouse")]


def _centred(c: str) -> pd.DataFrame:
    g = pd.read_parquet(T25 / f"{c}__genes.parquet").dropna(subset=["hsapiens_gene_id"])
    g = g.drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    g["da"] = g.delta_alpha_raw - g.delta_alpha_raw.median()
    return g


def panel_cross_tissue() -> plt.Figure:
    ref = pd.read_csv(T23 / "tissue_comparison.csv")
    ref = ref[ref.version == "raw_centred"].set_index("pair")
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 1.95), layout="constrained")
    for ax, (lc, tc, key, title) in zip(axes, PAIRS):
        L, T = _centred(lc), _centred(tc)
        ix = L.index.intersection(T.index)
        x, y = L.loc[ix, "da"], T.loc[ix, "da"]
        rho = spearmanr(x, y).statistic
        r = ref.loc[key]
        assert abs(rho - r.gene_rho) < 1e-9 and len(ix) == r.n_genes, (key, rho, r.gene_rho, len(ix))
        cm.fd.density_scatter(ax, x, y, s=1.4, alpha=0.2, color="#888888")
        mt = L.loc[ix, "symbol"].astype(str).str.startswith("MT-")
        ax.scatter(x[mt], y[mt], s=14, color="#5B3E80", edgecolor="white", lw=0.4, zorder=4,
                   label="mitochondrially encoded genes")
        ax.axhline(0, color=cm.MUTED, lw=0.5)
        ax.axvline(0, color=cm.MUTED, lw=0.5)
        lim = 1.08 * max(np.nanquantile(np.abs(np.r_[x, y]), 0.995), np.abs(np.r_[x[mt], y[mt]]).max())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Δα in liver (raw prior, centered)", fontsize=6.5)
        ax.set_title(title, fontsize=6.3)
        cm.fd.annotate_stat(ax, f"Spearman ρ = {signed(rho, 3)}, n = {len(ix):,}\nsign differs in {r.flip_rate_observed:.0%}"
                                f" of genes\n(expected if unrelated: {r.flip_rate_expected:.0%})", "lower right")
    axes[0].set_ylabel("Δα in testis (raw prior, centered)", fontsize=6.5)
    axes[0].legend(loc="upper left", fontsize=5.5, handletextpad=0.2, borderaxespad=0.1)
    return fig


def main(only=None) -> list:
    """Draw the panels of this script; ``only``: panel names ("family/name") to draw just those."""
    cm.setup()
    out = []
    if not only or LANDSCAPE_PANEL in only:  # at its exact size, with its preview beside those of the other panels
        fig = panel_landscape()
        out.append(cm.style.save_panel(fig, LANDSCAPE_PANEL, exact=True, preview=False, close=False))
        with plt.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(cm.PREVIEW / "fig4c_prior_landscape.png", format="png", dpi=200)
        plt.close(fig)
    for name, fn in [("fig4_celltype_drivers", panel_drivers), ("fig4_margin_support", panel_margin_support), ("fig4_cross_tissue", panel_cross_tissue)]:
        if not only or cm.style.panel_for(name) in only:
            out.append(cm.save(fn(), name, exact=name == "fig4_margin_support"))
    if not only or ENZYME_PANEL in only:  # at its exact size; the preview goes to figures/_work/panels/
        out.append(cm.style.save_panel(panel_drivers_enzymes(), ENZYME_PANEL, exact=True))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs="+", metavar="NAME", help="draw only these panels (family/name, as in figures/panels/)")
    for p in main(ap.parse_args().panel):
        print(p, cm.check_pdf(p))
