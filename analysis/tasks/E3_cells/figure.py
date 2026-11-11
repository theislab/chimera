"""E3 panels: home compartments, cell lineages and species-specific departures. Nine panels of four figures, saved
through figures/style.py to figures/panels/<figure>/; reads the result tables of results/analysis/E3 only.

Every panel is drawn at its size on the page and saved at that exact size, so the figures compose at scale 1:
main_specialized_cells (compartments, pattern_vs_level), supp_cell_compartments (compartments, pattern_vs_level),
supp_cell_lineages (lineages, named) and supp_primate_branches (branch, mtg, human_lineage). The two panels of the
main figure show the data of the two panels of supp_cell_compartments, drawn small enough to share one row.

    PYTHONPATH=$PWD python -m analysis.tasks.E3_cells.figure [panel ...]     # every panel, or the named ones, from the cached tables (draw_panels)
    CHIMERA_FIGURES_OUT=<folder> ... the same command                         # dry run: panels under <folder>/panels
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
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from analysis.tasks.E3_cells import rules as R  # noqa: E402
from analysis.tasks.H1_harmonise import rules as RU  # noqa: E402
from figures import style  # noqa: E402

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this task
E = R.OUT
REG = RU.cohorts()
WIDTH = style.TEXTWIDTH  # bp; a full-width panel
ROW_H = 165  # bp; height of the three panels that share the row of supp_primate_branches
S_TITLE, S_LABEL, S_TICK, S_TEXT = style.SIZE_TITLE, style.SIZE_LABEL, style.SIZE_TICK, style.SIZE_TEXT
INK, MUTED, RULE = style.INK, style.MUTED, style.RULE
NA_CELL = "#F2F2F2"  # heat-map cell without a value
GATE, PER_GENE, PROFILE = style.READOUT["delta_alpha"], style.READOUT["delta_alpha_per_gene"], style.READOUT["profile_correlation"]
LEVEL = style.Mark("#999999", "v", "level similarity at the home type (direct)")  # a direct level measure; not in the house table
# Species-specific departures: Chimera's count margin under each gate, and the direct pseudobulk analogue (keys of rates.csv, mtg_check.csv).
DEPARTURES = [("canonical", style.Mark(GATE.color, GATE.marker, "Chimera, protein-feature gate")),
              ("per-gene", style.Mark(PER_GENE.color, PER_GENE.marker, "Chimera, per-gene gate")),
              ("direct", style.Mark(PROFILE.color, PROFILE.marker, "direct pseudobulk"))]
COMP = ["specialised", "stromal", "vascular", "immune", "broad"]  # compartment keys of the result tables
COMP_LAB = {"specialised": "specialized cells", "stromal": "stromal", "vascular": "vascular", "immune": "immune", "broad": "no home type\n(broadly used)"}
COMP_COL = {"specialised": "#D55E00", "stromal": "#009E73", "vascular": "#0072B2", "immune": "#CC79A7", "broad": "#777777"}
COMP_MK = {"specialised": "o", "stromal": "s", "vascular": "^", "immune": "D"}
SPECIES = ["Human", "Chimpanzee", "Macaque", "Marmoset"]  # the four primates of the prefrontal-cortex fit, as named in the tables
CELL_TYPE = {"Micro": "microglia", "PVALB": "PVALB interneurons", "L3_5_IT": "L3-5 IT neurons"}
LEADER = dict(arrowstyle="-", lw=0.4, color=MUTED, shrinkA=0, shrinkB=1.5)  # line from a label to its point


def save(fig, name: str):
    """One panel at its exact size: figures/panels/<figure>/<name>.pdf."""
    return style.save_panel(fig, name, exact=True)


def signed(v, _=None) -> str:
    """Tick label with one decimal and a true minus sign."""
    return f"{v:.1f}".replace("-", "−")


def cohort_label(k: str) -> str:
    """Name of a cohort under a heading that already names its tissue: "DLPFC Human+Macaque, own ortholog table"."""
    name, tissue = style.cohort_name(REG[k]["label"]), REG[k]["tissue"]
    if name.lower().startswith(tissue + " "):
        name = name[len(tissue) + 1:]
    return name.replace(", own table", ",\nown ortholog table")


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axvline(0, color=RULE, lw=0.6, zorder=0)


def excess(universe="count_supported"):
    e = pd.read_csv(E / "excess.csv")
    return e[e.universe == universe]


def panel_compartments():
    """Panel a of the supplementary figure on home compartments (``supp_cell_compartments/compartments``, 486 x 150 bp;
    figure key ``supp_cell_compartments``). ``main_compartments`` draws the same rows and marks for the main figure.

    What it draws. Rows: the home compartment of a gene (specialized cells, stromal, vascular, immune, no home type).
    Four plots, one score each: Δα with the protein-feature gate, Δα with the per-gene gate, profile correlation, and
    level similarity at the home cell type. x: the excess of the genes of a compartment over matched genes, in SD
    units; one mark per cohort (filled, p < 0.01), the bar is the median over cohorts, and the gray count beside a
    row is the number of cohorts above zero.

    Reads ``results/analysis/E3/excess.csv`` and nothing else (columns ``universe``, ``level``, ``group``, ``score``,
    ``excess_d``, ``p``; rows of ``universe == count_supported`` and ``level == compartment``). The median and the
    count of cohorts above zero are taken from these rows at draw time.

        python -m analysis.tasks.E3_cells.figure supp_cell_compartments/compartments      # in chimera_env, from the repository root
    """
    e = excess()
    e = e[e.level == "compartment"]
    # the two gates first, then the direct baselines; a count of cohorts above zero beside every row
    cols = [("D", GATE.name, GATE), ("D_free", PER_GENE.name, PER_GENE), ("B1", PROFILE.name, PROFILE),
            ("DL", "level similarity at the\nhome type (direct)", LEVEL)]
    fig, axes = plt.subplots(1, 4, figsize=(style.inch(WIDTH), style.inch(150)), sharey=True, layout="constrained")
    rng = np.random.default_rng(0)
    for ax, (sc, title, mark) in zip(axes, cols):
        for i, c in enumerate(COMP):
            x = e[(e.group == c) & (e.score == sc)].dropna(subset=["excess_d"])
            y = len(COMP) - 1 - i + rng.uniform(-0.18, 0.18, len(x))
            sig = (x.p < 0.01).to_numpy()
            ax.scatter(x.excess_d[sig], y[sig], s=9, color=mark.color, marker=mark.marker, lw=0, alpha=0.85)
            ax.scatter(x.excess_d[~sig], y[~sig], s=9, facecolor="none", edgecolor=mark.color, marker=mark.marker, lw=0.6, alpha=0.85)
            med = x.excess_d.median()
            ax.plot([med, med], [len(COMP) - 1 - i - 0.32, len(COMP) - 1 - i + 0.32], color=INK, lw=1.2)
            npos, n = int((x.excess_d > 0).sum()), len(x)
            ax.text(1.02, len(COMP) - 1 - i, f"{npos}/{n} > 0", transform=ax.get_yaxis_transform(), fontsize=S_TEXT, va="center", color=MUTED)
        ax.set_title(title, loc="left", fontsize=S_TITLE)
        ax.set_xlim(-0.9, 1.3)  # holds every cohort (lowest -0.84, highest 1.19)
        ax.set_xticks([-0.5, 0, 0.5, 1])
        ax.xaxis.set_major_formatter(FuncFormatter(signed))
        ax.tick_params(labelsize=S_TICK)
        _clean(ax)
    axes[0].set_yticks(range(len(COMP)), [COMP_LAB[c] for c in COMP[::-1]])
    fig.supxlabel("excess over matched genes (SD)", fontsize=S_LABEL)
    return save(fig, "supp_cell_compartments/compartments")


def home_types():
    """One row per home cell type of the thread and sensitivity cohorts (``rules.THREAD``, ``rules.SENSITIVITY``): the
    excess of its genes over matched genes for every score (columns ``D_free``, ``DL``, ...) and its compartment
    (``comp``). Rows of ``excess.csv`` with ``level == subgroup``; genes without a home type are left out."""
    e = excess()
    e = e[(e.level == "subgroup") & (e.group != "broad")]
    th = list(R.THREAD.values()) + list(R.SENSITIVITY.values())
    t = e[e.cohort.isin(th)].pivot_table(index=["cohort", "group"], columns="score", values="excess_d").dropna(subset=["D_free", "DL"]).reset_index()
    comp = {}
    for c in th:
        for k, (cp, sg) in R.type_map(REG[c]["tissue"]).items():
            comp[(c, sg)] = cp
    t["comp"] = [comp.get((c, g), "immune") for c, g in zip(t.cohort, t.group)]
    return t


def panel_pattern_vs_level():
    t = home_types()
    fig, ax = plt.subplots(figsize=(style.inch(300), style.inch(250)), layout="constrained")
    for cp in COMP[:-1]:
        s = t[t.comp == cp]
        ax.scatter(-s.DL, s.D_free, s=12, color=COMP_COL[cp], marker=COMP_MK[cp], lw=0, label=COMP_LAB[cp], alpha=0.9)
    # named subgroups: (cohort, subgroup) -> label, offset of the label from its point (points), horizontal and vertical anchor
    lab = {("liv_hmouse", "endothelial"): ("liver sinusoidal\nendothelium", (8, -2), "left", "center"),
           ("liv_hmouse", "hepatocyte"): ("hepatocyte", (-6, -9), "right", "top"),
           ("kidney", "thick ascending limb"): ("kidney thick\nascending limb", (8, -3), "left", "center"),
           ("kidney", "thin limb"): ("kidney thin limb", (9, 9), "left", "bottom"),
           ("test_hmouse", "germline: spermatid"): ("spermatid", (12, 8), "left", "bottom"),
           ("vz_hmouse", "outflow: trabecular meshwork"): ("trabecular\nmeshwork", (-4, 12), "center", "bottom"),
           ("brain_m1_hmouse", "excitatory neuron"): ("excitatory\nneuron", (-32, 14), "right", "bottom"),
           ("islet", "endocrine: beta"): ("beta cell", (10, 0), "left", "center"),
           ("brain_m1_hmouse", "microglia"): ("microglia", (10, 0), "left", "center"),
           ("muscle_ageing", "muscle stem cell"): ("muscle stem cell", (8, 10), "left", "bottom")}
    for (c, g), (txt, off, ha, va) in lab.items():
        r = t[(t.cohort == c) & (t.group == g)]
        if len(r):
            ax.annotate(txt, (-r.DL.iloc[0], r.D_free.iloc[0]), fontsize=S_TEXT, xytext=off, textcoords="offset points", color=INK, ha=ha, va=va,
                        linespacing=1.0, arrowprops=LEADER)
    ax.axhline(0, color=RULE, lw=0.6, zorder=0)
    ax.axvline(0, color=RULE, lw=0.6, zorder=0)
    ax.set_xlim(-0.52, 1.02)  # room on the right for the label of the right-most named point
    ax.set_xlabel("direct level gap at the home type,\nexcess over matched genes (SD)", fontsize=S_LABEL)
    ax.set_ylabel(f"{PER_GENE.name},\nexcess over matched genes (SD)", fontsize=S_LABEL)
    ax.xaxis.set_major_formatter(FuncFormatter(signed))
    ax.yaxis.set_major_formatter(FuncFormatter(signed))
    ax.tick_params(labelsize=S_TICK)
    ax.legend(frameon=False, fontsize=S_TICK, loc="upper right", handletextpad=0.2, borderaxespad=0.1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return save(fig, "supp_cell_compartments/pattern_vs_level")


# ---------------------------------------------------------------- the main figure on specialized cells: two panels in one row
MAIN_H = 134.0  # bp; height of both panels
MAIN_TOP, MAIN_BOTTOM = 17.5, 18.5  # bp from the top and from the bottom of the page to the plot area, the same in both panels
MAIN_HEAD = (7.0, 14.2)  # bp from the top of the page: baselines of the two lines of a heading
MAIN_XTITLE = 3.1  # bp from the bottom of the page: baseline of the x title
M_HEAD, M_SUB, M_AXIS, M_ROW, M_TICK, M_NOTE = 6.5, 6.0, 6.5, 6.0, 5.5, 5.5  # pt: heading, its second line, axis title, row label, tick label, annotation
M_FLOOR = 5.0  # pt; no text below it
MAIN_ROW = {"specialised": "Specialized cells", "stromal": "Stromal", "vascular": "Vascular", "immune": "Immune", "broad": "No home type\n(broadly used)"}
MAIN_BAND = "#F2F2F2"  # behind the row of the specialized cells
MAIN_LEVEL = "#666666"  # marks of the level column
MAIN_HOME = {"specialised": ("Specialized", "o", 3.4), "stromal": ("Stromal", "s", 2.5), "vascular": ("Vascular", "^", 3.0),
             "immune": ("Immune", "D", 2.3)}  # compartment of a home cell type: name, marker, marker size (pt)
MAIN_FILLED, MAIN_OPEN = "#444444", "#999999"  # specialized cell types (filled); the other three compartments (open)
MAIN_LEADER = dict(arrowstyle="-", lw=0.5, color=MUTED, shrinkA=0, shrinkB=2.2)  # line from a label to its mark


def signed_tick(v, _=None) -> str:
    """Tick label with its sign and a true minus, without trailing zeros; zero is "0"."""
    return "0" if abs(v) < 1e-9 else f"{v:+g}".replace("-", "−")


def boxes_bp(fig, artists) -> np.ndarray:
    """Boxes (x0, y0, x1, y1) of drawn artists in bp from the lower left corner of the page; of an annotation, the text
    without its leader. Measured on a canvas of 720 dpi, where hinting no longer moves glyphs, so the boxes are those
    of the saved file."""
    dpi = fig.get_dpi()
    fig.set_dpi(720)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = np.array([(Text.get_window_extent(a, r) if isinstance(a, Text) else a.get_window_extent(r)).extents for a in artists]) * 72 / 720
    fig.set_dpi(dpi)
    return out.reshape(-1, 4)


def gap_bp(points, box) -> float:
    """Smallest distance (bp) from points (n x 2) to a box (x0, y0, x1, y1); 0 when a point lies inside."""
    x0, y0, x1, y1 = box
    return float(np.hypot(np.clip(points[:, 0], x0, x1) - points[:, 0], np.clip(points[:, 1], y0, y1) - points[:, 1]).min())


def check_text(fig, margin: float = 1.0, apart: float = 0.3) -> dict:
    """Every text of a figure is at least ``M_FLOOR`` pt, lies on the page with ``margin`` bp to spare, and no two
    texts come closer than ``apart`` bp. Returns {text artist: box in bp}."""
    fig.canvas.draw()  # tick labels exist after a draw
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    small = [t.get_text() for t in texts if t.get_fontsize() < M_FLOOR]
    assert not small, f"text under {M_FLOOR} pt: {small}"
    b = boxes_bp(fig, texts)
    w, h = fig.get_size_inches() * 72
    off = [t.get_text() for t, (x0, y0, x1, y1) in zip(texts, b, strict=True) if x0 < margin or y0 < margin or x1 > w - margin or y1 > h - margin]
    assert not off, f"text off the page: {off}"
    near = [(texts[i].get_text(), texts[j].get_text()) for i in range(len(texts)) for j in range(i + 1, len(texts))
            if min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0]) > -apart and min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1]) > -apart]
    assert not near, f"texts collide: {near}"
    return dict(zip(texts, b, strict=True))


def main_compartments():
    """Panel a of the main figure on specialized cells (``main_specialized_cells/compartments``, 359 x 134 bp; figure
    key ``main_specialized_cells``).

    What it draws. The rows, scores and marks of ``panel_compartments`` at the size of one row beside a second panel.
    Rows: the home compartment of a gene; the row of the specialized cells stands on a gray band, with its label and
    its counts in bold. Four plots, one score each, under a heading of two lines: Δα with the protein-feature gate, Δα
    with the per-gene gate, profile correlation, and level similarity at the home cell type. x: the excess of the
    genes of a compartment over matched genes, in SD units; one mark per cohort (filled, p < 0.01), the bar is the
    median over cohorts, and the count beside a row is the number of cohorts above zero out of the cohorts with that
    compartment. The marks keep the places they have in ``panel_compartments`` (same jitter).

    Reads ``results/analysis/E3/excess.csv`` and nothing else (rows of ``universe == count_supported`` and
    ``level == compartment``). Checks at draw time: every value lies inside the x range; the counts of the first row
    are 23/24, 24/24, 18/24 and 2/24; no text is under 5.0 pt, off the page or on another text.

        python -m analysis.tasks.E3_cells.figure main_specialized_cells/compartments      # in chimera_env, from the repository root
    """
    e = excess()
    e = e[e.level == "compartment"]
    cols = [("D", "Δα", "protein-feature gate", GATE.color, GATE.marker), ("D_free", "Δα", "per-gene gate", PER_GENE.color, PER_GENE.marker),
            ("B1", "Profile", "correlation", PROFILE.color, PROFILE.marker), ("DL", "Level similarity", "at the home type", MAIN_LEVEL, LEVEL.marker)]
    W, H = 359.0, MAIN_H
    # bp: the first axis, the width of an axis, from one axis to the next, from the end of an axis to the right edge of its count
    x0, aw, step, count = 62.0, 51.5, 74.25, 19.5
    ah, xlim = H - MAIN_TOP - MAIN_BOTTOM, (-0.9, 1.3)  # the x range holds every cohort (lowest -0.84, highest 1.19)
    fig = plt.figure(figsize=(style.inch(W), style.inch(H)))
    fig.add_artist(Rectangle((1.0 / W, (MAIN_BOTTOM + ah * (1 - 1 / len(COMP))) / H), (W - 2.0) / W, ah / len(COMP) / H, facecolor=MAIN_BAND,
                             edgecolor="none", zorder=-1))
    rng = np.random.default_rng(0)  # the jitter of panel_compartments, drawn in the same order
    first = []  # the counts of the first row
    for k, (sc, head, sub, color, marker) in enumerate(cols):
        ax = fig.add_axes([(x0 + k * step) / W, MAIN_BOTTOM / H, aw / W, ah / H], facecolor="none")
        for i, c in enumerate(COMP):
            x = e[(e.group == c) & (e.score == sc)].dropna(subset=["excess_d"])
            assert x.excess_d.between(*xlim).all(), f"{sc}, {c}: a value outside the x range {xlim}"
            y0 = len(COMP) - 1 - i
            y = y0 + rng.uniform(-0.18, 0.18, len(x))
            sig = (x.p < 0.01).to_numpy()
            ax.scatter(x.excess_d[sig], y[sig], s=9, color=color, marker=marker, lw=0, alpha=0.85)
            ax.scatter(x.excess_d[~sig], y[~sig], s=9, facecolor="none", edgecolor=color, marker=marker, lw=0.6, alpha=0.85)
            med = x.excess_d.median()
            ax.plot([med, med], [y0 - 0.32, y0 + 0.32], color=INK, lw=1.2)
            n = f"{int((x.excess_d > 0).sum())}/{len(x)}"
            if i == 0:
                first.append(n)
            ax.text((aw + count) / aw, y0, n, transform=ax.get_yaxis_transform(), ha="right", va="center_baseline", fontsize=M_ROW if i == 0 else M_NOTE,
                    fontweight="bold" if i == 0 else "normal", color=INK if i in (0, len(COMP) - 1) else MUTED)
        ax.axvline(0, color=RULE, lw=0.6, zorder=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.5, len(COMP) - 0.5)
        ticks = [-0.5, 0, 0.5, 1]
        ax.set_xticks(ticks, [signed_tick(v) if v != 0.5 else "" for v in ticks])  # the tick at +0.5 stays, without a label: beside "+1" the two would touch
        ax.set_yticks(range(len(COMP)) if k == 0 else [], [MAIN_ROW[c] for c in COMP[::-1]] if k == 0 else [])
        ax.tick_params(length=2.5, width=0.5, pad=1.5, labelsize=M_TICK)
        for j, label in enumerate(ax.get_yticklabels()):  # rows from the bottom; the first row in bold
            label.set(fontsize=M_ROW, fontweight="bold" if j == len(COMP) - 1 else "normal", linespacing=1.05)
        for side, spine in ax.spines.items():  # one y axis for the four plots: a count stands beside its own plot, not against the axis of the next
            spine.set_visible(side == "bottom" or (side == "left" and k == 0))
            spine.set_linewidth(0.6)
        fig.text((x0 + k * step) / W, 1 - MAIN_HEAD[0] / H, head, fontsize=M_HEAD, fontweight="bold", ha="left", va="baseline")
        fig.text((x0 + k * step) / W, 1 - MAIN_HEAD[1] / H, sub, fontsize=M_SUB, ha="left", va="baseline")
    assert first == ["23/24", "24/24", "18/24", "2/24"], f"cohorts above zero, specialized cells: {first}"
    right = x0 + (len(cols) - 1) * step + aw  # end of the last axis
    fig.text((x0 + right) / 2 / W, MAIN_XTITLE / H, "excess over matched genes (SD)", fontsize=M_AXIS, ha="center", va="baseline")
    fig.text((right + count) / W, MAIN_XTITLE / H, "n/N: cohorts above 0", fontsize=M_NOTE, color=MUTED, ha="right", va="baseline")
    check_text(fig)
    return save(fig, "main_specialized_cells/compartments")


def main_pattern_vs_level():
    """Panel b of the main figure on specialized cells (``main_specialized_cells/pattern_vs_level``, 120 x 134 bp; figure
    key ``main_specialized_cells``).

    What it draws. One mark per home cell type of the thread and sensitivity cohorts, the data of
    ``panel_pattern_vs_level`` in grays. x: level similarity at the home cell type, with the sign of the last plot of
    ``main_compartments`` (``panel_pattern_vs_level`` plots its negative, the level gap); y: Δα with the per-gene gate;
    both as the excess of the genes of the cell type over matched genes, in SD units. Specialized cell types are filled
    circles, drawn last; stromal, vascular and immune cell types are open marks. Six cell types are named. The plot
    area starts and ends at the heights of ``main_compartments``.

    Reads ``results/analysis/E3/excess.csv`` and nothing else (``home_types``). Checks at draw time: every value lies
    inside the axes; no text is under 5.0 pt, off the page or on another text; no name lies on a mark or on the key.

        python -m analysis.tasks.E3_cells.figure main_specialized_cells/pattern_vs_level      # in chimera_env, from the repository root
    """
    t = home_types()
    W, H = 120.0, MAIN_H
    left, right, xlim, ylim = 27.5, 2.0, (-0.9, 0.5), (-0.34, 1.04)
    assert t.DL.between(*xlim).all() and t.D_free.between(*ylim).all(), "a home cell type outside the axes"
    fig = plt.figure(figsize=(style.inch(W), style.inch(H)))
    ax = fig.add_axes([left / W, MAIN_BOTTOM / H, (W - left - right) / W, (H - MAIN_TOP - MAIN_BOTTOM) / H])
    ax.axhline(0, color=RULE, lw=0.6, zorder=0)
    ax.plot([0, 0], [-0.22, 0.84], color=RULE, lw=0.6, zorder=0)  # from the names at the bottom to the key
    for cp in ("stromal", "vascular", "immune"):
        s = t[t.comp == cp]
        ax.scatter(s.DL, s.D_free, s=MAIN_HOME[cp][2] ** 2, facecolor="none", edgecolor=MAIN_OPEN, marker=MAIN_HOME[cp][1], lw=0.6)
    s = t[t.comp == "specialised"]  # the marks that carry the finding, on top, each with a white edge
    ax.scatter(s.DL, s.D_free, s=MAIN_HOME["specialised"][2] ** 2, facecolor=MAIN_FILLED, edgecolor="white", marker="o", lw=0.4, zorder=3)
    # named cell types: (cohort, home cell type) -> name, offset of the name from its mark (bp), anchor of the name, start of the leader on the name
    named = {("liv_hmouse", "endothelial"): ("liver sinusoidal\nendothelium", (-12, -3.5), "left", "top", None),
             ("test_hmouse", "germline: spermatid"): ("spermatid", (-11.4, 9.9), "right", "center", (1, 0.5)),
             ("vz_hmouse", "outflow: trabecular meshwork"): ("trabecular\nmeshwork", (2.1, 17.3), "right", "bottom", (0.95, 0)),
             ("brain_m1_hmouse", "excitatory neuron"): ("excitatory\nneuron", (-41.2, 6.1), "left", "top", (1, 0.75)),
             ("liv_hmouse", "hepatocyte"): ("hepatocyte", (6.5, -12.0), "left", "top", (0.03, 1)),
             ("islet", "endocrine: beta"): ("beta cell", (0, -3.5), "center", "top", None)}
    names = []
    for (c, g), (txt, off, ha, va, start) in named.items():
        r = t[(t.cohort == c) & (t.group == g)]
        assert len(r) == 1, f"{c}, {g}: not one home cell type"
        names.append(ax.annotate(txt, (r.DL.iloc[0], r.D_free.iloc[0]), xytext=off, textcoords="offset points", fontsize=M_NOTE, color=INK, ha=ha, va=va,
                                 linespacing=1.05, zorder=4, arrowprops=dict(MAIN_LEADER, relpos=start) if start else None,
                                 bbox=dict(facecolor="white", edgecolor="none", pad=0.4)))  # a zero line stops at a name
    handles = [Line2D([], [], ls="", marker=mk, ms=ms - (0.4 if cp == "specialised" else 0), mew=0 if cp == "specialised" else 0.6,
                      mfc=MAIN_FILLED if cp == "specialised" else "none", mec=MAIN_OPEN, label=name) for cp, (name, mk, ms) in MAIN_HOME.items()]
    key = ax.legend(handles=handles, ncol=2, loc="upper left", bbox_to_anchor=(0, 1), frameon=False, fontsize=M_NOTE, borderpad=0, borderaxespad=0.1,
                    handlelength=0.9, handletextpad=0.25, labelspacing=0.15, columnspacing=1.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([-0.8, -0.4, 0, 0.4])
    ax.set_yticks([0, 0.4, 0.8])
    ax.xaxis.set_major_formatter(FuncFormatter(signed_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(signed_tick))
    ax.tick_params(labelsize=M_TICK, length=2.5, width=0.5, pad=1.5)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_linewidth(0.6)
    fig.text(left / W, 1 - MAIN_HEAD[0] / H, "By home cell type", fontsize=M_HEAD, fontweight="bold", ha="left", va="baseline")
    fig.text((left + (W - left - right) / 2) / W, MAIN_XTITLE / H, "level similarity (excess, SD)", fontsize=M_AXIS, ha="center", va="baseline")
    fig.text(2.0 / W, (MAIN_BOTTOM + (H - MAIN_TOP - MAIN_BOTTOM) / 2) / H, "Δα, per-gene gate (excess, SD)", fontsize=M_AXIS, rotation=90, ha="left",
             va="center")
    # no name on another text (check_text), on a mark or on the key; no mark on the key. A mark reaches 1.9 bp from its center.
    box = check_text(fig)
    marks = ax.transData.transform(t[["DL", "D_free"]].to_numpy()) * 72 / fig.get_dpi()
    kx0, ky0, kx1, ky1 = boxes_bp(fig, [key])[0]
    for a in names:
        bx0, by0, bx1, by1 = box[a]
        assert gap_bp(marks, box[a]) > 2.4, f"{a.get_text()!r} lies on a mark"
        assert bx0 > kx1 or bx1 < kx0 or by0 > ky1 or by1 < ky0, f"{a.get_text()!r} lies on the key"
    assert gap_bp(marks, (kx0, ky0, kx1, ky1)) > 3.0, "a mark lies on the key"
    return save(fig, "main_specialized_cells/pattern_vs_level")


LINEAGE_ROWS = [  # tissue, its home-type groups (top to bottom), x ticks
    ("liver", ["endothelial", "stellate/fibroblast", "macrophage", "cholangiocyte", "hepatocyte"], [0, 0.5, 1]),
    ("eye", ["outflow: trabecular meshwork", "ciliary muscle", "Schwann cell", "melanocyte", "cornea", "outflow: Schlemm canal"], [0, 0.2, 0.4]),
    ("cortex", ["excitatory neuron", "inhibitory neuron", "oligodendrocyte", "astrocyte", "microglia"], [-0.2, 0, 0.2]),
    ("testis", ["germline: spermatocyte", "germline: spermatid", "Sertoli", "germline: spermatogonia"], [-0.2, 0, 0.2, 0.4]),
]


def panel_lineages():
    """Home-type groups of four tissues: excess over matched genes in every cohort of the tissue, per-gene gate and profile correlation."""
    e = excess()
    e = e[e.level == "subgroup"]
    fig, axes = plt.subplots(1, 4, figsize=(style.inch(WIDTH), style.inch(150)), layout="constrained")
    for ax, (tis, groups, ticks) in zip(axes, LINEAGE_ROWS):
        cs = [c for c in REG if REG[c]["tissue"] == tis]
        for i, g in enumerate(groups):
            y0 = len(groups) - 1 - i
            for sc, mark, dy in (("D_free", PER_GENE, 0.13), ("B1", PROFILE, -0.13)):
                x = e[(e.group == g) & (e.score == sc) & e.cohort.isin(cs)].excess_d
                ax.scatter(x, np.full(len(x), y0 + dy), s=8, color=mark.color, marker=mark.marker, lw=0, alpha=0.8)
                if len(x):
                    ax.plot([x.median()] * 2, [y0 + dy - 0.12, y0 + dy + 0.12], color=mark.color, lw=1.1)
        ax.set_yticks(range(len(groups)), [g.replace("outflow: ", "").replace("germline: ", "") for g in groups[::-1]], fontsize=S_TICK)
        ax.set_title(f"{style.TISSUE[tis].name} ({len(cs)} cohorts)", loc="left", fontsize=S_TITLE)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(signed))
        ax.tick_params(axis="x", labelsize=S_TICK)
        _clean(ax)
    fig.supxlabel("excess over matched genes (SD)", fontsize=S_LABEL)
    handles = [Line2D([], [], ls="", marker=m.marker, color=m.color, ms=3.5, label=m.name) for m in (PER_GENE, PROFILE)]
    fig.legend(handles=handles, frameon=False, fontsize=S_TICK, loc="outside upper right", ncol=2, handletextpad=0.3, borderaxespad=0.2)
    return save(fig, "supp_cell_lineages/lineages")


NAMED = [("liv_hmouse", "endothelial", "Liver,\nsinusoidal endothelium"), ("vz_full", "outflow: trabecular meshwork", "Eye,\ntrabecular meshwork"),
         ("brain_m1_hmouse", "inhibitory neuron", "Cortex,\ninhibitory neurons"), ("brain_m1_hmouse", "excitatory neuron", "Cortex,\nexcitatory neurons")]
Z_MAX = 3.0  # color limit of the z maps; cells above it take the end color and print their value


def panel_named():
    """Seven named genes of four cell types: z of the per-gene gate in every cohort of the tissue, value in every cell."""
    n = pd.read_csv(E / "named_genes.csv")
    sets = []
    for c, g, title in NAMED:
        s = n[(n.cohort == c) & (n.subgroup == g) & (n.sign == "above") & (n.rep_same_sign == 1.0) & (n.n_peer_cohorts >= 4)].head(7)
        sets.append((title, s, [k for k in REG if REG[k]["tissue"] == REG[c]["tissue"]]))
    fig, axes = plt.subplots(1, 4, figsize=(style.inch(WIDTH), style.inch(190)), layout="constrained",
                             gridspec_kw=dict(width_ratios=[len(p) for _, _, p in sets]))
    for ax, (title, s, peers) in zip(axes, sets):
        M = s[[f"z__{k}" for k in peers]].to_numpy(float)
        ax.set_facecolor(NA_CELL)
        im = ax.imshow(np.ma.masked_invalid(M), cmap=style.DIVERGING, vmin=-Z_MAX, vmax=Z_MAX, aspect="auto", interpolation="nearest")
        for (i, j), v in np.ndenumerate(M):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=S_TEXT, color="white" if abs(v) > 0.6 * Z_MAX else INK)
        ax.set_yticks(range(len(s)), s.symbol, fontsize=S_TICK)
        ax.set_xticks(range(len(peers)), [cohort_label(k) for k in peers], rotation=90, fontsize=S_TEXT, linespacing=1.0)
        ax.set_title(title, loc="right", fontsize=S_TITLE, linespacing=1.05)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.016, pad=0.01, extend="max")
    cb.set_label(f"z of {PER_GENE.name}", fontsize=S_LABEL)
    cb.ax.tick_params(labelsize=S_TICK, length=2)
    cb.outline.set_visible(False)
    return save(fig, "supp_cell_lineages/named")


def departures_legend(fig):
    """The three series of the departure panels, above the axes."""
    handles = [Line2D([], [], ls="", marker=m.marker, color=m.color, ms=3.8, label=m.name) for _, m in DEPARTURES]
    fig.legend(handles=handles, frameon=False, fontsize=S_TICK, loc="outside upper left", ncol=1, handletextpad=0.3, borderaxespad=0.2)


def panel_branch():
    """Prefrontal cortex, four primates in one fit: share of gene by cell-type pairs in which a species departs, species in the
    order of their terminal branch length (printed under each name)."""
    rates = pd.read_csv(E / "departures" / "rates.csv")
    r = rates[rates.cohort == "brain_dlpfc"].groupby(["readout", "species"])[["tested", "departures"]].sum()
    r["rate"] = r.departures / r.tested
    bl = R.BRANCH_MY["brain_dlpfc"]
    sp = sorted(SPECIES, key=lambda s: bl[s])  # stable: human and chimpanzee share one branch length
    fig, ax = plt.subplots(figsize=(style.inch(186), style.inch(ROW_H)), layout="constrained")
    for (rd, mark), dx in zip(DEPARTURES, (-0.22, 0.0, 0.22)):
        ax.plot(np.arange(len(sp)) + dx, r.loc[rd].reindex(sp).rate.to_numpy() * 100, color=mark.color, marker=mark.marker, ms=4, lw=0.7)
    ax.set_xticks(range(len(sp)), [f"{s.lower()}\n{bl[s]:g}" for s in sp], fontsize=S_TICK)
    ax.set_xlim(-0.5, len(sp) - 0.5)
    ax.tick_params(axis="y", labelsize=S_TICK)
    ax.set_xlabel("species, terminal branch length (My)", fontsize=S_LABEL)
    ax.set_ylabel("species-specific departures\n(% of gene × cell type pairs)", fontsize=S_LABEL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    departures_legend(fig)
    return save(fig, "supp_primate_branches/branch")


def panel_mtg():
    """Three checks of the departures against level gaps of a held-out cortical region: rank correlation with its interval."""
    m = pd.read_csv(E / "departures" / "mtg_check.csv")
    tests = [("marmoset minus macaque branch vs MTG gap difference", "marmoset vs macaque\nbranch"),
             ("human departures vs MTG human-specific gap", "human branch"),
             ("chimpanzee departures vs MTG human-specific gap (control)", "chimpanzee branch\n(control)")]
    fig, ax = plt.subplots(figsize=(style.inch(150), style.inch(ROW_H)), layout="constrained")
    for i, (tt, lab) in enumerate(tests):
        for j, (rd, mark) in enumerate(DEPARTURES):
            r = m[(m.test == tt) & (m.readout == rd)].iloc[0]
            y = len(tests) - 1 - i + 0.2 - j * 0.2
            ax.errorbar(r.rho, y, xerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt=mark.marker, color=mark.color, ms=3.2, lw=1.0, capsize=0)
    ax.set_yticks(range(len(tests)), [t[1] for t in tests[::-1]], fontsize=S_TICK)
    ax.set_xticks([0, 0.1, 0.2])
    ax.xaxis.set_major_formatter(FuncFormatter(signed))
    ax.tick_params(axis="x", labelsize=S_TICK)
    fig.supxlabel("ρ with held-out MTG level gaps\n(partial on expression)", fontsize=S_LABEL)
    _clean(ax)
    departures_legend(fig)
    return save(fig, "supp_primate_branches/mtg")


HUMAN_LINEAGE = ["FOXP2", "CACNA1D", "DSCAM", "MELTF", "PKD2L1"]  # the human-lineage genes drawn, each in its named cell type


def panel_human_lineage():
    """Count margin per cell of each species for five human-lineage genes in their named cell type (per-gene gate fit)."""
    m = pd.read_csv(E / "departures" / "ma2022_genes.csv")
    m = m[(m.readout == "per-gene") & m.gene.isin(HUMAN_LINEAGE)]
    fig, ax = plt.subplots(figsize=(style.inch(136), style.inch(ROW_H)), layout="constrained")
    for i, r in enumerate(m.itertuples()):
        v = dict(x.split(":") for x in r.S_rows.split(";"))
        for j, s in enumerate(SPECIES):
            ax.bar(i + (j - 1.5) * 0.19, float(v[s]), width=0.18, color=style.SPECIES[s.lower()], label=s.lower() if i == 0 else None)
    ax.set_xticks(range(len(m)), [f"{g}\n{CELL_TYPE.get(t, t)}" for g, t in zip(m.gene, m.cell_type)], fontsize=S_TEXT, rotation=90, linespacing=1.0)
    ax.axhline(0, color=RULE, lw=0.6)
    ax.set_ylabel("count margin per cell\nin the named cell type", fontsize=S_LABEL)
    ax.yaxis.set_major_formatter(FuncFormatter(signed))
    ax.tick_params(axis="y", labelsize=S_TICK)
    ax.set_ylim(-0.5, 1.75)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend(frameon=False, fontsize=S_TICK, ncol=2, loc="outside upper center", handlelength=1.0, handletextpad=0.4, columnspacing=1.0)
    return save(fig, "supp_primate_branches/human_lineage")


# panel name in figures/panels -> the function that draws it
DRAW = {"main_specialized_cells/compartments": main_compartments, "main_specialized_cells/pattern_vs_level": main_pattern_vs_level,
        "supp_cell_compartments/compartments": panel_compartments, "supp_cell_compartments/pattern_vs_level": panel_pattern_vs_level,
        "supp_cell_lineages/lineages": panel_lineages, "supp_cell_lineages/named": panel_named,
        "supp_primate_branches/branch": panel_branch, "supp_primate_branches/mtg": panel_mtg, "supp_primate_branches/human_lineage": panel_human_lineage}


def draw_panels(names=None) -> list:
    """Draw the panels of this task from the cached tables of results/analysis/E3 and nothing else. ``names``: panel names
    as in ``DRAW`` (default: all nine). Returns the files written."""
    style.apply()
    return [DRAW[n]() for n in (DRAW if names is None else names)]


if __name__ == "__main__":
    for path in draw_panels(sys.argv[1:] or None):
        print(path)
