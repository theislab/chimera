"""H1 figures, one PDF per panel in figures/panels/<family>/, composed by figures/compose.py from figures/specs/. Supplementary: one figure per tissue with the standard panels of
liver and testis (same axes, palette and order), the recurrence heat map and the lineage heat map. Main: one reading across eight tissues. The panels read results/analysis/H1/ only; no model is loaded.

Panels reuse the forms of notebooks/_alpha_master_builder.py (A1.1-A1.3), _alpha_biology_simpler_builder.py (category boxes with d and q), _pathway_builder.py (Hallmark volcano),
analysis/tasks/T8_4_figures/fig4_additions.py (landscape, cell-type drivers), X5_sequence_axis/run.py (axis scatter). House style: figures/style.py. Every panel is drawn at its size on the page.

    PYTHONPATH=<chimera repo> python -m analysis.tasks.H1_harmonise.figures [--reference]     # every panel, from the cached tables (draw_panels)
    CHIMERA_FIGURES_OUT=<folder> ... the same command                                                # dry run: panels under <folder>/panels
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import transforms as mtransforms  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

from analysis.tasks.H1_harmonise import rules as RU  # noqa: E402
from analysis.tasks.H1_harmonise.recurrence import bh  # noqa: E402
from analysis.tasks.T8_4_figures import common as cm  # noqa: E402
from figures import style  # noqa: E402

fd = cm.fd
OUT = RU.OUT

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this task
INK, MUTED, RULE = style.INK, style.MUTED, style.RULE
S_TITLE, S_TICK, S_TEXT = style.SIZE_TITLE, style.SIZE_TICK, style.SIZE_TEXT  # axis labels take SIZE_LABEL from style.apply()
S_MARK = 5.2  # counts beside the marks of cut values
UP, DOWN = style.ABOVE, style.BELOW  # above and below the protein prior; positive and negative enrichment
BLUE, GREY, TREND = style.COUNT_EVIDENCE, style.IMPUTED, style.TREND  # genes with count evidence, without it, median trend
THEME_COL = list(style.GENE_THEME.values())  # lineage groups of the landscape, the colors of the liver landscape
CT_COL = list(style.LIVER_CELL_TYPE.values())  # cell types of the driver bars, the colors of the liver driver bars; the last is "other"
NA_CELL = "#F2F2F2"  # heat-map cell without an estimate
BOX_FACE, BAR_GREY, REF_GREY = "#dddddd", "#bdbdbd", "#888888"  # boxes, histogram bars, reference lines
MODEL_GREY = "#C4C4C4"  # the other models of a tissue beside the model of the figure (ink)
PRIOR = "protein prior π"
D_LABEL = "Δα (centered)"
YLIM_D = (-0.32, 0.32)
W_FULL, GAP = style.TEXTWIDTH, style.GAP
W3 = (W_FULL - 2 * GAP) / 3  # one of three panels in a row
# panel sizes in bp at print size: rows a b c / d e / f g h / i j k of a tissue figure, each row 486 bp with its gaps
SIZE = {"abc": (W3, 84), "d": (292, 154), "e": (W_FULL - GAP - 292, 154), "f": (160, 154), "g": (152, 154), "h": (W_FULL - 2 * GAP - 312, 154), "ijk": (W3, 90),
        "k_models": (W3, 110)}
BASE = {"abc": 25, "de": 25, "fgh": 57, "ijk": 33}  # points from the bottom of a panel to its x-axis, one value per row
_TISSUE_KEY = {"skeletal muscle": "muscle", "tumour myeloid": "tumor"}  # tissue names of rules.py that differ from the keys of style.TISSUE
TISSUE_LABEL = {t: style.TISSUE[_TISSUE_KEY.get(t, t)].name for t in ("liver", "testis", "bone marrow", "kidney", "islet", "eye", "cortex", "lung", "skeletal muscle", "tumour myeloid")}
FAMILY = {"bone marrow": "supp_tissue_bone_marrow", "kidney": "supp_tissue_kidney", "islet": "supp_tissue_islet", "eye": "supp_tissue_eye", "cortex": "supp_tissue_cortex",
          "lung": "supp_tissue_lung", "skeletal muscle": "supp_tissue_muscle", "tumour myeloid": "supp_tissue_tumor_myeloid"}
SUPP_PRIMARY = {**RU.PRIMARY, **RU.SENS_EXTRA, "tumour myeloid": "tumour_zilionis"}
SET_LABEL = {"heme Metabolism": "Heme Metabolism", "Pperoxisome": "Peroxisome", "TF": "Transcription factors", "TF_homeodomain": "Homeodomain TFs", "TF_C2H2_ZF": "C2H2 zinc-finger TFs", "TF_nuclear_rec": "Nuclear receptors", "protein_kinase": "Protein kinases",
             "receptor": "Receptors", "GPCR": "GPCRs", "ion_channel": "Ion channels", "SLC_transporter": "SLC transporters", "chromatin_reg": "Chromatin regulators", "enzyme": "Enzymes",
             "mitochondrial": "Mitochondrial proteins"}
# lineage groups: the keys of rules.LINEAGE stay as they are (they name rows of tables); these are the names drawn
THEME_LABEL = {"myofibre": "myofiber", "hepatocyte nutrient programme": "hepatocyte nutrient program"}
# marks on gene names (gene_label) and their key
MARK_KEY = {"^": "^ α > 0.95 in a seed (rank capped)", "§": "§ pair flagged by the ortholog audit, kept"}
# cell types: the labels of the data sets -> names for a reader
TYPE_LABEL = {
    # bone marrow, lung, lung-tumor myeloid
    "B": "B cell", "T": "T cell", "NK": "NK cell", "HSPC": "progenitor (HSPC)", "plasma": "plasma cell", "Macrophage": "macrophage", "Monocyte": "monocyte", "Neutrophil": "neutrophil",
    "DC": "dendritic cell", "cDC": "dendritic cell (cDC)", "pDC": "dendritic cell (pDC)",
    # kidney
    "pt": "proximal tubule", "tal": "thick ascending limb", "dct": "distal convoluted tubule", "cnt": "connecting tubule", "pc": "principal cell", "ic": "intercalated cell",
    "atl": "thin ascending limb", "dtl": "thin descending limb", "pec": "parietal epithelium", "pod": "podocyte", "ec": "endothelium", "fib": "fibroblast", "vsm_p": "smooth muscle",
    "pape": "papillary epithelium", "lymphoid": "lymphoid cell", "myeloid": "myeloid cell",
    # islet
    "alpha": "alpha cell", "beta": "beta cell", "delta": "delta cell", "pp": "PP cell", "ductal": "duct cell", "endothelial": "endothelium", "immune": "immune cell", "stellate": "stellate cell",
    # eye
    "trabecular meshwork cell": "trabecular meshwork", "ciliary muscle cell": "ciliary muscle", "vascular endothelial cell": "vascular endothelium", "canal endothelial cell": "Schlemm's canal",
    "ciliary body epithelial cell": "ciliary body epithelium", "corneal endothelial cell": "corneal endothelium", "corneal epithelial cell": "corneal epithelium",
    # cortex
    "Oligo": "oligodendrocyte", "Astro": "astrocyte", "Endo": "endothelium", "Micro": "microglia", "OPC": "oligodendrocyte precursor", "L2_3_IT": "L2/3 IT neuron", "L5_IT": "L5 IT neuron",
    "L6_IT": "L6 IT neuron", "L6_CT": "L6 CT neuron", "L5_ET": "L5 ET neuron", "L5_6_NP": "L5/6 NP neuron", "L6B": "L6b neuron", "PVALB": "PVALB interneuron", "SST": "SST interneuron",
    "SST_CHODL": "SST CHODL interneuron", "VIP": "VIP interneuron", "LAMP5": "LAMP5 interneuron", "SNCG": "SNCG interneuron",
    # lung
    "Macrophage_alveolar": "alveolar macrophage", "Macrophage_interstitial": "interstitial macrophage", "Fibroblast_alveolar": "alveolar fibroblast",
    "Fibroblast_adventitial": "adventitial fibroblast", "AT1": "alveolar type 1 cell", "AT2": "alveolar type 2 cell", "Aerocyte": "aerocyte", "Capillary": "capillary endothelium",
    "Artery": "arterial endothelium", "Vein": "venous endothelium", "Myofibroblast": "myofibroblast", "Pericyte": "pericyte",
    # skeletal muscle
    "FB": "fibroblast", "MuSC": "muscle stem cell", "SMC": "smooth muscle", "CapEC": "capillary endothelium", "ArtEC": "arterial endothelium", "VenEC": "venous endothelium", "Tenocyte": "tenocyte",
}


def type_label(t: str) -> str:
    return TYPE_LABEL.get(str(t), str(t))


def model_label(c: str) -> str:
    """A model of a tissue as it is named inside the figure of that tissue: the cohort name without the tissue."""
    lab = style.cohort_name(RU.cohorts()[c]["label"]).replace(", own table", ", own ortholog table")
    return re.sub(r"^(Eye|Cortex) ", "", lab)


def genes(c: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / "genes" / f"{c}.parquet")


def named(g: pd.DataFrame) -> pd.DataFrame:
    return g[g.nameable & g.symbol.notna()]


def gene_label(r) -> str:
    """^ alpha above 0.95 in a seed; section sign, pair flagged by the ortholog audit (kept, read with care)."""
    return f"{r.symbol}{'^' if r.saturated_flag else ''}{'§' if bool(r.o1_T2) else ''}"


def psave(fig: plt.Figure, name: str, exact: bool = False) -> None:
    """The one place a panel is saved: figures/panels/<family>/<name>.pdf, cropped to its content, or (``exact``) at the
    size of the figure, for panels whose axes share a baseline in their row (placed with ``X`` in the figure spec)."""
    style.save_panel(fig, name, exact=exact)


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def new_figure(size, base=None, **kw) -> plt.Figure:
    """A figure of ``size`` (width, height in bp at print size) whose content fills it, so that the saved panel has that
    size. ``base``: points from the bottom of the figure to the x-axis (see ``settle``)."""
    fig = plt.figure(figsize=(style.inch(size[0]), style.inch(size[1])), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, **kw)
    fig.baseline = base
    return fig


def settle(fig: plt.Figure) -> None:
    """Solve the layout once and keep it. Where the figure has a baseline (``fig.baseline``: points from the bottom of the
    figure to the x-axis), the axes that reach lowest are set on it, so that the panels of a row share it; a panel that
    needs more room under its axis than the baseline gives keeps the room it needs."""
    if fig.get_layout_engine() is None:
        return
    fig.draw_without_rendering()
    fig.set_layout_engine("none")
    base = getattr(fig, "baseline", None)
    if base is not None:
        low = min(a.get_position().y0 for a in fig.axes)
        b = max(base / (fig.get_figheight() * 72), low)
        for a in fig.axes:
            q = a.get_position()
            if q.y0 < low + 0.02:
                a.set_position([q.x0, b, q.width, q.y1 - b])
    # everything an axes carries (names beside it, keys under it) stays on the page, 2.5 points inside its edge
    m, (fw, fh) = 2.5 * fig.dpi / 72, (fig.bbox.width, fig.bbox.height)
    for _ in range(4):
        fig.draw_without_rendering()
        moved = False
        for a in fig.axes:
            bb, q = a.get_tightbbox(fig.canvas.get_renderer()), a.get_position()
            if bb is None:
                continue
            left, down, right, up = [max(0.0, v) for v in ((m - bb.x0) / fw, (m - bb.y0) / fh, (bb.x1 - fw + m) / fw, (bb.y1 - fh + m) / fh)]
            if max(left, down, right, up) > 1e-4:
                a.set_position([q.x0 + left, q.y0 + down, q.width - left - right, q.height - down - up])
                moved = True
        if not moved:
            break


def pt_per_unit(ax) -> tuple:
    """Points per data unit along x and y, after the layout is settled."""
    fig = ax.figure
    settle(fig)
    fig.draw_without_rendering()
    box = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return box.width * 72 / fig.dpi / abs(x1 - x0), box.height * 72 / fig.dpi / abs(y1 - y0)


def signed(v: float, nd: int = 2) -> str:
    """A signed number as printed on the panels: plus sign, true minus sign, and no sign on a value that rounds to zero."""
    s = f"{v:+.{nd}f}"
    return s[1:] if float(s) == 0 else s.replace("-", "−")


def note(ax, text: str, loc: str = "upper left", size: float = S_TEXT, patch: bool = False):
    """A statistic in a corner of the axes; on a white patch where it lies on a cloud of points."""
    x, y, ha, va = {"upper left": (0.03, 0.97, "left", "top"), "upper right": (0.97, 0.97, "right", "top"),
                    "lower left": (0.03, 0.04, "left", "bottom"), "lower right": (0.97, 0.04, "right", "bottom")}[loc]
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=INK, linespacing=1.15,
                   bbox=dict(fc="white", ec="none", alpha=0.8, pad=1) if patch else None)


def beyond(ax, x, y, color=MUTED, s: float = 7) -> int:
    """Points outside the axes are drawn on the frame as triangles that point outward, so no point is cut without a mark.
    Call after the limits are set. Returns the number of such points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    inside_y = (y >= y0) & (y <= y1)
    n = 0
    for sel, mk in ((y > y1, "^"), (y < y0, "v"), ((x < x0) & inside_y, "<"), ((x > x1) & inside_y, ">")):
        if sel.any():
            ax.scatter(np.clip(x[sel], x0, x1), np.clip(y[sel], y0, y1), marker=mk, s=s, color=color, linewidths=0, clip_on=False, zorder=6)
            n += int(sel.sum())
    return n


def beyond_count(ax, values, at: float, axis: str = "x") -> tuple:
    """The number of values beyond each end of an axis, printed at that end beside a triangle that points outward
    (box plots and histograms, where single points cannot be marked). ``at``: the position on the other axis."""
    v = np.asarray(values, float)
    lo, hi = sorted(ax.get_xlim() if axis == "x" else ax.get_ylim())
    n_lo, n_hi = int((v < lo).sum()), int((v > hi).sum())
    kw = dict(textcoords="offset points", fontsize=S_MARK, color=MUTED, annotation_clip=False, zorder=7)
    if axis == "x":
        if n_lo:
            ax.annotate(f"◂{n_lo}", (lo, at), xytext=(0.5, 1.2), ha="left", va="bottom", **kw)
        if n_hi:
            ax.annotate(f"{n_hi}▸", (hi, at), xytext=(-0.5, 1.2), ha="right", va="bottom", **kw)
    else:
        if n_lo:
            ax.annotate(f"▾{n_lo}", (at, lo), xytext=(2.5, 0.5), ha="left", va="bottom", **kw)
        if n_hi:
            ax.annotate(f"▴{n_hi}", (at, hi), xytext=(2.5, -0.5), ha="left", va="top", **kw)
    return n_lo, n_hi


# ---------------------------------------------------------------- standard panels
def p_scatter(ax, g):
    fd.density_scatter(ax, g.pi_raw, g.alpha)
    fd.style_alpha_axis(ax, "x")
    ax.set_xlabel(PRIOR)
    fd.style_alpha_axis(ax, "y")
    fd.add_identity_line(ax, lim=(0, 1))
    note(ax, f"Spearman $\\rho$ = {spearmanr(g.pi_raw, g.alpha)[0]:.3f}", "lower right", size=S_TICK)


def p_hist(ax, g):
    ax.hist(g.D, bins=fd.DELTA_ALPHA_BINS, color=BAR_GREY, edgecolor=REF_GREY, linewidth=0.4)
    ax.axvline(0, color=REF_GREY, linewidth=0.6, linestyle="--")
    fd.style_delta_alpha_axis(ax, "x")
    ax.set_xlabel(D_LABEL)
    ax.set_ylabel(fd.LABEL["n_genes"])
    note(ax, f"above +0.05: {(g.D > 0.05).mean():.0%}\nbelow −0.05: {(g.D < -0.05).mean():.0%}\nSD {g.D.std():.3f}", "upper left")
    beyond_count(ax, g.D, at=0, axis="x")


def p_overlay(ax, g):
    ax.hist(g.pi_raw, bins=fd.ALPHA_BINS, color="#bbbbbb", edgecolor="none", label=PRIOR, density=True)
    ax.hist(g.alpha, bins=fd.ALPHA_BINS, color=INK, histtype="step", linewidth=1.0, label=fd.LABEL["alpha"], density=True)
    fd.style_alpha_axis(ax, "x")
    ax.set_ylabel(fd.LABEL["density"])
    ax.legend(loc="upper left", fontsize=S_TICK, handlelength=1.2, handletextpad=0.4, borderpad=0.2, borderaxespad=0.2, labelspacing=0.25)
    ax.text(0.03, 0.66, f"$n$ = {len(g):,} genes\nmean $\\alpha$ = {g.alpha.mean():.3f}\nSD = {g.alpha.std():.3f}", transform=ax.transAxes, fontsize=S_TEXT, color=INK, va="top", linespacing=1.15)


def lineage_named(tissue, g):
    n = named(g).set_index("symbol")
    rows = []
    for (theme, syms), col in zip(RU.LINEAGE[tissue].items(), THEME_COL):
        for s in syms:
            if s in n.index:
                rows.append(dict(symbol=s, theme=theme, color=col, D=n.loc[s, "D"], pi=n.loc[s, "pi_raw"], var_name=n.loc[s, "var_name"] if "var_name" in n else None))
    return pd.DataFrame(rows, columns=["symbol", "theme", "color", "D", "pi", "var_name"]).astype({"D": float, "pi": float})


def top_named(g, k=10):
    n = named(g)
    return pd.concat([n.nlargest(k, "D"), n.nsmallest(k, "D")])


def side_by_side(xy: np.ndarray, d: float) -> np.ndarray:
    """Marker centers (points) moved apart along x where markers of diameter ``d`` would hide each other: centers closer
    than 0.6 d form a group, and a group is set side by side around its mean x, 0.7 d apart, in the order of x and y."""
    xy = np.array(xy, float)
    n = len(xy)
    group = list(range(n))
    for i in range(n):
        for j in range(i):
            if np.hypot(*(xy[i] - xy[j])) < 0.6 * d:
                gi, gj = group[i], group[j]
                group = [gj if k == gi else k for k in group]
    for k in set(group):
        idx = [i for i in range(n) if group[i] == k]
        if len(idx) > 1:
            idx = sorted(idx, key=lambda i: (xy[i, 0], xy[i, 1]))
            mid = np.mean([xy[i, 0] for i in idx])
            for r, i in enumerate(idx):
                xy[i, 0] = mid + (r - (len(idx) - 1) / 2) * 0.7 * d
    return xy


def _hits(seg, box, pad=0.0) -> bool:
    """Does the segment ((x0, y0), (x1, y1)) pass through the box (x0, y0, x1, y1)? Sampled along the segment."""
    (ax_, ay_), (bx_, by_) = seg
    for t in np.linspace(0, 1, 25):
        x, y = ax_ + t * (bx_ - ax_), ay_ + t * (by_ - ay_)
        if box[0] - pad <= x <= box[2] + pad and box[1] - pad <= y <= box[3] + pad:
            return True
    return False


def _cross(s1, s2) -> bool:
    """Do two segments cross?"""
    (a, b), (c, d) = np.asarray(s1, float), np.asarray(s2, float)
    o = lambda p, q, r: np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))  # noqa: E731
    return o(a, b, c) != o(a, b, d) and o(c, d, a) != o(c, d, b)


def _points(ax) -> tuple:
    """Width and height of the axes in points, and the transform from points (from the lower left of the axes) to display."""
    fig = ax.figure
    W, H = [v * 72 / fig.dpi for v in (ax.get_window_extent().width, ax.get_window_extent().height)]
    return W, H, mtransforms.Affine2D().scale(1 / W, 1 / H) + ax.transAxes


def _leader(ax, to_ax, p, q, r_mark) -> None:
    """A straight leader from the rim of the marker at ``p`` to ``q`` (points)."""
    ux, uy = q[0] - p[0], q[1] - p[1]
    n = np.hypot(ux, uy)
    if n > r_mark + 2:
        ax.plot([p[0] + ux * r_mark / n, q[0]], [p[1] + uy * r_mark / n, q[1]], transform=to_ax, color="#999999", lw=0.4, zorder=3, solid_capstyle="butt")


def near_labels(ax, idx, pts, names, sizes, markers, r_mark, keep_out=(), size=S_TEXT, shift=None, draw=True, cloud=None) -> tuple:
    """Names set directly beside their points, at fixed places that are the same on every run. Lengths in points from the
    lower left of the axes. The points ``idx`` are taken from the outside of the group inward; for each, places on three
    rings around it are tried, nearest first and the side away from the middle of the group first, and the first free
    one is taken: inside the axes, clear of every marker, of the names set before and of the boxes in ``keep_out``, and
    with at most two points of ``cloud`` (all points of the panel, in points) under the name.
    ``shift``: {name: (dx, dy)}, from the point to the center of the name, sets a place by hand.
    ``draw=False`` only finds the places. Returns (the points that found no place, {point: box of its name})."""
    W, H, to_ax = _points(ax)
    mk = np.asarray(markers, float)
    mid = mk.mean(axis=0)
    boxes, lost, placed = [], [], {}
    for i in sorted(idx, key=lambda i: -np.hypot(*(pts[i] - mid))):
        (px, py), (w, h) = pts[i], sizes[i]
        out = np.degrees(np.arctan2(py - mid[1], px - mid[0]))
        found = None
        if shift and shift.get(names[i]) is not None:
            dx, dy = shift[names[i]]
            found = (px + dx - w / 2, py + dy - h / 2, px + dx + w / 2, py + dy + h / 2)
        for r in () if found else r_mark + np.array([1.5, 4.0, 7.0]):
            for dev in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
                c, s = np.cos(np.radians(out + dev)), np.sin(np.radians(out + dev))
                cx, cy = px + r * c + c * w / 2, py + r * s + s * h / 2
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if box[0] < 1 or box[1] < 1 or box[2] > W - 1 or box[3] > H - 1:
                    continue
                if any(box[0] - 1 < b[2] and b[0] < box[2] + 1 and box[1] - 0.5 < b[3] and b[1] < box[3] + 0.5 for b in boxes + list(keep_out)):
                    continue
                if any(box[0] - r_mark - 0.8 < mx < box[2] + r_mark + 0.8 and box[1] - r_mark - 0.8 < my < box[3] + r_mark + 0.8 for mx, my in mk):
                    continue
                if cloud is not None and ((cloud[:, 0] > box[0]) & (cloud[:, 0] < box[2]) & (cloud[:, 1] > box[1]) & (cloud[:, 1] < box[3])).sum() > 2:
                    continue
                found = box
                break
            if found:
                break
        if found is None:
            lost.append(i)
            continue
        boxes.append(found)
        placed[i] = found
        if draw:
            ax.text((found[0] + found[2]) / 2, (found[1] + found[3]) / 2, names[i], transform=to_ax, fontsize=size, ha="center", va="center", color=INK, zorder=5)
            _leader(ax, to_ax, (px, py), (min(max(px, found[0]), found[2]), min(max(py, found[1]), found[3])), r_mark + 1.5)
    return lost, placed


def comb_labels(ax, idx, pts, names, markers, r_mark, blocked=(), size=S_TEXT, below=False, draw=True) -> list:
    """Names of crowded points, set upright in a row along the top of the axes (``below``: along the bottom) in the order
    of the points along x, each joined to its point by a straight leader. A slot that would stand on a marker or on a
    box of ``blocked`` is passed over; where two leaders would cross, the two names change places. ``draw=False`` only
    finds the places. Returns the leaders as segments."""
    if not len(idx):
        return []
    fig = ax.figure
    W, H, to_ax = _points(ax)
    r = fig.canvas.get_renderer()
    pitch = 1.2 * size
    lens = []
    for i in idx:
        t = ax.text(0, 0, names[i], fontsize=size)
        lens.append(t.get_window_extent(r).width * 72 / fig.dpi)
        t.remove()
    y_b = max(lens) + 2.5 if below else H - max(lens) - 2.5  # the end of the names that faces the points
    lo, hi = (0, y_b + 1) if below else (y_b - 1, H)
    taken = [(mx - r_mark - 1, mx + r_mark + 1) for mx, my in markers if lo - r_mark <= my <= hi + r_mark]
    taken += [(b[0] - 1, b[2] + 1) for b in blocked if b[1] < hi and lo < b[3]]
    order = sorted(idx, key=lambda i: (pts[i][0], -pts[i][1]))
    start = max(pitch / 2 + 1, np.mean([pts[i][0] for i in order]) - (len(order) - 1) / 2 * pitch)
    for _ in range(6):
        slots, s = [], start
        for _i in order:
            hit = [b for a, b in taken if a < s + pitch / 2 and s - pitch / 2 < b]
            while hit:
                s = max(hit) + pitch / 2 + 0.01
                hit = [b for a, b in taken if a < s + pitch / 2 and s - pitch / 2 < b]
            slots.append(s)
            s += pitch
        over = slots[-1] + pitch / 2 + 1 - W
        if over <= 0 or start <= pitch / 2 + 1:
            break
        start = max(pitch / 2 + 1, start - over)
    slot = dict(zip(order, slots))
    seg = lambda i: ((pts[i][0], pts[i][1]), (slot[i], y_b))  # noqa: E731
    for _ in range(200):
        pairs = [(i, j) for k, i in enumerate(order) for j in order[:k] if _cross(seg(i), seg(j))]
        if not pairs:
            break
        i, j = pairs[0]
        slot[i], slot[j] = slot[j], slot[i]
    for i in order if draw else ():
        ax.text(slot[i], y_b, names[i], transform=to_ax, rotation=90, ha="center", va="top" if below else "bottom", fontsize=size, color=INK, zorder=5)
        _leader(ax, to_ax, pts[i], (slot[i], y_b + (1 if below else -1)), r_mark)
    return [seg(i) for i in order]



# names of the landscape set by hand where the fixed rule leaves a collision: {tissue: {symbol: (dx, dy) in points from the point to the center of the
# name, or None for a place in the row along the top}}
LANDSCAPE_SHIFT: dict = {}


def p_landscape(g, tissue, size=SIZE["d"], base=None) -> plt.Figure:
    """All genes by protein prior and centered Δα, the lineage genes of the tissue in color, the 18 farthest from zero named.
    Lineage markers that would hide each other are set side by side along the prior; genes outside the axes are triangles on the frame."""
    fig = new_figure(size, base=base, hspace=0, wspace=0)
    gs = fig.add_gridspec(2, 2, width_ratios=[10, 1], height_ratios=[1, 8])
    ax = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=ax)
    right = fig.add_subplot(gs[1, 1], sharey=ax)
    fd.density_scatter(ax, g.pi_raw, g.D, s=1.6, alpha=0.25, color="#BBBBBB")
    ax.axhline(0, color=REF_GREY, lw=0.5, ls="--", zorder=2)
    ax.set_xlabel(PRIOR)
    ax.set_ylabel(D_LABEL)
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(-0.6, 0.6)
    beyond(ax, g.pi_raw, g.D, color="#999999")
    top.hist(g.pi_raw, bins=60, color="#999999", edgecolor="none")
    right.hist(g.D.clip(-0.6, 0.6), bins=60, color="#999999", edgecolor="none", orientation="horizontal")
    for a, hide in ((top, ("top", "right", "left")), (right, ("top", "right", "bottom"))):
        for sp in hide:
            a.spines[sp].set_visible(False)
    top.set_yticks([])
    top.tick_params(axis="x", bottom=False, labelbottom=False)
    right.set_xticks([])
    right.tick_params(axis="y", left=False, labelleft=False)
    nm = lineage_named(tissue, g)
    if not len(nm):
        return fig
    s_mark = 11
    d_mark = np.sqrt(s_mark)
    hs = [Line2D([], [], ls="none", marker="o", ms=d_mark, mfc=col, mec="white", mew=0.3, label=THEME_LABEL.get(theme, theme))
          for theme, col in nm.drop_duplicates("theme")[["theme", "color"]].itertuples(index=False)]
    leg = ax.legend(handles=hs, loc="lower left", fontsize=S_TEXT, handletextpad=0.1, borderaxespad=0.2, labelspacing=0.3, frameon=False)
    ux, uy = pt_per_unit(ax)
    (x0, _), (y0, _) = ax.get_xlim(), ax.get_ylim()
    xy = side_by_side(np.c_[(nm.pi - x0) * ux, (nm.D - y0) * uy], d_mark)
    nm = nm.assign(px=xy[:, 0], py=xy[:, 1], x_drawn=xy[:, 0] / ux + x0)
    for theme, s in nm.groupby("theme", sort=False):
        ax.scatter(s.x_drawn, s.D, s=s_mark, color=s.color.iloc[0], edgecolor="white", linewidth=0.3, zorder=4)
        beyond(ax, s.x_drawn, s.D, color=s.color.iloc[0], s=s_mark)
    sel = nm.reindex(nm.D.abs().sort_values(ascending=False).index).head(18)
    r = fig.canvas.get_renderer()
    sizes = []
    for sym in sel.symbol:
        t = ax.text(0, 0, sym, fontsize=S_TEXT)
        e = t.get_window_extent(r)
        sizes.append((e.width * 72 / fig.dpi, e.height * 72 / fig.dpi))
        t.remove()
    ab, lb = ax.get_window_extent(), leg.get_window_extent(r)
    k = 72 / fig.dpi
    keep_out = [((lb.x0 - ab.x0) * k, (lb.y0 - ab.y0) * k, (lb.x1 - ab.x0) * k, (lb.y1 - ab.y0) * k)]
    cloud = [((0.55 - x0) * ux, (-0.07 - y0) * uy, (0.96 - x0) * ux, (0.07 - y0) * uy)]  # where most genes lie
    # a name stands beside its point where the point has at most one neighbor and the place is free and off the cloud;
    # the names of the other points stand in a row along the top (genes below the prior: along the bottom)
    pts, mk, names = sel[["px", "py"]].to_numpy(), nm[["px", "py"]].to_numpy(), list(sel.symbol)
    shift = LANDSCAPE_SHIFT.get(tissue, {})
    alone = [i for i in range(len(pts)) if (np.hypot(*(mk - pts[i]).T) < 14).sum() <= 2 and not (names[i] in shift and shift[names[i]] is None)]
    zero = (0 - y0) * uy
    every = np.c_[(g.pi_raw.to_numpy() - x0) * ux, (g.D.to_numpy() - y0) * uy]
    def place(draw: bool) -> list:
        lost, placed = near_labels(ax, alone, pts, names, sizes, mk, d_mark / 2, keep_out=keep_out + cloud, shift=shift, draw=draw, cloud=every)
        rest = sorted(set(range(len(pts))) - set(alone) | set(lost))
        boxes = list(placed.values())
        segs = comb_labels(ax, [i for i in rest if pts[i][1] >= zero], pts, names, mk, d_mark / 2, blocked=boxes + keep_out, draw=draw)
        segs += comb_labels(ax, [i for i in rest if pts[i][1] < zero], pts, names, mk, d_mark / 2, blocked=boxes + keep_out, below=True, draw=draw)
        return [i for i, b in placed.items() if any(_hits(sg, b, 1.0) for sg in segs) and names[i] not in shift]

    crossed = place(False)
    while crossed:  # a name beside its point that a leader of the row would cross joins the row
        alone = [i for i in alone if i not in crossed]
        crossed = place(False)
    place(True)
    return fig


def p_categories(ax, g, tissue, members):
    cats = members[members.family == "broad"][["set", "hsapiens_gene_id"]]
    d = g.merge(cats, on="hsapiens_gene_id")
    order = d.groupby("set").D.median().sort_values().index.tolist()
    data = [d.loc[d.set == c, "D"].values for c in order]
    ax.boxplot(data, orientation="horizontal", positions=range(len(order)), widths=0.65, patch_artist=True, medianprops=dict(color=INK, linewidth=1.2),
               boxprops=dict(facecolor=BOX_FACE, edgecolor=INK, linewidth=0.5), whiskerprops=dict(color=INK, linewidth=0.5), capprops=dict(color=INK, linewidth=0.5),
               flierprops=dict(marker="o", markersize=1.0, markerfacecolor=INK, markeredgecolor="none", alpha=0.4))
    ax.axvline(0, color=REF_GREY, linewidth=0.6, linestyle="--", zorder=0)
    ax.set_xlim(-0.20, 0.20)
    ax.set_xticks([-0.2, 0.0, 0.2])
    ax.set_xlabel(D_LABEL)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=S_TEXT)
    ax.tick_params(axis="y", pad=1.5)
    sd = g.D.std()
    ps = [wilcoxon(x, alternative="two-sided", zero_method="wilcox")[1] if len(x) > 5 else 1.0 for x in data]
    q = bh(ps)
    # two columns right of the boxes: number of genes, and effect size d with the mark of significance
    kw = dict(va="center", fontsize=S_TEXT, color=INK, clip_on=False)
    col_n, col_d = [mtransforms.offset_copy(ax.get_yaxis_transform(), fig=ax.figure, x=dx, units="points") for dx in (21, 26)]  # from the right end of the axis
    ax.text(1, len(order) - 0.42, "$n$", ha="right", transform=col_n, **{**kw, "va": "bottom"})
    ax.text(1, len(order) - 0.42, "$d$", ha="left", transform=col_d, **{**kw, "va": "bottom"})
    for i, x in enumerate(data):
        ax.text(1, i, f"{len(x):,}", ha="right", transform=col_n, **kw)
        ax.text(1, i, signed(np.mean(x) / sd) + (style.SIGNIFICANT if q[i] < 0.05 else ""), ha="left", transform=col_d, **kw)
        beyond_count(ax, x, at=i, axis="x")


def clean_term(t) -> str:
    """The name of a Hallmark set as drawn: in full, with the spelling of SET_LABEL."""
    s = re.sub(r"^HALLMARK_", "", str(t).split("__")[-1]).replace("_", " ").strip()
    return SET_LABEL.get(s, s)


def p_volcano(ax, df, nes_lim=2.8, nlf_max=4.5):
    """Hallmark sets by NES and FDR q. Every set is drawn: a set whose q lies below the plotted range (q = 0 in the
    permutation test included) is a triangle above the range, and such sets are stacked where they would hide each other.
    The three significant sets with the largest NES on each side are named above the points, at fixed places."""
    fig = ax.figure
    d = df.copy()
    nlf = -np.log10(d["FDR q-val"].clip(1e-300, 1.0))
    d["capped"] = nlf > nlf_max
    d["nlf"] = nlf.clip(upper=nlf_max)
    d["sig"] = d["FDR q-val"] < 0.05
    col = np.where(d.NES > 0, UP, DOWN)
    y_lo, s_pt, s_ns = -0.15, 13, 9
    ax.axhline(-np.log10(0.05), color=REF_GREY, linewidth=0.5, linestyle="--", zorder=0)
    ax.set_xlim(-nes_lim, nes_lim)
    ax.set_ylim(y_lo, nlf_max)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["1", "0.1", "0.01", "0.001", "0.0001"])  # the axis is −log10 q, labeled with q
    ax.set_xlabel("NES (Δα-ranked GSEA)")
    ax.set_ylabel("FDR $q$")
    lab = pd.concat([sub.assign(a=sub.NES.abs()).nlargest(3, "a") for sub in (d[d.sig & (d.NES > 0)], d[d.sig & (d.NES < 0)])])
    mk = dict(ls="none", mew=0.3, mec=INK)
    hs = [Line2D([], [], marker="o", ms=np.sqrt(s_pt), mfc=UP, label="NES > 0", **mk)]
    if (d.sig & (d.NES < 0)).any():
        hs.append(Line2D([], [], marker="o", ms=np.sqrt(s_pt), mfc=DOWN, label="NES < 0", **mk))
    hs.append(Line2D([], [], marker="o", ms=np.sqrt(s_ns), mfc="#cccccc", mec="none", ls="none", label="not significant"))
    if d.capped.any():
        hs.append(Line2D([], [], marker="^", ms=np.sqrt(s_pt), mfc="#ffffff", label="$q$ below the plotted range", **mk))
    hs.append(Line2D([], [], color=REF_GREY, lw=0.5, ls="--", label="$q$ = 0.05"))
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 0), bbox_transform=mtransforms.blended_transform_factory(
        fig.transFigure, mtransforms.offset_copy(ax.transAxes, fig=fig, y=-25, units="points")), ncol=2, fontsize=S_TEXT, handlelength=1.3,
        handletextpad=0.4, columnspacing=1.0, labelspacing=0.25, borderpad=0, borderaxespad=0)
    # heights in points: the range of q, then the rows of sets below the range, then one line per name
    ux, _ = pt_per_unit(ax)
    h_pt = ax.get_window_extent().height * 72 / fig.dpi
    dm, lh = np.sqrt(s_pt), 1.22 * S_TEXT
    cap = d[d.capped].sort_values("NES")
    rows = []
    for x in cap.NES * ux:
        r = 0
        while any(rr == r and abs(x - xx) < 0.9 * dm for xx, rr in rows):
            r += 1
        rows.append((x, r))
    n_rows = max((r for _, r in rows), default=-1) + 1
    h_rows = n_rows * dm + (4 if n_rows else 0)
    h_range = h_pt - h_rows - len(lab) * lh - (3 if len(lab) else 0)
    uy = h_range / (nlf_max - y_lo)
    ax.set_ylim(y_lo, y_lo + h_pt / uy)
    y_of = lambda pt: y_lo + pt / uy  # noqa: E731
    d["y"] = d.nlf
    d.loc[cap.index, "y"] = [y_of(h_range + 3 + (r + 0.5) * dm) for _, r in rows]
    if n_rows:
        ax.axhline(y_of(h_range + 1.2), color=RULE, linewidth=0.5, linestyle=":", zorder=0)
    ax.plot([0, 0], [y_lo, y_of(h_range + h_rows)], color=REF_GREY, linewidth=0.5, zorder=0)  # the zero line ends under the names
    ns, sg = d[~d.sig], d[d.sig & ~d.capped]
    ax.scatter(ns.NES, ns.y, s=s_ns, c="#cccccc", alpha=0.8, linewidths=0, zorder=1)
    ax.scatter(sg.NES, sg.y, s=s_pt, c=col[d.index.get_indexer(sg.index)], linewidths=0.3, edgecolor=INK, zorder=2)
    ax.scatter(cap.NES, d.loc[cap.index, "y"], s=s_pt, marker="^", c=col[d.index.get_indexer(cap.index)], linewidths=0.3, edgecolor=INK, zorder=2)
    # names: one line each above the points; lower lines for smaller |NES|, the sets with NES < 0 below those with NES > 0.
    # A name ends (NES > 0) or starts (NES < 0) beside its point, each line a step further out, and a straight leader joins the two.
    w_pt = ax.get_window_extent().width * 72 / fig.dpi
    order = pd.concat([lab[lab.NES < 0].sort_values("a"), lab[lab.NES > 0].sort_values("a")])
    r = fig.canvas.get_renderer()
    side_rank = {1: 0, -1: 0}
    # the lines start above the rows of sets below the range, or, without such rows, just above the highest point
    first = h_range + h_rows + 3 if n_rows else min((d.y.max() - y_lo) * uy + dm / 2 + 4, h_pt - len(lab) * lh)
    for line, row in enumerate(order.itertuples()):
        sgn = 1 if row.NES > 0 else -1
        k = side_rank[sgn]
        side_rank[sgn] += 1
        name = clean_term(row.Term)
        yl = y_of(first + (line + 0.5) * lh)
        t = ax.text(0, yl, name, fontsize=S_TEXT, color=INK, va="center", ha="right" if sgn > 0 else "left", zorder=5)
        w = t.get_window_extent(r).width * 72 / fig.dpi
        x_pt = (row.NES + nes_lim) * ux + sgn * (4 * k - 2)  # the end of the name that faces its point
        x_pt = min(max(x_pt, w + 1), w_pt - 1) if sgn > 0 else max(min(x_pt, w_pt - w - 1), 1)
        t.set_x(x_pt / ux - nes_lim)
        ax.plot([row.NES, x_pt / ux - nes_lim], [d.y[row.Index] + 0.5 * dm / uy, yl - 0.42 * lh / uy], color="#999999", lw=0.4, zorder=3, solid_capstyle="butt")


def p_archetypes(ax, c):
    t = pd.read_csv(OUT / "archetypes" / f"{c}.csv")
    s = pd.read_csv(OUT / "archetypes" / "summary.csv")
    s = s[s.cohort == c]
    k = sorted(t.archetype.unique())
    ax.boxplot([t.D[t.archetype == a] for a in k], positions=k, widths=0.65, patch_artist=True, medianprops=dict(color=INK, linewidth=1.2), boxprops=dict(facecolor=BOX_FACE, edgecolor=INK, linewidth=0.5),
               whiskerprops=dict(color=INK, linewidth=0.5), capprops=dict(color=INK, linewidth=0.5), flierprops=dict(marker="o", markersize=0.8, markerfacecolor=INK, markeredgecolor="none", alpha=0.3))
    ax.axhline(0, color=REF_GREY, lw=0.6, ls="--", zorder=0)
    ax.set_ylim(*YLIM_D)
    ax.set_xticks(k)
    ax.set_xticklabels([f"$A_{a + 1}$" for a in k])
    ax.set_xlabel(f"GIN hidden-state archetype\n($k$ = {int(s.k.iloc[0])}; silhouette {s.silhouette.iloc[0]:.3f})")
    ax.set_ylabel(D_LABEL)
    for a in k:
        beyond_count(ax, t.D[t.archetype == a], at=a, axis="y")


def dominant_cells(c):
    f = OUT / "drivers" / f"{c}.parquet"
    if not f.exists():
        return None
    return pd.read_parquet(f)


def p_drivers(ax, c, g, tissue):
    """Fig 4d form: contribution of each cell type to S_g of the genes labelled in the landscape; top-weight types, the rest as other."""
    C = dominant_cells(c)
    nm = lineage_named(tissue, g)
    if C is None:
        ax.text(0.5, 0.5, "forward pass pending", transform=ax.transAxes, ha="center")
        return
    sel = pd.concat([nm.nlargest(7, "D"), nm.nsmallest(7, "D")]).drop_duplicates("symbol").sort_values("D")
    if len(sel) < 8:
        extra = top_named(g, 6)
        extra = extra[~extra.symbol.isin(sel.symbol)].assign(pi=lambda d: d.pi_raw, theme="", color="")[["symbol", "theme", "color", "D", "pi"]]
        sel = pd.concat([sel, extra]).sort_values("D")
    gs = g.set_index("symbol")
    shares = pd.read_csv(OUT / "drivers" / f"{c}__shares.csv").set_index("cell_type")
    tops = shares.sort_values("share_abs_S", ascending=False).index[:4].tolist()
    groups = [(t, [t], col) for t, col in zip(tops, CT_COL)] + [("other cell types", None, CT_COL[-1])]
    for y, r in enumerate(sel.itertuples()):
        vn = gs.index.get_loc(r.symbol)
        row = C.loc[g.index[g.symbol == r.symbol][0]]
        pos = neg = 0.0
        for name, ts, col in groups:
            v = row[[t for t in row.index if t not in tops]].sum() if ts is None else row.reindex(ts).sum()
            ax.barh(y, v, left=pos if v >= 0 else neg, color=col, height=0.72, lw=0)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        ax.plot([row.sum()], [y], marker="|", ms=5, color=INK, mew=0.9, zorder=4)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=0)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([f"{r.symbol} ({signed(r.D)})" for r in sel.itertuples()], fontsize=S_TEXT)
    ax.tick_params(axis="y", pad=1)
    ax.set_ylim(-0.6, len(sel) - 0.4)  # the largest Δα on top, as in the panel of the top genes
    ax.set_xlabel("contribution to $S_g$, by cell type\n(tick: total $S_g$; Δα in parentheses)", fontsize=S_TICK)
    w = shares.weight
    hs = [Patch(color=col, label=f"{type_label(name)} ({w.reindex(ts).sum():.2f})" if ts else name) for name, ts, col in groups]
    fig = ax.figure
    below = mtransforms.blended_transform_factory(fig.transFigure, mtransforms.offset_copy(ax.transAxes, fig=fig, y=-33, units="points"))
    for ncol in (2, 1):  # two columns where the names are short enough for the panel
        leg = ax.legend(handles=hs, title="cell type (share of cells)", title_fontsize=S_TEXT, loc="upper center", bbox_to_anchor=(0.5, 0), bbox_transform=below, ncol=ncol,
                        fontsize=S_TEXT, handlelength=0.9, handletextpad=0.4, columnspacing=0.8, labelspacing=0.25, borderpad=0, borderaxespad=0)
        fig.draw_without_rendering()
        if leg.get_window_extent().width * 72 / fig.dpi <= fig.get_figwidth() * 72 - 6:
            break


def p_top(ax, g, c, k=12):
    """Top count-supported genes by Delta-alpha, each direction, with the cell type carrying most of the margin."""
    n = named(g)
    C = dominant_cells(c)
    t = pd.concat([n.nlargest(k, "D"), n.nsmallest(k, "D").iloc[::-1]]).drop_duplicates("symbol")
    ys = np.arange(len(t))
    ax.barh(ys, t.D, color=[UP if v > 0 else DOWN for v in t.D], height=0.68, lw=0)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(ys)
    labels = [gene_label(r) for r in t.itertuples()]
    ax.set_yticklabels(labels, fontsize=S_TEXT)
    ax.tick_params(axis="y", pad=1.5)
    ax.set_ylim(len(t) - 0.4, -0.6)
    ax.set_xlim(-0.9, 0.9)  # holds every bar of the eight tissues (−0.85 to +0.60) and the cell-type names beside zero
    ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_xlabel(D_LABEL)
    key = [v for k_, v in MARK_KEY.items() if any(k_ in s for s in labels)]
    if key:
        ax.annotate("\n".join(key), (0.5, 0), xycoords=("figure fraction", "axes fraction"), xytext=(0, -26), textcoords="offset points", ha="center", va="top",
                    fontsize=S_TEXT, color=INK, linespacing=1.2, annotation_clip=False)
    if C is not None:
        names = [ax.text(-0.02 if d > 0 else 0.02, y, type_label(C.loc[vn].abs().idxmax()), fontsize=S_TEXT, color=MUTED, ha="right" if d > 0 else "left", va="center")
                 for y, vn, d in zip(ys, t.index, t.D)]
        # the axis grows to the left where a cell-type name left of zero would reach the gene names
        fig = ax.figure
        for _ in range(4):
            ux, _ = pt_per_unit(ax)
            r = fig.canvas.get_renderer()
            need = max([0.9] + [0.02 + (tx.get_window_extent(r).width * 72 / fig.dpi + 2.5) / ux for tx, d in zip(names, t.D) if d > 0])
            if need <= -ax.get_xlim()[0] + 1e-6:
                break
            ax.set_xlim(-need, 0.9)


def p_count(ax, g):
    """Delta-alpha against the count evidence on supported genes: the count dependence behind the reading."""
    u = g[g.count_support]
    x = pd.Series(u.S_ema).rank(pct=True).to_numpy()
    fd.density_scatter(ax, x, u.D, s=1.4, alpha=0.2, color=BLUE)
    ax.axhline(0, color=REF_GREY, lw=0.5, ls="--")
    ax.set_ylim(*YLIM_D)
    ax.set_xlim(0, 1)
    ax.set_xlabel("$S_g$ (rank, count-supported genes)")
    ax.set_ylabel(D_LABEL)
    beyond(ax, x, u.D, color=BLUE)
    note(ax, f"Spearman $\\rho$ = {signed(spearmanr(u.S_ema, u.D)[0])}", "lower right", size=S_TICK, patch=True)


def p_support(ax, g):
    """Fig 4e form: fraction of tail genes whose Delta-alpha is backed by count evidence of the same sign."""
    un = g.count_support.to_numpy()
    thr = np.median(g.S_ema.abs()[un])
    ok = g.count_support & (np.sign(g.S_ema) == np.sign(g.delta_alpha)) & (g.S_ema.abs() >= thr)
    tails = [(g.delta_alpha < -RU.TAIL, DOWN, "negative tail\nΔα < −0.05"), (g.delta_alpha > RU.TAIL, UP, "positive tail\nΔα > +0.05")]
    for k, (sub, col, lab) in enumerate(tails):
        f = ok[sub].mean() if sub.any() else 0.0
        ax.barh(k, f, color=col, height=0.6, lw=0)
        ax.text(f + 0.03, k, f"{f:.2f}  ($n$ = {int(sub.sum()):,})", va="center", fontsize=S_TEXT, color=INK)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([lab for _, _, lab in tails], fontsize=S_TICK, linespacing=1.1)
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(0, 1.75)  # room for the value beside a bar of 1.0; the axis is drawn from 0 to 1
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("fraction of tail genes backed by\ncount evidence of the same sign", fontsize=S_TICK)


def p_models(ax, tissue):
    """Centered Delta-alpha of every model of one tissue (eye, cortex), one row per model: the same histograms on one density
    scale, the model of the figure first and in ink, the others in gray, each named beside its row."""
    cs = [c for c, d in RU.cohorts().items() if d["tissue"] == tissue]
    cs = sorted(cs, key=lambda c: (c != RU.PRIMARY[tissue], c))
    bins = fd.DELTA_ALPHA_BINS
    hist = [np.histogram(genes(c).D, bins=bins, density=True)[0] for c in cs]
    top = max(h.max() for h in hist)
    for i, h in enumerate(hist):
        y0 = len(cs) - 1 - i
        y = y0 + np.r_[h, h[-1]] / top * 0.92
        ax.fill_between(bins, y0, y, step="post", color=INK if i == 0 else MODEL_GREY, lw=0)
        ax.plot(bins, y, drawstyle="steps-post", color=INK if i == 0 else MUTED, lw=0.5)
    fd.style_delta_alpha_axis(ax, "x")
    ax.set_xlabel(D_LABEL)
    ax.axvline(0, color=REF_GREY, lw=0.5, ls="--", zorder=0)
    ax.set_ylim(-0.05, len(cs))
    ax.set_yticks(np.arange(len(cs)) + 0.3)
    ax.set_yticklabels([model_label(c).replace(", own", ",\nown") for c in cs][::-1], fontsize=S_TEXT, linespacing=1.0)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)


def one(draw, size, *args, base=None, **kw) -> plt.Figure:
    """One axes on a figure of ``size`` (bp at print size), its x-axis ``base`` points above the bottom of the figure."""
    fig = new_figure(size, base=base)
    ax = fig.add_subplot()
    clean_axes(ax)
    draw(ax, *args, **kw)
    settle(fig)
    return fig


def supp_panels(tissue: str, c: str, family: str) -> None:
    """The eleven panels a to k of one tissue figure."""
    g = genes(c)
    g["var_name"] = g.index
    mem = pd.read_csv(OUT / "sets_membership.csv")
    gsea = pd.read_csv(OUT / "gsea" / f"{c}.csv")
    psave(one(p_scatter, SIZE["abc"], g, base=BASE["abc"]), f"{family}/a", exact=True)
    psave(one(p_hist, SIZE["abc"], g, base=BASE["abc"]), f"{family}/b", exact=True)
    psave(one(p_overlay, SIZE["abc"], g, base=BASE["abc"]), f"{family}/c", exact=True)
    psave(p_landscape(g, tissue, base=BASE["de"]), f"{family}/d", exact=True)
    psave(one(p_categories, SIZE["e"], g, tissue, mem, base=BASE["de"]), f"{family}/e", exact=True)
    psave(one(p_volcano, SIZE["f"], gsea, base=BASE["fgh"]), f"{family}/f", exact=True)
    psave(one(p_drivers, SIZE["g"], c, g, tissue, base=BASE["fgh"]), f"{family}/g", exact=True)
    psave(one(p_top, SIZE["h"], g, c, 8, base=BASE["fgh"]), f"{family}/h", exact=True)
    psave(one(p_archetypes, SIZE["ijk"], c, base=BASE["ijk"]), f"{family}/i", exact=True)
    psave(one(p_count, SIZE["ijk"], g, base=BASE["ijk"]), f"{family}/j", exact=True)
    if tissue in ("eye", "cortex"):
        psave(one(p_models, SIZE["k_models"], tissue, base=BASE["ijk"]), f"{family}/k", exact=True)
    else:
        psave(one(p_support, SIZE["ijk"], g, base=BASE["ijk"]), f"{family}/k", exact=True)


# ---------------------------------------------------------------- main candidate
def trend(ax, x, y, bins=14):
    q = pd.qcut(x, bins, duplicates="drop")
    m = pd.DataFrame({"x": x, "y": y}).groupby(q, observed=True).median()
    ax.plot(m.x, m.y, color=TREND, lw=1.1)


def cloud(ax, x, y, sup):
    ax.scatter(x[~sup], y[~sup], s=1.0, alpha=0.16, color=GREY, linewidths=0, rasterized=True)
    ax.scatter(x[sup], y[sup], s=1.0, alpha=0.2, color=BLUE, linewidths=0, rasterized=True)


def main_axis() -> plt.Figure:
    """Eight tissues in columns: the distribution of centered Δα, and Δα against protein identity and against dN/dS."""
    ax_t = pd.read_csv(OUT / "axis.csv")
    fig, axes = plt.subplots(3, len(RU.TISSUE_ORDER), figsize=(style.inch(W_FULL), style.inch(232)), sharey="row", layout="constrained", gridspec_kw=dict(height_ratios=[0.8, 1, 1]))
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    peak = 0.0
    ylim = (-0.56, YLIM_D[1])  # the range of the other panels, and room under the clouds for the correlations
    for j, t in enumerate(RU.TISSUE_ORDER):
        c = RU.PRIMARY[t]
        g = genes(c)
        r_all = ax_t[(ax_t.cohort == c) & (ax_t.universe == "all")].iloc[0]
        r_sup = ax_t[(ax_t.cohort == c) & (ax_t.universe == "count_support")].iloc[0]
        ax = axes[0, j]
        n, _, _ = ax.hist(g.D.clip(-0.4, 0.4), bins=fd.DELTA_ALPHA_BINS, color=BAR_GREY, edgecolor="none", density=True)
        peak = max(peak, n.max())
        ax.axvline(0, ymax=0.52, color=REF_GREY, lw=0.5, ls="--")
        ax.set_xlim(-0.3, 0.3)
        ax.set_xticks([-0.2, 0, 0.2])
        ax.set_xlabel(D_LABEL, fontsize=S_TICK)
        ax.set_title(TISSUE_LABEL[t], fontsize=S_TITLE, fontweight="bold", pad=3)
        beyond_count(ax, g.D, at=0, axis="x")
        # the note stands above the histogram: the density axis runs to twice the highest peak of the row
        ax.text(0.5, 0.97, f"SD {g.D.std():.3f}\n↑ {(g.D > 0.05).mean():.0%}  ↓ {(g.D < -0.05).mean():.0%}", transform=ax.transAxes, ha="center", va="top", fontsize=S_TEXT, color=INK, linespacing=1.15)
        ax = axes[1, j]
        d = g.dropna(subset=["dNdS"])
        d = d[d.dNdS > 0]
        sup = d.count_support.to_numpy()
        cloud(ax, d.pi_raw.to_numpy(), d.D.to_numpy(), sup)
        trend(ax, d.pi_raw.to_numpy(), d.D.to_numpy())
        ax.axhline(0, color="#333333", lw=0.5, ls=":")
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(*ylim)
        ax.set_yticks([-0.25, 0, 0.25])
        ax.set_xticks([0.4, 0.7, 1.0])
        ax.set_xlabel("protein identity π", fontsize=S_TICK)
        beyond(ax, d.pi_raw.to_numpy(), d.D.to_numpy(), s=5)
        note(ax, f"Δα ρ = {signed(r_all.rho_D_pi)}\n$S_g$ ρ = {signed(r_sup.rho_S_pi)}", "lower left")
        ax = axes[2, j]
        x = np.log10(d.dNdS.clip(lower=1e-3)).to_numpy()
        cloud(ax, x, d.D.to_numpy(), sup)
        trend(ax, x, d.D.to_numpy())
        ax.axhline(0, color="#333333", lw=0.5, ls=":")
        ax.set_xlim(-2.5, 0.3)
        ax.set_ylim(*ylim)
        ax.set_yticks([-0.25, 0, 0.25])
        ax.set_xticks([-2, -1, 0])
        ax.set_xlabel("dN/dS (log10)", fontsize=S_TICK)
        beyond(ax, x, d.D.to_numpy(), s=5)
        note(ax, f"Δα ρ = {signed(r_all.rho_D_dNdS)}\nπ ρ = {signed(r_all.rho_pi_dNdS)}", "lower left")
    axes[0, 0].set_ylim(0, 2.0 * peak)
    axes[0, 0].set_ylabel("density", fontsize=S_TICK)
    axes[1, 0].set_ylabel(D_LABEL, fontsize=S_TICK)
    axes[2, 0].set_ylabel(D_LABEL, fontsize=S_TICK)
    for ax in axes.ravel():
        ax.tick_params(labelsize=S_TEXT, length=2)
    hs = [Line2D([], [], ls="none", marker="o", ms=3, color=BLUE, label="count evidence"), Line2D([], [], ls="none", marker="o", ms=3, color=GREY, label="no count evidence (imputed from features)"),
          Line2D([], [], color=TREND, lw=1.1, label="median trend"), Line2D([], [], ls="none", label="↑ ↓  share of genes with Δα above +0.05, below −0.05")]
    fig.legend(handles=hs, loc="outside lower center", ncol=4, fontsize=S_TEXT, frameon=False, handletextpad=0.4, columnspacing=1.2)
    settle(fig)
    return fig


def label_set(k: str) -> str:
    fam, s = k.split("|", 1)
    return SET_LABEL.get(s, s) + {"hallmark": "", "class": " (class)", "broad": " (category)"}[fam]


def recurrence_rows(mode: str):
    R = pd.read_csv(OUT / "recurrence.csv")
    tabs = {k: R[R.readout == k].set_index("key") for k in R.readout.unique()}
    D = tabs["D"]
    ts = pd.read_csv(OUT / "tissue_specific.csv")
    ts = ts[(ts.readout == "D") & ts.tissue_specific]
    if mode == "strong":  # recurrent on D, on the count evidence, under both nulls, and matched on cell-type specificity
        keep = D.recurrent & tabs["D_tau"].recurrent & tabs["D_block"].recurrent & tabs["S"].recurrent & tabs["S_block"].recurrent
    else:
        keep = D.recurrent
    rec = D[keep]
    ab = rec[rec.direction == "above"].sort_values("stouffer", ascending=False)
    be = rec[rec.direction != "above"].sort_values("stouffer", ascending=False)
    keys = list(ab.index) + list(be.index)
    spec = [k for k in ts.sort_values("tissue").key if k not in keys]
    zc = [f"z_{t}" for t in RU.TISSUE_ORDER]
    opp = D[(D.n_up >= 1) & (D.n_down >= 1) & (D[zc].abs().max(1) >= 5) & ~D.index.isin(keys + spec) & (D.family == "hallmark")].sort_values("stouffer", ascending=False).index.tolist()
    return tabs, keys, len(ab), spec, opp


def main_recurrence(mode: str = "strong") -> plt.Figure:
    """Gene sets (rows) by tissue (columns): z of the set against matched genes, with the recurrence columns at the right.
    The rows stand in blocks with a heading each; the page is 486 bp wide, lengths below in bp."""
    tabs, keys, n_ab, spec, opp = recurrence_rows(mode)
    D, S, B, PI, W = tabs["D"], tabs["S"], tabs["B1"], tabs["pi"], tabs["dNdS"]
    blocks = [(h, r) for h, r in (("Recurrent, above the prior", keys[:n_ab]), ("Recurrent, below the prior", keys[n_ab:]),
                                  ("Called in one tissue only", spec), ("Opposite signs between tissues", opp)) if r]
    slots = [s for h, r in blocks for s in [("head", h)] + [("row", k) for k in r]]  # one line per heading and per set
    cols = RU.TISSUE_ORDER
    n, nc = len(slots), len(cols)
    rh, w_label, w_col, w_gap, w_side, top, bottom, pad = 7.7, 118.0, 29.0, 6.0, 25.0, 19.0, 46.0, 2.5
    Wb, Hb = pad + w_label + nc * w_col + w_gap + 5 * w_side + pad, top + n * rh + bottom
    fig = plt.figure(figsize=(style.inch(Wb), style.inch(Hb)))
    box = lambda x, y, w, h: [x / Wb, y / Hb, w / Wb, h / Hb]  # noqa: E731
    ax = fig.add_axes(box(pad + w_label, bottom, nc * w_col, n * rh))
    vmax = 8.0
    M = np.full((n, nc), np.nan)
    Q = np.full((n, nc), np.nan)
    for i, (kind, k) in enumerate(slots):
        if kind == "row":
            M[i], Q[i] = D.loc[k, [f"z_{t}" for t in cols]].to_numpy(float), D.loc[k, [f"q_{t}" for t in cols]].to_numpy(float)
    cmap = plt.get_cmap(style.DIVERGING).copy()
    cmap.set_bad("white")
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    yt = ax.get_yaxis_transform()
    x_left, x_right = -w_label / (nc * w_col), 1 + (w_gap + 5 * w_side) / (nc * w_col)
    for i, (kind, k) in enumerate(slots):
        if kind == "head":
            ax.text(x_left, i, k, transform=yt, ha="left", va="center", fontsize=S_TEXT + 0.3, fontweight="bold", color=INK, clip_on=False)
            if i:
                ax.plot([x_left, x_right], [i - 0.5, i - 0.5], transform=yt, color=INK, lw=0.6, clip_on=False)
            continue
        for j in range(nc):
            v = M[i, j]
            if not np.isfinite(v):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=NA_CELL, edgecolor="none"))
                continue
            ax.text(j, i, signed(v, 1) + (style.SIGNIFICANT if Q[i, j] < 0.05 else ""), ha="center", va="center", fontsize=S_TEXT, color="#ffffff" if abs(v) > 0.62 * vmax else INK)
    rows = [i for i, (kind, _) in enumerate(slots) if kind == "row"]
    ax.set_xticks(range(nc))
    ax.set_xticklabels([TISSUE_LABEL[t].replace(" ", "\n") for t in cols], fontsize=S_TICK, linespacing=0.95)
    ax.xaxis.tick_top()
    ax.set_yticks(rows)
    ax.set_yticklabels([label_set(slots[i][1]) for i in rows], fontsize=S_TEXT + 0.1)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax2 = fig.add_axes(box(pad + w_label + nc * w_col + w_gap, bottom, 5 * w_side, n * rh), sharey=ax)
    ax2.set_xlim(-0.5, 4.5)
    for j, hd in enumerate(["$q$", "$S_g$", "profile\ncorrelation", "π", "dN/dS"]):
        ax2.text(j, -0.75, hd, fontsize=S_TEXT + 0.1, ha="center", va="bottom", color=INK, linespacing=0.95)
    kw = dict(fontsize=S_TEXT, ha="center", va="center")
    for i in rows:
        k = slots[i][1]
        q = D.q_rec[k]
        ax2.text(0, i, f"{q:.3f}" if q >= 0.001 else "<.001", color=INK if k in keys else MUTED, **kw)
        for j, tab in zip((1, 2), (S, B)):
            r = tab.loc[k]
            ax2.text(j, i, f"{int(r.n_up)}↑{int(r.n_down)}↓", color=INK if r.recurrent else MUTED, **kw)
        for j, tab in zip((3, 4), (PI, W)):
            r = tab.loc[k]
            ax2.text(j, i, signed(r.mean_z, 1), color=INK if abs(r.mean_z) >= RU.Z_CALL else MUTED, **kw)
    ax2.axis("off")
    ax2.set_ylim(ax.get_ylim())
    # under the rows: the color scale at the left, the key of the cells and of the columns at the right
    cax = fig.add_axes(box(pad + 8, bottom - 13, 96, 5))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([-8, -4, 0, 4, 8])
    cb.set_ticklabels(["≤ −8", "−4", "0", "+4", "≥ +8"])
    cb.ax.tick_params(labelsize=S_TEXT, length=2, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cax.set_xlabel("set $z$ (color clipped at ±8)", fontsize=S_TEXT, labelpad=1.5)
    key = (f"Cells: $z$ of the set against matched genes of the tissue; {style.SIGNIFICANT} $q$ < 0.05; gray cell, fewer than {RU.MIN_SET} genes.\n"
           "Right: $q$, recurrence across tissues; $S_g$ and profile correlation, tissues with the set above (↑) and below (↓);\n"
           f"π and dN/dS, mean $z$ of the set over tissues. Gray type: not recurrent, or |mean $z$| below {RU.Z_CALL}.")
    fig.text((pad + w_label) / Wb, (bottom - 7) / Hb, key, ha="left", va="top", fontsize=S_TEXT, color=INK, linespacing=1.25)
    return fig


def main_lineage() -> plt.Figure:
    """Lineage marker sets (rows, with their own tissue) by tissue (columns): z against matched genes on Δα and on the
    count evidence. Outlined: the own tissue of the set. A cell without an estimate is light gray."""
    L = pd.read_csv(OUT / "lineage_z.csv")
    prim = {RU.PRIMARY[t]: t for t in RU.TISSUE_ORDER}
    L = L[L.cohort.isin(prim) & L.readout.isin(["D", "S"])].copy()
    L["tissue"] = L.cohort.map(prim)
    L["own"] = L.key.str.split("|").str[1]
    L["set"] = L.key.str.split("|").str[2]
    rows = [f"lineage|{t}|{s}" for t in RU.TISSUE_ORDER for s in RU.LINEAGE[t]]
    fig, axes = plt.subplots(1, 2, figsize=(style.inch(W_FULL), style.inch(7.7 * len(rows) + 78)), layout="constrained", sharey=True)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03)
    vmax = 8.0
    cmap = plt.get_cmap(style.DIVERGING).copy()
    cmap.set_bad(NA_CELL)
    for ax, rd, ttl in zip(axes, ["D", "S"], ["Δα", style.READOUT["count_evidence"].name]):
        P = L[L.readout == rd].pivot(index="key", columns="tissue", values="z").reindex(index=rows, columns=RU.TISSUE_ORDER)
        M = P.to_numpy(float)
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
        for i in range(M.shape[0]):
            own = RU.TISSUE_ORDER.index(rows[i].split("|")[1])
            ax.add_patch(Rectangle((own - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=0.9))
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, signed(M[i, j], 1), ha="center", va="center", fontsize=S_TEXT, color="#ffffff" if abs(M[i, j]) > 0.62 * vmax else INK)
        ax.set_xticks(range(len(RU.TISSUE_ORDER)))
        ax.set_xticklabels([TISSUE_LABEL[t] for t in RU.TISSUE_ORDER], fontsize=S_TICK, rotation=30, ha="right", rotation_mode="anchor")
        ax.tick_params(length=0)
        ax.set_title(ttl, fontsize=S_TITLE, pad=3)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"{THEME_LABEL.get(r.split('|')[2], r.split('|')[2])} ({TISSUE_LABEL[r.split('|')[1]].lower()})" for r in rows], fontsize=S_TEXT + 0.1)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", shrink=0.22, aspect=22, pad=0.01)
    cb.set_ticks([-8, -4, 0, 4, 8])
    cb.set_ticklabels(["−8", "−4", "0", "+4", "+8"])
    cb.ax.tick_params(labelsize=S_TEXT, length=2, pad=1.5)
    cb.outline.set_linewidth(0.4)
    cb.set_label("set $z$ against matched genes", fontsize=S_TEXT, labelpad=1.5)
    settle(fig)
    return fig


def main_genes(k_top=4, k_lin=5, k_down=5) -> plt.Figure:
    """Per tissue: the named genes farthest above the prior, the lineage genes of the tissue (bold), the genes farthest below."""
    nr = 2 if len(RU.TISSUE_ORDER) > 6 else 1
    nc = int(np.ceil(len(RU.TISSUE_ORDER) / nr))
    fig, axes = plt.subplots(nr, nc, figsize=(style.inch(W_FULL), style.inch(143 * nr)), layout="constrained", sharex=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    for ax, t in zip(np.ravel(axes), RU.TISSUE_ORDER):
        c = RU.PRIMARY[t]
        n = named(genes(c))
        n = n[~n.symbol.str.startswith("MT-")]  # mitochondrially encoded genes are not measured in every tissue and track RNA retention; see the caption
        panel = {s for v in RU.LINEAGE[t].values() for s in v}
        up = n[n.D > 0]
        lin = up[up.symbol.isin(panel)].nlargest(k_lin, "D")
        top = up[~up.symbol.isin(panel)].nlargest(k_top, "D")
        down = n[n.D < 0].nsmallest(k_down, "D").iloc[::-1]
        sel = pd.concat([top, lin, down])
        y = np.r_[np.arange(len(top)), len(top) + 0.5 + np.arange(len(lin)), len(top) + len(lin) + 1 + np.arange(len(down))]
        ax.barh(y, sel.D, color=[UP if v > 0 else DOWN for v in sel.D], height=0.68, lw=0)
        ax.axvline(0, color=MUTED, lw=0.5)
        ax.set_ylim(k_top + k_lin + k_down + 0.6, -0.6)
        ax.set_yticks([])
        ax.set_xlim(-0.9, 0.65)  # holds every bar of the eight tissues (−0.85 to +0.60)
        ax.set_xticks([-0.8, -0.4, 0, 0.4])
        ax.tick_params(labelsize=S_TEXT, length=2)
        if t in RU.TISSUE_ORDER[-nc:]:
            ax.set_xlabel(D_LABEL, fontsize=S_TICK)
        ax.set_title(TISSUE_LABEL[t], fontsize=S_TITLE, fontweight="bold", pad=3)
        ax.spines["left"].set_visible(False)
        for yy, r in zip(y, sel.itertuples()):
            ax.text(-0.015 if r.D > 0 else 0.015, yy, gene_label(r), fontsize=S_TEXT, ha="right" if r.D > 0 else "left", va="center", color=INK, fontweight="bold" if r.symbol in panel else "normal")
    settle(fig)
    return fig


# the panels that are not part of a tissue figure: name in figures/panels -> the function that draws it
SINGLE = {"main_tissues/axis": lambda: main_axis(), "main_tissues/genes": lambda: main_genes(), "main_tissues/recurrence": lambda: main_recurrence(),
          "supp_tissues_recurrence/recurrence_all": lambda: main_recurrence("all"), "supp_tissues_lineage/lineage": lambda: main_lineage()}


def draw_panels(tissues=None, single=None, reference: bool = False) -> None:
    """Draw the panels of this task from the cached tables of results/analysis/H1/ and nothing else: the eleven panels of
    each tissue figure (``tissues``: names as in rules.NEW_TISSUES; default all eight) and the panels of ``SINGLE``
    (``single``: their names; default all five). ``reference``: also the standard panels for liver and testis, to
    compare with the originals (extra_reference_liver, extra_reference_testis; in no figure)."""
    style.apply()
    for t in RU.NEW_TISSUES if tissues is None else tissues:
        supp_panels(t, SUPP_PRIMARY[t], FAMILY[t])
    if reference:
        for t in ("liver", "testis"):
            supp_panels(t, RU.PRIMARY[t], f"extra_reference_{t}")
    for name in SINGLE if single is None else single:
        psave(SINGLE[name](), name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", action="store_true", help="also render the standard panels for liver and testis, to compare with the originals")
    a = ap.parse_args()
    draw_panels(reference=a.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
