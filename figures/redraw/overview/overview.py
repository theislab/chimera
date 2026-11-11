"""Figure 1, the overview of Chimera: (a) problem, (b) inputs, (c) model, (d) outputs, (e) applications.

The whole figure is drawn here as vector shapes with embedded text (DejaVu Sans, TrueType); nothing is
pasted from another figure. One unit of every panel is one PostScript point at the size of the figure on
the page (486 pt wide, the text width of the manuscript), so the font sizes below are the printed sizes;
the smallest is 5.1 pt.

    python figures/redraw/overview/overview.py            # the four panel files, without letters, into
                                                          # figures/panels/main_overview/ (problem, inputs_model,
                                                          # outputs, applications); the figure is composed from
                                                          # them like every other one (key main_overview)
    python figures/redraw/overview/overview.py --check    # layout report: text outside a card, overlapping text
    python figures/redraw/overview/overview.py --whole DIR --png   # also the whole figure with letters, as one
                                                          # file, for a look without the compose tool

Panels (b) and (c) are one file, ``inputs_model``: the three inputs lead into the model with arrows, so the
two cards are drawn on one page and the figure spec places both letters under it (``LETTER_X``).

Inputs, all in ``assets/`` next to this file (made once by ``build_assets.py``):
    silhouettes.json                     outlines of the four species
    liver_human_monkey_delta_alpha.csv   panel (d) histogram: delta alpha of the liver Human+Monkey
                                         reference model, 8,430 genes, in the 80 bins of the paper's axis
    liver_human_monkey_umap.csv          panel (d) embedding: 1,500 cells of the same model
    eight_tissues.csv                    panel (e) strip: Spearman correlations of delta alpha and of dN/dS
                                         with protein identity in the eight tissues of the one-reading figure
Everything else in the figure is a schematic and uses no data: the two genes of panel (a), the dial of
panel (c), and in panel (d) the five example genes with their count evidence, gate and delta alpha (the
values printed under the dials are those of the drawn dials). The output is the same file on every run.

Colors follow the other figures: above the protein prior vermilion, below it blue; the shared decoder
takes the color of "above" (a gene the shared decoder explains moves above its prior) and the species-aware
residual decoder the color of "below"; species, liver cell types, delta alpha and the sequence measures as
in the other figures. Everything else is neutral: soft gray cards without frames, white tiles for data, and
one white tile with a dark outline, the gate, as the place the eye goes first.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle, Wedge  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
PANEL_DIR = Path(os.environ.get("CHIMERA_FIGURES_OUT", HERE.parents[1])) / "panels" / "main_overview"

# ---------------------------------------------------------------------------------------------- style
INK, SUB, MUTED, RULE = "#222222", "#555555", "#777777", "#BBBBBB"
ABOVE, BELOW = "#D55E00", "#0072B2"  # above / below the protein prior
NEUTRAL = "#BDBDBD"
CARD, TILE_EDGE, TRACK = "#F2F3F5", "#DADDE2", "#D9DCE1"  # panel card, edge of a white tile, scale of a dial
SLATE = "#4B515A"  # sequence and feature cells, the capsule
BLOCK, BLOCK_EDGE = "#E3E6EA", "#8B919A"  # encoder and gate network
SPECIES = {"human": "#222222", "macaque": "#0072B2", "mouse": "#D55E00", "zebrafish": "#009E73"}
CELLTYPE = {"hepatocytes": "#E69F00", "endothelial": "#56B4E9", "fibroblasts": "#009E73",
            "macrophages": "#CC79A7", "other": "#BBBBBB"}
CELLTYPE_OF = {"Hepatocytes": "hepatocytes", "Endothelial": "endothelial", "fibroblasts": "fibroblasts",
               "Kupffer cells": "macrophages", "Macrophages": "macrophages"}  # the rest: other
DELTA_ALPHA, SEQUENCE = "#D55E00", "#A6761D"  # delta alpha and the sequence measures in the other figures

TITLE, HEAD, BODY, SMALL, TINY = 7.6, 6.2, 6.0, 5.4, 5.1  # font sizes, pt on the page
LETTER = 7.0
LW_HAIR, LW, LW_FLOW = 0.5, 0.8, 1.25  # three line weights: axes and rules, connectors, the two decoder streams
FRAME = 1.2  # inset of a card from the edge of its file, so no ink touches the edge of a panel file
PAD = 7.5  # from the edge of a file to the text of a card

W = 486.0  # figure width, pt: the text width of the manuscript
BAND, VGAP, HGAP = 9.5, 6.0, 7.0  # letter band under a row, gap between rows, gap between files (as the compose tool)
ROWS = [  # (height, {file: width}); widths of a row and its gaps add up to W
    (144.0, {"problem": 112.0, "inputs_model": 367.0}),
    (148.0, {"outputs": 284.0, "applications": 195.0}),
]
SPLIT = (93.2, 102.6)  # in inputs_model: right edge of the inputs card, left edge of the model card
LETTER_X = {"problem": {"a": 56.0}, "inputs_model": {"b": 47.2, "c": 234.2}, "outputs": {"d": 142.0},
            "applications": {"e": 97.5}}  # panel letters: x of each letter in its file
H = sum(h for h, _ in ROWS) + len(ROWS) * BAND + (len(ROWS) - 1) * VGAP
FILES = {}  # name: (x, y, w, h), y from the bottom of the figure
_y = H
for _h, _row in ROWS:
    assert abs(sum(_row.values()) + HGAP * (len(_row) - 1) - W) < 1e-9, _row
    _y -= _h
    _x = 0.0
    for _k, _w in _row.items():
        FILES[_k] = (_x, _y, _w, _h)
        _x += _w + HGAP
    _y -= BAND + VGAP


def tint(color: str, f: float) -> tuple:
    """Color mixed with white; f = 1 is the color itself."""
    c = np.array(matplotlib.colors.to_rgb(color))
    return tuple(c * f + (1 - f))


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "mathtext.default": "it",
        "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": INK, "figure.dpi": 72,
        "path.simplify": False, "axes.unicode_minus": True,
    })


# ------------------------------------------------------------------------------------------ primitives
def T(ax, x, y, s, size=BODY, ha="left", va="baseline", color=INK, weight="normal", style="normal", rot=0):
    return ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, fontweight=weight, fontstyle=style,
                   rotation=rot, rotation_mode="anchor", clip_on=False, zorder=6)


def lines(ax, x, y, rows, size=BODY, ha="left", color=INK, pitch=None, **kw):
    """Stacked lines of text; y is the baseline of the first."""
    pitch = size * 1.16 if pitch is None else pitch
    return [T(ax, x, y - i * pitch, s, size=size, ha=ha, color=color, **kw) for i, s in enumerate(rows)]


def width(ax, s, size=BODY, weight="normal") -> float:
    """Width of a text in pt (figure dpi is 72, so one pixel is one point)."""
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight)
    w = t.get_window_extent(ax.figure.canvas.get_renderer()).width
    t.remove()
    return w


def card(ax, x0, y0, x1, y1):
    """The soft ground of a panel."""
    r = 5.0
    ax.add_patch(FancyBboxPatch((x0 + r, y0 + r), x1 - x0 - 2 * r, y1 - y0 - 2 * r,
                                boxstyle=f"round,pad={r},rounding_size={r}", fc=CARD, ec="none", zorder=0.5, clip_on=False))


def title(ax, x, y, name, sub=""):
    """Panel title, bold, with a subtitle in regular weight on the same line."""
    T(ax, x, y, name, size=TITLE, weight="bold")
    if sub:
        T(ax, x + width(ax, name, TITLE, "bold") + 4.5, y, sub, size=SMALL, color=SUB)


def tile(ax, x, y, w, h, r=3.0, fc="white", ec=TILE_EDGE, lw=LW_HAIR, z=1.5):
    p = FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r},rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, clip_on=False, joinstyle="round")
    ax.add_patch(p)
    return p


def _rounded(pts, r, closed=False):
    """Path through the points with corners of radius r."""
    pts = [np.asarray(p, float) for p in pts]
    n = len(pts)
    verts, codes = [], []
    idx = range(n) if closed else range(1, n - 1)
    if not closed:
        verts.append(pts[0])
        codes.append(MPath.MOVETO)
    for i in idx:
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
        a, b = p0 - p1, p2 - p1
        la, lb = np.hypot(*a), np.hypot(*b)
        rr = min(r, la / 2, lb / 2)
        s, e = p1 + a / la * rr, p1 + b / lb * rr
        verts.append(s)
        codes.append(MPath.LINETO if verts[:-1] else MPath.MOVETO)
        verts += [p1, e]
        codes += [MPath.CURVE3, MPath.CURVE3]
    if closed:
        verts.append(verts[0])
        codes.append(MPath.CLOSEPOLY)
    else:
        verts.append(pts[-1])
        codes.append(MPath.LINETO)
    return MPath(verts, codes)


def shape(ax, pts, fc, ec, lw=LW, r=1.6, z=2):
    """A filled polygon with softened corners."""
    p = PathPatch(_rounded(pts, r, closed=True), fc=fc, ec=ec, lw=lw, zorder=z, joinstyle="round", clip_on=False)
    ax.add_patch(p)
    return p


def node(ax, x, y, r=5.2, s="", fc="white", ec=INK, size=BODY, lw=LW):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=4, clip_on=False))
    if s:
        T(ax, x, y, s, size=size, ha="center", va="center_baseline")


def arrow(ax, pts, color=INK, lw=LW, dashed=False, head=True, hl=3.4, hw=2.8, r=2.4, z=3):
    """Line through pts with rounded corners and a filled head at the end."""
    pts = [np.asarray(p, float) for p in pts]
    end = pts[-1]
    if head:
        d = end - pts[-2]
        d = d / np.hypot(*d)
        n = np.array([-d[1], d[0]])
        base = end - hl * d
        ax.add_patch(Polygon([end, base + n * hw / 2, base - n * hw / 2], closed=True, fc=color, ec=color, lw=0.3,
                             zorder=z + 0.1, joinstyle="miter", clip_on=False))
        pts = pts[:-1] + [end - 0.8 * hl * d]
    ax.add_patch(PathPatch(_rounded(pts, r), fc="none", ec=color, lw=lw, ls=(0, (2.2, 1.5)) if dashed else "-",
                           zorder=z, capstyle="butt", joinstyle="round", clip_on=False))


def silhouette(ax, shapes, name, x, y, height, color):
    """Species outline with its lower left at (x, y); returns its width."""
    verts, codes = [], []
    for ring in shapes[name]["rings"]:
        r = np.asarray(ring) * height + [x, y]
        verts += r.tolist()
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(r) - 2) + [MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", zorder=3, clip_on=False))
    return shapes[name]["aspect"] * height


def cells(ax, x, y, values, cell, pitch, h, color=SLATE, r=0.6):
    """A row of small rounded cells, each filled with a tint of ``color``; returns the width of the row."""
    for i, f in enumerate(values):
        ax.add_patch(FancyBboxPatch((x + i * pitch + r, y + r), cell - 2 * r, h - 2 * r,
                                    boxstyle=f"round,pad={r},rounding_size={r}", fc=tint(color, float(f)), ec="none",
                                    zorder=3, clip_on=False))
    return (len(values) - 1) * pitch + cell


def strips(ax, x, y, mismatch, n=10, species=("human", "mouse"), marks=True, cell=3.4, pitch=4.0, h=3.6, gap=1.4):
    """Two aligned protein sequences, one per species; positions that differ are dark in the second. Returns the width."""
    for row, sp in enumerate(species):
        yy = y + (1 - row) * (h + gap)
        if marks:
            ax.add_patch(Rectangle((x - 3.2, yy), 1.5, h, fc=SPECIES[sp], ec="none", zorder=3, clip_on=False))
        cells(ax, x, yy, [0.78 if (row == 1 and i in mismatch) else 0.2 for i in range(n)], cell, pitch, h)
    return (n - 1) * pitch + cell


def profile(ax, x, y, w, h, series):
    """Expression of one gene across cell states, one line per species; series: {species: values in 0..1}."""
    ax.plot([x, x + w], [y, y], color=INK, lw=LW_HAIR, zorder=3, solid_capstyle="butt", clip_on=False)
    for sp, v in series.items():
        xs = x + (np.arange(len(v)) + 0.5) * w / len(v)
        ys = y + 1.8 + np.asarray(v) * (h - 3.6)
        ax.plot(xs, ys, color=SPECIES[sp], lw=1.0, zorder=4, solid_joinstyle="round", clip_on=False)
        ax.scatter(xs, ys, s=3.4, color=SPECIES[sp], linewidths=0, zorder=5, clip_on=False)
    for xs in x + (np.arange(len(v)) + 0.5) * w / len(v):
        ax.plot([xs, xs], [y, y - 1.5], color=INK, lw=LW_HAIR, zorder=3, clip_on=False)


def dial(ax, cx, cy, r, prior, alpha, track=2.4, imputed=False, needle=1.0, hub=1.25, base=True):
    """A half-circle scale from 0 (left) to 1 (right): tick at the prior, needle at alpha, arc between them."""
    ang = lambda v: 180.0 - 180.0 * v  # noqa: E731
    pt = lambda v, rr: (cx + rr * np.cos(np.radians(ang(v))), cy + rr * np.sin(np.radians(ang(v))))  # noqa: E731
    ax.add_patch(Wedge((cx, cy), r, 0, 180, width=track, fc=TRACK, ec="none", zorder=3, clip_on=False))
    col = MUTED if imputed else (ABOVE if alpha > prior else BELOW)
    if abs(alpha - prior) > 1e-6:
        lo, hi = sorted([ang(prior), ang(alpha)])
        ax.add_patch(Wedge((cx, cy), r, lo, hi, width=track, fc=tint(col, 0.55) if imputed else col, ec="none",
                           zorder=3.2, clip_on=False))
    ax.plot(*zip(pt(prior, r - track - 0.8), pt(prior, r + (0.34 * r if r > 12 else 2.3)), strict=True), color=INK, lw=0.9 * needle, zorder=3.6,
            solid_capstyle="butt", clip_on=False)  # the prior
    ax.plot(*zip((cx, cy), pt(alpha, r - 0.5 * track), strict=True), color=MUTED if imputed else INK, lw=needle,
            ls=(0, (1.6, 1.2)) if imputed else "-", zorder=3.8, solid_capstyle="round", clip_on=False)  # alpha
    ax.add_patch(Circle((cx, cy), hub, fc=MUTED if imputed else INK, ec="none", zorder=4, clip_on=False))
    if base:
        ax.plot([cx - r - 1.4, cx + r + 1.4], [cy, cy], color=RULE, lw=LW_HAIR, zorder=2.5, clip_on=False)
    return pt


# ---------------------------------------------------------------------------------------------- panels
def panel_problem(ax, w, h, shapes):
    card(ax, FRAME, FRAME, w - FRAME, h - FRAME)
    title(ax, PAD, h - 12.0, "Problem", "two measurements")
    pattern = np.array([0.2, 1.0, 0.55, 0.3, 0.15])  # one pattern across cell states
    genes = [  # caption of the protein, differing positions, profiles, caption of the use
        # upper card: the same pattern in both species, one species clearly above the other (a difference of level)
        (["conserved", "protein"], [6], {"human": 0.02 + 0.5 * pattern, "mouse": 0.47 + 0.5 * pattern},
         ["used at a", "different level"]),
        (["less conserved", "protein"], [1, 4, 7], {"human": [0.15, 0.3, 0.92, 0.5, 0.15], "mouse": [0.22, 0.36, 0.8, 0.42, 0.1]},
         ["used", "alike"]),
    ]
    # species key
    x = w - PAD
    for sp in ("mouse", "human"):
        x -= width(ax, sp, SMALL)
        T(ax, x, h - 24.6, sp, size=SMALL, color=SUB)
        hs = {"human": 9.6, "mouse": 6.2}[sp]
        x -= 2.2 + shapes[sp]["aspect"] * hs
        silhouette(ax, shapes, sp, x, h - 25.8, hs, SPECIES[sp])
        x -= 6.5
    th, x0 = 45.0, 6.0
    top = h - 28.5
    for i, (protein, mism, series, use) in enumerate(genes):
        y0 = top - th - i * (th + 3.5)
        tile(ax, x0, y0, w - 2 * x0, th)
        sw = strips(ax, x0 + 10.2, y0 + 25.5, mism, n=9)
        cx = x0 + 10.2 + sw / 2
        T(ax, cx, y0 + 17.9, "sequence", size=TINY, ha="center", color=MUTED)
        T(ax, cx, y0 + 10.6, protein[0], size=BODY, ha="center", weight="bold")
        T(ax, cx, y0 + 4.1, protein[1], size=BODY, ha="center")
        px, pw = x0 + 53.0, 40.0
        profile(ax, px, y0 + 24.8, pw, 16.8, series)
        T(ax, px + pw / 2, y0 + 17.9, "cell states", size=TINY, ha="center", color=MUTED)
        T(ax, px + pw / 2, y0 + 10.6, use[0], size=BODY, ha="center")
        T(ax, px + pw / 2, y0 + 4.1, use[1], size=BODY, ha="center", weight="bold")
    T(ax, w / 2, 14.6, "Deployment is per gene;", ha="center", weight="bold")
    T(ax, w / 2, 7.7, "sequence does not tell which.", ha="center")


def panel_inputs_model(ax, w, h, shapes):
    xb, xc = SPLIT
    card(ax, FRAME, FRAME, xb, h - FRAME)
    card(ax, xc, FRAME, w - FRAME, h - FRAME)
    sh, rs = ABOVE, BELOW
    y_enc, dy = 90.0, 17.5
    y_sh, y_rs = y_enc + dy, y_enc - dy
    y_prior, y_feat = 52.5, 23.6
    rng = np.random.default_rng(3)

    # ------------------------------------------------------------------ (b) inputs
    title(ax, PAD, h - 12.0, "Inputs")
    heights = {"human": 19.0, "macaque": 14.6, "mouse": 11.6, "zebrafish": 8.4}
    widths = {k: shapes[k]["aspect"] * v for k, v in heights.items()}
    x_left, x_right = PAD, xb - 7.5
    gap = (x_right - x_left - sum(widths.values())) / 3
    x = x_left
    for sp in heights:
        silhouette(ax, shapes, sp, x, y_enc + 9.6, heights[sp], SPECIES[sp])
        cx = x + widths[sp] / 2
        for j in range(3):
            cells(ax, cx - 5.45, y_enc - 5.45 + j * 3.8, rng.choice([0.14, 0.3, 0.5, 0.78], size=3), 3.3, 3.8, 3.3,
                  color=SPECIES[sp])
        x += widths[sp] + gap
    arrow(ax, [(x_right + 2.0, y_enc), (xc + 6.3, y_enc)])
    T(ax, PAD, y_enc - 14.6, "scRNA-seq counts $x$")
    lines(ax, PAD, y_enc - 21.2, ["several species,", "one-to-one orthologs"], size=SMALL, color=SUB)
    # protein prior
    sw = strips(ax, PAD, y_prior - 4.3, [2, 7, 10], n=12, marks=False)
    arrow(ax, [(PAD + sw + 3.0, y_prior), (232.0, y_prior)])
    T(ax, PAD, y_prior - 13.4, "protein prior $\\pi_g$")
    T(ax, PAD, y_prior - 19.8, "identity of the orthologs", size=SMALL, color=SUB)
    # gene features
    fw = cells(ax, PAD, y_feat - 2.8, rng.choice([0.14, 0.26, 0.4, 0.58, 0.8], size=13), 4.0, 4.6, 5.6)
    arrow(ax, [(PAD + fw + 3.0, y_feat), (xc + 6.3, y_feat)])
    T(ax, PAD, y_feat - 11.0, "gene features $f_g$")
    T(ax, PAD, y_feat - 17.2, "ESM-2, identity, expression", size=TINY, color=SUB)

    # ------------------------------------------------------------------ (c) model
    x0 = xc + 6.3
    title(ax, x0, h - 12.0, "Model")
    # encoder and latent
    xe0, xe1 = x0, x0 + 30.0
    shape(ax, [(xe0, y_enc - 13), (xe0, y_enc + 13), (xe1, y_enc + 7.5), (xe1, y_enc - 7.5)], fc=BLOCK, ec=BLOCK_EDGE)
    T(ax, (xe0 + xe1) / 2 - 0.3, y_enc, "Encoder", size=BODY, ha="center", va="center_baseline")
    xz = xe1 + 10.2
    arrow(ax, [(xe1, y_enc), (xz - 5.6, y_enc)], hl=3.0, hw=2.5)
    node(ax, xz, y_enc, 5.6, "$z$", fc="white", size=6.6)
    lines(ax, x0, y_enc + 29.0, ["cell state $z$,", "species-mixed", "(GRL + MMD)"], size=SMALL, color=SUB)
    # decoders
    xd0, xd1 = xz + 14.0, xz + 78.0
    xbr = xz + 8.6
    arrow(ax, [(xbr, y_enc), (xbr, y_sh), (xd0, y_sh)], color=sh, lw=LW_FLOW)
    arrow(ax, [(xbr, y_enc), (xbr, y_rs), (xd0, y_rs)], color=rs, lw=LW_FLOW)
    ax.plot([xz + 5.6, xbr], [y_enc, y_enc], color=INK, lw=LW, zorder=3.3, solid_capstyle="butt", clip_on=False)
    ax.add_patch(Circle((xbr, y_enc), 1.35, fc=INK, ec="none", zorder=3.5, clip_on=False))
    for yc, col, name, sub in ((y_sh, sh, "Shared decoder", "species-blind"), (y_rs, rs, "Residual decoder", "species-aware")):
        shape(ax, [(xd0, yc - 6.5), (xd0, yc + 6.5), (xd1, yc + 10.5), (xd1, yc - 10.5)], fc=tint(col, 0.16), ec=col)
        T(ax, (xd0 + xd1) / 2 + 1, yc + 1.4, name, size=BODY, ha="center")
        T(ax, (xd0 + xd1) / 2 + 1, yc - 5.2, sub, size=TINY, ha="center", color=SUB)
    T(ax, xd0 + 9.5, y_enc - 1.9, "species, batch", size=TINY, color=rs)
    arrow(ax, [(xd0 + 5.0, y_enc + 2.6), (xd0 + 5.0, y_rs + 7.4)], color=rs, lw=LW, hl=2.8, hw=2.4)
    # the gate: a white tile with a dial, the focal element
    gx0, gx1, gy0, gy1 = 232.0, w - FRAME - 5.0, FRAME + 5.0, y_rs - 13.4
    tile(ax, gx0, gy0 - 1.1, gx1 - gx0, gy1 - gy0, r=4.0, fc="#C9CDD3", ec="none", z=1.9)
    tile(ax, gx0, gy0, gx1 - gx0, gy1 - gy0, r=4.0, ec=INK, lw=0.9, z=2)
    T(ax, gx0 + 6.0, gy1 - 9.4, "Sequence-anchored $\\alpha$-gating", size=HEAD, weight="bold")
    cx, cy, r = gx0 + 27.0, gy0 + 10.8, 19.0
    prior, alpha = 0.36, 0.74
    pt = dial(ax, cx, cy, r, prior, alpha, track=3.6, needle=1.25, hub=1.9, base=False)
    for v, lab, col in ((0.0, "0", rs), (1.0, "1", sh)):
        T(ax, pt(v, r - 1.8)[0], cy - 6.2, lab, size=SMALL, ha="center", color=col)
    T(ax, *pt(prior, r + 9.6), "$\\pi_g$", size=BODY, ha="center", va="center")
    T(ax, pt(alpha, r + 6.2)[0] + 1.2, pt(alpha, r + 6.2)[1] - 0.6, "$\\alpha_g$", size=BODY, ha="center", va="center")
    T(ax, pt((prior + alpha) / 2, r + 4.6)[0] + 1.6, pt((prior + alpha) / 2, r + 4.6)[1], "$\\Delta\\alpha_g$", size=BODY,
      ha="center", va="center", color=sh)
    fx = gx0 + 56.0
    T(ax, fx, gy1 - 21.6, "$\\alpha_g=\\sigma\\mathrm{(logit}\\,\\pi_g+c_g\\mathrm{)}$", size=6.4)
    T(ax, fx, gy1 - 30.8, "$\\Delta\\alpha_g=\\alpha_g-\\pi_g$", size=6.4)
    lines(ax, fx, gy1 - 39.6, ["0: full residual", "1: shared decoder only"], size=TINY, color=SUB, pitch=5.9)
    # gate on the residual, sum, likelihood
    xs_ = 298.0
    arrow(ax, [(xd1, y_rs), (cx - 5.2, y_rs)], color=rs, lw=LW_FLOW)
    node(ax, cx, y_rs, 5.2, "×", size=7.0)
    arrow(ax, [(cx, gy1), (cx, y_rs - 5.2)], lw=0.9)
    T(ax, cx + 3.4, (gy1 + y_rs - 5.2) / 2 - 2.3, "$\\mathrm{1}-\\alpha_g$", size=BODY)
    arrow(ax, [(cx + 5.2, y_rs), (xs_, y_rs), (xs_, y_enc - 5.2)], color=rs, lw=LW_FLOW)
    arrow(ax, [(xd1, y_sh), (xs_, y_sh), (xs_, y_enc + 5.2)], color=sh, lw=LW_FLOW)
    node(ax, xs_, y_enc, 5.2, "+", size=7.0)
    lines(ax, xs_ - 13.6, y_enc + 0.9, ["zero-sum", "batch offset"], size=TINY, ha="right", color=SUB, pitch=5.9)
    arrow(ax, [(xs_ - 12.0, y_enc), (xs_ - 5.2, y_enc)], hl=2.8, hw=2.4)
    arrow(ax, [(xs_ + 5.2, y_enc), (xs_ + 14.0, y_enc)])
    T(ax, xs_ + 15.6, y_enc, "$\\mathrm{NB(}x\\,|\\,\\mu,\\theta\\mathrm{)}$", size=BODY, va="center_baseline")
    # count evidence: from the shared decoder alone, around the model, into the gate
    xt, yt, xr = xd1 + 8.0, h - 22.6, w - FRAME - 10.0
    ax.add_patch(Circle((xt, y_sh), 1.3, fc=sh, ec="none", zorder=3.4, clip_on=False))
    arrow(ax, [(xt, y_sh), (xt, yt), (xr, yt), (xr, gy1)], lw=0.9, dashed=True, r=3.5)
    s = "count evidence $S_g$:"
    T(ax, xt + 3.5, h - 12.0, s, size=BODY, weight="bold")
    T(ax, xt + 3.5 + width(ax, s, BODY, "bold") + 2.4, h - 12.0, "shared decoder alone", size=TINY, color=SUB)
    T(ax, xt + 3.5, h - 18.2, "against a per-species average", size=TINY, color=SUB)
    T(ax, xr - 3.0, gy1 + 4.6, "trains $c_g$", size=SMALL, ha="right", color=SUB)
    # the gate network and the two gates
    xg0, xg1 = x0, x0 + 22.0
    tile(ax, xg0, y_feat - 6.0, xg1 - xg0, 12.0, r=2.0, fc=BLOCK, ec=BLOCK_EDGE, lw=LW, z=2)
    T(ax, (xg0 + xg1) / 2, y_feat, "GIN", size=BODY, ha="center", va="center_baseline")
    arrow(ax, [(xg1, y_feat), (gx0, y_feat)])
    T(ax, xg1 + 4.0, y_feat + 2.4, "$c_g$", size=BODY)
    T(ax, gx0 - 4.5, y_prior + 2.4, "$\\pi_g$", size=BODY, ha="right")
    xn = xg1 + 3.5
    lines(ax, xn, y_prior - 9.2, ["protein-feature gate (default):", "$c_g=\\mathrm{GIN(}f_g\\mathrm{)}$, every gene"], size=TINY, pitch=6.1)
    lines(ax, xn, y_feat - 9.6, ["per-gene gate (companion):", "$c_g$ free, one offset per gene"], size=TINY, color=SUB, pitch=6.1)


def panel_outputs(ax, w, h, hist, cells_):
    card(ax, FRAME, FRAME, w - FRAME, h - FRAME)
    title(ax, PAD, h - 12.0, "Outputs", "per cell and per gene, from one fit")
    yh = h - 23.2  # baseline of the two block headings
    # ---- per cell: the embedding (real cells of one model)
    T(ax, PAD, yh, "Per cell:", weight="bold")
    T(ax, PAD + width(ax, "Per cell:", BODY, "bold") + 3.4, yh, "embedding $z$")
    rng = np.random.default_rng(1)
    order = rng.permutation(len(cells_["x"]))
    xs, ys = cells_["x"][order], cells_["y"][order]
    span = max(np.ptp(xs), np.ptp(ys))
    side = 44.0

    def umap(x0, y0, colors):
        px = x0 + side / 2 + (xs - (xs.min() + xs.max()) / 2) / span * (side - 4)
        py = y0 + side / 2 + (ys - (ys.min() + ys.max()) / 2) / span * (side - 4)
        tile(ax, x0, y0, side, side)
        ax.scatter(px, py, s=0.8, c=colors, linewidths=0, zorder=3, clip_on=False)

    sp_name = {"Human": "human", "Monkey": "macaque"}
    xl = PAD + side + 4.5
    y1 = yh - 5.5 - side
    umap(PAD, y1, [SPECIES[sp_name[sp]] for sp in cells_["species"][order]])
    T(ax, xl, y1 + side - 6.5, "by species", size=TINY, color=MUTED)
    for i, (lab, key) in enumerate([("human", "human"), ("monkey", "macaque")]):
        ax.add_patch(Circle((xl + 1.7, y1 + side - 12.6 - i * 6.5), 1.4, fc=SPECIES[key], ec="none", zorder=3, clip_on=False))
        T(ax, xl + 5.2, y1 + side - 14.5 - i * 6.5, lab, size=SMALL)
    ct = [CELLTYPE_OF.get(c, "other") for c in cells_["cell_type"][order]]
    y2 = y1 - 4.0 - side
    umap(PAD, y2, [CELLTYPE[c] for c in ct])
    T(ax, xl, y2 + side - 6.5, "by cell type", size=TINY, color=MUTED)
    for i, lab in enumerate(CELLTYPE):
        ax.add_patch(Circle((xl + 1.7, y2 + side - 12.6 - i * 6.4), 1.4, fc=CELLTYPE[lab], ec="none", zorder=3, clip_on=False))
        T(ax, xl + 5.2, y2 + side - 14.5 - i * 6.4, lab, size=SMALL)
    lines(ax, PAD, y2 - 7.4, ["species aligned: clustering,", "label transfer; the substrate", "of the gene readouts"],
          size=TINY, color=SUB, pitch=5.9)

    # ---- per gene: count evidence, gate, delta alpha (five example genes, schematic)
    gx, gr = 106.0, w - PAD
    T(ax, gx, yh, "Per gene:", weight="bold")
    T(ax, gx + width(ax, "Per gene:", BODY, "bold") + 3.4, yh, "five example orthologs (schematic)")
    pitch = 21.2
    cols = [gr - 10.8 - pitch * i for i in range(5)][::-1]
    sx0, sx1 = cols[0] - 10.8, cols[-1] + 10.8
    genes = [  # count evidence, prior, alpha (protein-feature gate), delta alpha read gene by gene
        (1.0, 0.50, 0.80, "+0.28"), (0.45, 0.80, 0.93, "+0.09"), (-0.5, 0.86, 0.70, "−0.12"),
        (-1.0, 0.70, 0.36, "−0.31"), (0.0, 0.62, 0.75, "0")]
    y0 = yh - 16.5
    T(ax, gx, y0 - 2.1, "count evidence $S_g$")
    ax.plot([cols[0] - 6.5, cols[3] + 6.5], [y0, y0], color=INK, lw=LW_HAIR, zorder=3, clip_on=False)
    for cx, (sg, _, _, _) in zip(cols[:4], genes[:4], strict=True):
        ax.add_patch(Rectangle((cx - 3.8, y0), 7.6, sg * 7.0, fc=ABOVE if sg > 0 else BELOW, ec="none", zorder=3.2,
                               clip_on=False))
    T(ax, cols[4], y0 - 1.9, "none", size=TINY, ha="center", color=MUTED)
    yd = y0 - 25.0
    tile(ax, sx0, yd - 3.6, sx1 - sx0, 18.2, r=4.0)
    lines(ax, gx, yd + 7.4, ["prior $\\pi_g$: tick", "gate $\\alpha_g$: needle"], pitch=7.0)
    for i, (cx, (_, p, a, _)) in enumerate(zip(cols, genes, strict=True)):
        dial(ax, cx, yd, 8.8, p, a, imputed=i == 4)
    yt = yd - 12.5
    T(ax, gx, yt, "$\\Delta\\alpha_g=\\alpha_g-\\pi_g$")
    T(ax, sx1, yt, "gray: imputed from protein features", size=TINY, ha="right", color=MUTED)
    T(ax, gx, yt - 7.0, "protein-feature gate", size=SMALL)
    T(ax, gx, yt - 13.7, "per-gene readout", size=SMALL)
    for i, (cx, (_, p, a, free)) in enumerate(zip(cols, genes, strict=True)):
        d = a - p
        col = MUTED if i == 4 else ABOVE if d > 0 else BELOW
        T(ax, cx, yt - 7.0, f"{d:+.2f}".replace("-", "−"), size=SMALL, ha="center", color=col,
          style="italic" if i == 4 else "normal")
        T(ax, cx, yt - 13.7, free, size=SMALL, ha="center", color=MUTED if i == 4 else ABOVE if free[0] == "+" else BELOW)
    # ---- delta alpha over all genes of one model (real)
    hx0, hx1, hy0, hy1 = gx + 55.0, sx1 - 51.0, 21.0, yt - 20.0
    left, right, n = hist
    top = n.max()
    for a, b, k in zip(left, right, n, strict=True):
        mid = (a + b) / 2
        col = ABOVE if mid > 0.05 else BELOW if mid < -0.05 else NEUTRAL
        xa = hx0 + (a + 0.4) / 0.8 * (hx1 - hx0)
        xb = hx0 + (b + 0.4) / 0.8 * (hx1 - hx0)
        ax.add_patch(Rectangle((xa, hy0), xb - xa, k / top * (hy1 - hy0), fc=col, ec="none", zorder=3, clip_on=False))
    ax.plot([hx0, hx1], [hy0, hy0], color=INK, lw=0.6, zorder=4, solid_capstyle="butt", clip_on=False)
    for v, lab in [(-0.4, "−0.4"), (-0.2, ""), (0.0, "0"), (0.2, ""), (0.4, "+0.4")]:
        xv = hx0 + (v + 0.4) / 0.8 * (hx1 - hx0)
        ax.plot([xv, xv], [hy0, hy0 - 2.0], color=INK, lw=LW_HAIR, zorder=4, clip_on=False)
        if lab:
            T(ax, xv, hy0 - 7.4, lab, size=TINY, ha="center")
    T(ax, (gx + sx1) / 2, 7.0, "$\\Delta\\alpha_g$ of one model: liver Human+Monkey (8,430 genes)", size=SMALL, ha="center")
    T(ax, hx0 - 4.0, hy1 - 5.0, "below the prior", size=SMALL, ha="right", color=BELOW, weight="bold")
    lines(ax, hx0 - 4.0, hy1 - 11.2, ["pattern or level", "parts between", "species"], size=TINY, ha="right", color=SUB, pitch=5.9)
    T(ax, hx1 + 4.0, hy1 - 5.0, "above the prior", size=SMALL, color=ABOVE, weight="bold")
    lines(ax, hx1 + 4.0, hy1 - 11.2, ["pattern shared", "across species"], size=TINY, color=SUB, pitch=5.9)


def panel_applications(ax, w, h, tissues):
    card(ax, FRAME, FRAME, w - FRAME, h - FRAME)
    title(ax, PAD, h - 12.0, "Applications", "of the readouts")
    yh = h - 23.2
    s = "Eight tissues, one reading:"
    T(ax, PAD, yh, s, weight="bold")
    T(ax, PAD + width(ax, s, BODY, "bold") + 3.4, yh, "a separate axis")
    # ---- strip: correlation with protein identity, tissue by tissue (real)
    tx0, tx1, ty0, ty1 = PAD, w - PAD, yh - 46.4, yh - 4.2
    tile(ax, tx0, ty0, tx1 - tx0, ty1 - ty0)
    ax0 = tx0 + 14.0
    pitch = (tx1 - 3.0 - ax0) / len(tissues)
    yz = ty1 - 17.4  # zero
    unit = 13.4  # points per unit of correlation
    for v, lab in ((0.0, "0"), (-1.0, "−1")):
        ax.plot([ax0, tx1 - 3.0], [yz + v * unit] * 2, color=RULE if v else INK, lw=LW_HAIR, zorder=2.5,
                ls="-" if not v else (0, (1.2, 1.4)), clip_on=False)
        T(ax, ax0 - 2.2, yz + v * unit, lab, size=TINY, ha="right", va="center_baseline", color=SUB)
    T(ax, tx0 + 3.0, yz - 0.5 * unit, "ρ", size=SMALL, ha="left", va="center_baseline", color=SUB)
    for i, t in enumerate(tissues):
        cx = ax0 + (i + 0.5) * pitch
        words = t["tissue"].capitalize().split()
        for j, word in enumerate(words):
            T(ax, cx, ty1 - 11.8 + (len(words) - 1 - j) * 5.6, word, size=TINY, ha="center")
        for dx, key, col, mk in ((-3.0, "rho_identity_dnds", SEQUENCE, "s"), (3.0, "rho_delta_alpha_identity", DELTA_ALPHA, "o")):
            v = t[key]
            ax.plot([cx + dx, cx + dx], [yz, yz + v * unit], color=col, lw=1.0, zorder=3, solid_capstyle="butt", clip_on=False)
            ax.scatter([cx + dx], [yz + v * unit], s=7.5 if mk == "o" else 6.5, marker=mk, color=col, linewidths=0, zorder=4,
                       clip_on=False)
    yl = ty0 + 3.2
    x = tx0 + 3.0
    T(ax, x, yl, "Spearman ρ with protein identity:", size=TINY, color=SUB)
    x += width(ax, "Spearman ρ with protein identity:", TINY) + 6.0
    for lab, col, mk in (("$\\Delta\\alpha$", DELTA_ALPHA, "o"), ("dN/dS", SEQUENCE, "s")):
        ax.scatter([x], [yl + 1.8], s=7.5 if mk == "o" else 6.5, marker=mk, color=col, linewidths=0, zorder=4, clip_on=False)
        T(ax, x + 3.6, yl, lab, size=TINY, color=SUB)
        x += 3.6 + width(ax, lab, TINY) + 7.0
    # ---- the other uses
    rows = [
        ("genes", "Genes, programs, cell types", "above or below the protein prior"),
        ("tests", "Independent tests", "held-out species, atlases, benchmark"),
        ("disease", "Disease genes, drug targets", "placed in healthy tissue (exploratory)"),
        ("atlas", "Protein-space atlas", "ESM-2 neighborhoods; residue attribution (exploratory)"),
        ("audit", "Ortholog audits", "mapping and annotation gaps"),
    ]
    y = ty0 - 4.0
    pitch = (y - 4.2) / len(rows)
    for key, name, desc in rows:
        icon(ax, key, PAD + 0.5, y - 11.9)
        T(ax, PAD + 19.5, y - 5.8, name, weight="bold")
        T(ax, PAD + 19.5, y - 11.6, desc, size=TINY, color=SUB)
        y -= pitch


def icon(ax, key, x, y):
    """Small pictogram in a box 15 wide and 10 high with its lower left at (x, y)."""
    rng = np.random.default_rng(7)
    if key == "genes":  # bars above and below the prior
        ax.plot([x + 7.5, x + 7.5], [y - 0.2, y + 10.2], color=INK, lw=LW_HAIR, zorder=3, clip_on=False)
        for i, v in enumerate([6.2, 3.8, -3.0, -6.6]):
            ax.add_patch(Rectangle((x + 7.5, y + 7.9 - i * 2.55), v, 1.8, fc=ABOVE if v > 0 else BELOW, ec="none",
                                   zorder=3.2, clip_on=False))
    elif key == "tests":  # estimates with intervals against a reference line
        ax.plot([x + 3.5, x + 3.5], [y - 0.2, y + 10.2], color=RULE, lw=0.6, zorder=3, clip_on=False)
        for i, (c, e) in enumerate([(9.4, 3.4), (7.4, 2.4), (10.8, 3.0)]):
            yy = y + 8.4 - i * 3.4
            ax.plot([x + c - e, x + c + e], [yy, yy], color=INK, lw=LW, zorder=4, clip_on=False)
            ax.scatter([x + c], [yy], s=4.6, color=INK, linewidths=0, zorder=5, clip_on=False)
    elif key == "disease":  # a capsule
        kw = dict(boxstyle="round,pad=1.4,rounding_size=3.1", lw=0.6, clip_on=False)
        body = FancyBboxPatch((x + 3.6, y + 3.3), 7.8, 3.4, fc="white", ec="none", zorder=3, **kw)
        ax.add_patch(body)
        half = Rectangle((x + 2.0, y + 1.5), 5.5, 7.0, fc=tint(SLATE, 0.45), ec="none", zorder=3.1)
        ax.add_patch(half)
        half.set_clip_path(body)
        ax.add_patch(FancyBboxPatch((x + 3.6, y + 3.3), 7.8, 3.4, fc="none", ec=SLATE, zorder=3.2, **kw))
        ax.plot([x + 7.5, x + 7.5], [y + 1.9, y + 8.1], color=SLATE, lw=0.6, zorder=3.3, clip_on=False)
    elif key == "atlas":  # neighborhoods of proteins, each with its sign
        for (cx, cy, col, k) in [(3.6, 7.3, ABOVE, 7), (10.8, 7.8, BELOW, 6), (7.4, 2.6, NEUTRAL, 6), (12.8, 2.8, ABOVE, 3)]:
            px = x + cx + rng.normal(0, 1.3, k)
            py = y + cy + rng.normal(0, 1.1, k)
            ax.scatter(px, py, s=1.7, color=col, linewidths=0, zorder=4, clip_on=False)
    elif key == "audit":  # a table with one flagged row
        for i in range(4):
            ax.add_patch(Rectangle((x + 1.5, y + 7.9 - i * 2.6), 12, 1.9, fc=tint("#E69F00", 0.85) if i == 2 else "#D5D8DD",
                                   ec="none", zorder=3, clip_on=False))


# -------------------------------------------------------------------------------------------- assembly
def _rows(path):
    with path.open() as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def load_assets():
    shapes = json.loads((ASSETS / "silhouettes.json").read_text())
    rows = _rows(ASSETS / "liver_human_monkey_delta_alpha.csv")
    hist = (np.array([float(r["bin_left"]) for r in rows]), np.array([float(r["bin_right"]) for r in rows]),
            np.array([float(r["n_genes"]) for r in rows]))
    rows = _rows(ASSETS / "liver_human_monkey_umap.csv")
    cells_ = {"x": np.array([float(r["umap1"]) for r in rows]), "y": np.array([float(r["umap2"]) for r in rows]),
              "species": np.array([r["species"] for r in rows]), "cell_type": np.array([r["cell_type"] for r in rows])}
    tissues = [{k: (v if k in ("tissue", "model") else float(v)) for k, v in r.items()} for r in _rows(ASSETS / "eight_tissues.csv")]
    return shapes, hist, cells_, tissues


def build(only: str | None = None):
    """The whole figure with its panel letters, or one panel file alone (no letter) on a page of its own size."""
    setup()
    shapes, hist, cells_, tissues = load_assets()
    places = FILES if only is None else {only: (0.0, 0.0, *FILES[only][2:])}
    fw, fh = (W, H) if only is None else FILES[only][2:]
    fig = plt.figure(figsize=(fw / 72, fh / 72), dpi=72)
    fig.patch.set_facecolor("white")
    axes = {}
    for k, (x, y, w, h) in places.items():
        ax = fig.add_axes([x / fw, y / fh, w / fw, h / fh])
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.axis("off")
        axes[k] = ax
    fig.canvas.draw()
    draw = {"problem": lambda ax, w, h: panel_problem(ax, w, h, shapes),
            "inputs_model": lambda ax, w, h: panel_inputs_model(ax, w, h, shapes),
            "outputs": lambda ax, w, h: panel_outputs(ax, w, h, hist, cells_),
            "applications": lambda ax, w, h: panel_applications(ax, w, h, tissues)}
    for k, ax in axes.items():
        draw[k](ax, *places[k][2:])
    if only is None:
        for k, (x, y, _, _) in FILES.items():
            for letter, lx in LETTER_X[k].items():
                fig.text((x + lx) / W, (y - BAND + 2.2) / H, f"({letter})", fontsize=LETTER, fontweight="bold", ha="center",
                         va="baseline", color=INK)
    return fig, axes


def check(fig, axes) -> int:
    """Report text that leaves its card or overlaps other text; returns the number of problems."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    smallest = min(t.get_fontsize() for ax in axes.values() for t in ax.texts)
    print(f"smallest font size {smallest} pt at 486 pt width")
    for k, ax in axes.items():
        x, y, w, h = FILES[k]
        m = FRAME + 2.6
        frames = [Bbox.from_bounds(x + m, y + m, w - 2 * m, h - 2 * m)]
        if k == "inputs_model":
            frames = [Bbox.from_extents(x + m, y + m, x + SPLIT[0] - 2.6, y + h - m),
                      Bbox.from_extents(x + SPLIT[1] + 2.6, y + m, x + w - m, y + h - m)]
        boxes = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts if t.get_text().strip()]
        for s, b in boxes:
            if not any(f.x0 <= b.x0 and b.x1 <= f.x1 and f.y0 <= b.y0 and b.y1 <= f.y1 for f in frames):
                bad += 1
                print(f"  ({k}) outside its card: {s!r}  x {b.x0 - x:.1f}-{b.x1 - x:.1f}  y {b.y0 - y:.1f}-{b.y1 - y:.1f}")
        for i, (s1, b1) in enumerate(boxes):
            for s2, b2 in boxes[i + 1:]:
                if b1.x0 < b2.x1 - 0.6 and b2.x0 < b1.x1 - 0.6 and b1.y0 < b2.y1 - 0.9 and b2.y0 < b1.y1 - 0.9:
                    bad += 1
                    print(f"  ({k}) text overlap: {s1!r} / {s2!r}")
    print("layout check:", "ok" if not bad else f"{bad} problems")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=PANEL_DIR, help="folder for the panel files")
    ap.add_argument("--whole", type=Path, default=None, help="folder for the whole figure with letters (overview_whole.pdf)")
    ap.add_argument("--png", action="store_true", help="with --whole: also a 200 dpi preview")
    ap.add_argument("--check", action="store_true", help="print the layout report")
    args = ap.parse_args()
    meta = {"Title": "Chimera overview", "Author": None, "Creator": None, "Producer": None, "CreationDate": None,
            "ModDate": None}
    fig, axes = build()
    bad = check(fig, axes) if args.check else 0
    if args.whole is not None:
        args.whole.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.whole / "overview_whole.pdf", metadata=meta)
        if args.png:
            fig.savefig(args.whole / "overview_whole.png", dpi=200)
    plt.close(fig)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        fig, _ = build(only=name)
        fig.savefig(args.out / f"{name}.pdf", metadata=meta)
        plt.close(fig)
    print("written", ", ".join(f"{n}.pdf" for n in FILES), "in", args.out, f"(figure {W:.0f} x {H:.0f} pt)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
