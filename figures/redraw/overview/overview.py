"""Figure 1, the overview of Chimera: (a) problem, (b) inputs, (c) model, (d) outputs, (e) applications.

The whole figure is drawn here as vector shapes with embedded text (DejaVu Sans, TrueType); nothing is
pasted from another figure. One unit of every panel is one PostScript point at the size of the figure on
the page (486 pt wide, the text width of the manuscript), so the font sizes below are the printed sizes;
the smallest is 6.2 pt, which stays above 5.5 pt if the figure is set at 0.9 of the text width.

    python figures/redraw/overview/overview.py            # the five panels, one file each, without letters, into
                                                                 # figures/panels/main_overview/ (problem, inputs, model,
                                                                 # outputs, applications); the figure is composed from
                                                                 # them like every other one (key main_overview)
    python figures/redraw/overview/overview.py --check    # layout report: text outside a panel, overlapping text
    python figures/redraw/overview/overview.py --whole DIR --png   # also the whole figure with letters, as one
                                                                 # file, for a look without the compose tool

Inputs, all in ``assets/`` next to this file (made once by ``build_assets.py``):
    silhouettes.json                     outlines of the four species
    liver_human_monkey_delta_alpha.csv   panel (d) histogram: delta alpha of the liver Human+Monkey
                                         reference model, 8,430 genes, in the 80 bins of the paper's axis
    liver_human_monkey_umap.csv          panel (d) embedding: 1,500 cells of the same model
Everything else in the figure is a schematic and uses no data: the two genes of panel (a), and in panel (d)
the five example genes with their count evidence, gate and delta alpha (the values printed under the dials
are those of the drawn dials). The output is the same file on every run.

Colors follow the other figures: above the protein prior vermilion, below it blue; the shared decoder
takes the color of "above" (a gene the shared decoder explains moves above its prior) and the species-aware
residual decoder the color of "below"; species and liver cell types as in the other figures.
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
from matplotlib.patches import Arc, Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
PANEL_DIR = Path(os.environ.get("CHIMERA_FIGURES_OUT", HERE.parents[1])) / "panels" / "main_overview"
PANEL_NAME = {"a": "problem", "b": "inputs", "c": "model", "d": "outputs", "e": "applications"}

# ---------------------------------------------------------------------------------------------- style
INK, MUTED, RULE = "#222222", "#777777", "#BBBBBB"
ABOVE, BELOW = "#D55E00", "#0072B2"  # above / below the protein prior
NEUTRAL = "#BDBDBD"
SPECIES = {"human": "#222222", "macaque": "#0072B2", "mouse": "#D55E00", "zebrafish": "#009E73"}
CELLTYPE = {"hepatocytes": "#E69F00", "endothelial": "#56B4E9", "fibroblasts": "#009E73",
            "macrophages": "#CC79A7", "other": "#BBBBBB"}
CELLTYPE_OF = {"Hepatocytes": "hepatocytes", "Endothelial": "endothelial", "fibroblasts": "fibroblasts",
               "Kupffer cells": "macrophages", "Macrophages": "macrophages"}  # the rest: other

TITLE, BODY, SMALL, LETTER = 7.6, 6.6, 6.2, 7.0  # font sizes, pt on the page
LH = 7.6  # line pitch of stacked text
FRAME = 1.2  # inset of a panel's frame from its edge, so no ink touches the edge of a panel file

W = 486.0  # figure width, pt: the text width of the manuscript
ROWS = [  # (height, {panel: width}); widths of a row and its gaps add up to W
    (160.0, {"a": 128.0, "b": 92.0, "c": 252.0}),
    (166.0, {"d": 303.0, "e": 176.0}),
]
BAND, VGAP, HGAP = 9.5, 6.0, 7.0  # letter band under a row, gap between rows, gap between panels (as the compose tool)
H = sum(h for h, _ in ROWS) + len(ROWS) * BAND + (len(ROWS) - 1) * VGAP
PANELS = {}  # name: (x, y, w, h), y from the bottom of the figure
_y = H
for _h, _row in ROWS:
    assert abs(sum(_row.values()) + HGAP * (len(_row) - 1) - W) < 1e-9, _row
    _y -= _h
    _x = 0.0
    for _k, _w in _row.items():
        PANELS[_k] = (_x, _y, _w, _h)
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


def lines(ax, x, y, rows, size=BODY, ha="left", color=INK, pitch=LH, **kw):
    """Stacked lines of text; y is the baseline of the first."""
    return [T(ax, x, y - i * pitch, s, size=size, ha=ha, color=color, **kw) for i, s in enumerate(rows)]


def width(ax, s, size=BODY, weight="normal") -> float:
    """Width of a text in pt (figure dpi is 72, so one pixel is one point)."""
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight)
    w = t.get_window_extent(ax.figure.canvas.get_renderer()).width
    t.remove()
    return w


def heading(ax, h, title, sub=""):
    """Panel title, bold, with a subtitle in regular weight on the same line."""
    T(ax, 6, h - 10.5, title, size=TITLE, weight="bold")
    if sub:
        T(ax, 6 + width(ax, title, TITLE, "bold") + 5, h - 10.5, sub, size=BODY, color=INK)


def box(ax, x, y, w, h, fc="white", ec=INK, lw=0.7, r=2.0, ls="-", z=2):
    p = FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r},rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, clip_on=False, joinstyle="round")
    ax.add_patch(p)
    return p


def poly(ax, pts, fc, ec, lw=0.7, z=2):
    p = Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=z, joinstyle="round", clip_on=False)
    ax.add_patch(p)
    return p


def node(ax, x, y, r=5.2, s="", fc="white", ec=INK, size=BODY, lw=0.7):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=4, clip_on=False))
    if s:
        T(ax, x, y, s, size=size, ha="center", va="center_baseline")


def arrow(ax, pts, color=INK, lw=0.8, dashed=False, head=True, hl=3.6, hw=2.9, z=3):
    """Polyline through pts with a filled head at the end."""
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
    xy = np.array(pts)
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, ls=(0, (2.4, 1.7)) if dashed else "-", zorder=z,
            solid_capstyle="butt", dash_capstyle="butt", solid_joinstyle="round", clip_on=False)


def silhouette(ax, shapes, name, x, y, height, color):
    """Species outline with its lower left at (x, y); returns its width."""
    verts, codes = [], []
    for ring in shapes[name]["rings"]:
        r = np.asarray(ring) * height + [x, y]
        verts += r.tolist()
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(r) - 2) + [MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", zorder=3, clip_on=False))
    return shapes[name]["aspect"] * height


def strips(ax, x, y, mismatch, n=10, colors=("human", "mouse"), cell=4.3, pitch=5.0, h=3.4, gap=1.5):
    """Two aligned protein sequences, one per species; positions that differ are dark in the second."""
    for row, sp in enumerate(colors):
        yy = y + (1 - row) * (h + gap)
        ax.add_patch(Rectangle((x - 3.0, yy), 1.6, h, fc=SPECIES[sp], ec="none", zorder=3, clip_on=False))
        for i in range(n):
            dark = row == 1 and i in mismatch
            ax.add_patch(Rectangle((x + i * pitch, yy), cell, h, fc="#4D4D4D" if dark else "#D9D9D9", ec="none",
                                   zorder=3, clip_on=False))
    return n * pitch - (pitch - cell)


def profile(ax, x, y, w, h, series):
    """Expression of one gene across cell states, one line per species; series: {species: values in 0..1}."""
    ax.plot([x, x + w], [y, y], color=INK, lw=0.6, zorder=3, solid_capstyle="butt", clip_on=False)
    for sp, v in series.items():
        xs = x + (np.arange(len(v)) + 0.5) * w / len(v)
        ys = y + 1.5 + np.asarray(v) * (h - 3.5)
        ax.plot(xs, ys, color=SPECIES[sp], lw=1.0, zorder=4, solid_joinstyle="round", clip_on=False)
        ax.scatter(xs, ys, s=3.6, color=SPECIES[sp], linewidths=0, zorder=5, clip_on=False)
    for xs in x + (np.arange(len(v)) + 0.5) * w / len(v):
        ax.plot([xs, xs], [y, y - 1.6], color=INK, lw=0.5, zorder=3, clip_on=False)


def dial(ax, cx, cy, r, prior, alpha, imputed=False):
    """A half-circle scale from 0 (left) to 1 (right): tick at the prior, needle at alpha, arc between them."""
    ang = lambda v: 180.0 - 180.0 * v  # noqa: E731
    pt = lambda v, rr: (cx + rr * np.cos(np.radians(ang(v))), cy + rr * np.sin(np.radians(ang(v))))  # noqa: E731
    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=0, theta2=180, color="#D5D5D5", lw=2.4, zorder=3,
                     capstyle="butt", clip_on=False))
    col = MUTED if imputed else (ABOVE if alpha > prior else BELOW)
    if abs(alpha - prior) > 1e-6:
        lo, hi = sorted([ang(prior), ang(alpha)])
        ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=lo, theta2=hi, color=tint(col, 0.55) if imputed else col,
                         lw=2.4, zorder=3.2, capstyle="butt", clip_on=False))
    ax.plot(*zip(pt(prior, r - 2.6), pt(prior, r + 3.0)), color=INK, lw=0.9, zorder=3.6, solid_capstyle="butt",
            clip_on=False)  # the prior
    ax.plot(*zip((cx, cy), pt(alpha, r - 1.6)), color=MUTED if imputed else INK, lw=1.0,
            ls=(0, (1.6, 1.2)) if imputed else "-", zorder=3.8, solid_capstyle="round", clip_on=False)  # alpha
    ax.add_patch(Circle((cx, cy), 1.25, fc=MUTED if imputed else INK, ec="none", zorder=4, clip_on=False))
    ax.plot([cx - r - 1.6, cx + r + 1.6], [cy, cy], color=RULE, lw=0.5, zorder=2.5, clip_on=False)


# ---------------------------------------------------------------------------------------------- panels
def panel_a(ax, w, h, shapes):
    heading(ax, h, "Problem", "two measurements")
    cards = [  # x of the card's center, header, differing positions, profiles, verdict
        (34.5, ["conserved", "protein"], [6], {"human": [0.18, 0.95, 0.45, 0.2, 0.12],
                                              "mouse": [0.15, 0.22, 0.3, 0.62, 0.2]}, "used differently"),
        (95.0, ["less conserved", "protein"], [1, 4, 7], {"human": [0.15, 0.3, 0.92, 0.5, 0.15],
                                                               "mouse": [0.22, 0.36, 0.8, 0.42, 0.1]}, "used alike"),
    ]
    for cx, head, mism, series, verdict in cards:
        lines(ax, cx, 134.0, head, ha="center")
        sw = 10 * 5.0 - 0.7
        strips(ax, cx - sw / 2 + 1.5, 111.5, mism)
        profile(ax, cx - 23, 64.0, 46, 36, series)
        T(ax, cx, 53.0, "cell states", size=SMALL, ha="center", color=MUTED)
        T(ax, cx, 42.0, verdict, size=SMALL, ha="center", weight="bold")
    T(ax, 7.4, 82.0, "expression", size=SMALL, ha="center", va="center", color=MUTED, rot=90)
    # species key
    x = 31.0
    for sp in ("human", "mouse"):
        ax.plot([x, x + 7], [33.2, 33.2], color=SPECIES[sp], lw=1.0, zorder=4, clip_on=False)
        ax.scatter([x + 3.5], [33.2], s=3.6, color=SPECIES[sp], linewidths=0, zorder=5, clip_on=False)
        T(ax, x + 9.5, 31.0, sp, size=SMALL)
        x += 9.5 + width(ax, sp, SMALL) + 9
    ax.plot([8, w - 8], [25.0, 25.0], color=RULE, lw=0.5, zorder=2, clip_on=False)
    lines(ax, w / 2, 15.6, ["Deployment is per gene;", "sequence does not tell which."], ha="center", pitch=7.6)


def panel_b(ax, w, h, shapes):
    heading(ax, h, "Inputs")
    # species and their count matrices
    heights = {"human": 20.0, "macaque": 15.0, "mouse": 12.0, "zebrafish": 8.6}
    widths = {k: shapes[k]["aspect"] * v for k, v in heights.items()}
    gap = (w - 14 - sum(widths.values())) / 3
    x = 7.0
    rng = np.random.default_rng(3)
    for sp in heights:
        silhouette(ax, shapes, sp, x, 122.0, heights[sp], SPECIES[sp])
        cx = x + widths[sp] / 2
        for i in range(3):
            for j in range(3):
                ax.add_patch(Rectangle((cx - 6.0 + i * 4.1, 105.5 + j * 4.1), 3.5, 3.5,
                                       fc=tint(SPECIES[sp], float(rng.choice([0.14, 0.3, 0.5, 0.75]))), ec="none",
                                       zorder=3, clip_on=False))
        x += widths[sp] + gap
    lines(ax, w / 2, 96.0, ["scRNA-seq counts $x$,", "several species,", "one-to-one orthologs"], ha="center")
    ax.plot([8, w - 8], [74.8, 74.8], color=RULE, lw=0.5, zorder=2, clip_on=False)
    # protein prior
    sw = 10 * 5.0 - 0.7
    strips(ax, w / 2 - sw / 2 + 1.5, 61.5, [2, 7])
    lines(ax, w / 2, 52.0, ["protein prior $\\pi_g$:", "identity of the orthologs"], ha="center")
    ax.plot([8, w - 8], [38.4, 38.4], color=RULE, lw=0.5, zorder=2, clip_on=False)
    # gene features
    v = rng.choice([0.12, 0.25, 0.4, 0.6, 0.85], size=13)
    for i, f in enumerate(v):
        ax.add_patch(Rectangle((w / 2 - 13 * 5.0 / 2 + i * 5.0 + 0.35, 28.8), 4.3, 4.6, fc=tint("#5B3E80", float(f)),
                               ec="none", zorder=3, clip_on=False))
    lines(ax, w / 2, 19.6, ["gene features $f_g$: ESM-2,", "identity, expression"], ha="center")


def panel_c(ax, w, h):
    heading(ax, h, "Model", "sequence-anchored $\\alpha$-gating")
    sh, rs = ABOVE, BELOW
    # ---- the gate (top lane)
    yf, yp = 135.5, 124.5
    T(ax, 6, yf - 2.2, "features $f_g$")
    arrow(ax, [(6 + width(ax, "features $f_g$") + 2.5, yf), (62, yf)])
    box(ax, 62, yf - 6.2, 25, 12.4, fc="#EEEEEE", ec=MUTED)
    T(ax, 74.5, yf, "GIN", ha="center", va="center_baseline")
    gx0, gx1 = 113.0, 207.0
    arrow(ax, [(87, yf), (gx0, yf)])
    T(ax, 100, yf + 2.6, "$c_g$", ha="center")
    T(ax, 6, yp - 2.2, "prior $\\pi_g$")
    arrow(ax, [(6 + width(ax, "prior $\\pi_g$") + 2.5, yp), (gx0, yp)])
    box(ax, gx0, 119.0, gx1 - gx0, 22.0, fc="white", ec=INK, lw=0.9, r=3.0)
    T(ax, (gx0 + gx1) / 2, 130.0, "$\\alpha_g=\\sigma\\mathrm{(logit}\\,\\pi_g+c_g\\mathrm{)}$", size=7.0, ha="center",
      va="center_baseline")
    # ---- encoder and latent
    ym = 80.0
    T(ax, 15.0, ym + 1.5, "$x$", size=7.2, ha="center")
    T(ax, 15.0, ym - 7.5, "counts", size=SMALL, ha="center", color=MUTED)
    arrow(ax, [(20.5, ym + 3.4), (28, ym + 3.4)])
    ym += 3.4
    poly(ax, [(28, ym - 12), (28, ym + 12), (62, ym + 6.5), (62, ym - 6.5)], fc="#EEEEEE", ec=MUTED)
    T(ax, 45, ym, "Encoder", ha="center", va="center_baseline")
    arrow(ax, [(62, ym), (70.6, ym)])
    node(ax, 76, ym, 5.4, "$z$", fc="#E4E4E4", size=7.0)
    lines(ax, 58, 113.4, ["cell state $z$,", "species-mixed", "(GRL + MMD)"], size=SMALL, ha="center", pitch=7.2)
    ax.plot([71.5, 74.6], [96.4, 89.8], color=MUTED, lw=0.5, zorder=3, clip_on=False)
    # ---- decoders
    x0, x1 = 92.0, 158.0
    yr, ys = ym + 16.6, ym - 21.4  # residual above, shared below
    arrow(ax, [(86.5, ym), (86.5, yr), (x0, yr)], color=rs, lw=1.0)
    arrow(ax, [(86.5, ym), (86.5, ys), (x0, ys)], color=sh, lw=1.0)
    ax.plot([81.4, 87.0], [ym, ym], color=INK, lw=0.8, zorder=3.3, solid_capstyle="butt", clip_on=False)
    poly(ax, [(x0, yr - 8), (x0, yr + 8), (x1, yr + 12), (x1, yr - 12)], fc=tint(rs, 0.2), ec=rs)
    lines(ax, (x0 + x1) / 2 + 1, yr + 1.6, ["Residual decoder", "species-aware"], ha="center", pitch=7.4)
    poly(ax, [(x0, ys - 8), (x0, ys + 8), (x1, ys + 12), (x1, ys - 12)], fc=tint(sh, 0.2), ec=sh)
    lines(ax, (x0 + x1) / 2 + 1, ys + 1.6, ["Shared decoder", "species-blind"], ha="center", pitch=7.4)
    T(ax, x0 + 4, yr + 13.2, "+ species, batch", size=SMALL, color=rs)
    # ---- gate on the residual, sum, likelihood
    xg, xs_ = 170.0, 188.0
    arrow(ax, [(x1, yr), (xg - 5.2, yr)], color=rs, lw=1.0)
    node(ax, xg, yr, 5.2, "×", size=7.5)
    arrow(ax, [(xg, 119.0), (xg, yr + 5.2)], color=INK, lw=0.9)
    T(ax, xg + 3.2, yr + 8.6, "$\\mathrm{1}-\\alpha_g$")
    yo = ym - 2.4
    arrow(ax, [(xg + 5.2, yr), (xs_, yr), (xs_, yo + 5.2)], color=rs, lw=1.0)
    arrow(ax, [(x1, ys), (xs_, ys), (xs_, yo - 5.2)], color=sh, lw=1.0)
    node(ax, xs_, yo, 5.2, "+", size=7.5)
    T(ax, xs_ - 13.0, yo - 2.2, "zero-sum batch offset", size=SMALL, ha="right")
    arrow(ax, [(xs_ - 11.6, yo), (xs_ - 5.2, yo)], hl=3.0, hw=2.5)
    arrow(ax, [(xs_ + 5.2, yo), (xs_ + 14.5, yo)])
    T(ax, xs_ + 16.5, yo - 2.3, "$\\mathrm{NB(}x\\,|\\,\\mu,\\theta\\mathrm{)}$")
    # ---- count evidence and its way back to the gate
    bx0, bx1, by0, by1 = 136.0, 238.0, 9.0, 41.0
    box(ax, bx0, by0, bx1 - bx0, by1 - by0, fc="white", ec=INK, lw=0.9, r=3.0)
    lines(ax, (bx0 + bx1) / 2, 31.4, ["count evidence $S_g$:"], ha="center", weight="bold")
    lines(ax, (bx0 + bx1) / 2, 23.4, ["shared decoder alone", "against a per-species average"], size=SMALL,
          ha="center", pitch=7.2)
    arrow(ax, [(xg + 5, ys), (xg + 5, by1)], color=sh, lw=0.9, dashed=True)
    xr = w - 6.5
    arrow(ax, [(bx1, 25.0), (xr, 25.0), (xr, 130.0), (gx1, 130.0)], color=INK, lw=0.9, dashed=True)
    T(ax, (gx1 + xr) / 2 + 1, 133.4, "trains $c_g$", ha="center")
    # ---- the two gates
    lines(ax, 6, 36.6, ["protein-feature gate (default):", "$c_g=\\mathrm{GIN(}f_g\\mathrm{)}$, every gene"], size=SMALL,
          pitch=7.4)
    lines(ax, 6, 18.4, ["per-gene gate (companion):", "$c_g$ free, one offset per gene"], size=SMALL, pitch=7.4)


def panel_d(ax, w, h, hist, cells):
    heading(ax, h, "Outputs", "per cell and per gene, from one fit")
    yh = h - 21.5  # baseline of the two block headings
    # ---- per cell: the embedding (real cells of one model)
    T(ax, 6, yh, "per cell: embedding $z$")
    rng = np.random.default_rng(1)
    order = rng.permutation(len(cells["x"]))
    xs, ys = cells["x"][order], cells["y"][order]
    span = max(np.ptp(xs), np.ptp(ys))
    side = 42.0

    def umap(x0, y0, colors):
        px = x0 + side / 2 + (xs - (xs.min() + xs.max()) / 2) / span * (side - 3)
        py = y0 + side / 2 + (ys - (ys.min() + ys.max()) / 2) / span * (side - 3)
        box(ax, x0, y0, side, side, fc="white", ec=RULE, lw=0.5, r=2.0)
        ax.scatter(px, py, s=0.85, c=colors, linewidths=0, zorder=3, clip_on=False)

    sp_name = {"Human": "human", "Monkey": "macaque"}
    y1 = yh - 8.0 - side
    umap(6, y1, [SPECIES[sp_name[sp]] for sp in cells["species"][order]])
    T(ax, 52.5, y1 + side - 7.5, "by species", size=SMALL, color=MUTED)
    for i, (lab, key) in enumerate([("human", "human"), ("monkey", "macaque")]):
        ax.add_patch(Circle((54.6, y1 + side - 14.5 - i * 7.4), 1.5, fc=SPECIES[key], ec="none", zorder=3, clip_on=False))
        T(ax, 58.6, y1 + side - 16.7 - i * 7.4, lab, size=SMALL)
    ct = [CELLTYPE_OF.get(c, "other") for c in cells["cell_type"][order]]
    y2 = y1 - 8.0 - side
    umap(6, y2, [CELLTYPE[c] for c in ct])
    T(ax, 52.5, y2 + side - 6.5, "by cell type", size=SMALL, color=MUTED)
    for i, lab in enumerate(CELLTYPE):
        ax.add_patch(Circle((54.6, y2 + side - 13.5 - i * 7.2), 1.5, fc=CELLTYPE[lab], ec="none", zorder=3, clip_on=False))
        T(ax, 58.6, y2 + side - 15.7 - i * 7.2, lab, size=SMALL)
    lines(ax, 6, y2 - 13.5, ["species aligned: clustering,", "label transfer; the substrate", "of the gene readouts"],
          size=SMALL, pitch=7.2)
    ax.plot([105.5, 105.5], [8, yh + 4], color=RULE, lw=0.5, zorder=2, clip_on=False)

    # ---- per gene: count evidence, gate, delta alpha (five example genes, schematic)
    gx, gr = 113.0, w - 6.0
    T(ax, gx, yh, "per gene: five example orthologs (schematic)")
    cols = [gr - 9.8 - 23.4 * i for i in range(5)][::-1]
    genes = [  # count evidence, prior, alpha (protein-feature gate), delta alpha read gene by gene
        (1.0, 0.50, 0.80, "+0.28"), (0.45, 0.80, 0.93, "+0.09"), (-0.5, 0.86, 0.70, "−0.12"),
        (-1.0, 0.70, 0.36, "−0.31"), (0.0, 0.62, 0.75, "0")]
    y0 = yh - 14.5
    T(ax, gx, y0 - 2.3, "count evidence $S_g$")
    ax.plot([cols[0] - 10, cols[3] + 10], [y0, y0], color=INK, lw=0.5, zorder=3, clip_on=False)
    for cx, (sg, _, _, _) in zip(cols[:4], genes[:4]):
        ax.add_patch(Rectangle((cx - 4.0, y0), 8.0, sg * 8.2, fc=ABOVE if sg > 0 else BELOW, ec="none", zorder=3.2,
                               clip_on=False))
    T(ax, cols[4], y0 - 2.2, "none", size=SMALL, ha="center", color=MUTED)
    yd = y0 - 26.5
    lines(ax, gx, yd + 6.4, ["prior $\\pi_g$: tick", "gate $\\alpha_g$: needle"], pitch=7.8)
    for i, (cx, (_, p, a, _)) in enumerate(zip(cols, genes)):
        dial(ax, cx, yd, 9.0, p, a, imputed=i == 4)
    T(ax, gr, yd - 9.4, "$\\alpha_g$ from 0 (full residual) to 1 (shared decoder only)", size=SMALL, ha="right",
      color=MUTED)
    yt = yd - 21.6
    T(ax, gx, yt, "$\\Delta\\alpha_g=\\alpha_g-\\pi_g$")
    T(ax, gx, yt - 8.0, "protein-feature gate", size=SMALL)
    T(ax, gx, yt - 15.6, "per-gene readout", size=SMALL)
    for i, (cx, (_, p, a, free)) in enumerate(zip(cols, genes)):
        d = a - p
        col = MUTED if i == 4 else ABOVE if d > 0 else BELOW
        T(ax, cx, yt - 8.0, f"{d:+.2f}".replace("-", "−"), size=SMALL, ha="center", color=col,
          style="italic" if i == 4 else "normal")
        T(ax, cx, yt - 15.6, free, size=SMALL, ha="center", color=MUTED if i == 4 else ABOVE if free[0] == "+" else BELOW)
    T(ax, gr, yt - 23.4, "gray: imputed from protein features", size=SMALL, ha="right", color=MUTED)
    # ---- delta alpha over all genes of one model (real)
    hx0, hx1, hy0, hy1 = 177.0, 241.0, 21.5, yt - 33.5
    left, right, n = hist
    top = n.max()
    for a, b, k in zip(left, right, n):
        mid = (a + b) / 2
        col = ABOVE if mid > 0.05 else BELOW if mid < -0.05 else NEUTRAL
        xa = hx0 + (a + 0.4) / 0.8 * (hx1 - hx0)
        xb = hx0 + (b + 0.4) / 0.8 * (hx1 - hx0)
        ax.add_patch(Rectangle((xa, hy0), xb - xa, k / top * (hy1 - hy0), fc=col, ec="none", zorder=3, clip_on=False))
    ax.plot([hx0, hx1], [hy0, hy0], color=INK, lw=0.6, zorder=4, solid_capstyle="butt", clip_on=False)
    for v, lab in [(-0.4, "−0.4"), (-0.2, ""), (0.0, "0"), (0.2, ""), (0.4, "+0.4")]:
        xv = hx0 + (v + 0.4) / 0.8 * (hx1 - hx0)
        ax.plot([xv, xv], [hy0, hy0 - 2.2], color=INK, lw=0.5, zorder=4, clip_on=False)
        if lab:
            T(ax, xv, hy0 - 8.6, lab, size=SMALL, ha="center")
    T(ax, (gx + gr) / 2, 5.0, "$\\Delta\\alpha_g$ of one model: liver, human + monkey (8,430 genes)", size=SMALL,
      ha="center")
    T(ax, hx0 - 3.5, (hy0 + hy1) / 2, "genes", size=SMALL, ha="center", va="center", rot=90, color=MUTED)
    ys_ = hy1 - 5.5
    T(ax, hx0 - 10, ys_, "below the prior", size=SMALL, ha="right", color=BELOW, weight="bold")
    lines(ax, hx0 - 10, ys_ - 7.4, ["pattern or level", "parts between", "species"], size=SMALL, ha="right", pitch=7.2)
    T(ax, hx1 + 4, ys_, "above the prior", size=SMALL, color=ABOVE, weight="bold")
    lines(ax, hx1 + 4, ys_ - 7.4, ["pattern shared", "across species"], size=SMALL, pitch=7.2)


def panel_e(ax, w, h):
    heading(ax, h, "Applications", "of the readouts")
    rows = [
        ("axis", "A separate axis", ["beside protein identity and dN/dS"]),
        ("genes", "Genes, programs, cell types", ["above or below the protein prior"]),
        ("tissues", "Eight tissues, one reading", ["liver, testis, bone marrow, kidney,", "islet, eye, cortex, lung"]),
        ("tests", "Independent tests", ["held-out species, atlases, benchmark"]),
        ("disease", "Disease genes, drug targets", ["placed in healthy tissue (exploratory)"]),
        ("atlas", "Protein-space atlas", ["deployment over ESM-2 neighborhoods;", "residue attribution (exploratory)"]),
        ("audit", "Ortholog audits", ["mapping and annotation gaps"]),
    ]
    y = h - 21.5
    for key, title, desc in rows:
        icon(ax, key, 7.0, y - 9.2)
        T(ax, 31.5, y - 2.4, title, weight="bold")
        lines(ax, 31.5, y - 9.9, desc, size=SMALL, pitch=7.2)
        y -= 18.6 + 7.2 * (len(desc) - 1)


def icon(ax, key, x, y):
    """Small pictogram in a box 19 wide and 14 high with its lower left at (x, y)."""
    rng = np.random.default_rng(7)
    if key == "axis":  # a cloud with no trend
        ax.plot([x + 1, x + 1, x + 18.5], [y + 13.5, y + 0.5, y + 0.5], color=INK, lw=0.6, zorder=3, clip_on=False,
                solid_joinstyle="miter")
        px = x + 3 + rng.random(13) * 14
        py = y + 6.8 + (rng.random(13) - 0.5) * 8.4
        ax.scatter(px, py, s=1.9, color=MUTED, linewidths=0, zorder=4, clip_on=False)
        ax.plot([x + 2.6, x + 17.6], [y + 6.8, y + 6.8], color=INK, lw=0.9, zorder=5, clip_on=False)
    elif key == "genes":  # bars above and below the prior
        ax.plot([x + 9.5, x + 9.5], [y, y + 14], color=INK, lw=0.5, zorder=3, clip_on=False)
        for i, v in enumerate([7.5, 4.6, -3.6, -8.0]):
            ax.add_patch(Rectangle((x + 9.5, y + 11.2 - i * 3.5), v, 2.5, fc=ABOVE if v > 0 else BELOW, ec="none",
                                   zorder=3.2, clip_on=False))
    elif key == "tissues":  # eight tiles
        cols = [tint(INK, f) for f in (0.75, 0.55, 0.4, 0.28, 0.6, 0.35, 0.8, 0.45)]
        for i in range(8):
            ax.add_patch(FancyBboxPatch((x + 1.2 + (i % 4) * 4.5, y + 2.3 + (1 - i // 4) * 5.2), 2.6, 3.3,
                                        boxstyle="round,pad=0.5,rounding_size=0.8", fc=cols[i], ec="none", zorder=3,
                                        clip_on=False))
    elif key == "tests":  # estimates with intervals against a reference line
        ax.plot([x + 6, x + 6], [y, y + 14], color=RULE, lw=0.6, zorder=3, clip_on=False)
        for i, (c, e) in enumerate([(12.0, 4.0), (9.5, 3.0), (13.5, 3.6)]):
            yy = y + 11.4 - i * 4.4
            ax.plot([x + c - e, x + c + e], [yy, yy], color=INK, lw=0.8, zorder=4, clip_on=False)
            ax.scatter([x + c], [yy], s=5.5, color=INK, linewidths=0, zorder=5, clip_on=False)
    elif key == "disease":  # a capsule
        ax.add_patch(FancyBboxPatch((x + 4.3, y + 4.8), 10.4, 4.4, boxstyle="round,pad=1.6,rounding_size=3.8",
                                    fc=tint(INK, 0.3), ec=INK, lw=0.6, zorder=3, clip_on=False))
        ax.add_patch(Rectangle((x + 9.5, y + 3.25), 6.9, 7.5, fc="white", ec="none", zorder=3.1, clip_on=False))
        ax.add_patch(FancyBboxPatch((x + 4.3, y + 4.8), 10.4, 4.4, boxstyle="round,pad=1.6,rounding_size=3.8",
                                    fc="none", ec=INK, lw=0.6, zorder=3.2, clip_on=False))
        ax.plot([x + 9.5, x + 9.5], [y + 3.2, y + 10.8], color=INK, lw=0.6, zorder=3.3, clip_on=False)
    elif key == "atlas":  # neighborhoods of proteins, each with its sign
        for (cx, cy, col, k) in [(5.0, 9.5, ABOVE, 7), (13.6, 10.4, BELOW, 6), (9.6, 3.4, NEUTRAL, 6),
                                 (16.2, 3.6, ABOVE, 3)]:
            px = x + cx + rng.normal(0, 1.7, k)
            py = y + cy + rng.normal(0, 1.5, k)
            ax.scatter(px, py, s=2.0, color=col, linewidths=0, zorder=4, clip_on=False)
    elif key == "audit":  # a table with one flagged row
        for i in range(4):
            ax.add_patch(Rectangle((x + 2, y + 10.6 - i * 3.4), 15, 2.6, fc=tint("#E69F00", 0.85) if i == 2 else
                                   "#DDDDDD", ec="none", zorder=3, clip_on=False))


# -------------------------------------------------------------------------------------------- assembly
def load_assets():
    shapes = json.loads((ASSETS / "silhouettes.json").read_text())
    with (ASSETS / "liver_human_monkey_delta_alpha.csv").open() as f:
        rows = list(csv.DictReader(r for r in f if not r.startswith("#")))
    hist = (np.array([float(r["bin_left"]) for r in rows]), np.array([float(r["bin_right"]) for r in rows]),
            np.array([float(r["n_genes"]) for r in rows]))
    with (ASSETS / "liver_human_monkey_umap.csv").open() as f:
        rows = list(csv.DictReader(r for r in f if not r.startswith("#")))
    cells = {"x": np.array([float(r["umap1"]) for r in rows]), "y": np.array([float(r["umap2"]) for r in rows]),
             "species": np.array([r["species"] for r in rows]), "cell_type": np.array([r["cell_type"] for r in rows])}
    return shapes, hist, cells


def build(only: str | None = None):
    """The whole figure with its panel letters, or one panel alone (no letter) on a page of its own size."""
    setup()
    shapes, hist, cells = load_assets()
    places = PANELS if only is None else {only: (0.0, 0.0, *PANELS[only][2:])}
    fw, fh = (W, H) if only is None else PANELS[only][2:]
    fig = plt.figure(figsize=(fw / 72, fh / 72), dpi=72)
    fig.patch.set_facecolor("white")
    axes = {}
    for k, (x, y, w, h) in places.items():
        ax = fig.add_axes([x / fw, y / fh, w / fw, h / fh])
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.axis("off")
        box(ax, FRAME, FRAME, w - 2 * FRAME, h - 2 * FRAME, fc="white", ec=RULE, lw=0.6, r=4.0, z=1)
        axes[k] = ax
    fig.canvas.draw()
    draw = {"a": lambda ax, w, h: panel_a(ax, w, h, shapes), "b": lambda ax, w, h: panel_b(ax, w, h, shapes),
            "c": panel_c, "d": lambda ax, w, h: panel_d(ax, w, h, hist, cells), "e": panel_e}
    for k, ax in axes.items():
        draw[k](ax, *places[k][2:])
    if only is None:
        for k, (x, y, w, h) in PANELS.items():
            fig.text((x + w / 2) / W, (y - BAND + 2.2) / H, f"({k})", fontsize=LETTER, fontweight="bold", ha="center",
                     va="baseline", color=INK)
    return fig, axes


def check(fig, axes) -> int:
    """Report text that leaves its panel or overlaps other text; returns the number of problems."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    smallest = min(t.get_fontsize() for ax in axes.values() for t in ax.texts)
    print(f"smallest font size {smallest} pt at 486 pt width ({0.9 * smallest:.2f} pt at 0.9 of the text width)")
    for k, ax in axes.items():
        x, y, w, h = PANELS[k]
        frame = Bbox.from_bounds(x + 3.0, y + 3.0, w - 6.0, h - 6.0)
        boxes = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts if t.get_text().strip()]
        for s, b in boxes:
            if b.x0 < frame.x0 or b.x1 > frame.x1 or b.y0 < frame.y0 or b.y1 > frame.y1:
                bad += 1
                print(f"  ({k}) outside the frame: {s!r}  x {b.x0 - x:.1f}-{b.x1 - x:.1f}  y {b.y0 - y:.1f}-{b.y1 - y:.1f}")
        for i, (s1, b1) in enumerate(boxes):
            for s2, b2 in boxes[i + 1:]:
                if b1.x0 < b2.x1 - 0.6 and b2.x0 < b1.x1 - 0.6 and b1.y0 < b2.y1 - 0.9 and b2.y0 < b1.y1 - 0.9:
                    bad += 1
                    print(f"  ({k}) text overlap: {s1!r} / {s2!r}")
    print("layout check:", "ok" if not bad else f"{bad} problems")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=PANEL_DIR, help="folder for the five panel files")
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
    for k, name in PANEL_NAME.items():
        fig, _ = build(only=k)
        fig.savefig(args.out / f"{name}.pdf", metadata=meta)
        plt.close(fig)
    print("written", ", ".join(f"{n}.pdf" for n in PANEL_NAME.values()), "in", args.out, f"(figure {W:.0f} x {H:.0f} pt)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
