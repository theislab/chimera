"""The integration benchmark on eleven cohorts: the seven liver and testis cohorts of ``fig3_integration`` and four
more (bone marrow, kidney, islet, eye Human+Mouse). Same three panels, same scoring and scaling; scaling is per cohort,
so the seven first columns are those of ``fig3_integration``.

Source: ``results/analysis/B2/integration/master_table_extended.csv``. A cohort joins by a row in
``B2_benchmark/integration/rules.py`` and a rerun of its ``score.py``. Integration is the substrate of Chimera's readouts,
not its aim.

The panels are drawn at print size: the panel with the method names and the one without, with a gap of 4 bp, fill the
text width (``MAP_W``), and a line of a heat map is 10.5 bp high, so no text is under 5.5 pt.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m analysis.tasks.T8_4_figures.fig3_integration_ext
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.tasks.T8_4_figures import common as cm
from analysis.tasks.T8_4_figures import fig3_integration as F3

SRC = cm.REPO / "results" / "analysis" / "B2" / "integration" / "master_table_extended.csv"
ADDED = [("bone_marrow", "Bone marrow 3 species"), ("kidney", "Kidney Human+Mouse"), ("islet", "Islet Human+Mouse"), ("vz_hmouse", "Eye Human+Mouse")]
# Geometry in inches. Two maps, the method names (LEFT), three margins and the gap of 4 bp between the panels fill 6.75 in.
LEFT, RIGHT, UNDER, TOP = 1.24, 0.02, 1.42, 0.02      # method names; right margin; cohort names and color bar; above the map
MAP_W = (6.75 - LEFT - 3 * RIGHT - 4 / 72) / 2        # the heat map of eleven cohorts
LINE = 10.5 / 72                                      # one line of a heat map: a method, or the heading of a group


def heat(M: pd.DataFrame, label: str, show_rows: bool, xlabels: list[str], n_orig: int) -> plt.Figure:
    layout, names = [], []
    for tier, ms in F3.TIERS:
        layout.append(None)
        names.append("")
        for m in ms:
            layout.append(m)
            names.append(F3.COL_LABEL.get(m, m))
    A = np.full((len(layout), M.shape[0]), np.nan)
    for i, m in enumerate(layout):
        if m is not None:
            A[i] = M[m].to_numpy()
    left = LEFT if show_rows else RIGHT
    mw, mh = MAP_W, LINE * len(layout)
    w, h = left + mw + RIGHT, UNDER + mh + TOP
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([left / w, UNDER / h, mw / w, mh / h])
    cax = fig.add_axes([(left + 0.05) / w, 0.3 / h, (1.75 - 0.1) / w, 0.06 / h])
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    best = {j: int(np.nanargmax(A[:, j])) if np.isfinite(A[:, j]).any() else -1 for j in range(A.shape[1])}
    for i, m in enumerate(layout):
        if m is None:
            continue
        for j in range(A.shape[1]):
            v = A[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.6 else cm.INK, fontweight="bold" if best[j] == i else "normal")
            if m == "Chimera" and j >= 4:  # labels through conditional MMD everywhere but liver
                ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False, ec=cm.INK, lw=0.6, ls=(0, (1.5, 1))))
    if show_rows:
        k = 0
        for tier, ms in F3.TIERS:
            tier = "no labels on liver; labels elsewhere (dashed)" if ms == ["Chimera"] else tier
            ax.text(-0.45, k + 0.1, tier, ha="left", va="center", fontsize=5.5, style="italic", color=cm.MUTED)
            k += len(ms) + 1
    ax.axvline(3.5, color="white", lw=2.0)
    ax.axvline(n_orig - 0.5, color="white", lw=3.0)  # original seven | added cohorts
    ax.set_xticks(range(M.shape[0]))
    ax.set_xticklabels(xlabels, rotation=90, ha="center", va="top", fontsize=5.5)
    ax.set_yticks([i for i, m in enumerate(layout) if m is not None])
    ax.set_yticklabels([n for n, m in zip(names, layout) if m is not None] if show_rows else [], fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"{label}\n(scaled within cohort, 1 = best)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.4)
    return fig


# ------------------------------------------------------------------------------------------------ main figure: composite strip
STRIP_PANEL = "main_integration/composite_strip"
CHIMERA_COLOR = cm.style.READOUT["delta_alpha"].color  # the color of the model's default readout
OTHER_COLOR = "#8C8C8C"
# The two tools whose composites the text quotes get marks of their own (still gray, still the shape of their use of
# labels): keyword arguments of ``ax.plot`` for each.
NAMED_METHODS = {"SATURN": dict(ms=3.6, mfc="#6E6E6E", mec="#222222", mew=0.45, alpha=1.0, zorder=2.6),
                 "CAMEX": dict(ms=3.5, mfc="white", mec="#555555", mew=0.7, alpha=1.0, zorder=2.5)}
LABEL_USE = [("no cell-type labels", "o"), ("within-species clusters", "^"), ("cell-type labels", "s")]  # the three first groups of fig3_integration.TIERS
TISSUE_OF = {"liv": "Liver", "tes": "Testis"}  # the first three letters of a cohort key of the seven liver and testis cohorts
# Composite score of Chimera on the eleven cohorts, in the order of the rows, as the stored table gives it (two decimals).
CHIMERA_COMPOSITE = [0.65, 0.45, 0.26, 0.60, 0.59, 0.59, 0.64, 0.41, 0.58, 0.69, 0.52]


def composite_strip() -> plt.Figure:
    """Main integration figure (``main_integration/composite_strip``, one column: 234 x 145 bp): the composite
    integration score of ten methods on eleven cohorts, one row per cohort.

    What it draws. x: the composite score, scaled within cohort (0 lowest, 1 highest on the scale of the benchmark).
    Each row is a cohort, grouped by tissue; each mark is a method. The nine other methods are gray; the marker shape
    says how a method uses cell-type labels (none, within-species clusters, cell-type labels), in the groups of the
    supplementary heat maps. Two of the nine have marks of their own and an entry in the key, because the text quotes
    their scores: SATURN (dark gray with an outline) and CAMEX (open). Chimera is drawn in color and joined by a line
    across the cohorts; it uses no cell-type labels on the liver cohorts and labels on the others, and its marker
    shape says so. No rank and no method name is printed beside the marks.

    Reads ``results/analysis/B2/integration/master_table_extended.csv`` and nothing else (110 rows, eleven cohorts by
    ten methods; columns ``dataset``, ``model``, ``total_score``). No value is computed: every mark is a stored score.
    The panel stops unless it draws 110 marks and the eleven scores of Chimera equal ``CHIMERA_COMPOSITE``.

        python -m analysis.tasks.T8_4_figures.fig3_integration_ext --panel main_integration/composite_strip     # in chimera_env, from the repository root
    """
    T = pd.read_csv(SRC)
    rows = [(d, cm.COHORT[c]) for d, c in F3.ROWS] + list(ADDED)
    tissue = [TISSUE_OF[c[:3]] for _, c in F3.ROWS] + [n.split(" ")[0] for _, n in ADDED]
    M = T.pivot(index="dataset", columns="model", values="total_score").reindex(index=[d for d, _ in rows], columns=F3.COLS)
    assert M.shape == (11, 10) and M.notna().all().all(), "master_table_extended.csv: expected eleven cohorts by ten methods"
    assert [round(float(v), 2) for v in M["Chimera"]] == CHIMERA_COMPOSITE, f"composite of Chimera in the table: {[round(float(v), 2) for v in M['Chimera']]}"
    shape = {m: mk for (_, ms), (_, mk) in zip(F3.TIERS[:3], LABEL_USE) for m in ms}
    W, H = 234.0, 145.0
    left, right, bottom, top = 76.0, 6.0, 31.0, 4.0
    fig = plt.figure(figsize=(cm.style.inch(W), cm.style.inch(H)))
    ax = fig.add_axes([left / W, bottom / H, (W - left - right) / W, (H - bottom - top) / H])
    y, ys = 0.0, []
    for i in range(len(rows)):  # a wider step where the tissue changes
        y += 1.0 + (0.45 if i and tissue[i] != tissue[i - 1] else 0.0)
        ys.append(y)
    tr = ax.get_yaxis_transform()
    n_marks = 0
    for (d, name), yy in zip(rows, ys):
        ax.plot([0, 1], [yy, yy], color="#E8E8E8", lw=0.5, zorder=0)
        for m in F3.COLS:
            if m == "Chimera":
                continue
            ax.plot(M.loc[d, m], yy, ls="none", marker=shape[m], **NAMED_METHODS.get(m, dict(ms=3.3, mfc=OTHER_COLOR, mec="white", mew=0.3, alpha=0.9, zorder=2)))
            n_marks += 1
        ax.text(-5.0 / (W - left - right), yy, name, transform=tr, ha="right", va="center", fontsize=5.5, color=cm.INK, clip_on=False)
    ax.plot(M["Chimera"].to_numpy(), ys, color=CHIMERA_COLOR, lw=0.6, alpha=0.7, zorder=3)
    for (d, _), yy, t in zip(rows, ys, tissue):
        ax.plot(M.loc[d, "Chimera"], yy, ls="none", marker="o" if t == "Liver" else "s", ms=4.6 if t == "Liver" else 4.2, mfc=CHIMERA_COLOR, mec="white", mew=0.5, zorder=4)
        n_marks += 1
    assert n_marks == 110, f"{n_marks} marks drawn, expected 110"
    ax.set_ylim(ys[-1] + 0.65, ys[0] - 0.65)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=5.5, length=2, pad=1.5)
    ax.set_xlabel("composite integration score (scaled within cohort)", fontsize=5.8, labelpad=1.5)
    from matplotlib.lines import Line2D

    kw = dict(ls="none", mec="white", mew=0.3)
    assert set(NAMED_METHODS) <= set(shape), "a named method is not one of the nine other methods"
    named = [Line2D([], [], ls="none", marker=shape[m], label=m, **{k: v for k, v in style_.items() if k != "zorder"}) for m, style_ in NAMED_METHODS.items()]
    hs = [Line2D([], [], marker="o", ms=4.4, mfc=CHIMERA_COLOR, color=CHIMERA_COLOR, lw=0.6, mec="white", mew=0.5, label="Chimera"), *named,
          Line2D([], [], marker="o", ms=3.3, mfc=OTHER_COLOR, label="seven other methods", **kw)]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(2.0 / W, 8.0 / H), ncol=4, fontsize=5.2, handletextpad=0.2, columnspacing=0.9, handlelength=1.5, frameon=False, borderpad=0, borderaxespad=0)
    hs = [Line2D([], [], marker=mk, ms=3.3, mfc="white", mec=cm.INK, mew=0.5, ls="none", label=name) for name, mk in LABEL_USE]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(2.0 / W, 1.0 / H), ncol=3, fontsize=5.2, handletextpad=0.1, columnspacing=0.8, handlelength=1.0, frameon=False, borderpad=0, borderaxespad=0)
    return fig


def main(only=None) -> list:
    """Draw the three heat maps of the supplementary figure and the strip of the main figure (``only``: panel names
    ("family/name") to draw just those)."""
    cm.setup()
    T = pd.read_csv(SRC)
    rows = [d for d, _ in F3.ROWS] + [d for d, _ in ADDED if d in set(T.dataset)]
    lab = [cm.COHORT[c] for _, c in F3.ROWS] + [n for d, n in ADDED if d in set(T.dataset)]
    out = []
    for k, (col, label) in enumerate(F3.SPECS):
        name = f"fig3{'abc'[k]}_ext_integration_{col.replace('total_score', 'composite')}"
        if only and cm.style.panel_for(name) not in only:
            continue
        M = T.pivot(index="dataset", columns="model", values=col).reindex(index=rows, columns=F3.COLS)
        fig = heat(M, label, k in (0, 2), lab, len(F3.ROWS))
        out.append(cm.save(fig, name, exact=True))
    if not only or STRIP_PANEL in only:  # at its exact size; the preview goes to figures/_work/panels/
        out.append(cm.style.save_panel(composite_strip(), STRIP_PANEL, exact=True))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs="+", metavar="NAME", help="draw only these panels (family/name, as in figures/panels/)")
    for p in main(ap.parse_args().panel):
        print(p, cm.check_pdf(p))
