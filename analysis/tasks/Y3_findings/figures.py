"""Y3 (EXPLORATORY): the two worked genes in the counts, panels a and b of the main validation figure
(``figures/panels/main_validation/worked_genes.pdf``, 317 x 120 bp, placed with the letters a and b).

What the panel draws. Two plots, each under a heading that names the finding its gene illustrates:

  left   "Less conserved protein, used alike": ZPBP2 on three-species testis. Log2(1 + CPM) of the pseudobulk per species
         over the testis stages, one line per species; the stage of the peak, shared by the three species, is shaded
         (SG, SC, rSD, eSD: spermatogonia, spermatocytes, round and elongated spermatids).
  right  "Conserved protein, used at a different level": ALAS1 on four-species liver, the three mammals on the four
         cell types they share. The bracket at the hepatocytes is the gap between mouse and human in log2 units.

Under each heading stand the gene and its cohort; in the upper right corner of the plot its reading: Δα against the raw
prior (in the color of "above" or "below" the prior) and the count evidence S_g, as stored. The protein identity of a gene
is not printed: the two genes are measured on different species sets, so the two values are not to be compared; the
caption gives each with its species set. Species colors are those of ``figures/style.py``, with one marker per species;
the species names stand at the ends of the lines. One y scale per plot.

Inputs, both written by ``worked_genes.py`` of this task; nothing is computed here:

  results/analysis/Y3/worked_pseudobulk.csv   columns symbol, species, type, log2cpm, cohort
                                              (rows of cohort ``test_3sp`` for ZPBP2 and ``liv_4sp`` for ALAS1)
  results/analysis/Y3/worked_readouts.csv     columns symbol, cohort, D_raw, S

The printed values are checked against the tables and against the paper: ZPBP2 Δα +0.17, S_g +0.055; ALAS1 Δα −0.24,
S_g −0.11; mouse 4.6 log2 above human in hepatocytes.

    python -m analysis.tasks.Y3_findings.figures          # in chimera_env, from the repository root
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.common import gene_ids  # noqa: E402
from figures import style  # noqa: E402

OUT = gene_ids.REPO_ROOT / "results" / "analysis" / "Y3"

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this panel
MARK = {"Human": "o", "Macaque": "s", "Monkey": "s", "Mouse": "^", "Zebrafish": "D"}  # one marker per species, beside its color
NICE = {"Monkey": "Macaque", "Other_somatic": "other", "fibroblasts": "Fibro-\nblasts", "Hepatocytes": "Hepato-\ncytes", "Cholangio": "Cholangio-\ncytes",
        "Endothelial": "Endo-\nthelial"}  # the testis stages keep their short names (SG, SC, rSD, eSD: the caption spells them out)
S_HEAD, S_NAME, S_TICK, S_NOTE = 6.5, 6.0, 5.5, 5.5  # type in pt at print size: heading and axis title; numbers of the y axis and species names; gene line, readings and the names of the cell types under the x axis (at 6.0 pt two of them would touch); notes
LINE, MARKER = 1.3, 3.5  # pt: the lines of the species and their markers
# Sizes in bp: one panel with two plots of the same size (in the figure it stands beside a third panel in one row); the
# page width is on the 0.24 bp grid of the layout tool.
PANEL_W, PANEL_H = 316.8, 120.0
LEFT, BETWEEN, RIGHT, ABOVE_AXES, BELOW_AXES = 22.0, 9.0, 2.5, 31.0, 17.0
PLOT_W = (PANEL_W - 2 * LEFT - BETWEEN - RIGHT) / 2
# The two genes: cohort, order of the cell types or stages, species, heading, where, what the paper prints
# (Δα, S_g), the digits of S_g, the type that is marked, and the room for the species names at the right
# of the lines (in steps of the x axis).
WORKED = [
    dict(cohort="test_3sp", order=["SG", "SC", "rSD", "eSD", "Sertoli", "Other_somatic"], species=["Human", "Macaque", "Mouse"],
         heading="Less conserved protein,\nused alike", where="testis 3 species", printed=(0.17, 0.055), s_digits=3, mark="SC", room=1.6),
    dict(cohort="liv_4sp", order=["Hepatocytes", "Cholangio", "Endothelial", "fibroblasts"], species=["Human", "Monkey", "Mouse"],
         heading="Conserved protein,\nused at a different level", where="liver 4 species", printed=(-0.24, -0.11), s_digits=2, mark="Hepatocytes", room=1.08),
]
HEPATOCYTE_GAP = 4.6  # log2, mouse above human, as the paper prints it


def signed(v: float, nd: int) -> str:
    return f"{v:+.{nd}f}".replace("-", "−")


def lines(ax, t: pd.DataFrame, order: list, species: list) -> dict:
    """One line per species over the types of ``order``; the species names stand at the ends of the lines. Returns
    species -> its values in that order."""
    out = {}
    for s in species:
        v = t[t.species == s].set_index("type").log2cpm.reindex(order).to_numpy(float)
        col = style.SPECIES[NICE.get(s, s).lower()]
        ax.plot(range(len(order)), v, color=col, marker=MARK[s], ms=MARKER, lw=LINE, mec="white", mew=0.3, zorder=3)
        i = np.where(np.isfinite(v))[0][-1]
        ax.annotate(NICE.get(s, s), (i, v[i]), xytext=(4.5, 0), textcoords="offset points", va="center", fontsize=S_NAME, color=col)
        out[s] = v
    ax.set_xticks(range(len(order)), [NICE.get(k, k) for k in order])
    ax.tick_params(axis="x", labelsize=S_TICK, length=2.5, pad=1.5)
    ax.tick_params(axis="y", labelsize=S_NAME, length=2.5, pad=1.5)
    for lab in ax.get_xticklabels():
        lab.set_linespacing(0.95)
    return out


def draw_panels() -> None:
    """The panel from the two cached tables (worked_pseudobulk.csv, worked_readouts.csv); see the module docstring."""
    style.apply()
    d = pd.read_csv(OUT / "worked_pseudobulk.csv")
    r = pd.read_csv(OUT / "worked_readouts.csv")
    fig = plt.figure(figsize=(style.inch(PANEL_W), style.inch(PANEL_H)))
    x0 = LEFT
    h_ax = PANEL_H - ABOVE_AXES - BELOW_AXES
    for w in WORKED:
        g = r[r.cohort == w["cohort"]].iloc[0]
        delta, s_g = w["printed"]
        assert (round(g.D_raw, 2), round(g.S, w["s_digits"])) == (delta, s_g), \
            f"{g.symbol}: {g.D_raw:.3f}, {g.S:.4f} in worked_readouts.csv, the paper prints {w['printed']}"
        ax = fig.add_axes([x0 / PANEL_W, BELOW_AXES / PANEL_H, PLOT_W / PANEL_W, h_ax / PANEL_H])
        v = lines(ax, d[d.cohort == w["cohort"]], w["order"], w["species"])
        n, k = len(w["order"]), w["order"].index(w["mark"])
        lo, hi = np.nanmin(list(v.values())), np.nanmax(list(v.values()))
        ax.set_ylim(lo - 0.07 * (hi - lo), hi + 0.2 * (hi - lo))
        if w["cohort"] == "test_3sp":  # the stage at which the three species peak
            assert all(int(np.nanargmax(x)) == k for x in v.values()), "the three species do not peak at the marked stage"
            ax.set_xlim(-0.4, n - 1 + w["room"])
            ax.axvspan(k - 0.32, k + 0.32, color=style.NULL_BAND, lw=0, zorder=0)
            ax.text(k, 1.0, "shared peak", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=S_NOTE, color=style.MUTED)
        else:  # the gap between mouse and human in hepatocytes
            gap = v["Mouse"][k] - v["Human"][k]
            assert round(gap, 1) == HEPATOCYTE_GAP, f"hepatocytes, mouse minus human: {gap:.2f} log2 in worked_pseudobulk.csv, the paper prints {HEPATOCYTE_GAP}"
            ax.set_xlim(-0.58, n - 1 + w["room"])
            xb = k - 0.24
            ax.annotate("", (xb, v["Mouse"][k]), (xb, v["Human"][k]), arrowprops=dict(arrowstyle="<->", color=style.MUTED, lw=0.7, shrinkA=0, shrinkB=0, mutation_scale=5))
            ax.text(xb - 0.06, (v["Mouse"][k] + v["Human"][k]) / 2, f"{HEPATOCYTE_GAP} log$_2$", rotation=90, ha="right", va="center", fontsize=S_NOTE, color=style.MUTED)
        ax.set_ylabel(r"log$_2$(1 + CPM)", fontsize=S_HEAD, labelpad=1)
        # above the plot: the finding, then the gene and its cohort; in the free upper right corner of the plot: its reading
        xl = (x0 - LEFT + 2.0) / PANEL_W
        fig.text(xl, (PANEL_H - 1.2) / PANEL_H, w["heading"], ha="left", va="top", fontsize=S_HEAD, fontweight="bold", color=style.INK, linespacing=1.12)
        t = fig.text(xl, (PANEL_H - 22.6) / PANEL_H, g.symbol, ha="left", va="baseline", fontsize=S_TICK, fontweight="bold", color=style.INK)
        fig.canvas.draw()
        fig.text(xl + t.get_window_extent().width / fig.bbox.width, (PANEL_H - 22.6) / PANEL_H, f", {w['where']}", ha="left", va="baseline", fontsize=S_TICK, color=style.INK)
        col = style.ABOVE if delta > 0 else style.BELOW
        for j, (text, c, weight) in enumerate(((f"Δα {signed(delta, 2)}", col, "bold"), ("$S_g$ " + signed(s_g, w["s_digits"]), style.INK, "normal"))):
            ax.text(0.99, 0.985 - j * 6.6 / h_ax, text, transform=ax.transAxes, ha="right", va="top", fontsize=S_TICK, color=c, fontweight=weight)
        x0 += PLOT_W + BETWEEN + LEFT
    style.save_panel(fig, "main_validation/worked_genes", exact=True)


def main():
    draw_panels()


if __name__ == "__main__":
    main()
