"""Y3 (EXPLORATORY): supplementary panel, two worked genes in the counts.

Log2(1 + CPM) per species and cell type or stage (worked_pseudobulk.csv), one line per species, types ordered by the
human value within each panel. Species colors from figures/style.py, one marker per species. One y scale per plot.
Types without enough cells in a species are gaps. Writes the panel figures/panels/supp_worked_genes/worked_genes.pdf, drawn
at its size on the page, from the two cached tables.

Usage: PYTHONPATH=$PWD python -m analysis.tasks.Y3_findings.figures
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

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this figure
MARK = {"Human": "o", "Macaque": "s", "Monkey": "s", "Mouse": "^", "Zebrafish": "D"}  # one marker per species, beside its color
NICE = {"Monkey": "Macaque", "Other_somatic": "other\nsomatic", "fibroblasts": "Fibroblasts", "Hepatocytes": "Hepatocytes", "Cholangio": "Cholangiocytes",
        "Endothelial": "Endothelial\ncells"}
IDENTITY = "protein identity π"
# Sizes in bp: one panel at the text width with two plots (LEFT + testis + BETWEEN + LEFT + liver + 4 = text width). The species
# names stand at the ends of the lines, inside the axes.
PANEL_W, PANEL_H = style.TEXTWIDTH, 170.0
PLOT_W = {"testis": 190.0, "liver": 218.0}
LEFT, BETWEEN, ABOVE_AXES, BELOW_AXES = 27.0, 20.0, 13.0, 22.0


def panel(ax, t, order, species, title, identity):
    for s in species:
        v = t[t.species == s].set_index("type").log2cpm.reindex(order).to_numpy(float)
        ax.plot(range(len(order)), v, color=style.SPECIES[NICE.get(s, s).lower()], marker=MARK[s], ms=4.0, lw=style.LW_DATA, mec="white", mew=0.4,
                label=NICE.get(s, s))
        i = np.where(np.isfinite(v))[0][-1]
        ax.annotate(NICE.get(s, s), (i, v[i]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=style.SIZE_TICK, color=style.INK)
    ax.set_xticks(range(len(order)), [NICE.get(k, k) for k in order])
    ax.set_xlim(-0.3, len(order) - 0.3)
    ax.set_title(title, loc="left", pad=3)
    ax.text(0.985, 0.97, f"{IDENTITY} = {identity:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=style.SIZE_TEXT)
    ax.set_ylabel(r"log$_2$(1 + CPM)")


def draw_panels() -> None:
    """The panel from the two cached tables (worked_pseudobulk.csv, worked_readouts.csv); nothing is computed."""
    style.apply()
    d = pd.read_csv(OUT / "worked_pseudobulk.csv")
    r = pd.read_csv(OUT / "worked_readouts.csv")
    fig = plt.figure(figsize=(style.inch(PANEL_W), style.inch(PANEL_H)))
    x0 = LEFT
    for key, cohort, order, species, note in (
            ("testis", "test_3sp", ["SG", "SC", "rSD", "eSD", "Sertoli", "Other_somatic"], ["Human", "Macaque", "Mouse"], ""),
            ("liver", "liv_4sp", ["Hepatocytes", "Cholangio", "Endothelial", "fibroblasts"], ["Human", "Monkey", "Mouse"], " (the three mammals)")):
        g = r[r.cohort == cohort].iloc[0]
        ax = fig.add_axes([x0 / PANEL_W, BELOW_AXES / PANEL_H, PLOT_W[key] / PANEL_W, (PANEL_H - ABOVE_AXES - BELOW_AXES) / PANEL_H])
        panel(ax, d[d.cohort == cohort], order, species, f"{g.symbol}, {style.cohort_name(cohort, lower=True)}{note}", g.prior_raw)
        x0 += PLOT_W[key] + BETWEEN + LEFT
    style.save_panel(fig, "supp_worked_genes/worked_genes", exact=True)


def main():
    draw_panels()


if __name__ == "__main__":
    main()
