"""E6 panels (exploratory): published species differences on the readouts. Two panels of the figure
supp_published_differences (matched, conserved), saved through figures/style.py to figures/panels/; reads
results/analysis/E6/primary.csv only.

Both panels are drawn at the text width with the same axes box and the same AUROC range and are saved at their exact
size, so the figure composes at scale 1 and the 0.5 lines of the two panels line up.

    PYTHONPATH=$PWD python -m analysis.tasks.E6_known_differences.figures [panel ...]     # both panels, or the named ones, from the cached table (draw_panels)
    CHIMERA_FIGURES_OUT=<folder> ... the same command                                      # dry run: panels under <folder>/panels
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analysis.tasks.E6_known_differences import rules as RU  # noqa: E402
from figures import style  # noqa: E402

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this task
WIDTH = style.TEXTWIDTH  # bp; both panels fill the text width
LEFT, RIGHT, PITCH = 258.0, 8.0, 12.0  # bp: column of the row labels, right margin, height of one row
SPREAD = 0.325  # of a row: the marks of a row sit within this distance above and below its center
XLIM = (0.1, 0.8)  # AUROC range of both panels; holds every interval of both
S_LABEL, S_TICK, S_TEXT = style.SIZE_LABEL, style.SIZE_TICK, style.SIZE_TEXT
INK, MUTED = style.INK, style.MUTED
LEVEL = style.Mark("#999999", "v", "detection gap, no model")  # a direct level measure; not in the house table
# column prefix in primary.csv -> name, color and marker; in the order of the house table
SCORES = {"S_can": style.READOUT["count_evidence"], "D_can": style.READOUT["delta_alpha"], "D_free": style.READOUT["delta_alpha_per_gene"],
          "B1": style.READOUT["profile_correlation"], "LVL": LEVEL}
SAME_DATA = "\u2021"  # double dagger after the tissue of a list that was derived from the same data as the cohort
NOTE = f"{SAME_DATA} list derived from the same data as the cohort"

# (assignment, set) -> (tissue, the published list, cohort it is read in); a double dagger marks a list from the cohort's own data
LABEL = {
    ("jor_hs", "human_specific__core"): ("Cortex", "human-specific DE, MTG (Jorstad 2023)", "DLPFC 4 primates"),
    ("jor_hc", "human_vs_chimp_divergent__core"): ("Cortex", "human vs chimpanzee DE, MTG (Jorstad 2023)", "DLPFC H+Chimp"),
    ("cag_am", "ape_vs_macaque_divergent__core"): ("Cortex", "ape vs macaque DE, cingulate (Caglayan 2023)", "DLPFC H+Macaque"),
    ("cag_hs", "human_specific__core"): ("Cortex", "human-specific DE, cingulate (Caglayan 2023)", "DLPFC 4 primates"),
    ("khr_hs", "human_specific__core"): ("Cortex", "human-specific DE, ACC (Khrameeva 2020)", "DLPFC 4 primates"),
    ("hodge", "human_mouse_divergent__core"): ("Cortex", "human-mouse divergent, MTG/VISp (Hodge 2019)", "M1 H+Mouse"),
    ("benner", "human_enriched__core"): ("Islet", "human-enriched in beta cells (Benner 2014)", "Islet H+Mouse"),
    ("benner", "mouse_enriched__core"): ("Islet", "mouse-enriched in beta cells (Benner 2014)", "Islet H+Mouse"),
    ("he2021", "species_divergent"): ("Kidney", "species-specific, glomerulus (He 2021)", "Kidney H+Mouse"),
    ("kim2024", "human_enriched__core"): ("Kidney", "human-enriched, fetal (Kim 2024)", "Kidney H+Mouse"),
    ("kim2024", "mouse_enriched__core"): ("Kidney", "mouse-enriched, fetal (Kim 2024)", "Kidney H+Mouse"),
    ("shami_hm", "species_specific"): ("Testis", "stage-unique, human-mouse (Shami 2020)", "Testis H+Mouse"),
    ("shami_hm", "species_divergent"): ("Testis", "phase-shifted, human-mouse (Shami 2020)", "Testis H+Mouse"),
    ("shami_hq", "species_specific"): ("Testis", "stage-unique, human-macaque (Shami 2020)", "Testis H+Macaque"),
    ("aiz_hep", "nonconserved_zonation"): ("Liver", "non-conserved zonation, hepatocytes (Aizarani 2019)", "Liver H+Mouse"),
    ("ezran_lung_circ", "primate_depleted__core"): ("Lung" + SAME_DATA, "primate-depleted (Tabula Microcebus)", "Lung 3 species"),
    ("ezran_lung_circ", "primate_enriched__core"): ("Lung" + SAME_DATA, "primate-enriched (Tabula Microcebus)", "Lung 3 species"),
    ("trav_circ", "species_divergent__core"): ("Lung" + SAME_DATA, "human vs mouse DE (Travaglini 2020)", "Lung 3 species"),
    ("ezran_bm_circ", "primate_enriched__core"): ("Bone marrow" + SAME_DATA, "primate-enriched (Tabula Microcebus)", "Bone marrow"),
    ("bakken_circ", "human_only_marker__core"): ("Cortex" + SAME_DATA, "human-only subclass markers (Bakken 2021)", "M1 H+Mouse"),
}
# (assignment, set) -> (tissue, the divergent list against the conserved list of the same paper)
CTRL_LABEL = {
    ("hodge", "human_mouse_divergent__core"): ("Cortex", "human-mouse divergent (top 3%) vs shared markers (Hodge 2019)"),
    ("bakken_circ", "human_only_marker__core"): ("Cortex" + SAME_DATA, "human-only vs conserved markers (Bakken 2021)"),
    ("he2021", "species_divergent"): ("Kidney", "species-specific vs conserved, glomerulus (He 2021)"),
    ("kim2024", "human_enriched__core"): ("Kidney", "human-enriched vs conserved, fetal (Kim 2024)"),
    ("benner", "human_enriched__core"): ("Islet", "human-enriched vs common, beta cells (Benner 2014)"),
    ("benner", "mouse_enriched__core"): ("Islet", "mouse-enriched vs common, beta cells (Benner 2014)"),
    ("shami_hm", "species_specific"): ("Testis", "stage-unique vs in-phase, human-mouse (Shami 2020)"),
    ("shami_hm", "species_divergent"): ("Testis", "phase-shifted vs in-phase, human-mouse (Shami 2020)"),
    ("shami_hq", "species_specific"): ("Testis", "stage-unique vs in-phase, human-macaque (Shami 2020)"),
    ("aiz_hep", "nonconserved_zonation"): ("Liver", "non-conserved vs conserved zonation, hepatocytes (Aizarani 2019)"),
    ("ezran_bm_circ", "primate_enriched__core"): ("Bone marrow" + SAME_DATA, "primate-enriched vs conserved (Tabula Microcebus)"),
    ("ezran_bm_circ", "primate_depleted__core"): ("Bone marrow" + SAME_DATA, "primate-depleted vs conserved (Tabula Microcebus)"),
    ("ezran_lung_circ", "primate_depleted__core"): ("Lung" + SAME_DATA, "primate-depleted vs conserved (Tabula Microcebus)"),
    ("trav_circ", "species_divergent__core"): ("Lung" + SAME_DATA, "human vs mouse DE vs conserved (Travaglini 2020)"),
}


def save(fig, name: str):
    """One panel at its exact size: figures/panels/supp_published_differences/<name>.pdf."""
    return style.save_panel(fig, f"supp_published_differences/{name}", exact=True)


def forest(P, keys, labels, scores, xlabel, legend_cols):
    """One forest panel at the text width: a row per list, a mark with its interval per score, the legend above the axes
    and the note on the double dagger under the row labels. The axes box is the same for every panel."""
    n = len(keys)
    top, bot = 9.0 * -(-len(scores) // legend_cols) + 6.0, 27.0  # bp: legend rows above; ticks, axis label and note below
    H = n * PITCH + top + bot
    fig = plt.figure(figsize=(style.inch(WIDTH), style.inch(H)))
    ax = fig.add_axes([LEFT / WIDTH, bot / H, (WIDTH - LEFT - RIGHT) / WIDTH, n * PITCH / H])
    off = np.linspace(-SPREAD, SPREAD, len(scores))
    for i, k in enumerate(keys):
        y = n - 1 - i
        r = P.loc[k]
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#F3F3F3", lw=0, zorder=0)
        for sc, o in zip(scores, off):
            a, lo, hi = r[f"auroc_{sc}"], r[f"lo_{sc}"], r[f"hi_{sc}"]
            if pd.isna(a):
                continue
            ax.plot([lo, hi], [y - o, y - o], color=SCORES[sc].color, lw=0.8, zorder=2, solid_capstyle="round")  # first score on top
            ax.plot(a, y - o, marker=SCORES[sc].marker, ms=3.4, color=SCORES[sc].color, mew=0, zorder=3)
    ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1], fontsize=S_TEXT)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlim(*XLIM)
    ax.set_xticks(np.arange(1, 9) / 10)
    ax.set_xlabel(xlabel, fontsize=S_LABEL)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=S_TICK)
    h = [Line2D([], [], marker=SCORES[sc].marker, color=SCORES[sc].color, mew=0, lw=0.8, ms=3.4, label=SCORES[sc].name) for sc in scores]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=legend_cols, frameon=False, fontsize=S_TICK,
              handlelength=1.6, columnspacing=1.0, borderaxespad=0.3)
    fig.text(6 / WIDTH, 5 / H, NOTE, fontsize=S_TEXT, ha="left", va="bottom", color=INK)
    return fig


def row_label(tissue: str, text: str, n) -> str:
    return f"{tissue}: {text} (n = {int(n):,})"


def p_matched():
    """Every published list in its cohort: stratified AUROC of the listed genes against matched genes, five scores."""
    P = pd.read_csv(RU.OUT / "primary.csv")
    P = P[P.comparison == "vs_matched"].set_index(["assign_id", "set"])
    keys = [k for k in LABEL if k in P.index]
    labels = [row_label(LABEL[k][0], LABEL[k][1], P.loc[k, "n"]) for k in keys]
    fig = forest(P, keys, labels, list(SCORES), "stratified AUROC against matched genes", 3)
    return save(fig, "matched")


def p_conserved():
    """Divergent genes against the conserved genes of the same paper: stratified AUROC, four scores (no protein-feature gate)."""
    P = pd.read_csv(RU.OUT / "primary.csv")
    P = P[P.comparison == "vs_conserved_control"].set_index(["assign_id", "set"])
    keys = [k for k in CTRL_LABEL if k in P.index]
    labels = [row_label(*CTRL_LABEL[k], P.loc[k, "n"]) for k in keys]
    fig = forest(P, keys, labels, ["S_can", "D_free", "B1", "LVL"], "stratified AUROC against the same paper's conserved genes", 2)
    return save(fig, "conserved")


# panel name in figures/panels -> the function that draws it
DRAW = {"supp_published_differences/matched": p_matched, "supp_published_differences/conserved": p_conserved}


def draw_panels(names=None) -> list:
    """Draw the panels of this task from results/analysis/E6/primary.csv and nothing else. ``names``: panel names as in
    ``DRAW`` (default: both). Returns the files written."""
    style.apply()
    return [DRAW[n]() for n in (DRAW if names is None else names)]


def main() -> int:
    for path in draw_panels(sys.argv[1:] or None):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
