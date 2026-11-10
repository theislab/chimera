"""E6 candidate panels in the style of the harmonised panels (H1/E1). Run from chimera/:

    PYTHONPATH=$PWD mamba run -n chimera_env python -m analysis.tasks.E6_known_differences.figures

Panels in results/analysis/E6/panels/ (pdf + png). Composed with F1_figures/compose.py --spec
analysis/tasks/E6_known_differences/specs_e6.py, CHIMERA_MANUSCRIPT pointed at a scratch folder (E6 does not write into
the manuscript).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analysis.tasks.E6_known_differences import rules as RU  # noqa: E402
from analysis.tasks.H1_harmonise.figures import panel_rc  # noqa: E402
from analysis.tasks.T8_4_figures import common as cm  # noqa: E402

PANELS = RU.OUT / "panels"
# the paper's convention (T8_4_figures/common.STY): Δα orange circle, S_g blue square, profile correlation black triangle,
# level measure grey; the protein-feature gate in amber diamonds
SCORES = [("S_can", "Count evidence $S_g$", "#0072B2", "s"),
          ("D_free", r"$\Delta\alpha$, per-gene gate", "#D55E00", "o"),
          ("D_can", r"$\Delta\alpha$, protein-feature gate", "#E69F00", "D"),
          ("B1", "Pseudobulk profile correlation", "#222222", "^"),
          ("LVL", "Detection gap (no model)", "#999999", "v")]
INK, MUTED = "#222222", "#777777"

LABEL = {
    ("jor_hs", "human_specific__core"): ("Cortex", "Human-specific DE, MTG (Jorstad 2023)", "DLPFC 4 primates"),
    ("jor_hc", "human_vs_chimp_divergent__core"): ("Cortex", "Human vs chimpanzee DE, MTG (Jorstad 2023)", "DLPFC H+Chimp"),
    ("cag_am", "ape_vs_macaque_divergent__core"): ("Cortex", "Ape vs macaque DE, cingulate (Caglayan 2023)", "DLPFC H+Macaque"),
    ("cag_hs", "human_specific__core"): ("Cortex", "Human-specific DE, cingulate (Caglayan 2023)", "DLPFC 4 primates"),
    ("khr_hs", "human_specific__core"): ("Cortex", "Human-specific DE, ACC (Khrameeva 2020)", "DLPFC 4 primates"),
    ("hodge", "human_mouse_divergent__core"): ("Cortex", "Human-mouse divergent, MTG/VISp (Hodge 2019)", "M1 H+Mouse"),
    ("benner", "human_enriched__core"): ("Islet", "Human-enriched in beta cells (Benner 2014)", "Islet H+Mouse"),
    ("benner", "mouse_enriched__core"): ("Islet", "Mouse-enriched in beta cells (Benner 2014)", "Islet H+Mouse"),
    ("he2021", "species_divergent"): ("Kidney", "Species-specific, glomerulus (He 2021)", "Kidney H+Mouse"),
    ("kim2024", "human_enriched__core"): ("Kidney", "Human-enriched, fetal (Kim 2024)", "Kidney H+Mouse"),
    ("kim2024", "mouse_enriched__core"): ("Kidney", "Mouse-enriched, fetal (Kim 2024)", "Kidney H+Mouse"),
    ("shami_hm", "species_specific"): ("Testis", "Stage-unique, human-mouse (Shami 2020)", "Testis H+Mouse"),
    ("shami_hm", "species_divergent"): ("Testis", "Phase-shifted, human-mouse (Shami 2020)", "Testis H+Mouse"),
    ("shami_hq", "species_specific"): ("Testis", "Stage-unique, human-macaque (Shami 2020)", "Testis H+Macaque"),
    ("aiz_hep", "nonconserved_zonation"): ("Liver", "Non-conserved zonation, hepatocytes (Aizarani 2019)", "Liver H+Mouse"),
    ("ezran_lung_circ", "primate_depleted__core"): ("Lung*", "Primate-depleted (Tabula Microcebus)", "Lung 3 species"),
    ("ezran_lung_circ", "primate_enriched__core"): ("Lung*", "Primate-enriched (Tabula Microcebus)", "Lung 3 species"),
    ("trav_circ", "species_divergent__core"): ("Lung*", "Human vs mouse DE (Travaglini 2020)", "Lung 3 species"),
    ("ezran_bm_circ", "primate_enriched__core"): ("Marrow*", "Primate-enriched (Tabula Microcebus)", "Bone marrow"),
    ("bakken_circ", "human_only_marker__core"): ("Cortex*", "Human-only subclass markers (Bakken 2021)", "M1 H+Mouse"),
}
CTRL_LABEL = {
    ("hodge", "human_mouse_divergent__core"): "Cortex: Hodge divergent (top 3%) vs shared markers",
    ("bakken_circ", "human_only_marker__core"): "Cortex*: Bakken human-only vs conserved markers",
    ("he2021", "species_divergent"): "Kidney: He glomerular specific vs conserved",
    ("kim2024", "human_enriched__core"): "Kidney: Kim human-enriched vs conserved",
    ("benner", "human_enriched__core"): "Islet: Benner human-enriched vs common",
    ("benner", "mouse_enriched__core"): "Islet: Benner mouse-enriched vs common",
    ("shami_hm", "species_specific"): "Testis: Shami unique vs in-phase (H-M)",
    ("shami_hm", "species_divergent"): "Testis: Shami phase-shifted vs in-phase (H-M)",
    ("shami_hq", "species_specific"): "Testis: Shami unique vs in-phase (H-Q)",
    ("aiz_hep", "nonconserved_zonation"): "Liver: Aizarani non-conserved vs conserved zonation",
    ("ezran_bm_circ", "primate_enriched__core"): "Marrow*: primate-enriched vs conserved",
    ("ezran_bm_circ", "primate_depleted__core"): "Marrow*: primate-depleted vs conserved",
    ("ezran_lung_circ", "primate_depleted__core"): "Lung*: primate-depleted vs conserved",
    ("trav_circ", "species_divergent__core"): "Lung*: Travaglini DE vs conserved",
}


def save(fig, name):
    PANELS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PANELS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(PANELS / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def forest(ax, P, keys, labels, scores, xlim):
    n = len(keys)
    off = np.linspace(-0.28, 0.28, len(scores))
    for i, k in enumerate(keys):
        y = n - 1 - i
        r = P.loc[k]
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#F3F3F3", lw=0, zorder=0)
        for (sc, _, col, mk), o in zip(scores, off):
            a, lo, hi = r[f"auroc_{sc}"], r[f"lo_{sc}"], r[f"hi_{sc}"]
            if pd.isna(a):
                continue
            ax.plot([lo, hi], [y + o, y + o], color=col, lw=0.8, zorder=2, solid_capstyle="round")
            ax.plot(a, y + o, marker=mk, ms=3.4, color=col, mec=INK, mew=0.3, zorder=3)
    ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(*xlim)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def p_matched():
    P = pd.read_csv(RU.OUT / "primary.csv")
    P = P[P.comparison == "vs_matched"].set_index(["assign_id", "set"])
    keys = [k for k in LABEL if k in P.index]
    labels = [f"{LABEL[k][0]}: {LABEL[k][1]} (n {int(P.loc[k, 'n'])})" for k in keys]
    fig, ax = plt.subplots(figsize=(3.3, 0.2 * len(keys) + 0.7))
    forest(ax, P, keys, labels, SCORES, (0.15, 0.85))
    ax.set_xlabel("Stratified AUROC, published species-different genes vs matched genes\n"
                  "(< 0.5: placed below genes of equal expression, cell-type specificity and protein identity; * same data as the cohort)")
    h = [Line2D([], [], marker=m, color=c, mec=INK, mew=0.3, lw=0.8, ms=3.4, label=l) for _, l, c, m in SCORES]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handlelength=1.6, columnspacing=1.0)
    save(fig, "e6_matched")


def p_conserved():
    P = pd.read_csv(RU.OUT / "primary.csv")
    P = P[P.comparison == "vs_conserved_control"].set_index(["assign_id", "set"])
    keys = [k for k in CTRL_LABEL if k in P.index]
    labels = [f"{CTRL_LABEL[k]}  (n {int(P.loc[k, 'n'])})" for k in keys]
    sc = [SCORES[0], SCORES[1], SCORES[3], SCORES[4]]
    fig, ax = plt.subplots(figsize=(2.9, 0.2 * len(keys) + 0.7))
    forest(ax, P, keys, labels, sc, (0.08, 0.72))
    ax.set_xlabel("Stratified AUROC, published divergent genes vs the same paper's conserved genes\n(< 0.5: divergent placed below conserved; * same data as the cohort)")
    h = [Line2D([], [], marker=m, color=c, mec=INK, mew=0.3, lw=0.8, ms=3.4, label=l) for _, l, c, m in sc]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              handlelength=1.6, columnspacing=1.0)
    save(fig, "e6_conserved")


def main():
    cm.setup()
    panel_rc()
    p_matched()
    p_conserved()


if __name__ == "__main__":
    main()
