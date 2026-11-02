"""Supplementary panels: seed reliability (new), and panels made by their own tasks, copied unchanged.

New here:
  supp_celltype_seed_stability  re-render of T2.5 ``fig_supp_seed_stability`` from ``seed_stability.csv``:
                          agreement between seeds of each cell type's contribution to S_g.
  supp_seed_reliability   pairwise seed agreement of Delta-alpha, alpha and S_g per cohort (T3.3
                          ``seed_pairs.csv``) with the reliability of the seed mean (``reliability.csv``).
Copied (made and checked by their tasks; every copy is re-checked for fonts and run names):
  supp_covariate_audit        T2.4 ``fig_supp_covariate_audit``  (7-cohort covariate audit)
  supp_celltype_closure       T2.5 ``fig_supp_closure``
  supp_relations_density      X4 ``fig_relations_density``
  supp_relations_heatmap      X4 ``fig_relations_heatmap``
  supp_integration_raw_metrics      T5.5 ``raw_metrics_10models``
  supp_integration_purity_view      T5.5 ``composite_mixing_bio_10models`` (with raw kNN species purity)
Already in the manuscript folder, checked only: ``supp_functional_layer``, ``fig_known_measures``,
``fig_sequence_axis``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from revision.tasks.T8_4_figures import common as cm

REL_COHORTS = ["liv_hmonkey", "liv_hmouse", "liv_4sp", "test_hmouse", "test_3sp"]
READOUTS = [("spearman_D", "D", "Δα"), ("spearman_S", "S", "count-evidence margin $S_g$"),
            ("spearman_alpha", None, "α")]

COPIES = {
    "supp_covariate_audit": cm.RES / "T2_4" / "fig_supp_covariate_audit.pdf",
    "supp_celltype_closure": cm.RES / "T2_5" / "fig_supp_closure.pdf",
    "supp_relations_density": cm.RES / "X4" / "figures" / "fig_relations_density.pdf",
    "supp_relations_heatmap": cm.RES / "X4" / "figures" / "fig_relations_heatmap.pdf",
    "supp_integration_raw_metrics": cm.REPO / "notebooks/figures/baselines_master_ext/raw_metrics_10models.pdf",
    "supp_integration_purity_view": cm.REPO / "notebooks/figures/baselines_master_ext/composite_mixing_bio_10models.pdf",
}
EXISTING = ["supp_functional_layer", "fig_known_measures", "fig_sequence_axis"]


def panel_seed_reliability() -> plt.Figure:
    pairs = pd.read_csv(cm.RES / "T3_3" / "seed_pairs.csv")
    rel = pd.read_csv(cm.RES / "T3_3" / "reliability.csv").set_index("cohort")
    fig, ax = plt.subplots(figsize=(4.4, 2.6), layout="constrained")
    for i, c in enumerate(REL_COHORTS):
        p = pairs[pairs.cohort == c]
        for k, (col, key, _) in enumerate(READOUTS):
            color, mk = (cm.STY[key][0], cm.STY[key][1]) if key else ("#999999", "^")
            y = i + (k - 1) * 0.24
            rng = np.random.default_rng(i * 10 + k)
            yy = y + rng.uniform(-0.07, 0.07, len(p))
            ax.scatter(p[col], yy, s=7, marker=mk, color=color, alpha=0.75, lw=0, zorder=3)
        r = rel.loc[c]
        rv = r.rel_all if np.isfinite(r.rel_all) else r.rel_core
        ax.text(1.02, i, f"{rv:.2f}" if np.isfinite(rv) else "not estimated\n(3 seeds)",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6 if np.isfinite(rv) else 5)
    ax.text(1.02, -0.75, "reliability of\nseed-mean Δα", transform=ax.get_yaxis_transform(), ha="left",
            va="bottom", fontsize=5.5, color=cm.MUTED)
    ax.set_yticks(range(len(REL_COHORTS)))
    ax.set_yticklabels([f"{cm.COHORT[c]}\n({int(rel.loc[c, 'n_seeds'])} seeds)" for c in REL_COHORTS],
                       fontsize=6)
    ax.set_ylim(len(REL_COHORTS) - 0.5, -0.6)
    ax.set_xlim(-0.05, 1.0)
    ax.axvline(0, color=cm.MUTED, lw=0.6)
    ax.set_xlabel("agreement between two training seeds (Spearman ρ across genes; one point per seed pair)",
                  fontsize=6.3)
    fig.legend(handles=[Line2D([], [], marker=(cm.STY[k][1] if k else "^"), ls="none", ms=4,
                               color=(cm.STY[k][0] if k else "#999999"), label=lab) for _, k, lab in READOUTS],
               loc="outside lower center", ncol=3, fontsize=6)
    return fig


def panel_celltype_seed_stability() -> plt.Figure:
    st = pd.read_csv(cm.RES / "T2_5" / "seed_stability.csv")
    cohorts = [c for c in ["liv_hmonkey", "liv_hmouse"] if c in set(st.cohort)]
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5.6, 2.4), layout="constrained", sharex=True)
    for ax, c in zip(np.atleast_1d(axes), cohorts):
        d = st[st.cohort == c]
        types = d.groupby("cell_type").weight.first().sort_values(ascending=False)
        for i, (t, w) in enumerate(types.items()):
            v = d[d.cell_type == t].rho
            ax.plot([v.min(), v.max()], [i, i], color=cm.MUTED, lw=0.8, zorder=2)
            ax.scatter(v, [i] * len(v), s=10, color=cm.STY["S"][0] if w >= 0.10 else "#999999", zorder=3, lw=0)
        ax.set_yticks(range(len(types)))
        ax.set_yticklabels(["total margin (all cell types)" if t == "(S_g total)" else f"{t} ({w:.2f})"
                            for t, w in types.items()], fontsize=5.8)
        ax.set_ylim(len(types) - 0.5, -0.5)
        ax.axvline(0.9, color=cm.RULE, lw=0.6, ls=(0, (2, 2)))
        ax.set_title(f"{cm.COHORT[c]} (seeds 42, 43, 44)", fontsize=6.5)
    fig.supxlabel("agreement between seeds of a cell type's contribution to $S_g$ (Spearman ρ across genes;\n"
                  "points: seed pairs; in brackets: cell-type weight; blue: weight at least 0.10)", fontsize=6.2)
    return fig


def main() -> list:
    cm.setup()
    out = [cm.save(panel_seed_reliability(), "supp_seed_reliability"),
           cm.save(panel_celltype_seed_stability(), "supp_celltype_seed_stability")]
    for name, src in COPIES.items():
        out.append(cm.copy_pdf(src, name))
    out += [cm.FIG_OUT / f"{n}.pdf" for n in EXISTING]
    return out


if __name__ == "__main__":
    for p in main():
        print(p.name, cm.check_pdf(p))
