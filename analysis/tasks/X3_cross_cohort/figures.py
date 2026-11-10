"""X3 (EXPLORATORY): figure drafts from the X3 CSVs. PDF with embedded fonts (TrueType, type 42).

Writes results/analysis/X3/figures/{fig_main_protein_classes,fig_supp_*}.pdf (+ .png previews).
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from analysis.common import gene_ids  # noqa: E402

sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
import _figure_design as FD  # noqa: E402

OUT = gene_ids.REPO_ROOT / "results" / "analysis" / "X3"
FIG = OUT / "figures"
ORDER = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp", "test_hmac", "test_hmouse", "test_3sp"]
CLABEL = {"liv_hmonkey": "L H+Mo", "liv_hmouse": "L H+Mm", "liv_hzeb": "L H+Zf", "liv_4sp": "L 4-sp",
          "test_hmac": "T H+Mac", "test_hmouse": "T H+Mm", "test_3sp": "T 3-sp"}
CLASSES = [("TF", "TFs (all)"), ("TF_nuclear_rec", "  nuclear receptor"), ("TF_homeodomain", "  homeodomain"),
           ("TF_forkhead", "  forkhead"), ("TF_C2H2_ZF", "  C2H2 zinc finger"), ("protein_kinase", "protein kinases"),
           ("receptor", "receptors"), ("GPCR", "  GPCR"), ("ion_channel", "ion channels"),
           ("SLC_transporter", "SLC transporters"), ("chromatin_reg", "chromatin regulators"),
           ("enzyme", "enzymes (all EC)"), ("EC4", "  lyases (EC4)"), ("EC6", "  ligases (EC6)"),
           ("mitochondrial", "mitochondrial")]


def save(fig, name):
    FD.save_figure(fig, name, out_dir=FIG, formats=("pdf", "png"), close=True)


def dot_matrix(ax, tab: pd.DataFrame, title: str, lim: float = 0.5, show_rows: bool = True):
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    cmap = plt.get_cmap("RdBu_r")
    for i, (k, _) in enumerate(CLASSES):
        for j, c in enumerate(ORDER):
            r = tab[(tab.cls == k) & (tab.cohort == c)]
            if r.empty or not np.isfinite(r.excess_d.iloc[0]):
                ax.scatter(j, i, marker="x", s=6, color="#bbbbbb", lw=0.5)
                continue
            v, p = float(r.excess_d.iloc[0]), float(r.p.iloc[0])
            col = cmap(norm(np.clip(v, -lim, lim)))
            size = 8 + 90 * min(abs(v), lim) / lim
            if p < 0.01:
                ax.scatter(j, i, s=size, color=col, edgecolor="#333333", lw=0.3)
            else:
                ax.scatter(j, i, s=size, facecolor="white", edgecolor=col, lw=0.8)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([CLABEL[c] for c in ORDER], fontsize=5.5, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    if show_rows:
        ax.set_yticklabels([lab for _, lab in CLASSES], fontsize=6)
    else:
        ax.tick_params(labelleft=False)
    ax.set_ylim(len(CLASSES) - 0.5, -0.5)
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.axvline(3.5, color="#999999", lw=0.5)
    ax.set_title(title, fontsize=7, loc="left")
    return norm, cmap


def fig_main_classes():
    d = pd.read_csv(OUT / "class_excess.csv")
    d = d[d.matching == "C4"]
    e = pd.read_csv(OUT / "esm_class_loadings.csv")
    panels = [(d[d.score == "D_raw"], "a  Chimera $D$"),
              (e[e.score == "non_esm_rD"], "b  $D$, non-ESM part"),
              (d[d.score == "S"], "c  evidence $S_g$"),
              (d[d.score == "B1b_raw"], "d  B1b profile r"),
              (d[d.score == "B1c_raw"], "e  B1c level")]
    fig, axes = plt.subplots(1, 5, figsize=(FD.DOUBLE_W + 0.6, 3.1), sharey=True)
    for i, (ax, (tab, title)) in enumerate(zip(axes, panels)):
        norm, cmap = dot_matrix(ax, tab, title, show_rows=(i == 0))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("matched excess (SD units)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    save(fig, "fig_main_protein_classes")


def fig_supp_depth():
    s = pd.read_csv(OUT / "depth_summary.csv").set_index("cohort")
    a = pd.read_csv(OUT / "depth_gene_agreement.csv")
    m = pd.read_csv(OUT / "depth_species_margin.csv")
    fig, axes = plt.subplots(1, 4, figsize=(FD.DOUBLE_W, 2.1), gridspec_kw=dict(width_ratios=[1, 1, 1.3, 1]))
    liv = ["liv_hmonkey", "liv_hmouse", "liv_hzeb"]
    sp = ["Monkey", "Mouse", "Zebrafish"]
    x = np.arange(3)
    ax = axes[0]
    ax.plot(x, s.loc[liv, "median_pair_identity"], "-o", color="#222222", ms=3, label="protein identity")
    ax.plot(x, s.loc[liv, "median_B1b_raw_common"], "-s", color="#777777", ms=3, label="B1b profile r (median)")
    ax.plot(x, s.loc[liv, "frac_tail_s42"], "-^", color="#0072B2", ms=3, label=r"Chimera: share $|\Delta\alpha|>0.05$")
    ax.set_xticks(x)
    ax.set_xticklabels(sp, fontsize=6)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=5, frameon=False, loc="lower left")
    ax.set_title("a  liver pairs by depth", fontsize=7, loc="left")
    ax = axes[1]
    ax.plot(x, s.loc[liv, "frac_S_neg_of_unmasked"], "-o", color="#D55E00", ms=3, label=r"$S_g<0$ (of unmasked)")
    ax.plot(x, s.loc[liv, "frac_S_masked"], "-s", color="#999999", ms=3, label=r"$S_g$ masked")
    ax.set_xticks(x)
    ax.set_xticklabels(sp, fontsize=6)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=5, frameon=False, loc="upper left")
    ax.set_title("b  count evidence by depth", fontsize=7, loc="left")
    ax = axes[2]
    combos = [("liv_hmonkey", "liv_hmouse", "Mo-Mm"), ("liv_hmonkey", "liv_hzeb", "Mo-Zf"),
              ("liv_hmouse", "liv_hzeb", "Mm-Zf"), ("test_hmac", "test_hmouse", "T Mac-Mm")]  # noqa: E501
    scores = [("D_raw", "#0072B2", r"$D$"), ("S", "#D55E00", r"$S_g$"), ("B1b_raw", "#777777", "B1b")]
    for k, (sc, col, lab) in enumerate(scores):
        v = [a[(a.a == p) & (a.b == q) & (a.score == sc)].rho_p.iloc[0] for p, q, _ in combos]
        ax.bar(np.arange(4) + (k - 1) * 0.26, v, width=0.24, color=col, label=lab)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([c[2] for c in combos], fontsize=5.5, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel(r"gene-level $\rho_p$ between pairs")
    ax.legend(fontsize=5, frameon=False)
    ax.set_title("c  same genes, two partners", fontsize=7, loc="left")
    ax = axes[3]
    rows = [("liv_4sp", "Monkey"), ("liv_4sp", "Mouse"), ("liv_4sp", "Zebrafish"), ("test_3sp", "Macaque"),
            ("test_3sp", "Mouse")]
    v = [m[(m.cohort == c) & (m.species == s_)].frac_negative.iloc[0] for c, s_ in rows]
    ax.bar(range(5), v, color=[FD.species_color(s_) for _, s_ in rows])
    ax.set_xticks(range(5))
    ax.set_xticklabels(["L Mo", "L Mm", "L Zf", "T Mac", "T Mm"], fontsize=5.5, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of genes, species margin < 0")
    ax.set_title("d  per-species margin (joint fits)", fontsize=7, loc="left")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_supp_divergence_depth")


def fig_supp_esm():
    r = pd.read_csv(OUT / "esm_cv_r2.csv")
    mo = pd.read_csv(OUT / "esm_morans_i.csv")
    cp = pd.read_csv(OUT / "summary_coupling.csv").set_index("cohort")
    fig, axes = plt.subplots(1, 3, figsize=(FD.DOUBLE_W, 2.2), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    sc = [("rD", "#0072B2", r"Chimera $D$"), ("rS", "#D55E00", r"count evidence $S_g$"),
          ("rB1b", "#777777", "baseline B1b")]
    for ax, tab, col, ylab, title in [(axes[0], r, "cv_r2_esm64", r"CV $R^2$ from 64 ESM-2 PCs", "a  predictable from protein"),
                                      (axes[1], mo, "morans_I", "Moran's I, ESM-2 15-NN graph", "b  smooth over protein space")]:
        for k, (s_, cc, lab) in enumerate(sc):
            key = "target" if col == "cv_r2_esm64" else "score"
            v = [tab[(tab.cohort == c) & (tab[key] == s_)][col].iloc[0] for c in ORDER]
            ax.bar(np.arange(7) + (k - 1) * 0.26, v, width=0.24, color=cc, label=lab)
        ax.axhline(0, color="#999999", lw=0.5)
        ax.set_xticks(range(7))
        ax.set_xticklabels([CLABEL[c] for c in ORDER], fontsize=5.5, rotation=60, ha="right", rotation_mode="anchor")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=7, loc="left")
    axes[0].legend(fontsize=5, frameon=False, loc="upper left")
    ax = axes[2]
    for c in ORDER:
        col = "#0072B2" if c.startswith("liv") else "#D55E00"
        ax.scatter(cp.loc[c, "cv_r2_esm64_residual_D"], cp.loc[c, "rho_p_D_S"], s=14, color=col)
        ax.annotate(CLABEL[c], (cp.loc[c, "cv_r2_esm64_residual_D"], cp.loc[c, "rho_p_D_S"]), fontsize=5,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(r"CV $R^2$ of $D$ from ESM-2")
    ax.set_ylabel(r"$\rho_p(D, S_g)$")
    ax.set_xlim(-0.03, 0.7)
    ax.set_ylim(0, 0.55)
    ax.set_title("c  features vs evidence", fontsize=7, loc="left")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_supp_esm_substrate")


def fig_supp_celltypes():
    """Share of negative (top) and positive (bottom) S_g mass per cell type over its cell weight."""
    s = pd.read_csv(OUT / "agreement_celltype_shares.csv")
    s = s[s.group == "all"]
    liv = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp"]
    types = ["Hepatocytes", "Endothelial", "fibroblasts", "Cholangio", "Macrophages", "Kupffer cells"]
    tes = ["test_hmac", "test_hmouse", "test_3sp"]
    stages = ["SG", "SC", "rSD", "eSD", "Sertoli", "Other_somatic"]
    greys = ["#bbbbbb", "#888888", "#555555", "#111111"]
    fig, axes = plt.subplots(2, 2, figsize=(FD.DOUBLE_W, 3.4), sharex="col")
    for col, (cs, tt, tis) in enumerate([(liv, types, "liver"), (tes, stages, "testis")]):
        for row, (key, lab) in enumerate([("neg_per_weight", "negative"), ("pos_per_weight", "positive")]):
            ax = axes[row, col]
            for k, c in enumerate(cs):
                x = s[s.cohort == c].set_index("cell_type").reindex(tt)
                off = (k - (len(cs) - 1) / 2) * 0.8 / len(cs)
                ax.bar(np.arange(len(tt)) + off, np.log2(x[key]), width=0.8 / len(cs),
                       color=greys[k + (4 - len(cs))], label=CLABEL[c])
            ax.axhline(0, color="#999999", lw=0.5)
            ax.set_ylabel(f"log$_2$({lab} share / weight)", fontsize=6)
            ax.set_title(f"{'abcd'[row * 2 + col]}  {tis}: {lab} $S_g$ mass", fontsize=7, loc="left")
            if row == 0:
                ax.legend(fontsize=5, frameon=False, ncol=2)
            else:
                ax.set_xticks(range(len(tt)))
                ax.set_xticklabels(tt, fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    save(fig, "fig_supp_celltype_evidence")


def main():
    FD.apply_design()
    fig_main_classes()
    fig_supp_depth()
    fig_supp_esm()
    fig_supp_celltypes()
    print(f"figures in {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
