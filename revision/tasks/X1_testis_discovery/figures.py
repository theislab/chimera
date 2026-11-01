"""X1 (EXPLORATORY): figure drafts from the X1 CSVs. PDF with embedded fonts (TrueType, type 42).

Writes results/revision/X1/figures/{fig_main_stage_deployment,fig_supp_*}.pdf (+ .png previews).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from revision.common import gene_ids  # noqa: E402

sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
import _figure_design as FD  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leads as LD  # noqa: E402

OUT = gene_ids.REPO_ROOT / "results" / "revision" / "X1"
FIG = OUT / "figures"
COH = {"test_hmac": ("H+Macaque", FD.species_color("Macaque"), "o"),
       "test_hmouse": ("H+Mouse", FD.species_color("Mouse"), "s"),
       "test_3sp": ("3-species", "#222222", "^")}
GROUP_LABEL = {"SG": "SG", "SC": "SC", "rSD": "rSD", "eSD": "eSD", "Sertoli": "Sert",
               "Other_somatic": "Som", "broad/low": "broad"}
SCORE_LABEL = {"D_raw": r"Chimera $D$ (raw prior)", "D_clamp": r"Chimera $D$ (clamped prior)",
               "S_g_ema": r"Chimera margin $S_g$", "B1b_raw": "Baseline B1b (profile r)",
               "B1b": "Baseline B1b (residualised)", "B1c": "Baseline B1c (level)",
               "PB_dc_own": "Baseline: -|stage diff.|", "PB_lfc_own": "Baseline: -|stage log2FC|"}


SHORT = {"D_raw": r"$D$ raw prior", "D_clamp": r"$D$ clamped prior", "S_g_ema": r"margin $S_g$",
         "B1b_raw": "B1b (profile r)", "B1c": "B1c (level)", "PB_dc_own": "-|stage diff.|"}


def save(fig, name):
    FD.save_figure(fig, name, out_dir=FIG, formats=("pdf", "png"), close=True)


def excess_panel(ax, ex, score, matching, peak_def="main", groups=LD.GROUPS, ylim=None, title=None):
    x = np.arange(len(groups))
    for j, (c, (lab, col, mk)) in enumerate(COH.items()):
        d = ex[(ex.cohort == c) & (ex.run_label == c) & (ex.score == score) & (ex.matching == matching)
               & (ex.peak_def == peak_def)].set_index("peak").reindex(groups)
        sig = d.p.to_numpy() < 0.01
        xx = x + (j - 1) * 0.22
        ax.scatter(xx[sig], d.excess_d.to_numpy()[sig], s=16, marker=mk, color=col, label=lab, zorder=3,
                   edgecolor="white", linewidth=0.4)
        ax.scatter(xx[~sig], d.excess_d.to_numpy()[~sig], s=16, marker=mk, facecolor="white", edgecolor=col,
                   linewidth=0.8, zorder=3)
    ax.axhline(0, color="#999999", lw=0.5, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABEL[g] for g in groups], fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("matched excess (SD units)")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title or SCORE_LABEL[score], fontsize=7, loc="left")


def fig_main():
    ex = pd.read_csv(OUT / "stage_excess.csv")
    grp = pd.read_csv(OUT / "stage_groups.csv")
    att = pd.read_csv(OUT / "stage_attribution.csv")
    g = LD.load("test_hmouse")
    g["peak"] = LD.peak_stage(g)
    fig, axes = plt.subplots(1, 4, figsize=(FD.DOUBLE_W, 2.2), gridspec_kw=dict(width_ratios=[1, 1.2, 1.2, 1.1]))
    # a: protein identity by human peak stage
    ax = axes[0]
    data = [g.loc[g.peak == k, "prior_raw"].to_numpy() * 100 for k in LD.GROUPS]
    q = np.array([np.percentile(d, [25, 50, 75]) for d in data])
    x = np.arange(len(LD.GROUPS))
    ax.vlines(x, q[:, 0], q[:, 2], color="#444444", lw=1.6)
    ax.scatter(x, q[:, 1], s=14, color="#222222", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABEL[k] for k in LD.GROUPS], fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("protein identity (%)")
    ax.set_title("a  protein conservation", fontsize=7, loc="left")
    # b, c: matched excess, Chimera vs baseline, same y scale
    lim = (-0.3, 0.45)
    excess_panel(axes[1], ex, "D_raw", "C_pat", ylim=lim, title=r"b  Chimera $D$ (raw prior)")
    excess_panel(axes[2], ex, "B1b_raw", "C_pat", ylim=lim, title="c  baseline B1b (profile r)")
    axes[2].set_ylabel("")
    axes[1].legend(loc="upper right", fontsize=5.5, frameon=False)
    # d: where the SC-peak margin sits (positive S_g share per stage / stage weight)
    ax = axes[3]
    a = att[att.peak == "SC"]
    for j, (c, (lab, col, mk)) in enumerate(COH.items()):
        d = a[a.run_label == c].set_index("stage").reindex(LD.STAGES)
        ax.bar(np.arange(6) + (j - 1) * 0.26, d.enrich_pos, width=0.24, color=col, label=lab)
    ax.axhline(1, color="#999999", lw=0.5)
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels([GROUP_LABEL[k] for k in LD.STAGES], fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("positive-$S_g$ share / stage weight")
    ax.set_title("d  SC-peak genes: where $S_g$ sits", fontsize=7, loc="left")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_main_stage_deployment")


def fig_supp_sensitivity():
    ex = pd.read_csv(OUT / "stage_excess.csv")
    scores = ["D_raw", "D_clamp", "S_g_ema", "B1b_raw", "B1c", "PB_dc_own"]
    fig, axes = plt.subplots(3, 6, figsize=(FD.DOUBLE_W, 4.6), sharey="row")
    rows = [("main", "C", "peak >= 1 log2, matched on C"), ("main", "C_pat_pool", "matched on C + tau + pooled depth"),
            ("strict", "C_pat", "peak >= 2 log2, matched on C + tau")]
    for i, (pdef, m, rlab) in enumerate(rows):
        for jj, s in enumerate(scores):
            ax = axes[i, jj]
            excess_panel(ax, ex, s, m, peak_def=pdef, groups=LD.STAGES, title=SHORT[s] if i == 0 else "")
            if jj:
                ax.set_ylabel("")
            else:
                ax.set_ylabel(f"{rlab}\nexcess (SD)", fontsize=6)
            ax.tick_params(axis="x", labelsize=5)
    axes[0, 0].legend(loc="lower right", fontsize=5, frameon=False)
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    save(fig, "fig_supp_stage_sensitivity")


def fig_supp_seeds():
    f = OUT / "seed_robustness.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(FD.SINGLE_W * 1.6, 2.2), sharey=True)
    for ax, score in zip(axes, ["D_raw", "S_g_ema"]):
        for j, c in enumerate(["test_hmouse", "test_3sp"]):
            lab, col, mk = COH[c]
            d = s[(s.cohort == c) & (s.score == score) & (s.n_runs == 1)]
            for k, grp in enumerate(["SC", "rSD", "eSD", "SG"]):
                v = d[d.peak == grp].excess_d.to_numpy()
                ax.scatter(np.full(len(v), k + (j - 0.5) * 0.3), v, s=10, marker=mk, color=col,
                           label=lab if k == 0 else None, edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="#999999", lw=0.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["SC", "rSD", "eSD", "SG"])
        ax.set_title(SCORE_LABEL[score] + ", per seed", fontsize=7, loc="left")
    axes[0].set_ylabel("matched excess (C + tau, SD)")
    axes[0].legend(loc="upper right", fontsize=5.5, frameon=False)
    fig.tight_layout()
    save(fig, "fig_supp_seed_robustness")


def fig_supp_x():
    x = pd.read_csv(OUT / "x_chromosome.csv")
    grp = pd.read_csv(OUT / "stage_groups.csv")
    fig, axes = plt.subplots(1, 2, figsize=(FD.SINGLE_W * 1.8, 2.2), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axes[0]
    d = grp[grp.cohort == "test_hmouse"].set_index("peak").reindex(LD.GROUPS)
    ax.bar(np.arange(7), d.frac_X * 100, color="#444444", width=0.6)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels([GROUP_LABEL[k] for k in LD.GROUPS], fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("X-linked genes (%)")
    ax.set_title("a  X share by human peak stage", fontsize=7, loc="left")
    ax = axes[1]
    labs = ["test_hmac", "test_hmouse", "test_3sp", "liv_hmonkey", "liv_hmouse", "liv_4sp"]
    sc = ["D_raw", "D_clamp", "S_g_ema", "B1b_raw"]
    mk = {"D_raw": "o", "D_clamp": "D", "S_g_ema": "s", "B1b_raw": "^"}
    cl = {"D_raw": "#0072B2", "D_clamp": "#56B4E9", "S_g_ema": "#009E73", "B1b_raw": "#D55E00"}
    for j, s in enumerate(sc):
        d = x[(x.subset == "all") & (x.matching == "C") & (x.score == s)].set_index("run_label").reindex(labs)
        xx = np.arange(len(labs)) + (j - 1.5) * 0.17
        ok = d.excess_d.notna().to_numpy()
        sig = (d.p < 0.01).to_numpy()
        ax.scatter(xx[ok & sig], d.excess_d[ok & sig], marker=mk[s], s=14, color=cl[s], label=SCORE_LABEL[s])
        ax.scatter(xx[ok & ~sig], d.excess_d[ok & ~sig], marker=mk[s], s=14, facecolor="white", edgecolor=cl[s])
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(["T H+Mac", "T H+Mouse", "T 3-sp", "L H+Mon", "L H+Mouse", "L 4-sp"], fontsize=5.5)
    ax.set_ylabel("X vs autosomes, matched excess (SD)")
    ax.set_title("b  X-linked genes, testis (T) and liver (L)", fontsize=7, loc="left")
    ax.legend(loc="upper right", fontsize=5, frameon=False)
    fig.tight_layout()
    save(fig, "fig_supp_x_chromosome")


def fig_supp_depth():
    a, b = LD.by_ensg(LD.load("test_hmac")), LD.by_ensg(LD.load("test_hmouse"))
    j = a.join(b, lsuffix="_mac", rsuffix="_mus", how="inner")
    st = pd.read_csv(OUT / "depth_stage_3sp.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FD.DOUBLE_W, 2.2))
    for ax, col, lab in [(axes[0], "alpha", r"Chimera $\alpha$"), (axes[1], "B1b_raw", "baseline B1b (profile r)")]:
        ok = j[f"{col}_mac"].notna() & j[f"{col}_mus"].notna()
        ax.hexbin(j.loc[ok, f"{col}_mac"], j.loc[ok, f"{col}_mus"], gridsize=45, cmap="Greys", bins="log",
                  mincnt=1, linewidths=0)
        lo = min(j.loc[ok, f"{col}_mac"].min(), j.loc[ok, f"{col}_mus"].min())
        hi = max(j.loc[ok, f"{col}_mac"].max(), j.loc[ok, f"{col}_mus"].max())
        ax.plot([lo, hi], [lo, hi], color="#D55E00", lw=0.6)
        from scipy.stats import spearmanr
        r = spearmanr(j.loc[ok, f"{col}_mac"], j.loc[ok, f"{col}_mus"])[0]
        ax.set_xlabel(f"{lab}, H+Macaque")
        ax.set_ylabel(f"{lab}, H+Mouse")
        FD.annotate_stat(ax, f"rho = {r:.2f}\nmedian {j.loc[ok, col + '_mac'].median():.2f} vs "
                             f"{j.loc[ok, col + '_mus'].median():.2f}", "upper left")
    axes[0].set_title("a  same genes, two partners", fontsize=7, loc="left")
    axes[1].set_title("b", fontsize=7, loc="left")
    ax = axes[2]
    x = np.arange(6)
    ax.bar(x - 0.18, st.median_absdc_Macaque, width=0.34, color=FD.species_color("Macaque"), label="Macaque")
    ax.bar(x + 0.18, st.median_absdc_Mouse, width=0.34, color=FD.species_color("Mouse"), label="Mouse")
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABEL[k] for k in st.stage], fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("median |species-centred log2 diff.|\nto human (pseudobulk)")
    ax.set_title("c  baseline divergence by stage (3-sp)", fontsize=7, loc="left")
    ax.legend(loc="upper left", fontsize=5.5, frameon=False)
    fig.tight_layout()
    save(fig, "fig_supp_divergence_depth")


def fig_supp_mtdna():
    m = pd.read_csv(OUT / "mtdna_stage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(FD.SINGLE_W * 1.9, 2.2))
    ax = axes[0]
    for j, (c, (lab, col, mk)) in enumerate(COH.items()):
        d = m[m.cohort == c][[f"C_{k}" for k in LD.STAGES]].sum()
        ax.bar(np.arange(6) + (j - 1) * 0.26, d.to_numpy(), width=0.24, color=col, label=lab)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels([GROUP_LABEL[k] for k in LD.STAGES], fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(r"sum of stage contributions $C_{g,k}$")
    ax.set_title("a  11 mtDNA genes: margin by stage", fontsize=7, loc="left")
    ax.legend(loc="lower left", fontsize=5.5, frameon=False)
    ax = axes[1]
    for s, c in (("Macaque", "test_hmac"), ("Mouse", "test_hmouse")):
        d = m[m.cohort == c][[f"lfc_{s}_{k}" for k in LD.STAGES]].mean()
        ax.plot(np.arange(6), d.to_numpy(), marker="o", color=FD.species_color(s), label=s, ms=4)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels([GROUP_LABEL[k] for k in LD.STAGES], fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("mean log2 fold change to human\n(pseudobulk, mtDNA genes)")
    ax.set_title("b  species level of mtDNA transcripts", fontsize=7, loc="left")
    ax.legend(loc="lower left", fontsize=5.5, frameon=False)
    fig.tight_layout()
    save(fig, "fig_supp_mtdna")


def main():
    FD.apply_design()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_main()
    fig_supp_sensitivity()
    fig_supp_seeds()
    fig_supp_x()
    fig_supp_depth()
    fig_supp_mtdna()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
