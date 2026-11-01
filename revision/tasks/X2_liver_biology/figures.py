"""X2 (EXPLORATORY): figure drafts from the X2 CSVs. PDF with embedded fonts (TrueType, type 42).

Writes results/revision/X2/figures/{fig_main_liver_drivers, fig_supp_liver_zonation,
fig_supp_liver_presence, fig_supp_liver_named_genes}.pdf (+ .png previews).
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from revision.common import gene_ids  # noqa: E402

sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
import _figure_design as FD  # noqa: E402

OUT = gene_ids.REPO_ROOT / "results" / "revision" / "X2"
FIG = OUT / "figures"
SPECIES = ["Human", "Monkey", "Mouse", "Zebrafish"]
COH = [("liv_4sp", "4-sp"), ("liv_hmonkey", "H+Monkey"), ("liv_hmouse", "H+Mouse"), ("liv_hzeb", "H+Zebrafish")]
SETS = [("fig4c:Urea cycle / GNG", "Urea / GNG (Fig 4c)"), ("zonation:zPP_Mouse", "Periportal, mouse-defined"),
        ("zonation:zPC_Mouse", "Pericentral, mouse-defined"), ("mtDNA", "mtDNA-encoded"),
        ("OXPHOS_nuclear", "OXPHOS, nuclear"), ("fig6:A0", "Archetype A0"), ("fig6:A2", "Archetype A2"),
        ("fig4c:Liver support", "Liver support (Fig 4c)")]
TEXT = "#222222"
OFFS = {"PCK1": 0.25, "ARG1": -0.05, "ASL": -0.3, "IGF1": 0.15, "CPS1": -0.1}
MUTED = "#777777"


def save(fig, name):
    FD.save_figure(fig, name, out_dir=FIG, formats=("pdf", "png"), close=True)


def _drv(cohort, matching="C_hep", universe="all"):
    d = pd.read_csv(OUT / "drivers.csv")
    return d[(d.cohort == cohort) & (d.matching == matching) & (d.universe == universe)]


def panel_heat(ax, cax):
    d = _drv("liv_4sp")
    cols = [("D", r"$D$"), ("S_ema_cons", r"$S_g$"), ("B1b_raw", "B1b"), ("lfc__Monkey__Hepatocytes", "lfc Mon"),
            ("lfc__Mouse__Hepatocytes", "lfc Mus"), ("lfc__Zebrafish__Hepatocytes", "lfc Zeb")]
    cols += [(f"Cs__{s}", s[:3]) for s in SPECIES]
    cols += [("Ck__Hepatocytes", "Hep"), ("Ck__Endothelial", "Endo")]
    M = np.full((len(SETS), len(cols)), np.nan)
    P = np.ones_like(M)
    for i, (s, _) in enumerate(SETS):
        for j, (c, _) in enumerate(cols):
            r = d[(d.set == s) & (d.score == c)]
            if len(r):
                M[i, j], P[i, j] = r.excess_d.iloc[0], r.p.iloc[0]
    lim = 3.0
    im = ax.imshow(np.clip(M, -lim, lim), cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]) and P[i, j] < 0.01:
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=4.6,
                        color="white" if abs(M[i, j]) > 1.8 else TEXT)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], fontsize=5.5, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(SETS)))
    ax.set_yticklabels([s[1] for s in SETS], fontsize=5.5)
    for x in (2.5, 5.5, 9.5):
        ax.axvline(x, color="white", lw=1.5)
    for xc, lab in ((1.0, "D, S, B1b"), (4.0, "hep. log2FC vs human"), (7.5, "margin by species"),
                    (10.5, "by type")):
        ax.text(xc, -0.62, lab, ha="center", va="bottom", fontsize=5, color=MUTED)
    ax.set_title("a  4-sp: matched excess (SD); number if p < 0.01", fontsize=7, loc="left", pad=14)
    cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5)
    cb.set_label("matched excess (SD), clipped at 3", fontsize=5.5)


def panel_cohorts(ax):
    d = pd.read_csv(OUT / "drivers.csv")
    d = d[(d.matching == "C_hep") & (d.universe == "all")]
    show = [("fig4c:Urea cycle / GNG", "Urea"), ("zonation:zPP_Mouse", "PP"), ("mtDNA", "mtDNA")]
    mk = {"D": ("o", "#0072B2", r"Chimera $D$"), "B1b_raw": ("s", "#D55E00", "baseline B1b")}
    ticks, labels = [], []
    for i, (c, clab) in enumerate(COH):
        for k, (s_, lab) in enumerate(show):
            x0 = i * 4 + k
            ticks.append(x0)
            labels.append(lab)
            for j, (sc, (m, col, sl)) in enumerate(mk.items()):
                r = d[(d.cohort == c) & (d.set == s_) & (d.score == sc)]
                if not len(r) or not np.isfinite(r.excess_d.iloc[0]):
                    continue
                y, p = r.excess_d.iloc[0], r.p.iloc[0]
                kw = dict(color=col) if p < 0.01 else dict(facecolor="white", edgecolor=col)
                ax.scatter(x0 + (j - 0.5) * 0.35, y, marker=m, s=14, zorder=3,
                           label=sl if (i == 0 and k == 0) else None, **kw)
        ax.text(i * 4 + 1, 3.0, clab, ha="center", fontsize=5.5, color=TEXT)
        if i:
            ax.axvline(i * 4 - 1, color="#e5e5e5", lw=0.6)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=5, rotation=90)
    ax.set_ylim(-3.0, 3.5)
    ax.set_ylabel("matched excess (SD)")
    ax.set_title("b  same sets in each liver cohort (filled p < 0.01)", fontsize=7, loc="left", pad=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=5.5, frameon=False, ncol=2)


def panel_mt(ax):
    s = pd.read_csv(OUT / "mtdna_species.csv")
    mk = {"liv_4sp": "o", "liv_hmonkey": "^", "liv_hmouse": "s", "liv_hzeb": "D"}
    for _, r in s.iterrows():
        ax.scatter(r.mean_detection_mt_hep, r.mean_Cs_mt - r.mean_Cs_nonmt, marker=mk[r.cohort], s=18,
                   color=FD.species_color(r.species), edgecolor="white", linewidth=0.4, zorder=3)
    for c, m in mk.items():
        ax.scatter([], [], marker=m, color=MUTED, s=14, label=dict(COH)[c])
    for sp_ in SPECIES:
        ax.scatter([], [], marker="o", color=FD.species_color(sp_), s=14, label=sp_)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_xlabel("mtDNA gene detection in hepatocytes")
    ax.set_ylabel(r"$C_{g,s}$, mtDNA minus other")
    r = spearmanr(s.mean_detection_mt_hep, s.mean_Cs_mt - s.mean_Cs_nonmt)[0]
    FD.annotate_stat(ax, f"Spearman {r:.2f} (n = {len(s)})", "lower right")
    ax.set_title("c  mtDNA margin follows mt capture", fontsize=7, loc="left")
    ax.legend(loc="upper left", fontsize=4.8, frameon=False, ncol=2, handletextpad=0.2, columnspacing=0.6)


def fig_main():
    FD.apply_design()
    fig = plt.figure(figsize=(FD.DOUBLE_W, 4.6))
    ax_h = fig.add_axes([0.17, 0.20, 0.40, 0.70])
    cax = fig.add_axes([0.25, 0.035, 0.25, 0.018])
    panel_heat(ax_h, cax)
    panel_cohorts(fig.add_axes([0.68, 0.60, 0.31, 0.30]))
    panel_mt(fig.add_axes([0.68, 0.10, 0.31, 0.36]))
    save(fig, "fig_main_liver_drivers")


def fig_supp_zonation():
    FD.apply_design()
    ref = pd.read_csv(OUT / "zonation" / "gene_zonation_reference.csv")
    lit = pd.read_csv(OUT / "zonation" / "literature_check.csv")
    fid = pd.read_csv(OUT / "zonation" / "decoder_fidelity.csv")
    lat = pd.read_csv(OUT / "zonation" / "latent_zonation.csv")
    fig, axes = plt.subplots(2, 2, figsize=(FD.DOUBLE_W, 4.6))
    axes = axes.ravel()
    ax = axes[0]
    li = lit[lit.in_vocab == True].drop_duplicates("gene")  # noqa: E712
    for _, r in li.iterrows():
        col = "#0072B2" if r.expected_sign < 0 else ("#D55E00" if r.expected_sign > 0 else MUTED)
        ax.scatter(r.obs_slope_Human, r.obs_slope_Mouse, color=col, s=12, zorder=3)
        ax.text(r.obs_slope_Human + 0.08, r.obs_slope_Mouse + OFFS.get(r.gene, 0), r.gene, fontsize=5, color=TEXT,
                va="center")
    ax.axhline(0, color="#999999", lw=0.5)
    ax.axvline(0, color="#999999", lw=0.5)
    ax.set_xlabel("human PC - PP (log2)")
    ax.set_ylabel("mouse PC - PP (log2)")
    ax.set_title("a  literature genes (blue PP, red PC)", fontsize=6.5, loc="left")
    ax = axes[1]
    both = (ref.det_Human >= 0.1) & (ref.det_Mouse >= 0.1)
    zon = both & (ref.zPC_Human | ref.zPP_Human | ref.zPC_Mouse | ref.zPP_Mouse)
    ax.scatter(ref.loc[both & ~zon, "obs_slope_Human"], ref.loc[both & ~zon, "obs_slope_Mouse"], s=2,
               color="#cccccc", rasterized=True)
    ax.scatter(ref.loc[zon, "obs_slope_Human"], ref.loc[zon, "obs_slope_Mouse"], s=6, color=TEXT)
    con = pd.read_csv(OUT / "zonation" / "conservation.csv").set_index("genes")
    FD.annotate_stat(ax, f"zonated (n={int(con.loc['zonated_either', 'n'])}):\nrho {con.loc['zonated_either', 'spearman_H_M']:.2f}, "
                         f"same sign {con.loc['zonated_either', 'sign_concordance'] * 100:.0f}%", "upper left")
    ax.axhline(0, color="#999999", lw=0.5)
    ax.axvline(0, color="#999999", lw=0.5)
    ax.set_xlabel("human PC - PP (log2)")
    ax.set_ylabel("mouse PC - PP (log2)")
    ax.set_title("b  baseline: zonation agreement", fontsize=6.5, loc="left")
    ax = axes[2]
    f = fid[(fid.genes == "zonated_either") & fid.species.isin(["Human", "Mouse"])]
    for k, (c, lab) in enumerate([("liv_hmouse", "H+Mouse"), ("liv_4sp", "4-sp")]):
        for j, sp_ in enumerate(["Human", "Mouse"]):
            v = f[(f.cohort == c) & (f.species == sp_)].spearman_pred_obs.to_numpy()
            ax.scatter(np.full(len(v), k * 2.5 + j), v, color=FD.species_color(sp_), s=12, zorder=3,
                       label=sp_ if k == 0 else None)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_xticks([0.5, 3.0])
    ax.set_xticklabels(["H+Mouse", "4-sp"])
    ax.set_ylabel("rho(decoder slope, observed)")
    ax.set_title("c  shared decoder vs zonation", fontsize=6.5, loc="left")
    ax.legend(fontsize=5, frameon=False, loc="lower left")
    ax = axes[3]
    for k, (c, lab) in enumerate([("liv_hmouse", "H+Mouse"), ("liv_4sp", "4-sp")]):
        for j, (kind, tr, te, lab2) in enumerate([("cv5", "Human", "Human", "H->H"), ("cv5", "Mouse", "Mouse", "M->M"),
                                                 ("transfer", "Mouse", "Human", "M->H"),
                                                 ("transfer", "Human", "Mouse", "H->M")]):
            v = lat[(lat.cohort == c) & (lat.kind == kind) & (lat.train == tr) & (lat.test == te)].spearman.to_numpy()
            col = FD.species_color(te) if kind == "cv5" else MUTED
            ax.scatter(np.full(len(v), k * 5 + j), v, color=col, s=10, marker="o" if kind == "cv5" else "x", zorder=3)
    ax.set_xticks([0, 1, 2, 3, 5, 6, 7, 8])
    ax.set_xticklabels(["Human CV", "Mouse CV", "Mouse to Human", "Human to Mouse"] * 2, fontsize=5, rotation=60,
                       ha="right", rotation_mode="anchor")
    ax.text(1.5, 0.93, "H+Mouse", ha="center", fontsize=6)
    ax.text(6.5, 0.93, "4-sp", ha="center", fontsize=6)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_ylim(-0.35, 1.0)
    ax.set_ylabel("rho(predicted, zone score)")
    ax.set_title("d  zonation in z: within vs across", fontsize=6.5, loc="left")
    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    save(fig, "fig_supp_liver_zonation")


def fig_supp_presence():
    FD.apply_design()
    p = pd.read_csv(OUT / "presence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(FD.SINGLE_W * 2, 2.0))
    ax = axes[0]
    x = np.arange(len(COH))
    for j, sp_ in enumerate(SPECIES):
        v = [p[(p.cohort == c) & (p.species == sp_)] for c, _ in COH]
        y = [r.frac_zero.iloc[0] * 100 if len(r) else np.nan for r in v]
        ax.bar(x + (j - 1.5) * 0.2, y, width=0.18, color=FD.species_color(sp_), label=sp_)
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in COH], fontsize=6)
    ax.set_ylabel("genes with no count (%)")
    ax.set_title("a  structural zeros per species", fontsize=7, loc="left")
    ax.legend(fontsize=5.5, frameon=False)
    ax = axes[1]
    a = p[p.species == "any"].set_index("cohort").reindex([c for c, _ in COH])
    w = 0.25
    ax.bar(x - w, a.frac_zero * 100, width=w, color="#dddddd", label="all genes")
    ax.bar(x, a.frac_zero_in_D_bottom5 * 100, width=w, color="#444444", label=r"bottom 5 % of $D$")
    ax.bar(x + w, a.frac_zero_in_D_top5 * 100, width=w, color="#999999", label=r"top 5 % of $D$")
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in COH], fontsize=6)
    ax.set_ylabel("zero in any species (%)")
    ax.set_title("b  where structural zeros sit in D", fontsize=7, loc="left")
    ax.legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    save(fig, "fig_supp_liver_presence")


def fig_supp_named():
    FD.apply_design()
    n = pd.read_csv(OUT / "named_genes.csv")
    n = n[n.in_4sp_vocab == True].copy()  # noqa: E712
    order = {"confident": 0, "caveated": 1, "do-not-highlight": 2}
    n["o"] = n["class"].map(order)
    n = n.sort_values(["o", "D"])
    col = {"confident": "#222222", "caveated": "#0072B2", "do-not-highlight": "#bbbbbb"}
    fig, ax = plt.subplots(figsize=(FD.DOUBLE_W, 2.3))
    x = np.arange(len(n))
    for cl, c in col.items():
        m = (n["class"] == cl).to_numpy()
        ax.errorbar(x[m], n.D[m], yerr=n.sd_delta_alpha[m].fillna(0), fmt="o", ms=3.5, color=c, lw=0.6,
                    label=cl)
    for c, mk_ in (("liv_hmonkey", "^"), ("liv_hmouse", "s"), ("liv_hzeb", "D")):
        ax.scatter(x, n[f"D_{c}"], marker=mk_, s=6, facecolor="white", edgecolor=MUTED, lw=0.5, zorder=1,
                   label=dict(COH)[c])
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(n.symbol, fontsize=4.8, rotation=90)
    ax.set_ylabel(r"$D$ (4-sp consensus, seed SD)")
    ax.set_title("Named figure genes: 4-sp consensus D, class by the fixed robustness rule; open marks: two-species cohorts",
                 fontsize=6.5, loc="left")
    ax.legend(fontsize=5, frameon=False, ncol=6, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_supp_liver_named_genes")


def main():
    fig_main()
    fig_supp_zonation()
    fig_supp_presence()
    fig_supp_named()


if __name__ == "__main__":
    main()
