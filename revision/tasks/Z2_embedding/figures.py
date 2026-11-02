"""Z2 figure panels, each one PDF at print size, then composed with F1_figures.compose (spec in figure_spec.py; output kept under results/revision/Z2, not in the manuscript).
Okabe-Ito hues in fixed order, a distinct marker per series; the dataviz validator needs node, which this host lacks, so the palette is not validated by script.
python -m revision.tasks.Z2_embedding.figures"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

from revision.tasks.Z2_embedding import rules as RU

OUT = RU.OUT / "figures"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"
CH, SC = "#0072B2", "#D55E00"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "axes.linewidth": 0.5, "axes.edgecolor": MUTED, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
COHORT = {"liv_hmouse": "Liver H+Mouse", "liv_4sp": "Liver 4-sp", "liv_hmonkey": "Liver H+Monkey", "test_hmouse": "Testis H+Mouse", "test_3sp": "Testis 3-sp", "test_hmac": "Testis H+Macaque", "kidney": "Kidney", "islet": "Islet", "bone_marrow": "Bone marrow"}
TASKS = [("liv_hmouse", "hep_zone", "zone"), ("liv_4sp", "hep_zone", "zone"), ("liv_hmonkey", "hep_zone", "zone"), ("kidney", "pt_segment", "PT S1-S3"), ("kidney", "pt_state", "PT state"), ("islet", "beta_stress", "beta UPR"),
         ("bone_marrow", "ery_maturation", "erythroid"), ("bone_marrow", "neu_maturation", "neutrophil"), ("test_hmouse", "stage_SC", "SC stage"), ("test_hmouse", "stage_SG", "SG stage"), ("test_3sp", "stage_SC", "SC stage"), ("test_hmac", "stage_SD", "SD stage")]


def panel_gap():
    w = pd.read_csv(RU.OUT / "within_label.csv")
    w = w[w.metric == "auroc_lr"]
    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    n = 0
    labs = []
    for c, t, name in TASKS:
        r = w[(w.cohort == c) & (w.task == t)]
        if not len(r) or "scVI" not in w or not np.isfinite(r.scVI.iloc[0]):
            continue
        r = r.iloc[0]
        y = -n
        ax.plot([r.Chimera, r.scVI], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle="round")
        ax.plot([r.Chimera_lo, r.Chimera_hi], [y, y], color=CH, lw=0.8, zorder=2)
        ax.plot([r.scVI_lo, r.scVI_hi], [y, y], color=SC, lw=0.8, zorder=2)
        ax.scatter(r.Chimera, y, s=14, color=CH, marker="o", zorder=3, label="Chimera" if n == 0 else None, edgecolor="white", linewidth=0.3)
        ax.scatter(r.scVI, y, s=14, color=SC, marker="s", zorder=3, label="scVI" if n == 0 else None, edgecolor="white", linewidth=0.3)
        labs.append(f"{COHORT[c]}, {name}")
        n += 1
    ax.set_yticks(-np.arange(n), labs)
    ax.set_xlabel("Within-label AUROC, in the model (line: seed range)")
    ax.axvline(0.5, color=GRID, lw=0.6, zorder=0)
    ax.legend(loc="lower left", handletextpad=0.1, fontsize=6)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    fig.savefig(OUT / "z2_a.pdf", bbox_inches="tight")


def panel_spectrum():
    import glob
    p = pd.concat(pd.read_csv(f) for f in glob.glob(str(RU.OUT / "snr_pcs_*.csv")))
    p = p[p.cohort.isin(["liv_hmouse", "test_hmouse"])]
    fig, axs = plt.subplots(1, 2, figsize=(3.0, 1.45), sharey=True)
    for ax, c in zip(axs, ["liv_hmouse", "test_hmouse"]):
        for me, col, mk in (("Chimera", CH, "o"), ("scVI", SC, "s")):
            g = p[(p.cohort == c) & (p.method == me)].groupby("pc").snr.median()
            ax.plot(g.index, g.values, color=col, lw=1, marker=mk, ms=2.8, label=me)
        ax.axhline(1, color=MUTED, lw=0.5, ls=(0, (3, 2)))
        ax.set_yscale("log")
        ax.set_title(COHORT[c], fontsize=6, color=INK, pad=2)
        ax.set_xticks([1, 5, 10])
        ax.set_xlabel("Axis of $z$, ranked")
    axs[0].set_ylabel("Signal / posterior noise")
    axs[0].legend(loc="lower left", handletextpad=0.3, fontsize=6)
    axs[1].text(10, 1.2, "signal = noise", ha="right", va="bottom", fontsize=5.5, color=MUTED)
    fig.savefig(OUT / "z2_b.pdf", bbox_inches="tight")


def panel_seeds():
    t = pd.read_csv(RU.OUT / "perseed_liver.csv")
    fig, ax = plt.subplots(figsize=(1.75, 1.6))
    ax.scatter(t.mixing, t.rho_self, s=12, color=CH, edgecolor="white", linewidth=0.3, zorder=3)
    s51 = t[t.seed == 51].iloc[0]
    ax.annotate("s51", (s51.mixing, s51.rho_self), xytext=(-9, 2), textcoords="offset points", fontsize=5.5, color=INK)
    ax.set_xlabel("Mixing entropy")
    ax.set_ylabel(r"Hepatocyte zone $\rho_{self}$")
    fig.savefig(OUT / "z2_c.pdf", bbox_inches="tight")


GROUPS = {"Mixing terms": ["adv100", "adv50", "adv0", "mmd10", "mmd0", "adv0mmd0", "cmmd", "adv50mmd10", "adv20mmd10", "adv50mmd5", "adv20mmd5"], "KL and prior": ["fb05", "fb2", "novamp", "klw5"], "Size": ["nl20", "nl30", "ep300", "zb0", "res01"], "Input": ["logx", "logcp", "logx_adv0mmd0"]}
GC = {"Mixing terms": ("#D55E00", "o"), "KL and prior": ("#009E73", "^"), "Size": ("#CC79A7", "D"), "Input": ("#E69F00", "v")}


def panel_trade():
    t = pd.read_csv(RU.OUT / "factorial_levels.csv")
    t = t[t.cohort == "liv_hmouse"]
    b = t[t.level == "base"].iloc[0]
    fig, axs = plt.subplots(1, 3, figsize=(5.2, 1.75), sharey=True)
    for ax, x, xl in zip(axs, ["mixing", "rho_DS", "rho_D_consensus"], ["Mixing entropy", r"$\rho(\Delta\alpha, S_{ema})$", r"$\rho(\Delta\alpha$, canonical consensus)"]):
        ax.axvline(b[x], color=GRID, lw=0.7, zorder=0)
        ax.axhline(b.hep_rho_self, color=GRID, lw=0.7, zorder=0)
        ax.scatter(b[x], b.hep_rho_self, s=26, color=CH, marker="*", zorder=4, label="canonical recipe" if ax is axs[0] else None)
        for g, lv in GROUPS.items():
            q = t[t.level.isin(lv)]
            col, mk = GC[g]
            ax.scatter(q[x], q.hep_rho_self, s=11, color=col, marker=mk, edgecolor="white", linewidth=0.3, zorder=3, label=g if ax is axs[0] else None)
        ax.set_xlabel(xl)
    axs[0].set_ylabel(r"Hepatocyte zone $\rho_{self}$")
    axs[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.08), handletextpad=0.1, fontsize=5.5, labelspacing=0.25)
    fig.savefig(OUT / "z2_d.pdf", bbox_inches="tight")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel_gap(); panel_spectrum(); panel_seeds(); panel_trade()


if __name__ == "__main__":
    main()
