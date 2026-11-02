"""Chimera's readouts among known gene measures and gene groups, on the same genes.

Row 1: Spearman matrix of sequence, constraint, expression-evolution and expression measures with
Delta-alpha and S_g. Row 2: known gene groups on the identity-Delta-alpha plane (median, interquartile
bars). Rows 3-5: Delta-alpha against LOEUF, EVEE and profile correlation. Delta-alpha is D; blue genes
have count evidence, grey genes are masked.

Usage: python revision/tasks/X5_sequence_axis/known.py
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.common import cohorts as C
from revision.common import controls as K
from revision.common import gene_ids

sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
import _figure_design as fd  # noqa: E402

RES = gene_ids.REPO_ROOT / "results" / "revision"
OUT = RES / "X5"
COHORTS = [("liv_hmouse", "Liver, Human+Mouse"), ("liv_4sp", "Liver, 4 species"), ("test_hmouse", "Testis, Human+Mouse")]
BLUE, GREY = "#2b6ca3", "#9a9a9a"
ORDER = [("pi", "protein identity $\\pi$"), ("logw", "dN/dS (log10)"), ("LOEUF", "LOEUF"), ("esm", "ESM-2 distance"),
         ("evee", "EVEE (−log variance)"), ("B1b", "profile correlation"), ("B2b", "specificity conservation"),
         ("expr", "expression"), ("tau", "cell-type specificity"), ("len", "CDS length"),
         ("D", "$\\Delta\\alpha$"), ("S", "$S_g$")]
GROUPS = [("C1", "housekeeping", "#3a7d44"), ("C2", "core machinery", "#7a4fa3"), ("mt", "mtDNA", "#c8501e"),
          ("TF", "TFs", "#111111"), ("V2", "interferon, inflammatory", "#b8860b"), ("V3", "extracellular", "#2aa0a0"),
          ("V1", "xenobiotic, bile acid", "#c0392b")]


def load(c, gold, tfs):
    d = pd.read_csv(RES / f"T3_3/{c}__consensus.csv")
    a = C.read_adata(c)
    pr = C.raw_prior(c, C.CANONICAL[c], a.var["conservation_prior"].to_numpy(dtype=float))
    cov = K.cohort_covariates(c, adata=a, prior_raw=pr).reset_index(drop=True)
    d["expr"], d["len"] = cov["log_cp10k_spmean"], cov["log_cds_length"]
    d["tau"] = cov["tau"] if "tau" in cov else np.nan
    d["esm"] = np.asarray(a.varm["gene_features"])[:, 64]
    d = d.set_index("hsapiens_gene_id")
    d = d[~d.index.duplicated()].join(gold)
    b = pd.read_csv(RES / f"T4_3/baselines_consensusD/baselines__{c}.csv").set_index("hsapiens_gene_id")
    d = d.join(b[[x for x in ("B1b_raw", "B2b") if x in b]]).rename(columns={"B1b_raw": "B1b"})
    d["pi"], d["S"] = d.prior_raw, d.S_ema
    d["logw"] = np.log10(d.dNdS.where(d.dNdS > 0).clip(lower=1e-3))
    d["evee"] = d.G_H2 if c.startswith("test") else d.G_H1
    d["mt"] = d.symbol.astype(str).str.startswith("MT-")
    d["TF"] = d.index.isin(tfs)
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fd.apply_design()
    gold = pd.read_csv(RES / "T4_3/golds/gold_table.csv", index_col=0)
    lam = pd.read_csv(gene_ids.REPO_ROOT / "data/revision/x3_annotations/lambert2018_DatabaseExtract_v_1.01.csv")
    idc = [c for c in lam.columns if "Ensembl" in c][0]
    tfc = [c for c in lam.columns if c.replace(" ", "").lower() in ("istf?", "istf")][0]
    tfs = set(lam.loc[lam[tfc].astype(str).str.lower().str.startswith("yes"), idc])
    # Fits one page with its caption at scale 1 (F1).
    fig = plt.figure(figsize=(fd.DOUBLE_W, 7.75))
    gs1 = fig.add_gridspec(1, 3, left=0.145, right=0.905, top=0.975, bottom=0.735, wspace=0.12)
    gs2 = fig.add_gridspec(4, 3, left=0.085, right=0.985, top=0.565, bottom=0.05, wspace=0.27, hspace=0.5,
                           height_ratios=[1.3, 1, 1, 1])
    rows, grp = [], []
    for j, (c, title) in enumerate(COHORTS):
        d = load(c, gold, tfs)
        keys = [k for k, _ in ORDER if d[k].notna().sum() > 100]
        names = [dict(ORDER)[k] for k in keys]
        M = d[keys].corr(method="spearman", min_periods=200).to_numpy()
        ax = fig.add_subplot(gs1[0, j])
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(keys)), names, rotation=90, fontsize=5.5)
        ax.set_yticks(range(len(keys)), names if j == 0 else [""] * len(keys), fontsize=5.5)
        for a_ in range(len(keys)):
            for b_ in range(len(keys)):
                if keys[a_] in ("D", "S") or keys[b_] in ("D", "S"):
                    ax.text(b_, a_, f"{M[a_, b_]:.1f}".replace("0.", "."), ha="center", va="center", fontsize=4.4,
                            color="white" if abs(M[a_, b_]) > 0.55 else "#222222")
        ax.set_title(title, fontsize=8)
        ax.grid(False)
        if j == 2:
            cax = fig.add_axes([0.925, 0.77, 0.012, 0.17])
            fig.colorbar(im, cax=cax)
            cax.set_title("$\\rho$", fontsize=7)
            cax.tick_params(labelsize=5.5)
        for s in ax.spines.values():
            s.set_visible(False)
        for a_, ka in enumerate(keys):
            for b_, kb in enumerate(keys):
                rows.append(dict(cohort=c, a=ka, b=kb, rho=M[a_, b_], n=int(d[[ka, kb]].dropna().shape[0])))
        sup = (d.S.abs() > 1e-6).to_numpy()
        ax = fig.add_subplot(gs2[0, j])
        ax.scatter(d.pi[~sup], d.D[~sup], s=1, alpha=0.12, color=GREY, linewidths=0, rasterized=True)
        ax.scatter(d.pi[sup], d.D[sup], s=1, alpha=0.12, color=BLUE, linewidths=0, rasterized=True)
        ax.axhline(0, color="#333333", lw=0.6, ls=":")
        for k, lab, col in GROUPS:
            g = d[d[k].astype(bool)].dropna(subset=["pi", "D"])
            if len(g) < 8:
                continue
            x, y = g.pi.median(), g.D.median()
            ax.errorbar(x, y, xerr=[[x - g.pi.quantile(.25)], [g.pi.quantile(.75) - x]],
                        yerr=[[y - g.D.quantile(.25)], [g.D.quantile(.75) - y]], fmt="o", ms=3.2, color=col, lw=0.8,
                        label=lab, zorder=6)
            grp.append(dict(cohort=c, group=k, label=lab, n=len(g), median_pi=x, median_D=y,
                            q25_D=g.D.quantile(.25), q75_D=g.D.quantile(.75)))
        q = d.D.quantile([.002, .998]); low = min(q.iloc[0], d.D[d.mt].quantile(.1) - 0.02); high = max(q.iloc[1], d.D[d.mt].quantile(.9) + 0.02)
        ax.set_ylim(low, high); ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("protein identity $\\pi$", fontsize=7)
        if j == 0:
            ax.set_ylabel("$\\Delta\\alpha$", fontsize=8)
        if j == 1:
            hl = ax.get_legend_handles_labels()
            fig.legend(*hl, fontsize=6.5, frameon=False, loc="center", bbox_to_anchor=(0.53, 0.59), ncol=7,
                       handletextpad=0.1, columnspacing=0.8)
        for i, (k, lab) in enumerate((("LOEUF", "LOEUF"), ("evee", "EVEE (−log variance)"), ("B1b", "profile correlation"))):
            ax = fig.add_subplot(gs2[1 + i, j])
            s = d.dropna(subset=[k, "D"]); m = (s.S.abs() > 1e-6).to_numpy()
            ax.scatter(s[k][~m], s.D[~m], s=1.1, alpha=0.18, color=GREY, linewidths=0, rasterized=True)
            ax.scatter(s[k][m], s.D[m], s=1.1, alpha=0.2, color=BLUE, linewidths=0, rasterized=True)
            t = s.groupby(pd.qcut(s[k], 14, duplicates="drop"), observed=True)[[k, "D"]].median()
            ax.plot(t[k], t.D, color="#c8501e", lw=1.2)
            ax.axhline(0, color="#333333", lw=0.6, ls=":")
            ax.set_ylim(*d.D.quantile([.002, .998]))
            fd.annotate_stat(ax, f"$\\rho = {spearmanr(s[k], s.D)[0]:+.2f}$")
            ax.set_xlabel(lab, fontsize=7)
            if j == 0:
                ax.set_ylabel("$\\Delta\\alpha$", fontsize=8)
    pd.DataFrame(rows).to_csv(OUT / "known_measures_rho.csv", index=False)
    pd.DataFrame(grp).to_csv(OUT / "known_groups.csv", index=False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_known_measures.{ext}", dpi=300)
    t = pd.DataFrame(rows)
    p = t[t.b.isin(["D", "S"]) & ~t.a.isin(["D"])].pivot_table(index=["cohort", "a"], columns="b", values="rho").round(2)
    print(p.to_string())
    print(pd.DataFrame(grp).round(3).to_string())


if __name__ == "__main__":
    main()
