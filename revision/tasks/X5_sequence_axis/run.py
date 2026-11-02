"""Sequence measures against Chimera's readouts on the same genes, in the style of the prior-vs-alpha panels.

Rows (same genes in each cohort): dN/dS vs identity; alpha vs identity; Delta-alpha vs identity;
Delta-alpha vs dN/dS. Identity is the raw prior; alpha and Delta-alpha are seed consensus values.
Delta-alpha is D (median-centred, against the prior the run anchors on). Blue genes have count
evidence (S_g nonzero), grey genes are masked and their Delta-alpha is not a count call.

Usage: python revision/tasks/X5_sequence_axis/run.py
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.common import gene_ids

sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
import _figure_design as fd  # noqa: E402

RES = gene_ids.REPO_ROOT / "results" / "revision"
OUT = RES / "X5"
COHORTS = [("liv_hmouse", "Liver, Human+Mouse"), ("liv_4sp", "Liver, 4 species"), ("test_hmouse", "Testis, Human+Mouse")]
BLUE, GREY = "#2b6ca3", "#9a9a9a"
LIMS = {"liv_hmouse": (-0.32, 0.22), "liv_4sp": (-0.42, 0.22), "test_hmouse": (-0.05, 0.12)}


def trend(ax, x, y, bins=14):
    q = pd.qcut(x, bins, duplicates="drop")
    m = pd.DataFrame({"x": x, "y": y}).groupby(q, observed=True).median()
    ax.plot(m.x, m.y, color="#c8501e", lw=1.3)


def cloud(ax, x, y, supported):
    ax.scatter(x[~supported], y[~supported], s=1.2, alpha=0.18, color=GREY, linewidths=0, rasterized=True)
    ax.scatter(x[supported], y[supported], s=1.2, alpha=0.22, color=BLUE, linewidths=0, rasterized=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fd.apply_design()
    gold = pd.read_csv(RES / "T4_3/golds/gold_table.csv", index_col=0)[["dNdS"]]
    fig, axes = plt.subplots(4, 3, figsize=(fd.DOUBLE_W, 7.4), constrained_layout=True)
    rows = []
    for j, (c, title) in enumerate(COHORTS):
        d = pd.read_csv(RES / f"T3_3/{c}__consensus.csv").set_index("hsapiens_gene_id")
        d = d[~d.index.duplicated()].join(gold).dropna(subset=["dNdS", "D", "alpha", "prior_raw"])
        d = d[d.dNdS > 0]
        pi = d.prior_raw.to_numpy()
        logw = np.log10(d.dNdS.to_numpy().clip(1e-3))
        a, D = d.alpha.to_numpy(), d.D.to_numpy()
        sup = (d.S_ema.abs() > 1e-6).to_numpy()
        panels = [
            (pi, logw, "protein identity $\\pi$", "dN/dS (log10)", "pi_vs_dNdS"),
            (pi, a, "protein identity $\\pi$", "$\\alpha$", "pi_vs_alpha"),
            (pi, D, "protein identity $\\pi$", "$\\Delta\\alpha$", "pi_vs_D"),
            (logw, D, "dN/dS (log10)", "$\\Delta\\alpha$", "dNdS_vs_D"),
        ]
        for i, (x, y, xl, yl, name) in enumerate(panels):
            ax = axes[i, j]
            cloud(ax, x, y, sup)
            if name == "pi_vs_alpha":
                ax.plot([0.4, 1], [0.4, 1], color="#333333", lw=0.8, ls="--")
            else:
                trend(ax, x, y)
            if name != "pi_vs_dNdS" and name != "pi_vs_alpha":
                ax.axhline(0, color="#333333", lw=0.6, ls=":")
                ax.set_ylim(*LIMS[c])
            if x is pi:
                ax.set_xlim(0.4, 1.0)
            r, rs = spearmanr(x, y)[0], spearmanr(x[sup], y[sup])[0]
            fd.annotate_stat(ax, f"$\\rho = {r:+.2f}$")
            ax.set_xlabel(xl)
            if j == 0:
                ax.set_ylabel(yl)
            if i == 0:
                ax.set_title(title)
            rows.append(dict(cohort=c, panel=name, n=len(x), rho=r, n_count_evidence=int(sup.sum()), rho_count_evidence=rs))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "sequence_axis_rho.csv", index=False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_sequence_axis.{ext}", dpi=300)
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
