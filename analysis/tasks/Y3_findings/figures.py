"""Y3 (EXPLORATORY): supplementary panel, two worked genes in the counts.

Log2(1 + CPM) per species and cell type or stage (worked_pseudobulk.csv), one line per species, types ordered by the
human value within each panel. Species colours and markers fixed by notebooks/_figure_design.py. One y scale per panel.
Types without enough cells in a species are gaps. Writes results/analysis/Y3/figures/supp_worked_genes.{pdf,png}.

Usage: PYTHONPATH=$PWD python -m analysis.tasks.Y3_findings.figures
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.common import gene_ids  # noqa: E402

sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
import _figure_design as fd  # noqa: E402

OUT = gene_ids.REPO_ROOT / "results" / "analysis" / "Y3"
MARK = {"Human": "o", "Macaque": "s", "Monkey": "s", "Mouse": "^", "Zebrafish": "D"}
NICE = {"Monkey": "Macaque", "Other_somatic": "other\nsomatic", "fibroblasts": "Fibro.", "Hepatocytes": "Hepato.", "Cholangio": "Cholangio.",
        "Endothelial": "Endo."}


def panel(ax, t, order, species, title, ylab=True):
    for s in species:
        v = t[t.species == s].set_index("type").log2cpm.reindex(order).to_numpy(float)
        ax.plot(range(len(order)), v, color=fd.species_color(s), marker=MARK[s], ms=4.2, lw=1.1, mec="white", mew=0.4, label=NICE.get(s, s))
        i = np.where(np.isfinite(v))[0][-1]
        ax.annotate(NICE.get(s, s), (i, v[i]), xytext=(5, 0), textcoords="offset points", va="center", fontsize=6.5, color="#222222")
    ax.set_xticks(range(len(order)), [NICE.get(k, k) for k in order], fontsize=6.5)
    ax.set_xlim(-0.3, len(order) - 0.3)
    ax.set_title(title, fontsize=7.5, loc="left")
    if ylab:
        ax.set_ylabel(r"log$_2$(1 + CPM)")


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fd.apply_design()
    d = pd.read_csv(OUT / "worked_pseudobulk.csv")
    r = pd.read_csv(OUT / "worked_readouts.csv")
    t_gene = r.loc[r.cohort == "test_3sp", "symbol"].iloc[0]
    l_gene = r.loc[r.cohort == "liv_4sp", "symbol"].iloc[0]
    fig, axs = plt.subplots(1, 2, figsize=(fd.DOUBLE_W, 2.5), constrained_layout=True, gridspec_kw=dict(width_ratios=[6, 4.3]))
    tt = d[d.cohort == "test_3sp"]
    panel(axs[0], tt, ["SG", "SC", "rSD", "eSD", "Sertoli", "Other_somatic"], ["Human", "Macaque", "Mouse"], t_gene)
    ll = d[d.cohort == "liv_4sp"]
    panel(axs[1], ll, ["Hepatocytes", "Cholangio", "Endothelial", "fibroblasts"], ["Human", "Monkey", "Mouse"], l_gene, ylab=False)
    fd.save_figure(fig, "supp_worked_genes", out_dir=OUT / "figures", formats=("pdf", "png"), close=True)


if __name__ == "__main__":
    main()
