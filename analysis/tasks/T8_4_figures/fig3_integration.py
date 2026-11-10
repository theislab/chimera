"""Fig 3 (integration), re-rendered with SATURN, SATURN-Leiden and CAMEX and with label tiers.

Three panels on the same rows and columns (T5.5 main view, coordinator 2026-11-02): species mixing
score, cell-type conservation score, composite (0.4 mixing + 0.6 conservation, as frozen). Columns are
grouped by how each method uses cell-type labels in training. Chimera uses none on liver and
cell-type-conditional MMD on testis; its testis cells carry a dashed outline. Row best in bold.
Source: ``notebooks/cache/baselines_master_ext/master_table_ext.csv`` (T5.5, 10 models, s42).
The raw kNN species purity view stays in the Supplement (T5.5 file ``composite_mixing_bio_10models``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.tasks.T8_4_figures import common as cm

SRC = cm.REPO / "notebooks" / "cache" / "baselines_master_ext" / "master_table_ext.csv"
ROWS = [("panc_hmonkey", "liv_hmonkey"), ("panc_hmouse", "liv_hmouse"), ("panc_hzeb", "liv_hzeb"),
        ("panc_4sp", "liv_4sp"), ("test_hmac", "test_hmac"), ("test_hmouse", "test_hmouse"),
        ("test_3sp", "test_3sp")]
TIERS = [("no cell-type labels", ["PCA", "Harmony", "scVI-sp", "scVI-donor", "sysVI", "CAMEX"]),
         ("within-species clusters", ["SATURN-Leiden"]),
         ("cell-type labels", ["scANVI", "SATURN"]),
         ("no labels on liver; labels on testis (dashed)", ["Chimera"])]
COLS = [m for _, ms in TIERS for m in ms]
COL_LABEL = {"scVI-sp": "scVI (species batch)", "scVI-donor": "scVI (donor batch)",
             "SATURN-Leiden": "SATURN (Leiden clusters)"}
SPECS = [("mixing", "species mixing score"), ("biology", "cell-type conservation score"),
         ("total_score", "composite score")]


def heat(M: pd.DataFrame, label: str, show_rows: bool) -> plt.Figure:
    """Methods as rows, grouped by label use under a header row; cohorts as columns."""
    layout, names = [], []
    for tier, ms in TIERS:
        layout.append(None)
        names.append("")
        for m in ms:
            layout.append(m)
            names.append(COL_LABEL.get(m, m))
    A = np.full((len(layout), M.shape[0]), np.nan)
    for i, m in enumerate(layout):
        if m is not None:
            A[i] = M[m].to_numpy()
    # Fixed axes geometry in inches, so the heatmap has the same size in all three panels and the row
    # fills the text width at scale 1 (F1): 6.75 in = row labels + 3 heatmaps + 2 gaps of the grid.
    left = 1.24 if show_rows else 0.0
    mw, right = 1.75, 0.02
    w, h = left + mw + right, 3.72
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([left / w, 1.42 / h, mw / w, 2.28 / h])
    cax = fig.add_axes([(left + 0.05) / w, 0.3 / h, (mw - 0.1) / w, 0.06 / h])
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    best = {j: int(np.nanargmax(A[:, j])) for j in range(A.shape[1])}
    for i, m in enumerate(layout):
        if m is None:
            continue
        for j in range(A.shape[1]):
            v = A[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if v > 0.6 else cm.INK, fontweight="bold" if best[j] == i else "normal")
            if m == "Chimera" and ROWS[j][1].startswith("test"):
                ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False, ec=cm.INK, lw=0.6,
                                           ls=(0, (1.5, 1))))
    if show_rows:
        k = 0
        for tier, ms in TIERS:
            ax.text(-0.45, k + 0.1, tier, ha="left", va="center", fontsize=5.3, style="italic", color=cm.MUTED)
            k += len(ms) + 1
    ax.axvline(3.5, color="white", lw=2.0)
    ax.set_xticks(range(len(ROWS)))
    ax.set_xticklabels([cm.COHORT[c] for _, c in ROWS], rotation=90, ha="center", va="top", fontsize=5.5)
    ax.set_yticks([i for i, m in enumerate(layout) if m is not None])
    ax.set_yticklabels([n for n, m in zip(names, layout) if m is not None] if show_rows else [], fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"{label}\n(scaled within panel, 1 = best)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(0.4)
    return fig


def main() -> list:
    cm.setup()
    T = pd.read_csv(SRC)
    out = []
    for k, (col, label) in enumerate(SPECS):
        M = T.pivot(index="dataset", columns="model", values=col).loc[[d for d, _ in ROWS], COLS]
        fig = heat(M, label, show_rows=(k == 0))
        out.append(cm.save(fig, f"fig3{'abc'[k]}_integration_{col.replace('total_score', 'composite')}", exact=True))
    return out


if __name__ == "__main__":
    for p in main():
        print(p, cm.check_pdf(p))
