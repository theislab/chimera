"""Fig 3 with the added cohorts (bone marrow, kidney, islet, eye H+Mouse) beside the original seven. Same three panels, same scoring and scaling as ``fig3_integration``; scaling is per cohort, so the seven original columns are unchanged.
Source: results/analysis/B2/integration/master_table_extended.csv. A cohort joins by a row in B2_benchmark/integration/rules.py and a rerun of its score.py. Integration is the substrate of Chimera's readouts, not its aim.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.tasks.T8_4_figures import common as cm
from analysis.tasks.T8_4_figures import fig3_integration as F3

SRC = cm.REPO / "results" / "analysis" / "B2" / "integration" / "master_table_extended.csv"
ADDED = [("bone_marrow", "bone marrow, 3 species"), ("kidney", "kidney, human + mouse"), ("islet", "islet, human + mouse"), ("vz_hmouse", "eye, human + mouse")]


def heat(M: pd.DataFrame, label: str, show_rows: bool, xlabels: list[str], n_orig: int) -> plt.Figure:
    layout, names = [], []
    for tier, ms in F3.TIERS:
        layout.append(None)
        names.append("")
        for m in ms:
            layout.append(m)
            names.append(F3.COL_LABEL.get(m, m))
    A = np.full((len(layout), M.shape[0]), np.nan)
    for i, m in enumerate(layout):
        if m is not None:
            A[i] = M[m].to_numpy()
    left = 1.24 if show_rows else 0.0
    mw, right = 1.75 * M.shape[0] / 7, 0.02
    w, h = left + mw + right, 3.72
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([left / w, 1.42 / h, mw / w, 2.28 / h])
    cax = fig.add_axes([(left + 0.05) / w, 0.3 / h, (1.75 - 0.1) / w, 0.06 / h])
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    best = {j: int(np.nanargmax(A[:, j])) if np.isfinite(A[:, j]).any() else -1 for j in range(A.shape[1])}
    for i, m in enumerate(layout):
        if m is None:
            continue
        for j in range(A.shape[1]):
            v = A[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if v > 0.6 else cm.INK, fontweight="bold" if best[j] == i else "normal")
            if m == "Chimera" and j >= 4:  # labels through conditional MMD everywhere but liver
                ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False, ec=cm.INK, lw=0.6, ls=(0, (1.5, 1))))
    if show_rows:
        k = 0
        for tier, ms in F3.TIERS:
            tier = "no labels on liver; labels elsewhere (dashed)" if ms == ["Chimera"] else tier
            ax.text(-0.45, k + 0.1, tier, ha="left", va="center", fontsize=5.3, style="italic", color=cm.MUTED)
            k += len(ms) + 1
    ax.axvline(3.5, color="white", lw=2.0)
    ax.axvline(n_orig - 0.5, color="white", lw=3.0)  # original seven | added cohorts
    ax.set_xticks(range(M.shape[0]))
    ax.set_xticklabels(xlabels, rotation=90, ha="center", va="top", fontsize=5.5)
    ax.set_yticks([i for i, m in enumerate(layout) if m is not None])
    ax.set_yticklabels([n for n, m in zip(names, layout) if m is not None] if show_rows else [], fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"{label}\n(scaled within cohort, 1 = best)", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(0.4)
    return fig


def main() -> list:
    cm.setup()
    T = pd.read_csv(SRC)
    rows = [d for d, _ in F3.ROWS] + [d for d, _ in ADDED if d in set(T.dataset)]
    lab = [cm.COHORT[c] for _, c in F3.ROWS] + [n for d, n in ADDED if d in set(T.dataset)]
    out = []
    for k, (col, label) in enumerate(F3.SPECS):
        M = T.pivot(index="dataset", columns="model", values=col).reindex(index=rows, columns=F3.COLS)
        fig = heat(M, label, k in (0, 2), lab, len(F3.ROWS))
        out.append(cm.save(fig, f"fig3{'abc'[k]}_ext_integration_{col.replace('total_score', 'composite')}", exact=True))
    return out


if __name__ == "__main__":
    for p in main():
        print(p, cm.check_pdf(p))
