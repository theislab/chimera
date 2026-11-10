"""Mean Δα per gene program in two fits, redrawn at print size: the eleven panels of ``supp_alpha_robustness``.

    supp_alpha_robustness/seeds_42_44.pdf ... tissues_hmouse.pdf     one scatter per pair of fits (table ``PANELS``)

Each point is one gene program: the mean Δα (α minus prior) of its genes in the fit on the x axis and in the fit on the
y axis, over the genes the two fits share. Programs, the genes they hold and the rule that keeps a program (five shared
genes or more, "other" left out) follow ``category_means_aligned`` of ``notebooks/_alpha_master_builder.py``; Spearman ρ
and Pearson r are computed over all programs of a panel, as there.

Every program is drawn. Most programs lie in a narrow window around zero and one or a few lie far outside it, so each
axis is linear inside the window of its row and compressed by ``SQUEEZE`` outside it; the outer part is shaded, a
dotted line marks where it starts, and its ticks carry the true values. Both axes of a panel use the same scale, so
the identity line stays straight. Programs in the outer part are named.

Inputs: ``final_state.npz`` (α and prior) of the nine stored runs; the gene names of the processed data objects; the
ortholog tables (through ``notebooks/_paper_utils.py``) and the program vocabulary ``notebooks/_gene_categories.py``.
All panels have one page size and one axes position, so they are composed with ``X(...)``.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m figures.redraw.robustness [panel ...]
"""
from __future__ import annotations

import sys
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures import style
from figures.redraw import _common as C

FIGURE = "supp_alpha_robustness"
SQUEEZE = 5.0            # an axis is compressed by this factor outside the window of its row
MIN_GENES = 5            # a program is shown when the two fits share at least this many of its genes

# Fits: key -> (run folder, data object, tissue of the ortholog table, name on the axis).
SEED_RUN = {42: C.RUN["liver_hmonkey"], 43: "iter90-150ep-s43", 44: "iter90-150ep-s44"}
FIT = {f"liver_hmonkey_s{s}": (r, C.ADATA["liver_hmonkey"], "pancreas", f"{C.COHORT['liver_hmonkey']}, seed {s}") for s, r in SEED_RUN.items()}
FIT.update({k: (C.RUN[k], C.ADATA[k], "pancreas" if k.startswith("liver") else "testis", C.COHORT[k]) for k in C.RUN})

# Rows of the figure: window in which the axes are linear, limits of the axes, ticks (all in units of Δα).
ROW = {
    "seeds": dict(window=(-0.085, 0.075), limits=(-0.20, 0.075), ticks=(-0.2, -0.05, 0, 0.05)),
    "liver": dict(window=(-0.145, 0.075), limits=(-0.45, 0.075), ticks=(-0.4, -0.1, 0)),
    "testis": dict(window=(-0.04, 0.06), limits=(-0.04, 0.14), ticks=(-0.04, 0, 0.04, 0.12)),
    "tissues": dict(window=(-0.10, 0.075), limits=(-0.28, 0.17), ticks=(-0.2, -0.05, 0, 0.05, 0.15)),
}
# Panels in the order of the figure: name -> (row, fit on x, fit on y, whether the number of shared genes is printed).
PANELS = {
    "seeds_42_44": ("seeds", "liver_hmonkey_s42", "liver_hmonkey_s44", False),
    "seeds_42_43": ("seeds", "liver_hmonkey_s42", "liver_hmonkey_s43", False),
    "seeds_43_44": ("seeds", "liver_hmonkey_s43", "liver_hmonkey_s44", False),
    "liver_4sp_hmouse": ("liver", "liver_4sp", "liver_hmouse", True),
    "liver_4sp_hmonkey": ("liver", "liver_4sp", "liver_hmonkey", True),
    "liver_4sp_hzeb": ("liver", "liver_4sp", "liver_hzeb", True),
    "testis_3sp_hmac": ("testis", "testis_3sp", "testis_hmac", True),
    "testis_3sp_hmouse": ("testis", "testis_3sp", "testis_hmouse", True),
    "testis_hmac_hmouse": ("testis", "testis_hmac", "testis_hmouse", True),
    "tissues_hmonkey_hmac": ("tissues", "liver_hmonkey", "testis_hmac", True),
    "tissues_hmouse": ("tissues", "liver_hmouse", "testis_hmouse", True),
}

# Page of one panel, bp: three panels and two gaps fill the text width; the axes are square.
PAGE_W, PAGE_H = (style.TEXTWIDTH - 2 * style.GAP) / 3, 139.0
AX_LEFT, AX_BOTTOM, AX_SIDE = 42.0, 30.0, 90.0
OUTER = "#F2F2F2"        # shade of the compressed part of the axes


def minus(text: str) -> str:
    return text.replace("-", "−")


@lru_cache(maxsize=None)
def gene_names(adata_file: str) -> np.ndarray:
    import anndata as ad

    path = C.DATA / adata_file
    C.need(path)
    a = ad.read_h5ad(path, backed="r")
    names = np.asarray(a.var_names)
    a.file.close()
    return names


@lru_cache(maxsize=None)
def delta_alpha(fit: str) -> np.ndarray:
    path = C.RUNS / FIT[fit][0] / "final_state.npz"
    C.need(path)
    with np.load(path, allow_pickle=False) as f:
        return f["alpha"] - f["prior"]


def program_means(fit_x: str, fit_y: str) -> tuple:
    """(number of shared genes, table of the programs kept: mean Δα in the two fits and number of genes)."""
    C.design()                                   # puts the notebooks folder on the path
    import _gene_categories as gc                # the program vocabulary
    import _paper_utils as pu                    # ortholog tables and gene names

    names_x, names_y = gene_names(FIT[fit_x][1]), gene_names(FIT[fit_y][1])
    table_x, table_y = pu.cohort_for(FIT[fit_x][2]), pu.cohort_for(FIT[fit_y][2])
    ix, iy = pu.shared_genes(names_x, table_x, names_y, table_y)
    shared = names_x[ix]
    symbols = pu.annotate_genes_with_names(np.asarray(shared), table_x)
    programs = gc.annotate_extended(np.asarray(shared), symbols)["category"].values
    df = pd.DataFrame({"program": programs, "x": delta_alpha(fit_x)[ix], "y": delta_alpha(fit_y)[iy]})
    sizes = df.groupby("program", observed=True).size()
    keep = [p for p in gc.CATEGORY_ORDER_EXTENDED if p != "other" and sizes.get(p, 0) >= MIN_GENES]
    means = df.groupby("program", observed=True)[["x", "y"]].mean().loc[keep]
    means["n"] = [int(sizes[p]) for p in keep]
    return len(shared), means


def squeezed(window: tuple) -> tuple:
    """Forward and inverse of the axis scale: the identity inside the window, slope 1 / SQUEEZE outside it."""
    lo, hi = window

    def forward(v):
        v = np.asarray(v, dtype=float)
        return np.where(v < lo, lo + (v - lo) / SQUEEZE, np.where(v > hi, hi + (v - hi) / SQUEEZE, v))

    def inverse(u):
        u = np.asarray(u, dtype=float)
        return np.where(u < lo, lo + (u - lo) * SQUEEZE, np.where(u > hi, hi + (u - hi) * SQUEEZE, u))

    return forward, inverse


def tick_text(v: float) -> str:
    return "0" if v == 0 else minus(f"{v:g}")


def draw(name: str) -> list:
    from scipy import stats

    row, fit_x, fit_y, print_genes = PANELS[name]
    n_shared, means = program_means(fit_x, fit_y)
    window, limits, ticks = ROW[row]["window"], ROW[row]["limits"], ROW[row]["ticks"]
    x, y = means["x"].to_numpy(), means["y"].to_numpy()
    if min(x.min(), y.min()) < limits[0] or max(x.max(), y.max()) > limits[1]:
        raise SystemExit(f"{name}: a program lies outside the axes {limits}: widen ROW['{row}']['limits']")
    rho, r = stats.spearmanr(x, y).correlation, stats.pearsonr(x, y).statistic

    style.apply()
    fig = plt.figure(figsize=(style.inch(PAGE_W), style.inch(PAGE_H)))
    ax = fig.add_axes([AX_LEFT / PAGE_W, AX_BOTTOM / PAGE_H, AX_SIDE / PAGE_W, AX_SIDE / PAGE_H])
    forward, inverse = squeezed(window)
    ax.set_xscale("function", functions=(forward, inverse))
    ax.set_yscale("function", functions=(forward, inverse))
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    # the compressed parts of the axes: shaded, with a dotted line where they start
    for edge, end in zip(window, limits):
        if edge != end:
            ax.axvspan(min(edge, end), max(edge, end), color=OUTER, linewidth=0, zorder=0)
            ax.axhspan(min(edge, end), max(edge, end), color=OUTER, linewidth=0, zorder=0)
            ax.axvline(edge, color=C.RULE, linewidth=0.5, linestyle=":", zorder=1)
            ax.axhline(edge, color=C.RULE, linewidth=0.5, linestyle=":", zorder=1)
    ax.axhline(0, color="#CCCCCC", linewidth=0.4, linestyle="--", zorder=1)
    ax.axvline(0, color="#CCCCCC", linewidth=0.4, linestyle="--", zorder=1)
    ax.plot(limits, limits, color="#888888", linewidth=0.6, linestyle="--", zorder=2)
    ax.scatter(x, y, s=9, color=C.INK, alpha=0.75, linewidths=0, zorder=3)
    for program, px, py in zip(means.index, x, y):      # programs in the compressed part are named
        if not (window[0] <= px <= window[1] and window[0] <= py <= window[1]):
            left = forward(px) < forward(np.mean(limits))
            ax.annotate(C.category(program), (px, py), xytext=(4 if left else -4, 0), textcoords="offset points",
                        ha="left" if left else "right", va="center", fontsize=style.SIZE_TEXT, color=C.INK, zorder=4)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([tick_text(t) for t in ticks])
    ax.set_yticklabels([tick_text(t) for t in ticks])
    ax.minorticks_off()
    ax.tick_params(pad=2)
    ax.set_xlabel(f"{FIT[fit_x][3]}\nmean Δα per program", linespacing=1.15, labelpad=2)
    ax.set_ylabel(f"{FIT[fit_y][3]}\nmean Δα per program", linespacing=1.15, labelpad=2)
    head = f"$k$ = {len(means)} programs" + (f", $n$ = {n_shared:,} shared genes" if print_genes else "")
    stat = minus(f"Spearman $\\rho$ = {rho:.3f}, Pearson $r$ = {r:.3f}")
    fig.text((PAGE_W - 3) / PAGE_W, (AX_BOTTOM + AX_SIDE + 3) / PAGE_H, head + "\n" + stat, ha="right", va="bottom",
             fontsize=style.SIZE_TEXT, color=C.INK, linespacing=1.2)
    return [C.save_panel(fig, FIGURE, name, exact=True)]


def main(names: list) -> list:
    out = []
    for name in names or list(PANELS):
        out += draw(name)
    return out


if __name__ == "__main__":
    for path in main(sys.argv[1:]):
        print(path)
