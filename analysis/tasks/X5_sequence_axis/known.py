"""Chimera's readouts among known gene measures and gene groups, on the same genes.

Row 1: Spearman matrix of sequence, constraint, expression-evolution and expression measures with
Delta-alpha and S_g. Row 2: known gene groups on the identity-Delta-alpha plane (median, interquartile
bars). Rows 3-5: Delta-alpha against LOEUF, EVEE and profile correlation. Delta-alpha is D; blue genes
have count evidence, grey genes are masked.

Usage: python analysis/tasks/X5_sequence_axis/known.py         # the two tables (reads the count matrices), then the figure
       python -c "from analysis.tasks.X5_sequence_axis import known; known.draw_panels()"      # the figure alone, from the cached tables
The figure is one panel, figures/panels/supp_known_measures/known_measures.pdf, drawn at its size on the page (house style: figures/style.py).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.tasks.X5_sequence_axis.run import RASTER_DPI, beyond, key_handles, note, signed
from figures import style

RES = gene_ids.REPO_ROOT / "results" / "analysis"
OUT = RES / "X5"
COHORTS = [(c, style.cohort_name(c)) for c in ("liv_hmouse", "liv_4sp", "test_hmouse")]
# Measures of the correlation maps, in the order of their rows: key in the tables, name on the figure. The last two are Chimera's readouts.
ORDER = [("pi", "protein identity π"), ("logw", "log₁₀ dN/dS"), ("LOEUF", "LOEUF"), ("esm", "ESM-2 distance"),
         ("evee", "EVEE (−log variance)"), ("B1b", "profile correlation"), ("B2b", "specificity similarity"),
         ("expr", "expression level"), ("tau", "cell-type specificity τ"), ("len", "coding length"),
         ("D", "Δα"), ("S", "count evidence $S_g$")]
READOUT_SHORT = {"D": "Δα", "S": "$S_g$"}  # headings of the two columns that carry the values
# Gene groups: key, label in the table known_groups.csv.
GROUPS = [("C1", "housekeeping"), ("C2", "core machinery"), ("mt", "mtDNA"), ("TF", "TFs"), ("V2", "interferon, inflammatory"), ("V3", "extracellular"),
          ("V1", "xenobiotic, bile acid")]

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this figure
BLUE, GREY, TREND = style.COUNT_EVIDENCE, style.IMPUTED, style.TREND  # genes with count evidence, genes without it, median trend
INK, MUTED, REF_LINE = style.INK, style.MUTED, "#888888"  # REF_LINE: zero lines
S_TITLE, S_TICK, S_TEXT = style.SIZE_TITLE, style.SIZE_TICK, style.SIZE_TEXT
NA_CELL = "#D9D9D9"  # map cell without a value: darker than the center of the diverging map, so it cannot be read as a value near zero
# Gene groups on the figure: color, marker, name.
GROUP_MARK = {"C1": style.Mark(style.GENE_THEME["housekeeping"], "o", "housekeeping"), "C2": style.Mark("#CC79A7", "s", "core machinery"),
              "mt": style.Mark(style.GENE_THEME["mitochondrially encoded"], "D", "mitochondrially encoded"), "TF": style.Mark(INK, "^", "transcription factors"),
              "V2": style.Mark("#E69F00", "v", "interferon, inflammatory"), "V3": style.Mark("#009E73", "P", "extracellular"),
              "V1": style.Mark("#D55E00", "X", "xenobiotic, bile acid")}


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


def annotations() -> tuple:
    """The table of gene annotations (dN/dS, LOEUF, EVEE, gene lists) and the set of human transcription factors."""
    gold = pd.read_csv(RES / "T4_3/golds/gold_table.csv", index_col=0)
    lam = pd.read_csv(gene_ids.REPO_ROOT / "data/analysis/x3_annotations/lambert2018_DatabaseExtract_v_1.01.csv")
    idc = [c for c in lam.columns if "Ensembl" in c][0]
    tfc = [c for c in lam.columns if c.replace(" ", "").lower() in ("istf?", "istf")][0]
    tfs = set(lam.loc[lam[tfc].astype(str).str.lower().str.startswith("yes"), idc])
    return gold, tfs


def gene_table(c, gold, tfs):
    """The genes of one cohort for the gene clouds, from cached tables only (consensus, annotations, baselines): the columns of `load` that do
    not need the count matrix."""
    d = pd.read_csv(RES / f"T3_3/{c}__consensus.csv").set_index("hsapiens_gene_id")
    d = d[~d.index.duplicated()].join(gold)
    b = pd.read_csv(RES / f"T4_3/baselines_consensusD/baselines__{c}.csv").set_index("hsapiens_gene_id")
    d = d.join(b[[x for x in ("B1b_raw", "B2b") if x in b]]).rename(columns={"B1b_raw": "B1b"})
    d["pi"], d["S"] = d.prior_raw, d.S_ema
    d["evee"] = d.G_H2 if c.startswith("test") else d.G_H1
    d["mt"] = d.symbol.astype(str).str.startswith("MT-")
    d["TF"] = d.index.isin(tfs)
    return d


def group(d, k):
    """The genes of one group that have identity and Δα; None for a group of fewer than eight genes."""
    g = d[d[k].astype(bool)].dropna(subset=["pi", "D"])
    return g if len(g) >= 8 else None


def cell_value(v: float) -> str:
    """A correlation in a map cell: one decimal, a true minus sign, no sign on a value that rounds to zero."""
    r = round(float(v), 1)
    return "0.0" if r == 0 else f"{r:.1f}".replace("-", "−")


def draw_panels() -> None:
    """The figure, from cached tables only: the correlation maps from results/analysis/X5/known_measures_rho.csv, the gene clouds and gene
    groups from the consensus, annotation and baseline tables. No count matrix is read."""
    style.apply()
    gold, tfs = annotations()
    rho = pd.read_csv(OUT / "known_measures_rho.csv")
    keys = [k for k, _ in ORDER]
    names = dict(ORDER)
    known, read = [k for k in keys if k not in READOUT_SHORT], [k for k in keys if k in READOUT_SHORT]
    # page and maps, in bp
    W, cw, sw, cgap, map_gap, map_left = style.TEXTWIDTH, 8.7, 16.5, 2.5, 7.0, 70.0
    map_w, map_h = len(known) * cw + cgap + len(read) * sw, len(keys) * cw + cgap
    left, right, gap_x, row_gap, ax_h2, ax_h, bottom = 40.0, 8.0, 28.0, 23.0, 72.0, 56.0, 43.0
    ax_w = (W - left - right - 2 * gap_x) / 3
    y_rows = [bottom + k * (ax_h + row_gap) for k in (2, 1, 0)]  # the three rows of clouds against one measure, top to bottom
    y_groups = y_rows[0] + ax_h + row_gap  # the row of gene groups
    y_maps = y_groups + ax_h2 + 42.0
    H = y_maps + map_h + 13.0
    fig = plt.figure(figsize=(style.inch(W), style.inch(H)))
    xe = np.r_[np.arange(len(known) + 1) * cw, len(known) * cw + cgap + np.arange(len(read) + 1) * sw]  # cell edges; one narrow empty column and row set the readouts apart
    ye = np.r_[np.arange(len(known) + 1) * cw, len(known) * cw + cgap + np.arange(len(read) + 1) * cw]
    idx = [i for i in range(len(keys) + 1) if i != len(known)]
    xc, yc = (xe[:-1] + xe[1:])[idx] / 2, (ye[:-1] + ye[1:])[idx] / 2
    for j, (c, title) in enumerate(COHORTS):
        d = gene_table(c, gold, tfs)
        # correlation map: every measure keeps its row and column in every cohort; a measure without values is drawn as empty cells
        M = rho[rho.cohort == c].pivot(index="a", columns="b", values="rho").reindex(index=keys, columns=keys).to_numpy(float)
        A = np.full((len(keys) + 1, len(keys) + 1), np.nan)
        A[np.ix_(idx, idx)] = M
        E = np.full_like(A, np.nan)
        E[np.ix_(idx, idx)] = np.where(np.isnan(M), 1.0, np.nan)
        ax = fig.add_axes([(map_left + j * (map_w + map_gap)) / W, y_maps / H, map_w / W, map_h / H])
        ax.pcolormesh(xe, ye, np.ma.masked_invalid(E), cmap=ListedColormap([NA_CELL]), vmin=0, vmax=2, edgecolors="white", linewidth=0.3)
        im = ax.pcolormesh(xe, ye, np.ma.masked_invalid(A), cmap=style.DIVERGING, vmin=-1, vmax=1, edgecolors="white", linewidth=0.3)
        ax.set_xlim(0, map_w)
        ax.set_ylim(map_h, 0)
        for a_, ka in enumerate(keys):
            for kb in read:
                b_ = keys.index(kb)
                if ka != kb and np.isfinite(M[a_, b_]):
                    ax.text(xc[b_], yc[a_], cell_value(M[a_, b_]), ha="center", va="center", fontsize=S_TEXT, color="white" if abs(M[a_, b_]) > 0.55 else INK)
        ax.set_yticks(yc)
        ax.set_yticklabels([names[k] for k in keys] if j == 0 else [""] * len(keys), fontsize=S_TICK)
        ax.set_xticks([xc[keys.index(k)] for k in read])
        ax.set_xticklabels([READOUT_SHORT[k] for k in read], fontsize=S_TICK)
        ax.tick_params(length=0, pad=2)
        ax.text(len(known) * cw / 2, map_h + 2.5, "columns in the order of the rows", ha="center", va="top", fontsize=S_TEXT, color=MUTED)
        ax.set_title(title, fontsize=S_TITLE, pad=3)
        for s in ax.spines.values():
            s.set_visible(False)
        if j == len(COHORTS) - 1:
            cax = fig.add_axes([(map_left + (j + 1) * (map_w + map_gap) - map_gap + 5) / W, (y_maps + 0.22 * map_h) / H, 5.5 / W, 0.7 * map_h / H])
            cb = fig.colorbar(im, cax=cax)
            cb.set_ticks([-1, -0.5, 0, 0.5, 1])
            cb.ax.tick_params(labelsize=S_TICK, length=2)
            cb.outline.set_linewidth(0.4)
            cax.set_title("ρ", fontsize=S_TICK, pad=3)
            fig.text((map_left + (j + 1) * (map_w + map_gap) - map_gap + 3) / W, (y_maps + 0.01 * map_h) / H, "gray:\nno value", fontsize=S_TEXT, color=MUTED, ha="left", va="bottom",
                     linespacing=1.1)
        # gene groups on the plane of identity and Δα
        sup = (d.S.abs() > 1e-6).to_numpy()
        ax = fig.add_axes([(left + j * (ax_w + gap_x)) / W, y_groups / H, ax_w / W, ax_h2 / H])
        ax.scatter(d.pi[~sup], d.D[~sup], s=1, alpha=0.12, color=GREY, linewidths=0, rasterized=True)
        ax.scatter(d.pi[sup], d.D[sup], s=1, alpha=0.12, color=BLUE, linewidths=0, rasterized=True)
        ax.axhline(0, color=REF_LINE, lw=0.6, ls=":")
        q = d.D.quantile([.002, .998]); low = min(q.iloc[0], d.D[d.mt].quantile(.1) - 0.02); high = max(q.iloc[1], d.D[d.mt].quantile(.9) + 0.02)
        ax.set_ylim(low, high); ax.set_xlim(0.5, 1.0)
        beyond(ax, d.pi, d.D, sup)
        for k, _ in GROUPS:
            g = group(d, k)
            if g is None:
                continue
            mk = GROUP_MARK[k]
            x, y = g.pi.median(), g.D.median()
            ax.errorbar(x, y, xerr=[[x - g.pi.quantile(.25)], [g.pi.quantile(.75) - x]],
                        yerr=[[y - g.D.quantile(.25)], [g.D.quantile(.75) - y]], fmt=mk.marker, ms=3.4, color=mk.color, mec="white", mew=0.4, lw=0.8, zorder=6)
            if k == "mt":  # named beside its mark, on the side away from the other groups where the axes leave room
                below = y < 0 and (g.D.quantile(.25) - low) / (high - low) * ax_h2 > 10
                ax.annotate(mk.name, (x, g.D.quantile(.25 if below else .75)), xytext=(0, -3 if below else 3), textcoords="offset points", ha="center",
                            va="top" if below else "bottom", fontsize=S_TEXT, color=mk.color, zorder=7, bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
        ax.yaxis.set_major_formatter(FuncFormatter(signed))
        ax.set_xlabel(names["pi"], labelpad=2)
        if j == 0:
            ax.set_ylabel("Δα")
        # Δα against one measure
        for i, k in enumerate(("LOEUF", "evee", "B1b")):
            ax = fig.add_axes([(left + j * (ax_w + gap_x)) / W, y_rows[i] / H, ax_w / W, ax_h / H])
            s = d.dropna(subset=[k, "D"]); m = (s.S.abs() > 1e-6).to_numpy()
            ax.scatter(s[k][~m], s.D[~m], s=1.1, alpha=0.18, color=GREY, linewidths=0, rasterized=True)
            ax.scatter(s[k][m], s.D[m], s=1.1, alpha=0.2, color=BLUE, linewidths=0, rasterized=True)
            t = s.groupby(pd.qcut(s[k], 14, duplicates="drop"), observed=True)[[k, "D"]].median()
            ax.plot(t[k], t.D, color=TREND, lw=1.1)
            ax.axhline(0, color=REF_LINE, lw=0.6, ls=":")
            ax.set_ylim(*d.D.quantile([.002, .998]))
            beyond(ax, s[k], s.D, m)
            note(ax, f"ρ = {spearmanr(s[k], s.D)[0]:+.2f}".replace("-", "−"))
            ax.yaxis.set_major_formatter(FuncFormatter(signed))
            ax.set_xlabel(names[k], labelpad=2)
            if j == 0:
                ax.set_ylabel("Δα")
    hg = [Line2D([], [], marker=GROUP_MARK[k].marker, ls="-", lw=0.8, color=GROUP_MARK[k].color, mec="white", mew=0.4, ms=3.6, label=GROUP_MARK[k].name) for k, _ in GROUPS]
    fig.legend(handles=hg, loc="lower center", bbox_to_anchor=(0.5, (y_groups + ax_h2 + 5) / H), ncol=4, handletextpad=0.5, columnspacing=1.6, borderaxespad=0, labelspacing=0.3,
               title="gene groups: median and interquartile range", title_fontsize=S_TICK)
    fig.legend(handles=key_handles(), loc="lower center", bbox_to_anchor=(0.5, 2 / H), ncol=4, handletextpad=0.5, columnspacing=1.4, borderaxespad=0, labelspacing=0.3)
    with plt.rc_context({"savefig.dpi": RASTER_DPI}):
        style.save_panel(fig, "supp_known_measures/known_measures", exact=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gold, tfs = annotations()
    rows, grp = [], []
    for c, _ in COHORTS:
        d = load(c, gold, tfs)
        keys = [k for k, _ in ORDER if d[k].notna().sum() > 100]
        M = d[keys].corr(method="spearman", min_periods=200).to_numpy()
        for a_, ka in enumerate(keys):
            for b_, kb in enumerate(keys):
                rows.append(dict(cohort=c, a=ka, b=kb, rho=M[a_, b_], n=int(d[[ka, kb]].dropna().shape[0])))
        for k, lab in GROUPS:
            g = group(d, k)
            if g is None:
                continue
            x, y = g.pi.median(), g.D.median()
            grp.append(dict(cohort=c, group=k, label=lab, n=len(g), median_pi=x, median_D=y,
                            q25_D=g.D.quantile(.25), q75_D=g.D.quantile(.75)))
    pd.DataFrame(rows).to_csv(OUT / "known_measures_rho.csv", index=False)
    pd.DataFrame(grp).to_csv(OUT / "known_groups.csv", index=False)
    draw_panels()
    t = pd.DataFrame(rows)
    p = t[t.b.isin(["D", "S"]) & ~t.a.isin(["D"])].pivot_table(index=["cohort", "a"], columns="b", values="rho").round(2)
    print(p.to_string())
    print(pd.DataFrame(grp).round(3).to_string())


if __name__ == "__main__":
    main()
