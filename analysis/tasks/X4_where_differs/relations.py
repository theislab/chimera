"""EXPLORATORY addition: how consensus delta-alpha and S_g relate to sequence constraint, expression,
specificity, length and direct baselines. Descriptive; no gold is used; unfavourable relations are shown as they are.

Readouts: D (T3_3_consensus consensus, the scoreboard readout: clamped prior on testis), D_raw (raw prior), S (S_ema).
Relations: dN/dS and LOEUF (human; dN/dS is human-mouse in every cohort), pi_raw, expression (species-balanced
log CP10K), specificity tau (X3_cross_cohort column, one definition in all cohorts), log CDS length, B1b, B2b.
raw = Spearman. adjusted = rho_p of readout and relation given C (pi_raw, expression, detection, CDS length) without
the relation itself; for expression, detection is also left out (they are nearly the same quantity).
CI: block bootstrap on ESM modules where T4_3_scoreboard built them, gene bootstrap elsewhere (ci_kind).
seed_lo, seed_hi: range of the raw Spearman over the cohort's seeds, each seed's delta-alpha or S on its own.

Writes results/analysis/X4/relations.csv and the two panels figures/panels/supp_relations_heatmap/heatmap.pdf and
figures/panels/supp_relations_density/density.pdf, each drawn at its size on the page (house style: figures/style.py).

    python -m analysis.tasks.X4_where_differs.relations                                               # the table, then the two panels
    python -c "from analysis.tasks.X4_where_differs import relations; relations.draw_panels()"        # the two panels alone, from the cached tables
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from analysis.common import controls as K  # noqa: E402
from analysis.tasks.T4_3_scoreboard import inputs as I  # noqa: E402
from analysis.tasks.X4_where_differs import common as CM  # noqa: E402
from figures import style  # noqa: E402

OUT = CM.OUT
COHORTS = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp", "test_hmac", "test_hmouse", "test_3sp"]
COV = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]
SCATTER = ["liv_hmouse", "liv_4sp", "test_hmouse"]

# ---------------------------------------------------------------- house style (figures/style.py) and the names of the two figures
LABEL = {c: style.cohort_name(c) for c in COHORTS}
# The properties, in the order of the rows of the heat maps and of the columns of the gene clouds: key in the tables, name on the figure.
REL = {"dNdS": "dN/dS", "prior_raw": "protein identity π", "LOEUF": "LOEUF", "log_cp10k_spmean": "expression level",
       "tau": "cell-type specificity τ", "log_cds_length": "coding length", "B1b_raw": style.READOUT["profile_correlation"].name,
       "B2b": style.READOUT["specificity_similarity"].name}
# The same properties as axis labels of the gene clouds, with the scale they are drawn on.
REL_AXIS = {"dNdS": "log₁₀ dN/dS", "prior_raw": "protein\nidentity π", "LOEUF": "LOEUF", "log_cp10k_spmean": "expression level\n(log CP10K)",
            "tau": "cell-type\nspecificity τ", "log_cds_length": "coding length\n(log₁₀ bp)", "B1b_raw": "profile\ncorrelation",
            "B2b": "specificity\nsimilarity"}
READ = {"D": "Δα", "D_raw": "Δα on the raw prior", "S": style.READOUT["count_evidence"].name}  # rows of maps
READ_AXIS = {"D": "Δα", "S": "$S_g$"}  # y axes of the gene clouds
INK, MUTED, RULE, TREND = style.INK, style.MUTED, style.RULE, style.TREND
NA_CELL = "#F2F2F2"  # map cell without a value (the property is not defined for the cohort)
RASTER_DPI = 360  # rasterized hexagon maps: above 300 ppi on the page


def signed(v: float) -> str:
    """A correlation with its sign and a true minus sign; a value that rounds to zero carries no sign."""
    return "0.00" if abs(v) < 0.005 else f"{v:+.2f}".replace("-", "−")


def covs_for(v):
    c = [x for x in COV if x != v]
    if v == "log_cp10k_spmean":
        c = [x for x in c if x != "detection_spmean"]
    return c


def gene_table(c):
    """The genes of a cohort with the readouts and the properties they are related to, from the cached tables."""
    t = pd.read_csv(CM.RES / f"T3_3/{c}__consensus.csv")
    x = pd.read_parquet(CM.RES / f"X3/genes/{c}.parquet")
    if not (t.var_name.to_numpy() == x.var_name.to_numpy()).all():
        raise AssertionError(c)
    g = pd.DataFrame(dict(var_name=t.var_name, hsapiens_gene_id=t.hsapiens_gene_id, symbol=t.symbol, D=t.D, S=t.S_ema, n_seeds=t.n_seeds))
    d = t.alpha - t.prior_raw
    g["D_raw"] = d - d.median()
    for k in ("prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length", "tau", "B1b_raw", "B2b"):
        g[k] = x[k].to_numpy()
    gold = pd.read_csv(CM.RES / "T4_3/golds/gold_table.csv", index_col=0)[["dNdS", "LOEUF"]]
    return g.join(gold, on="hsapiens_gene_id")


def table(c):
    g = gene_table(c)
    f = CM.RES / f"T4_3/cache/{c}__inputs.parquet"
    if f.exists():
        g["block"] = pd.read_parquet(f).set_index("var_name")["block"].reindex(g.var_name).to_numpy()
        kind = "ESM block"
    else:
        g["block"] = np.arange(len(g))
        kind = "gene"
    try:
        runs = I.seed_runs_used(c)
    except Exception:  # noqa: BLE001 - single-seed cohorts are not in the pooling table
        runs = {}
    sd, ss = [], []
    for r in runs.values():
        st = K.read_run(r)
        sd.append(st["alpha"] - st["prior"])
        ss.append(st["S_g_ema"])
    return g, kind, (np.column_stack(sd) if sd else None), (np.column_stack(ss) if ss else None)


def compute():
    units, rows = [], []
    frames = {}
    for c in COHORTS:
        g, kind, Sd, Ss = table(c)
        frames[c] = g
        for v in REL:
            ok = g[v].notna().to_numpy()
            for r in READ:
                n = int((ok & g[r].notna()).sum())
                rec = dict(cohort=c, n_seeds=int(g.n_seeds.max()), readout=r, relation=v, n=n, ci_kind=kind)
                if n < 30:
                    rows.append(dict(rec, raw=np.nan))
                    continue
                rec["raw"] = spearmanr(g.loc[ok, r], g.loc[ok, v])[0]
                M = {"D": Sd, "S": Ss}.get(r)
                if M is not None and M.shape[1] > 1:
                    vals = [spearmanr(M[ok, j], g.loc[ok, v])[0] for j in range(M.shape[1])]
                    rec.update(seed_lo=min(vals), seed_hi=max(vals))
                rows.append(rec)
                units.append(dict(g=g, gold=v, covs=covs_for(v), scores=[r], draws=500, label=dict(cohort=c, readout=r, relation=v)))
    adj = CM.run_units(units).rename(columns={"rho_p": "adjusted", "ci_lo": "adj_lo", "ci_hi": "adj_hi", "cov": "adjusted_given", "n": "n_adj"})
    t = pd.DataFrame(rows).merge(adj[["cohort", "readout", "relation", "adjusted", "adj_lo", "adj_hi", "adjusted_given", "n_adj"]],
                                 on=["cohort", "readout", "relation"], how="left")
    K.write_csv(t, OUT / "relations.csv")
    print(t.pivot_table(index=["readout", "relation"], columns="cohort", values="raw").round(2)[COHORTS].to_string(), flush=True)
    return t, frames


def density(t, frames, cohorts=None):
    """Gene clouds: one column per property (x axis, in the row order of the heat maps), two rows per cohort (y axis: Δα,
    then S_g). Plots of a column share their x axis and plots of a row their y axis, so tick labels stand once, under
    the last row and left of the first column. `frames` holds the gene tables of all SCATTER cohorts (the x window of a
    property is set over all of them); `cohorts` are the ones drawn (default: all of SCATTER)."""
    style.apply()
    cohorts = list(SCATTER if cohorts is None else cohorts)
    rels = list(REL)
    # geometry, bp
    W = style.TEXTWIDTH
    left, right, top, bottom = 45.0, 7.0, 11.0, 30.0  # cohort name, axis name and tick labels; last tick label; key; tick labels and property names
    gap_x, gap_block, n_line, rho_line, side = 5.0, 4.0, 6.5, 10.0, 48.5
    plot_w = (W - left - right - gap_x * (len(rels) - 1)) / len(rels)
    block_h = n_line + 2 * (rho_line + side)
    H = top + len(cohorts) * block_h + gap_block * (len(cohorts) - 1) + bottom
    edge = 0.03  # room around the window of the hexagon grid, so the hexagons on its border are drawn whole
    fig = plt.figure(figsize=(style.inch(W), style.inch(H)))
    lims = {(c, r): tuple(frames[c][r].quantile([0.002, 0.998])) for c in cohorts for r in ("D", "S")}
    lowest = {}  # the lowest plot of each column carries the tick labels of the x axis
    for k, c in enumerate(cohorts):
        g = frames[c]
        block_top = H - top - k * (block_h + gap_block)
        rows_y = [block_top - n_line - (a + 1) * (rho_line + side) for a in range(2)]  # bottom of the Δα and of the S_g plots
        fig.text(5.5 / W, (rows_y[1] + rows_y[0] + side) / 2 / H, LABEL[c], rotation=90, ha="center", va="center", fontsize=style.SIZE_TITLE,
                 fontweight="bold")
        fig.add_artist(Line2D([12.0 / W] * 2, [rows_y[1] / H, (rows_y[0] + side) / H], color=RULE, lw=0.8))  # the two rows of the cohort
        for a, r in enumerate(("D", "S")):
            y0 = rows_y[a]
            fig.text(18.0 / W, (y0 + side / 2) / H, READ_AXIS[r], rotation=90, ha="center", va="center", fontsize=style.SIZE_LABEL)
            for i, v in enumerate(rels):
                x0 = left + i * (plot_w + gap_x)
                q = t[(t.cohort == c) & (t.readout == r) & (t.relation == v)].iloc[0]
                ok = g[v].notna()
                if ok.sum() < 30:
                    fig.text((x0 + plot_w / 2) / W, (y0 + side / 2) / H, "not defined\nfor this cohort", ha="center", va="center",
                             fontsize=style.SIZE_TEXT, color=MUTED)
                    continue
                ax = fig.add_axes([x0 / W, y0 / H, plot_w / W, side / H])
                x = np.log10(g.loc[ok, v].clip(lower=1e-3)) if v == "dNdS" else g.loc[ok, v]
                y = g.loc[ok, r]
                xl = (-3, 0.5) if v == "dNdS" else tuple(pd.concat([frames[s][v] for s in SCATTER]).quantile([0.002, 0.998]))
                yl = lims[(c, r)]
                ax.hexbin(x, y, gridsize=28, extent=(*xl, *yl), cmap=style.SEQUENTIAL, norm=LogNorm(vmin=1), mincnt=1, linewidths=0, rasterized=True)
                b = pd.qcut(x.rank(method="first"), 10, labels=False)
                m = pd.DataFrame(dict(x=x.to_numpy(), y=y.to_numpy(), b=b.to_numpy())).groupby("b").median()
                ax.plot(m.x, m.y, color=TREND, lw=0.9)
                ax.set_xlim(xl[0] - edge * (xl[1] - xl[0]), xl[1] + edge * (xl[1] - xl[0]))
                ax.set_ylim(yl[0] - edge * (yl[1] - yl[0]), yl[1] + edge * (yl[1] - yl[0]))
                ax.locator_params(axis="both", nbins=3)
                ax.tick_params(length=2, pad=1.5, labelleft=i == 0, labelbottom=False, bottom=False)
                fig.text(x0 / W, (y0 + side + 1.5) / H, f"ρ = {signed(q.raw)}", ha="left", va="bottom", fontsize=style.SIZE_TICK)
                if a == 0:  # the same genes in both rows of the cohort
                    assert int(q.n) == int(t[(t.cohort == c) & (t.readout == "S") & (t.relation == v)].iloc[0].n), (c, v)
                    fig.text(x0 / W, (y0 + side + rho_line + 1.5) / H, f"n = {int(q.n):,}", ha="left", va="bottom", fontsize=style.SIZE_TEXT,
                             color=MUTED)
                lowest[i] = ax
    for i, v in enumerate(rels):
        lowest[i].tick_params(labelbottom=True, bottom=True)
        fig.text((left + i * (plot_w + gap_x) + plot_w / 2) / W, (bottom - 11.5) / H, REL_AXIS[v], ha="center", va="top", fontsize=style.SIZE_LABEL,
                 linespacing=1.0)
    # key, above the first row: the color of a hexagon and the line
    kx, ky, kw, kh, steps = left, H - top + 3.0, 36.0, 4.5, 24
    cmap = plt.get_cmap(style.SEQUENTIAL)
    fig.text(kx / W, (ky + kh / 2) / H, "genes per hexagon: 1", ha="left", va="center", fontsize=style.SIZE_TEXT)
    kx += 58.0
    for s in range(steps):
        fig.add_artist(Rectangle(((kx + s * kw / steps) / W, ky / H), kw / steps / W * 1.05, kh / H, transform=fig.transFigure, fc=cmap(s / (steps - 1)), ec="none"))
    fig.text((kx + kw + 2.5) / W, (ky + kh / 2) / H, "most in the plot (log scale)", ha="left", va="center", fontsize=style.SIZE_TEXT)
    kx += kw + 96.0
    fig.add_artist(Line2D([kx / W, (kx + 12.0) / W], [(ky + kh / 2) / H] * 2, color=TREND, lw=0.9))
    fig.text((kx + 14.5) / W, (ky + kh / 2) / H, "median of the readout in ten equal-count bins of the property", ha="left", va="center",
             fontsize=style.SIZE_TEXT)
    with plt.rc_context({"savefig.dpi": RASTER_DPI}):
        style.save_panel(fig, "supp_relations_density/density", exact=True)


def heat_one(ax, tt, value, ylabels, xlabels):
    """One map: properties (rows) by cohorts (columns), the value in every cell; a cell without a value is gray and empty."""
    P = tt.pivot(index="relation", columns="cohort", values=value).reindex(index=list(REL), columns=COHORTS)
    n_rows, n_cols = P.shape
    ax.set_facecolor(NA_CELL)
    im = ax.pcolormesh(np.arange(n_cols + 1) - 0.5, np.arange(n_rows + 1) - 0.5, np.ma.masked_invalid(P.to_numpy(dtype=float)), cmap=style.DIVERGING,
                       norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), edgecolors="white", linewidth=0.3)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    for i in range(n_rows):
        for j in range(n_cols):
            v = P.iat[i, j]
            if not pd.isna(v):
                ax.text(j, i, signed(v), ha="center", va="center", fontsize=style.SIZE_TEXT, color="white" if abs(v) > 0.55 else INK)
    seeds = tt.groupby("cohort").n_seeds.max()
    ax.set_xticks(range(n_cols), [LABEL[c] + (" (1 seed)" if seeds[c] == 1 else "") for c in COHORTS] if xlabels else [""] * n_cols,
                  rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n_rows), [REL[v] for v in REL] if ylabels else [""] * n_rows)
    ax.tick_params(length=0)
    ax.axvline(3.5, color="white", lw=2)  # liver cohorts | testis cohorts
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def heatmap(t):
    """Three rows of maps (the readouts) by two columns (Spearman and partial rank correlation); property names left of the
    first column, cohort names under the last row, one color bar."""
    style.apply()
    reads, kinds = ("D", "D_raw", "S"), (("raw", "Spearman ρ"), ("adjusted", "partial ρ given the other covariates"))
    # geometry, bp
    W = style.TEXTWIDTH
    left, right, top, bottom = 92.0, 46.0, 12.0, 80.0  # readout and property names; color bar; column headings; cohort names and notes
    gap_x, gap_y, cell_h = 8.0, 6.0, 15.0
    map_w, map_h = (W - left - right - gap_x) / 2, cell_h * len(REL)
    H = top + len(reads) * map_h + (len(reads) - 1) * gap_y + bottom
    fig = plt.figure(figsize=(style.inch(W), style.inch(H)))
    im = None
    for i, r in enumerate(reads):
        tt = t[t.readout == r]
        y0 = H - top - (i + 1) * map_h - i * gap_y
        fig.text(6.5 / W, (y0 + map_h / 2) / H, READ[r], rotation=90, ha="center", va="center", fontsize=style.SIZE_TITLE)
        for j, (value, heading) in enumerate(kinds):
            ax = fig.add_axes([(left + j * (map_w + gap_x)) / W, y0 / H, map_w / W, map_h / H])
            im = heat_one(ax, tt, value, j == 0, i == len(reads) - 1)
            if i == 0:
                ax.set_title(heading, pad=3)
    cax = fig.add_axes([(W - right + 8.0) / W, (H - top - 1.5 * map_h - gap_y - 60.0) / H, 7.0 / W, 120.0 / H])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0.0" if v == 0 else f"{v:+.1f}".replace("-", "−")))
    cb.ax.tick_params(length=1.5)
    cb.set_label("rank correlation", fontsize=style.SIZE_LABEL)
    cb.outline.set_visible(False)
    # notes under the cohort names
    note = dict(ha="left", va="bottom", fontsize=style.SIZE_TEXT, color=INK)
    fig.text(2.0 / W, 10.0 / H, "Partial ρ: partial rank correlation given the other covariates (identity, expression, detection, coding length).", **note)
    fig.add_artist(Rectangle((2.0 / W, 2.2 / H), 9.0 / W, 5.5 / H, transform=fig.transFigure, fc=NA_CELL, ec=RULE, lw=0.4))
    fig.text(13.5 / W, 2.0 / H, "not defined for the cohort.   Liver cohorts are anchored on the raw prior: their columns of the middle maps repeat the top maps.",
             **note)
    with plt.rc_context({"savefig.dpi": RASTER_DPI}):  # the color bar is the one raster part
        style.save_panel(fig, "supp_relations_heatmap/heatmap", exact=True)


def draw_panels() -> None:
    """The two panels from the cached tables (relations.csv and the gene tables of the three cohorts of the gene clouds);
    nothing is computed."""
    t = pd.read_csv(OUT / "relations.csv")
    density(t, {c: gene_table(c) for c in SCATTER})
    heatmap(t)


def main():
    t, frames = compute()
    density(t, frames)
    heatmap(t)
    print("relations written", flush=True)


if __name__ == "__main__":
    main()
