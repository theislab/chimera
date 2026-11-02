"""X4 (EXPLORATORY, owner addition): how consensus delta-alpha and S_g relate to sequence constraint, expression,
specificity, length and direct baselines. Descriptive; no gold is used; unfavourable relations are shown as they are.

Readouts: D (T3.3 consensus, the scoreboard readout: clamped prior on testis), D_raw (raw prior), S (S_ema).
Relations: dN/dS and LOEUF (human; dN/dS is human-mouse in every cohort), pi_raw, expression (species-balanced
log CP10K), specificity tau (X3 column, one definition in all cohorts), log CDS length, B1b, B2b.
raw = Spearman. adjusted = rho_p of readout and relation given C (pi_raw, expression, detection, CDS length) without
the relation itself; for expression, detection is also left out (they are nearly the same quantity).
CI: block bootstrap on ESM modules where T4.3 built them, gene bootstrap elsewhere (ci_kind).
seed_lo, seed_hi: range of the raw Spearman over the cohort's seeds, each seed's delta-alpha or S on its own.

Writes results/revision/X4/relations.csv and figures/fig_relations_{density,heatmap}.{pdf,png}.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from revision.common import controls as K  # noqa: E402
from revision.tasks.T4_3_scoreboard import inputs as I  # noqa: E402
from revision.tasks.X4_where_differs import common as CM  # noqa: E402

sys.path.insert(0, str(CM.REPO / "notebooks"))
import _figure_design as FD  # noqa: E402

OUT = CM.OUT
FIG = OUT / "figures"
COHORTS = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp", "test_hmac", "test_hmouse", "test_3sp"]
LABEL = {"liv_hmonkey": "Liver H+Monkey", "liv_hmouse": "Liver H+Mouse", "liv_hzeb": "Liver H+Zebrafish", "liv_4sp": "Liver 4-species",
         "test_hmac": "Testis H+Macaque", "test_hmouse": "Testis H+Mouse", "test_3sp": "Testis 3-species"}
REL = {"dNdS": "dN/dS", "prior_raw": "Protein identity π", "LOEUF": "LOEUF (LoF tolerance)", "log_cp10k_spmean": "Expression (log CP10K)",
       "tau": "Cell-type specificity τ", "log_cds_length": "log CDS length", "B1b_raw": "B1b (profile correlation)",
       "B2b": "B2b (specificity cosine)"}
READ = {"D": r"$\Delta\alpha$", "D_raw": r"$\Delta\alpha$ on the raw prior", "S": r"$S_g$"}
COV = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]
SCATTER = ["liv_hmouse", "liv_4sp", "test_hmouse"]


def covs_for(v):
    c = [x for x in COV if x != v]
    if v == "log_cp10k_spmean":
        c = [x for x in c if x != "detection_spmean"]
    return c


def table(c):
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
    g = g.join(gold, on="hsapiens_gene_id")
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


def density(t, frames):
    rels = list(REL)
    cols = [(c, r) for c in SCATTER for r in ("D", "S")]
    fig, axes = plt.subplots(len(rels), len(cols), figsize=(FD.DOUBLE_W, 9.4), squeeze=False)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.945, bottom=0.03, wspace=0.42, hspace=0.42)
    lims = {}
    for c, r in cols:
        y = frames[c][r]
        lims[(c, r)] = tuple(y.quantile([0.002, 0.998]))
    for i, v in enumerate(rels):
        xall = pd.concat([frames[c][v] for c in SCATTER])
        for j, (c, r) in enumerate(cols):
            ax = axes[i, j]
            g = frames[c]
            q = t[(t.cohort == c) & (t.readout == r) & (t.relation == v)].iloc[0]
            ok = g[v].notna()
            if ok.sum() < 30:
                ax.axis("off")
                ax.text(0.5, 0.5, "not defined\nfor 4 species", transform=ax.transAxes, ha="center", va="center", fontsize=6, color="#555555")
                continue
            x = np.log10(g.loc[ok, v].clip(lower=1e-3)) if v == "dNdS" else g.loc[ok, v]
            y = g.loc[ok, r]
            xl = (-3, 0.5) if v == "dNdS" else tuple(xall.quantile([0.002, 0.998]))
            yl = lims[(c, r)]
            ax.hexbin(x, y, gridsize=28, extent=(*xl, *yl), cmap="Blues", norm=LogNorm(vmin=1), mincnt=1, linewidths=0, rasterized=True)
            b = pd.qcut(x.rank(method="first"), 10, labels=False)
            m = pd.DataFrame(dict(x=x.to_numpy(), y=y.to_numpy(), b=b.to_numpy())).groupby("b").median()
            ax.plot(m.x, m.y, color="#D55E00", lw=0.9)
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            ax.locator_params(axis="both", nbins=3)
            ax.tick_params(labelsize=5, length=1.5, pad=1)
            ax.text(0.03, 0.97, f"ρ {q.raw:+.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.5, fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.4))
            ax.text(0.97, 0.03, f"n {int(q.n):,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5, color="#444444",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.3))
            if j == 0:
                ax.set_ylabel(REL[v].replace(" (", "\n(") + ("\n(log10)" if v == "dNdS" else ""), fontsize=6.5, labelpad=3)
            if i == 0:
                ax.set_title(READ[r], fontsize=7.5, pad=3)
    for k, c in enumerate(SCATTER):
        x0 = axes[0, 2 * k].get_position().x0
        x1 = axes[0, 2 * k + 1].get_position().x1
        fig.text((x0 + x1) / 2, 0.972, LABEL[c], ha="center", va="bottom", fontsize=8, fontweight="bold")
    FD.save_figure(fig, "fig_relations_density", out_dir=FIG, formats=("pdf", "png"), close=True)


def heat_one(ax, tt, value, title, ylabels):
    P = tt.pivot(index="relation", columns="cohort", values=value).reindex(index=list(REL), columns=COHORTS)
    im = ax.imshow(P.to_numpy(dtype=float), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P.iat[i, j]
            ax.text(j, i, "n/a" if pd.isna(v) else f"{v:+.2f}".replace("+0.", "+.").replace("-0.", "−."), ha="center", va="center", fontsize=5.6,
                    color="#777777" if pd.isna(v) else ("white" if abs(v) > 0.55 else "#222222"))
    seeds = tt.groupby("cohort").n_seeds.max()
    ax.set_xticks(range(len(COHORTS)), [LABEL[c] + ("*" if seeds[c] == 1 else "") for c in COHORTS], fontsize=5.6, rotation=30, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(range(len(REL)), [REL[v] for v in REL] if ylabels else [""] * len(REL), fontsize=6)
    ax.set_title(title, fontsize=7.5)
    ax.tick_params(length=0)
    ax.axvline(3.5, color="white", lw=2)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def heatmap(t):
    fig, axes = plt.subplots(3, 2, figsize=(FD.DOUBLE_W, 7.4))
    fig.subplots_adjust(left=0.16, right=0.93, top=0.965, bottom=0.095, wspace=0.05, hspace=0.42)
    fig.text(0.01, 0.006, "Adjusted: rank partial correlation given the other covariates (identity, expression, detection, CDS length). * one seed. "
                          "n/a: B2b is not defined for these cohorts.", fontsize=6, color="#444444")
    im = None
    for i, r in enumerate(("D", "D_raw", "S")):
        tt = t[t.readout == r]
        heat_one(axes[i, 0], tt, "raw", f"{READ[r]}: raw Spearman", True)
        im = heat_one(axes[i, 1], tt, "adjusted", f"{READ[r]}: adjusted", False)
    cax = fig.add_axes([0.945, 0.3, 0.012, 0.4])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=6, length=1.5)
    cb.outline.set_visible(False)
    FD.save_figure(fig, "fig_relations_heatmap", out_dir=FIG, formats=("pdf", "png"), close=True)


def main():
    FD.apply_design()
    FIG.mkdir(parents=True, exist_ok=True)
    t, frames = compute()
    density(t, frames)
    heatmap(t)
    print("relations written", flush=True)


if __name__ == "__main__":
    main()
