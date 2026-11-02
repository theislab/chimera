"""Y3 (EXPLORATORY): what the spermatocyte-peak and hepatocyte catabolic genes look like in the counts.

No training and no new association with a gold: read-outs and golds are copied from the X4 gene tables, and
pseudobulks are the X4 / T4.2 ones (log2(1 + CPM) from raw counts, per species and cell type or stage, mean
over libraries). The cell counts and raw UMI totals are recomputed here from the training h5ad.

Worked genes, by a stated rule (the tables were already read in X1, X2 and X4, so the rule is descriptive):
    testis   of the 17 X4 named spermatocyte genes (reread_x1_named.csv): strict spermatocyte peak, all three golds
             present, largest residual Murat pattern r_G_H4.
    liver    of the 13 X4-nameable 4-species core genes (reread_x2_named.csv): hepatocyte detection >= 0.25 in human,
             macaque and mouse, both independent golds present, lowest EVEE level value (G_H1).
    Amendment after the first run, before any text: the first liver rule was largest |S| with counts in all four
    species, which gave ASL. ASL is a poor example of its class: macaque cells of every type detect it at 66-100%
    and zebrafish has 3 UMI in 5,289 hepatocytes, so its macaque profile is nearly flat. It stays a robust margin call,
    named in the Fig 4 text, but it does not show "pattern shared, level differs" in raw counts. The rule above
    was written after seeing the table, so the choice of worked gene is illustration, not evidence.

Set-level checks, on the same tables:
    testis   X1 SC-peak group (human-defined): share of genes whose peak stage in mouse and macaque is SC too.
    liver    X2 list of 63 nuclear genes: share with the hepatocyte as top shared type per species; hepatocyte
             log2(1 + CPM) median per species and median absolute gap per species pair, beside matched-free background.

Writes results/revision/Y3/worked_*.csv.

Usage: PYTHONPATH=$PWD python -m revision.tasks.Y3_findings.worked_genes
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp

from revision.common import cohorts as C
from revision.common import gene_ids
from revision.tasks.T4_2_gene_baselines import run as T42

RES = gene_ids.REPO_ROOT / "results" / "revision"
OUT = RES / "Y3"
STAGES = ["SG", "SC", "rSD", "eSD", "Sertoli", "Other_somatic"]


def tidy(g: pd.DataFrame, symbol: str) -> pd.DataFrame:
    r = g[g.symbol == symbol].iloc[0]
    rows = []
    for col in [k for k in g.columns if k.startswith("x_")]:
        _, sp_, *typ = col.split("_")
        typ = "_".join(typ)
        rows.append(dict(symbol=symbol, species=sp_, type=typ, log2cpm=r[col], detection=r.get(f"f_{sp_}_{typ}", np.nan)))
    return pd.DataFrame(rows)


def raw_counts(c: str, symbol: str, g: pd.DataFrame, coarse: bool) -> pd.DataFrame:
    a = C.read_adata(c)
    vn = g.loc[g.symbol == symbol, "var_name"].iloc[0]
    j = list(a.var_names).index(vn)
    x = a.X[:, j]
    x = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
    lab = a.obs["cell_type"].astype(str)
    lab = lab.map(T42.TESTIS_COARSE) if coarse else lab
    return (pd.DataFrame(dict(species=a.obs["species"].astype(str).to_numpy(), type=lab.to_numpy(), umi=x, expressing=x > 0))
            .groupby(["species", "type"]).agg(n_cells=("umi", "size"), umi_total=("umi", "sum"), n_expressing=("expressing", "sum"))
            .reset_index().assign(symbol=symbol))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # testis
    named = pd.read_csv(RES / "X4/reread_x1_named.csv")
    n = named[(named.peak_strict == "SC")].dropna(subset=["G_H2", "G_H4", "G_H4_primates"]).sort_values("r_G_H4", ascending=False)
    t_gene = n.symbol.iloc[0]
    t3 = pd.read_parquet(RES / "X4/genes/test_3sp.parquet")
    tm = pd.read_parquet(RES / "X4/genes/test_hmouse.parquet")
    print("testis candidates\n", n[["symbol", "prior_raw", "dNdS", "S", "B1b_raw", "G_H4", "r_G_H4"]].round(3).to_string())
    # liver
    nm = pd.read_csv(RES / "X4/reread_x2_named.csv")
    nm = nm[(nm.cohort == "liv_4sp") & nm.nameable].copy()
    dc = pd.read_csv(RES / "X2/divergent_conserved.csv")
    dc4 = dc[dc.cohort == "liv_4sp"].set_index("symbol")
    nm["complete"] = nm.symbol.map(dc4["complete"])
    print("liver candidates\n", nm[["symbol", "prior_raw", "S", "B1b_raw", "complete"]].round(3).sort_values("S").to_string())
    l4 = pd.read_parquet(RES / "X4/genes/liv_4sp.parquet")
    det = l4.set_index("symbol").loc[nm.symbol, [f"f_{s_}_Hepatocytes" for s_ in ("Human", "Monkey", "Mouse")]]
    keep_l = nm[(det.min(axis=1).to_numpy() >= 0.25)].dropna(subset=["G_H1", "G_H3"]).sort_values("G_H1")
    print("liver rule\n", keep_l[["symbol", "S", "G_H1", "G_H3"]].round(3).to_string())
    l_gene = keep_l.symbol.iloc[0]
    lm = pd.read_parquet(RES / "X4/genes/liv_hmouse.parquet")
    print("worked genes:", t_gene, l_gene, flush=True)
    keep = ["symbol", "prior_raw", "dNdS", "LOEUF", "D_raw", "S", "signfrac_D", "signfrac_S", "margin_support", "masked", "B1b_raw", "B1c_raw",
            "G_H1", "G_H2", "G_H3", "G_H4", "G_H4_primates"]
    s = []
    for name, g, gene in (("test_3sp", t3, t_gene), ("test_hmouse", tm, t_gene), ("liv_4sp", l4, l_gene), ("liv_hmouse", lm, l_gene)):
        s.append(g.loc[g.symbol == gene, [k for k in keep if k in g.columns]].assign(cohort=name))
    pd.concat(s).to_csv(OUT / "worked_readouts.csv", index=False)
    pd.concat([tidy(t3, t_gene).assign(cohort="test_3sp"), tidy(tm, t_gene).assign(cohort="test_hmouse"),
               tidy(l4, l_gene).assign(cohort="liv_4sp"), tidy(lm, l_gene).assign(cohort="liv_hmouse")]).to_csv(OUT / "worked_pseudobulk.csv", index=False)
    pd.concat([raw_counts("test_3sp", t_gene, t3, True).assign(cohort="test_3sp"),
               raw_counts("liv_4sp", l_gene, l4, False).assign(cohort="liv_4sp"),
               raw_counts("liv_4sp", "ASL", l4, False).assign(cohort="liv_4sp")]).to_csv(OUT / "worked_raw_counts.csv", index=False)
    # set level, testis
    sc = pd.read_csv(RES / "X1/sc_peak_genes.csv")
    print("sc_peak columns", sc.columns.tolist()[:30], flush=True)
    sp3 = ["Human", "Macaque", "Mouse"]
    cols = {s_: [f"x_{s_}_{t}" for t in STAGES] for s_ in sp3}
    missing = [k for v in cols.values() for k in v if k not in t3.columns]
    if missing:
        print("missing stage columns", missing[:8], [k for k in t3.columns if k.startswith("x_")][:12], flush=True)
    h = t3[[k for k in cols["Human"] if k in t3]].to_numpy(float)
    # human-defined SC-peak: X1 rule, peak minus mean >= 1 log2 and detection >= 0.10 in the peak stage
    peak_h = np.nanargmax(h, 1)
    margin = np.nanmax(h, 1) - np.nanmean(h, 1)
    det = t3[[f"f_Human_{t}" for t in STAGES]].to_numpy(float)
    is_sc = (peak_h == STAGES.index("SC")) & (margin >= 1) & (det[:, STAGES.index("SC")] >= 0.10)
    rows = [dict(check="n_human_SC_peak", value=int(is_sc.sum()), note="must equal X1 stage_groups n 1103")]
    for s_ in sp3[1:]:
        m = t3[cols[s_]].to_numpy(float)
        ok = np.isfinite(m).all(1) & (np.nanmax(m, 1) > 0)
        pk = np.argmax(np.nan_to_num(m), 1)
        rows.append(dict(check=f"SC_group_peak_SC_in_{s_}", value=float((pk[is_sc & ok] == 1).mean()), n=int((is_sc & ok).sum()),
                         note="background: all expressed genes " + f"{(pk[ok] == 1).mean():.3f}"))
        rows.append(dict(check=f"SC_group_peak_SC_or_rSD_in_{s_}", value=float(np.isin(pk[is_sc & ok], [1, 2]).mean()), n=int((is_sc & ok).sum()),
                         note="background " + f"{np.isin(pk[ok], [1, 2]).mean():.3f}"))
    pd.DataFrame(rows).to_csv(OUT / "worked_sets_testis.csv", index=False)
    # set level, liver
    genes63 = [g_ for g_ in dc4.index if not g_.startswith("MT-")]
    in63 = l4.symbol.isin(genes63).to_numpy()
    spl = ["Human", "Monkey", "Mouse", "Zebrafish"]
    types = sorted({k.split("_", 2)[2] for k in l4.columns if k.startswith("x_")})
    rows = []
    for s_ in spl:
        cs = [f"x_{s_}_{t}" for t in types if f"x_{s_}_{t}" in l4]
        m = l4[cs].to_numpy(float)
        top = np.array(cs)[np.argmax(np.nan_to_num(m), 1)]
        hep = l4[f"x_{s_}_Hepatocytes"].to_numpy(float)
        rows.append(dict(species=s_, n_types=len(cs), share_top_hepatocyte_set=float(np.mean(top[in63] == f"x_{s_}_Hepatocytes")),
                         share_top_hepatocyte_all=float(np.mean(top == f"x_{s_}_Hepatocytes")),
                         hep_log2cpm_median_set=float(np.nanmedian(hep[in63])), hep_log2cpm_median_all=float(np.nanmedian(hep))))
    pd.DataFrame(rows).to_csv(OUT / "worked_sets_liver.csv", index=False)
    rows = []
    for a_, b_ in itertools.combinations(spl, 2):
        d = (l4[f"x_{a_}_Hepatocytes"] - l4[f"x_{b_}_Hepatocytes"]).abs().to_numpy(float)
        expr = (l4[[f"x_{a_}_Hepatocytes", f"x_{b_}_Hepatocytes"]].min(axis=1) > 0).to_numpy()
        rows.append(dict(pair=f"{a_}-{b_}", median_abs_gap_set=float(np.nanmedian(d[in63])), median_abs_gap_all=float(np.nanmedian(d)),
                         median_abs_gap_set_expr_both=float(np.nanmedian(d[in63 & expr])), median_abs_gap_all_expr_both=float(np.nanmedian(d[expr])),
                         n_set_expr_both=int((in63 & expr).sum())))
    pd.DataFrame(rows).to_csv(OUT / "worked_sets_liver_gaps.csv", index=False)
    # level-matched hepatocyte gap: set genes against non-set genes in the same decile of pair-mean hepatocyte level
    # (genes expressed in both species); per list, 63 genes and the 15 X4 core genes
    core = pd.read_csv(RES / "X4/reread_x2_named.csv")
    core = core[core.cohort == "liv_4sp"].symbol.tolist()
    rows = []
    for name, flag in (("prog63_noMT", in63), ("core15", l4.symbol.isin(core).to_numpy())):
        for a_, b_ in itertools.combinations(spl, 2):
            x, y = l4[f"x_{a_}_Hepatocytes"].to_numpy(float), l4[f"x_{b_}_Hepatocytes"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y) & (np.minimum(x, y) > 0)
            t = pd.DataFrame(dict(q=pd.qcut(((x + y) / 2)[ok], 10, labels=False, duplicates="drop"), gap=np.abs(x - y)[ok], st=flag[ok]))
            exp = t[t.st].q.map(t[~t.st].groupby("q").gap.median())
            rows.append(dict(set=name, pair=f"{a_}-{b_}", n=int(t.st.sum()), median_gap=float(t[t.st].gap.median()), median_gap_level_matched=float(exp.median()),
                             share_above_matched=float((t[t.st].gap.to_numpy() > exp.to_numpy()).mean())))
    pd.DataFrame(rows).to_csv(OUT / "worked_sets_liver_gaps_matched.csv", index=False)


if __name__ == "__main__":
    main()
