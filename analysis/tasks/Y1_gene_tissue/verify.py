"""Independent recomputation of the headline numbers. No shared statistics code: S and B1 re-read from each task's own table, joined on ENSG,
covariates recomputed from raw counts here, cubic polynomials of rank instead of splines, no block machinery.

    python -I launch.py analysis.tasks.Y1_gene_tissue.verify
"""

from __future__ import annotations

import itertools

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from scipy.stats import rankdata

from analysis.common import gene_ids

ROOT = gene_ids.REPO_ROOT
RES = ROOT / "results" / "analysis"
SRC = {"liver": ("T3_3/liv_hmouse__consensus.csv", "T4_2/baselines__liv_hmouse.csv", "data/processed/camex_human_mouse.h5ad", "liv_hmouse"),
       "testis": ("T3_3/test_hmouse__consensus.csv", "T4_2/baselines__test_hmouse.csv", "data/processed/camex_testis_human_mouse.h5ad", "test_hmouse"),
       "kidney": ("P13/readout/kidney_genes.parquet", None, "data/analysis/disease_kidney/kidney_healthy_train.h5ad", "kidney"),
       "islet": ("T7_9/readout/islet_genes.parquet", None, "data/analysis/disease_islet/islet_healthy_train.h5ad", "testis"),
       "eye": ("T11/readout/genes_hmouse.parquet", None, "data/analysis/van_zyl/van_zyl_hmouse.h5ad", "van_zyl")}


def read(t):
    a_, b_, h, tab = SRC[t]
    if a_.endswith("csv"):
        g = pd.read_csv(RES / a_).set_index("hsapiens_gene_id")
        b = pd.read_csv(RES / b_).set_index("hsapiens_gene_id")
        S, B = g["S_ema"], b["B1b_raw"]
    else:
        g = pd.read_parquet(RES / a_).set_index("hsapiens_gene_id")
        S, B = g["S_ema"], g["B1b_raw"] if "B1b_raw" in g else g["E_human_mouse"]
    a = ad.read_h5ad(ROOT / h)
    X = a.X.tocsr() if issparse(a.X) else csr_matrix(a.X)
    sp = a.obs["species"].astype(str).str.lower().to_numpy()
    L = np.asarray(X.sum(1)).ravel()
    det, lc = [], []
    Xb = (X > 0).astype(float)
    Xn = X.multiply(1e4 / L[:, None]).tocsr()
    Xn.data = np.log1p(Xn.data)
    for s in ("human", "mouse"):
        det.append(np.asarray(Xb[sp == s].mean(0)).ravel())
        lc.append(np.asarray(Xn[sp == s].mean(0)).ravel())
    ids = gene_ids.ensembl(tab, a.var_names).to_numpy()
    cov = pd.DataFrame(dict(expr=np.mean(lc, 0), det=np.mean(det, 0)), index=ids)
    cov = cov[cov.index.notna()]
    d = pd.DataFrame(dict(S=S, B=B)).join(cov)
    prior = g["prior_raw"]
    d["prior"] = prior
    return d


def ranks(x):
    return rankdata(x) / len(x)


def poly(d, cols):
    return np.column_stack([np.ones(len(d))] + [ranks(d[c].to_numpy()) ** k for c in cols for k in (1, 2, 3)])


def res(y, Z):
    r = ranks(y)
    return r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]


def main():
    T = {t: read(t) for t in SRC}
    gold = pd.read_csv(RES / "T4_3" / "golds" / "gold_table.csv", index_col=0)
    Gk = pd.read_parquet(RES / "Y1" / "gold" / "gold_kidney.parquet")["G"]
    Gp = pd.read_parquet(RES / "Y1" / "gold" / "gold_pancreas.parquet")["G"]
    G = {"liver": gold["G_H3"], "testis": gold["G_H4"], "kidney": Gk, "islet": Gp}
    rows = []
    for t in G:
        # support in the matched tissue and in the source tissue only (pairwise), five tissues read here; bone marrow left to the main code
        for src in SRC:
            d = T[t].join(T[src][["S", "B"]].add_suffix("_src"), how="inner").join(G[t].rename("G"), how="inner")
            d = d[(d["S"].abs() >= 1e-6) & (d["S_src"].abs() >= 1e-6) & d["B"].notna() & d["B_src"].notna()].dropna(subset=["G", "expr", "det", "prior"])
            Z = poly(d, ["prior", "expr", "det"])
            rg = res(d["G"].to_numpy(), Z)
            rows.append(dict(gold=t, source=src, n=len(d), rho_S=np.corrcoef(rg, res(d["S_src"].to_numpy(), Z))[0, 1], rho_B1=np.corrcoef(rg, res(d["B_src"].to_numpy(), Z))[0, 1],
                             own_S=np.corrcoef(rg, res(d["S"].to_numpy(), Z))[0, 1], own_B1=np.corrcoef(rg, res(d["B"].to_numpy(), Z))[0, 1]))
    r = pd.DataFrame(rows)
    r["d_S"], r["d_B1"] = r.own_S - r.rho_S, r.own_B1 - r.rho_B1
    r.to_csv(RES / "Y1" / "verify_pairwise.csv", index=False)
    print(r[r.gold != r.source].groupby("gold")[["d_S", "d_B1"]].mean().round(3))
    # within gene
    w = []
    for a, b in itertools.combinations(G, 2):
        d = T[a].join(T[b], lsuffix="_a", rsuffix="_b", how="inner").join(G[a].rename("Ga"), how="inner").join(G[b].rename("Gb"), how="inner")
        d = d[(d["S_a"].abs() >= 1e-6) & (d["S_b"].abs() >= 1e-6) & d["B_a"].notna() & d["B_b"].notna()].dropna(subset=["Ga", "Gb", "expr_a", "expr_b", "prior_a"])
        Za, Zb = poly(d, ["prior_a", "expr_a", "det_a"]), poly(d, ["prior_b", "expr_b", "det_b"])
        z = lambda col, Z: (lambda e: (e - e.mean()) / e.std())(res(d[col].to_numpy(), Z))  # noqa: E731
        dG = z("Ga", Za) - z("Gb", Zb)
        w.append(dict(pair=f"{a}|{b}", n=len(d), S=np.corrcoef(dG, z("S_a", Za) - z("S_b", Zb))[0, 1], B1=np.corrcoef(dG, z("B_a", Za) - z("B_b", Zb))[0, 1]))
    w = pd.DataFrame(w)
    w.to_csv(RES / "Y1" / "verify_within.csv", index=False)
    print(w.round(3).to_string())
    # gene share on genes unmasked in the five tissues read
    ix = None
    for t in T:
        u = T[t][(T[t]["S"].abs() >= 1e-6) & T[t]["B"].notna()].dropna(subset=["expr", "det"]).index
        ix = u if ix is None else ix.intersection(u)
    Zs = pd.DataFrame({t: res(T[t].loc[ix, "S"].to_numpy(), poly(T[t].loc[ix], ["expr", "det"])) for t in T})
    Zb = pd.DataFrame({t: res(T[t].loc[ix, "B"].to_numpy(), poly(T[t].loc[ix], ["expr", "det"])) for t in T})
    print("n", len(ix), "gene share S", Zs.corr().where(np.triu(np.ones((5, 5)), 1) > 0).stack().mean().round(3), "B1", Zb.corr().where(np.triu(np.ones((5, 5)), 1) > 0).stack().mean().round(3))


if __name__ == "__main__":
    main()
