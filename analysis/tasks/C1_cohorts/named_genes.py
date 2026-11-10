"""Named genes per cohort, with counts and ortholog checked gene by gene (the check that analysis/tasks/O1_orthologs motivates). Strict support (rule of H1_harmonise); not in zero_genes.tsv; both tails of D.
Columns: symbol, D, S_ema, B1, pi, tau, per species detection and counts over the whole training object, each partner's ID, name, identity, GOC, WGA, top class by |contribution| in each species row.

    python analysis/tasks/C1_cohorts/named_genes.py <cohort> ... [--k 15]
"""
from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd

from analysis.common import gene_ids
from analysis.tasks.C1_cohorts import configs as CF
from analysis.tasks.C1_cohorts import tests_c1 as T


def one(c: str, k: int) -> pd.DataFrame:
    g = pd.read_parquet(T.H1 / "genes" / f"{c}.parquet")
    zf = CF.RES / c / "zero_genes.tsv"
    parent = CF.SUBSETS.get(c, {}).get("parent", c)
    flag = set()
    if zf.exists() and zf.stat().st_size > 2:
        flag = set(pd.read_csv(zf, sep="\t")["symbol"].dropna())
    tab = gene_ids.load_table(c)
    ids = pd.Index(g.index).str.replace("ortho_", "", regex=False).astype(int)
    a = ad.read_h5ad(CF.h5ad(c))
    X = a.X.tocsr()
    sp_ = a.obs["species"].astype(str).to_numpy()
    fw = T.fw_tables(c) if (CF.RES / c / "drivers").exists() else None
    s = g[g.strict_support & ~g.symbol.isin(flag)].sort_values("D")
    pick = pd.concat([s.head(k).assign(tail="below"), s.tail(k).assign(tail="above")])
    out = []
    pos = {v: i for i, v in enumerate(g.index)}
    for v, r in pick.iterrows():
        j = pos[v]
        row = dict(cohort=c, tail=r["tail"], symbol=r.symbol, D=r.D, S_ema=r.S_ema, B1=r.B1, pi=r.pi_raw, tau=r.tau, saturated=bool(r.saturated_flag))
        for nm in np.unique(sp_):
            x = X[sp_ == nm][:, j]
            row[f"det_{nm}"] = float((x > 0).mean())
            row[f"counts_{nm}"] = float(x.sum())
        t = tab.loc[ids[j]]
        for col in t.index:
            if col.endswith(("_gene_id", "_gene_name", "_identity", "_goc", "_wga")) and not col.startswith("hsapiens") and col != "mean_identity":
                row[col] = t[col]
        if fw is not None:
            vi = list(fw["var_names"]).index(v)
            for si, nm in enumerate(fw["species"]):
                rows = np.flatnonzero(fw["sp"] == si)
                C = fw["S"][rows, vi] * fw["w"][rows]
                row[f"top_{nm}"] = fw["labels"][rows][np.abs(C).argmax()]
                row[f"topC_{nm}"] = float(C[np.abs(C).argmax()])
        out.append(row)
    t = pd.DataFrame(out)
    t.to_csv(CF.RES / c / "named_genes.tsv", sep="\t", index=False)
    return t


if __name__ == "__main__":
    k = 15
    for c in [x for x in sys.argv[1:] if not x.startswith("--")]:
        t = one(c, k)
        print(c, len(t))
