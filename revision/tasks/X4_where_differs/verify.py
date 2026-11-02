"""X4 (EXPLORATORY): independent recomputation of headline numbers. Uses no X4 or T4.3 statistics code and no X4 gene table:
source files joined on ENSG, polynomial (degree 3) rank covariates instead of splines. Writes results/revision/X4/verify.csv."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = "results/revision/"


def tab(c):
    t = pd.read_csv(f"{R}T3_3/{c}__consensus.csv").set_index("hsapiens_gene_id")
    i = pd.read_parquet(f"{R}T4_3/cache/{c}__inputs.parquet").set_index("hsapiens_gene_id")
    b = pd.read_csv(f"{R}T4_3/baselines_consensusD/baselines__{c}.csv").set_index("hsapiens_gene_id")
    g = pd.read_csv(f"{R}T4_3/golds/gold_table.csv", index_col=0)
    t = t[["D", "S_ema", "alpha", "prior_raw"]].join(i[["log_cp10k_spmean", "detection_spmean", "log_cds_length", "tau_shared"]])
    t = t.join(b[["B1b_raw", "B1c_raw", "B1z_raw", "B2b"]]).join(g[["G_H1", "G_H2", "G_H3", "G_H4", "H4_nsp"]])
    t["Sdir"] = pd.read_parquet(f"{R}X4/cache/Sdir__{c}.parquet").set_index("var_name")["Sdir"].reindex(pd.read_csv(f"{R}T3_3/{c}__consensus.csv").var_name).to_numpy()
    return t


def poly(T, cols):
    Z = [np.ones(len(T))]
    for c in cols:
        r = rankdata(T[c].to_numpy(float))
        r = (r - r.mean()) / r.std()
        Z += [r, r ** 2, r ** 3]
    return np.column_stack(Z)


def pr(T, y, x, covs):
    Z = poly(T, covs)
    e = [v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0] for v in (rankdata(T[y].to_numpy(float)), rankdata(T[x].to_numpy(float)))]
    return float(np.corrcoef(*e)[0, 1])


C = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]
P = C + ["tau_shared"]
B = ["B1b_raw", "B1c_raw", "B1z_raw", "B2b"]
rows = []
for c, gold, cv in (("liv_hmouse", "G_H1", C), ("liv_hmouse", "G_H3", P), ("test_hmouse", "G_H2", C), ("test_hmouse", "G_H4", P + ["H4_nsp"])):
    t = tab(c)
    T = t.dropna(subset=[gold] + cv + B + ["S_ema", "Sdir"])
    rows.append(dict(cohort=c, gold=gold, n=len(T), S=pr(T, gold, "S_ema", cv), Sdir=pr(T, gold, "Sdir", cv), B1b=pr(T, gold, "B1b_raw", cv),
                     S_given_all_direct=pr(T, gold, "S_ema", cv + B), Sdir_given_all_direct=pr(T, gold, "Sdir", cv + B),
                     S_given_Sdir=pr(T, gold, "S_ema", cv + ["Sdir"]), Sdir_given_S=pr(T, gold, "Sdir", cv + ["S_ema"])))
v = pd.DataFrame(rows)
v.to_csv(f"{R}X4/verify.csv", index=False)
print(v.round(3).to_string())
