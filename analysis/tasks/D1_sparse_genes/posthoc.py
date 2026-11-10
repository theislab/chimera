"""Post hoc, chosen after a_rho.csv was read; labeled so. (1) D on masked genes given C plus all 69 protein columns; (2) with and without an exact zero in a species; (3) forward margin against
label-free and labeled direct scores on masked genes, head to head. Writes results/analysis/D1/posthoc_*.csv.
    python -m analysis.tasks.D1_sparse_genes.posthoc
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
from joblib import Parallel, delayed

from analysis.common import gene_ids
from analysis.tasks.D1_sparse_genes import common as CM
from analysis.tasks.D1_sparse_genes import rules as RU
from analysis.tasks.H1_harmonise import rules as H1R


def scalars(c, t):
    d = H1R.COHORTS[c]
    a = ad.read_h5ad(d["h5ad"], backed="r")
    ids = pd.Index(np.asarray(gene_ids.ensembl(d["table"], a.var_names)))
    F = pd.DataFrame(np.asarray(a.varm["gene_features"])[:, [64, 65, 69, 70, 71]], index=ids, columns=["f_var", "f_id", "f_sd", "f_max", "f_min"])
    a.file.close()
    F = F[~F.index.duplicated(keep=False)].reindex(t.hsapiens_gene_id)
    F.index = t.index
    return F


def one(c, kind, gold, covs):
    t = CM.load(c)
    F = scalars(c, t)
    keep = [k for k in F.columns if F[k].std() > 1e-9]
    for k in keep:
        t[k] = F[k]
    r0 = (t.masked & t.clean).to_numpy() & t[gold].notna().to_numpy()
    if r0.sum() < 200:
        return [], []
    lab = dict(cohort=c, kind=kind)
    pcs = [f"esm{i}" for i in range(64)] + keep
    a = [dict(x, test="given_C_protein69") for x in CM.rho_rows(t, gold, covs, ["D"], r0, None, RU.N_DRAWS_GRID, lab, extra_linear=pcs)]
    z = t.o1_zero_any.fillna(False).to_numpy(bool)
    for nm, rw in (("zero_in_a_species", r0 & z), ("counts_in_all_species", r0 & ~z)):
        if rw.sum() >= CM.MIN_N:
            a += [dict(x, test=nm) for x in CM.rho_rows(t, gold, covs, ["D", "Sdir", "S_fwd"], rw, "D", RU.N_DRAWS_GRID, lab)]
    b = [dict(x, test="fwd_h2h") for x in CM.rho_rows(t, gold, covs, ["S_fwd", "Sdir", "Lfc"], r0, "S_fwd", RU.N_DRAWS, lab)]
    return a, b


def main():
    u = [(c, k, g, cv) for c in RU.GOLD_COHORTS for k, g, cv in CM.golds(c)]
    res = Parallel(n_jobs=10, backend="loky")(delayed(one)(*x) for x in u)
    pd.DataFrame([r for x in res for r in x[0]]).to_csv(RU.RES / "posthoc_masked.csv", index=False)
    pd.DataFrame([r for x in res for r in x[1]]).to_csv(RU.RES / "posthoc_fwd.csv", index=False)


if __name__ == "__main__":
    main()
