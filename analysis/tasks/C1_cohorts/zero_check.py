"""Checklist item 2b. Per cohort and species, before training: genes with exact zero in the training cells of a species; their total under ID in the species' whole source (all units
and classes, ID and, for symbol matrices, name), and whether a gene quiet in one species is well expressed in the other species of the object.

Output results/analysis/C1/<cohort>/zero_genes.tsv and one line per species.

    python analysis/tasks/C1_cohorts/zero_check.py <cohort> ...
"""
from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd

from analysis.common import gene_ids
from analysis.tasks.C1_cohorts import configs as CF


def one(c: str) -> None:
    parent = CF.SUBSETS.get(c, {}).get("parent", c)
    cc = CF.COHORTS[parent]
    src = CF.SOURCES[cc["source"]]
    a = ad.read_h5ad(CF.h5ad(c))
    tab = gene_ids.load_table(c)
    ids = pd.Index(a.var_names).str.replace("ortho_", "", regex=False).astype(int)
    X = a.X.tocsr()
    sp_lab = a.obs["species"].astype(str).to_numpy()
    rows = []
    det = {}
    for s in cc["species"]:
        nm = src["species"][s]["name"]
        if nm in set(sp_lab):
            det[nm] = np.asarray((X[sp_lab == nm] > 0).mean(0)).ravel()
    for s in cc["species"]:
        nm = src["species"][s]["name"]
        if nm not in det:
            continue
        z = det[nm] == 0
        low = det[nm] < 1e-3
        others = np.max([v for k, v in det.items() if k != nm], axis=0)
        pb = ad.read_h5ad(CF.BUILD / "species" / cc["source"] / s / "pseudobulk.h5ad")
        tot = pd.Series(np.asarray(pb.X.sum(0)).ravel(), index=pb.var_names)
        gid = tab.loc[ids, f"{s}_gene_id"].to_numpy()
        sym = tab.loc[ids, "hsapiens_gene_name"].to_numpy()
        for j in np.flatnonzero(low & (others >= 0.05)):
            rows.append(dict(cohort=c, species=nm, gene_id=gid[j], symbol=sym[j], det_species=float(det[nm][j]), det_max_other=float(others[j]), total_in_source=float(tot.get(gid[j], np.nan)), n_units=pb.n_obs))
        print(c, nm, "exact zero in training cells", int(z.sum()), "below 0.1% and at least 5% in another species", int((low & (others >= 0.05)).sum()), "source total 0", int(sum(tot.get(g, 0) == 0 for g in gid[z])), flush=True)
    t = pd.DataFrame(rows).sort_values("det_max_other", ascending=False) if rows else pd.DataFrame()
    t.to_csv(CF.RES / c / "zero_genes.tsv", sep="\t", index=False)
    print(t.head(14).to_string() if len(t) else "none", flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        one(c)
