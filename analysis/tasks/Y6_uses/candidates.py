"""Y6 use 1, PH4: flagged candidates per species with counts and Ensembl reasons. The flag: S_row at or beyond the odd-tuned T_lowconf threshold of the row, or among the 1% most negative S_gs of U_expr.

    PYTHONPATH=$PWD python -m analysis.tasks.Y6_uses.candidates
"""

from __future__ import annotations

import gzip
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import gene_ids
from analysis.tasks.Y6_uses import rules as RU
from analysis.tasks.Y6_uses.annot_genome import GTF
from analysis.tasks.Y6_uses.use1 import OUT

H5 = {"liv_4sp": gene_ids.h5ad_path("liv_4sp"), "test_3sp": gene_ids.h5ad_path("test_3sp"), "bone_marrow": gene_ids.h5ad_path("bone_marrow"), "vz_full": gene_ids.h5ad_path("van_zyl")}


def gtf_names(path):
    names = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 3)
            if p[2] != "gene":
                continue
            g = re.search(r'gene_id "([^"]+)"', line).group(1)
            n = re.search(r'gene_name "([^"]+)"', line)
            b = re.search(r'gene_biotype "([^"]+)"', line)
            names[g] = (n.group(1) if n else "", b.group(1) if b else "", p[0])
    return names


def main():
    import anndata as ad

    D = pd.read_parquet(OUT / "posthoc_rows.parquet")
    m = pd.read_csv(OUT / "metrics.csv")
    rows = []
    for fit in RU.JOINT:
        a = ad.read_h5ad(H5[fit])
        key = "species" if "species" in a.obs else "Species"
        spc = a.obs[key].astype(str).to_numpy()
        X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
        var = pd.Index([str(v) for v in a.var_names])
        for lab, pre in RU.FITS[fit]["species"].items():
            x = D[(D.fit == fit) & (D.species_label == lab) & (D.det_other >= RU.DET_EXPR)].copy()
            th = m[(m.fit == fit) & (m.species_label == lab) & (m.score == "S_row") & (m.universe == "expr") & (m.label == "T_lowconf")].threshold
            cut1 = np.quantile(x.S_row, 0.99)
            flag = x.S_row >= cut1
            if len(th) and np.isfinite(th.iloc[0]):
                flag |= x.S_row >= th.iloc[0]
            x = x[flag].sort_values("S_gs")
            names = gtf_names(GTF[pre][110])
            inv = {}
            for g, (n, b, c) in names.items():
                if n:
                    inv.setdefault(n.upper(), []).append(g)
            idx = var.get_indexer(x.var_name)
            cs = np.asarray(X[spc == lab][:, idx].sum(0)).ravel()
            ch = np.asarray(X[spc == RU.FITS[fit]["human"]][:, idx].sum(0)).ravel()
            x["counts_s"], x["counts_human"] = cs, ch
            x["n_cells_s"], x["n_cells_human"] = int((spc == lab).sum()), int((spc == RU.FITS[fit]["human"]).sum())
            x["species_gene_name"] = x.gene_id.map(lambda g: names.get(g, ("", "", ""))[0])
            x["contig"] = x.gene_id.map(lambda g: names.get(g, ("", "", ""))[2])
            x["same_symbol_genes"] = [";".join(sorted(g2 for g2 in inv.get(str(s).upper(), []) if g2 != g)) for s, g in zip(x.symbol, x.gene_id)]
            x["reason_orthology"] = (x.high_conf == 0)
            x["reason_no_synteny"] = (x.goc == 0) & (x.wga == 0)
            x["reason_gone_116"] = x.homology_type_116.isna() | (x.homology_type_116 != "ortholog_one2one")
            x["reason_new_model"] = x.T_newasm.eq(1)
            x["reason_no_3utr"] = x.T_no3utr.eq(1)
            x["reason_single_exon"] = (x.ex_s <= 1) & (x.ex_human >= 3)
            rows.append(x[["fit", "species_label", "species", "symbol", "hsapiens_gene_id", "gene_id", "species_gene_name", "contig", "same_symbol_genes", "identity_s", "S_gs", "S_human", "S_ema", "D", "E_hs", "det_s", "det_other", "counts_s", "counts_human", "n_cells_s", "n_cells_human", "cp_s", "cp_other", "ex_s", "ex_human", "goc", "wga", "high_conf", "homology_type_110", "homology_type_116", "human_partners_116", "T_lowconf", "T_later", "T_newasm", "T_no3utr", "reason_orthology", "reason_no_synteny", "reason_gone_116", "reason_new_model", "reason_no_3utr", "reason_single_exon", "split"]])
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "candidates.csv", index=False)
    print(out.groupby(["fit", "species_label"]).agg(n=("symbol", "size"), lowconf=("T_lowconf", "sum"), later=("T_later", "sum"), newasm=("T_newasm", "sum"), nosyn=("reason_no_synteny", "sum"), single=("reason_single_exon", "sum")))


if __name__ == "__main__":
    main()
