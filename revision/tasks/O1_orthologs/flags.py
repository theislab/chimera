"""O1 step 1c: flags per cohort, gene and species; counts and shares per cohort and species. Counts of the objects only for exact zeros; no readout or gold.

    PYTHONPATH=$PWD python -m revision.tasks.O1_orthologs.flags
"""

from __future__ import annotations

import re
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from revision.tasks.O1_orthologs import rules as RU

warnings.filterwarnings("ignore")


def sp_key(x: str) -> str:
    k = str(x).lower().replace(". ", "_").replace(" ", "_")
    m = {"human": "hsapiens", "monkey": "mmulatta", "macaque": "mmulatta", "mouse": "mmusculus", "zebrafish": "drerio", "mouse_lemur": "mmurinus", "lemur": "mmurinus", "pig": "sscrofa",
         "macaca_fascicularis": "mfascicularis", "m_fascicularis": "mfascicularis", "macaca_mulatta": "mmulatta", "m_mulatta": "mmulatta"}
    return m[k]


def main():
    P = pd.read_parquet(RU.RES / "table_pairs_flags.parquet")
    P["R2"] = P.T1 | P.F_retro | P.F_named | (P.F_lowconf & ~((P.goc_score >= 50) & (P.wga_coverage >= 50)))
    M = pd.read_parquet(RU.RES / "mapping_rows.parquet")
    rows, gene_rows = [], []
    pair_flags = ["F_absent", "F_ncrna", "F_type", "F_biotype", "F_lowconf", "F_nosynt", "F_retro", "F_named_raw", "F_named", "F_later", "F_earlier", "T1", "T2"]
    for c, v in RU.COHORTS.items():
        a = ad.read_h5ad(v["h5ad"], backed="r")
        tab = pd.read_csv(RU.TABLES[v["table"]]).set_index("ortholog_group_id")
        var = pd.Index(a.var_names.astype(str))
        gi = var.str.extract(r"^ortho_(\d+)$")[0].astype(int).to_numpy()
        g = pd.DataFrame(dict(cohort=c, var_name=var, ortholog_group_id=gi, hsapiens_gene_id=tab.hsapiens_gene_id.reindex(gi).to_numpy(), symbol=tab.hsapiens_gene_name.reindex(gi).to_numpy()))
        labs = a.obs["species"].astype(str).to_numpy()
        a2 = a.to_memory()
        X = sp.csc_matrix(a2.X)
        spc = sorted(set(labs))
        zero_any = np.zeros(len(g), bool)
        for s in spc:
            t = np.asarray(X[labs == s].sum(0)).ravel()
            k = sp_key(s)
            g[f"zero_{k}"] = t == 0
            zero_any |= t == 0
        g["zero_any"] = zero_any
        partners = [sp_key(s) for s in spc if sp_key(s) != "hsapiens"]
        cols_all = [c_[:-8] for c_ in tab.columns if c_.endswith("_gene_id") and c_ != "hsapiens_gene_id"]
        pt = P[P.table == v["table"]].set_index(["species", "ortholog_group_id"])
        for k in cols_all:
            sub = pt.xs(k, level="species").reindex(gi)
            for f in ["T1", "T2", "F_later", "R2"]:
                g[f"{f}_{k}"] = sub[f].fillna(False).astype(bool).to_numpy()
        for tier in ("T1", "T2", "R2"):
            g[tier] = g[[f"{tier}_{k}" for k in partners]].any(axis=1)
            g[f"{tier}_table"] = g[[f"{tier}_{k}" for k in cols_all]].any(axis=1)
        g["T2L"] = g.T2 | g[[f"F_later_{k}" for k in partners]].any(axis=1)
        wrong = np.zeros(len(g), bool)
        false0 = np.zeros(len(g), bool)
        if v["table"] in ("liver", "testis") and c in RU.CANONICAL:
            mm = M[M.table == v["table"]]
            for s in spc:
                x = mm[mm.obj_species == s].set_index("ortholog_group_id").reindex(gi)
                wrong |= x.M_wrong_row.fillna(False).astype(bool).to_numpy()
                false0 |= x.M_false_zero.fillna(False).astype(bool).to_numpy()
        g["M_wrong_row"], g["M_false_zero"] = wrong, false0
        g["T3"] = g.T2 | g.M_wrong_row | g.M_false_zero
        g["T3b"] = g.T2L | g.M_wrong_row | g.M_false_zero
        g["R3"] = g.R2 | g.M_wrong_row | g.M_false_zero
        g["TZ"] = g.T3 | g.zero_any
        gene_rows.append(g)
        for k in partners:
            x = pt.xs(k, level="species").reindex(gi)
            r = dict(cohort=c, label=v["label"], table=v["table"], species=k, n_cells=int((np.array([sp_key(s) for s in labs]) == k).sum()), n_pairs=len(gi))
            for f in pair_flags:
                r[f] = int(x[f].fillna(False).astype(bool).sum())
            r["zero_species"] = int(g[f"zero_{k}"].sum())
            r["goc_lt50"] = int((x.goc_score < 50).sum())
            r["wga_lt50"] = int((x.wga_coverage < 50).sum())
            r["goc_median"], r["wga_median"] = float(x.goc_score.median()), float(x.wga_coverage.median())
            rows.append(r)
        print(c, len(g), {t: int(g[t].sum()) for t in ["T1", "T2", "T2L", "M_wrong_row", "M_false_zero", "T3", "zero_any", "TZ"]}, "table-wide T2", int(g.T2_table.sum()), spc, flush=True)
        a.file.close()
    G = pd.concat(gene_rows, ignore_index=True)
    G.to_parquet(RU.RES / "flags_by_cohort.parquet.tmp")
    (RU.RES / "flags_by_cohort.parquet.tmp").replace(RU.RES / "flags_by_cohort.parquet")
    R = pd.DataFrame(rows)
    for f in pair_flags + ["zero_species", "goc_lt50", "wga_lt50"]:
        R[f"{f}_share"] = R[f] / R.n_pairs
    R.to_csv(RU.RES / "audit_by_cohort_species.csv", index=False)
    S = G.groupby("cohort", sort=False)[["T1", "T2", "R2", "T2L", "M_wrong_row", "M_false_zero", "T3", "R3", "zero_any", "TZ", "T2_table"]].agg(["sum", "mean"])
    S.columns = [f"{a}_{'n' if b == 'sum' else 'share'}" for a, b in S.columns]
    S.insert(0, "n_genes", G.groupby("cohort", sort=False).size())
    S.to_csv(RU.RES / "audit_by_cohort.csv")
    pd.set_option("display.width", 230, "display.max_columns", 40, "display.max_rows", 100)
    print(R[["cohort", "species", "n_pairs", "F_ncrna", "F_biotype", "F_lowconf", "F_nosynt", "F_retro", "F_named", "F_later", "T1", "T2", "T2_share", "zero_species"]].round(3).to_string())
    print(S[[c for c in S.columns if c.endswith("_n") or c == "n_genes"]].to_string())


if __name__ == "__main__":
    main()
