"""EXPLORATORY: gene tables per cohort = X4_where_differs tables (readouts, baselines, covariates) + annotations + no-model terms.

No association is computed here. Prints coverage counts only.
Usage (from chimera/): PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m analysis.tasks.Y2_functional.build
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.common import gene_ids
from analysis.tasks.Y2_functional import rules as RU

REPO = gene_ids.REPO_ROOT
RES = REPO / "results/analysis"
OUT = RES / "Y2"
SYSTEMS = list(RU.MP_SYSTEMS)


def add_systems(t: pd.DataFrame) -> pd.DataFrame:
    a = t["M_any"].fillna(False).astype(bool) & t["H_any"].fillna(False).astype(bool)
    t["in_A"] = a
    new = {}
    for k in SYSTEMS:
        m, h = t[f"M_{k}"].fillna(False).astype(bool), t[f"H_{k}"].fillna(False).astype(bool)
        new[f"U_{k}"] = a & (m | h)
        new[f"y_{k}"] = (m & h).astype(float).where(new[f"U_{k}"])
        # IMPC-only mouse side, on IMPC-tested genes
        mi = t[f"Mimpc_{k}"].fillna(False).astype(bool)
        ai = t["Mimpc_any"].fillna(False).astype(bool) & t["H_any"].fillna(False).astype(bool) & t["impc_tested"].fillna(False).astype(bool)
        new[f"Ui_{k}"] = ai & (mi | h)
        new[f"yi_{k}"] = (mi & h).astype(float).where(new[f"Ui_{k}"])
    return pd.concat([t, pd.DataFrame(new, index=t.index)], axis=1)


def global_blocks(tables: dict[str, pd.DataFrame]) -> pd.Series:
    """One partition of genes over cohorts: the liver block where a gene has one; otherwise its testis block, merged into the
    liver block most of that testis block's shared genes sit in; a testis block with no shared gene stays its own."""
    li = tables["liver"].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")["block"].astype(str).radd("L")
    te = tables["testis"].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")["block"].astype(str).radd("T")
    shared = li.index.intersection(te.index)
    vote = pd.DataFrame({"t": te.reindex(shared), "l": li.reindex(shared)}).groupby("t")["l"].agg(lambda x: x.mode().iat[0])
    only = te.drop(shared)
    assigned = only.map(vote).fillna(only)
    return pd.concat([li, assigned])


def main():
    ph = pd.read_parquet(OUT / "annot/pheno.parquet")
    reg = pd.read_parquet(OUT / "annot/regulatory.parquet")
    reg = reg.drop(columns=["chrom", "strand", "tss", "biotype", "abc_chrom"], errors="ignore")
    tabs = {}
    for tissue, c in RU.COHORTS.items():
        g = pd.read_parquet(RES / f"X4/genes/{c}.parquet")
        d = pd.read_parquet(RES / f"X4/cache/Sdir__{c}.parquet")[["var_name", "Sdir", "Sdir_pattern", "Sdir_level"]]
        keep = [k for k in g.columns if not k.startswith(("cls_", "C_", "x_", "f_", "esm"))]
        t = g[keep].merge(d, on="var_name", how="left")
        tabs[tissue] = t
    gb = global_blocks(tabs)
    gb.to_frame("gblock").to_parquet(OUT / "annot/global_blocks.parquet")
    for tissue, t in tabs.items():
        t = t.merge(ph.drop(columns=["ncbi", "mouse_ncbi", "mgi", "mouse_ens"], errors="ignore"), left_on="hsapiens_gene_id", right_index=True, how="left")
        t = t.merge(reg, left_on="hsapiens_gene_id", right_index=True, how="left")
        t = add_systems(t)
        t["gblock"] = t["hsapiens_gene_id"].map(gb)
        t["ms"] = t["margin_support"].astype(bool)
        t["un"] = t["S"].abs() >= 1e-6
        tabs[tissue] = t
        t.to_parquet(OUT / f"genes_{tissue}.parquet")
    # replication: bone marrow and kidney. Margin flag recomputed by the T2_5_celltype_margin rule on the consensus (no canonical-seed table here).
    gs = {"bone_marrow": RES / "P12/readout/bm_genes.parquet", "kidney": RES / "P13/readout/kidney_genes.parquet"}
    # four-species liver, only for the fixed X2_liver_biology list in (c)
    g = pd.read_parquet(RES / "X4/genes/liv_4sp.parquet")
    t = g[[k for k in g.columns if not k.startswith(("cls_", "C_", "x_", "f_", "esm"))]]
    t = t.merge(ph.drop(columns=["ncbi", "mouse_ncbi", "mgi", "mouse_ens"], errors="ignore"), left_on="hsapiens_gene_id", right_index=True, how="left")
    t = t.merge(reg, left_on="hsapiens_gene_id", right_index=True, how="left")
    t = add_systems(t)
    t["ms"], t["un"] = t["margin_support"].astype(bool), t["S"].abs() >= 1e-6
    t.to_parquet(OUT / "genes_liver4sp.parquet")
    for tissue, f in gs.items():
        g = pd.read_parquet(f)
        g = g.rename(columns={"S_ema": "S", "B1_raw": "B1b_raw"})
        if "tau_shared" not in g:
            g["tau_shared"] = g["tau"]
        gold = pd.read_csv(RES / "T4_3/golds/gold_table.csv", index_col=0)[["dNdS", "LOEUF"]]
        t = g.merge(gold, left_on="hsapiens_gene_id", right_index=True, how="left")
        t = t.merge(ph.drop(columns=["ncbi", "mouse_ncbi", "mgi", "mouse_ens"], errors="ignore"), left_on="hsapiens_gene_id", right_index=True, how="left")
        t = t.merge(reg, left_on="hsapiens_gene_id", right_index=True, how="left")
        t = add_systems(t)
        un = t["S"].abs() >= 1e-6
        thr = t.loc[un, "S"].abs().median()
        t["un"] = un
        t["ms"] = un & (np.sign(t["S"]) == np.sign(t["D"])) & (t["S"].abs() >= thr)
        t.to_parquet(OUT / f"genes_{tissue}.parquet")
        tabs[tissue] = t
    rows = []
    for tissue, t in tabs.items():
        for u in ("ms", "un"):
            m = t[u].to_numpy(bool)
            base = dict(tissue=tissue, universe=u, n=int(m.sum()), in_A=int((m & t.in_A.to_numpy(bool)).sum()))
            for k in SYSTEMS:
                uk = t[f"U_{k}"].to_numpy(bool) & m
                base[f"U_{k}"] = int(uk.sum())
                base[f"both_{k}"] = int(np.nansum(t.loc[uk, f"y_{k}"]))
                base[f"Ui_{k}"] = int((t[f"Ui_{k}"].to_numpy(bool) & m).sum())
            base["lethal_known"] = int(t.loc[m, "lethal_known"].notna().sum())
            base["phylop_promoter"] = int(t.loc[m, "phylop_promoter"].notna().sum())
            base["utr5"] = int(t.loc[m, "phylop_utr5"].notna().sum())
            base["utr3"] = int(t.loc[m, "phylop_utr3"].notna().sum())
            base["dNdS"] = int(t.loc[m, "dNdS"].notna().sum())
            base["LOEUF"] = int(t.loc[m, "LOEUF"].notna().sum())
            rows.append(base)
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "coverage.csv", index=False)
    print(c.T.to_string())


if __name__ == "__main__":
    main()
