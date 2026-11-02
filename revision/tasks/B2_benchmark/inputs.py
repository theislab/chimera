"""Per-cohort gene table for scoring: covariates, ESM blocks, golds, static scores. Keyed by var_name within a cohort and human ENSG across cohorts.

Canonical cohorts: T4.3 cache (same covariates, tau, blocks as the scoreboard). New cohorts: H1 gene table (same schema for all), T4.3 CDS lengths, Y1 ESM blocks.
Golds: T4.3 gold table, Y1 atlas golds; EVEE kidney; atlas level golds (golds_c.py). dN/dS and LOEUF oriented so that higher means more constrained.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from revision.common import controls as K
from revision.tasks.B2_benchmark import rules as RU

REPO = RU.REPO
RES = REPO / "results" / "revision"
CANON = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp", "test_hmac", "test_hmouse", "test_3sp"]


def gold_table() -> pd.DataFrame:
    g = pd.read_csv(RES / "T4_3/golds/gold_table.csv", low_memory=False).set_index("hsapiens_gene_id")
    Y = RES / "Y1/gold"
    for f, col in (("gold_kidney", "G_kidney"), ("gold_kidney_fine", "G_kidney_fine"), ("gold_kidney_nephron", "G_kidney_nephron"),
                   ("gold_pancreas", "G_pancreas"), ("gold_pancreas_noacinar", "G_pancreas_noacinar"),
                   ("gold_islet_fine", "G_islet_fine"), ("gold_islet_noacinar", "G_islet_noacinar")):
        p = Y / f"{f}.parquet"
        if p.exists():
            s = pd.read_parquet(p)["G"]
            g = g.join(s.rename(col), how="outer")
    z = zipfile.ZipFile(REPO / "data/revision/gold/G1/evolutionary_statistics.zip")
    var = pd.read_csv(io.BytesIO(z.read("evolutionary_statistics/var.txt")), sep="\t")
    m = pd.read_csv(RES / "T4_1/evee_symbol_to_ensg.tsv", sep="\t")
    ok = m.ensg.notna().to_numpy()
    v = var["kidney"].to_numpy(float)
    ev = pd.Series(np.where(v > 0, -np.log(np.where(v > 0, v, 1.0)), np.nan), index=m.ensg)[ok]
    g = g.join(ev[~ev.index.duplicated()].rename("G_evee_kidney"), how="outer")
    lv = RU.RES / "inputs" / "atlas_level_golds.parquet"
    if lv.exists():
        g = g.join(pd.read_parquet(lv), how="outer")
    coe = pd.read_csv(REPO / "data/revision/bone_marrow/shay_coe.tsv", sep="\t").drop_duplicates("hsapiens_gene_id", keep=False).set_index("hsapiens_gene_id")
    g = g.join(coe[["coe", "divergent", "human_log2_range_lineage_means", "mouse_log2_range_lineage_means"]].rename(columns={"coe": "G_coe"}), how="outer")
    return g


def cohort_table(cohort: str) -> pd.DataFrame:
    """var_name, hsapiens_gene_id, covariates, block, tau_shared, H1 readouts, golds, static scores. All 14+ cohorts in one schema."""
    h = pd.read_parquet(RES / f"H1/genes/{cohort}.parquet").set_index("hsapiens_gene_id")
    if cohort in CANON and (RES / f"T4_3/cache/{cohort}__inputs.parquet").exists():
        c = pd.read_parquet(RES / f"T4_3/cache/{cohort}__inputs.parquet")
        t = c[["hsapiens_gene_id", "var_name", "prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length", "tau_shared", "block"]].set_index("hsapiens_gene_id")
    else:
        a = pd.read_parquet(RES / "Y1/matrices/blocks.parquet")["block"]
        cds = K.cds_table()
        t = pd.DataFrame(index=h.index)
        from revision.tasks.H1_harmonise import rules as H1R

        tb = __import__("revision.common.gene_ids", fromlist=["x"]).load_table(H1R.COHORTS[cohort]["table"])
        vn = pd.Series("ortho_" + tb.index.astype(str), index=tb["hsapiens_gene_id"].values)
        t["var_name"] = vn[~vn.index.duplicated(keep=False)].reindex(t.index)
        t["prior_raw"] = h["pi_raw"]
        t["log_cp10k_spmean"] = h["expr"]
        t["detection_spmean"] = h["detection"]
        t["log_cds_length"] = cds["log_cds_length"].reindex(t.index)
        t["tau_shared"] = h["tau"]
        t["block"] = a.reindex(t.index)
        # genes outside the joint module table: each its own block, numbered after the others
        miss = t["block"].isna()
        t.loc[miss, "block"] = np.arange(miss.sum()) + int(a.max()) + 1
    for k in ("D", "D_raw", "S_ema", "B1", "count_support", "identity_human_mouse", "expr", "symbol"):
        t[k] = h[k].reindex(t.index)
    g = gold_table()
    for k in g.columns:
        t[k] = g[k].reindex(t.index)
    # H1's dN/dS, LOEUF oriented
    t["dNdS_score"] = -h["dNdS"].reindex(t.index).astype(float)
    t["LOEUF_score"] = -h["LOEUF"].reindex(t.index).astype(float)
    t["pi_score"] = t["prior_raw"]
    return t.reset_index()
