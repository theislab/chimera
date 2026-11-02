"""Y6 use 4: cis-regulatory evidence by proxy, rules.py USE 4. Human-chimpanzee hybrid allele counts against human-macaque readouts; islet is human-mouse.

Matched contexts, closest available: liver H+Monkey hepatocytes vs hybrid hepatocyte progenitors (HP); eye H+M. mulatta melanocytes vs retinal pigment epithelium (RPE; both pigmented, the eye object has no RPE type); islet
H+Mouse beta cells vs pancreatic progenitors (PP); testis H+Macaque has no matched type (negative context, level gap pooled).
Brain and pluripotent golds (MN, iPS, cortical spheroids) have no fit, read only as a descriptive row against the same readouts.

    PYTHONPATH=$PWD python -m revision.tasks.Y6_uses.use4
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from revision.common import gene_ids
from revision.tasks.T4_3_scoreboard import stats as S
from revision.tasks.X4_where_differs import common as CM
from revision.tasks.Y6_uses import rules as RU

CM.N_JOBS = 8
OUT = RU.OUT / "use4"
FIT = {
    "liv_hmonkey": dict(h5=gene_ids.h5ad_path("liv_hmonkey"), key="cell_type", type="Hepatocytes", human="Human", other="Monkey", match="HP", pre="mmulatta", table="data/processed/ortholog_table.csv"),
    "vz_hmmul": dict(h5=RU.ROOT / "data/revision/van_zyl/van_zyl_hmmul.h5ad", key="cell_type_harmonised", type="Melanocyte", human="human", other="macaca_mulatta", match="RPE", pre="mmulatta", table="data/revision/van_zyl/ortholog_table_van_zyl.csv", genes="T11"),
    "islet": dict(h5=RU.ROOT / "data/revision/disease_islet/islet_healthy_train.h5ad", key="cell_type", type="beta", human="Human", other="Mouse", match="PP", pre="mmusculus", table="data/processed/ortholog_table_testis.csv"),
    "test_hmac": dict(h5=gene_ids.h5ad_path("test_hmac"), key=None, type=None, human="Human", other="Macaque", match=None, pre="mmulatta", table="data/processed/ortholog_table_testis.csv"),
}
SP_KEYS = ["species", "Species"]


def level_gap(d):
    a = ad.read_h5ad(d["h5"])
    sp_key = next(k for k in SP_KEYS if k in a.obs)
    spc = a.obs[sp_key].astype(str).to_numpy()
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    mask = np.ones(a.n_obs, bool)
    if d["key"] is not None:
        col = d["key"] if d["key"] in a.obs else next(c for c in ("cell_type", "cell_ontology_class") if c in a.obs)
        labs = a.obs[col].astype(str)
        if d["type"] not in set(labs):
            raise KeyError(f"{d['type']} not in {sorted(set(labs))[:30]}")
        mask = labs.to_numpy() == d["type"]
    out = {}
    n = {}
    for s in (d["human"], d["other"]):
        m = mask & (spc == s)
        n[s] = int(m.sum())
        c = np.asarray(X[m].sum(0)).ravel()
        out[s] = np.log2(1 + 1e6 * c / max(c.sum(), 1))
    diff = out[d["human"]] - out[d["other"]]
    ok = (out[d["human"]] > 0) | (out[d["other"]] > 0)
    gap = np.abs(diff - np.median(diff[ok]))
    return pd.Series(-gap, index=[str(v) for v in a.var_names]), n


def table(fit):
    d = FIT[fit]
    if d.get("genes") == "T11":
        g = pd.read_parquet(RU.RES / "T11/readout/genes_hmmul.parquet")
        g = g.rename(columns={"prior_raw": "pi_raw", "B1_raw": "B1", "log_cp10k_spmean": "expr", "detection_spmean": "detection"}).set_index("var_name", drop=False)
    else:
        g = pd.read_parquet(RU.RES / f"H1/genes/{fit}.parquet")
        g = g.assign(var_name=g.index)
    tab = pd.read_csv(RU.ROOT / d["table"]).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    g["identity_s"] = g.hsapiens_gene_id.map(tab[f"{d['pre']}_identity"])
    gap, n = level_gap(d)
    g["gap_neg"] = gap.reindex(g.var_name).to_numpy()
    blocks = pd.read_parquet(RU.RES / "Y1/matrices/blocks.parquet").block
    g["block"] = g.hsapiens_gene_id.map(blocks).fillna(-1)
    g.loc[g.block < 0, "block"] = (pd.Series(np.arange((g.block < 0).sum())).to_numpy() % 300) + 100000
    return g, n


def add_golds(g):
    A = pd.read_parquet(RU.OUT / "gold/ase_reference.parquet")
    sym = g.symbol.astype(str)
    for c in A.columns:
        g[c] = sym.map(A[c]).to_numpy()
    for t in RU.ASE_TYPES + ["iPS", "CS"]:
        g[f"L_{t}"] = -g[f"a_{t}"].abs()
    for t in RU.ASE_TYPES:
        g[f"logn_{t}"] = np.log1p(g[f"n_{t}"])
    return g


def hyb_reliability():
    """Agreement of the two hybrid lines per type, one reference, for the record."""
    d = pd.read_csv(RU.DATA / "ase_wang2024/GSE232948_RNAseq_ASE_CM_HP_MN_PP_SKM_RPE_GRCh38.txt.gz", sep="\t").drop_duplicates("gene").set_index("gene")
    out = []
    for t in RU.ASE_TYPES:
        vals = []
        for h in ("Hyb1", "Hyb2"):
            hc = [c for c in d.columns if f"_{t}_{h}_" in c and c.endswith("_human")]
            cc = [c for c in d.columns if f"_{t}_{h}_" in c and c.endswith("_chimp")]
            a, b = d[hc].sum(1), d[cc].sum(1)
            v = np.log2((a + 1) / (b + 1)).where((a + b) >= RU.MIN_ALLELIC / 2)
            vals.append(v - v.median())
        p = pd.concat(vals, axis=1).dropna()
        out.append(dict(type=t, n=len(p), spearman=p.corr(method="spearman").iloc[0, 1]))
    return pd.DataFrame(out)


def contrast(g, fit, match, scores, covs, rows):
    others = [t for t in RU.ASE_TYPES if t != match]
    g = g.copy()
    g["L_other"] = g[[f"L_{t}" for t in others]].mean(axis=1)
    cols = [f"L_{match}", "L_other"] + scores + covs
    m = g[cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1) & rows
    T = g[m]
    Z = S.design(T, covs)
    res = []
    arr = {}
    for gold in (f"L_{match}", "L_other"):
        R = np.column_stack([S.midrank(T[gold].to_numpy(float))] + [S.midrank(T[s].to_numpy(float)) for s in scores])
        est = S.rho_p_matrix(R, Z)[1:]
        B = S.bootstrap(R, Z, RU.DRAWS_RHO, np.random.default_rng(RU.SEED), T["block"].to_numpy())[:, 1:]
        arr[gold] = (est, B)
    for k, s in enumerate(scores):
        d = arr[f"L_{match}"][0][k] - arr["L_other"][0][k]
        lo, hi = np.percentile(arr[f"L_{match}"][1][:, k] - arr["L_other"][1][:, k], [2.5, 97.5])
        res.append(dict(fit=fit, match=match, score=s, n=len(T), rho_matched=arr[f"L_{match}"][0][k], rho_other=arr["L_other"][0][k], delta=d, lo=lo, hi=hi))
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hyb_reliability().to_csv(OUT / "hybrid_line_agreement.csv", index=False)
    rows, con, meta = [], [], {}
    for fit, d in FIT.items():
        g, n = table(fit)
        meta[fit] = n
        g = add_golds(g)
        for c in ["expr", "detection", "tau", "pi_raw", "S_ema", "D", "B1"]:
            if c not in g:
                raise KeyError(f"{fit}: {c}")
        covs = ["pi_raw", "expr", "detection", "tau"]
        support = (np.abs(g.S_ema) >= 1e-6).to_numpy()
        scores = ["S_ema", "D", "B1", "gap_neg", "identity_s"]
        golds = ["P_cis"] + [f"L_{t}" for t in RU.ASE_TYPES] + ["L_iPS", "L_CS"]
        for gold in golds:
            extra = [f"logn_{gold[2:]}"] if gold[2:] in RU.ASE_TYPES else []
            rows += CM.rho_unit(g, gold, covs + extra, scores, rows=support, ref="S_ema", draws=RU.DRAWS_RHO, label=dict(fit=fit, matched=gold == f"L_{d['match']}", context=d["match"]))
        if d["match"]:
            con += contrast(g, fit, d["match"], scores, covs, pd.Series(support, index=g.index))
    pd.DataFrame(rows).to_csv(OUT / "rho.csv", index=False)
    pd.DataFrame(con).to_csv(OUT / "tissue_contrast.csv", index=False)
    pd.Series({k: str(v) for k, v in meta.items()}).to_csv(OUT / "cells_in_level_gap.csv")


if __name__ == "__main__":
    main()
