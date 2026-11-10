"""EXPLORATORY: one gene table per cohort. No association with a gold is computed here.

Readouts as X3_cross_cohort and X4_where_differs (T3_3_consensus consensus D, mean S_g EMA), T4_2_gene_baselines baselines (T4_3_scoreboard
consensus-D run for B3a where it exists), gold table, Compara age and paralogs, classes, per-type expression, variability and level gap.
The direct margin of X4_where_differs is taken from its cache, or computed here for the three cohorts it did not cover.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from analysis.common import cohorts as CO
from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.T4_3_scoreboard import inputs as I
from analysis.tasks.T4_3_scoreboard import stats as ST
from analysis.tasks.Y4_regions import annotations as AN
from analysis.tasks.Y4_regions import rules as RU

REPO = gene_ids.REPO_ROOT
RES = REPO / "results/analysis"
OUT = RES / "Y4"
GDIR = OUT / "genes"
CACHE = OUT / "cache"
HARMONISE = {"Kupffer cells": "Macrophages"}
MIN_CELLS = 20


def type_stats(c: str, a) -> pd.DataFrame:
    """Per gene: per-type mean log1p CP10K per species, peak type, spread, variances, level gap to the partner."""
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    L = np.maximum(np.asarray(X.sum(1)).ravel(), 1)
    lg = sp.diags(1e4 / L).dot(X).tocsr()
    lg.data = np.log1p(lg.data)
    spc = a.obs["species"].astype(str).to_numpy()
    lab = a.obs["cell_type"].astype(str)
    lab = lab.map(T42.TESTIS_COARSE) if c.startswith("test") else lab.map(lambda x: HARMONISE.get(x, x))
    lab = lab.to_numpy().astype(str)
    species = sorted(set(spc))
    types = sorted(set(lab) - {"nan"})
    G = X.shape[1]
    M, Vw, Vt = {}, {}, {}
    out = {}
    for s in species:
        ms = spc == s
        Y = lg[ms]
        m1 = np.asarray(Y.mean(0)).ravel()
        m2 = np.asarray(Y.multiply(Y).mean(0)).ravel()
        Vt[s] = m2 - m1 ** 2
        within = np.zeros(G)
        wsum = 0.0
        for t in types:
            mt = ms & (lab == t)
            n = int(mt.sum())
            if n < MIN_CELLS:
                continue
            Z = lg[mt]
            a1 = np.asarray(Z.mean(0)).ravel()
            a2 = np.asarray(Z.multiply(Z).mean(0)).ravel()
            M[(s, t)] = a1
            out[f"y_{s}_{t}"] = a1
            within += n * (a2 - a1 ** 2)
            wsum += n
        Vw[s] = within / max(wsum, 1)
    df = pd.DataFrame(out, index=a.var_names)
    df["logvar_total"] = np.log(np.mean([Vt[s] for s in species], 0) + 1e-6)
    df["logvar_within"] = np.log(np.mean([Vw[s] for s in species], 0) + 1e-6)
    prof = []
    for s in species:
        have = [t for t in types if (s, t) in M]
        P = np.full((G, len(types)), np.nan)
        for t in have:
            P[:, types.index(t)] = M[(s, t)]
        prof.append(P)
    prof = np.stack(prof)  # species x genes x types
    centred = prof - np.nanmean(prof, 2, keepdims=True)
    df["spread_y"] = np.nanmean(np.nanstd(prof, 2), 0)
    mp = np.nanmean(centred, 0)
    top = np.nanargmax(np.where(np.isnan(mp), -np.inf, mp), 1)
    srt = np.sort(np.where(np.isnan(mp), -np.inf, mp), 1)
    silent = np.nanmax(np.nanmean(prof, 0), 1) < 0.01
    df["peak_y"] = np.array(types, dtype=object)[top]
    df.loc[silent, "peak_y"] = None
    df["peak_margin"] = srt[:, -1] - srt[:, -2]
    # peak agreement and level gap, Human against the partner (X4_where_differs definition on these per-type means)
    part = RU.PARTNER[c]
    shared = [t for t in types if ("Human", t) in M and (part, t) in M]
    if len(shared) >= 2:
        H = np.column_stack([M[("Human", t)] for t in shared])
        O = np.column_stack([M[(part, t)] for t in shared])
        df["peak_agree"] = (H.argmax(1) == O.argmax(1)) & ~silent
        d = O - H
        ok = (H.max(1) > 0.01) & (O.max(1) > 0.01)
        d = d - np.median(d[ok], 0)
        df["shift_y"] = d.mean(1)
        df["abs_shift_y"] = np.abs(d.mean(1))
        df["shape_y"] = d.std(1)
        msd = (d ** 2).mean(1)
        df["frac_level_y"] = np.where(msd > 0, d.mean(1) ** 2 / np.where(msd > 0, msd, 1), np.nan)
        df["n_shared_types_y"] = len(shared)
    # species imbalance of counts
    det = np.column_stack([np.asarray((X[spc == s] > 0).mean(0)).ravel() for s in species])
    df["det_gap_y"] = det.max(1) - det.min(1)
    return df


def sdir(c: str) -> pd.DataFrame:
    f = RES / f"X4/cache/Sdir__{c}.parquet"
    g = CACHE / f"Sdir__{c}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    if not g.exists():
        from analysis.tasks.X4_where_differs import direct_margin as DM

        DM.run(c).to_parquet(g)
    return pd.read_parquet(g)


def local_moran(t: pd.DataFrame, value: str, knn: int = 15) -> np.ndarray:
    """Local agreement of the residual of ``value`` (given C) with its ESM neighbors; no gold enters."""
    from sklearn.neighbors import NearestNeighbors

    ok = t[RU.C + [value]].notna().all(1).to_numpy()
    T = t.loc[ok]
    F = T[[f"esm{j}" for j in range(64)]].to_numpy(float)
    Z = ST.design(T, RU.C)
    e = ST.resid(ST.midrank(T[value].to_numpy(float)), Z)
    e = e / e.std()
    _, nb = NearestNeighbors(n_neighbors=knn + 1).fit(F).kneighbors(F)
    out = np.full(len(t), np.nan)
    out[ok] = e * e[nb[:, 1:]].mean(1)
    return out


ID_COLS = {"Monkey": "mmulatta_identity", "Macaque": "mmulatta_identity", "Mouse": "mmusculus_identity",
           "Zebrafish": "drerio_identity"}


def true_identity(c: str, genes: pd.Series) -> np.ndarray:
    """Uncapped protein identity (0-1), mean over the cohort's non-human species. pi_raw is capped at 0.95, so it cannot
    separate genes above that; this column can."""
    tab = pd.read_csv(gene_ids.REPO_ROOT / "data/processed" / gene_ids.TABLES[c]).drop_duplicates("hsapiens_gene_id")
    tab = tab.set_index("hsapiens_gene_id").reindex(genes)
    spp = {"liv_hmonkey": ["Monkey"], "liv_hmouse": ["Mouse"], "liv_hzeb": ["Zebrafish"],
           "liv_4sp": ["Monkey", "Mouse", "Zebrafish"], "test_hmac": ["Macaque"], "test_hmouse": ["Mouse"],
           "test_3sp": ["Macaque", "Mouse"]}[c]
    return np.nanmean(np.column_stack([tab[ID_COLS[s]].to_numpy(float) for s in spp]), 1) / 100.0


def build(c: str, gold: pd.DataFrame) -> pd.DataFrame:
    a = CO.read_adata(c)
    gene_ids.assert_table_matches(a, gene_ids.load_table(c))
    x3 = pd.read_parquet(RES / f"X3/genes/{c}.parquet").set_index("var_name").reindex(a.var_names)
    cons = pd.read_csv(RES / f"T3_3/{c}__consensus.csv").set_index("var_name").reindex(a.var_names)
    t = pd.DataFrame(index=pd.Index(a.var_names, name="var_name"))
    t["hsapiens_gene_id"] = x3["hsapiens_gene_id"]
    t["symbol"] = x3["symbol"]
    for k in ("prior_raw", "log_cp10k_spmean", "detection_spmean", "detection_sprange", "log_cds_length", "margin_support",
              "lacks_margin_support", "tail", "D_raw"):
        t[k] = x3[k]
    t["S"] = x3["S"]
    t["D"] = cons["D"]
    t["sign_consistency"] = cons["sign_consistency"]
    t["n_seeds"] = cons["n_seeds"]
    t["masked"] = np.abs(t["S"]) < 1e-6
    F = np.asarray(a.varm["gene_features"], dtype=float)
    for j in range(64):
        t[f"esm{j}"] = F[:, j]
    cache = RES / f"T4_3/cache/{c}__inputs.parquet"
    if cache.exists():
        ci = pd.read_parquet(cache).set_index("var_name").reindex(a.var_names)
        t["block"] = ci["block"]
        t["tau_shared"] = ci["tau_shared"]
        t["nonzero_all_species"] = ci["nonzero_all_species"]
        assert np.allclose(ci["prior_raw"].to_numpy(), t["prior_raw"].to_numpy(), atol=1e-6)
    else:
        t["block"] = I.esm_blocks(F[:, :64])[0]
        t["tau_shared"] = I.tau_shared(a, c)
        t["nonzero_all_species"] = I.nonzero_all_species(a)
    b = pd.read_csv(RES / f"T4_2/baselines__{c}.csv").set_index("var_name")
    bc = RES / f"T4_3/baselines_consensusD/baselines__{c}.csv"
    if bc.exists():
        b = pd.read_csv(bc).set_index("var_name")
    for k in RU.BASELINES + ["B3b"]:
        t[k] = b[k].reindex(a.var_names)
    d = sdir(c).set_index("var_name").reindex(a.var_names)
    for k in RU.BESIDE:
        t[k] = d[k]
    t = t.join(gold, on="hsapiens_gene_id")
    t = t.join(type_stats(c, a))
    t["id_true"] = true_identity(c, t["hsapiens_gene_id"])
    t["chrom"] = t["hsapiens_gene_id"].map(AN.chrom_table())
    t["split"] = AN.split_of(t["chrom"].fillna("none"))
    cp = AN.compara()
    t = t.join(cp, on="hsapiens_gene_id")
    t["n_paralog"] = t["n_paralog"].fillna(0)
    t["paralog_same_chrom"] = t["paralog_same_chrom"].astype("boolean").fillna(False).astype(bool)
    t["paralog_on_odd"] = t["paralog_on_odd"].astype("boolean").fillna(False).astype(bool)
    ok = t[["log_cp10k_spmean", "logvar_total"]].notna().all(1).to_numpy()
    t["hv_resid"] = np.nan
    t.loc[ok, "hv_resid"] = ST.resid(ST.midrank(t.loc[ok, "logvar_total"].to_numpy(float)),
                                     ST.design(t.loc[ok], ["log_cp10k_spmean"]))
    t["moran_S"] = local_moran(t, "S")
    if c.startswith("liv"):
        ys = [k for k in t.columns if k.startswith("y_") and k.endswith("_Hepatocytes")]
        enr = []
        for k in ys:
            s = k.split("_")[1]
            cols = [x for x in t.columns if x.startswith(f"y_{s}_")]
            enr.append(t[k] - t[cols].mean(1))
        t["hep_enrich_y"] = np.mean(enr, 0)
    cl = AN.class_columns(pd.Index(t["hsapiens_gene_id"]), t.set_index("hsapiens_gene_id")["symbol"])
    cl.index = t.index
    t = pd.concat([t, cl], axis=1)
    return t.reset_index()


def patch_identity():
    """Add id_true to tables built before the column existed."""
    for c in RU.COHORTS:
        f = GDIR / f"{c}.parquet"
        t = pd.read_parquet(f)
        t["id_true"] = true_identity(c, t["hsapiens_gene_id"])
        t.to_parquet(f)
        print(c, t["id_true"].describe().round(3).to_dict(), int((t.id_true >= 0.99).sum()), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "patch_identity":
        return patch_identity()
    GDIR.mkdir(parents=True, exist_ok=True)
    AN.compara()
    gold = pd.read_csv(RES / "T4_3/golds/gold_table.csv", index_col=0).drop(columns=["chromosome"])

    def one(c):
        t = build(c, gold)
        t.to_parquet(GDIR / f"{c}.parquet")
        print(c, t.shape, t.split.value_counts().to_dict(), flush=True)

    Parallel(n_jobs=4, backend="loky")(delayed(one)(c) for c in RU.COHORTS)


if __name__ == "__main__":
    main()
