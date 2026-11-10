"""D1 gene tables, one per cohort: readouts, covariates, golds, flags, strata, feature-only comparators. No association with a gold is computed here.
    python -m analysis.tasks.D1_sparse_genes.build paralogs      closest paralog pairs from Compara 110
    python -m analysis.tasks.D1_sparse_genes.build <cohort>      writes results/analysis/D1/genes/<cohort>.parquet, features/<cohort>.npy
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("SLURM_CPUS_PER_TASK", "4"))

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from analysis.common import gene_ids
from analysis.tasks.B2_benchmark import inputs as IN
from analysis.tasks.B2_benchmark import perturb as P
from analysis.tasks.D1_sparse_genes import rules as RU
from analysis.tasks.H1_harmonise import rules as H1R
from analysis.tasks.T4_2_gene_baselines import run as T42

COMPARA = RU.REPO / "data/analysis/ensembl110/Compara.110.protein_default.homologies.tsv.gz"
PARA = RU.RES / "cache" / "paralogs.parquet"
ODD = {str(i) for i in range(1, 23, 2)}
EVEN = {str(i) for i in range(2, 23, 2)}


def paralogs() -> None:
    use = ["gene_stable_id", "species", "identity", "homology_type", "homology_gene_stable_id", "homology_species"]
    para = []
    for ch in pd.read_csv(COMPARA, sep="\t", usecols=use, chunksize=2_000_000, dtype=str):
        ch = ch[(ch.species == "homo_sapiens") & ch.homology_type.eq("within_species_paralog") & ch.homology_species.eq("homo_sapiens")]
        para.append(ch[["gene_stable_id", "homology_gene_stable_id", "identity"]])
    p = pd.concat(para, ignore_index=True)
    p["identity"] = pd.to_numeric(p["identity"], errors="coerce")
    PARA.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(PARA)
    print(len(p), flush=True)


def knn_mean(E: np.ndarray, train: np.ndarray, values: dict[str, np.ndarray], k: int = RU.KNN) -> dict[str, np.ndarray]:
    """Mean of each value over the k nearest train genes in E, self excluded."""
    idx = np.flatnonzero(train)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E[idx])
    ind = nn.kneighbors(E, return_distance=False)
    out = {}
    pick = idx[ind]
    self_hit = pick == np.arange(len(E))[:, None]
    w = (~self_hit).astype(float)
    w[~self_hit.any(1), -1] = 0.0  # non-train gene: first k neighbours
    w /= w.sum(1, keepdims=True)
    for n, v in values.items():
        vv = v[pick]
        mask = np.isfinite(vv)
        out[n] = (np.where(mask, vv, 0.0) * w).sum(1) / np.maximum((mask * w).sum(1), 1e-9)
    return out


def feature_only(t: pd.DataFrame, F: np.ndarray, train: np.ndarray, S: str = "S_ema") -> pd.DataFrame:
    """Ridge and neighbour comparators built on genes in ``train`` only; a gene outside it is predicted by models that never saw it."""
    Fp = F[:, RU.PROT_COLS]
    E = F[:, RU.ESM_COLS]
    out = pd.DataFrame(index=t.index)
    tg = {"S": t[S].to_numpy(float), "Sdir": t["Sdir"].to_numpy(float), "B1": t["B1"].to_numpy(float), "D": t["D"].to_numpy(float)}
    for n, v in tg.items():
        y = np.where(train & np.isfinite(v), v, np.nan)
        out[f"R_{n}_p"] = T42.cross_fit_ridge(Fp, y)[0]
        out[f"R_{n}_72"] = T42.cross_fit_ridge(F, y)[0]
        out.attrs[f"R2_{n}_p"] = None
    k = knn_mean(E, train, {f"kNN_{n}": v for n, v in tg.items()})
    for n, v in k.items():
        out[n] = v
    b = t["block"].to_numpy()
    sv = pd.Series(np.where(train, tg["S"], np.nan), index=t.index)
    grp = sv.groupby(b)
    su, cn = grp.transform("sum").to_numpy(), grp.transform("count").to_numpy()
    own = np.where(train & np.isfinite(sv.to_numpy()), sv.to_numpy(), 0.0)
    cnt = cn - (train & np.isfinite(sv.to_numpy())).astype(int)
    out["Block_S"] = np.where(cnt > 0, (su - own) / np.maximum(cnt, 1), np.nan)
    return out


def para_s(t: pd.DataFrame, train: np.ndarray, S: str = "S_ema") -> pd.Series:
    p = pd.read_parquet(PARA)
    tr = pd.Series(t[S].to_numpy(float)[train], index=t["hsapiens_gene_id"].to_numpy()[train])
    tr = tr[~tr.index.duplicated()]
    p = p[p.homology_gene_stable_id.isin(tr.index)].sort_values("identity", ascending=False).drop_duplicates("gene_stable_id")
    return pd.Series(p.homology_gene_stable_id.map(tr).to_numpy(), index=p.gene_stable_id.to_numpy())


def build(c: str) -> None:
    d = H1R.COHORTS[c]
    a = ad.read_h5ad(d["h5ad"])
    ids = pd.Index(np.asarray(gene_ids.ensembl(d["table"], a.var_names)), name="hsapiens_gene_id")
    t = IN.cohort_table(c)
    h = pd.read_parquet(RU.REPO / f"results/analysis/H1/genes/{c}.parquet").set_index("hsapiens_gene_id")
    t = t.set_index("hsapiens_gene_id")
    pos = pd.Series(np.arange(len(ids)), index=ids)
    pos = pos[~pos.index.duplicated(keep=False)]
    t = t[t.index.isin(pos.index) & t.index.notna()]
    p = pos.reindex(t.index).to_numpy()
    for k in ("ppass", "nameable", "detection", "tau", "o1_T2", "o1_M_wrong_row", "o1_M_false_zero", "o1_zero_any", "pi_raw"):
        t[k] = h[k].reindex(t.index)
    b = pd.read_parquet(RU.B2.RES / "track_C" / f"baselines__{c}.parquet").set_index("hsapiens_gene_id")
    for k in ("B1", "B1b", "B1c", "B1z", "B2b", "B3a", "Sdir", "Lfc"):
        t[k] = b[k].reindex(t.index)
    f = RU.RES / "fwd" / f"{c}.parquet"
    if f.exists():
        fw = pd.read_parquet(f)
        t["S_fwd"] = fw["S_fwd"][~fw.index.duplicated(keep=False)].reindex(t.index)
    else:
        t["S_fwd"] = np.nan
    t["masked"] = t["S_ema"].abs() < 1e-6
    bad = sum(t[k].fillna(False).astype(bool) for k in ("o1_T2", "o1_M_wrong_row", "o1_M_false_zero")).astype(bool)
    t["clean"] = ~bad
    X = P.csr(a.X)
    spc = a.obs["species"].astype(str).to_numpy()
    ne = np.column_stack([np.asarray((X[spc == s] > 0).sum(0)).ravel() for s in sorted(set(spc))])
    t["n_expr_min"] = ne[p].min(1)
    t["n_expr_total"] = ne[p].sum(1)
    t["no_direct"] = t["masked"] & t["B1"].isna()
    t["sp_sparse"] = t["masked"] & (t["n_expr_min"] < RU.MIN_EXPR_CELLS)
    ch = pd.read_csv(RU.REPO / "results/analysis/X1/genes/chromosome_ensembl110.csv", dtype={"chrom": str}).set_index("hsapiens_gene_id")["chrom"]
    t["chrom"] = ch[~ch.index.duplicated()].reindex(t.index)
    t["split"] = np.where(t["chrom"].isin(ODD), "odd", np.where(t["chrom"].isin(EVEN), "even", "none"))
    t = t.reset_index()
    F = np.asarray(a.varm["gene_features"], dtype=float)[p]
    for i in range(64):
        t[f"esm{i}"] = F[:, i]
    train = ~t["masked"].to_numpy()
    fo = feature_only(t, F, train)
    for k in fo.columns:
        t[k] = fo[k].to_numpy()
    t["B3b"] = T42.cross_fit_ridge(F, t["D"].to_numpy(float))[0]
    t["Para_S"] = para_s(t, train).reindex(t.hsapiens_gene_id).to_numpy()
    t["pi"] = t["prior_raw"]
    RU.RES.joinpath("genes").mkdir(parents=True, exist_ok=True)
    t.to_parquet(RU.RES / "genes" / f"{c}.parquet")
    print(c, t.shape, "masked", int(t.masked.sum()), "clean masked", int((t.masked & t.clean).sum()), "no_direct", int((t.no_direct & t.clean).sum()),
          "sp_sparse", int((t.sp_sparse & t.clean).sum()), "S_fwd", int(t.S_fwd.notna().sum()), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "paralogs":
        paralogs()
    else:
        build(sys.argv[1])
