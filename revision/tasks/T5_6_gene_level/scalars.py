"""T5.6: per-ortholog scalars of SATURN and CAMEX (chimera_env, reads run outputs only).

Fixed before any tool scalar was joined to a gold standard (05 section 5, A4 action 5). The scalars
are our derivations, a fair-faith attempt, not outputs the tools were built to give.

Cohorts liv_hmouse, test_hmouse; seeds 42, 43, 44. Arms: ``saturn_native`` (T5.3 s42, T5.6 s43, s44),
``saturn_1to1`` (Adapter B), ``camex`` (T5.3). A pair (h, m) is the cohort's 1:1 ortholog; symbols
from the ortholog table.

SATURN, w = final gene-to-macrogene weights (``*_genes_to_macrogenes.pkl``):
  S_deploy  PRIMARY. r_m = Pearson across the cohort's shared cell types (T4.2 ``all_species_types``;
            testis through the T4.2 coarse map) of human and mouse cell-type means of the macrogene
            value (final h5ad ``obsm["macrogenes"]``). S_deploy = sum_m wbar_m r_m / sum_m wbar_m,
            wbar_m = (w_h,m + w_m,m) / 2; macrogenes with undefined r_m leave both sums.
  S_cos     cos(w_h, w_m). S_shift = S_cos(final) - S_cos(initial, ``centroids_s<seed>.pkl``).
  S_topk    Jaccard of the top 10 macrogenes of h and of m. S_top1: same top macrogene (0/1).
CAMEX, h = gene embedding (``gene_integration_hidden.h5ad``):
  C_cos     PRIMARY. cos(h_h, h_m).
  C_module  same Leiden module on the joint 15-NN graph of all human and mouse gene embeddings;
            resolution 1.0 (C_module_r1), sweep 0.5 and 2 (igraph flavour, 2 iterations, seed 0).
Secondary scalars are exploratory. The seed value of a scalar is the mean over the three seeds and
needs a value in all three, matching D_3 (mean of alpha - prior over seeds 42-44).

Writes results/revision/T5_6/<cohort>__<arm>.csv (one row per gene the arm covers) and, for T5.6
analysis.py, results/revision/T5_6/modules/<cohort>__<arm>__s<seed>.npz.
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from revision.common import gene_ids
from revision.tasks.T4_2_gene_baselines import run as T42
from revision.tasks.T4_3_scoreboard import inputs as I

REPO = gene_ids.REPO_ROOT
RES = REPO / "results/revision"
OUT = RES / "T5_6"
STEMS = {"liv_hmouse": "camex_human_mouse", "test_hmouse": "camex_testis_human_mouse"}
SEEDS = [42, 43, 44]
CAMEX_DATASET = {"liv_hmouse": ("raw-liver-human-Martin", "raw-liver-mouse-Martin"),
                 "test_hmouse": ("raw-testis-human-Murat", "raw-testis-mouse-Murat")}
TOPK = 10
RESOLUTIONS = {"C_module_r05": 0.5, "C_module_r1": 1.0, "C_module_r2": 2.0}
SATURN_SCALARS = ["S_deploy", "S_cos", "S_shift", "S_topk", "S_top1"]
CAMEX_SCALARS = ["C_cos"] + list(RESOLUTIONS)


def base_table(cohort: str) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    d = I.load(cohort, RES / "T4_3/cache", consensus=True)
    tab = d["tab"]
    seeds = sorted(d["meta"]["seeds"])
    idx = [seeds.index(s) for s in SEEDS]
    M3 = d["M"][:, idx]
    t = gene_ids.load_table(cohort)
    ids = tab["var_name"].str.replace("ortho_", "", regex=False).astype(int)
    tab = tab.copy()
    tab["symbol_h"] = t.loc[ids, "hsapiens_gene_name"].values
    tab["symbol_m"] = t.loc[ids, "mmusculus_gene_name"].values
    return tab, M3, idx


def saturn_dir(arm: str, stem: str, seed: int) -> Path:
    if arm == "saturn_1to1":
        return RES / f"T5_6/saturn_1to1/{stem}-s{seed}"
    return RES / (f"T5_3/saturn/{stem}-s42" if seed == 42 else f"T5_6/saturn/{stem}-s{seed}")


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)


def macrogene_r(work: Path, cohort: str, stem: str) -> np.ndarray:
    f = [p for p in glob.glob(str(work / "saturn_results/*.h5ad"))
         if not p.endswith("_pretrain.h5ad") and "_ep_" not in Path(p).name]
    if len(f) != 1:
        raise SystemExit(f"{work}: expected one final h5ad, found {f}")
    a = ad.read_h5ad(f[0])
    p = ad.read_h5ad(gene_ids.h5ad_path(cohort), backed="r")
    obs = p.obs[["species", "cell_type"]].astype(str).copy()
    p.file.close()
    a = a[obs.index.to_numpy()]
    lab = obs["cell_type"]
    if cohort.startswith("test"):
        lab = lab.map(T42.TESTIS_COARSE)
    lab = lab.to_numpy()
    spc = obs["species"].to_numpy()
    X = np.asarray(a.obsm["macrogenes"])
    types = I.shared_types(cohort)
    H = np.column_stack([X[(spc == "Human") & (lab == t)].mean(0) for t in types])
    M = np.column_stack([X[(spc == "Mouse") & (lab == t)].mean(0) for t in types])
    H0, M0 = H - H.mean(1, keepdims=True), M - M.mean(1, keepdims=True)
    den = np.sqrt((H0 ** 2).sum(1) * (M0 ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (H0 * M0).sum(1) / den, np.nan)


def saturn_seed(arm: str, cohort: str, tab: pd.DataFrame, seed: int, mod_dir: Path) -> pd.DataFrame:
    stem = STEMS[cohort]
    w = saturn_dir(arm, stem, seed)
    pk = glob.glob(str(w / "saturn_results/*_genes_to_macrogenes.pkl"))
    if len(pk) != 1:
        raise SystemExit(f"{w}: no unique macrogene pickle")
    final = pickle.load(open(pk[0], "rb"))
    init = pickle.load(open(w / f"centroids_s{seed}.pkl", "rb"))
    keys_h = "human_" + tab["symbol_h"].astype(str)
    keys_m = "mouse_" + tab["symbol_m"].astype(str)
    ok = np.array([a in final and b in final for a, b in zip(keys_h, keys_m)])
    Wh = np.vstack([final[k] for k in keys_h[ok]]).astype(np.float64)
    Wm = np.vstack([final[k] for k in keys_m[ok]]).astype(np.float64)
    Ih = np.vstack([init[k] for k in keys_h[ok]]).astype(np.float64)
    Im = np.vstack([init[k] for k in keys_m[ok]]).astype(np.float64)
    r = macrogene_r(w, cohort, stem)
    good = np.isfinite(r)
    wbar = (Wh + Wm) / 2
    dep = (wbar[:, good] * r[good]).sum(1) / wbar[:, good].sum(1)
    sc_f, sc_i = cos(Wh, Wm), cos(Ih, Im)
    th = np.argsort(-Wh, axis=1)[:, :TOPK]
    tm = np.argsort(-Wm, axis=1)[:, :TOPK]
    jac = np.array([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(th, tm)])
    res = pd.DataFrame({"var_name": tab["var_name"].to_numpy()[ok], "S_deploy": dep, "S_cos": sc_f,
                        "S_shift": sc_f - sc_i, "S_topk": jac, "S_top1": (th[:, 0] == tm[:, 0]).astype(float)})
    # Module membership for L2: weight >= 0.5, else the top 10 of the species (A4 action 6)
    idx = np.where(ok)[0]
    rows, cols = [], []
    n_m = Wh.shape[1]
    for W in (Wh, Wm):
        for m in range(n_m):
            g = np.where(W[:, m] >= 0.5)[0]
            if len(g) == 0:
                g = np.argsort(-W[:, m])[:TOPK]
            rows.extend([m] * len(g))
            cols.extend(idx[g].tolist())
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_m, len(tab)))
    A.data[:] = 1
    hard = np.full(len(tab), -1, dtype=int)
    hard[idx] = wbar.argmax(1)
    # top genes per macrogene (human symbols) for the native arm
    np.savez_compressed(mod_dir / f"{cohort}__{arm}__s{seed}.npz", indptr=A.indptr, indices=A.indices,
                        shape=np.array(A.shape), hard=hard, macrogene_r=r, ok=ok)
    print(f"{cohort} {arm} s{seed}: {int(ok.sum())} pairs, {int(good.sum())} macrogenes with r, "
          f"weight>=0.5 per macrogene (median, human) {np.median((Wh >= 0.5).sum(0))}", flush=True)
    return res.set_index("var_name")


def camex_seed(cohort: str, tab: pd.DataFrame, seed: int, mod_dir: Path) -> pd.DataFrame:
    stem = STEMS[cohort]
    a = ad.read_h5ad(RES / f"T5_3/camex/{stem}-s{seed}/gene_integration_hidden.h5ad")
    a.obs_names = a.obs_names.astype(str)
    a.obs["gene"] = a.obs_names
    a.obs_names_make_unique()
    dh, dm = CAMEX_DATASET[cohort]
    b = a.obs["batch"].astype(str).to_numpy()
    X = np.asarray(a.X, dtype=np.float64)
    gene = a.obs["gene"].to_numpy()
    pos_h = {g: i for i, g in zip(np.where(b == dh)[0], gene[b == dh])}
    pos_m = {g: i for i, g in zip(np.where(b == dm)[0], gene[b == dm])}
    if not pos_h or not pos_m:
        raise SystemExit(f"{cohort}: dataset names not found in {sorted(set(b))}")
    ih = tab["symbol_h"].map(pos_h)
    im = tab["symbol_m"].map(pos_m)
    ok = (ih.notna() & im.notna()).to_numpy()
    ih, im = ih[ok].astype(int).to_numpy(), im[ok].astype(int).to_numpy()
    res = pd.DataFrame({"var_name": tab["var_name"].to_numpy()[ok], "C_cos": cos(X[ih], X[im])})
    g = ad.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32))
    g.obsm["emb"] = X.astype(np.float32)
    sc.pp.neighbors(g, n_neighbors=15, use_rep="emb", random_state=0)
    labs = {}
    for name, res_v in RESOLUTIONS.items():
        sc.tl.leiden(g, resolution=res_v, flavor="igraph", n_iterations=2, directed=False, random_state=0, key_added=name)
        lab = g.obs[name].astype(int).to_numpy()
        labs[name] = lab
        res[name] = (lab[ih] == lab[im]).astype(float)
    # L2 and native arm: gene pair belongs to the modules of its two members
    lab = labs["C_module_r1"]
    idx = np.where(ok)[0]
    n_m = int(lab.max()) + 1
    rows = np.concatenate([lab[ih], lab[im]])
    cols = np.concatenate([idx, idx])
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_m, len(tab)))
    A.data[:] = 1
    hard = np.full(len(tab), -1, dtype=int)
    hard[idx] = lab[ih]
    np.savez_compressed(mod_dir / f"{cohort}__camex__s{seed}.npz", indptr=A.indptr, indices=A.indices,
                        shape=np.array(A.shape), hard=hard, ok=ok, n_genes_embedding=X.shape[0])
    print(f"{cohort} camex s{seed}: {int(ok.sum())} pairs, {n_m} modules at res 1.0, {X.shape[0]} gene nodes",
          flush=True)
    return res.set_index("var_name")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=list(STEMS))
    ap.add_argument("--arms", nargs="+", default=["saturn_native", "saturn_1to1", "camex"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--outdir", type=Path, default=OUT)
    a = ap.parse_args()
    mod_dir = a.outdir / "modules"
    mod_dir.mkdir(parents=True, exist_ok=True)
    for cohort in a.cohorts:
        tab, M3, _ = base_table(cohort)
        d3 = M3.mean(1)
        for arm in a.arms:
            frames = {}
            for s in a.seeds:
                frames[s] = camex_seed(cohort, tab, s, mod_dir) if arm == "camex" else saturn_seed(arm, cohort, tab, s, mod_dir)
            names = CAMEX_SCALARS if arm == "camex" else SATURN_SCALARS
            out = tab[["var_name", "hsapiens_gene_id", "symbol_h", "symbol_m", "prior_raw", "log_cp10k_spmean",
                       "detection_spmean", "log_cds_length", "block"]].copy()
            out["D_3"] = d3 - np.median(d3)
            for k in range(M3.shape[1]):
                out[f"D_s{SEEDS[k]}"] = M3[:, k]
            for name in names:
                cols = []
                for s in a.seeds:
                    c = f"{name}_s{s}"
                    out[c] = frames[s][name].reindex(out["var_name"]).to_numpy()
                    cols.append(c)
                out[name] = out[cols].mean(1, skipna=False)
            out = out[out[names[0]].notna()]
            out.to_csv(a.outdir / f"{cohort}__{arm}.csv", index=False)
            print(f"{cohort} {arm}: {len(out)} of {len(tab)} genes in all seeds", flush=True)


if __name__ == "__main__":
    main()
