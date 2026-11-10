"""Per-cohort inputs of the scoreboard: covariates C (and tau for C_pat), ESM PCs, ESM blocks,
per-seed D, consensus D, baselines and the T2.5 margin-support flag. Gene key: human ENSG.

Readout (05 section 2): D = mean over the pooled seeds of (alpha - prior), minus the cohort
median, from the T3.3 consensus table; per-seed D is re-read from each run's final_state.npz
(read-only) for the leave-one-seed-out range and rel(D|C), and the mean is checked against
the consensus column. D_raw = mean alpha - pi_raw, minus its median (the alpha - pi_raw
sensitivity; identical to D on unclamped liver recipes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.tasks.T3_3_consensus import run as T33
from analysis.tasks.T4_2_gene_baselines import run as T42

REPO = gene_ids.REPO_ROOT
T33_DIR = REPO / "results/analysis/T3_3"
T42_DIR = REPO / "results/analysis/T4_2"
T25_DIR = REPO / "results/analysis/T2_5"
BASELINE_RAW = {"B1b": "B1b_raw", "B1c": "B1c_raw", "B1z": "B1z_raw", "B2b": "B2b", "B3a": "B3a"}
N_PCS = 64
KNN = 15
TARGET_MEDIAN = 20


def shared_types(cohort: str) -> list[str]:
    return json.loads((T42_DIR / f"summary__{cohort}.json").read_text())["all_species_types"]


def tau_shared(adata, cohort: str) -> np.ndarray:
    """Cell-type specificity tau over the types shared by all species, mean over species.

    A species with no counts in any shared type has no defined tau and is left out of the
    mean; a gene with no counts in any species gets tau = 0."""
    import scipy.sparse as sp

    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    X = X.astype(np.float64)
    lib = np.asarray(X.sum(1)).ravel()
    cp = sp.diags(1e4 / np.maximum(lib, 1)).dot(X).tocsr()
    lab = adata.obs["cell_type"].astype(str)
    if C.is_testis(cohort):
        lab = lab.map(T42.TESTIS_COARSE)
    lab = lab.to_numpy()
    spc = adata.obs["species"].astype(str).to_numpy()
    types = shared_types(cohort)
    taus = []
    for s in sorted(set(spc)):
        M = np.column_stack([np.asarray(cp[(spc == s) & (lab == t)].mean(0)).ravel() for t in types])
        t_s = C.tau_index(M)
        t_s[M.max(1) <= 0] = np.nan  # no expression in this species' shared types: tau undefined
        taus.append(t_s)
    T = np.column_stack(taus)
    out = np.full(T.shape[0], 0.0)  # expressed in no species: no specificity, tau = 0
    ok = np.isfinite(T).any(1)
    out[ok] = np.nanmean(T[ok], 1)
    return out


def nonzero_all_species(adata) -> np.ndarray:
    """True where the gene has a nonzero count total in every species of the cohort."""
    import scipy.sparse as sp

    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    spc = adata.obs["species"].astype(str).to_numpy()
    ok = np.ones(adata.n_vars, bool)
    for s in sorted(set(spc)):
        ok &= np.asarray(X[spc == s].sum(0)).ravel() > 0
    return ok


def esm_blocks(F: np.ndarray, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Leiden modules of a 15-NN graph on the 64 ESM PCs; resolution by bisection on a log scale
    so that the median module size is as close as possible to 20 genes. No gold is used."""
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(X=np.zeros((F.shape[0], 1), dtype=np.float32))
    a.obsm["X_esm"] = np.asarray(F, dtype=np.float32)
    sc.pp.neighbors(a, n_neighbors=KNN, use_rep="X_esm", transformer="sklearn", random_state=seed)

    def run(res):
        sc.tl.leiden(a, resolution=res, flavor="igraph", n_iterations=2, directed=False, random_state=seed,
                     key_added="m")
        lab = a.obs["m"].astype(int).to_numpy()
        return lab, float(np.median(np.bincount(lab)))

    lo, hi = np.log(0.5), np.log(500.0)
    best = None
    for _ in range(18):
        mid = (lo + hi) / 2
        lab, med = run(float(np.exp(mid)))
        if best is None or abs(med - TARGET_MEDIAN) < abs(best[1] - TARGET_MEDIAN):
            best = (lab, med, float(np.exp(mid)))
        if med > TARGET_MEDIAN:
            lo = mid
        else:
            hi = mid
    lab, med, res = best
    sizes = np.bincount(lab)
    return lab, dict(resolution=res, n_modules=int(len(sizes)), median_size=med,
                     q10=float(np.quantile(sizes, 0.1)), q90=float(np.quantile(sizes, 0.9)), max=int(sizes.max()))


def seed_runs_used(cohort: str) -> dict[int, str]:
    pool = {p["cohort"]: p for p in json.loads((T33_DIR / "seed_pooling.json").read_text())}
    used = pool[cohort]["used"]
    runs = {**T33.MPS.get(cohort, {}), **T33.cpu_runs(cohort)}
    # CPU seeds take precedence over an MPS run of the same number (T3.3 col())
    cpu = T33.cpu_runs(cohort)
    return {s: (cpu[s] if s in cpu else runs[s]) for s in used}


def seed_runs_available(cohort: str) -> dict[int, str]:
    """All finished seed runs (null-check mode, before the consensus exists)."""
    mps = T33.finished(T33.MPS.get(cohort, {}))
    cpu = T33.finished(T33.cpu_runs(cohort))
    return {**mps, **cpu}


def load(cohort: str, cache_dir: Path, consensus: bool = True) -> dict:
    """Everything the scoreboard needs for one cohort. ``consensus=False`` (null-check mode)
    builds a stand-in D from the finished seed runs; it is only ever used permuted."""
    cache = cache_dir / f"{cohort}__inputs.parquet"
    meta_f = cache_dir / f"{cohort}__inputs.json"
    if cache.exists() and meta_f.exists():
        tab = pd.read_parquet(cache)
        meta = json.loads(meta_f.read_text())
    else:
        a = C.read_adata(cohort)
        gene_ids.assert_table_matches(a, gene_ids.load_table(cohort))
        canon = C.CANONICAL[cohort]
        praw = C.raw_prior(cohort, canon, a.var["conservation_prior"].to_numpy(dtype=float))
        cov = K.cohort_covariates(cohort, adata=a, prior_raw=praw)
        tab = cov[["hsapiens_gene_id"] + K.COVARIATE_COLUMNS].copy()
        tab["tau_shared"] = tau_shared(a, cohort)
        tab["nonzero_all_species"] = nonzero_all_species(a)
        F = np.asarray(a.varm["gene_features"], dtype=float)
        for j in range(N_PCS):
            tab[f"esm{j}"] = F[:, j]
        blk, binfo = esm_blocks(F[:, :N_PCS])
        tab["block"] = blk
        tab["var_name"] = tab.index
        tab = tab.reset_index(drop=True)
        meta = dict(blocks=binfo, shared_types=shared_types(cohort), canonical=canon)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tab.to_parquet(str(cache) + ".tmp")
        Path(str(cache) + ".tmp").replace(cache)
        meta_f.write_text(json.dumps(meta, indent=2))
    tab = tab.copy()
    # seeds
    runs = seed_runs_used(cohort) if consensus else seed_runs_available(cohort)
    M = np.column_stack([K.read_run(r)["D"] for s, r in sorted(runs.items())])
    A = np.column_stack([K.read_run(r)["alpha"] for s, r in sorted(runs.items())])
    tab["S_ema"] = np.column_stack([K.read_run(r)["S_g_ema"] for s, r in sorted(runs.items())]).mean(1)
    meta["seeds"] = {int(s): r for s, r in sorted(runs.items())}
    mean_da = M.mean(1)
    tab["D"] = mean_da - np.median(mean_da)
    a_mean = A.mean(1)
    draw = a_mean - tab["prior_raw"].to_numpy()
    tab["D_raw"] = draw - np.median(draw)
    if consensus:
        cons = pd.read_csv(T33_DIR / f"{cohort}__consensus.csv")
        if not (cons["var_name"].to_numpy() == tab["var_name"].to_numpy()).all():
            raise AssertionError(f"{cohort}: consensus gene order differs")
        err = float(np.abs(cons["D"].to_numpy() - tab["D"].to_numpy()).max())
        if err > 1e-5:
            raise AssertionError(f"{cohort}: consensus D differs from the seed mean by {err}")
        meta["consensus_check_max_abs"] = err
        err_s = float(np.abs(cons["S_ema"].to_numpy() - tab["S_ema"].to_numpy()).max())
        if err_s > 1e-5:
            raise AssertionError(f"{cohort}: consensus S_ema differs from the seed mean by {err_s}")
        tab["D"] = cons["D"].to_numpy()
    # baselines
    b = pd.read_csv(T42_DIR / f"baselines__{cohort}.csv").set_index("var_name")
    for k, col in BASELINE_RAW.items():
        tab[k] = b[col].reindex(tab["var_name"]).to_numpy()
    sp_cols = [c for c in b.columns if c.startswith("n_types_det_")]
    det = b[sp_cols].reindex(tab["var_name"]).to_numpy()
    tab["low_detection"] = (det == 0).any(1)
    # margin support (T2.5, canonical run)
    t25 = T25_DIR / f"{cohort}__genes.parquet"
    if t25.exists():
        mg = pd.read_parquet(t25).set_index("var_name")
        for c in ("tail", "margin_support", "lacks_margin_support"):
            tab[c] = mg[c].reindex(tab["var_name"]).to_numpy()
    return dict(tab=tab, M=M, meta=meta, F=tab[[f"esm{j}" for j in range(N_PCS)]].to_numpy())


def b3b(F72: np.ndarray, d: np.ndarray) -> np.ndarray:
    """B3b on a given D: the T4.2 cross-fitted ridge (5 folds, seed 0) on the 72 features."""
    return T42.cross_fit_ridge(F72, np.asarray(d, dtype=float))[0]


def features72(cohort: str) -> np.ndarray:
    a = C.read_adata(cohort, backed=True)
    F = np.asarray(a.varm["gene_features"], dtype=float)
    a.file.close()
    return F
