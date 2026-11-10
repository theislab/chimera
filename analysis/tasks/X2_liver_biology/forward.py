"""X2 (EXPLORATORY): checkpoint forward pass with hepatocytes resolved along the porto-central axis.

Same per-cell log-likelihood ratio as T2.5 (``T2_5_celltype_margin/run.py``: shared decoder on
the posterior-mean z against the model's per-species null rate, NB with the trained per-batch
dispersion), summed per (species, cell type, zone). Hepatocytes get a zone (PP, mid, PC:
within-species tertiles of a landmark zonation score, ``sets.ZONE_SCORE_*``); other types get
zone "all". Also accumulated per group:

* ``sum_wt``: the same ratio against a within-type null (Laplace-smoothed rate of each
  (species, cell type), as T2.5, but on the posterior-mean z; T2.5 used one posterior sample);
* ``sum_pred``: the shared decoder's predicted composition mu1 / sum(mu1), i.e. what the
  species-blind decoder expects for these cells;
* ``sum_obs`` and ``sum_lib``: observed counts and library sizes.

Zonation score per hepatocyte (sets.py amendment 1): from the raw full-transcriptome object of
the species, restricted to the hepatocytes of the cohort; log1p CP10K of each out-of-vocabulary
landmark detected in >= 5 % of them, smoothed over a within-species 15-NN graph (20 PCs of
2,000 HVGs of the raw object, plus self, row-normalised), z-scored; score = mean(z, pericentral)
- mean(z, periportal). Axis check per species: Spearman(PC part, PP part); the zonation
analysis uses only species where it is <= AXIS_RHO (rule fixed before D was read).

No training; reads run dirs and data/processed only. Writes results/analysis/X2/forward/.

Usage: python analysis/tasks/X2_liver_biology/forward.py --label liv_4sp [--threads 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common import cohorts as C
from analysis.common import gene_ids

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "T2_5_celltype_margin"))
import sets as ST  # noqa: E402
import run as T25  # noqa: E402  (load_components, nb_llr, RUNS)

OUT = gene_ids.REPO_ROOT / "results" / "analysis" / "X2" / "forward"
EXTRA = {f"liv_4sp_s{s}": ("liv_4sp", f"rev-liv_4sp-s{s}") for s in (43, 44, 45, 46)}
EXTRA.update({f"liv_hmouse_s{s}": ("liv_hmouse", f"rev-liv_hmouse-s{s}") for s in range(45, 52)})
MIN_DET = 0.05
AXIS_RHO = -0.3


RAW_DIR = gene_ids.REPO_ROOT / "data" / "raw" / "camex" / "1liver" / "dataset"


def _smoother(Xs, lib):
    """Row-normalised (15-NN connectivities + identity) of a within-species hepatocyte PCA (raw genes)."""
    import anndata as ad
    import scanpy as sc
    import scipy.sparse as sp

    a = ad.AnnData(Xs.astype(np.float32))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    sc.pp.pca(a, n_comps=20, random_state=0)
    sc.pp.neighbors(a, n_neighbors=15, random_state=0)
    W = a.obsp["connectivities"] + sp.eye(a.n_obs)
    return sp.csr_matrix(W.multiply(1.0 / W.sum(1)))


def _side_score(Xs, var, genes, lib, W, s, side, version, info):
    zs = []
    for gname in genes:
        j = np.where(var == gname)[0]
        if not len(j):
            info.append(dict(species=s, side=side, gene=gname, version=version, status="absent_from_raw"))
            continue
        v = np.asarray(Xs[:, j].sum(1)).ravel()  # duplicated raw symbols are summed
        det = float((v > 0).mean())
        if det < MIN_DET:
            info.append(dict(species=s, side=side, gene=gname, version=version, status="undetected", detection=det))
            continue
        e = W @ np.log1p(1e4 * v / np.maximum(lib, 1))
        zs.append((e - e.mean()) / e.std())
        info.append(dict(species=s, side=side, gene=gname, version=version, status="used", detection=det))
    return np.mean(zs, 0) if zs else None


def zonation_scores(adata, cohort: str):
    """Per-hepatocyte porto-central score from the raw full-transcriptome object of each species."""
    import anndata as ad
    import scipy.sparse as sp

    obs = adata.obs
    rows, info = [], []
    hep = (obs["cell_type"].astype(str) == "Hepatocytes").to_numpy()
    for s in sorted(obs["species"].astype(str).unique()):
        m = hep & (obs["species"].astype(str) == s).to_numpy()
        if not m.any():
            continue
        raw = ad.read_h5ad(RAW_DIR / ST.RAW_LIVER[s])
        cells = obs.index[m]
        missing = cells.difference(raw.obs_names)
        if len(missing):
            raise AssertionError(f"{s}: {len(missing)} hepatocytes not in the raw object")
        raw = raw[cells]
        Xs = raw.X.tocsr() if sp.issparse(raw.X) else sp.csr_matrix(raw.X)
        var = np.asarray(raw.var_names, dtype=str)
        lib = np.asarray(Xs.sum(1)).ravel()
        W = _smoother(Xs, lib)
        out = dict(cell=cells, species=s)
        for version, extra in (("primary", None), ("extended", ST.ZONE_SCORE_NATIVE_EXTRA[s])):
            parts = {}
            for side in ("PC", "PP"):
                genes = list(ST.ZONE_SCORE_NATIVE[s][side]) + (list(extra[side]) if extra else [])
                parts[side] = _side_score(Xs, var, genes, lib, W, s, side, version, info)
            if parts["PC"] is None and parts["PP"] is None:
                score = np.full(m.sum(), np.nan)
            elif parts["PP"] is None:
                score = parts["PC"]
            elif parts["PC"] is None:
                score = -parts["PP"]
            else:
                score = parts["PC"] - parts["PP"]
            tag = "" if version == "primary" else "_ext"
            out[f"pc_score{tag}"] = parts["PC"] if parts["PC"] is not None else np.nan
            out[f"pp_score{tag}"] = parts["PP"] if parts["PP"] is not None else np.nan
            out[f"score{tag}"] = score
            if np.isfinite(score).all():
                q = np.quantile(score, [1 / 3, 2 / 3])
                zone = np.where(score <= q[0], "PP", np.where(score <= q[1], "mid", "PC"))
            else:
                zone = np.full(m.sum(), "NA")
            out[f"zone{tag}"] = zone
            if parts["PC"] is not None and parts["PP"] is not None:
                rho = float(pd.Series(parts["PC"]).corr(pd.Series(parts["PP"]), method="spearman"))
            else:
                rho = np.nan
            info.append(dict(species=s, side="PC_vs_PP", gene="spearman", version=version, status="axis_check",
                             detection=rho))
        rows.append(pd.DataFrame(out))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(info)


def forward(label: str, threads: int, batch_size: int = 512) -> None:
    import scipy.sparse as sp
    import torch

    cohort, run = T25.RUNS[label] if label in T25.RUNS else EXTRA[label]
    torch.set_num_threads(threads)
    t0 = time.time()
    run_dir = C.RUNS_DIR / run
    cfg = json.loads((run_dir / "config.json").read_text())
    adata = C.read_adata(cohort)
    comp = T25.load_components(run_dir, adata, cfg)
    fs = C.load_state(run)
    if not np.allclose(comp["S_ema"], fs["S_g_ema"], atol=1e-6):
        raise AssertionError(f"{run}: checkpoint S_g_ema differs from final_state.npz")
    zt, zinfo = zonation_scores(adata, cohort)
    zone = pd.Series("all", index=adata.obs.index, dtype=object)
    zone.loc[zt["cell"].to_numpy()] = zt["zone"].to_numpy()
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    N, G = X.shape
    s_idx = comp["species_codes"]
    b_idx = comp["batch_codes"]
    species = comp["species_names"]
    ct = adata.obs["cell_type"].astype(str).to_numpy()
    sp_obs = adata.obs["species"].astype(str).to_numpy()
    if not (np.array(species)[s_idx] == sp_obs).all():
        raise AssertionError("species codes do not match obs")
    lab = np.array([f"{species[s]}|{t}|{z}" for s, t, z in zip(s_idx, ct, zone.to_numpy())])
    groups = sorted(set(lab))
    gi = {g: i for i, g in enumerate(groups)}
    cg = np.array([gi[x] for x in lab])
    # within-type null per (species, cell type), pooled over zones
    st_lab = np.array([f"{species[s]}|{t}" for s, t in zip(s_idx, ct)])
    st_groups = sorted(set(st_lab))
    sti = {g: i for i, g in enumerate(st_groups)}
    stg = np.array([sti[x] for x in st_lab])
    lib = np.asarray(X.sum(1)).ravel().astype(np.float64)
    p_wt = np.zeros((len(st_groups), G))
    for j in range(len(st_groups)):
        m = stg == j
        p_wt[j] = (np.asarray(X[m].sum(0)).ravel() + 1.0) / (lib[m].sum() + G)
    p_wt = torch.tensor(p_wt)
    K = len(groups)
    acc = {k: np.zeros((K, G)) for k in ("sum_l", "sum_wt", "sum_pred", "sum_obs")}
    sum_lib = np.zeros(K)
    zmean = np.zeros((N, int(cfg.get("n_latent", 10))), dtype=np.float32)
    with torch.no_grad():
        for st in range(0, N, batch_size):
            idx = np.arange(st, min(st + batch_size, N))
            x = torch.tensor(X[idx].toarray(), dtype=torch.float32)
            s_t = torch.tensor(s_idx[idx], dtype=torch.long)
            b_t = torch.tensor(b_idx[idx], dtype=torch.long)
            L = x.sum(1, keepdim=True).double()
            zmu, _ = comp["encode"](x, s_t, b_t)
            if zmean.shape[1] != zmu.shape[1]:
                zmean = np.zeros((N, zmu.shape[1]), dtype=np.float32)
            zmean[idx] = zmu.numpy()
            xd = x.double()
            th = comp["theta"][b_t]
            mu1 = torch.exp(comp["shared_rate"](zmu).double()) * L
            Lt = mu1.sum(1, keepdim=True).clamp(min=1.0)
            mu0 = (comp["p_null"][s_t] * Lt).clamp(min=1e-8)
            mu0w = (p_wt[torch.tensor(stg[idx], dtype=torch.long)] * Lt).clamp(min=1e-8)
            g_idx = cg[idx]
            np.add.at(acc["sum_l"], g_idx, T25.nb_llr(xd, mu1, mu0, th).numpy())
            np.add.at(acc["sum_wt"], g_idx, T25.nb_llr(xd, mu1, mu0w, th).numpy())
            np.add.at(acc["sum_pred"], g_idx, (mu1 / Lt).numpy())
            np.add.at(acc["sum_obs"], g_idx, xd.numpy())
            np.add.at(sum_lib, g_idx, L.squeeze(1).numpy())
    n = np.bincount(cg, minlength=K)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{label}.npz", groups=np.array(groups), n=n, sum_lib=sum_lib,
                        var_names=np.array(adata.var_names, dtype=str), run=np.array(run),
                        source=np.array(comp["source"]), **acc)
    np.save(OUT / f"{label}__zmean.npy", zmean)
    zt.to_csv(OUT / f"{label}__zones.csv", index=False)
    zinfo.to_csv(OUT / f"{label}__zone_landmarks.csv", index=False)
    # check: summed over groups, the species-null sum per species equals the T2.5 per-species sum
    print(f"  {label}: {run}, {comp['source']}, {N} cells, {K} groups, {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    for lab in args.label:
        forward(lab, args.threads)


if __name__ == "__main__":
    main()
