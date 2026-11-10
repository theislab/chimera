"""Y3 (EXPLORATORY): does the liver-pattern information of B1z belong to Chimera's latent?

X6 found that profile correlation over Leiden clusters of Chimera's latent (B1z) carries the most unique
liver-pattern information (0.104), and left open whether another integrator's latent does as well. This
applies the T4.2 B1z recipe, unchanged, to the other latents already on disk. No training. Chosen after the
scoreboard and after X6, so it is exploratory and changes no status, row or modifier.

Definitions, fixed before any association below was computed:
    cohorts   liv_hmouse, test_hmouse (X6's two cohorts).
    latents   Chimera canonical s42 (must reproduce T4.2 B1z_raw); scVI s42 with species as batch and with
              library as batch; SATURN s42 and CAMEX s42 as scored in the benchmark (z_mean.npy).
    recipe    T4_2 leiden_clusters (15-NN exact, resolution 1, igraph, fixed seed), pseudobulk per library with
              at least 20 cells per cluster, pooled_E over the species pair.
    golds     G_H3 (C_pat), G_H4 (C_pat + H4_nsp), G_H4_primates (C_pat + H4_npri), as X6; level golds G_H1, G_H2
              (C) reported beside them.
    universe  one per gold and variant: genes with the gold, covariates, D, S, the X6 panel, B1z_chimera and that variant.
              Amendment after the first run, in which no association was computed because the joint universe was
              empty: scVI and SATURN latents leave at most 1-9 Leiden clusters holding both species, so B1z is
              undefined for most genes on them. The cluster table is therefore itself a result.
    tests     (a) rho_p of each variant given covariates; delta to Chimera's B1z, joint block bootstrap.
              (b) each variant given covariates + the other simple baselines B1b, B1c, B2b, B3a (X6's 'other
              simple baselines'). (c) Chimera's B1z given the variant, and the variant given Chimera's B1z.

Writes results/analysis/Y3/b1z_latents.csv, b1z_clusters.csv and cache/b1z__<cohort>.parquet.

Usage: PYTHONPATH=$PWD python -m analysis.tasks.Y3_findings.latent_b1z
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import cohorts as C
from analysis.common import gene_ids
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.X4_where_differs import common as CM

OUT = gene_ids.REPO_ROOT / "results" / "analysis" / "Y3"
CACHE = OUT / "cache"
RUNS = gene_ids.REPO_ROOT / "training" / "runs"
COHORTS = ["liv_hmouse", "test_hmouse"]
LIB = {"liv_hmouse": "batch", "test_hmouse": "sample_batch"}
OTHERS = ["B1b_raw", "B1c_raw", "B2b", "B3a"]
BASE = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]
PAT = BASE + ["tau_shared"]
GOLDS = {"liv_hmouse": [("G_H1", BASE), ("G_H3", PAT)],
         "test_hmouse": [("G_H2", BASE), ("G_H4", PAT + ["H4_nsp"]), ("G_H4_primates", PAT + ["H4_npri"])]}


def latents(c: str, a) -> dict[str, np.ndarray]:
    stem = gene_ids.H5ADS[c]
    out = {"chimera": T42.chimera_latent(c, a)}
    out["scvi_species"] = np.load(RUNS / "baselines_scvi" / f"{stem}-s42-e100-nl10" / "z_mean.npy")
    out["scvi_library"] = np.load(RUNS / "baselines_scvi" / f"{stem}-s42-e100-nl10-bk_{LIB[c]}" / "z_mean.npy")
    out["saturn"] = np.load(RUNS / "baselines_saturn" / f"{stem}-s42" / "z_mean.npy")
    out["camex"] = np.load(RUNS / "baselines_camex" / f"{stem}-s42" / "z_mean.npy")
    for k, z in out.items():
        if z.shape[0] != a.n_obs:
            raise SystemExit(f"{c} {k}: {z.shape[0]} cells vs {a.n_obs}")
    return out


def b1z(c: str) -> tuple[pd.DataFrame, list[dict]]:
    p = CACHE / f"b1z__{c}.parquet"
    if p.exists() and (OUT / "b1z_clusters.csv").exists():
        old = pd.read_csv(OUT / "b1z_clusters.csv")
        keep = old[(old.cohort == c) & old.n_clusters.notna()][["cohort", "latent", "n_clusters", "n_clusters_all_species"]]
        return pd.read_parquet(p), keep.to_dict("records")
    a = C.read_adata(c)
    X = (a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)).astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    lib = a.obs[LIB[c]].astype(str).to_numpy()
    species = sorted(set(spc))
    tab, info = pd.DataFrame({"var_name": np.array(a.var_names, dtype=str)}), []
    for k, z in latents(c, a).items():
        cl = T42.leiden_clusters(np.asarray(z, dtype=np.float32))
        x, f, _ = T42.pseudobulk(X, spc, cl, lib, T42.MIN_CELLS_CLUSTER)
        tab[f"B1z_{k}"] = T42.pooled_E(x, f, species, a.n_vars)[0]
        shared = sum(all((s, t) in x for s in species) for t in sorted(set(cl)))
        info.append(dict(cohort=c, latent=k, n_clusters=int(pd.Series(cl).nunique()), n_clusters_all_species=shared))
        print(c, k, info[-1], flush=True)
    tab.to_parquet(p)
    return tab, info


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    rows, infos = [], []
    for c in COHORTS:
        tab, info = b1z(c)
        infos += info
        g = CM.load(c).merge(tab, on="var_name", how="left")
        delta = (g["B1z_chimera"] - g["B1z_raw"]).abs().max()
        print(c, "reproduction of T4.2 B1z_raw, max abs diff", delta, flush=True)
        infos[-1]["repro_max_abs_diff"] = float(delta)
        var = [f"B1z_{k}" for k in ("chimera", "scvi_species", "scvi_library", "saturn", "camex")]
        units = []
        for v in var[1:]:
            ok = g[["D", "S"] + OTHERS + ["B1z_chimera", v]].notna().all(1).to_numpy()
            infos.append(dict(cohort=c, latent=v, n_genes_defined=int(g[v].notna().sum()), n_genes_chimera=int(g["B1z_chimera"].notna().sum()),
                              n_genes_common_panel=int(ok.sum())))
            if ok.sum() < 30:
                continue
            for gold, covs in GOLDS[c]:
                base = dict(cohort=c, variant=v)
                units.append(dict(g=g, gold=gold, covs=covs, scores=["B1z_chimera", v], rows=ok, label=dict(base, given="none"), ref="B1z_chimera"))
                for w in ("B1z_chimera", v):
                    units.append(dict(g=g, gold=gold, covs=covs, scores=[w], rows=ok, extra_spline=OTHERS, label=dict(base, given="other simple baselines")))
                units.append(dict(g=g, gold=gold, covs=covs, scores=["B1z_chimera"], rows=ok, extra_spline=[v], label=dict(base, given=v)))
                units.append(dict(g=g, gold=gold, covs=covs, scores=[v], rows=ok, extra_spline=["B1z_chimera"], label=dict(base, given="B1z_chimera")))
        if units:
            rows += CM.run_units(units).to_dict("records")
    pd.DataFrame(rows).to_csv(OUT / "b1z_latents.csv", index=False)
    pd.DataFrame(infos).to_csv(OUT / "b1z_clusters.csv", index=False)


if __name__ == "__main__":
    main()
