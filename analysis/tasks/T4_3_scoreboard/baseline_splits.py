"""Split halves of the baselines for their noise ceilings (05 section 3: "Baselines: same split
method"). The training cohorts have fewer than 4 donors per species (liver: one library per
species; testis H+Mouse: 2 mouse libraries), so the halves are random cell halves, labelled
technical: cells are split 50/50 within each (species, cell type, library) group, and each half
goes through the T4.2 pipeline (E, L, B2b; Leiden clusters of Chimera's z from the full data;
B3a refitted on the half's E) with the unit cell minimum halved (15 cells per type, 10 per cluster)
so that each half keeps the group set of the full data (liver mouse cholangiocytes have 35 cells).
100 splits, seeded.
No gold and no D is read.

Usage: python -m analysis.tasks.T4_3_scoreboard.baseline_splits --cohorts liv_hmouse,test_hmouse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import cohorts as C
from analysis.common import gene_ids
from analysis.tasks.T4_2_gene_baselines import run as T42

N_SPLITS = 100
SEED = 20261102
COLS = ["B1b", "B1c", "B2b", "B1z", "B3a"]


def scores(X, species, ctype, cl, library, F, n):
    spl = sorted(set(species))
    x, f, _ = T42.pseudobulk(X, species, ctype, library, T42.MIN_CELLS_TYPE // 2)
    E, _, _ = T42.pooled_E(x, f, spl, n)
    L, _, B2, _, groups = T42.level_and_specificity(x, f, spl, n)
    xz, fz, _ = T42.pseudobulk(X, species, cl, library, T42.MIN_CELLS_CLUSTER // 2)
    Ez, _, _ = T42.pooled_E(xz, fz, spl, n)
    b3a = T42.cross_fit_ridge(F, np.asarray(E, float))[0]
    b3a[~np.isfinite(E)] = np.nan  # only genes with a target carry a split value
    if len(groups) < 4:
        B2 = np.full(n, np.nan)
    return dict(B1b=E, B1c=L, B2b=B2, B1z=Ez, B3a=b3a)


def run_cohort(cohort: str, outdir: Path, n_splits: int) -> None:
    co = C.load_cohort(cohort, keep_adata=True, covariates=False)
    a = co.adata
    X = (a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)).astype(np.float64)
    n = a.n_vars
    species = a.obs["species"].astype(str).to_numpy()
    tis = T42.tissue(cohort)
    library = a.obs[T42.LIBRARY_KEY[tis]].astype(str).to_numpy()
    lab = a.obs["cell_type"].astype(str)
    if tis == "testis":
        lab = lab.map(T42.TESTIS_COARSE)
    ctype = lab.to_numpy()
    z = T42.chimera_latent(cohort, a)
    cl = T42.leiden_clusters(z)
    F = np.asarray(co.state["gene_features"], dtype=float)
    rng = np.random.default_rng(SEED)
    grp = pd.Series([f"{s}|{t}|{l}" for s, t, l in zip(species, ctype, library)])
    gidx = [np.flatnonzero(grp.to_numpy() == g) for g in grp.unique()]
    out = {c: np.full((n_splits, 2, n), np.nan, dtype=np.float32) for c in COLS}
    for i in range(n_splits):
        half = np.zeros(len(species), bool)
        for idx in gidx:
            p = rng.permutation(idx)
            half[p[: len(p) // 2]] = True
        for h, mk in enumerate((half, ~half)):
            rows = np.flatnonzero(mk)
            sc = scores(X[rows], species[rows], ctype[rows], cl[rows], library[rows], F, n)
            for c in COLS:
                out[c][i, h] = sc[c]
        print(f"[{cohort}] split {i + 1}/{n_splits}", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = outdir / f"baseline_splits__{cohort}.tmp.npz"
    np.savez_compressed(tmp, var_names=np.asarray(a.var_names), **out)
    tmp.replace(outdir / f"baseline_splits__{cohort}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", default="liv_hmouse,test_hmouse")
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--outdir", type=Path, default=gene_ids.REPO_ROOT / "results/analysis/T4_3/baseline_splits")
    a = ap.parse_args()
    import torch

    torch.set_num_threads(4)
    for c in a.cohorts.split(","):
        run_cohort(c, a.outdir, a.n_splits)


if __name__ == "__main__":
    main()
