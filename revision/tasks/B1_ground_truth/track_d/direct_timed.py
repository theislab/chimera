"""B1 Track D: the direct pipeline a user would run, timed per step on one object: pseudobulk with labels; B1 and B1c, B2b; Sdir; B3a."""

from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np

import anndata as ad

from revision.tasks.B1_ground_truth import alter as A
from revision.tasks.B1_ground_truth import rules as RU
from revision.tasks.T4_2_gene_baselines import run as T42
from revision.tasks.X6_adds_beyond.degrade import sdir


def main():
    job, h5ad, cohort = sys.argv[1:4]
    t = {}
    t0 = time.time()
    a = ad.read_h5ad(RU.REPO / h5ad)
    t["read"] = time.time() - t0
    X = A.csr(a.X).astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    lab = A.labels(a, cohort)
    lib = a.obs[RU.COHORTS[cohort]["library_key"]].astype(str).to_numpy()
    species = sorted(set(spc))
    G = a.n_vars
    t1 = time.time()
    x, f, _ = T42.pseudobulk(X, spc, lab, lib, T42.MIN_CELLS_TYPE)
    t["pseudobulk"] = time.time() - t1
    t1 = time.time()
    E, _, _ = T42.pooled_E(x, f, species, G)
    t["B1"] = time.time() - t1
    t1 = time.time()
    T42.level_and_specificity(x, f, species, G)
    t["B1c_B2b"] = time.time() - t1
    t1 = time.time()
    sdir(X, spc, lab)
    t["Sdir"] = time.time() - t1
    t1 = time.time()
    T42.cross_fit_ridge(np.asarray(a.varm["gene_features"], dtype=float), np.asarray(E, dtype=float))
    t["B3a"] = time.time() - t1
    t["total"] = time.time() - t0
    t["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    out = RU.RES / "track_d"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{job}.direct.json").write_text(json.dumps(t, indent=1))
    print(t)


if __name__ == "__main__":
    main()
