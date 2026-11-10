"""Calibration of the pre-specified set test on random sets from a cell-type-dominated gene pool.

Uniform random sets are exchangeable within strata by construction, so they cannot show the
failure that analysis/tasks/X2_liver_biology found for hepatocyte gene sets. Here random sets of matched size are
drawn from the pool used there (liver: top quartile of ``hep_enrich``) or its testis analogue (genes whose
dominant cell type in analysis/tasks/T2_5_celltype_margin is the commonest one), then go through the same statistic: pi_raw x
expression decile cells, the 5:1 background rule, stratified AUROC, set labels permuted within
kept strata. Real D is used; no gold or gene set is read. Reports the fraction of nominal
one-sided p < 0.05 in each direction (5% if calibrated). 40 is the set size of X2_liver_biology.

Usage: python -m analysis.tasks.T4_3_scoreboard.calibration_dominant [--n-sets 200 --n-perm 2000]
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.tasks.T4_3_scoreboard import inputs as I
from analysis.tasks.T4_3_scoreboard import stats as S

RES = gene_ids.REPO_ROOT / "results/analysis"
COV = list(K.COVARIATE_COLUMNS)
SIZES = {"liv_hmouse": [40, 109, 131, 153], "liv_4sp": [40, 109, 133, 155], "test_hmouse": [40, 139, 320]}
DATA = {}


def one(job):
    c, pool, size, n_sets, n_perm = job
    T, m = DATA[c]
    idx = np.flatnonzero(m[pool])
    score = T["D"].to_numpy(float)
    lab = S.cells(T["prior_raw"].to_numpy(float), T["log_cp10k_spmean"].to_numpy(float))
    rng = np.random.default_rng(zlib.crc32(f"{c}:{pool}:{size}".encode()))
    pu, pl, au = [], [], []
    for _ in range(n_sets):
        ins = np.zeros(len(T), bool)
        ins[rng.choice(idx, size, replace=False)] = True
        keep = S.kept_strata(ins, lab)
        if ins[keep].sum() < 3:
            continue
        a = S.strat_auroc(score[keep], ins[keep], lab[keep])
        null = S.auroc_perm(score[keep], ins[keep], lab[keep], n_perm, rng)
        au.append(a)
        pu.append(S.p_one_sided(a, null))
        pl.append(S.p_one_sided(a, null, lower=True))
    pu, pl = np.array(pu), np.array(pl)
    return dict(cohort=c, pool=pool, pool_size=len(idx), size=size, n_sets=len(pu), mean_auroc=float(np.mean(au)),
                frac_above_lt_005=float((pu < .05).mean()), frac_below_lt_005=float((pl < .05).mean()), n_perm=n_perm)


def load():
    for c in SIZES:
        d = I.load(c, RES / "T4_3/cache", consensus=True)
        T = d["tab"]
        T = T[T[COV + ["D"]].notna().all(1)].reset_index(drop=True)
        pools = {"any_gene": np.ones(len(T), bool)}
        if c.startswith("liv"):
            g = pd.read_parquet(RES / f"X2/genes/{c}.parquet").set_index("var_name")["hep_enrich"]
            v = g.reindex(T["var_name"]).to_numpy()
            pools["hep_top_quartile"] = v >= np.nanquantile(g.to_numpy(), 0.75)
        else:
            g = pd.read_parquet(RES / f"T2_5/{c}__genes.parquet").set_index("var_name")["dominant_cell_type"]
            v = g.reindex(T["var_name"])
            pools[f"dominant_{v.mode().iloc[0]}"] = (v == v.mode().iloc[0]).to_numpy()
        DATA[c] = (T, pools)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sets", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=RES / "T4_3/calibration_dominant.csv")
    a = ap.parse_args()
    load()
    jobs = [(c, p, s, a.n_sets, a.n_perm) for c, ss in SIZES.items() for p in DATA[c][1] for s in ss]
    import multiprocessing as mp

    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(one, jobs)
    df = pd.DataFrame(rows)
    tmp = a.out.with_name(a.out.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(a.out)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
