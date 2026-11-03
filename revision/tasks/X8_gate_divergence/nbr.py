"""X8 step (b), protein neighbourhoods beyond covariates. Added after gap.py showed that expression-matched random neighbours predict a
gene's per-gene reading as well as its ESM-2 neighbours in several cohorts (testis, liver 4 species), so the raw neighbour statistic
mixes protein similarity with expression.

Per cohort, on the universe: z_free and z_can as rank residuals on df-4 splines of expression, detection, identity (pi_raw), tau and
the detection gap (rf, rc; |S| is not a covariate here). 15-NN on the 64 ESM-2 PCs within the universe; m_f, m_c = neighbour means.
  share_free   rho(rf, m_f): do a gene's protein neighbours share its count reading beyond covariates?
  share_can    rho(rc, m_c): the same for the canonical reading.
  follow       rho(rc, m_f): does the canonical reading follow the neighbours' count reading?
  pull         partial rho(rc, m_f | rf): does the neighbourhood's count reading shape the canonical reading beyond the gene's own?
  unlike       rho(delta_r, rf - m_f) with delta_r = rf - rc, beside its value with random neighbours (the mechanical part).
Null for each: the same statistic with 15 random genes of the universe as neighbours (N_NULL_NBR draws). Also by neighbour distance
tertile (do near neighbours share more?) and k in (5, 15, 50).

    PYTHONPATH=$PWD python -m revision.tasks.X8_gate_divergence.nbr [--jobs 8]
Output: results/revision/X8/nbr_resid.csv, nbr_resid_k.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from revision.tasks.X8_gate_divergence import gap as G  # noqa: E402
from revision.tasks.X8_gate_divergence import rules as RU  # noqa: E402

OUT = RU.OUT
COVS = ["expr", "detection", "pi_raw", "tau", "det_gap"]


def pcor(a, b, c) -> float:
    ra, rb, rc_ = (pd.Series(x).rank().to_numpy() for x in (a, b, c))
    X = np.c_[np.ones(len(rc_)), rc_]
    ea = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    eb = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])


def stats(rf, rc, idx) -> dict:
    mf, mc = rf[idx].mean(1), rc[idx].mean(1)
    dr = rf - rc
    return dict(share_free=G.rho(rf, mf), share_can=G.rho(rc, mc), follow=G.rho(rc, mf), pull=pcor(rc, mf, rf), unlike=G.rho(dr, rf - mf))


def one(c: str) -> tuple[list[dict], list[dict]]:
    from sklearn.neighbors import NearestNeighbors

    t = pd.read_parquet(OUT / "genes" / f"{c}.parquet")
    t = t.dropna(subset=COVS)
    Z = G.design([t[x].to_numpy(float) for x in COVS])
    rf, rc = G.resid_rank(t.z_free.to_numpy(), Z), G.resid_rank(t.z_can.to_numpy(), Z)
    E = G.esm(c).reindex(t.index)
    F = E[[f"esm{j}" for j in range(RU.N_PCS)]].to_numpy(float)
    nn = NearestNeighbors(n_neighbors=51).fit(F)
    dist, idx = nn.kneighbors(F)
    idx, dist = idx[:, 1:], dist[:, 1:]
    rng = np.random.default_rng([RU.SEED, 11, sum(map(ord, c))])
    obs = stats(rf, rc, idx[:, :RU.KNN])
    null = pd.DataFrame([stats(rf, rc, rng.integers(0, len(rf), (len(rf), RU.KNN))) for _ in range(RU.N_NULL_NBR)])
    row = dict(cohort=c, n=len(rf))
    for k, v in obs.items():
        row[k] = v
        row[f"{k}_null_mean"] = float(null[k].mean())
        row[f"{k}_null_lo"], row[f"{k}_null_hi"] = float(null[k].quantile(0.025)), float(null[k].quantile(0.975))
    # near versus far neighbourhoods: tertiles of the gene's mean neighbour distance
    md = dist[:, :RU.KNN].mean(1)
    ter = pd.qcut(md, 3, labels=False)
    for q in range(3):
        w = ter == q
        mf = rf[idx[:, :RU.KNN]].mean(1)
        row[f"share_free_dist_T{q + 1}"] = G.rho(rf[w], mf[w])
        row[f"follow_dist_T{q + 1}"] = G.rho(rc[w], mf[w])
    ks = []
    for k in (5, 15, 50):
        s = stats(rf, rc, idx[:, :k])
        ks.append(dict(cohort=c, k=k, **s))
    print(c, {k: round(v, 3) for k, v in obs.items()}, flush=True)
    return [row], ks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    cs = list(RU.cohorts())
    with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(one, cs))
    pd.DataFrame([r for x in res for r in x[0]]).to_csv(OUT / "nbr_resid.csv", index=False)
    pd.DataFrame([r for x in res for r in x[1]]).to_csv(OUT / "nbr_resid_k.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
