"""X8 step (b), sets: H1's set machinery unchanged (Hallmarks, X3 gene classes, first-submission categories; INT of the readout within
the universe; permutation within pi x expression strata; z, two-sided p, BH within cohort; recurrence over the eight thread tissues,
R >= 3 and q_rec < 0.05; tissue-specific sets), with readouts delta, delta_res, z_free and z_can on X8's universe.

    PYTHONPATH=$PWD python -m analysis.tasks.X8_gate_divergence.sets [--jobs 10]
Output: results/analysis/X8/set_z.csv, set_recurrence.csv, set_tissue_specific.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from analysis.tasks.H1_harmonise import recurrence as H1REC  # noqa: E402
from analysis.tasks.X8_gate_divergence import rules as RU  # noqa: E402

OUT = RU.OUT
READOUTS = ["delta", "delta_res", "z_free", "z_can"]


def one(args):
    c, readout, membership, n_perm = args
    u = pd.read_parquet(OUT / "genes" / f"{c}.parquet").dropna(subset=[readout, "expr", "pi_raw"])
    st = H1REC.strata(u, True)
    v = H1REC.inv_norm_rank(u[readout].to_numpy(float)).astype(np.float32)
    ix = pd.Index(u.hsapiens_gene_id)
    sets = sorted(membership.key.unique())
    si = {s: i for i, s in enumerate(sets)}
    mm = membership[membership.hsapiens_gene_id.isin(ix)]
    M = sp.csr_matrix((np.ones(len(mm), np.float32), (mm.key.map(si).to_numpy(), ix.get_indexer(mm.hsapiens_gene_id))), shape=(len(sets), len(ix)))
    sizes = np.asarray(M.sum(1)).ravel()
    rng = np.random.default_rng([RU.SEED, sum(map(ord, c)) * 1000 + sum(map(ord, readout))])
    obs = np.asarray(M @ v).ravel() / np.maximum(sizes, 1)
    null = H1REC.perm_scores(v, st, M, n_perm, rng)
    mu, sd = null.mean(0), null.std(0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)
    z = (obs - mu) / sd
    zn = (null - mu) / sd
    p = (1 + (np.abs(zn) >= np.abs(z)[None, :]).sum(0)) / (n_perm + 1)
    t = pd.DataFrame(dict(cohort=c, readout=readout, key=sets, n=sizes.astype(int), mean_value=obs, z=z, p=p))
    t.loc[t.n < H1REC.RU.MIN_SET, ["z", "p"]] = np.nan
    t["q"] = H1REC.bh(t.p.to_numpy())
    return t, zn.astype(np.float32), sets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=10_000)
    a = ap.parse_args()
    cs = list(RU.cohorts())
    allg = pd.concat([pd.read_parquet(OUT / "genes" / f"{c}.parquet")[["hsapiens_gene_id", "symbol"]] for c in cs])
    gs = H1REC.gene_sets(allg)
    gs["key"] = gs.family + "|" + gs.set
    mem = gs[["key", "hsapiens_gene_id"]]
    th = RU.thread()
    prim = list(th.values())
    jobs = [(c, r, mem, a.n_perm) for c in prim for r in READOUTS]
    jobs += [(c, r, mem, 2000) for c in cs if c not in prim for r in ("delta", "z_free", "z_can")]
    with ProcessPoolExecutor(a.jobs, mp_context=__import__("multiprocessing").get_context("spawn")) as ex:
        res = list(ex.map(one, jobs))
    long = pd.concat([x[0] for x in res], ignore_index=True)
    long.to_csv(OUT / "set_z.csv", index=False)
    idx = {(j[0], j[1]): i for i, j in enumerate(jobs)}
    fam = {k: k.split("|")[0] for k in mem.key.unique()}
    sets_all = res[0][2]
    recs, tss = [], []
    for r in READOUTS:
        Z = {}
        for t, c in th.items():
            tab, zn, _ = res[idx[(c, r)]]
            Z[t] = (tab.z.to_numpy(), zn, tab.p.to_numpy())
        rec, ts = H1REC.recurrence_table(Z, list(th), sets_all, fam, RU.R_MIN)
        rec.insert(0, "readout", r)
        recs.append(rec)
        if len(ts):
            ts.insert(0, "readout", r)
            tss.append(ts)
    rec = pd.concat(recs, ignore_index=True)
    rec.to_csv(OUT / "set_recurrence.csv", index=False)
    if tss:
        pd.concat(tss, ignore_index=True).to_csv(OUT / "set_tissue_specific.csv", index=False)
    print(rec.groupby("readout").recurrent.sum())
    print(rec[rec.recurrent].sort_values(["readout", "q_rec"])[["readout", "key", "R", "direction", "q_rec", "mean_z"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
