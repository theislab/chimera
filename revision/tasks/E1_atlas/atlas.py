"""E1 step 2: set statistic per cohort x read-out (definitions in rules.py). H1's strata, permutation and block nulls reused unchanged.

Writes results/revision/E1/set_stats.parquet (one row per cohort x read-out x set) and, for the eight-tissue thread, null sign
indicators and tissue-programme null z in results/revision/E1/cache/ (read by summarize.py).

    PYTHONPATH=$PWD python -m revision.tasks.E1_atlas.atlas [--jobs 12] [--quick]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp

from revision.tasks.E1_atlas import rules as RU
from revision.tasks.H1_harmonise import rules as H1R
from revision.tasks.H1_harmonise.recurrence import bh, block_scores, inv_norm_rank, perm_scores, strata

CACHE = RU.OUT / "cache"
THREAD_READOUTS = ("D_can", "D_free", "S", "B1", "pi", "dNdS", "D_can_tau", "D_free_tau", "D_can_block", "D_free_block")
OTHER_READOUTS = ("D_can", "D_free")


def cohort_table(c: str) -> pd.DataFrame:
    """Canonical and free per-gene tables joined on the human Ensembl ID; universe = count-supported in both fits."""
    a = pd.read_parquet(RU.CAN / f"{c}.parquet").dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id")
    b = pd.read_parquet(RU.FREE / f"{c}.parquet").dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id")
    b = b.set_index("hsapiens_gene_id")[["D", "S_ema", "count_support", "nameable", "strict_support", "saturated_flag"]].add_suffix("_free")
    g = a.set_index("hsapiens_gene_id").join(b, how="inner").reset_index()
    g = g.rename(columns={"D": "D_can", "S_ema": "S"})
    g["universe"] = g.count_support & g.count_support_free
    return g


def tissue_programme_keys(membership_keys: list[str]) -> list[str]:
    out = []
    for ids in RU.TISSUE_PROGRAMMES.values():
        for x in ids:
            if x.startswith("hallmark:"):
                k = "hallmark|" + x.split(":", 1)[1]
            elif x.startswith("R-HSA"):
                k = next((k for k in (f"reactome_top|{x}", f"reactome|{x}") if k in membership_keys), None)
            else:
                k = next((k for k in (f"goslim|{x}", f"go_tissue|{x}") if k in membership_keys), None)
            if k and k not in out:
                out.append(k)
    return out


def one(args):
    c, readout, n_perm, keys, mem, thread = args
    t0 = time.time()
    g = cohort_table(c)
    base = readout.replace("_tau", "").replace("_block", "")
    col = {"D_can": "D_can", "D_free": "D_free", "S": "S", "B1": "B1", "pi": "pi_raw", "dNdS": "dNdS", "noise": "noise"}[base]
    u = g[g.universe].copy()
    if base == "noise":  # mechanics only
        u["noise"] = np.random.default_rng(0).normal(size=len(u))
    if base == "dNdS":
        u = u[u.dNdS > 0].assign(dNdS=lambda d: np.log10(d.dNdS.clip(lower=1e-3)))
    u = u.dropna(subset=[col, "expr", "pi_raw"])
    st = strata(u, base not in ("pi", "dNdS"), "pi_raw", with_tau=readout.endswith("_tau"))
    v = inv_norm_rank(u[col].to_numpy(float)).astype(np.float32)
    ix = pd.Index(u.hsapiens_gene_id)
    si = {k: i for i, k in enumerate(keys)}
    mm = mem[mem.hsapiens_gene_id.isin(ix)]
    M = sp.csr_matrix((np.ones(len(mm), np.float32), (mm.key.map(si).to_numpy(), ix.get_indexer(mm.hsapiens_gene_id))), shape=(len(keys), len(ix)))
    sizes = np.asarray(M.sum(1)).ravel()
    rng = np.random.default_rng([RU.SEED, sum(map(ord, c)) * 1000 + sum(map(ord, readout))])
    if readout.endswith("_block"):
        v = (v - pd.Series(v).groupby(st).transform("mean").to_numpy()).astype(np.float32)
        tissue = H1R.cohorts()[c]["tissue"].replace(" ", "_")
        yf = RU.RES / "Y1" / "cache" / f"tissue__{tissue}.parquet"
        tb = pd.read_parquet(yf) if yf.exists() else pd.read_parquet(RU.RES / "H1" / "cache" / f"blocks__{c}.parquet")
        blk = pd.Series(tb.block.to_numpy(), index=tb.hsapiens_gene_id).groupby(level=0).first().reindex(ix)
        codes = pd.factorize(blk.fillna(-1).astype(str))[0]
        null = block_scores(v, codes, M, n_perm, rng)
    else:
        null = perm_scores(v, st, M, n_perm, rng)
    obs = np.asarray(M @ v).ravel() / np.maximum(sizes, 1)
    mu, sd = null.mean(0), null.std(0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)
    z = (obs - mu) / sd
    zn = (null - mu) / sd
    p = (1 + (np.abs(zn) >= np.abs(z)[None, :]).sum(0)) / (n_perm + 1)
    small = sizes < RU.MIN_SET
    z[small], p[small] = np.nan, np.nan
    t = pd.DataFrame(dict(cohort=c, readout=readout, key=keys, n=sizes.astype(int), score=obs, null_mean=mu, null_sd=sd, effect=np.where(small, np.nan, obs - mu), z=z, p=p))
    t["library"] = t.key.str.split("|").str[0]
    t["q"] = t.groupby("library").p.transform(lambda s: pd.Series(bh(s.to_numpy()), index=s.index))
    if thread:
        zn[:, small] = np.nan
        sign = np.zeros(zn.shape, np.int8)
        sign[np.nan_to_num(zn, nan=0.0) >= RU.Z_CALL] = 1
        sign[np.nan_to_num(zn, nan=0.0) <= -RU.Z_CALL] = -1
        tp = [si[k] for k in tissue_programme_keys(keys)]
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE / f"{c}__{readout}.npz", sign=sign, zn_tp=zn[:, tp].astype(np.float32))
    print(f"{c} {readout} n_genes={len(ix)} sets={int((~small).sum())} {time.time() - t0:.0f}s", flush=True)
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--quick", action="store_true", help="mechanics check: one cohort, 200 draws, no output table")
    a = ap.parse_args()
    mem = pd.read_parquet(RU.OUT / "sets" / "membership.parquet")[["key", "hsapiens_gene_id"]]
    keys = sorted(mem.key.unique())
    cohorts = list(H1R.cohorts())
    thread = [RU.PRIMARY[t] for t in RU.TISSUES]
    if a.quick:
        t = one(("kidney", "noise", 1000, keys, mem, False))
        print(t.groupby("library").agg(n_sets=("z", lambda s: s.notna().sum()), sd_z=("z", "std"), frac_p05=("p", lambda s: (s < 0.05).mean())))
        return 0
    jobs = []
    for c in thread:
        for r in THREAD_READOUTS:
            jobs.append((c, r, RU.N_PERM_BLOCK if r.endswith("_block") else RU.N_PERM_THREAD, keys, mem, True))
    for c in cohorts:
        if c not in thread:
            for r in OTHER_READOUTS:
                jobs.append((c, r, RU.N_PERM_OTHER, keys, mem, False))
    jobs.sort(key=lambda j: -j[2])
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, jobs))
    out = pd.concat(res, ignore_index=True)
    out["thread"] = out.cohort.isin(thread)
    out.to_parquet(RU.OUT / "set_stats.parquet")
    print(out.groupby(["readout"]).size())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
