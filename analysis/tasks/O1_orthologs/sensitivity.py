"""O1 part B: the T4.3 scoreboard with flagged genes left out of every test. D, baselines, draws and seeds as frozen; only the gene universes shrink.
Written to results/analysis/O1/scoreboard_<tier>/; the official folder is not touched. See rules.py and the ledger.

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.sensitivity --tier T2 [--workers 12]
    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.sensitivity --random-drop
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json

import numpy as np
import pandas as pd

from analysis.tasks.O1_orthologs import rules as RU
from analysis.tasks.T4_3_scoreboard import inputs as I
from analysis.tasks.T4_3_scoreboard import run as R
from analysis.tasks.T4_3_scoreboard import stats as S

TIERS = ["T1", "T2", "T2L", "MAP", "T3", "TZ", "T2T", "R2", "R3"]
TOOLS = RU.REPO / "results/analysis/T5_6/tools_primary.csv"


def flag_sets(tier: str) -> dict[str, set]:
    G = pd.read_parquet(RU.RES / "flags_by_cohort.parquet")
    G["MAP"] = G.M_wrong_row | G.M_false_zero
    G["T2T"] = G.T2_table
    return {c: set(x.var_name[x[tier]]) for c, x in G.groupby("cohort")}


def patch(tier: str) -> None:
    drop = flag_sets(tier)
    o_load, o_u, o_su = I.load, R.universe, R.set_universe

    def load(cohort, cache, consensus=True):
        d = o_load(cohort, cache, consensus=consensus)
        d["tab"]["o1_keep"] = ~d["tab"]["var_name"].isin(drop[cohort])
        return d

    def universe(tab, gold, cov, scores, subset=None):
        return o_u(tab, gold, cov, scores, subset) & tab["o1_keep"].to_numpy()

    def set_universe(tab, subset=None):
        return o_su(tab, subset) & tab["o1_keep"].to_numpy()

    I.load, R.universe, R.set_universe = load, universe, set_universe


def _one_drop(args):
    h, tier, k, seed, n_draw = args
    v = R.PRIMARY[h]
    tab = R.DATA[v["cohort"]]["tab"]
    m = R.universe(tab, v["gold"], v["cov"], ["D"])
    T = tab.loc[m].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    out = np.empty(n_draw)
    cov = R.COVSETS[v["cov"]]
    for i in range(n_draw):
        keep = np.ones(len(T), bool)
        keep[rng.choice(len(T), k, replace=False)] = False
        U = T.loc[keep]
        Rk = np.column_stack([S.midrank(U[v["gold"]].to_numpy(float)), S.midrank(U["D"].to_numpy(float))])
        out[i] = S.rho_p_matrix(Rk, S.design(U, cov))[1]
    return h, tier, k, out


def random_drop(workers: int, n_draw: int = 1000) -> None:
    import multiprocessing as mp

    R.CFG.update(draws=1, n_perm=1, n_boot_sets=1, n_perm_sets=1, n_calib=1, n_perm_calib=1, mode="real")
    R.prepare("real", 1)
    jobs = []
    for tier in TIERS:
        d = flag_sets(tier)
        for h, v in R.PRIMARY.items():
            tab = R.DATA[v["cohort"]]["tab"]
            m = R.universe(tab, v["gold"], v["cov"], ["D"])
            k = int(tab.loc[m, "var_name"].isin(d[v["cohort"]]).sum())
            if k:
                jobs.append((h, tier, k, RU.SEED + len(jobs), n_draw))
    with mp.get_context("fork").Pool(workers) as p:
        res = p.map(_one_drop, jobs, chunksize=1)
    rows = [dict(hyp=h, tier=t, k_dropped=k, n_draw=len(x), mean=float(x.mean()), q025=float(np.quantile(x, 0.025)), q975=float(np.quantile(x, 0.975)), sd=float(x.std(ddof=1))) for h, t, k, x in res]
    pd.DataFrame(rows).to_csv(RU.RES / "random_drop.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=TIERS)
    ap.add_argument("--random-drop", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.random_drop:
        random_drop(a.workers)
        return
    patch(a.tier)
    out = RU.RES / f"scoreboard_{a.tier}"
    if out.resolve() == (RU.REPO / "results/analysis/T4_3").resolve():
        raise SystemExit("never the official folder")
    argv = ["--mode", "real", "--outdir", str(out), "--workers", str(a.workers)]
    if TOOLS.exists():
        argv += ["--tools", str(TOOLS)]
    R.main(argv)
    (out / "O1_tier.json").write_text(json.dumps(dict(tier=a.tier), indent=1))


if __name__ == "__main__":
    main()
