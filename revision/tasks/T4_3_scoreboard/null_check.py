"""Mechanics check of the scoreboard statistics on null data (no real association involved).

Check A, permuted D: in each replicate the stand-in D (seed mean of the finished runs) is
permuted across genes (seeded), then the four primary units (H1-H4) are run with reduced draw
counts. Under this null the one-sided permutation p-values should be uniform and the block
bootstrap CIs should cover 0 at about the nominal rate.
Check B, permuted gold: the real baselines (and the permuted-D replicate) against a permuted
gold; same expectations for their rho_p CIs.
Check C, set tests on random gene sets: covered by run.py (calibration_sets.csv).

Usage: python -m revision.tasks.T4_3_scoreboard.null_check --outdir DIR [--reps 200]
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest

from revision.tasks.T4_3_scoreboard import run as RUN

BASE = {}


def replicate(rep: int) -> list[dict]:
    rng = np.random.default_rng(10_000 + rep)
    out = []
    for h, v in RUN.PRIMARY.items():
        c = v["cohort"]
        tab0 = BASE[c]["tab"]
        tab = tab0.copy()
        n = len(tab)
        # Check A: permuted D
        p = rng.permutation(n)
        tab["D"] = tab0["D"].to_numpy()[p]
        RUN.DATA[c] = dict(BASE[c], tab=tab, M=BASE[c]["M"][p])
        u = RUN.unit(f"nullA:{h}:{rep}", "null", "permuted_D", h, c, v["gold"], v["cov"], ["D"], perm="one",
                     draws=RUN.CFG["draws"])
        r = RUN.run_unit(u)["rows"][0]
        out.append(dict(check="A_permuted_D", hyp=h, rep=rep, score="D", rho_p=r["rho_p"], p=r["p_perm"],
                        ci_lo=r["ci_lo"], ci_hi=r["ci_hi"]))
        # Check B: real baselines (and D) against a permuted gold
        tab = tab0.copy()
        g = tab[v["gold"]].to_numpy()
        ok = np.flatnonzero(np.isfinite(g))
        g2 = g.copy()
        g2[ok] = g[ok][rng.permutation(len(ok))]
        tab[v["gold"]] = g2
        RUN.DATA[c] = dict(BASE[c], tab=tab)
        for s in RUN.J + ["B3b"]:
            u = RUN.unit(f"nullB:{h}:{s}:{rep}", "null", "permuted_gold", h, c, v["gold"], v["cov"], [s],
                         draws=RUN.CFG["draws"])
            r = RUN.run_unit(u)["rows"][0]
            out.append(dict(check="B_permuted_gold", hyp=h, rep=rep, score=s, rho_p=r["rho_p"], p=np.nan,
                            ci_lo=r["ci_lo"], ci_hi=r["ci_hi"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=18)
    a = ap.parse_args()
    RUN.CFG.update(draws=a.draws, n_perm=a.n_perm)
    RUN.prepare("null-D", 0)
    for c in ("liv_hmouse", "test_hmouse"):
        BASE[c] = dict(RUN.DATA[c])
    import multiprocessing as mp

    with mp.get_context("fork").Pool(a.workers) as pool:
        res = [x for r in pool.imap_unordered(replicate, range(a.reps)) for x in r]
    df = pd.DataFrame(res)
    df["covers_0"] = (df.ci_lo <= 0) & (df.ci_hi >= 0)
    summ = []
    for (chk, h, s), g in df.groupby(["check", "hyp", "score"]):
        row = dict(check=chk, hyp=h, score=s, reps=len(g), mean_rho_p=g.rho_p.mean(), sd_rho_p=g.rho_p.std(),
                   ci_coverage_of_0=g.covers_0.mean(), mean_ci_width=(g.ci_hi - g.ci_lo).mean())
        if g.p.notna().any():
            row.update(frac_p_lt_005=(g.p < 0.05).mean(), frac_p_lt_010=(g.p < 0.10).mean(),
                       mean_p=g.p.mean(), ks_uniform_p=kstest(g.p, "uniform").pvalue)
        summ.append(row)
    S = pd.DataFrame(summ)
    a.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.outdir / "null_check_reps.csv", index=False)
    S.to_csv(a.outdir / "null_check_summary.csv", index=False)
    (a.outdir / "null_check.json").write_text(json.dumps(dict(reps=a.reps, draws=a.draws, n_perm=a.n_perm), indent=2))
    pd.set_option("display.width", 200)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
