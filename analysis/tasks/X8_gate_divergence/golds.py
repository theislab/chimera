"""X8 step (d): does the gap carry information about independent references? B2 Track C golds through B2's engine and C1's cortex
held-out-region golds through C1's rho_table, on X8's universe (genes count-supported in both gates).

Per unit: rho_p(G, x | C) for x in delta, delta_res, z_free, z_can, r_f|c, r_c|f; and each gate's reading given the other
(the other's z added to the design as a df-4 spline). Called genes: mean residual gold (rank of G given C) of up minus down calls.

    PYTHONPATH=$PWD python -m analysis.tasks.X8_gate_divergence.golds trackc|cortex [--jobs 8]
Output: results/analysis/X8/golds_trackc.csv, golds_calls.csv, golds_cortex.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.tasks.B2_benchmark import engine as EN  # noqa: E402
from analysis.tasks.B2_benchmark import rules as B2R  # noqa: E402
from analysis.tasks.B2_benchmark import rules_c as RC  # noqa: E402
from analysis.tasks.B2_benchmark import track_c as TC  # noqa: E402
from analysis.tasks.X8_gate_divergence import rules as RU  # noqa: E402

OUT = RU.OUT
SCORES = ["delta", "delta_res", "z_free", "z_can", "r_f_given_c", "r_c_given_f"]
N_DRAWS = B2R.N_DRAWS_C


def x8(c: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / "genes" / f"{c}.parquet")


def unit(u: dict) -> tuple[list[dict], list[dict]]:
    c, kind, gold, covs = u["cohort"], u["kind"], u["gold"], u["covs"]
    t = TC.table(c)
    g = x8(c).set_index("hsapiens_gene_id")
    for k in SCORES + ["call"]:
        t[k] = g[k].reindex(t.hsapiens_gene_id).to_numpy(float)
    G = t[gold].to_numpy(float) if gold in t else np.full(len(t), np.nan)
    U = np.isfinite(G) & t[covs].notna().all(1).to_numpy() & t["block"].notna().to_numpy() & np.isfinite(t[SCORES].to_numpy(float)).all(1)
    base = dict(cohort=c, kind=kind, gold=gold, n=int(U.sum()))
    if U.sum() < B2R.STRATUM_MIN:
        return [dict(**base, method="", value=np.nan)], []
    task = EN.Task(t, N_DRAWS, B2R.useed("X8", c, kind))
    sc = {k: t[k].to_numpy(float) for k in SCORES}
    task.add("main", U, G, sc, covs)
    task.add("given_free", U, G, {"z_can": sc["z_can"]}, covs + ["z_free"])
    task.add("given_can", U, G, {"z_free": sc["z_free"]}, covs + ["z_can"])
    task.run()
    rows = [dict(**{k: v for k, v in base.items() if k != "n"}, **r) for r in task.rows()]
    # called genes: residual gold given C, up minus down, ESM-module bootstrap
    idx = np.flatnonzero(U)
    Z = EN.design_for(t.iloc[idx], covs)
    r = pd.Series(G[idx]).rank().to_numpy()
    res = (r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]) / len(r)
    cl = t["call"].to_numpy(float)[idx]
    blk = t["block"].to_numpy()[idx]
    rng = np.random.default_rng(B2R.useed("X8calls", c, kind))
    ub, inv = np.unique(blk, return_inverse=True)
    mem = [np.flatnonzero(inv == i) for i in range(len(ub))]

    def stat(ix):
        a, b = res[ix][cl[ix] > 0], res[ix][cl[ix] < 0]
        return (a.mean() if len(a) else np.nan) - (b.mean() if len(b) else np.nan), (a.mean() if len(a) else np.nan), (b.mean() if len(b) else np.nan)

    obs = stat(np.arange(len(res)))
    bs = np.array([stat(np.concatenate([mem[i] for i in rng.integers(0, len(ub), len(ub))])) for _ in range(2000)])
    q = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))  # noqa: E731
    calls = [dict(**base, n_up=int((cl > 0).sum()), n_down=int((cl < 0).sum()), diff=obs[0], diff_lo=q(bs[:, 0])[0], diff_hi=q(bs[:, 0])[1],
                  mean_up=obs[1], up_lo=q(bs[:, 1])[0], up_hi=q(bs[:, 1])[1], mean_down=obs[2], down_lo=q(bs[:, 2])[0], down_hi=q(bs[:, 2])[1])]
    return rows, calls


def trackc(jobs: int) -> None:
    units = [dict(cohort=c, kind=k, gold=g, covs=cv) for c in RC.COHORTS_C if (OUT / "genes" / f"{c}.parquet").exists() for k, g, cv in RC.COHORTS_C[c]["golds"]]
    with ProcessPoolExecutor(jobs, mp_context=__import__("multiprocessing").get_context("spawn")) as ex:
        res = list(ex.map(unit, units))
    pd.DataFrame([r for x in res for r in x[0]]).to_csv(OUT / "golds_trackc.csv", index=False)
    pd.DataFrame([r for x in res for r in x[1]]).to_csv(OUT / "golds_calls.csv", index=False)


def cortex(n_perm: int, n_boot: int) -> None:
    from analysis.tasks.C1_cohorts import tests_c1 as T1

    rng = np.random.default_rng(RU.SEED)
    rows, calls = [], []
    for c, gl in T1.BRAIN_GOLDS.items():
        g = T1.genes(c)
        x = x8(c)
        g = g[g["var_name"].isin(x.index)].copy()
        for k in SCORES + ["call"]:
            g[k] = x[k].reindex(g["var_name"]).to_numpy(float)
        for gname in gl:
            gg = pd.read_parquet(T1.GOLD / f"{gname}.parquet").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
            u = g.assign(gold=gg["gold"].reindex(g["hsapiens_gene_id"]).to_numpy())
            lab = dict(cohort=c, gold=gname)
            rows += [dict(**r, group="main") for r in T1.rho_table(u, "gold", SCORES, n_perm, n_boot, rng, lab)]
            rows += [dict(**r, group="given_free") for r in T1.rho_table(u, "gold", ["z_can"], n_perm, n_boot, rng, lab, extra_nuis=("z_free",))]
            rows += [dict(**r, group="given_can") for r in T1.rho_table(u, "gold", ["z_free"], n_perm, n_boot, rng, lab, extra_nuis=("z_can",))]
            v = u.dropna(subset=["gold", *T1.C_COLS])
            Z = T1.S.design(v, T1.C_COLS)
            r = pd.Series(v["gold"].to_numpy(float)).rank().to_numpy()
            res = (r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]) / len(r)
            cl = v["call"].to_numpy(float)
            calls.append(dict(**lab, n=len(v), n_up=int((cl > 0).sum()), n_down=int((cl < 0).sum()), mean_up=float(res[cl > 0].mean()), mean_down=float(res[cl < 0].mean()),
                              diff=float(res[cl > 0].mean() - res[cl < 0].mean())))
    pd.DataFrame(rows).to_csv(OUT / "golds_cortex.csv", index=False)
    pd.DataFrame(calls).to_csv(OUT / "golds_cortex_calls.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    if a.cmd == "trackc":
        trackc(a.jobs)
    elif a.cmd == "cortex":
        cortex(a.n_perm, a.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
