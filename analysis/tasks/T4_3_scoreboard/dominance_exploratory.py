"""EXPLORATORY, post hoc, written after the Family 2 results were seen. Enters no status, row or
modifier. It measures how far the pre-specified results depend on cell-type dominance, the
confounder that ``calibration_dominant.py`` found the pi x expression strata do not control.

Dominance: liver, top quartile of ``hep_enrich`` of analysis/tasks/X2_liver_biology; testis, dominant type of
analysis/tasks/T2_5_celltype_margin = commonest type.
- The 9 Family 2 tests with dominance added to the strata (same AUROC, 5:1 rule, permutation).
- H1-H4 and the multi-species estimates (family F3_figs4_6) with dominance added to the design (liver: spline of rank
  hep_enrich, as C_hep of X2_liver_biology; testis: the binary flag), block bootstrap, stratified permutation
  with the flag added to the strata.

Usage: python -m analysis.tasks.T4_3_scoreboard.dominance_exploratory
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"

import zlib

import numpy as np
import pandas as pd

from analysis.tasks.T4_3_scoreboard import inputs as I
from analysis.tasks.T4_3_scoreboard import run as RUN
from analysis.tasks.T4_3_scoreboard import stats as S

RES = RUN.RES
OUT = RES / "T4_3/exploratory_dominance.csv"
N_DRAWS, N_PERM, N_BOOT = 10_000, 10_000, 2_000
DATA = {}


def load():
    gold = pd.read_csv(RES / "T4_3/golds/gold_table.csv", index_col=0)
    for c in ("liv_hmouse", "test_hmouse", "liv_4sp", "test_3sp"):
        d = I.load(c, RES / "T4_3/cache", consensus=True)
        T = d["tab"].join(gold, on="hsapiens_gene_id")
        if c.startswith("liv"):
            g = pd.read_parquet(RES / f"X2/genes/{c}.parquet").set_index("var_name")["hep_enrich"]
            T["dom_value"] = g.reindex(T["var_name"]).to_numpy()
            T["dom_flag"] = T["dom_value"] >= np.nanquantile(g.to_numpy(), 0.75)
        else:
            g = pd.read_parquet(RES / f"T2_5/{c}__genes.parquet").set_index("var_name")["dominant_cell_type"]
            v = g.reindex(T["var_name"])
            T["dom_flag"] = (v == v.mode().iloc[0]).to_numpy()
            T["dom_value"] = T["dom_flag"].astype(float)
        for s in ("C1", "C2", "V1", "V2", "V3"):
            T[s] = T[s].eq(True)
        DATA[c] = T


def set_job(j):
    tis, s = j
    c = RUN.SET_COHORT[tis]
    T = DATA[c]
    T = T[T[RUN.COV + ["D", "dom_value"]].notna().all(1)]
    ins = T[s].to_numpy(bool)
    x = T["D"].to_numpy(float)
    lab = S.cells(T["prior_raw"].to_numpy(float), T["log_cp10k_spmean"].to_numpy(float)) * 2 + T["dom_flag"].to_numpy(int)
    keep = S.kept_strata(ins, lab)
    rng = np.random.default_rng(zlib.crc32(f"dom:{tis}:{s}".encode()))
    a = S.strat_auroc(x[keep], ins[keep], lab[keep])
    below = s.startswith("V")
    null = S.auroc_perm(x[keep], ins[keep], lab[keep], N_PERM, rng)
    A, _ = S.auroc_boot(x[keep], ins[keep], lab[keep], N_BOOT, rng, auprc=False)
    lo, hi = S.pct_ci(A)
    return dict(kind="set", id=f"{tis}:{s}", n_set=int(ins[keep].sum()), estimate=a, ci_lo=lo, ci_hi=hi,
                p=S.p_one_sided(a, null, lower=below), frac_set_dominated=float(T["dom_flag"].to_numpy()[ins].mean()),
                frac_universe_dominated=float(T["dom_flag"].mean()), direction="below" if below else "above")


def rho_job(j):
    h, c = j
    v = RUN.PRIMARY[h]
    T = DATA[c]
    cols = RUN.COVSETS[v["cov"]]
    m = T[[v["gold"], "D", "dom_value"] + cols].notna().all(1).to_numpy()
    U = T.loc[m]
    R = np.column_stack([S.midrank(U[v["gold"]]), S.midrank(U["D"])])
    Z0 = S.design(U, cols)
    if c.startswith("liv"):
        Z = S.design(U, cols + ["dom_value"])
    else:
        Z = S.design(U, cols, extra=U[["dom_value"]].to_numpy(float))
    rng = np.random.default_rng(zlib.crc32(f"dom:{h}:{c}".encode()))
    est = S.rho_p_matrix(R, Z)[1]
    Bb = S.bootstrap(R, Z, N_DRAWS, rng, U["block"].to_numpy())[:, 1]
    lab = S.strata(U["prior_raw"].to_numpy(float), U["log_cp10k_spmean"].to_numpy(float)) * 2 + U["dom_flag"].to_numpy(int)
    eD, eG = S.resid(R[:, 1], Z), S.resid(R[:, 0], Z)
    null = S.perm_null(eD, eG, lab, N_PERM, rng)
    lo, hi = S.pct_ci(Bb)
    return dict(kind="rho", id=f"{h}:{c}", n=int(m.sum()), rho_p_without=float(S.rho_p_matrix(R, Z0)[1]), estimate=float(est),
                ci_lo=lo, ci_hi=hi, p=S.p_one_sided(float(est), null))


def main():
    load()
    sets = [(t, s) for t, ss in RUN.SETS.items() for s in ss]
    rhos = [(h, v["cohort"]) for h, v in RUN.PRIMARY.items()] + [(h, c) for h, c in RUN.MULTI.items()]
    import multiprocessing as mp

    with mp.get_context("fork").Pool(16) as pool:
        a = pool.map(set_job, sets)
        b = pool.map(rho_job, rhos)
    df = pd.DataFrame(a + b)
    m = df.kind == "set"
    df.loc[m, "q_bh"] = S.bh(df.loc[m, "p"].tolist())
    tmp = OUT.with_name(OUT.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(OUT)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
