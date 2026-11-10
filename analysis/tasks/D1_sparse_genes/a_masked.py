"""D1 (a) and (d): masked genes on every cohort with a gold, coverage, gold-free controls, incremental rho. Rows in results/analysis/D1/a_*.csv.
    python -m analysis.tasks.D1_sparse_genes.a_masked units      list of units
    python -m analysis.tasks.D1_sparse_genes.a_masked run        all units, joblib over SLURM cpus
    python -m analysis.tasks.D1_sparse_genes.a_masked goldfree   gold-free controls, no gold read
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.tasks.D1_sparse_genes import common as CM
from analysis.tasks.D1_sparse_genes import rules as RU
from analysis.tasks.T4_3_scoreboard import stats as ST

ALWAYS = ["D", "S_fwd", "Sdir", "Lfc", "B3a", "B3b", "R_S_p", "R_S_72", "R_Sdir_p", "R_Sdir_72", "R_B1_p", "R_B1_72", "R_D_p", "kNN_S", "kNN_Sdir", "kNN_B1", "kNN_D", "Block_S", "random"]
PARTIAL = ["B1", "B1c", "B1z", "B2b", "Para_S"]   # defined on part of the genes: each on its own coverage, beside D on the same genes
STRATA = {"masked": lambda t: t.masked, "no_direct": lambda t: t.no_direct, "sp_sparse": lambda t: t.sp_sparse, "unmasked": lambda t: ~t.masked}


def rows_of(t, stratum, universe, split):
    m = STRATA[stratum](t).to_numpy(bool)
    if universe == "clean":
        m = m & t.clean.to_numpy(bool)
    return m & CM.split_rows(t, split)


def unit(c, kind, gold, covs, stratum, universe, split):
    t = CM.load(c)
    t["random"] = np.random.default_rng(RU.useed("random", c)).random(len(t))
    r0 = rows_of(t, stratum, universe, split)
    lab = dict(cohort=c, kind=kind, stratum=stratum, universe=universe, split=split)
    out = []
    draws = RU.N_DRAWS if (stratum, universe, split) == ("masked", "clean", "all") else RU.N_DRAWS_GRID
    if not (t[gold].notna().to_numpy() & r0).sum() >= CM.MIN_N:
        return [dict(lab, gold=gold, score="", rho_p=np.nan, n=int((t[gold].notna().to_numpy() & r0).sum()), group="too_few")]
    al = [s for s in ALWAYS if t.loc[r0, s].notna().mean() > 0.9]
    for r in CM.rho_rows(t, gold, covs, al, r0, "D", draws, lab):
        out.append(dict(r, group="always"))
    cp = [x for x in covs if x != "prior_raw"]
    for r in CM.rho_rows(t, gold, cp, ["pi"], r0, None, draws, lab):
        out.append(dict(r, group="pi_given_C_without_pi"))
    for s in PARTIAL:
        if s not in t or t.loc[r0, s].notna().sum() < CM.MIN_N:
            out.append(dict(lab, gold=gold, score=s, rho_p=np.nan, n=int((t[gold].notna().to_numpy() & r0 & t[s].notna().to_numpy()).sum()), group="too_few"))
            continue
        for r in CM.rho_rows(t, gold, covs, ["D", s], r0, "D", draws, lab):
            out.append(dict(r, group=f"own_{s}"))
    if stratum == "masked" and universe == "clean":  # (d)
        pcs = [f"esm{i}" for i in range(64)]
        if (t[gold].notna().to_numpy() & r0).sum() >= 200:
            for r in CM.rho_rows(t, gold, covs, ["D", "R_S_p"], r0, None, draws, lab, extra_linear=pcs):
                out.append(dict(r, group="given_C_64PCs"))
        for r in CM.rho_rows(t, gold, covs, ["D", "S_fwd"], r0, None, draws, lab, extra_spline=["kNN_S", "Block_S"]):
            out.append(dict(r, group="given_C_kNN_Block"))
        for r in CM.rho_rows(t, gold, covs, ["D"], r0, None, draws, lab, extra_spline=["R_S_p"]):
            out.append(dict(r, group="given_C_R_S_p"))
        for r in CM.rho_rows(t, gold, covs, ["R_S_p", "R_Sdir_p", "R_B1_p"], r0, "R_S_p", draws, lab):
            out.append(dict(r, group="target"))
    return out


def units():
    u = []
    for c in RU.GOLD_COHORTS:
        for kind, gold, covs in CM.golds(c):
            for st in STRATA:
                for spl in ("all", "odd", "even"):
                    u.append((c, kind, gold, covs, st, "clean", spl))
            u.append((c, kind, gold, covs, "masked", "allgenes", "all"))
    return u


def run():
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    res = Parallel(n_jobs=n, backend="loky", batch_size=1)(delayed(unit)(*x) for x in units())
    d = pd.DataFrame([r for x in res for r in x])
    d.to_csv(RU.RES / "a_rho.csv", index=False)
    print(d.shape, flush=True)


def goldfree():
    rows = []
    for c in RU.ALL_COHORTS:
        t = CM.load(c)
        m = (t.masked & t.clean).to_numpy()
        um = (~t.masked & t.clean).to_numpy()
        Z = ST.design(t.loc[m], RU.COV_C) if m.sum() > 100 else None
        E = t[[f"esm{i}" for i in range(64)]].to_numpy(float)
        r = dict(cohort=c, n_genes=len(t), n_masked=int(t.masked.sum()), n_masked_clean=int(m.sum()), n_no_direct=int((t.no_direct & t.clean).sum()), n_sp_sparse=int((t.sp_sparse & t.clean).sum()),
                 cov_B1=float(t.loc[m, "B1"].notna().mean()), cov_B1c=float(t.loc[m, "B1c"].notna().mean()), cov_B1z=float(t.loc[m, "B1z"].notna().mean()), cov_B2b=float(t.loc[m, "B2b"].notna().mean()),
                 cov_Sdir=float(t.loc[m, "Sdir"].notna().mean()), cov_Lfc=float(t.loc[m, "Lfc"].notna().mean()), cov_S_fwd=float(t.loc[m, "S_fwd"].notna().mean()), cov_D=float(t.loc[m, "D"].notna().mean()),
                 cov_Para_S=float(t.loc[m, "Para_S"].notna().mean()), median_expressing_cells=float(t.loc[m, "n_expr_total"].median()), S_nonzero=0.0)
        for s in ("kNN_S", "kNN_D", "R_S_p", "R_D_p", "B3b", "kNN_Sdir", "R_Sdir_p", "Sdir", "S_fwd", "prior_raw"):
            ok = t.loc[m, s].notna().to_numpy()
            if ok.sum() < 100:
                continue
            r[f"rho_D_{s}"] = float(spearmanr(t.loc[m, "D"][ok], t.loc[m, s][ok])[0])
            if s != "prior_raw" and ok.all():
                r[f"prho_D_{s}"] = CM.partial(t.loc[m, "D"].to_numpy(float), t.loc[m, s].to_numpy(float), Z)
        # cross-fitted R2 of D on the 64 ESM PCs, within masked and within unmasked (X4 mechanism_masked_r2); and trained on unmasked, applied on masked
        for nm, rows_ in (("masked", m), ("unmasked", um)):
            y = t.loc[rows_, "D"].to_numpy(float)
            X = E[rows_]
            pred = np.zeros(len(y))
            for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
                mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
                pred[te] = mdl.fit(X[tr], y[tr]).predict(X[te])
            r[f"R2_cv_ESM_{nm}"] = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(E[um], t.loc[um, "D"].to_numpy(float))
        p = mdl.predict(E[m])
        y = t.loc[m, "D"].to_numpy(float)
        r["R2_unmasked_to_masked"] = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        y = t.loc[m, "S_ema"].to_numpy(float)
        r["S_nonzero"] = float((np.abs(y) > 1e-6).mean())
        rows.append(r)
    pd.DataFrame(rows).to_csv(RU.RES / "a_goldfree.csv", index=False)


if __name__ == "__main__":
    {"run": run, "goldfree": goldfree, "units": lambda: print(len(units()))}[sys.argv[1]]()
