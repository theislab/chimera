"""D1 (b): depth experiment on genes that become sparse. Rows in results/revision/D1/b_rho.csv, per-gene frames in b_genes/.
    python -m revision.tasks.D1_sparse_genes.b_depth
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from revision.tasks.B2_benchmark import rules as B2
from revision.tasks.D1_sparse_genes import build as BD
from revision.tasks.D1_sparse_genes import common as CM
from revision.tasks.D1_sparse_genes import rules as RU
from revision.tasks.T4_2_gene_baselines import run as T42

METHODS = ["D", "S_fwd", "B1", "B1c", "Sdir", "Lfc", "B3a"]
FULL_ALIAS = {"S": "S_ema"}


def load_runs(c: str, setting: str):
    """Healthy replicates of a depth: B2 fits where they exist, D1 fits otherwise."""
    fr, used = [], []
    for r in range(3):
        for p, h in ((RU.RES / "scores" / f"d1-{c}-{setting}-r{r}.parquet", RU.RES / "health" / f"d1-{c}-{setting}-r{r}.json"),
                     (B2.RES / "scores" / f"b2-{c}-{setting}-r{r}.parquet", B2.RES / "health" / f"b2-{c}-{setting}-r{r}.json")):
            if p.exists() and h.exists():
                hh = json.loads(h.read_text())
                if not hh.get("discard", False):
                    fr.append(pd.read_parquet(p).set_index("hsapiens_gene_id"))
                    used.append(dict(run=p.stem, **{k: hh.get(k) for k in ("mix_entropy", "adv_acc_final", "rho_D_S", "frac_S_zero", "tail_pos", "tail_neg", "sat_hi", "active_dims", "n_cells", "lib_median")}))
                break
    return fr, used


def frame(c: str, setting: str):
    t = CM.load(c).set_index("hsapiens_gene_id")
    fr, used = load_runs(c, setting)
    if not fr:
        return None, used
    F = np.column_stack([t[f"esm{i}"].to_numpy(float) for i in range(64)])
    for m in METHODS + ["S_g"]:
        M = np.column_stack([f[m].reindex(t.index).to_numpy(float) if m in f else np.full(len(t), np.nan) for f in fr])
        with np.errstate(invalid="ignore"):
            n = np.isfinite(M).sum(1)
            t[f"{m}_f"] = np.where(n > 0, np.nansum(M, 1) / np.maximum(n, 1), np.nan)
    Ms = np.column_stack([np.abs(f["S_g"].reindex(t.index).to_numpy(float)) < 1e-6 for f in fr])
    t["masked_f"] = Ms.all(1)
    t["sparse_f"] = (~t.masked).to_numpy() & t.clean.to_numpy() & t["masked_f"].to_numpy()
    t["always_masked"] = t.masked.to_numpy() & t.clean.to_numpy()
    tr = (~t["masked_f"]).to_numpy()
    y = {"S": t["S_g_f"].to_numpy(float), "Sdir": t["Sdir_f"].to_numpy(float), "D": t["D_f"].to_numpy(float), "B1": t["B1_f"].to_numpy(float)}
    Fp = np.column_stack([F, t[["prior_raw"]].to_numpy(float)])  # protein columns available in every setting
    for n, v in y.items():
        yy = np.where(tr & np.isfinite(v), v, np.nan)
        t[f"R_{n}_f"] = T42.cross_fit_ridge(Fp, yy)[0]
    k = BD.knn_mean(F, tr, {f"kNN_{n}_f": v for n, v in y.items()})
    for n, v in k.items():
        t[n] = v
    return t.reset_index(), used


def ceiling(c: str, t: pd.DataFrame, rows: np.ndarray) -> dict:
    """Seed against the consensus of the other seeds, on these genes, given C."""
    h = pd.read_parquet(RU.REPO / f"results/revision/H1/genes/{c}.parquet").set_index("hsapiens_gene_id").reindex(t.hsapiens_gene_id)
    cols = [k for k in h.columns if k.startswith("alpha_s")]
    A = h[cols].to_numpy(float)[rows]
    pr = h["pi_clamped"].to_numpy(float)[rows]
    D = A - pr[:, None]
    Z = CM.ST.design(t.loc[rows], RU.COV_C)
    v, vr = [], []
    for j in range(D.shape[1]):
        o = np.delete(D, j, 1).mean(1)
        v.append(CM.partial(D[:, j], o, Z))
        vr.append(float(spearmanr(D[:, j], o)[0]))
    return dict(ceil_partial=float(np.mean(v)), ceil_raw=float(np.mean(vr)), n_seeds=len(cols))


def cell(c, setting, t, setn, rows, used):
    out = []
    lab = dict(cohort=c, setting=setting, set=setn, n_reps=len(used))
    tt = t.copy()
    tt.rename(columns={"S_ema": "S_full"}, inplace=True)
    for m in METHODS:
        tt[f"{m}_full"] = tt[m]
    F = list(RU.COV_C)
    all_f = [f"{m}_f" for m in METHODS] + ["R_S_f", "R_Sdir_f", "R_B1_f", "R_D_f", "kNN_S_f", "kNN_Sdir_f"]
    ful = ["D_full", "S_fwd_full", "Sdir_full", "Lfc_full", "B3a_full", "R_S_p", "kNN_S", "S_full"]
    # 1 own agreement
    for m in METHODS:
        rw = rows & tt[f"{m}_full"].notna().to_numpy() & tt[f"{m}_f"].notna().to_numpy()
        if rw.sum() >= CM.MIN_N:
            r = CM.X4.rho_unit(tt, f"{m}_full", F, [f"{m}_f"], rows=rw, label=lab, draws=RU.N_DRAWS_GRID)
            for x in r:
                x.update(metric="own_agreement", target=f"{m}_full", method=m, coverage=float(rw.sum() / max(rows.sum(), 1)),
                         raw=float(spearmanr(tt.loc[rw, f"{m}_full"], tt.loc[rw, f"{m}_f"])[0]))
                out.append(x)
        else:
            out.append(dict(lab, metric="own_agreement", target=f"{m}_full", method=m, n=int(rw.sum()), coverage=float(rw.sum() / max(rows.sum(), 1))))
    # 2 common target, each score on its own coverage with D_f beside; S_full and Sdir_full
    for target in ("S_full", "Sdir_full"):
        for s in all_f:
            if s == "D_f":
                continue
            rw = rows & tt[target].notna().to_numpy() & tt[s].notna().to_numpy() & tt["D_f"].notna().to_numpy()
            if rw.sum() < CM.MIN_N:
                out.append(dict(lab, metric="common_target", target=target, method=s[:-2], n=int(rw.sum()), coverage=float(rw.sum() / max(rows.sum(), 1))))
                continue
            for x in CM.X4.rho_unit(tt, target, F, ["D_f", s], rows=rw, label=lab, draws=RU.N_DRAWS_GRID, ref="D_f"):
                x.update(metric="common_target", target=target, method=x["score"][:-2], coverage=float(rw.sum() / max(rows.sum(), 1)), pair=s[:-2])
                out.append(x)
    # 3 gold: at this depth, and the same genes at full depth
    for kind, gold, covs in CM.golds(c):
        for spl in ("all", "odd", "even"):
            r0 = rows & CM.split_rows(tt, spl) & tt[gold].notna().to_numpy()
            if r0.sum() < CM.MIN_N:
                out.append(dict(lab, metric="gold", target=gold, kind=kind, split=spl, n=int(r0.sum())))
                continue
            for s in all_f + ful:
                if s in ("D_f",) or s not in tt:
                    continue
                rw = r0 & tt[s].notna().to_numpy() & tt["D_f"].notna().to_numpy()
                if rw.sum() < CM.MIN_N:
                    continue
                for x in CM.X4.rho_unit(tt, gold, covs, ["D_f", s], rows=rw, label=lab, draws=RU.N_DRAWS_GRID, ref="D_f"):
                    x.update(metric="gold", target=gold, kind=kind, split=spl, method=x["score"], coverage=float(rw.sum() / max(r0.sum(), 1)), pair=s)
                    out.append(x)
            pi = rows & CM.split_rows(tt, spl) & tt[gold].notna().to_numpy()
            for x in CM.X4.rho_unit(tt, gold, [k for k in covs if k != "prior_raw"], ["pi"], rows=pi, label=lab, draws=RU.N_DRAWS_GRID):
                x.update(metric="gold", target=gold, kind=kind, split=spl, method="pi", coverage=1.0, pair="pi")
                out.append(x)
    return out


def one(c, setting):
    t, used = frame(c, setting)
    if t is None:
        return [], [dict(cohort=c, setting=setting, n_reps=0)], []
    rows_out = []
    ceil = []
    for setn in ("sparse_f", "always_masked"):
        rows = t[setn].to_numpy(bool)
        ceil.append(dict(cohort=c, setting=setting, set=setn, n=int(rows.sum()), **ceiling(c, t, rows)))
        rows_out += cell(c, setting, t, setn, rows, used)
    (RU.RES / "b_genes").mkdir(exist_ok=True)
    keep = [k for k in t.columns if not k.startswith("esm")]
    t[keep].to_parquet(RU.RES / "b_genes" / f"{c}__{setting}.parquet")
    u = [dict(cohort=c, setting=setting, n_reps=len(used), **x) for x in used]
    return rows_out, u, ceil


def main():
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    todo = [(c, s) for c in RU.DEPTH_COHORTS for s in RU.DEPTHS]
    res = Parallel(n_jobs=min(n, len(todo)), backend="loky")(delayed(one)(*x) for x in todo)
    pd.DataFrame([r for x in res for r in x[0]]).to_csv(RU.RES / "b_rho.csv", index=False)
    pd.DataFrame([r for x in res for r in x[1]]).to_csv(RU.RES / "b_health.csv", index=False)
    pd.DataFrame([r for x in res for r in x[2]]).to_csv(RU.RES / "b_ceiling.csv", index=False)


if __name__ == "__main__":
    main()
