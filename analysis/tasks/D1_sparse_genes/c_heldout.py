"""D1 (c): held-out chromosomes. Rows in results/analysis/D1/c_rho.csv.
    python -m analysis.tasks.D1_sparse_genes.c_heldout
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from analysis.tasks.D1_sparse_genes import build as BD
from analysis.tasks.D1_sparse_genes import common as CM
from analysis.tasks.D1_sparse_genes import rules as RU
from analysis.tasks.T4_2_gene_baselines import run as T42

TARGETS = ["D_full", "S_full", "Sdir_full", "B1_full"]


def runs_of(c, fold, arm):
    fr, used = [], []
    for r in RU.HOLD_SEEDS:
        run = f"d1-{c}-ho{fold}_{arm}-r{r}"
        p, h = RU.RES / "scores" / f"{run}.parquet", RU.RES / "health" / f"{run}.json"
        if p.exists() and h.exists():
            hh = json.loads(h.read_text())
            used.append(dict(run=run, **{k: hh.get(k) for k in ("discard", "mix_entropy", "adv_acc_final", "rho_D_S", "frac_S_zero", "tail_pos", "tail_neg", "sat_hi", "active_dims")}))
            if not hh.get("discard", False):
                f = pd.read_parquet(p).set_index("hsapiens_gene_id")
                fr.append(f[~f.index.duplicated(keep=False)])
    return fr, used


def frame(c, fold):
    t = CM.load(c).set_index("hsapiens_gene_id")
    fp, up = runs_of(c, fold, "P")
    fe, ue = runs_of(c, fold, "E")
    if not fp and not fe:
        return None, up + ue
    held = (fp or fe)[0]["held_out"].reindex(t.index).fillna(False).to_numpy(bool)
    t["held"] = held
    for nm, fr in (("P", fp), ("E", fe)):
        if not fr:
            continue
        t[f"D_{nm}"] = np.mean([f["D"].reindex(t.index).to_numpy(float) for f in fr], 0)
        t[f"S_{nm}"] = np.mean([f["S_g"].reindex(t.index).to_numpy(float) for f in fr], 0)
        t[f"Sfit_masked_{nm}"] = np.abs(t[f"S_{nm}"]) < 1e-6
    if fe and "D_E0" in fe[0]:
        t["D_E0"] = np.mean([f["D_E0"].reindex(t.index).to_numpy(float) for f in fe], 0)
    E = t[[f"esm{i}" for i in range(64)]].to_numpy(float)
    Fp = np.column_stack([E, t[["prior_raw"]].to_numpy(float)])
    base = "P" if fp else "E"
    tr = (~held) & ~t[f"Sfit_masked_{base}"].to_numpy()
    y = {"S": t[f"S_{base}"].to_numpy(float), "Sdir": t["Sdir"].to_numpy(float), "B1": t["B1"].to_numpy(float), "D": t[f"D_{base}"].to_numpy(float)}
    for n, v in y.items():
        yy = np.where(tr & np.isfinite(v), v, np.nan)
        t[f"Rh_{n}"] = T42.cross_fit_ridge(Fp, yy)[0]
    k = BD.knn_mean(E, tr, {f"kNNh_{n}": v for n, v in y.items()})
    for n, v in k.items():
        t[n] = v
    para = BD.para_s(t.reset_index(), tr, S=f"S_{base}")
    t["Parah_S"] = para.reindex(t.index).to_numpy()
    p = pd.read_parquet(BD.PARA)
    p = p[p.identity >= RU.PARALOG_ID]
    outside = set(t.index[~held])
    hasp = set(p.loc[p.homology_gene_stable_id.isin(outside), "gene_stable_id"])
    t["paralog_outside"] = t.index.isin(hasp)
    t = t.rename(columns={"S_ema": "S_full"})
    for m in ("D", "Sdir", "B1"):
        t[f"{m}_full"] = t[m]
    return t.reset_index(), up + ue


def ceiling(c, t, rows):
    h = pd.read_parquet(RU.REPO / f"results/analysis/H1/genes/{c}.parquet").set_index("hsapiens_gene_id").reindex(t.hsapiens_gene_id)
    cols = [k for k in h.columns if k.startswith("alpha_s")]
    D = h[cols].to_numpy(float)[rows] - h["pi_clamped"].to_numpy(float)[rows][:, None]
    Z = CM.ST.design(t.loc[rows], RU.COV_C)
    return dict(ceil_partial=float(np.mean([CM.partial(D[:, j], np.delete(D, j, 1).mean(1), Z) for j in range(D.shape[1])])),
                ceil_raw=float(np.mean([spearmanr(D[:, j], np.delete(D, j, 1).mean(1))[0] for j in range(D.shape[1])])))


def score_set(c, fold, t, setn, rows, used):
    out = []
    preds = [k for k in ("D_P", "D_E", "D_E0", "Rh_D", "Rh_S", "Rh_Sdir", "Rh_B1", "kNNh_D", "kNNh_S", "kNNh_Sdir", "kNNh_B1", "Parah_S", "pi") if k in t]
    if setn == "trained_unmasked":  # reference only: the same fit on genes it read
        preds = [k for k in ("D_P", "D_E") if k in t]
    ref = "D_P" if "D_P" in t else "D_E"
    lab = dict(cohort=c, fold=fold, set=setn, n_seeds_P=sum(1 for u in used if "_P-" in u["run"]))
    tg = [(x, x, list(RU.COV_C), "measured") for x in TARGETS] + [(gold, gold, covs, kind) for kind, gold, covs in CM.golds(c)]
    for name, col, covs, kind in tg:
        r0 = rows & t[col].notna().to_numpy()
        if r0.sum() < CM.MIN_N:
            out.append(dict(lab, target=name, kind=kind, n=int(r0.sum())))
            continue
        measured = ["D_full", "S_full", "Sdir_full", "B1_full"] if kind != "measured" else []
        for s in preds + measured:
            use_covs = [k for k in covs if k != "prior_raw"] if s == "pi" else covs
            rw = r0 & t[s].notna().to_numpy() & t[ref].notna().to_numpy()
            if rw.sum() < CM.MIN_N:
                continue
            sc = [s] if s in ("pi", ref) else [ref, s]
            for x in CM.X4.rho_unit(t, col, use_covs, sc, rows=rw, label=lab, draws=RU.N_DRAWS_GRID, ref=ref if len(sc) == 2 else None):
                if x["score"] == s:
                    x.update(target=name, kind=kind, raw=float(spearmanr(t.loc[rw, col], t.loc[rw, s])[0]), raw_D=float(spearmanr(t.loc[rw, col], t.loc[rw, ref])[0]))
                    out.append(x)
    return out


def one(c, fold):
    t, used = frame(c, fold)
    if t is None:
        return [], [], []
    rows = []
    cl = t.clean.to_numpy(bool)
    held = t.held.to_numpy(bool)
    mk = t.masked.to_numpy(bool)
    sets = {"held_unmasked": held & cl & ~mk, "held_masked": held & cl & mk, "held_unmasked_noparalog": held & cl & ~mk & ~t.paralog_outside.to_numpy(bool),
            "trained_unmasked": (~held) & cl & ~mk}
    ce = []
    for n, r in sets.items():
        ce.append(dict(cohort=c, fold=fold, set=n, n=int(r.sum()), **(ceiling(c, t, r) if r.sum() > 60 else {})))
        rows += score_set(c, fold, t, n, r, used)
    (RU.RES / "c_genes").mkdir(exist_ok=True)
    t[[k for k in t.columns if not k.startswith("esm")]].to_parquet(RU.RES / "c_genes" / f"{c}__{fold}.parquet")
    return rows, [dict(cohort=c, fold=fold, **u) for u in used], ce


def main():
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", 6))
    todo = [(c, f) for c in RU.DEPTH_COHORTS for f in "AB"]
    res = Parallel(n_jobs=min(n, len(todo)), backend="loky")(delayed(one)(*x) for x in todo)
    pd.DataFrame([r for x in res for r in x[0]]).to_csv(RU.RES / "c_rho.csv", index=False)
    pd.DataFrame([r for x in res for r in x[1]]).to_csv(RU.RES / "c_health.csv", index=False)
    pd.DataFrame([r for x in res for r in x[2]]).to_csv(RU.RES / "c_ceiling.csv", index=False)


if __name__ == "__main__":
    main()
