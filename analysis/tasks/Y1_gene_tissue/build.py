"""Y1: one gene table per tissue, profiles, direct margins and B1 cell halves. No gold is read and no two tissues are joined.

    python -I launch.py analysis.tasks.Y1_gene_tissue.build <tissue> [--no-direct] [--no-halves]

Writes results/analysis/Y1/cache/tissue__<t>.parquet, profiles__<t>.npz, Sdir__<t>.parquet, b1_halves__<t>.npz,
tissue_meta__<t>.json. A repro check against the T4.2 or task table's B1 is asserted: same E to 1e-9 on the same object and units.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.X4_where_differs import direct_margin as DM
from analysis.tasks.Y1_gene_tissue import common as Y
from analysis.tasks.Y1_gene_tissue import rules as RU


def readout_table(t: str, var_names) -> pd.DataFrame:
    """D, D_raw, S_ema, ppass, B1 (E of the human-mouse pair), prior_raw, block, per the tissue's own task."""
    R = Y.RES
    if t in ("liver", "testis"):
        c = Y.TISSUES[t]["table"]
        g = pd.read_parquet(R / "X4" / "genes" / f"{c}.parquet").set_index("var_name")
        cs = pd.read_csv(R / "T3_3" / f"{c}__consensus.csv").set_index("var_name")
        tt = pd.read_parquet(R / "T2_5" / f"{c}__genes.parquet").set_index("var_name")
        out = pd.DataFrame(dict(S=cs["S_ema"], D=g["D"], D_raw=g["D_raw"], B1=g["B1b_raw"], prior_raw=g["prior_raw"],
                                log_cds_length=g["log_cds_length"], block=g["block"], ppass=tt["ppass"]))
        for cl in [c_ for c_ in g.columns if c_.startswith("cls_")]:
            out[cl] = g[cl]
    elif t == "kidney":
        g = pd.read_parquet(R / "P13/readout/kidney_genes.parquet").set_index("var_name")
        out = pd.DataFrame(dict(S=g["S_ema"], D=g["D"], D_raw=g["D_raw"], B1=g["B1b_raw"], prior_raw=g["prior_raw"],
                                log_cds_length=g["log_cds_length"], block=g["block"], ppass=g["ppass"]))
    elif t == "islet":
        g = pd.read_parquet(R / "T7_9/readout/islet_genes.parquet").set_index("var_name")
        out = pd.DataFrame(dict(S=g["S_ema"], D=g["D"], D_raw=g["D_raw"], B1=g["B1b_raw"], prior_raw=g["prior_raw"],
                                log_cds_length=g["log_cds_length"], block=g["block"], ppass=g["ppass"]))
    elif t == "bone_marrow":
        g = pd.read_parquet(R / "P12/readout/bm_genes.parquet").set_index("var_name")
        out = pd.DataFrame(dict(S_ema_all=g["S_ema"], D=g["D"], D_raw=g["D_raw"], B1=g["E_human_mouse"], prior_raw=g["prior_raw"],
                                log_cds_length=g["log_cds_length"], block=g["block"], ppass=g["ppass"]))
    else:
        g = pd.read_parquet(R / "T11/readout/genes_hmouse.parquet").set_index("var_name")
        out = pd.DataFrame(dict(S=g["S_ema"], D=g["D"], D_raw=g["D_raw"], B1=g["B1_raw"], prior_raw=g["prior_raw"],
                                log_cds_length=g["log_cds_length"], block=g["block"], ppass=g["ppass"]))
    return out.reindex(var_names)


def seed_S(t: str, var_names, ppass: np.ndarray) -> pd.DataFrame:
    """S per seed: S_g_ema, or on bone marrow ppass x the human and mouse rows of the T2.5 forward margin."""
    d = Y.TISSUES[t]
    S = {}
    if t == "bone_marrow":
        for s in d["runs"]:
            fw = dict(np.load(Y.RES / "P12" / "drivers" / f"fw_s{s}.npz", allow_pickle=False))
            if list(fw["var_names"]) != list(var_names):
                raise AssertionError("bone marrow: var_names differ")
            sp = [str(x) for x in fw["species"]]
            hi, mi = sp.index("human"), sp.index("mouse")
            n = np.asarray(fw["n_species"], float)
            r = (fw["sp_sum_meanz"][hi] / n[hi] + fw["sp_sum_meanz"][mi] / n[mi]) / 2
            ema = np.load(Y.ROOT / "training/runs" / d["runs"][s] / "final_state.npz")["S_g_ema"]
            S[s] = np.where(np.abs(ema) >= RU.MASK_EPS, ppass * r, 0.0)
    else:
        for s, run in d["runs"].items():
            S[s] = K.read_run(run)["S_g_ema"]
    return pd.DataFrame(S, index=var_names)


def profiles(a, h, m, t):
    X = Y.csr(a.X)
    spc = a.obs["species"].astype(str).to_numpy()
    lab = a.obs["y1_group"].astype(str).to_numpy()
    lib = a.obs[Y.TISSUES[t]["unit"]].astype(str).to_numpy()
    x, f, units = T42.pseudobulk(X, spc, lab, lib, T42.MIN_CELLS_TYPE)
    groups = T42.shared_groups(x, [h, m], [h, m])
    return X, spc, lab, lib, x, f, units, groups


def covariates(a, X, spc, lab, x, h, m, groups) -> pd.DataFrame:
    Xb = X.copy()
    Xb.data = (Xb.data > 0).astype(np.float64)
    L = np.asarray(X.sum(1)).ravel()
    cp = (X.multiply(1e4 / np.maximum(L, 1)[:, None])).tocsr()
    cp.data = np.log1p(cp.data)
    expr, det, xh = [], [], {}
    for s in (h, m):
        expr.append(np.asarray(cp[spc == s].mean(0)).ravel())
        det.append(np.asarray(Xb[spc == s].mean(0)).ravel())
    A = np.stack([np.column_stack([x[(s, g)] for g in groups]) for s in (h, m)])  # species, genes, types
    # tau over shared types (T4.3 tau_shared): on CP10K means, species without counts left out of the mean
    taus = []
    cp10 = X.multiply(1e4 / np.maximum(L, 1)[:, None]).tocsr()
    for s in (h, m):
        M = np.column_stack([np.asarray(cp10[(spc == s) & (lab == g)].mean(0)).ravel() for g in groups])
        ts = C.tau_index(M)
        ts[M.max(1) <= 0] = np.nan
        taus.append(ts)
    T = np.column_stack(taus)
    tau = np.zeros(T.shape[0])
    ok = np.isfinite(T).any(1)
    tau[ok] = np.nanmean(T[ok], 1)
    amp = A.std(2, ddof=0).mean(0)
    top = A.max(2)
    dom = (top - (A.sum(2) - top) / max(len(groups) - 1, 1)).mean(0)
    shift = (A[1] - A[0]).mean(1)  # mouse minus human, mean over shared types, log2 CPM
    profile_shape = A.mean(0)
    return pd.DataFrame(dict(expr=np.mean(expr, 0), det=np.mean(det, 0), det_human=det[0], det_mouse=det[1], expr_human=expr[0],
                             expr_mouse=expr[1], tau=tau, amp=amp, dom=dom, shift=shift, abs_shift=np.abs(shift),
                             top_type=np.array(groups)[profile_shape.argmax(1)]), index=a.var_names)


def b1_halves(X, spc, lab, lib, h, m, n, n_splits, seed):
    """T4.3 baseline_splits rule: cells split 50/50 within (species, group, library), unit minimum halved."""
    rng = np.random.default_rng(seed)
    key = pd.Series(spc + "|" + lab + "|" + lib).groupby(spc + "|" + lab + "|" + lib).indices
    A, B = np.full((n_splits, n), np.nan, np.float32), np.full((n_splits, n), np.nan, np.float32)
    for b in range(n_splits):
        hf = np.zeros(len(spc), bool)
        for _, i in key.items():
            hf[rng.permutation(i)[: len(i) // 2]] = True
        for arr, mk in ((A, hf), (B, ~hf)):
            x, f, _ = T42.pseudobulk(X[mk], spc[mk], lab[mk], lib[mk], T42.MIN_CELLS_TYPE // 2)
            g = T42.shared_groups(x, [h, m], [h, m])
            r, _ = T42.pair_corr(x, f, h, m, g, n)
            arr[b] = r
    return A, B


def direct(t: str, a_hm) -> pd.DataFrame:
    if t in ("liver", "testis"):
        return pd.read_parquet(Y.RES / "X4" / "cache" / f"Sdir__{Y.TISSUES[t]['table']}.parquet")
    b = a_hm.copy()
    b.obs["cell_type"] = b.obs["y1_group"].values
    C.read_adata = lambda cohort, backed=False: b
    return DM.run(f"y1_{t}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tissue")
    p.add_argument("--no-direct", action="store_true")
    p.add_argument("--no-halves", action="store_true")
    a_ = p.parse_args()
    t = a_.tissue
    d = Y.TISSUES[t]
    a = Y.read(t)
    gene_ids.assert_table_matches(a, gene_ids.load_table(d["table"]))
    Y.CACHE.mkdir(parents=True, exist_ok=True)
    tb = readout_table(t, a.var_names)
    tb.insert(0, "hsapiens_gene_id", gene_ids.ensembl(d["table"], a.var_names).values)
    tb.insert(1, "symbol", gene_ids.symbols(d["table"], a.var_names).values)
    Sm = seed_S(t, a.var_names, tb["ppass"].to_numpy(float))
    if t == "bone_marrow":
        tb["S"] = Sm.mean(1)
    else:
        if not np.allclose(Sm.mean(1).to_numpy(), tb["S"].to_numpy(float), atol=1e-6):
            raise AssertionError(f"{t}: S is not the mean S_g_ema of the listed seeds")
    a_hm, h, m = Y.hm_view(a)
    X, spc, lab, lib, x, f, units, groups = profiles(a_hm, h, m, t)
    E, _ = T42.pair_corr(x, f, h, m, groups, a_hm.n_vars)
    rep = np.nanmax(np.abs(E - tb["B1"].to_numpy(float))) if np.isfinite(E).any() else np.nan
    same = np.array_equal(np.isfinite(E), np.isfinite(tb["B1"].to_numpy(float)))
    if not (rep < 1e-6 and same):
        raise AssertionError(f"{t}: B1 does not reproduce the task table, max diff {rep}, same defined {same}")
    cov = covariates(a_hm, X, spc, lab, x, h, m, groups)
    tb = tb.join(cov)
    Ma = Sm.to_numpy()
    unm = np.abs(Ma) >= RU.MASK_EPS
    tb["sign_stable_S"] = (np.sign(Ma) == np.sign(Ma[:, [0]])).all(1) & unm.all(1)
    tb["masked"] = ~(np.abs(tb["S"].to_numpy(float)) >= RU.MASK_EPS)
    tb["support"] = (~tb["masked"]) & (tb["ppass"] >= RU.PPASS_MIN)
    thr = np.median(np.abs(tb.loc[~tb["masked"], "S"]))
    tb["strict"] = tb["support"] & (np.abs(tb["S"]) >= thr) & tb["sign_stable_S"]
    if not tb["hsapiens_gene_id"].dropna().is_unique:
        raise AssertionError(f"{t}: duplicate human IDs")
    Sm.columns = [f"S_s{c}" for c in Sm.columns]
    tb = tb.join(Sm)
    tb["n_seeds"] = Sm.shape[1]
    tb.reset_index(names="var_name").to_parquet(Y.CACHE / f"tissue__{t}.parquet")
    # profiles: pooled counts per species x group, for worked genes
    types = sorted({l for l in lab})
    P = {"var_names": np.array(a_hm.var_names, dtype=str), "types": np.array(types), "shared": np.array(groups), "species": np.array([h, m])}
    L = np.asarray(X.sum(1)).ravel()
    for s in (h, m):
        cnt, ex, nc, ls = [], [], [], []
        for g in types:
            mk = (spc == s) & (lab == g)
            cnt.append(np.asarray(X[mk].sum(0)).ravel())
            ex.append(np.asarray((X[mk] > 0).sum(0)).ravel())
            nc.append(int(mk.sum()))
            ls.append(float(L[mk].sum()))
        P[f"counts_{s}"], P[f"nexp_{s}"] = np.array(cnt, np.float32), np.array(ex, np.float32)
        P[f"ncells_{s}"], P[f"libsum_{s}"] = np.array(nc), np.array(ls)
    for g in groups:
        for s in (h, m):
            P[f"x_{s}_{g}"] = x[(s, g)].astype(np.float32)
    np.savez_compressed(Y.CACHE / f"profiles__{t}.npz", **P)
    meta = dict(tissue=t, n_cells=dict(pd.Series(spc).value_counts()), n_cells_fit=int(a.n_obs), n_genes=int(a.n_vars),
                seeds=list(map(int, d["runs"])), K=len(groups), groups=groups, n_units=int(units["used"].sum()),
                median_library={s: float(np.median(L[spc == s])) for s in (h, m)}, n_B1=int(np.isfinite(E).sum()),
                n_support=int(tb["support"].sum()), n_strict=int(tb["strict"].sum()), repro_B1_max_diff=float(rep))
    Y.write_json(Y.CACHE / f"tissue_meta__{t}.json", meta)
    print(meta, flush=True)
    if not a_.no_direct:
        direct(t, a_hm).to_parquet(Y.CACHE / f"Sdir__{t}.parquet")
    if not a_.no_halves:
        A, B = b1_halves(X, spc, lab, lib, h, m, a_hm.n_vars, RU.N_HALVES, RU.SEED)
        np.savez_compressed(Y.CACHE / f"b1_halves__{t}.npz", a=A, b=B, var_names=np.array(a_hm.var_names, dtype=str))
    print("done", t, flush=True)


if __name__ == "__main__":
    main()
