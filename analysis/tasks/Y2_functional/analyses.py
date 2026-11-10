"""Y2 (EXPLORATORY, Family 3): tests of rules.py. Sections: pheno, viability, regulatory, discordant.

--mechanics replaces every readout and baseline by independent noise and writes to mechanics/, so that the machinery can
be checked with no association read.
Usage (from chimera/): PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m analysis.tasks.Y2_functional.analyses [--mechanics] [--sections ...]
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm, rankdata

from analysis.common import gene_ids
from analysis.tasks.T4_3_scoreboard import stats as S
from analysis.tasks.X4_where_differs import common as CM
from analysis.tasks.Y2_functional import rules as RU

REPO = gene_ids.REPO_ROOT
RES = REPO / "results/analysis"
OUT = RES / "Y2"
SCORES = RU.SCORES + ["random"]
PRIMARY_SCORES = ["D", "S", "B1b_raw", "B1c_raw", "B1z_raw", "B2b", "B3a"]
PUB = ["log_pubmed_h", "log_pubmed_m"]
N_JOBS = int(os.environ.get("Y2_JOBS", 16))


def z_design(T: pd.DataFrame, covs: list[str]) -> np.ndarray:
    if len(T) >= RU.SPLINE_MIN:
        return S.design(T, covs)
    Z = [np.ones(len(T))]
    for c in covs:
        r = rankdata(T[c].to_numpy(float))
        Z.append((r - r.mean()) / (r.std() + 1e-12))
    return np.column_stack(Z)


class Draws:
    """Block multiplicities shared by every cell: same draw, same blocks, in every cohort that uses this partition."""

    def __init__(self, labels, n_draws: int, seed: int):
        self.index = {b: i for i, b in enumerate(sorted(set(labels)))}
        rng = np.random.default_rng(seed)
        B = len(self.index)
        self.M = np.vstack([np.bincount(rng.integers(0, B, B), minlength=B) for _ in range(n_draws)]).astype(np.int32)

    def weights(self, labels) -> np.ndarray:
        return self.M[:, [self.index[b] for b in labels]].astype(float)


def make_cell(t: pd.DataFrame, rows, y: str, covs: list[str], scores: list[str], block: str, y_sign: float = 1.0) -> dict | None:
    cols = [y] + covs + scores
    m = t[cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy() & np.asarray(rows, bool)
    if m.sum() < 30:
        return None
    T = t.loc[m]
    R = np.column_stack([rankdata(y_sign * T[y].to_numpy(float))] + [rankdata(T[s].to_numpy(float)) for s in scores])
    if np.ptp(R[:, 0]) == 0:
        return None
    return dict(R=R, Z=z_design(T, covs), labels=T[block].astype(str).to_list(), scores=scores, n=int(m.sum()),
                n_pos=float(T[y].sum()) if set(np.unique(T[y])) <= {0.0, 1.0} else np.nan, genes=T["hsapiens_gene_id"].to_list())


def eval_cell(c: dict, D: Draws) -> tuple[np.ndarray, np.ndarray]:
    est = S.rho_p_matrix(c["R"], c["Z"])[1:]
    W = D.weights(c["labels"])
    B = np.vstack([S.weighted_rho(c["R"], c["Z"], W[b])[1:] for b in range(W.shape[0])])
    return est, B


def summar(est, draws) -> dict:
    d = draws[np.isfinite(draws)]
    if len(d) < 20 or not np.isfinite(est):
        return dict(est=est, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
    lo, hi = np.quantile(d, [0.025, 0.975])
    sd = d.std()
    return dict(est=float(est), ci_lo=float(lo), ci_hi=float(hi), p=float(2 * norm.sf(abs(est) / sd)) if sd > 0 else np.nan)


def bh_col(df: pd.DataFrame, rows=None) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    idx = df.index if rows is None else df.index[rows]
    p = df.loc[idx, "p"]
    p = p.dropna()
    if len(p):
        out[p.index] = S.bh(p.tolist())
    return out


def load(mechanics: bool) -> dict[str, pd.DataFrame]:
    tabs = {k: pd.read_parquet(OUT / f"genes_{k}.parquet") for k in ("liver", "testis", "bone_marrow", "kidney", "liver4sp")}
    rng = np.random.default_rng(RU.SEED)
    for k, t in tabs.items():
        t["random"] = rng.random(len(t))
        t["dNdS"] = t["dNdS"].where(t["dNdS"] > 0)
        if mechanics:
            for s in RU.SCORES:
                if s in t:
                    t[s] = rng.random(len(t))
        if "block" in t:
            t["block"] = t["block"].astype(str)
    return tabs


# ------------------------------------------------------------------ (a)
ADJ = {
    "C_Y2": lambda k: list(RU.C_Y2),
    "no_pubmed": lambda k: [c for c in RU.C_Y2 if c not in PUB],
    "C_only": lambda k: ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"],
    "pleiotropy": lambda k: list(RU.C_Y2) + ["n_mp_sys", "n_hp_sys"],
    "with_spread": lambda k: list(RU.C_Y2) + ["spread"],
}


def universe(t, u):
    return t[u].to_numpy(bool)


def pheno(tabs, out: dict):
    L, T = tabs["liver"], tabs["testis"]
    labels = list(L["gblock"].astype(str)) + list(T["gblock"].astype(str))
    D = Draws(labels, RU.DRAWS, RU.SEED)
    defs = []
    for u in ("ms", "un"):
        for adj in ADJ:
            defs.append((u, adj, "system", "hepmet", "repro"))
        for name in ("impc", "clingen"):
            defs.append((u, "C_Y2", name, "hepmet", "repro"))
        defs.append((u, "C_Y2", "narrow", "hepatic", "male"))
    cells = {}
    keys = []
    for u, adj, mode, a, b in defs:
        for tissue, tt in (("liver", L), ("testis", T)):
            for sys_ in (a, b):
                key = (u, adj, mode, tissue, sys_)
                ycol, ucol = (f"yi_{sys_}", f"Ui_{sys_}") if mode == "impc" else (f"y_{sys_}", f"U_{sys_}")
                rows = universe(tt, u) & tt[ucol].to_numpy(bool)
                if mode == "clingen":
                    rows = rows & tt["clingen_ok"].fillna(False).to_numpy(bool)
                cov = ADJ[adj](sys_)
                c = make_cell(tt, rows, ycol, cov, SCORES, "gblock")
                if c is not None:
                    cells[key] = c
                    keys.append(key)
    # all other systems on the primary adjustment, for the matrix and negative control
    for u in ("ms", "un"):
        for tissue, tt in (("liver", L), ("testis", T)):
            for sys_ in ("heme", "renal", "eye"):
                key = (u, "C_Y2", "system", tissue, sys_)
                c = make_cell(tt, universe(tt, u) & tt[f"U_{sys_}"].to_numpy(bool), f"y_{sys_}", RU.C_Y2, SCORES, "gblock")
                if c is not None:
                    cells[key] = c
                    keys.append(key)
    res = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(eval_cell)(cells[k], D) for k in keys)
    E = dict(zip(keys, res))
    rows = []
    for k, (est, B) in E.items():
        c = cells[k]
        for j, s in enumerate(c["scores"]):
            rows.append(dict(zip(("universe", "adj", "mode", "tissue", "system"), k)) | dict(score=s, n=c["n"], n_pos=c["n_pos"]) | summar(est[j], B[:, j]))
    cell_t = pd.DataFrame(rows)
    prim = (cell_t["adj"] == "C_Y2") & (cell_t["mode"] == "system") & cell_t["system"].isin(["hepmet", "repro"]) & cell_t["score"].isin(PRIMARY_SCORES)
    cell_t["q"] = np.nan
    for u in ("ms", "un"):
        m = prim & (cell_t["universe"] == u)
        cell_t.loc[m, "q"] = bh_col(cell_t, m)
    out["pheno_cells"] = cell_t
    # contrasts
    con = []
    for u, adj, mode, a, b in defs:
        ks = [(u, adj, mode, "liver", a), (u, adj, mode, "liver", b), (u, adj, mode, "testis", b), (u, adj, mode, "testis", a)]
        if not all(k in E for k in ks):
            continue
        ests = [E[k][0] for k in ks]
        draws = [E[k][1] for k in ks]
        score = cells[ks[0]]["scores"]
        d_est = 0.5 * ((ests[0] - ests[1]) + (ests[2] - ests[3]))
        d_B = 0.5 * ((draws[0] - draws[1]) + (draws[2] - draws[3]))
        l_est, l_B = ests[0] - ests[1], draws[0] - draws[1]
        t_est, t_B = ests[2] - ests[3], draws[2] - draws[3]
        ref = {x: score.index(x) for x in ("B1b_raw", "B1c_raw", "B1z_raw", "B2b", "B3a")}
        for j, s in enumerate(score):
            base = dict(universe=u, adj=adj, mode=mode, sys_a=a, sys_b=b, score=s, n_cells="/".join(str(cells[k]["n"]) for k in ks))
            con.append(base | dict(stat="Delta_match") | summar(d_est[j], d_B[:, j]))
            con.append(base | dict(stat="liver_diff") | summar(l_est[j], l_B[:, j]))
            con.append(base | dict(stat="testis_diff") | summar(t_est[j], t_B[:, j]))
            if s in ("D", "S", "Sdir", "Sdir_pattern", "Sdir_level"):
                for r, jr in ref.items():
                    con.append(base | dict(stat=f"minus_{r}") | summar(d_est[j] - d_est[jr], d_B[:, j] - d_B[:, jr]))
    con = pd.DataFrame(con)
    con["q"] = np.nan
    for u in ("ms", "un"):
        m = (con["universe"] == u) & (con["adj"] == "C_Y2") & (con["mode"] == "system") & (((con["stat"] == "Delta_match") & con["score"].isin(PRIMARY_SCORES)) | ((con["stat"] == "minus_B1b_raw") & con["score"].isin(["D", "S"])))
        con.loc[m, "q"] = bh_col(con, m)
    out["pheno_contrast"] = con


def _logit_fit(X, y, w, ridge=1e-3, it=25):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        eta = np.clip(X @ b, -20, 20)
        p = 1 / (1 + np.exp(-eta))
        W = w * p * (1 - p) + 1e-9
        A = (X.T * W) @ X + ridge * np.eye(X.shape[1])
        g = X.T @ (w * (y - p)) - ridge * b
        step = np.linalg.solve(A, g)
        b = b + step
        if np.abs(step).max() < 1e-6:
            break
    return b


def _inter_one(T, labels, ycol, mcol, score, covs, D: Draws):
    def zr(x):
        r = rankdata(x)
        return (r - r.mean()) / (r.std() + 1e-12)

    x, m = zr(T[score].to_numpy(float)), T[mcol].to_numpy(float)
    X = np.column_stack([np.ones(len(T)), m, x, m * x] + [zr(T[c].to_numpy(float)) for c in covs])
    y = T[ycol].to_numpy(float)
    est = _logit_fit(X, y, np.ones(len(y)))[3]
    W = D.weights(labels)
    B = np.array([_logit_fit(X, y, W[b])[3] for b in range(W.shape[0])])
    return est, B


def interaction(tabs, out: dict):
    L, T = tabs["liver"], tabs["testis"]
    D = Draws(list(L["gblock"].astype(str)) + list(T["gblock"].astype(str)), 500, RU.SEED + 1)
    scores = ["D", "S", "Sdir_pattern", "Sdir_level", "B1b_raw", "B1c_raw", "B1z_raw", "B2b", "random"]
    jobs = []
    for u in ("ms", "un"):
        for tissue, tt in (("liver", L), ("testis", T)):
            for sys_ in ("hepmet", "repro"):
                cols = [f"H_{sys_}", f"M_{sys_}"] + scores + RU.C_Y2
                ok = tt[cols].astype(float).replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy() & universe(tt, u) & tt["in_A"].to_numpy(bool)
                TT = tt.loc[ok].copy()
                for k in (f"H_{sys_}", f"M_{sys_}"):
                    TT[k] = TT[k].astype(float)
                labels = TT["gblock"].astype(str).to_list()
                for s in scores:
                    jobs.append(((u, tissue, sys_, s, len(TT)), (TT[[f"H_{sys_}", f"M_{sys_}", s] + RU.C_Y2], labels, f"H_{sys_}", f"M_{sys_}", s, RU.C_Y2, D)))
    res = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(_inter_one)(*a) for _, a in jobs)
    E = {k[:4]: r for (k, _), r in zip(jobs, res)}
    rows = []
    for (k, _), r in zip(jobs, res):
        rows.append(dict(zip(("universe", "tissue", "system", "score", "n"), k)) | dict(stat="beta_MxX") | summar(*r))
    for u in ("ms", "un"):
        for s in scores:
            a = E[(u, "liver", "hepmet", s)]; b = E[(u, "liver", "repro", s)]; c = E[(u, "testis", "repro", s)]; d = E[(u, "testis", "hepmet", s)]
            rows.append(dict(universe=u, tissue="both", system="match", score=s, stat="Delta_match_logit") | summar(0.5 * ((a[0] - b[0]) + (c[0] - d[0])), 0.5 * ((a[1] - b[1]) + (c[1] - d[1]))))
        for s in ("D", "S"):
            def dm(sc):
                a = E[(u, "liver", "hepmet", sc)]; b = E[(u, "liver", "repro", sc)]; c = E[(u, "testis", "repro", sc)]; d = E[(u, "testis", "hepmet", sc)]
                return 0.5 * ((a[0] - b[0]) + (c[0] - d[0])), 0.5 * ((a[1] - b[1]) + (c[1] - d[1]))
            e1, b1 = dm(s); e2, b2 = dm("B1b_raw")
            rows.append(dict(universe=u, tissue="both", system="match", score=s, stat="minus_B1b_logit") | summar(e1 - e2, b1 - b2))
    out["pheno_interaction"] = pd.DataFrame(rows)


def matrix(tabs, out: dict):
    """Replication: each tissue's scores against the four systems; own ESM blocks; diagonal minus mean off-diagonal."""
    diag = {"liver": "hepmet", "testis": "repro", "bone_marrow": "heme", "kidney": "renal"}
    rows = []
    for tissue, own in diag.items():
        t = tabs[tissue]
        scores = [s for s in ("D", "S", "B1b_raw", "B1c_raw", "B2b", "random") if s in t and t[s].notna().sum() > 100]
        D = Draws(t["block"].astype(str).tolist(), RU.DRAWS, RU.SEED + 7)
        for u in ("ms", "un"):
            est, draws, ns = {}, {}, {}
            for sys_ in diag.values():
                c = make_cell(t, universe(t, u) & t[f"U_{sys_}"].to_numpy(bool), f"y_{sys_}", RU.C_Y2, scores, "block")
                if c is None:
                    continue
                est[sys_], draws[sys_] = eval_cell(c, D)
                ns[sys_] = c["n"]
                for j, s in enumerate(scores):
                    rows.append(dict(tissue=tissue, universe=u, system=sys_, matched=sys_ == own, score=s, stat="cell", n=c["n"], n_pos=c["n_pos"]) | summar(est[sys_][j], draws[sys_][:, j]))
            off = [k for k in est if k != own]
            for j, s in enumerate(scores):
                e = est[own][j] - np.mean([est[k][j] for k in off])
                B = draws[own][:, j] - np.mean([draws[k][:, j] for k in off], axis=0)
                rows.append(dict(tissue=tissue, universe=u, system=own, matched=True, score=s, stat="diag_minus_offdiag") | summar(e, B))
                if s != "B1b_raw":
                    j2 = scores.index("B1b_raw")
                    e2 = est[own][j2] - np.mean([est[k][j2] for k in off])
                    B2 = draws[own][:, j2] - np.mean([draws[k][:, j2] for k in off], axis=0)
                    rows.append(dict(tissue=tissue, universe=u, system=own, matched=True, score=s, stat="diag_minus_offdiag_minus_B1b") | summar(e - e2, B - B2))
    out["pheno_matrix"] = pd.DataFrame(rows)


# ------------------------------------------------------------------ (a2)
def viability(tabs, out: dict):
    rows = []
    for tissue in ("liver", "testis", "bone_marrow", "kidney"):
        t = tabs[tissue].copy()
        scores = [s for s in SCORES if s in t and t[s].notna().sum() > 100]
        lethal = t["lethal_known"]
        con = t["LOEUF"] < RU.LOEUF_CONSTRAINED
        ess = t["depmap_common"]
        t["U_L"] = lethal.notna() & t["LOEUF"].notna() & ((lethal == 1) | con)
        t["y_L"] = ((lethal == 1) & con).astype(float).where(t["U_L"])
        t["U_E"] = lethal.notna() & ess.notna() & ((lethal == 1) | (ess == 1))
        t["y_E"] = ((lethal == 1) & (ess == 1)).astype(float).where(t["U_E"])
        t["neg_dep"] = -t["depmap_mean"]
        D = Draws(t["block"].astype(str).tolist(), RU.DRAWS, RU.SEED + 11)
        cov_l = [c for c in RU.C_Y2 if c != "LOEUF"]
        spec = [("y_L", "U_L", cov_l, None), ("y_E", "U_E", RU.C_Y2, None), ("y_L_impc", "U_L", cov_l, "impc_tested"),
                ("y_E_impc", "U_E", RU.C_Y2, "impc_tested"), ("neg_dep_lethal", None, RU.C_Y2, "lethal")]
        t["y_L_impc"], t["y_E_impc"], t["neg_dep_lethal"] = t["y_L"], t["y_E"], t["neg_dep"]
        for u in ("ms", "un"):
            for y, U, cov, flt in spec:
                r = universe(t, u)
                if U:
                    r = r & t[U].to_numpy(bool)
                if flt == "impc_tested":
                    r = r & t["impc_tested"].fillna(False).to_numpy(bool)
                if flt == "lethal":
                    r = r & (lethal == 1).to_numpy()
                c = make_cell(t, r, y, cov, scores, "block")
                if c is None:
                    continue
                e, B = eval_cell(c, D)
                for j, s in enumerate(scores):
                    rows.append(dict(tissue=tissue, universe=u, outcome=y, score=s, n=c["n"], n_pos=c["n_pos"]) | summar(e[j], B[:, j]))
    r = pd.DataFrame(rows)
    r["q"] = np.nan
    m = r["outcome"].isin(["y_L", "y_E"]) & r["tissue"].isin(["liver", "testis"]) & r["score"].isin(PRIMARY_SCORES)
    r.loc[m, "q"] = bh_col(r, m)
    out["viability"] = r


# ------------------------------------------------------------------ (b)
def regulatory(tabs, out: dict):
    rows, cmp_rows = [], []
    no_pub = [c for c in RU.C_Y2 if c not in PUB]
    for tissue in ("liver", "testis", "bone_marrow", "kidney"):
        t = tabs[tissue].copy()
        scores = [s for s in SCORES if s in t and t[s].notna().sum() > 100]
        t["log_abc"] = np.log1p(t["abc_links_mean"])
        t["log_abc_liver"] = np.log1p(t["abc_links_liver"])
        t["neg_logw"] = -np.log(t["dNdS"])
        t["neg_loeuf"] = -t["LOEUF"]
        D = Draws(t["block"].astype(str).tolist(), RU.DRAWS, RU.SEED + 13)
        outs = [("phylop_promoter", ["gc_promoter", "log_length_promoter"]), ("phylop_utr5", ["gc_utr5", "log_length_utr5"]),
                ("phylop_utr3", ["gc_utr3", "log_length_utr3"]), ("phylop_fc_promoter", ["gc_promoter", "log_length_promoter"]),
                ("phylop_fc_utr3", ["gc_utr3", "log_length_utr3"]), ("promoter_cgi", []), ("tata", []), ("log_abc", [])]
        if tissue == "liver":
            outs.append(("log_abc_liver", []))
        for u in ("ms", "un"):
            for y, extra in outs:
                c = make_cell(t, universe(t, u), y, no_pub + extra, scores, "block")
                if c is None:
                    continue
                e, B = eval_cell(c, D)
                for j, s in enumerate(scores):
                    rows.append(dict(tissue=tissue, universe=u, outcome=y, score=s, adj="C_REG", n=c["n"], n_pos=c["n_pos"]) | summar(e[j], B[:, j]))
                if "B1b_raw" in scores:
                    jb = scores.index("B1b_raw")
                    for j, s in enumerate(scores):
                        if s in ("D", "S"):
                            rows.append(dict(tissue=tissue, universe=u, outcome=y, score=s, adj="C_REG", stat="minus_B1b", n=c["n"]) | summar(e[j] - e[jb], B[:, j] - B[:, jb]))
            # regulatory vs coding on one gene set
            comp = [("phylop_promoter", ["gc_promoter", "log_length_promoter"]), ("phylop_utr3", ["gc_utr3", "log_length_utr3"]),
                    ("phylop_utr5", ["gc_utr5", "log_length_utr5"]), ("phylop_cds", ["gc_cds"]), ("neg_logw", []), ("neg_loeuf", [])]
            need = [y for y, _ in comp] + [x for _, ex in comp for x in ex] + RU.C_BASE + scores
            ok = t[need].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy() & universe(t, u)
            res = {}
            for y, extra in comp:
                c = make_cell(t, ok, y, RU.C_BASE + extra, scores, "block")
                if c is not None:
                    res[y] = eval_cell(c, D)
                    n = c["n"]
            for j, s in enumerate(scores):
                for y in res:
                    cmp_rows.append(dict(tissue=tissue, universe=u, score=s, stat=y, n=n) | summar(res[y][0][j], res[y][1][:, j]))
                for a in ("phylop_promoter", "phylop_utr3", "phylop_utr5"):
                    for b in ("phylop_cds", "neg_logw", "neg_loeuf"):
                        cmp_rows.append(dict(tissue=tissue, universe=u, score=s, stat=f"{a}-{b}", n=n) | summar(res[a][0][j] - res[b][0][j], res[a][1][:, j] - res[b][1][:, j]))
    r = pd.DataFrame(rows)
    r["q"] = np.nan
    if "stat" not in r:
        r["stat"] = np.nan
    m = r["tissue"].isin(["liver", "testis"]) & r["stat"].isna() & r["score"].isin(PRIMARY_SCORES) & r["outcome"].isin(["phylop_promoter", "phylop_utr5", "phylop_utr3", "promoter_cgi", "tata", "log_abc"])
    for u in ("ms", "un"):
        mm = m & (r["universe"] == u)
        r.loc[mm, "q"] = bh_col(r, mm)
    out["regulatory"] = r
    c = pd.DataFrame(cmp_rows)
    out["regulatory_vs_coding"] = c


# ------------------------------------------------------------------ (c)
def classes(t: pd.DataFrame, mechanics=False) -> dict:
    C4 = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]
    pi = CM.tertile(t["prior_raw"])
    out = {}
    un, ms = t["un"].to_numpy(bool), t["ms"].to_numpy(bool)
    rD = CM.tertile(CM.residual_rank(t, "D", C4).where(pd.Series(un, index=t.index)))
    S_ = t["S"].to_numpy(float)
    out["PClow_D_ms"] = ((pi == 3) & (rD == 1)).to_numpy() & ms & (S_ < 0), ms
    out["PDhigh_D_ms"] = ((pi == 1) & (rD == 3)).to_numpy() & ms & (S_ > 0), ms
    for name, col in (("S", "S"), ("B1b", "B1b_raw"), ("B1c", "B1c_raw")):
        if col not in t or t[col].notna().sum() < 100:
            continue
        r = CM.tertile(CM.residual_rank(t, col, C4).where(pd.Series(un, index=t.index)))
        out[f"PClow_{name}_un"] = ((pi == 3) & (r == 1)).to_numpy() & un, un
        out[f"PDhigh_{name}_un"] = ((pi == 1) & (r == 3)).to_numpy() & un, un
    return out


def discordant(tabs, out: dict):
    rows, sizes, member = [], [], []
    x2 = pd.read_csv(RES / "X2/divergent_conserved.csv")
    x2_ids = set(x2.loc[x2["cohort"].eq("liv_4sp"), "hsapiens_gene_id"])
    for tissue in ("liver", "testis", "liver4sp"):
        t = tabs[tissue].copy()
        t["log_abc"] = np.log1p(t["abc_links_mean"])
        for k in ("M_any", "H_any", "mouse_lethal", "promoter_cgi", "tata"):
            t[k] = t[k].astype(float)
        match = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length", "tau_shared", "log_pubmed_h"]
        if tissue != "testis":
            match = match + ["hep_enrich"]
        cl = {}
        if tissue == "liver4sp":
            cl["X2_list_63"] = ((t["hsapiens_gene_id"].isin(x2_ids) & ~t["symbol"].astype(str).str.startswith("MT-")).to_numpy(), t["ms"].to_numpy(bool))
        else:
            cl = classes(t)
            cl["X2_list_63"] = ((t["hsapiens_gene_id"].isin(x2_ids) & ~t["symbol"].astype(str).str.startswith("MT-")).to_numpy(), t["un"].to_numpy(bool))
        sys_ = {"liver": "hepmet", "liver4sp": "hepmet", "testis": "repro"}[tissue]
        off = {"hepmet": "repro", "repro": "hepmet"}[sys_]
        outcomes = ["M_any", "H_any", "mouse_lethal", "phylop_promoter", "phylop_utr5", "phylop_utr3", "promoter_cgi", "tata", "log_abc"]
        for name, (flag, pool) in cl.items():
            sizes.append(dict(tissue=tissue, cls=name, n=int(flag.sum()), n_pool=int(pool.sum())))
            for g, s in zip(t.loc[flag, "hsapiens_gene_id"], t.loc[flag, "symbol"]):
                member.append(dict(tissue=tissue, cls=name, hsapiens_gene_id=g, symbol=s))
            for y in outcomes + [f"y_{sys_}", f"y_{off}"]:
                f, p = flag, pool
                if y.startswith("y_"):
                    u = t[f"U_{y[2:]}"].to_numpy(bool)
                    f, p = flag & u, pool & u
                rows.append(dict(tissue=tissue, cls=name, outcome=y) | CM.matched(t, f, p, y, match, key=f"Y2|{tissue}|{name}"))
    r = pd.DataFrame(rows)
    r["q"] = np.nan
    for (ti, cn), g in r.groupby(["tissue", "cls"]):
        p = g["p"].dropna()
        if len(p):
            r.loc[p.index, "q"] = S.bh(p.tolist())
    out["discordant"] = r
    out["discordant_sizes"] = pd.DataFrame(sizes)
    out["discordant_members"] = pd.DataFrame(member)


SECTIONS = {"pheno": pheno, "interaction": interaction, "matrix": matrix, "viability": viability, "regulatory": regulatory, "discordant": discordant}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanics", action="store_true")
    ap.add_argument("--sections", nargs="*", default=list(SECTIONS))
    a = ap.parse_args()
    dest = OUT / "mechanics" if a.mechanics else OUT
    dest.mkdir(parents=True, exist_ok=True)
    tabs = load(a.mechanics)
    for s in a.sections:
        out: dict = {}
        SECTIONS[s](tabs, out)
        for k, v in out.items():
            v.to_csv(dest / f"{k}.csv", index=False)
            print(s, k, v.shape, flush=True)


if __name__ == "__main__":
    main()
