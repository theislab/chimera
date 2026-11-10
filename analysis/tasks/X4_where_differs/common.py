"""X4 (EXPLORATORY): shared loaders and statistics. rho_p as in T4.3 (``stats.py`` reused)."""

from __future__ import annotations

import os
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.common import gene_ids
from analysis.tasks.T4_3_scoreboard import stats as S
from analysis.tasks.X4_where_differs import rules as RU

REPO = gene_ids.REPO_ROOT
RES = REPO / "results" / "analysis"
OUT = RES / "X4"
GDIR = OUT / "genes"
CACHE = OUT / "cache"
N_JOBS = 18


def tissue(c: str) -> str:
    return "testis" if c.startswith("test") else "liver"


def load(c: str) -> pd.DataFrame:
    return pd.read_parquet(GDIR / f"{c}.parquet")


def useed(*parts) -> int:
    return (zlib.crc32("|".join(map(str, parts)).encode()) ^ RU.SEED) & 0x7FFFFFFF


def residual_rank(g: pd.DataFrame, col: str, covs: list[str]) -> pd.Series:
    """Rank residual of col on spline(covs), standardised; NaN where anything is missing."""
    ok = g[col].notna() & g[covs].notna().all(1)
    out = pd.Series(np.nan, index=g.index)
    if ok.sum() < 50:
        return out
    T = g.loc[ok]
    e = S.resid(S.midrank(T[col].to_numpy(float)), S.design(T, covs))
    out[ok] = e / e.std()
    return out


def tertile(x: pd.Series) -> pd.Series:
    """1, 2, 3 by global tertile over defined values; NaN elsewhere."""
    r = x.rank(method="first")
    n = x.notna().sum()
    return np.ceil(3 * r / n).clip(1, 3)


def rho_unit(g: pd.DataFrame, gold: str, covs: list[str], scores: list[str], rows=None, label=None,
             extra_spline=None, extra_linear=None, draws=RU.DRAWS, ref=None) -> list[dict]:
    """rho_p of each score with gold on rows, block bootstrap CI; ``ref`` adds score - ref deltas."""
    extra_spline = list(extra_spline or [])
    extra_linear = list(extra_linear or [])
    cols = [gold] + covs + extra_spline + extra_linear + scores
    m = g[cols].replace([np.inf, -np.inf], np.nan).notna().all(1).to_numpy()
    if rows is not None:
        m &= np.asarray(rows, bool)
    base = dict(label or {})
    base.update(gold=gold, cov="+".join(covs + extra_spline + extra_linear), n=int(m.sum()))
    if m.sum() < 30:
        return [dict(base, score=s, rho_p=np.nan, note="fewer than 30 genes") for s in scores]
    T = g.loc[m]
    R = np.column_stack([S.midrank(T[gold].to_numpy(float))] + [S.midrank(T[s].to_numpy(float)) for s in scores])
    Z = S.design(T, covs + extra_spline, extra=T[extra_linear].to_numpy(float) if extra_linear else None)
    est = S.rho_p_matrix(R, Z)[1:]
    rng = np.random.default_rng(useed(gold, *scores, *covs, *extra_spline, *extra_linear, str(label), int(m.sum())))
    B = S.bootstrap(R, Z, draws, rng, T["block"].to_numpy())[:, 1:]
    out = []
    for k, s in enumerate(scores):
        lo, hi = S.pct_ci(B[:, k])
        r = dict(base, score=s, rho_p=float(est[k]), ci_lo=lo, ci_hi=hi)
        if ref is not None and ref in scores and s != ref:
            j = scores.index(ref)
            dl, dh = S.pct_ci(B[:, k] - B[:, j])
            r.update(ref=ref, delta=float(est[k] - est[j]), delta_lo=dl, delta_hi=dh)
        out.append(r)
    return out


def run_units(units: list[dict]) -> pd.DataFrame:
    """units: dicts of kwargs for rho_unit with key g naming a frame in the module-level GENES."""
    res = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(_call)(u) for u in units)
    return pd.DataFrame([r for x in res for r in x])


def _call(u):
    u = dict(u)
    g = u.pop("g")
    gold = u.pop("gold")
    covs = u.pop("covs")
    scores = u.pop("scores")
    return rho_unit(g, gold, covs, scores, **u)


def matched(g: pd.DataFrame, flag, pool, value: str, covs: list[str], key: str = "", nn: int = RU.NN) -> dict:
    """Matched excess of ``value`` in ``flag`` genes against nearest ``pool`` genes (rules.py)."""
    flag, pool = np.asarray(flag, bool), np.asarray(pool, bool)
    v = g[value].to_numpy(float)
    Zr = g[covs].to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(Zr).all(1)
    uni = ok & (flag | pool)
    fi, pi = np.where(ok & flag)[0], np.where(ok & pool & ~flag)[0]
    r = dict(n_set=len(fi), n_pool=len(pi))
    if len(fi) < 5 or len(pi) < nn:
        return dict(r, excess_d=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan, mean_set=np.nan, mean_null=np.nan)
    idx = np.where(uni)[0]
    Zu = np.column_stack([rankdata(Zr[idx, j]) / len(idx) for j in range(Zr.shape[1])])
    pos = pd.Series(np.arange(len(idx)), index=idx)
    Zf, Zp = Zu[pos[fi].to_numpy()], Zu[pos[pi].to_numpy()]
    _, nbr = NearestNeighbors(n_neighbors=nn).fit(Zp).kneighbors(Zf)
    nv = v[pi][nbr]
    d = v[fi] - nv.mean(1)
    sd = float(np.std(v[idx], ddof=1))
    rng = np.random.default_rng(useed("matched", key, value, len(fi), len(pi)))
    # block CI of the mean difference
    blocks = g["block"].to_numpy()[fi]
    ub, inv = np.unique(blocks, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(ub))
    cnts = np.bincount(inv, minlength=len(ub)).astype(float)
    pick = rng.integers(0, len(ub), (RU.DRAWS, len(ub)))
    boot = sums[pick].sum(1) / np.maximum(cnts[pick].sum(1), 1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    # null by neighbour draws, as X1-X3
    ch = rng.integers(0, nn, (RU.NULL_DRAWS, len(fi)))
    draws = nv[np.arange(len(fi))[None, :], ch].mean(1)
    obs, c = v[fi].mean(), draws.mean()
    p = (1 + np.sum(np.abs(draws - c) >= abs(obs - c))) / (RU.NULL_DRAWS + 1)
    return dict(r, mean_set=float(obs), mean_null=float(c), excess_d=float((obs - c) / sd), ci_lo=float(lo / sd),
                ci_hi=float(hi / sd), p=float(p))


def non_esm(g: pd.DataFrame, col: str, covs: list[str], seed: int = 0) -> pd.Series:
    """Residual rank of col on spline(covs) minus its 5-fold ridge prediction from the 64 ESM PCs."""
    e = residual_rank(g, col, covs)
    ok = e.notna().to_numpy()
    X = g.loc[ok, [f"esm{i}" for i in range(64)]].to_numpy(float)
    y = e[ok].to_numpy()
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
        pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
    out = pd.Series(np.nan, index=g.index)
    out[ok] = y - pred
    return out
