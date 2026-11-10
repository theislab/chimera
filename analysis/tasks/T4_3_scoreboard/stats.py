"""Statistics of the validation scoreboard (pre-specified analysis plan, sections 3, 5, 6). Pure functions, no file access.

Conventions
- Ranks are mid-ranks taken once within the universe U; the spline design is built once in U
  (``controls.design``: intercept + natural cubic spline, df 4, knots at rank quartiles, of the
  rank of each covariate). Bootstrap draws keep these ranks and this basis and refit the OLS
  coefficients with the draw's multiplicity weights ("refit residuals in each draw").
- rho_p = Pearson of the OLS residuals of rank(score) and rank(G) on the same design.
- Permutation null: residuals of D permuted within strata (pi_raw decile x expression decile,
  strata under 20 genes merged within their pi decile); T recomputed as the Pearson of the
  permuted D residuals with the G residuals; p = (1 + #{T_b >= T}) / (B + 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from analysis.common import controls as K


# ---------------------------------------------------------------- design, residuals
def midrank(x) -> np.ndarray:
    return rankdata(np.asarray(x, dtype=float))


def design(cov: pd.DataFrame, columns: list[str], extra: np.ndarray | None = None) -> np.ndarray:
    """Spline design of ``columns`` (ranks within these rows); ``extra`` linear columns appended."""
    Z = K.design(cov, columns=columns)
    if extra is not None:
        E = np.asarray(extra, dtype=float)
        E = (E - E.mean(0)) / (E.std(0) + 1e-12)
        Z = np.hstack([Z, E])
    return Z


def resid(Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    beta = np.linalg.lstsq(Z, Y, rcond=None)[0]
    return Y - Z @ beta


def corr_first(E: np.ndarray) -> np.ndarray:
    """Pearson of column 0 with every column of E (columns already mean-zero residuals)."""
    num = E[:, 0] @ E
    den = np.sqrt((E ** 2).sum(0) * (E[:, 0] ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def rho_p_matrix(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """R: ranks, column 0 = G. Returns rho_p of every column with G (entry 0 is 1)."""
    return corr_first(resid(R, Z))


def weighted_rho(R: np.ndarray, Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """rho_p of every column with column 0 under multiplicity weights w (one bootstrap draw)."""
    m = w > 0
    Rm, Zm, wm = R[m], Z[m], w[m]
    ZtW = Zm.T * wm
    beta = np.linalg.lstsq(ZtW @ Zm, ZtW @ Rm, rcond=None)[0]
    E = Rm - Zm @ beta
    Ew = E * wm[:, None]
    num = Ew[:, 0] @ E
    den = np.sqrt((Ew * E).sum(0) * (Ew[:, 0] * E[:, 0]).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


# ---------------------------------------------------------------- strata
def deciles(x: np.ndarray) -> np.ndarray:
    r = midrank(x)
    return pd.qcut(r, 10, labels=False, duplicates="drop").astype(int)


def strata(pi_raw: np.ndarray, expr: np.ndarray, min_size: int = 20) -> np.ndarray:
    """pi_raw decile x expression decile; within each pi decile, expression cells are merged
    in expression order until each group has >= min_size genes (a short last group joins the
    previous one)."""
    pd_, ed = deciles(pi_raw), deciles(expr)
    lab = np.full(len(pi_raw), -1)
    nxt = 0
    for p in np.unique(pd_):
        groups, cur = [], []
        for e in np.unique(ed[pd_ == p]):
            cur.append(e)
            if ((pd_ == p) & np.isin(ed, cur)).sum() >= min_size:
                groups.append(cur)
                cur = []
        if cur:
            if groups:
                groups[-1] = groups[-1] + cur
            else:
                groups.append(cur)
        for g in groups:
            lab[(pd_ == p) & np.isin(ed, g)] = nxt
            nxt += 1
    assert (lab >= 0).all()
    return lab


def cells(pi_raw: np.ndarray, expr: np.ndarray | None) -> np.ndarray:
    """Unmerged decile cells (sets; plan, section 4); expression only when pi_raw is None."""
    if pi_raw is None:
        return deciles(expr)
    return deciles(pi_raw) * 10 + deciles(expr)


# ---------------------------------------------------------------- permutation test
def perm_null(eD: np.ndarray, eG: np.ndarray, lab: np.ndarray, n_perm: int, rng: np.random.Generator,
              chunk: int = 250) -> np.ndarray:
    """T_b = Pearson(permuted eD, eG), eD permuted within strata."""
    eD = np.asarray(eD, float)
    eG = np.asarray(eG, float)
    eDc = eD - eD.mean()
    eGc = eG - eG.mean()
    den = np.sqrt((eDc ** 2).sum() * (eGc ** 2).sum())
    groups = [np.flatnonzero(lab == s) for s in np.unique(lab)]
    out = np.empty(n_perm)
    n = len(eD)
    done = 0
    while done < n_perm:
        c = min(chunk, n_perm - done)
        P = np.empty((c, n), dtype=np.int64)
        for idx in groups:
            P[:, idx] = idx[np.argsort(rng.random((c, len(idx))), axis=1)]
        out[done:done + c] = (eDc[P] @ eGc) / den
        done += c
    return out


def p_one_sided(T: float, null: np.ndarray, lower: bool = False) -> float:
    cnt = (null <= T).sum() if lower else (null >= T).sum()
    return float((1 + cnt) / (len(null) + 1))


def p_two_sided(T: float, null: np.ndarray) -> float:
    return float((1 + (np.abs(null) >= abs(T)).sum()) / (len(null) + 1))


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj.tolist()


def bh(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)[::-1]
    adj = np.empty(m)
    run = 1.0
    for k, j in enumerate(o):
        rank = m - k
        run = min(run, p[j] * m / rank)
        adj[j] = run
    return adj.tolist()


# ---------------------------------------------------------------- bootstrap
def block_weights(block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiplicity weights of one block-bootstrap draw (blocks resampled with replacement)."""
    ub, inv = np.unique(block, return_inverse=True)
    pick = rng.integers(0, len(ub), len(ub))
    cnt = np.bincount(pick, minlength=len(ub))
    return cnt[inv].astype(float)


def gene_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)


def bootstrap(R: np.ndarray, Z: np.ndarray, n_draws: int, rng: np.random.Generator,
              block: np.ndarray | None) -> np.ndarray:
    """Draws x columns of rho_p (column 0, G with itself, dropped by the caller)."""
    out = np.empty((n_draws, R.shape[1]))
    n = R.shape[0]
    for b in range(n_draws):
        w = block_weights(block, rng) if block is not None else gene_weights(n, rng)
        out[b] = weighted_rho(R, Z, w)
    return out


def pct_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def simultaneous_ci(est: np.ndarray, draws: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray, float]:
    """max-|t| simultaneous intervals: est +- c * se, c the level quantile of max_j |t_j|."""
    se = draws.std(0, ddof=1)
    t = np.abs((draws - draws.mean(0)) / se)
    c = float(np.quantile(t.max(1), level))
    return est - c * se, est + c * se, c


# ---------------------------------------------------------------- reliability
def split_rel(halves_a: np.ndarray, halves_b: np.ndarray, Z: np.ndarray, rows: np.ndarray | None = None) -> dict:
    """rel = 2r/(1+r), r = mean over splits of the Pearson of the residualized ranks of the two
    halves. Each split uses the genes where both halves are defined; ranks and the design are
    taken within those genes (rows of Z with the covariates of U)."""
    rs, ns = [], []
    for a, b in zip(halves_a, halves_b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 50:
            continue
        Zs = Z[ok]
        ea = resid(midrank(a[ok]), Zs)
        eb = resid(midrank(b[ok]), Zs)
        rs.append(float(np.corrcoef(ea, eb)[0, 1]))
        ns.append(int(ok.sum()))
    if not rs:
        return dict(r=np.nan, rel=np.nan, n_splits=0, n_median=0)
    r = float(np.mean(rs))
    return dict(r=r, rel=2 * r / (1 + r), n_splits=len(rs), n_median=int(np.median(ns)))


# ---------------------------------------------------------------- sign statistic (3a)
def sign_concordance(d: np.ndarray, g: np.ndarray, lab: np.ndarray) -> float:
    sd, sg = np.zeros(len(d)), np.zeros(len(g))
    for s in np.unique(lab):
        m = lab == s
        sd[m] = np.sign(d[m] - np.median(d[m]))
        sg[m] = np.sign(g[m] - np.median(g[m]))
    ok = (sd != 0) & (sg != 0)
    return float((sd[ok] == sg[ok]).mean())


# ---------------------------------------------------------------- stratified AUROC / AUPRC
def kept_strata(inset: np.ndarray, lab: np.ndarray, ratio: int = 5, absolute: bool = False) -> np.ndarray:
    """Strata kept for a set test (plan, section 4): at least 5 background genes per set gene
    (n_bg >= 5 n_set). ``absolute`` (Family 3 sensitivity): at least 5 background genes."""
    keep = np.zeros(len(lab), bool)
    for s in np.unique(lab):
        m = lab == s
        n1 = int(inset[m].sum())
        n0 = int(m.sum()) - n1
        if n1 >= 1 and n0 >= (ratio if absolute else ratio * n1):
            keep |= m
    return keep


def strat_auroc(score: np.ndarray, inset: np.ndarray, lab: np.ndarray) -> float:
    """Mean over set genes of the fraction of same-stratum background genes they exceed (ties 1/2)."""
    tot, n1 = 0.0, 0
    for s in np.unique(lab):
        m = lab == s
        a, b = score[m & inset], np.sort(score[m & ~inset])
        if len(a) == 0 or len(b) == 0:
            continue
        lt = np.searchsorted(b, a, "left")
        le = np.searchsorted(b, a, "right")
        tot += ((lt + le) / 2 / len(b)).sum()
        n1 += len(a)
    return tot / n1 if n1 else np.nan


def strat_auprc(score: np.ndarray, inset: np.ndarray, lab: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    pct = np.empty(len(score))
    for s in np.unique(lab):
        m = np.flatnonzero(lab == s)
        pct[m] = midrank(score[m]) / len(m)
    return float(average_precision_score(inset.astype(int), pct))


def auroc_perm(score: np.ndarray, inset: np.ndarray, lab: np.ndarray, n_perm: int,
               rng: np.random.Generator) -> np.ndarray:
    """Null of the stratified AUROC: set labels permuted within strata."""
    out = np.empty(n_perm)
    strata_ = [np.flatnonzero(lab == s) for s in np.unique(lab)]
    info = []
    for idx in strata_:
        n1 = int(inset[idx].sum())
        if n1 == 0 or n1 == len(idx):
            continue
        info.append((midrank(score[idx]), n1, len(idx) - n1))
    N1 = sum(i[1] for i in info)
    for b0 in range(0, n_perm, 500):
        c = min(500, n_perm - b0)
        acc = np.zeros(c)
        for r, n1, n0 in info:
            pick = np.argsort(rng.random((c, len(r))), axis=1)[:, :n1]
            Rsum = r[pick].sum(1)
            U = Rsum - n1 * (n1 + 1) / 2
            acc += U / n0
        out[b0:b0 + c] = acc / N1
    return out


def auroc_boot(score: np.ndarray, inset: np.ndarray, lab: np.ndarray, n_boot: int,
               rng: np.random.Generator, auprc: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Stratified bootstrap: set and background genes resampled within each kept stratum."""
    idx = {s: (np.flatnonzero((lab == s) & inset), np.flatnonzero((lab == s) & ~inset)) for s in np.unique(lab)}
    A, P = np.empty(n_boot), np.full(n_boot, np.nan)
    for b in range(n_boot):
        sel = []
        for s, (i1, i0) in idx.items():
            if len(i1):
                sel.append(i1[rng.integers(0, len(i1), len(i1))])
            if len(i0):
                sel.append(i0[rng.integers(0, len(i0), len(i0))])
        sel = np.concatenate(sel)
        A[b] = strat_auroc(score[sel], inset[sel], lab[sel])
        if auprc:
            P[b] = strat_auprc(score[sel], inset[sel], lab[sel])
    return A, P
