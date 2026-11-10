"""EXPLORATORY: compute T1-T4 for every unit on one split (definitions in rules.py).

`python -m analysis.tasks.Y4_regions.screen coverage`  coverage only, no gold value compared
`python -m analysis.tasks.Y4_regions.screen discovery` odd autosomes, all regions and decoys
Even genes are read by replicate.py only, after the carried list is committed.
"""

from __future__ import annotations

import os
import sys
import zlib

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.tasks.T4_3_scoreboard import stats as ST
from analysis.tasks.Y4_regions import regions as RG
from analysis.tasks.Y4_regions import rules as RU

REPO = gene_ids.REPO_ROOT
OUT = REPO / "results/analysis/Y4"
GDIR = OUT / "genes"
N_JOBS = int(os.environ.get("Y4_JOBS", 18))
_CACHE: dict = {}


def tissue(c: str) -> str:
    return "testis" if c.startswith("test") else "liver"


def table(c: str) -> pd.DataFrame:
    if c not in _CACHE:
        t = pd.read_parquet(GDIR / f"{c}.parquet")
        _CACHE[c] = (t, RG.regions(t, c))
    return _CACHE[c][0]


def regs(c: str) -> list[dict]:
    table(c)
    return _CACHE[c][1]


def useed(*p) -> int:
    return (zlib.crc32("|".join(map(str, p)).encode()) ^ RU.SEED) & 0x7FFFFFFF


def linear_design(T: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = [np.ones((len(T), 1))]
    for c in cols:
        r = ST.midrank(T[c].to_numpy(float))
        parts.append(((r - r.mean()) / (r.std() + 1e-12))[:, None])
    return np.hstack(parts)


def zdesign(T: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return ST.design(T, cols) if len(T) >= RU.SPLINE_MIN else linear_design(T, cols)


def ranks(T: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([ST.midrank(T[c].to_numpy(float)) for c in cols])


class Part:
    """One gene subset for a bundle: ranks (column 0 the gold), design, position of its genes in the block index."""

    def __init__(self, T, cols, Z, bidx):
        self.R, self.Z, self.b = ranks(T, cols), Z, bidx
        self.cols, self.n = cols, len(T)
        self.est = ST.rho_p_matrix(self.R, self.Z)

    def draw(self, wb):
        w = wb[self.b].astype(float)
        if (w > 0).sum() < self.Z.shape[1] + 2:  # a draw that left too few genes in a small part
            return np.full(self.R.shape[1], np.nan)
        try:
            return ST.weighted_rho(self.R, self.Z, w)
        except np.linalg.LinAlgError:
            return np.full(self.R.shape[1], np.nan)


def j_of(t: pd.DataFrame) -> list[str]:
    """Baselines defined for at least half of the cohort's genes."""
    return [b for b in RU.BASELINES if t[b].notna().mean() >= 0.5]


def bundle(c: str, gold: str, covs: list[str], name: str, mask: np.ndarray, split: str, draws: int) -> list[dict]:
    t = table(c)
    base = t[[gold] + covs].notna().all(1).to_numpy() & t["split"].eq(split).to_numpy()
    for x in RU.READOUTS:
        base &= t[x].notna().to_numpy()
    own, rest = base & mask, base & ~mask
    if own.sum() < RU.MIN_GENES:
        return []
    blocks, inv = np.unique(t.loc[base, "block"].to_numpy(), return_inverse=True)
    binv = pd.Series(inv, index=np.flatnonzero(base))
    bidx = lambda m: binv.loc[np.flatnonzero(m)].to_numpy()
    To = t.loc[own]
    P_own = Part(To, [gold] + RU.READOUTS, zdesign(To, covs), bidx(own))
    P_rest = None
    if rest.sum() >= RU.MIN_GENES:
        Tr = t.loc[rest]
        P_rest = Part(Tr, [gold] + RU.READOUTS, zdesign(Tr, covs), bidx(rest))
    Jc = j_of(t)
    # a baseline defined for under half of this region's genes is dropped from its J and recorded
    J = [b for b in Jc if To[b].notna().mean() >= 0.5]
    com = own & t[J + RU.BESIDE].notna().all(1).to_numpy() if J else np.zeros(len(t), bool)
    P_com = P_inc = P_inc_sd = None
    if com.sum() >= RU.MIN_GENES:
        Tc = t.loc[com]
        Zc = zdesign(Tc, covs)
        P_com = Part(Tc, [gold] + RU.READOUTS + J + RU.BESIDE, Zc, bidx(com))
        Zi = zdesign(Tc, covs + J)
        P_inc = Part(Tc, [gold] + RU.READOUTS + ["Sdir"], Zi, bidx(com))
        P_inc_sd = Part(Tc, [gold] + RU.READOUTS, zdesign(Tc, covs + J + ["Sdir"]), bidx(com))
    rng = np.random.default_rng(useed(c, gold, name, split))
    nb = len(blocks)
    B_own = np.empty((draws, 3))
    B_rest = np.full((draws, 3), np.nan)
    B_com = np.full((draws, 3 + len(J) + len(RU.BESIDE)), np.nan)
    B_inc = np.full((draws, 4), np.nan)
    B_inc_sd = np.full((draws, 3), np.nan)
    for i in range(draws):
        wb = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        B_own[i] = P_own.draw(wb)
        if P_rest is not None:
            B_rest[i] = P_rest.draw(wb)
        if P_com is not None:
            B_com[i] = P_com.draw(wb)
            B_inc[i] = P_inc.draw(wb)
            B_inc_sd[i] = P_inc_sd.draw(wb)

    def ci_p(est, d):
        d = d[np.isfinite(d)]
        if len(d) < 20:
            return np.nan, np.nan, np.nan, np.nan
        lo, hi = np.quantile(d, [0.025, 0.975])
        se = d.std(ddof=1)
        return float(lo), float(hi), float(se), float(2 * norm.sf(abs(est) / max(se, 1e-9)))

    out = []
    for k, x in enumerate(RU.READOUTS, start=1):
        e = float(P_own.est[k])
        lo, hi, se, p = ci_p(e, B_own[:, k])
        r = dict(cohort=c, gold=gold, region=name, readout=x, split=split, n=int(own.sum()),
                 n_blocks=int(pd.unique(To["block"]).size), rho=e, lo=lo, hi=hi, se=se, p_T1=p,
                 J=",".join(J), J_dropped=",".join(sorted(set(Jc) - set(J))))
        if P_rest is not None:
            er = float(P_rest.est[k])
            lo2, hi2, _, p2 = ci_p(e - er, B_own[:, k] - B_rest[:, k])
            r.update(rho_rest=er, n_rest=int(rest.sum()), T2=e - er, T2_lo=lo2, T2_hi=hi2, p_T2=p2)
        if P_com is not None:
            ec = P_com.est
            r.update(n_common=P_com.n, rho_common=float(ec[k]))
            nj = 3 + len(J)
            bj = ec[3:nj]
            for j, b in enumerate(J):
                r[f"rho_{b}"] = float(bj[j])
            for j, b in enumerate(RU.BESIDE):
                r[f"rho_{b}"] = float(ec[nj + j])
            if len(J):
                dd = B_com[:, k][:, None] - B_com[:, 3:nj]
                for j, b in enumerate(J):
                    lo3, hi3, _, _ = ci_p(float(ec[k] - bj[j]), dd[:, j])
                    r[f"delta_{b}"] = float(ec[k] - bj[j])
                    r[f"delta_{b}_lo"], r[f"delta_{b}_hi"] = lo3, hi3
                t3 = float((ec[k] - bj).min())
                lo3, hi3, _, p3 = ci_p(t3, dd.min(1))
                r.update(T3=t3, T3_lo=lo3, T3_hi=hi3, p_T3=p3, T3_best=J[int(np.argmax(bj))])
            e4 = float(P_inc.est[k])
            lo4, hi4, _, p4 = ci_p(e4, B_inc[:, k])
            r.update(T4=e4, T4_lo=lo4, T4_hi=hi4, p_T4=p4, T4_Sdir=float(P_inc.est[3]))
            e5 = float(P_inc_sd.est[k])
            lo5, hi5, _, _ = ci_p(e5, B_inc_sd[:, k])
            r.update(T4s=e5, T4s_lo=lo5, T4s_hi=hi5)
        out.append(r)
    return out


def tasks_for(c: str) -> list[tuple]:
    t = table(c)
    tk = []
    for gold, (_, covs) in RU.GOLDS[tissue(c)].items():
        for i, reg in enumerate(regs(c)):
            if RG.usable(t, reg, gold, covs):
                tk.append((c, gold, i))
    return tk


def _run(c, gold, i, split, draws):
    reg = regs(c)[i]
    covs = RU.GOLDS[tissue(c)][gold][1]
    res = bundle(c, gold, covs, reg["name"], reg["mask"], split, draws)
    for r in res:
        r.update(axis=reg["axis"], mech=reg["mech"], seen=reg["seen"])
    return res


def decoy_masks(c: str, gold: str) -> list[tuple[str, np.ndarray]]:
    """Random whole ESM modules until the region holds the target number of usable odd genes."""
    t = table(c)
    covs = RU.GOLDS[tissue(c)][gold][1]
    ok = t[[gold] + covs].notna().all(1).to_numpy() & t["split"].eq("odd").to_numpy()
    rng = np.random.default_rng(useed("decoy", c, gold))
    b = t["block"].to_numpy()
    ub = np.unique(b)
    cnt = pd.Series(b[ok]).value_counts()
    out = []
    for size in RU.DECOY_SIZES:
        for k in range(RU.DECOY_N):
            perm = rng.permutation(ub)
            s = np.cumsum(cnt.reindex(perm).fillna(0).to_numpy())
            pick = perm[: int(np.searchsorted(s, size) + 1)]
            out.append((f"decoy{size}_{k}", np.isin(b, pick)))
    return out


def _decoy(c, gold, name, mask, split, draws):
    covs = RU.GOLDS[tissue(c)][gold][1]
    res = bundle(c, gold, covs, name, mask, split, draws)
    for r in res:
        r.update(axis="decoy", mech=False, seen=False)
    return res


def coverage():
    rows = []
    for c in RU.COHORTS:
        t = table(c)
        print(c, len(t), t.split.value_counts().to_dict(), {b: round(t[b].notna().mean(), 2) for b in RU.BASELINES},
              flush=True)
        for gold, (_, covs) in RU.GOLDS[tissue(c)].items():
            ok = t[[gold] + covs].notna().all(1).to_numpy()
            for reg in regs(c):
                m = ok & reg["mask"]
                rows.append(dict(cohort=c, gold=gold, region=reg["name"], axis=reg["axis"], mech=reg["mech"],
                                 seen=reg["seen"], n_odd=int((m & t.split.eq("odd").to_numpy()).sum()),
                                 n_even=int((m & t.split.eq("even").to_numpy()).sum()),
                                 usable=RG.usable(t, reg, gold, covs)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "coverage.csv", index=False)
    print(df.groupby("cohort").usable.agg(["size", "sum"]))
    print(df[df.usable].groupby("axis").region.nunique())
    print("regions tested anywhere", df[df.usable].region.nunique(), "units", int(df.usable.sum()) * 2)


def bh(p: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p.index)
    m = p.notna()
    out[m] = ST.bh(p[m].to_numpy().tolist())
    return out


def discovery():
    tasks = []
    for c in RU.COHORTS:
        tasks += tasks_for(c)
    print("bundles", len(tasks), flush=True)
    res = Parallel(n_jobs=N_JOBS, backend="loky", batch_size=4)(
        delayed(_run)(c, g, i, "odd", RU.BOOT_SCREEN) for c, g, i in tasks)
    df = pd.DataFrame([r for x in res for r in x])
    for k in ("T1", "T2", "T3", "T4"):
        df[f"q_{k}"] = bh(df[f"p_{k}"])
    df.to_csv(OUT / "discovery_units.csv", index=False)
    print("units", len(df), "regions", df.region.nunique(), flush=True)
    dec = []
    for c in RU.COHORTS:
        for gold in RU.GOLDS[tissue(c)]:
            dec += [(c, gold, n, m) for n, m in decoy_masks(c, gold)]
    res = Parallel(n_jobs=N_JOBS, backend="loky", batch_size=4)(
        delayed(_decoy)(c, g, n, m, "odd", RU.BOOT_SCREEN) for c, g, n, m in dec)
    dd = pd.DataFrame([r for x in res for r in x])
    dd.to_csv(OUT / "discovery_decoys.csv", index=False)
    print("decoys", len(dd), flush=True)


if __name__ == "__main__":
    {"coverage": coverage, "discovery": discovery}[sys.argv[1]]()
