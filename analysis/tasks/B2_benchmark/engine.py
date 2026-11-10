"""B2 scoring engine: rho_p, joint block bootstrap and head-to-head of the plan's section 3 and 5, reusing the T4.3 statistics.

A task holds groups. A group is (rows, columns, design): ranks are taken once within the rows and the design is built once there; each draw reweights
blocks over all genes of the task, so every group of a task shares its draws and differences across groups have CIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.common import controls as K
from analysis.tasks.B2_benchmark import rules as RU
from analysis.tasks.T4_3_scoreboard import stats as ST


def design_for(cov: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Spline df 4 of each covariate rank from SPLINE_MIN genes up, intercept + linear ranks below (Y4). A covariate with one value is dropped."""
    cols = [c for c in columns if cov[c].nunique() > 1]
    if len(cov) >= RU.SPLINE_MIN:
        return K.design(cov, columns=cols)
    parts = [np.ones((len(cov), 1))]
    for c in cols:
        r = pd.Series(cov[c].to_numpy(dtype=float)).rank().to_numpy()
        parts.append(((r - r.mean()) / (r.std() + 1e-12))[:, None])
    return np.hstack(parts)


class Task:
    """All statistics of one task on one set of draws."""

    def __init__(self, table: pd.DataFrame, n_draws: int, seed: int):
        self.t = table
        self.block = table["block"].to_numpy()
        self.n_draws = n_draws
        self.seed = seed
        self.groups = []
        self.ub, self.inv = np.unique(self.block, return_inverse=True)

    def add(self, name, rows: np.ndarray, target: np.ndarray, scores: dict[str, np.ndarray], cov_cols: list[str]):
        """target and scores are full-length arrays; rows a boolean mask. Column 0 of R is the target."""
        idx = np.flatnonzero(rows)
        names = list(scores)
        R = np.column_stack([ST.midrank(target[idx])] + [ST.midrank(scores[k][idx]) for k in names])
        Z = design_for(self.t.iloc[idx], cov_cols)
        self.groups.append(dict(name=name, idx=idx, R=R, Z=Z, names=names, est=ST.rho_p_matrix(R, Z)[1:]))

    def run(self) -> None:
        rng = np.random.default_rng(self.seed)
        for g in self.groups:
            g["draws"] = np.empty((self.n_draws, len(g["names"])))
        for b in range(self.n_draws):
            cnt = np.bincount(rng.integers(0, len(self.ub), len(self.ub)), minlength=len(self.ub))
            w = cnt[self.inv].astype(float)
            for g in self.groups:
                wg = w[g["idx"]]
                g["draws"][b] = ST.weighted_rho(g["R"], g["Z"], wg)[1:] if (wg > 0).sum() > g["Z"].shape[1] + 2 else np.nan

    def rows(self) -> list[dict]:
        out = []
        for g in self.groups:
            for j, k in enumerate(g["names"]):
                d = g["draws"][:, j]
                lo, hi = ST.pct_ci(d) if np.isfinite(d).sum() > 10 else (np.nan, np.nan)
                out.append(dict(group=g["name"], method=k, value=float(g["est"][j]), ci_lo=lo, ci_hi=hi, n=len(g["idx"]), n_blocks=len(np.unique(self.block[g["idx"]]))))
        return out

    def get(self, group: str, method: str) -> tuple[float, np.ndarray]:
        g = next(x for x in self.groups if x["name"] == group)
        j = g["names"].index(method)
        return float(g["est"][j]), g["draws"][:, j]

    def h2h(self, focal: list[str], comparators: list[str], group: str) -> list[dict]:
        """Delta = focal - comparator inside a group: percentile CI, and simultaneous CI by max-|t| over comparators of one focal."""
        g = next(x for x in self.groups if x["name"] == group)
        out = []
        for fo in focal:
            if fo not in g["names"]:
                continue
            comps = [c for c in comparators if c in g["names"] and c != fo]
            if not comps:
                continue
            est = np.array([g["est"][g["names"].index(fo)] - g["est"][g["names"].index(c)] for c in comps])
            D = np.column_stack([g["draws"][:, g["names"].index(fo)] - g["draws"][:, g["names"].index(c)] for c in comps])
            ok = np.isfinite(D).all(1)
            if ok.sum() < 10:
                continue
            sl, sh, _ = ST.simultaneous_ci(est, D[ok], 0.95)
            for i, c in enumerate(comps):
                lo, hi = ST.pct_ci(D[ok, i])
                out.append(dict(group=group, focal=fo, comparator=c, delta=float(est[i]), ci_lo=lo, ci_hi=hi, sim_lo=float(sl[i]), sim_hi=float(sh[i]), n=len(g["idx"])))
        return out


def side_h2h(task: Task, group: str, method: str, focal: list[str]) -> list[dict]:
    """A method on its own genes: its group holds it and the focal readouts on the same genes."""
    out = []
    for fo in focal:
        e1, d1 = task.get(group, fo)
        e2, d2 = task.get(group, method)
        D = d1 - d2
        lo, hi = ST.pct_ci(D[np.isfinite(D)])
        out.append(dict(group=group, focal=fo, comparator=method, delta=e1 - e2, ci_lo=lo, ci_hi=hi, sim_lo=np.nan, sim_hi=np.nan,
                        n=len(next(x for x in task.groups if x["name"] == group)["idx"])))
    return out
