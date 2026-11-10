"""X4 (EXPLORATORY): the T2.5 margin with no neural model (rules.py, Direct margin).

For each cell of (species s, type k): mu1 = rbar_gk L_c, rbar_gk = mean over species of the species' rate in
k (a type present in one species uses its own rate); mu0 = p_gs L_c, p_gs = rate of g in s. NB log likelihood
ratio with gene-wise moment dispersion; S_gsk = mean over cells; w_sk = n_sk / (n_s K); Cdir_gk = sum_s w_sk
S_gsk; Sdir_g = sum_k Cdir_gk = sum_s Cspdir_gs. Laplace smoothing as ``precompute_p_null`` (0.5 count). Cell labels are used.

Pattern and level parts (added after the mechanism notes, to test them). With mu_p = rbar_gk c_gs L_c, where c_gs rescales the
shared profile to species s's own total for g, Sdir = LLR(mu_p vs mu0) + LLR(mu1 vs mu_p): the first term is the gain from the
shared profile at the species' own level (pattern), the second the cost of forcing one level on both species (level). Exact.

Writes results/analysis/X4/cache/Sdir__<cohort>.parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import cohorts as C
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.X4_where_differs import common as CM
from analysis.tasks.X4_where_differs import rules as RU


def nb_ll_diff(x, mu1, mu0, th):
    """log NB(x | mu1) - log NB(x | mu0), the gamma terms cancel."""
    return (x * (np.log(mu1) - np.log(mu0)) - (x + th) * (np.log(th + mu1) - np.log(th + mu0)))


def run(c: str) -> pd.DataFrame:
    a = C.read_adata(c)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    lab = a.obs["cell_type"].astype(str)
    if c.startswith("test"):
        lab = lab.map(T42.TESTIS_COARSE)
    lab = lab.to_numpy()
    L = np.asarray(X.sum(1)).ravel()
    G = X.shape[1]
    species = sorted(set(spc))
    K = len(species)
    groups = sorted({(s, t) for s, t in zip(spc, lab)})
    rate, n_sk, Lsum = {}, {}, {}
    for s, t in groups:
        m = (spc == s) & (lab == t)
        n_sk[(s, t)] = int(m.sum())
        Lsum[(s, t)] = L[m].sum()
        rate[(s, t)] = (np.asarray(X[m].sum(0)).ravel() + 0.5) / (L[m].sum() + 0.5 * G)
    p = {s: (np.asarray(X[spc == s].sum(0)).ravel() + 0.5) / (L[spc == s].sum() + 0.5 * G) for s in species}
    types = sorted({t for _, t in groups})
    rbar = {}
    for t in types:
        have = [s for s in species if (s, t) in rate]
        rbar[t] = np.mean([rate[(s, t)] for s in have], 0)
    # gene-wise moment dispersion around the (s, k) rate
    num, den = np.zeros(G), np.zeros(G)
    for s, t in groups:
        m = np.where((spc == s) & (lab == t))[0]
        for i in range(0, len(m), 2000):
            mm = m[i:i + 2000]
            mu = np.outer(L[mm], rate[(s, t)])
            xx = np.asarray(X[mm].todense())
            num += (((xx - mu) ** 2) - mu).sum(0)
            den += (mu ** 2).sum(0)
    phi = np.clip(num / np.maximum(den, 1e-12), 1e-3, 1e3)
    th = 1.0 / phi
    Cdir = {t: np.zeros(G) for t in types}
    Csp = {sp_: np.zeros(G) for sp_ in species}
    Cpat = {t: np.zeros(G) for t in types}
    Lspecies = {sp_: sum(Lsum[g_] for g_ in groups if g_[0] == sp_) for sp_ in species}
    cfac = {sp_: p[sp_] / (sum(Lsum[(s_, t_)] * rbar[t_] for (s_, t_) in groups if s_ == sp_) / Lspecies[sp_]) for sp_ in species}
    for s, t in groups:
        m = np.where((spc == s) & (lab == t))[0]
        tot = np.zeros(G)
        pat = np.zeros(G)
        for i in range(0, len(m), 2000):
            mm = m[i:i + 2000]
            xx = np.asarray(X[mm].todense())
            mu1 = np.outer(L[mm], rbar[t])
            mu0 = np.outer(L[mm], p[s])
            mup = np.outer(L[mm], rbar[t] * cfac[s])
            tot += nb_ll_diff(xx, mu1, mu0, th).sum(0)
            pat += nb_ll_diff(xx, mup, mu0, th).sum(0)
        w = n_sk[(s, t)] / (np.sum(spc == s) * K)
        Cdir[t] += w * tot / n_sk[(s, t)]
        Csp[s] += w * tot / n_sk[(s, t)]
        Cpat[t] += w * pat / n_sk[(s, t)]
    df = pd.DataFrame({f"Cdir_{t}": v for t, v in Cdir.items()}, index=pd.Index(a.var_names, name="var_name"))
    df["Sdir"] = df.sum(1)
    for s, v in Csp.items():
        df[f"Cspdir_{s}"] = v
    for t, v in Cpat.items():
        df[f"Cpat_{t}"] = v
        df[f"Clev_{t}"] = Cdir[t] - v
    df["Sdir_pattern"] = df[[f"Cpat_{t}" for t in types]].sum(1)
    df["Sdir_level"] = df["Sdir"] - df["Sdir_pattern"]
    return df.reset_index()


def main():
    CM.CACHE.mkdir(parents=True, exist_ok=True)
    for c in RU.COHORTS:
        run(c).to_parquet(CM.CACHE / f"Sdir__{c}.parquet")
        print("done", c, flush=True)


if __name__ == "__main__":
    main()
