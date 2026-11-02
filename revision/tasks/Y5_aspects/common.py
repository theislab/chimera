"""Y5 shared statistics. Rank partial correlation as T4.3/X4; block bootstrap over gene modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from revision.common import gene_ids
from revision.tasks.T4_3_scoreboard import stats as ST
from revision.tasks.Y5_aspects import rules as RU

REPO = gene_ids.REPO_ROOT
OUT = REPO / "results" / "revision" / "Y5"
GEN = OUT / "genes"
C4 = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]
C3 = ["prior_raw", "log_cp10k_spmean", "detection_spmean"]
CPAT = C4 + ["tau_shared"]


def genes(c: str) -> pd.DataFrame:
    return pd.read_parquet(GEN / f"{c}.parquet")


def scan(T: pd.DataFrame, x: str, ys: list[str], covs: list[str], draws: int = RU.DRAWS_SCAN, seed: int = 0,
         extra: np.ndarray | None = None, delta_to: str | None = None) -> pd.DataFrame:
    """rho_p of x with each y on rows T (already the universe). CI, p from block bootstrap; optional delta to a reference y."""
    R = np.column_stack([ST.midrank(T[x].to_numpy(float))] + [ST.midrank(T[y].to_numpy(float)) for y in ys])
    Z = ST.design(T, covs, extra=extra)
    est = ST.rho_p_matrix(R, Z)[1:]
    rng = np.random.default_rng(seed)
    B = ST.bootstrap(R, Z, draws, rng, T["block"].to_numpy())[:, 1:]
    sd = np.nanstd(B, axis=0)
    out = []
    for k, y in enumerate(ys):
        lo, hi = ST.pct_ci(B[:, k])
        p = float(2 * norm.sf(abs(est[k]) / sd[k])) if sd[k] > 0 else 1.0
        r = dict(x=x, y=y, rho_p=float(est[k]), ci_lo=lo, ci_hi=hi, p=p, n=len(T), covs="+".join(covs))
        if delta_to is not None and delta_to in ys:
            j = ys.index(delta_to)
            dB = np.abs(B[:, k]) - np.abs(B[:, j])
            r[f"absdelta_vs_{delta_to}"] = float(abs(est[k]) - abs(est[j]))
            r["absdelta_lo"], r["absdelta_hi"] = ST.pct_ci(dB)
        out.append(r)
    return pd.DataFrame(out)


def one_sided(est: float, sd_p: float) -> float:
    return sd_p / 2


def bh_table(t: pd.DataFrame, p: str = "p") -> pd.Series:
    return pd.Series(ST.bh(t[p].fillna(1).tolist()), index=t.index)


def useed(*parts) -> int:
    import zlib
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF
