"""Y5 shared statistics. Rank partial correlation as T4.3/X4; block bootstrap over gene modules."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.common import gene_ids
from analysis.tasks.T4_3_scoreboard import stats as ST
from analysis.tasks.Y5_aspects import rules as RU

REPO = gene_ids.REPO_ROOT
OUT = REPO / "results" / "analysis" / "Y5"
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
        if not np.isfinite(B[:, k]).any():
            out.append(dict(x=x, y=y, rho_p=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan, n=len(T), covs="+".join(covs)))
            continue
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


def forward_runs(c: str):
    return sorted((OUT / "forward").glob(f"{c}_s*.npz"))


def contribs(c: str) -> pd.DataFrame:
    """Healthy-seed mean of C_gk (coarse types) and Csp_s, from the Y5 forward pass (T2.5 definitions)."""
    from analysis.tasks.T4_2_gene_baselines import run as T42
    acc = {}
    for f in forward_runs(c):
        with np.load(f, allow_pickle=False) as d:
            sp_names = d["species"].astype(str)
            ns = d["n_species"].astype(float)
            S_n = len(sp_names)
            grp = d["cell_type__groups"]
            cs = d["cell_type__sum_meanz"]
            var = d["var_names"].astype(str)
            one = {}
            for i, (s, lab, n) in enumerate(grp):
                lab = str(lab)
                k = T42.TESTIS_COARSE.get(lab, lab) if c.startswith("test") else lab
                v = cs[i] / (ns[int(s)] * S_n)
                one[f"C_{k}"] = one.get(f"C_{k}", 0) + v
                one[f"Csp_{sp_names[int(s)]}"] = one.get(f"Csp_{sp_names[int(s)]}", 0) + cs[i] / ns[int(s)]
            for k, v in one.items():
                acc.setdefault(k, []).append(v)
    return pd.DataFrame({k: np.mean(v, 0) for k, v in acc.items()}, index=pd.Index(var, name="var_name")).reset_index()


def gsk(c: str, key: str = "cell_type", min_cells: int = 30) -> tuple[np.ndarray, list[tuple[str, str]], np.ndarray]:
    """Healthy-seed mean S_gsk (genes x (species, label)) from the forward pass."""
    arrs, names = [], None
    for f in forward_runs(c):
        with np.load(f, allow_pickle=False) as d:
            sp_names = d["species"].astype(str)
            grp = d[f"{key}__groups"]
            cs = d[f"{key}__sum_meanz"]
            keep = [i for i, (s, l, n) in enumerate(grp) if n >= min_cells]
            nm = [(str(sp_names[int(grp[i][0])]), str(grp[i][1])) for i in keep]
            arrs.append(np.column_stack([cs[i] / grp[i][2] for i in keep]))
            var = d["var_names"].astype(str)
            names = nm
    return np.mean(arrs, 0), names, var
