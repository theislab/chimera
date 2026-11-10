"""X2 (EXPLORATORY): shared statistics for the liver analyses.

Matched excess (as T2.6 / X1): for each set gene, the 40 nearest non-set genes in standardised
rank space of the covariates C = (prior_raw, species-balanced log CP10K, species-balanced
detection, log CDS length); one drawn per set gene, B draws; excess = set mean - null mean,
d = excess / SD of the score over all genes, two-sided empirical p. Matching C_hep adds tau and
hepatocyte enrichment. Uncentred delta-alpha is
never tested against zero. Residual scale: rank, OLS on natural-spline design of C
(``controls.design``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.common import controls as K
from analysis.common import gene_ids

OUT = gene_ids.REPO_ROOT / "results" / "analysis" / "X2"
MATCH_COLS = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]
# C_hep: C + cell-type specificity (tau) + hepatocyte enrichment, so a set of hepatocyte genes is
# compared with other hepatocyte genes of the same expression, prior and length
MATCH_HEP = MATCH_COLS + ["tau", "hep_enrich"]
MATCHINGS = {"C": MATCH_COLS, "C_hep": MATCH_HEP}
NN = 40


def load_genes(cohort: str) -> pd.DataFrame:
    g = pd.read_parquet(OUT / "genes" / f"{cohort}.parquet")
    for c in g.columns:
        if c in ("tail", "margin_support", "margin_support_forward", "lacks_margin_support", "complete") or \
                c.startswith(("zero__", "absent__")):
            g[c] = g[c].fillna(False).astype(bool)
    return g


def rank01(x: pd.Series) -> np.ndarray:
    return (x.rank(method="average") / x.notna().sum()).to_numpy()


def knn_null(values, Z, set_idx, pool, B, rng, nn=NN):
    d = ((Z[set_idx][:, None, :] - Z[pool][None, :, :]) ** 2).sum(-1)
    nb = np.argpartition(d, min(nn, len(pool) - 1), axis=1)[:, :nn]
    nbrs = pool[nb]
    pick = rng.integers(0, nn, size=(len(set_idx), B))
    return values[np.take_along_axis(nbrs, pick, axis=1)].mean(0)


def set_excess(g: pd.DataFrame, cohort: str, sets: dict, scores: list, B: int = 2000, seed: int = 0,
               matchings: dict | None = None) -> list[dict]:
    rows = []
    for mname, cols in (matchings or MATCHINGS).items():
        for r in _set_excess(g, cohort, sets, scores, B, seed, cols):
            r["matching"] = mname
            rows.append(r)
    return rows


def _set_excess(g, cohort, sets, scores, B, seed, match_cols) -> list[dict]:
    g = g.reset_index(drop=True) if "var_name" in g.columns else g.reset_index()
    rows = []
    okc = g[match_cols].notna().all(1).to_numpy()
    Z = np.full((len(g), len(match_cols)), np.nan)
    Z[okc] = np.column_stack([rank01(g.loc[okc, c]) for c in match_cols])
    for j, (name, mask) in enumerate(sets.items()):
        mask = np.asarray(mask, bool)
        for k, sc in enumerate(scores):
            if sc not in g.columns:
                continue
            v = g[sc].to_numpy(dtype=float)
            ok = okc & np.isfinite(v)
            idx = np.where(mask & ok)[0]
            pool = np.where(~mask & ok)[0]
            r = dict(cohort=cohort, set=name, score=sc, n_set=int(mask.sum()), n=len(idx),
                     symbols=";".join(sorted(map(str, g.loc[idx, "symbol"].dropna())))[:600])
            if len(idx) < 3:
                rows.append(r)
                continue
            rng = np.random.default_rng(seed + 1000 * j + k)
            null = knn_null(v, Z, idx, pool, B, rng)
            obs = v[idx].mean()
            sd = v[ok].std(ddof=1)
            mu = null.mean()
            r.update(set_mean=obs, null_mean=mu, excess=obs - mu, excess_d=(obs - mu) / sd if sd > 0 else np.nan,
                     p=(1 + (np.abs(null - mu) >= abs(obs - mu)).sum()) / (B + 1),
                     frac_set_negative=float((v[idx] < np.median(v[ok])).mean()))
            rows.append(r)
    return rows


def residualise(g: pd.DataFrame, col: str) -> pd.Series:
    ok = g[MATCH_COLS + [col]].notna().all(1)
    Z = K.design(g.loc[ok, MATCH_COLS], MATCH_COLS)
    out = pd.Series(np.nan, index=g.index)
    out.loc[ok] = K.residual_rank(g.loc[ok, col].to_numpy(), Z)
    return out
