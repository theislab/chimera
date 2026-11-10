"""Shared pieces of the control analyses (T3.4-T3.6): seed consensus, covariate residuals, floor, calls.

Definitions follow the pre-specified analysis plan (sections 2, 3 and 9):

* D for a run is ``alpha - prior`` from ``final_state.npz`` (prior as the run anchored on it,
  i.e. clamped on testis recipes). The consensus is the mean over seeds; the cohort median
  is irrelevant for rank statistics and is not subtracted here.
* Covariates C: natural cubic splines (df 4, knots at the rank quartiles) of rank raw prior,
  rank species-mean log1p CP10K, rank species-mean detection and rank log human CDS length.
* Residual scale: rank each score, residualise on C by OLS, Pearson of the residuals.
* Floor F: for each of the cohort's seeds, rho(seed, mean of the other seeds), on the same scale.
  Calls (residual scale): preserved if rho >= min F, lost if rho < 0.5 * median F, partial otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.common import gene_ids

# Seeds that make up D per cohort (canonical s42, MPS seeds where they exist, T3.2 CPU seeds).
# The T3.1 seed-pooling rule may drop a seed; pass ``drop`` to the functions below.
SEED_SETS = {
    "liv_hmouse": {42: "xds-hmouse-iter90-150ep-s42", 43: "xds-hmouse-iter90-150ep-s43",
                   44: "xds-hmouse-iter90-150ep-s44", 45: "rev-liv_hmouse-s45", 46: "rev-liv_hmouse-s46"},
    "test_hmouse": {42: "testis-hmouse-iter111-pclamp-coc-200ep-s42", 43: "rev-test_hmouse-s43",
                    44: "rev-test_hmouse-s44", 45: "rev-test_hmouse-s45", 46: "rev-test_hmouse-s46"},
    "liv_4sp": {42: "xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42", 43: "rev-liv_4sp-s43",
                44: "rev-liv_4sp-s44", 45: "rev-liv_4sp-s45", 46: "rev-liv_4sp-s46"},
}
COVARIATE_COLUMNS = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]
COUNT_COVARIATES = ["log_cp10k_spmean", "detection_spmean"]  # move with the counts
SLOT_COVARIATES = ["prior_raw", "log_cds_length"]  # stay with the sequence
CDS_CACHE = gene_ids.REPO_ROOT / "results" / "analysis" / "T2_4" / "cds_features.csv"


def run_finished(run: str) -> bool:
    d = C.RUNS_DIR / run
    return (d / "summary.json").exists() and (d / "final_state.npz").exists()


def read_run(run: str) -> dict:
    """D, S_g_ema, correction, prior, alpha and the last epoch of one run (read-only)."""
    st = C.load_state(run)
    out = {k: np.asarray(st[k], dtype=float) for k in ("alpha", "prior", "correction", "S_g_ema")}
    out["D"] = out["alpha"] - out["prior"]
    import json

    summ = json.loads((C.RUNS_DIR / run / "summary.json").read_text())
    out["summary"] = summ
    return out


def seed_runs(cohort: str, drop: tuple[int, ...] = (), require_all: bool = False) -> dict[int, str]:
    runs = {s: r for s, r in SEED_SETS[cohort].items() if s not in drop}
    done = {s: r for s, r in runs.items() if run_finished(r)}
    missing = sorted(set(runs) - set(done))
    if missing and require_all:
        raise FileNotFoundError(f"{cohort}: seed runs not finished: {[runs[s] for s in missing]}")
    if missing:
        print(f"  {cohort}: using {len(done)} of {len(runs)} seeds; unfinished {[runs[s] for s in missing]}",
              flush=True)
    return done


def seed_matrix(cohort: str, runs: dict[int, str], key: str = "D") -> pd.DataFrame:
    return pd.DataFrame({s: read_run(r)[key] for s, r in runs.items()})


def cds_table() -> pd.DataFrame:
    if CDS_CACHE.exists():
        return pd.read_csv(CDS_CACHE).set_index("hsapiens_gene_id")
    from analysis.common import cds as CDS

    return CDS.human_cds_table()


def cohort_covariates(cohort: str, adata=None, prior_raw: np.ndarray | None = None,
                      cds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-gene covariates C for ``cohort`` (or for ``adata``, a control built on it)."""
    if adata is None:
        adata = C.read_adata(cohort)
    g = C.gene_covariates(adata)
    g["hsapiens_gene_id"] = gene_ids.ensembl(cohort, adata.var_names).values
    g["prior_raw"] = (np.asarray(prior_raw, dtype=float) if prior_raw is not None
                      else adata.var["conservation_prior"].to_numpy(dtype=float))
    cds = cds_table() if cds is None else cds
    g = g.join(cds[["log_cds_length"]], on="hsapiens_gene_id")
    return g


def ns_basis(x: np.ndarray, df: int = 4) -> np.ndarray:
    """Natural cubic spline basis (no intercept), ``df`` columns, interior knots at quantiles.

    Truncated-power form (Hastie, Tibshirani and Friedman, eq. 5.4-5.5): boundary knots at
    min and max, ``df - 1`` interior knots at the quantiles 1/df, ..., (df-1)/df. Knots that
    coincide with a boundary or with each other (heavy ties, e.g. the testis raw prior with a
    third of genes at its 0.95 cap) are dropped, which lowers the column count for that covariate.
    """
    x = np.asarray(x, dtype=float)
    inner = np.unique(np.quantile(x, np.arange(1, df) / df))
    inner = inner[(inner > x.min()) & (inner < x.max())]
    knots = np.r_[x.min(), inner, x.max()]
    K = len(knots)
    if K < 3:
        return ((x - x.mean()) / (x.std() + 1e-12))[:, None]

    def d(k):
        num = np.maximum(x - knots[k], 0) ** 3 - np.maximum(x - knots[K - 1], 0) ** 3
        return num / (knots[K - 1] - knots[k])

    cols = [x] + [d(k) - d(K - 2) for k in range(K - 2)]
    B = np.column_stack(cols)
    return (B - B.mean(0)) / (B.std(0) + 1e-12)


def design(cov: pd.DataFrame, columns=COVARIATE_COLUMNS, df: int = 4) -> np.ndarray:
    """Intercept plus a spline of the rank of each covariate. Rows with NaN must be removed first."""
    parts = [np.ones((len(cov), 1))]
    for c in columns:
        r = pd.Series(cov[c].to_numpy(dtype=float)).rank().to_numpy()
        parts.append(ns_basis(r, df))
    return np.hstack(parts)


def residual_rank(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    r = pd.Series(np.asarray(y, dtype=float)).rank().to_numpy()
    beta = np.linalg.lstsq(Z, r, rcond=None)[0]
    return r - Z @ beta


def rho_pair(a: np.ndarray, b: np.ndarray, Za: np.ndarray | None = None, Zb: np.ndarray | None = None) -> dict:
    """Spearman, and Pearson of covariate residuals of the ranks (each score on its own C)."""
    out = dict(spearman=float(spearmanr(a, b)[0]))
    if Za is not None:
        ea = residual_rank(a, Za)
        eb = residual_rank(b, Zb if Zb is not None else Za)
        out["residual"] = float(np.corrcoef(ea, eb)[0, 1])
    return out


def floor(M: pd.DataFrame, Z: np.ndarray | None, rows: np.ndarray | None = None) -> pd.DataFrame:
    """Leave-one-seed-out rho of each seed against the mean of the others, both scales."""
    out = []
    rows = np.arange(len(M)) if rows is None else rows
    for s in M.columns:
        others = M.drop(columns=s).mean(1).to_numpy()
        v = M[s].to_numpy()
        r = dict(seed=s, spearman=float(spearmanr(v, others)[0]))
        if Z is not None:
            r["residual"] = rho_pair(v[rows], others[rows], Z)["residual"]
        out.append(r)
    return pd.DataFrame(out)


def call(rho: float, F: np.ndarray) -> str:
    F = np.asarray(F, dtype=float)
    F = F[np.isfinite(F)]
    if not np.isfinite(rho) or len(F) == 0:
        return "NA"
    if rho >= F.min():
        return "preserved"
    if rho < 0.5 * np.median(F):
        return "lost"
    return "partial"


def consensus_excluding(M: pd.DataFrame, seed: int) -> np.ndarray:
    cols = [s for s in M.columns if s != seed]
    return M[cols].mean(1).to_numpy()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
