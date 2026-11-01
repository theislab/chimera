"""Canonical cohorts, run state and per-gene covariates shared by the T2 analyses.

Read-only on ``training/runs`` and ``data/processed``. Gene identity goes through
``gene_ids`` only; cross-tissue joins use ``hsapiens_gene_id``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from revision.common import gene_ids

RUNS_DIR = gene_ids.REPO_ROOT / "training" / "runs"
MIN_CELLS_FQR = 20  # chimera/_constants.py: genes need >= 20 expressing cells in a 256-cell batch
BATCH_SIZE = 256

# cohort -> canonical s42 run (07_EXPERIMENT_MATRIX.md)
CANONICAL = {
    "liv_hmonkey": "iter90-150ep-s42",
    "liv_hmouse": "xds-hmouse-iter90-150ep-s42",
    "liv_hzeb": "xds-hzeb-iter85-s42",
    "liv_4sp": "xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42",
    "test_hmac": "testis-hmac-iter109-pclamp085-200ep-s42",
    "test_hmouse": "testis-hmouse-iter111-pclamp-coc-200ep-s42",
    "test_3sp": "testis-unified-iter115-pclamp-coc-100ep-s42",
}
SEED_RUNS = {
    "liv_hmonkey": ["iter90-150ep-s42", "iter90-150ep-s43", "iter90-150ep-s44"],
    "liv_hmouse": ["xds-hmouse-iter90-150ep-s42", "xds-hmouse-iter90-150ep-s43",
                   "xds-hmouse-iter90-150ep-s44"],
}
COHORT_ORDER = list(CANONICAL)
LABELS = {
    "liv_hmonkey": "Liver H+Monkey", "liv_hmouse": "Liver H+Mouse", "liv_hzeb": "Liver H+Zebrafish",
    "liv_4sp": "Liver 4-sp", "test_hmac": "Testis H+Macaque", "test_hmouse": "Testis H+Mouse",
    "test_3sp": "Testis 3-sp",
}
# liver cohort, testis cohort, label. The first two share the species pair.
TISSUE_PAIRS = [
    ("liv_hmonkey", "test_hmac", "H+Monkey vs H+Macaque"),
    ("liv_hmouse", "test_hmouse", "H+Mouse vs H+Mouse"),
    ("liv_4sp", "test_3sp", "4-sp vs 3-sp"),
]
GIN_COLUMNS = {"esm64": list(range(64)), "esm_var": [64], "mean_identity": [65],
               "expr3": [66, 67, 68], "identity3": [69, 70, 71]}


def is_testis(cohort: str) -> bool:
    return cohort.startswith("test")


def load_state(run: str) -> dict:
    """Arrays of ``final_state.npz`` plus ``config``; never writes to the run dir."""
    d = RUNS_DIR / run
    with np.load(d / "final_state.npz", allow_pickle=False) as z:
        out = {k: np.asarray(z[k]) for k in z.files}
    out["config"] = json.loads((d / "config.json").read_text())
    return out


def raw_prior(cohort: str, run: str, adata_prior: np.ndarray) -> np.ndarray:
    """Unclamped prior: ``conservation_prior_raw.npy`` on testis, the h5ad prior on liver."""
    f = RUNS_DIR / run / "conservation_prior_raw.npy"
    if f.exists():
        raw = np.load(f)
        if not np.allclose(raw, adata_prior, atol=1e-6):
            raise AssertionError(f"{run}: conservation_prior_raw.npy differs from h5ad prior")
        return raw.astype(float)
    if is_testis(cohort):
        raise FileNotFoundError(f"{run}: testis run without conservation_prior_raw.npy")
    return np.asarray(adata_prior, dtype=float)


def read_adata(cohort: str, backed: bool = False):
    import anndata as ad

    path = gene_ids.DATA_PROCESSED / f"{gene_ids.H5ADS[cohort]}.h5ad"
    return ad.read_h5ad(path, backed="r" if backed else None)


def _csr(X):
    import scipy.sparse as sp

    return X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)


def tau_index(M: np.ndarray) -> np.ndarray:
    """Cell-type specificity tau on log1p means (genes x types)."""
    M = np.log1p(M)
    mx = M.max(1, keepdims=True)
    mx[mx == 0] = 1
    return ((1 - M / mx).sum(1)) / (M.shape[1] - 1)


def gene_covariates(adata) -> pd.DataFrame:
    """Per-gene covariates, definitions as in the earlier read-only analysis, plus species-balanced versions.

    Pooled over cells: ``log_mean`` = log1p mean count, ``detection``, ``log_var``.
    ``tau`` uses species-balanced cell-type means of CP10K (types with >= 20 cells,
    species-type groups with >= 10 cells). ``ppass`` is the probability that a gene
    passes the >= 20 expressing-cell mask in a 256-cell batch under sqrt-species sampling.
    Species-balanced (05 section 3): mean over species of species mean log1p CP10K, and of
    species detection fractions.
    """
    import scipy.sparse as sp
    from scipy.stats import binom

    X = _csr(adata.X).astype(np.float64)
    sp_lab = adata.obs["species"].astype(str).values
    ct_lab = adata.obs["cell_type"].astype(str).values
    lib = np.asarray(X.sum(1)).ravel()
    m = np.asarray(X.mean(0)).ravel()
    Xb = X.copy()
    Xb.data = (Xb.data > 0).astype(np.float64)
    det = np.asarray(Xb.mean(0)).ravel()
    X2 = X.copy()
    X2.data = X2.data ** 2
    var = np.asarray(X2.mean(0)).ravel() - m ** 2
    cpm = sp.diags(1e4 / np.maximum(lib, 1)).dot(X).tocsr()
    lcpm = cpm.copy()
    lcpm.data = np.log1p(lcpm.data)

    species = sorted(set(sp_lab))
    t = pd.crosstab(ct_lab, sp_lab)
    cts = [k for k in t.index if t.loc[k].sum() >= 20]
    Mct = np.zeros((adata.n_vars, len(cts)))
    for j, k in enumerate(cts):
        vals = []
        for s in species:
            mk = (ct_lab == k) & (sp_lab == s)
            if mk.sum() >= 10:
                vals.append(np.asarray(cpm[mk].mean(0)).ravel())
        Mct[:, j] = np.mean(vals, 0)
    pn = np.asarray(adata.uns["p_null"], dtype=float)
    logpn = np.log(pn)

    sps, inv = np.unique(sp_lab, return_inverse=True)
    cnt = np.bincount(inv)
    w = np.sqrt(len(sp_lab) / cnt)[inv]
    w /= w.sum()
    pdet = np.asarray(Xb.T.dot(w)).ravel()
    ppass = 1 - binom.cdf(MIN_CELLS_FQR - 1, BATCH_SIZE, pdet)

    sp_lcpm = np.stack([np.asarray(lcpm[sp_lab == s].mean(0)).ravel() for s in species])
    sp_det = np.stack([np.asarray(Xb[sp_lab == s].mean(0)).ravel() for s in species])
    out = pd.DataFrame(dict(
        log_mean=np.log1p(m), detection=det, log_var=np.log1p(var), tau=tau_index(Mct),
        species_logp_range=logpn.max(0) - logpn.min(0), ppass=ppass,
        log_cp10k_spmean=sp_lcpm.mean(0), detection_spmean=sp_det.mean(0),
        detection_sprange=sp_det.max(0) - sp_det.min(0),
    ), index=pd.Index(adata.var_names, name="var_name"))
    for i, s in enumerate(species):
        out[f"detection_{s}"] = sp_det[i]
    return out


@dataclass
class Cohort:
    """Canonical run state joined to gene identity and covariates for one cohort."""

    name: str
    run: str
    genes: pd.DataFrame
    state: dict = field(repr=False)
    adata: object = field(default=None, repr=False)


def load_cohort(cohort: str, run: str | None = None, keep_adata: bool = False,
                covariates: bool = True) -> Cohort:
    run = run or CANONICAL[cohort]
    st = load_state(run)
    a = read_adata(cohort, backed=not (covariates or keep_adata))
    if len(st["alpha"]) != a.n_vars:
        raise AssertionError(f"{run}: {len(st['alpha'])} genes vs h5ad {a.n_vars}")
    gene_ids.assert_table_matches(a, gene_ids.load_table(cohort))
    vn = pd.Index(a.var_names, name="var_name")
    praw = raw_prior(cohort, run, a.var["conservation_prior"].to_numpy(dtype=float))
    g = pd.DataFrame(index=vn)
    g["hsapiens_gene_id"] = gene_ids.ensembl(cohort, vn).values
    g["symbol"] = gene_ids.symbols(cohort, vn).values
    g["alpha"] = st["alpha"].astype(float)
    g["prior"] = st["prior"].astype(float)
    g["prior_raw"] = praw
    g["delta_alpha"] = g["alpha"] - g["prior"]
    g["delta_alpha_raw"] = g["alpha"] - g["prior_raw"]
    g["correction"] = st["correction"].astype(float)
    g["S_g_ema"] = st["S_g_ema"].astype(float)
    F = np.asarray(a.varm["gene_features"], dtype=float)
    g["mean_identity_feature"] = F[:, 65]
    if covariates:
        g = g.join(gene_covariates(a))
    st["gene_features"] = F
    if not keep_adata:
        if a.isbacked:
            a.file.close()
        a = None
    return Cohort(cohort, run, g, st, a)


def expression_residual(values: pd.Series, log_mean: pd.Series, n_bins: int = 50) -> pd.Series:
    """Remove the running mean over ``n_bins`` log-mean quantile bins (reference definition)."""
    q = pd.qcut(log_mean.rank(method="first"), n_bins, labels=False)
    return values - values.groupby(q).transform("mean")
