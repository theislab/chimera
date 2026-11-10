"""T3.5: count-permutation control h5ads (keep sequence, destroy counts).

One gene permutation ``perm`` per control: slot ``g`` of the control keeps everything that
belongs to the sequence of gene ``g`` (``var_names``, ``var["conservation_prior"]``, the ESM-2
and identity columns of ``varm["gene_features"]``) and receives the counts of gene ``perm[g]``
in every species: ``X_new[:, g] = X[:, perm[g]]`` and ``uns["p_null"][:, g] = p_null[:, perm[g]]``
(``p_null`` is ``(K, G)``; ``layers`` is empty, so ``X`` is the only count matrix).

Arms (07_EXPERIMENT_MATRIX names ``permA``, ``permB``, ``permC``):

* ``a``: free permutation of all genes; the three expression scalars
  (``gene_features`` columns 66-68: log mean, log variance, fraction non-zero) stay at their
  slots, so features are fully detached from the counts.
* ``b``: free permutation; the expression scalars move with the counts
  (``F_new[:, 66:69] = F[perm, 66:69]``), so only sequence is detached.
* ``c``: permutation within 20 quantile bins of the pooled log1p mean count of the h5ad;
  scalars unchanged. Counts are exchanged only between genes of similar expression.

Per-cell libraries do not change. ``obs`` is copied unchanged (``total_counts_mt`` and
``pct_counts_mt`` there describe the original genes; training does not read them).
Seeds and the permutation are stored next to each h5ad (``<name>.perm.npz``, ``<name>.json``).

Usage::

    python analysis/tasks/T3_5_count_perm/make_control_h5ad.py [--only NAME ...] [--outdir DIR] [--force]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import gene_ids
from analysis.common.outputs import git_sha

from chimera._data_utils import precompute_p_null

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROLS_TSV = os.path.join(HERE, "controls.tsv")
OUT_DIR = gene_ids.REPO_ROOT / "data" / "analysis" / "controls"
EXPR_COLS = [66, 67, 68]
N_BINS = 20


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pooled_log_mean(X) -> np.ndarray:
    return np.log1p(np.asarray(X.mean(0), dtype=np.float64).ravel())


def expression_bins(X, n_bins: int = N_BINS) -> np.ndarray:
    lm = pooled_log_mean(X)
    return pd.qcut(pd.Series(lm).rank(method="first"), n_bins, labels=False).to_numpy()


def make_permutation(X, arm: str, seed: int, n_bins: int = N_BINS):
    rng = np.random.default_rng(seed)
    G = X.shape[1]
    if arm in ("a", "b"):
        return rng.permutation(G), None
    if arm != "c":
        raise ValueError(f"unknown arm {arm!r}")
    bins = expression_bins(X, n_bins)
    perm = np.arange(G)
    for k in range(n_bins):
        m = np.where(bins == k)[0]
        perm[m] = m[rng.permutation(len(m))]
    return perm, bins


def build(src: ad.AnnData, arm: str, perm: np.ndarray) -> ad.AnnData:
    if len(src.layers) or len(src.obsm) or set(src.varm.keys()) != {"gene_features"} \
            or set(src.uns.keys()) != {"p_null"} or len(src.obsp) or len(src.varp):
        raise AssertionError("unexpected AnnData slots; the control would leave counts behind")
    X = src.X.tocsr() if sp.issparse(src.X) else sp.csr_matrix(src.X)
    Xn = X[:, perm].tocsr()
    Xn.sort_indices()
    F = np.asarray(src.varm["gene_features"]).copy()
    if arm == "b":
        F[:, EXPR_COLS] = F[perm][:, EXPR_COLS]
    p_null = np.asarray(src.uns["p_null"], dtype=np.float64)[:, perm].copy()
    out = ad.AnnData(X=Xn, obs=src.obs.copy(), var=src.var.copy())
    out.varm["gene_features"] = F
    out.uns["p_null"] = p_null
    return out


def verify(src: ad.AnnData, new: ad.AnnData, arm: str, perm: np.ndarray, bins) -> dict:
    """Checks that the control is what the design says; raises on any failure."""
    X0 = src.X.tocsr()
    X1 = new.X.tocsr()
    G = src.n_vars
    tot0 = np.asarray(X0.sum(0), dtype=np.float64).ravel()
    tot1 = np.asarray(X1.sum(0), dtype=np.float64).ravel()
    lib0 = np.asarray(X0.sum(1), dtype=np.float64).ravel()
    lib1 = np.asarray(X1.sum(1), dtype=np.float64).ravel()
    F0 = np.asarray(src.varm["gene_features"])
    F1 = np.asarray(new.varm["gene_features"])
    pn = np.asarray(new.uns["p_null"])
    pn_re = precompute_p_null(new, species_key="species")
    keep_cols = [j for j in range(F0.shape[1]) if not (arm == "b" and j in EXPR_COLS)]
    # spot check of whole columns, including zeros
    rng = np.random.default_rng(0)
    probe = rng.choice(G, size=min(50, G), replace=False)
    col_equal = all(np.array_equal(X1[:, g].toarray(), X0[:, perm[g]].toarray()) for g in probe)
    lm_new = pooled_log_mean(X1)
    checks = dict(
        shape_equal=new.shape == src.shape,
        is_permutation=bool(np.array_equal(np.sort(perm), np.arange(G))),
        X_finite=bool(np.isfinite(X1.data).all()),
        X_nonneg=bool((X1.data >= 0).all()),
        nnz_equal=int(X0.nnz) == int(X1.nnz),
        gene_totals_moved=bool(np.array_equal(tot1, tot0[perm])),
        cell_libraries_equal=bool(np.allclose(lib1, lib0, rtol=0, atol=1e-6)),
        probe_columns_equal=bool(col_equal),
        p_null_moved=bool(np.array_equal(pn, np.asarray(src.uns["p_null"])[:, perm])),
        p_null_matches_recompute=bool(np.allclose(pn, pn_re, rtol=1e-6, atol=0)),
        p_null_finite_positive=bool(np.isfinite(pn).all() and (pn > 0).all()),
        p_null_shape=list(pn.shape),
        features_finite=bool(np.isfinite(F1).all()),
        sequence_features_in_place=bool(np.array_equal(F1[:, keep_cols], F0[:, keep_cols])),
        expr_scalars_moved=(bool(np.array_equal(F1[:, EXPR_COLS], F0[perm][:, EXPR_COLS])) if arm == "b"
                            else None),
        expr_scalars_in_place=(bool(np.array_equal(F1[:, EXPR_COLS], F0[:, EXPR_COLS])) if arm != "b"
                               else None),
        var_names_equal=bool(np.array_equal(new.var_names, src.var_names)),
        prior_in_place=bool(np.array_equal(new.var["conservation_prior"].to_numpy(),
                                           src.var["conservation_prior"].to_numpy())),
        obs_equal=bool(new.obs.equals(src.obs)),
        within_bins=(bool(np.array_equal(bins[perm], bins)) if bins is not None else None),
    )
    failed = [k for k, v in checks.items() if v is False]
    stats = dict(
        frac_genes_moved=float((perm != np.arange(G)).mean()),
        spearman_slot_logmean_new_vs_old=float(pd.Series(lm_new).corr(pd.Series(pooled_log_mean(X0)),
                                                                       method="spearman")),
        spearman_slot_logmean_new_vs_feature66=float(pd.Series(lm_new).corr(pd.Series(F1[:, 66]),
                                                                             method="spearman")),
        spearman_canonical_logmean_vs_feature66=float(pd.Series(pooled_log_mean(X0)).corr(
            pd.Series(F0[:, 66]), method="spearman")),
        n_genes_zero_total=int((tot1 == 0).sum()),
    )
    if failed:
        raise AssertionError(f"verification failed: {failed}")
    return dict(checks=checks, stats=stats, failed=failed)


def make_one(row: pd.Series, outdir, force: bool = False) -> dict:
    name = row["control"]
    dest = outdir / f"{name}.h5ad"
    meta_path = outdir / f"{name}.json"
    if dest.exists() and meta_path.exists() and not force:
        print(f"{name}: exists; skipping (use --force)", flush=True)
        return json.loads(meta_path.read_text())
    src_path = gene_ids.DATA_PROCESSED / f"{gene_ids.H5ADS[row['cohort']]}.h5ad"
    print(f"{name}: reading {src_path.name}", flush=True)
    src = ad.read_h5ad(src_path)
    src.X = src.X.tocsr() if sp.issparse(src.X) else sp.csr_matrix(src.X)
    perm, bins = make_permutation(src.X, row["arm"], int(row["perm_seed"]))
    new = build(src, row["arm"], perm)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".tmp.h5ad")
    new.write_h5ad(tmp)
    os.replace(tmp, dest)
    back = ad.read_h5ad(dest)
    ver = verify(src, back, row["arm"], perm, bins)
    np.savez_compressed(outdir / f"{name}.perm.npz", perm=perm,
                        bins=bins if bins is not None else np.full(len(perm), -1),
                        perm_seed=int(row["perm_seed"]))
    meta = dict(control=name, cohort=row["cohort"], arm=row["arm"], perm_seed=int(row["perm_seed"]),
                train_seed=int(row["train_seed"]), convention="X_new[:, g] = X[:, perm[g]]",
                n_bins=N_BINS if row["arm"] == "c" else None, expr_feature_cols=EXPR_COLS,
                scalars="move with counts" if row["arm"] == "b" else "stay at slots",
                source=str(src_path), source_sha256=sha256(src_path),
                h5ad=str(dest), h5ad_sha256=sha256(dest), n_obs=int(back.n_obs), n_vars=int(back.n_vars),
                git_sha=git_sha(), built_at=_dt.datetime.now().isoformat(timespec="seconds"), **ver)
    tmpj = meta_path.with_name(meta_path.name + ".tmp")
    tmpj.write_text(json.dumps(meta, indent=2, default=str))
    os.replace(tmpj, meta_path)
    print(f"{name}: ok, sha256 {meta['h5ad_sha256'][:16]}, {json.dumps(ver['stats'])}", flush=True)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, help="control names from controls.tsv")
    ap.add_argument("--outdir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    tab = pd.read_csv(CONTROLS_TSV, sep="\t")
    if args.only:
        tab = tab[tab["control"].isin(args.only)]
    from pathlib import Path

    metas = [make_one(r, Path(args.outdir), args.force) for _, r in tab.iterrows()]
    print(json.dumps({m["control"]: m["h5ad_sha256"] for m in metas}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
