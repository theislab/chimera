"""Matched-cell-type h5ad for liver 4-sp (Hepatocytes + Endothelial only).

Mirrors ``training/build_subset.py`` with its keep-all-genes behavior: cells are subset,
every gene is kept (8,879), ``var`` and ``varm["gene_features"]`` are carried over unchanged
(the full-cohort StandardScaled features, as ``build_subset.py`` does), ``species`` and
``batch`` lose unused categories, and ``uns["p_null"]`` is recomputed on the subset with
``chimera._data_utils.precompute_p_null``. Hepatocytes and Endothelial are the only cell types
present in all four species.

Usage::

    python analysis/tasks/T3_6_matched/make_matched_h5ad.py [--outdir DIR] [--force]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from analysis.common import gene_ids
from analysis.common.outputs import git_sha

from chimera._data_utils import precompute_p_null

COHORT = "liv_4sp"
CELL_TYPES = ("Hepatocytes", "Endothelial")
NAME = "liv_4sp_matchA"
OUT_DIR = gene_ids.REPO_ROOT / "data" / "analysis" / "controls"
N_GENES = 8879


def sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build(src: ad.AnnData) -> ad.AnnData:
    keep = src.obs["cell_type"].astype(str).isin(CELL_TYPES).to_numpy()
    sub = src[keep].copy()
    sub.X = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    sub.obs["species"] = sub.obs["species"].astype(str).astype("category")
    sub.obs["batch"] = sub.obs["batch"].astype(str).astype("category")
    sub.obs["cell_type"] = sub.obs["cell_type"].astype(str).astype("category")
    sub.uns["p_null"] = precompute_p_null(sub, species_key="species")
    return sub


def verify(src: ad.AnnData, new: ad.AnnData) -> dict:
    X = new.X.tocsr()
    pn = np.asarray(new.uns["p_null"])
    species = sorted(new.obs["species"].astype(str).unique())
    tot = np.asarray(X.sum(0)).ravel()
    tab = new.obs.groupby(["species", "cell_type"], observed=True).size().unstack(fill_value=0)
    checks = dict(
        n_vars=int(new.n_vars) == N_GENES == int(src.n_vars),
        var_names_equal=bool(np.array_equal(new.var_names, src.var_names)),
        var_equal=bool(new.var.equals(src.var)),
        gene_features_equal=bool(np.array_equal(np.asarray(new.varm["gene_features"]),
                                                np.asarray(src.varm["gene_features"]))),
        only_matched_types=set(new.obs["cell_type"].astype(str)) == set(CELL_TYPES),
        all_species_both_types=bool((tab.reindex(columns=list(CELL_TYPES)).fillna(0) > 0).all().all()
                                    and len(tab) == src.obs["species"].nunique()),
        X_finite=bool(np.isfinite(X.data).all()),
        X_nonneg=bool((X.data >= 0).all()),
        p_null_shape=list(pn.shape) == [len(species), N_GENES],
        p_null_matches_recompute=bool(np.allclose(pn, precompute_p_null(new), rtol=1e-12, atol=0)),
        p_null_finite_positive=bool(np.isfinite(pn).all() and (pn > 0).all()),
        p_null_differs_from_full=bool(not np.allclose(pn, np.asarray(src.uns["p_null"]))),
        n_cells_expected=int(new.n_obs) == int(src.obs["cell_type"].astype(str).isin(CELL_TYPES).sum()),
    )
    failed = [k for k, v in checks.items() if v is False]
    if failed:
        raise AssertionError(f"verification failed: {failed}")
    stats = dict(n_obs=int(new.n_obs), n_obs_source=int(src.n_obs),
                 cells=json.loads(tab.to_json()),
                 batches=sorted(new.obs["batch"].astype(str).unique().tolist()),
                 n_genes_zero_total=int((tot == 0).sum()), n_genes_lt3_cells=int(
                     (np.asarray((X > 0).sum(0)).ravel() < 3).sum()))
    return dict(checks=checks, stats=stats, failed=failed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    dest = outdir / f"{NAME}.h5ad"
    meta_path = outdir / f"{NAME}.json"
    if dest.exists() and meta_path.exists() and not args.force:
        print(f"{dest} exists; nothing to do (use --force)")
        return 0
    src_path = gene_ids.DATA_PROCESSED / f"{gene_ids.H5ADS[COHORT]}.h5ad"
    src = ad.read_h5ad(src_path)
    new = build(src)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".tmp.h5ad")
    new.write_h5ad(tmp)
    os.replace(tmp, dest)
    back = ad.read_h5ad(dest)
    ver = verify(src, back)
    meta = dict(control=NAME, cohort=COHORT, cell_types=list(CELL_TYPES), genes="all kept",
                gene_features="carried over unchanged (full-cohort scaler)", p_null="recomputed on subset",
                source=str(src_path), source_sha256=sha256(src_path), h5ad=str(dest),
                h5ad_sha256=sha256(dest), git_sha=git_sha(),
                built_at=_dt.datetime.now().isoformat(timespec="seconds"), **ver)
    tmpj = meta_path.with_name(meta_path.name + ".tmp")
    tmpj.write_text(json.dumps(meta, indent=2, default=str))
    os.replace(tmpj, meta_path)
    print(json.dumps(dict(sha256=meta["h5ad_sha256"], **ver["stats"]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
