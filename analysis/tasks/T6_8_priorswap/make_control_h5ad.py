"""Prior swap: pi' = 0.5 pi + 0.5 smooth(pi, STRING direct graph), through data preparation only.

smooth(pi, G)[g] = mean of pi over the neighbors of g in the STRING 12.0 physical graph (direct
``experiments`` >= 400, same graph as F2, restricted to the cohort's genes); a gene without
neighbors keeps its own pi. Only ``var['conservation_prior']`` changes. Counts, ``p_null``, obs and
``gene_features`` (including the identity columns 65, 69-71 and the ESM PCs) are the canonical
ones. Canonical prior kept in ``var['conservation_prior_canonical']``.

    python analysis/tasks/T6_8_priorswap/make_control_h5ad.py
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import gene_ids
from analysis.common.outputs import git_sha
from analysis.tasks.T6_common import functional as FN

COHORT = "liv_hmouse"
NAME = "liv_hmouse_priorswap"
OUT_DIR = gene_ids.REPO_ROOT / "data" / "analysis" / "controls"
THR = 400
W = 0.5


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    src_path = gene_ids.DATA_PROCESSED / f"{gene_ids.H5ADS[COHORT]}.h5ad"
    src = ad.read_h5ad(src_path)
    ids = gene_ids.ensembl(COHORT, src.var_names).reset_index(drop=True)
    pi = src.var["conservation_prior"].to_numpy(dtype=float)
    # duplicated or missing human IDs get no edges through graph_arrays' lookup
    uniq = ids.where(~ids.duplicated(keep=False))
    uniq = uniq.fillna(pd.Series([f"_none{i}" for i in range(len(ids))]))
    e = FN.graph_arrays(FN.string_edges("9606", THR), uniq)
    G = len(pi)
    A = sp.csr_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(G, G))
    deg = np.asarray(A.sum(1)).ravel()
    smooth = pi.copy()
    has = deg > 0
    smooth[has] = (A @ pi)[has] / deg[has]
    new = (1 - W) * pi + W * smooth
    out = src.copy()
    out.var["conservation_prior_canonical"] = pi
    out.var["conservation_prior"] = new.astype(src.var["conservation_prior"].dtype)
    dest = OUT_DIR / f"{NAME}.h5ad"
    tmp = dest.with_name(dest.stem + ".tmp.h5ad")
    out.write_h5ad(tmp)
    os.replace(tmp, dest)
    back = ad.read_h5ad(dest)
    p1 = back.var["conservation_prior"].to_numpy(dtype=float)
    checks = dict(
        X_equal=bool((back.X != src.X).nnz == 0),
        p_null_equal=bool(np.array_equal(back.uns["p_null"], src.uns["p_null"])),
        features_equal=bool(np.array_equal(back.varm["gene_features"], src.varm["gene_features"])),
        obs_equal=bool(back.obs.equals(src.obs)),
        var_names_equal=bool(np.array_equal(back.var_names, src.var_names)),
        prior_in_unit=bool((p1 >= 0).all() and (p1 <= 1).all()),
        unconnected_unchanged=bool(np.allclose(p1[~has], pi[~has], atol=1e-6)),
    )
    if not all(checks.values()):
        raise AssertionError(f"verification failed: {checks}")
    stats = dict(n_genes=int(G), n_edges=int(len(e)), n_connected=int(has.sum()),
                 frac_connected=float(has.mean()),
                 spearman_new_vs_canonical=float(pd.Series(p1).corr(pd.Series(pi), method="spearman")),
                 pearson_new_vs_canonical=float(np.corrcoef(p1, pi)[0, 1]),
                 mean_abs_change=float(np.abs(p1 - pi).mean()), max_abs_change=float(np.abs(p1 - pi).max()),
                 frac_changed_gt_001=float((np.abs(p1 - pi) > 0.01).mean()),
                 min_new=float(p1.min()), max_new=float(p1.max()))
    meta = dict(control=NAME, cohort=COHORT, recipe="pi' = 0.5 pi + 0.5 smooth(pi, STRING experiments>=400)",
                source=str(src_path), source_sha256=sha256(src_path), h5ad=str(dest), h5ad_sha256=sha256(dest),
                git_sha=git_sha(), built_at=_dt.datetime.now().isoformat(timespec="seconds"), checks=checks, **stats)
    p = OUT_DIR / f"{NAME}.json"
    p.write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
