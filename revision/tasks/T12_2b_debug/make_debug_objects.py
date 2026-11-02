"""T12.2b: diagnostic objects for the bone-marrow training failure (health only, no readout).

Each object is drawn from a parent bone-marrow h5ad by a rule on cell counts and per-cell
totals only. Genes, ``var``, ``varm`` and the cohort prior are kept as in the parent;
``uns['p_null']`` is recomputed on the drawn cells (``chimera._data_utils.precompute_p_null``,
as ``training/build_subset.py`` does).

Rules (``--rule``):

* ``prop``: proportional stratified draw of ``--n`` cells. Each species x cell_ontology_class
  stratum keeps round(n * stratum share) cells, drawn uniformly without replacement
  (numpy default_rng(--seed) over sorted cell IDs). The composition of the parent is kept, so
  the object differs from the parent only in size (hypothesis 1: dataset scale).
* ``prop_libq``: as ``prop``, after removing, per species, cells whose total count is above
  the species' 99th percentile (hypothesis 2: extreme libraries).
* ``prop_thin``: as ``prop``, then every count is binomially thinned with probability
  ``--thin`` (numpy default_rng(--seed)), so every library shrinks by that factor and relative
  depth across species is kept (hypothesis 2: absolute sequencing depth; 1/3 brings the human
  median, 5,579, near the testis human median, 1,718).

    python revision/tasks/T12_2b_debug/make_debug_objects.py --rule prop --n 25000 \
        --out data/revision/bone_marrow/debug/bm_dbg_prop25k.h5ad
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from chimera._data_utils import precompute_p_null  # noqa: E402

PARENT = REPO / "data" / "revision" / "bone_marrow" / "bone_marrow_unified.h5ad"
STRATA = ("species", "cell_ontology_class")


def proportional(obs: pd.DataFrame, eligible: np.ndarray, n: int, seed: int) -> np.ndarray:
    names = np.asarray(obs.index)
    key = obs[list(STRATA)].astype(str).agg("|".join, axis=1).to_numpy()
    total = int(eligible.sum())
    keep = np.zeros(len(obs), dtype=bool)
    for k in sorted(np.unique(key[eligible])):
        idx = np.flatnonzero((key == k) & eligible)
        m = min(len(idx), int(round(n * len(idx) / total)))
        order = idx[np.argsort(names[idx], kind="stable")]
        keep[np.random.default_rng(seed).choice(order, size=m, replace=False)] = True
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, default=PARENT)
    ap.add_argument("--rule", choices=["prop", "prop_libq", "prop_thin"], required=True)
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thin", type=float, default=1 / 3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import anndata as ad
    import scipy.sparse as sp

    a = ad.read_h5ad(args.parent)
    X = sp.csr_matrix(a.X)
    lib = np.asarray(X.sum(1)).ravel()
    eligible = np.ones(a.n_obs, dtype=bool)
    rec: dict = {"parent": str(args.parent), "rule": args.rule, "n": args.n, "seed": args.seed}
    if args.rule == "prop_libq":
        sp_ = a.obs["species"].astype(str).to_numpy()
        rec["lib_q99"] = {}
        for s in sorted(set(sp_)):
            m = sp_ == s
            q = float(np.quantile(lib[m], 0.99))
            rec["lib_q99"][s] = q
            eligible[m & (lib > q)] = False
        rec["removed_above_q99"] = int((~eligible).sum())
    keep = proportional(a.obs, eligible, args.n, args.seed)
    sub = a[keep].copy()
    if args.rule == "prop_thin":
        Xs = sp.csr_matrix(sub.X)
        Xs.data = np.random.default_rng(args.seed).binomial(Xs.data.astype(np.int64), args.thin).astype(np.float32)
        Xs.eliminate_zeros()
        sub.X = Xs
        rec["thin"] = args.thin
    sub.uns["p_null"] = precompute_p_null(sub, species_key="species")
    for c in sub.obs.columns:
        if isinstance(sub.obs[c].dtype, pd.CategoricalDtype):
            sub.obs[c] = sub.obs[c].cat.remove_unused_categories()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(args.out)
    sl = np.asarray(sp.csr_matrix(sub.X).sum(1)).ravel()
    rec["shape"] = list(sub.shape)
    rec["counts"] = pd.crosstab(sub.obs["cell_ontology_class"], sub.obs["species"]).to_dict()
    rec["lib_max_by_species"] = {s: float(sl[(sub.obs["species"] == s).to_numpy()].max())
                                 for s in sub.obs["species"].astype(str).unique()}
    args.out.with_suffix(".json").write_text(json.dumps(rec, indent=2, default=str))
    print(json.dumps({k: rec[k] for k in ("rule", "shape", "lib_max_by_species")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
