"""Pre-training check that a new cohort h5ad follows the canonical Chimera conventions.

Usage: python -m revision.common.check_cohort <h5ad> [--batch-key species] [--celltype-key cell_ontology_class]

Fails (exit 1) unless:
* X holds raw non-negative integer counts and ``layers`` is empty (Chimera reads X as counts);
* ``obs`` has ``species``, the batch key and the cell-type key; every batch maps to one species;
* the batch key gives at most 2 batches per species: canonical recipes use batch == species, and
  several batches per species is a documented failure (iter94, ``training/workflow.md``);
* ``var['conservation_prior']`` is finite in [0, 1]; ``varm['gene_features']`` has 72 finite columns;
* ``uns['p_null']`` has shape (n_species, n_genes), non-negative, rows summing to about 1.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def check(path: str, batch_key: str = "species", celltype_key: str | None = "cell_ontology_class",
          n_rows: int = 5000) -> list[str]:
    import anndata as ad

    a = ad.read_h5ad(path, backed="r")
    errs = []
    X = a.X[: min(n_rows, a.n_obs)]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if not np.allclose(X, np.round(X)) or X.min() < 0:
        errs.append("X is not raw non-negative integer counts")
    if len(a.layers):
        errs.append(f"layers not empty: {list(a.layers)}")
    for k in ["species", batch_key] + ([celltype_key] if celltype_key else []):
        if k not in a.obs:
            errs.append(f"obs lacks {k!r}")
    if "species" in a.obs and batch_key in a.obs:
        per = a.obs.groupby(batch_key, observed=True)["species"].nunique()
        if (per != 1).any():
            errs.append(f"batches spanning several species: {per[per != 1].index.tolist()[:5]}")
        nb = a.obs.groupby("species", observed=True)[batch_key].nunique()
        if (nb > 2).any():
            errs.append(f"more than 2 batches per species with batch_key={batch_key!r}: {nb.to_dict()} "
                        "(multi-batch-per-species fails, iter94; train with --batch-key species)")
    pr = np.asarray(a.var.get("conservation_prior", np.array([np.nan])), dtype=float)
    if not (np.isfinite(pr).all() and pr.min() >= 0 and pr.max() <= 1):
        errs.append("var['conservation_prior'] missing or outside [0, 1]")
    F = a.varm.get("gene_features") if "gene_features" in a.varm else None
    if F is None or np.asarray(F).shape != (a.n_vars, 72) or not np.isfinite(np.asarray(F)).all():
        errs.append("varm['gene_features'] missing, not (n_genes, 72), or non-finite")
    P = np.asarray(a.uns.get("p_null", np.empty(0)))
    n_sp = a.obs["species"].nunique() if "species" in a.obs else -1
    if P.shape != (n_sp, a.n_vars) or (P < 0).any() or not np.allclose(P.sum(1), 1, atol=1e-3):
        errs.append(f"uns['p_null'] shape {P.shape}, expected ({n_sp}, {a.n_vars}), non-negative rows summing to 1")
    a.file.close()
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5ad")
    ap.add_argument("--batch-key", default="species")
    ap.add_argument("--celltype-key", default="cell_ontology_class")
    a = ap.parse_args()
    errs = check(a.h5ad, a.batch_key, a.celltype_key or None)
    for e in errs:
        print("FAIL:", e)
    print("OK" if not errs else f"{len(errs)} problem(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
