"""Build a 2-species subset of the unified CAMEX AnnData for fast iteration.

Rationale
---------
The full unified AnnData has 4 species: Human (4,125), Monkey (8,483), Mouse (3,966),
Zebrafish (5,530). For fast iteration we need a reproducible 2-species subset. We
choose **Human + Monkey** because:

* Both are primates (mean conservation prior is high but non-degenerate), which gives
  the GIN a biologically meaningful signal to learn from.
* Monkey has 2 batches (CS172, CS173) — this is the only species in the dataset with
  within-species batch variation. Picking Monkey as one of the two means the
  `ConstrainedBatchOffset` is actually exercised (non-trivial zero-sum constraint).
* ~12.6k cells total → each epoch is ~50 batches of 256, small enough to train to 50
  epochs in minutes on MPS and 200 epochs in ~1 hour.

We recompute p_null on the subset (it is species-dependent) and we optionally re-standardize
the GIN features on the subset gene-set (conservatively off by default; the full-dataset
scaler is usually fine since most genes are shared).

Outputs to data/processed/:
    camex_human_monkey.h5ad   -- 2-species unified AnnData with recomputed p_null
    subset_config.json         -- metadata (species chosen, cell counts, date)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chimera._data_utils import precompute_p_null  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


DEFAULT_SPECIES = ("Human", "Monkey")


def build_subset(
    source_path: Path,
    species: tuple[str, ...],
    dest_path: Path,
    drop_empty_genes: bool = True,
    min_cells_per_gene: int = 3,
) -> ad.AnnData:
    """Build the subset and write to disk.

    Parameters
    ----------
    source_path
        Path to the full unified h5ad.
    species
        Species names to keep (must exist in adata.obs['species']).
    dest_path
        Output h5ad path.
    drop_empty_genes
        If True, drop genes with fewer than ``min_cells_per_gene`` cells
        expressing them in the subset. This prevents the subset from carrying
        genes that are only informative in the dropped species. ``var`` metadata
        (including ``conservation_prior``, ``gene_features``, ``p_null``) is
        subset accordingly.
    """
    logger.info("Loading %s", source_path)
    adata = ad.read_h5ad(source_path)

    available = set(adata.obs["species"].unique())
    missing = [s for s in species if s not in available]
    if missing:
        raise ValueError(f"Species {missing} not in dataset. Available: {sorted(available)}")

    logger.info("Selecting species: %s", species)
    mask = adata.obs["species"].isin(list(species)).values
    sub = adata[mask].copy()

    # Reset species category ordering so that batch_to_species indexing is
    # dense [0..K-1] starting from the kept species only.
    sub.obs["species"] = sub.obs["species"].astype(str).astype("category")
    # Keep original batch labels as categorical but drop unused categories.
    sub.obs["batch"] = sub.obs["batch"].astype(str).astype("category")

    logger.info(
        "After species filter: %d cells, %d genes. Species counts: %s. Batches: %s",
        sub.n_obs,
        sub.n_vars,
        sub.obs["species"].value_counts().to_dict(),
        sub.obs["batch"].value_counts().to_dict(),
    )

    if drop_empty_genes:
        X = sub.X
        if hasattr(X, "toarray"):
            nonzero_per_gene = (X != 0).sum(axis=0)
            nonzero_per_gene = np.asarray(nonzero_per_gene).ravel()
        else:
            nonzero_per_gene = (X != 0).sum(axis=0)

        gene_mask = nonzero_per_gene >= min_cells_per_gene
        n_dropped = int((~gene_mask).sum())
        if n_dropped:
            logger.info(
                "Dropping %d / %d genes expressed in fewer than %d cells after species filter.",
                n_dropped,
                len(gene_mask),
                min_cells_per_gene,
            )
        sub = sub[:, gene_mask].copy()

    # p_null must be recomputed: it is species- and gene-indexed. The old p_null
    # was (4, G_full); after subsetting we need (K_subset, G_subset).
    logger.info("Recomputing p_null on the subset...")
    p_null_new = precompute_p_null(sub, species_key="species")
    sub.uns["p_null"] = p_null_new
    logger.info(
        "p_null shape %s, min %.2e, max %.2e, median %.2e",
        p_null_new.shape,
        float(p_null_new.min()),
        float(p_null_new.max()),
        float(np.median(p_null_new)),
    )

    # varm['gene_features'] carries the 72-D StandardScaled features from the full
    # dataset. We keep them as-is: they are structural (ESM-2 + conservation scalars)
    # and do not depend on which species we subset to, except for the expression
    # mean/var/nonzero scalars (last 3 of the 72). Those were computed over the full
    # four-species set; for a conservative baseline we leave them, and we note this
    # in the subset_config. If downstream diagnostics flag mis-scaling we can swap in
    # a subset-recomputed scaler here.

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(dest_path)
    logger.info("Wrote subset to %s", dest_path)

    meta = {
        "source": str(source_path),
        "species": list(species),
        "n_obs": int(sub.n_obs),
        "n_vars": int(sub.n_vars),
        "batches": list(sub.obs["batch"].unique()),
        "species_cell_counts": {k: int(v) for k, v in sub.obs["species"].value_counts().to_dict().items()},
        "drop_empty_genes": drop_empty_genes,
        "min_cells_per_gene": min_cells_per_gene,
        "conservation_prior_stats": {
            "min": float(sub.var["conservation_prior"].min()),
            "max": float(sub.var["conservation_prior"].max()),
            "mean": float(sub.var["conservation_prior"].mean()),
            "std": float(sub.var["conservation_prior"].std()),
        },
        "gene_features_stats": {
            "shape": list(sub.varm["gene_features"].shape),
            "mean_abs": float(np.abs(sub.varm["gene_features"]).mean()),
            "std": float(sub.varm["gene_features"].std()),
            "note": "Carried over from full-dataset StandardScaler; see build_subset.py",
        },
        "p_null_stats": {
            "shape": list(p_null_new.shape),
            "min": float(p_null_new.min()),
            "max": float(p_null_new.max()),
            "median": float(np.median(p_null_new)),
        },
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }

    meta_path = dest_path.parent / f"{dest_path.stem}_config.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote metadata to %s", meta_path)

    return sub


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/processed/camex_pancreas_unified.h5ad"),
        help="Path to the full unified h5ad.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=list(DEFAULT_SPECIES),
        help=f"Species names to retain. Default {DEFAULT_SPECIES}.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Output h5ad path. Default: data/processed/camex_<species>_<species>.h5ad",
    )
    parser.add_argument(
        "--min-cells-per-gene",
        type=int,
        default=3,
        help="Drop genes with fewer than this many cells expressing them in the subset.",
    )
    parser.add_argument(
        "--keep-all-genes",
        action="store_true",
        help="If set, keep all genes even if they are zero in the subset.",
    )
    args = parser.parse_args()

    if args.dest is None:
        tag = "_".join(s.lower() for s in args.species)
        args.dest = Path(f"data/processed/camex_{tag}.h5ad")

    build_subset(
        source_path=args.source,
        species=tuple(args.species),
        dest_path=args.dest,
        drop_empty_genes=not args.keep_all_genes,
        min_cells_per_gene=args.min_cells_per_gene,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
