"""Cross-dataset alpha reproducibility and multi-seed stability (Task 9.6).

Tests whether alpha is a robust gene property (reproducible across datasets
and random seeds) or a dataset-specific artefact.

Two analyses:
  1. cross_dataset_reproducibility  -- Train on two datasets with overlapping
     species, compare alpha for shared genes via Spearman rank correlation.
  2. multi_seed_stability           -- Train the same config with multiple
     seeds, compute per-gene alpha standard deviation and flag unreliable genes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from chimera._model import ChimeraModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Core analysis functions
# ------------------------------------------------------------------


def cross_dataset_reproducibility(
    adata_a,
    adata_b,
    dataset_a_name: str = "pancreas",
    dataset_b_name: str = "testis",
    max_epochs: int = 200,
    seed: int = 42,
) -> dict:
    """Train Chimera on two different datasets with overlapping species.
    Compare alpha for the same genes across datasets."""

    import scvi

    # --- Train on dataset A ---
    logger.info("Training on dataset A (%s) for %d epochs ...", dataset_a_name, max_epochs)
    ChimeraModel.setup_anndata(adata_a, species_key="species", batch_key="batch")
    model_a = ChimeraModel.from_config(adata_a, "D")
    scvi.settings.seed = seed
    model_a.train(max_epochs=max_epochs)
    alpha_a = model_a.get_alpha()
    delta_a = model_a.get_delta_alpha()
    genes_a = set(adata_a.var_names)

    # --- Train on dataset B ---
    logger.info("Training on dataset B (%s) for %d epochs ...", dataset_b_name, max_epochs)
    ChimeraModel.setup_anndata(adata_b, species_key="species", batch_key="batch")
    model_b = ChimeraModel.from_config(adata_b, "D")
    scvi.settings.seed = seed
    model_b.train(max_epochs=max_epochs)
    alpha_b = model_b.get_alpha()
    delta_b = model_b.get_delta_alpha()
    genes_b = set(adata_b.var_names)

    # --- Compare on shared gene set ---
    # Use dict-based O(1) lookup instead of list.index() which is O(G) per call
    # (total O(G^2) for G shared genes -- unacceptable for G~15K)
    shared_genes = sorted(genes_a & genes_b)
    name_to_idx_a = {g: i for i, g in enumerate(adata_a.var_names)}
    name_to_idx_b = {g: i for i, g in enumerate(adata_b.var_names)}
    idx_a = [name_to_idx_a[g] for g in shared_genes]
    idx_b = [name_to_idx_b[g] for g in shared_genes]

    alpha_a_shared = alpha_a[idx_a]
    alpha_b_shared = alpha_b[idx_b]
    delta_a_shared = delta_a[idx_a]
    delta_b_shared = delta_b[idx_b]

    # Full gene set correlation
    r_alpha, p_alpha = spearmanr(alpha_a_shared, alpha_b_shared)
    r_delta, p_delta = spearmanr(delta_a_shared, delta_b_shared)

    # Stratified: high-|delta_alpha| genes only (tails are more meaningful)
    tail_mask = (np.abs(delta_a_shared) > 0.1) | (np.abs(delta_b_shared) > 0.1)
    if tail_mask.sum() > 20:
        r_tail, _ = spearmanr(alpha_a_shared[tail_mask], alpha_b_shared[tail_mask])
    else:
        r_tail = float("nan")

    logger.info(
        "Cross-dataset results: %d shared genes, alpha Spearman r=%.4f (p=%.2e), "
        "tail r=%.4f (%d tail genes)",
        len(shared_genes), r_alpha, p_alpha, r_tail, int(tail_mask.sum()),
    )

    return {
        "n_shared_genes": len(shared_genes),
        "alpha_spearman_r": float(r_alpha),
        "alpha_spearman_p": float(p_alpha),
        "delta_alpha_spearman_r": float(r_delta),
        "delta_alpha_spearman_p": float(p_delta),
        "tail_genes_spearman_r": float(r_tail),
        "n_tail_genes": int(tail_mask.sum()),
        "interpretation": {
            "r > 0.5": "alpha is a gene property -- robust cross-dataset metric",
            "0.3 < r < 0.5": "moderate reproducibility -- tissue-dependent component present",
            "r < 0.3": "alpha is dataset-specific -- tissue-dependent conservation, also a novel finding",
        },
        "dataset_a": dataset_a_name,
        "dataset_b": dataset_b_name,
    }


def multi_seed_stability(
    adata,
    n_seeds: int = 5,
    max_epochs: int = 200,
) -> dict:
    """Train the same config with multiple seeds, compute per-gene alpha stability."""

    import scvi

    seeds = [42, 123, 456, 789, 1024][:n_seeds]
    alpha_runs = []

    for s in seeds:
        logger.info("Training seed %d / %d ...", s, seeds[-1])
        adata_copy = adata.copy()
        ChimeraModel.setup_anndata(adata_copy, species_key="species", batch_key="batch")
        model = ChimeraModel.from_config(adata_copy, "D")
        scvi.settings.seed = s
        model.train(max_epochs=max_epochs)
        alpha_runs.append(model.get_alpha())

    alpha_matrix = np.stack(alpha_runs, axis=0)  # (n_seeds, G)
    alpha_mean = alpha_matrix.mean(axis=0)       # (G,)
    alpha_std = alpha_matrix.std(axis=0)         # (G,)

    # Pairwise Spearman between all seed pairs
    pairwise_r = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            r, _ = spearmanr(alpha_runs[i], alpha_runs[j])
            pairwise_r.append(r)

    logger.info(
        "Multi-seed results: mean pairwise r=%.4f, mean alpha std=%.4f, "
        "%d / %d unreliable genes (std > 0.05)",
        float(np.mean(pairwise_r)),
        float(alpha_std.mean()),
        int((alpha_std > 0.05).sum()),
        len(alpha_std),
    )

    return {
        "n_seeds": n_seeds,
        "mean_pairwise_spearman_r": float(np.mean(pairwise_r)),
        "min_pairwise_spearman_r": float(np.min(pairwise_r)),
        "mean_alpha_std": float(alpha_std.mean()),
        "median_alpha_std": float(np.median(alpha_std)),
        "frac_unreliable": float((alpha_std > 0.05).mean()),
        "n_unreliable_genes": int((alpha_std > 0.05).sum()),
        "alpha_mean": alpha_mean,
        "alpha_std": alpha_std,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset alpha reproducibility and multi-seed stability (Task 9.6).",
    )
    sub = parser.add_subparsers(dest="command")

    # -- cross-dataset --
    cross_p = sub.add_parser("cross-dataset", help="Compare alpha across two datasets.")
    cross_p.add_argument("--adata-a", required=True, help="Path to .h5ad for dataset A.")
    cross_p.add_argument("--adata-b", required=True, help="Path to .h5ad for dataset B.")
    cross_p.add_argument("--name-a", default="pancreas", help="Label for dataset A.")
    cross_p.add_argument("--name-b", default="testis", help="Label for dataset B.")
    cross_p.add_argument("--max-epochs", type=int, default=200)
    cross_p.add_argument("--seed", type=int, default=42)
    cross_p.add_argument("--output", default=None, help="Path to save JSON results.")

    # -- multi-seed --
    seed_p = sub.add_parser("multi-seed", help="Train with multiple seeds on one dataset.")
    seed_p.add_argument("--adata", required=True, help="Path to .h5ad.")
    seed_p.add_argument("--n-seeds", type=int, default=5)
    seed_p.add_argument("--max-epochs", type=int, default=200)
    seed_p.add_argument("--output", default=None, help="Path to save JSON results.")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    import anndata as ad

    if args.command == "cross-dataset":
        logger.info("Loading dataset A: %s", args.adata_a)
        adata_a = ad.read_h5ad(args.adata_a)
        logger.info("Loading dataset B: %s", args.adata_b)
        adata_b = ad.read_h5ad(args.adata_b)

        results = cross_dataset_reproducibility(
            adata_a,
            adata_b,
            dataset_a_name=args.name_a,
            dataset_b_name=args.name_b,
            max_epochs=args.max_epochs,
            seed=args.seed,
        )

    elif args.command == "multi-seed":
        logger.info("Loading dataset: %s", args.adata)
        adata = ad.read_h5ad(args.adata)

        results = multi_seed_stability(
            adata,
            n_seeds=args.n_seeds,
            max_epochs=args.max_epochs,
        )
        # Strip non-serialisable numpy arrays before JSON dump
        results = {
            k: v for k, v in results.items()
            if not isinstance(v, np.ndarray)
        }

    # Print results
    print(json.dumps(results, indent=2))

    # Optionally save to file
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
