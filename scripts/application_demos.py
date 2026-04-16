"""Application demonstrations for Chimera v2.4 cross-species integration.

Implements four analysis functions (Task 10.4) that use a trained model:
  1. per_species_pair_integration  -- (10.4d) per-gene, per-species-pair
     effective integration from log(mu_full / mu_shared_only)
  2. detect_convergent_evolution   -- (10.4a) flags cell-type pairs
     clustering together via low-alpha genes
  3. phylogenetic_alpha_analysis   -- (10.4b) stratifies alpha by
     phylogenetic distance between species pairs
  4. species_aware_imputation      -- (10.4c) imputes expression:
     high-alpha via shared decoder, low-alpha within species

These are post-training analysis scripts; a trained Chimera model
and an AnnData object are required.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Application 13 (Task 10.4d) — Per-species-pair integration
# NOTE: defined first because phylogenetic_alpha_analysis calls it.
# ------------------------------------------------------------------


def per_species_pair_integration(model, adata, species_key="species"):
    """Compute (genes x species_pairs) matrix of effective integration.

    Uses log(mu_full / mu_shared_only) as proxy for residual magnitude
    per cell, then averages absolute values per species to get per-gene
    divergence.

    Parameters
    ----------
    model
        Trained Chimera model with ``get_expression`` and
        ``get_shared_only_expression`` methods.
    adata : AnnData
        Annotated data matrix.
    species_key : str
        Column in ``adata.obs`` identifying species.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_genes, n_species_pairs)`` with columns
        ``"{sp_a}_vs_{sp_b}"``.
    """
    species = adata.obs[species_key].unique()
    full_expr = model.get_expression(adata)  # (N, G) counts-scale
    # library_size=None uses actual per-cell library sizes so that
    # shared-only expression matches the counts-scale of full_expr.
    # Default library_size=1.0 would make log_residual dominated by
    # log(library_size) ~ 9.2.
    shared_expr = model.get_shared_only_expression(adata, library_size=None)

    # Residual in log-space: log(mu_full) - log(mu_shared_only)
    log_residual = np.log(full_expr + 1e-8) - np.log(shared_expr + 1e-8)  # (N, G)

    results = {}
    for sp_a, sp_b in itertools.combinations(species, 2):
        mask_a = (adata.obs[species_key] == sp_a).values
        mask_b = (adata.obs[species_key] == sp_b).values
        # Mean absolute residual per gene for each species
        mean_res_a = np.abs(log_residual[mask_a]).mean(axis=0)  # (G,)
        mean_res_b = np.abs(log_residual[mask_b]).mean(axis=0)  # (G,)
        # Effective divergence: average of both species' residual magnitudes
        pair_divergence = (mean_res_a + mean_res_b) / 2  # (G,)
        results[f"{sp_a}_vs_{sp_b}"] = pair_divergence

    return pd.DataFrame(results, index=adata.var_names)


# ------------------------------------------------------------------
# Application 3 (Task 10.4a) — Detection of convergent evolution
# ------------------------------------------------------------------


def detect_convergent_evolution(
    model,
    adata,
    label_key="cell_type",
    species_key="species",
    knn_k=30,
    alpha_threshold=0.3,
    overlap_threshold=0.1,
):
    """Flag cell-type pairs that cluster together but via low-alpha genes.

    For each cross-species cell-type pair that clusters together
    (kNN overlap > *overlap_threshold*), compute the mean alpha of
    the top-100 genes driving cluster similarity (by shared expression
    variance).  If the mean alpha is below *alpha_threshold*, the pair
    is flagged as a potential convergent evolution candidate.

    Parameters
    ----------
    model
        Trained Chimera model.
    adata : AnnData
        Annotated data matrix.
    label_key : str
        Column in ``adata.obs`` with cell-type labels.
    species_key : str
        Column in ``adata.obs`` identifying species.
    knn_k : int
        Number of nearest neighbours.
    alpha_threshold : float
        Alpha cutoff below which a pair is flagged as convergent.
    overlap_threshold : float
        Minimum cross-species kNN overlap to consider a pair.

    Returns
    -------
    pd.DataFrame
        One row per evaluated cell-type / species pair with columns
        ``cell_type``, ``species_a``, ``species_b``, ``knn_overlap``,
        ``mean_alpha_top100``, and ``convergent_candidate``.
    """
    from sklearn.neighbors import NearestNeighbors

    z = model.get_latent_representation(adata)
    alpha = model.get_alpha()
    species = adata.obs[species_key].values
    labels = adata.obs[label_key].values
    unique_species = np.unique(species)

    # Build kNN graph in z-space
    nn = NearestNeighbors(n_neighbors=knn_k).fit(z)
    _, indices = nn.kneighbors(z)

    results = []
    for ct in np.unique(labels):
        for sp_a, sp_b in itertools.combinations(unique_species, 2):
            mask_a = (labels == ct) & (species == sp_a)
            mask_b = (labels == ct) & (species == sp_b)
            if mask_a.sum() < 10 or mask_b.sum() < 10:
                continue
            # kNN overlap: fraction of sp_a cells whose neighbours
            # include sp_b cells
            idx_a = np.where(mask_a)[0]
            idx_b_set = set(np.where(mask_b)[0])
            cross_neighbors = np.mean(
                [len(set(indices[i]) & idx_b_set) / knn_k for i in idx_a]
            )
            if cross_neighbors < overlap_threshold:
                continue
            # Genes driving similarity: top genes by shared expression
            # variance
            shared_expr = model.get_shared_only_expression(
                adata[mask_a | mask_b]
            )
            gene_importance = np.var(shared_expr, axis=0)  # (G,)
            top_genes = np.argsort(gene_importance)[-100:]  # top 100
            mean_alpha_top = alpha[top_genes].mean()

            results.append(
                {
                    "cell_type": ct,
                    "species_a": sp_a,
                    "species_b": sp_b,
                    "knn_overlap": cross_neighbors,
                    "mean_alpha_top100": mean_alpha_top,
                    "convergent_candidate": mean_alpha_top < alpha_threshold,
                }
            )

    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Application 5 (Task 10.4b) — Phylogenetically structured cell atlases
# ------------------------------------------------------------------


def phylogenetic_alpha_analysis(model, adata, species_tree, species_key="species"):
    """Stratify alpha by phylogenetic distance between species pairs.

    Parameters
    ----------
    model
        Trained Chimera model.
    adata : AnnData
        Annotated data matrix.
    species_tree : dict
        Mapping of ``(species_a, species_b)`` tuples to phylogenetic
        distances (float).
    species_key : str
        Column in ``adata.obs`` identifying species.

    Returns
    -------
    pd.DataFrame
        One row per species pair with phylogenetic distance, mean
        residual magnitude, Spearman correlation of alpha vs residual,
        and count of clade-specific genes.

    Notes
    -----
    Calls :func:`per_species_pair_integration` (Task 10.4d).
    """
    from scipy import stats

    alpha = model.get_alpha()  # (G,)
    # Per-species-pair residual divergence (Application 13, Task 10.4d)
    pair_residuals = per_species_pair_integration(model, adata, species_key)

    results = []
    for (sp_a, sp_b), phylo_dist in species_tree.items():
        col = f"{sp_a}_vs_{sp_b}"
        col_rev = f"{sp_b}_vs_{sp_a}"
        if col in pair_residuals.columns:
            residual_mag = pair_residuals[col].values
        elif col_rev in pair_residuals.columns:
            residual_mag = pair_residuals[col_rev].values
        else:
            continue
        # Correlation: do more distant species show larger residuals?
        # Use [0] indexing instead of .statistic for scipy <1.7 compat
        r_alpha_res = stats.spearmanr(alpha, residual_mag)[0]
        results.append(
            {
                "species_a": sp_a,
                "species_b": sp_b,
                "phylo_distance": phylo_dist,
                "mean_residual_magnitude": residual_mag.mean(),
                "spearman_alpha_vs_residual": r_alpha_res,
                # Low-alpha genes with large residuals = clade-specific
                # innovations
                "n_clade_specific": int(
                    ((alpha < 0.3) & (residual_mag > np.median(residual_mag))).sum()
                ),
            }
        )
    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Application 10 (Task 10.4c) — Species-aware expression imputation
# ------------------------------------------------------------------


def species_aware_imputation(model, adata, alpha_threshold=0.7):
    """Impute expression: high-alpha genes via shared decoder, low-alpha within species.

    For high-alpha genes (alpha > *alpha_threshold*), expression is
    imputed using the shared decoder only (species-agnostic).  For
    low-alpha genes, the full prediction including the species-specific
    residual is used.

    Both ``get_expression`` and ``get_shared_only_expression`` must use
    the same library-size scale.  ``get_expression()`` returns
    counts-scale (real library size); passing ``library_size=None`` to
    ``get_shared_only_expression`` uses actual per-cell library sizes
    to match.

    Parameters
    ----------
    model
        Trained Chimera model.
    adata : AnnData
        Annotated data matrix.
    alpha_threshold : float
        Genes with alpha above this value are imputed via the shared
        decoder.

    Returns
    -------
    imputed : np.ndarray
        Shape ``(n_cells, n_genes)`` of imputed expression values.
    high_alpha_genes : np.ndarray
        Boolean mask of shape ``(n_genes,)`` indicating which genes
        used the shared-only pathway.
    """
    alpha = model.get_alpha()
    # Both must use same library-size scale (counts-scale).
    shared_expr = model.get_shared_only_expression(adata, library_size=None)
    full_expr = model.get_expression(adata)

    # High-alpha: shared decoder (species-agnostic)
    # Low-alpha : full prediction (species-specific residual preserved)
    imputed = full_expr.copy()
    high_alpha_genes = alpha > alpha_threshold
    imputed[:, high_alpha_genes] = shared_expr[:, high_alpha_genes]
    return imputed, high_alpha_genes


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    """Command-line interface for running application demonstrations."""
    parser = argparse.ArgumentParser(
        description="Chimera v2.4 application demonstrations (Task 10.4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to saved Chimera model directory.",
    )
    parser.add_argument(
        "--adata-path",
        type=str,
        required=True,
        help="Path to AnnData (.h5ad) file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/applications",
        help="Directory to write output tables (default: results/applications).",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="cell_type",
        help="Column in adata.obs for cell-type labels (default: cell_type).",
    )
    parser.add_argument(
        "--species-key",
        type=str,
        default="species",
        help="Column in adata.obs for species identity (default: species).",
    )
    parser.add_argument(
        "--app",
        type=str,
        nargs="+",
        choices=["convergent", "phylo", "imputation", "pair_integration", "all"],
        default=["all"],
        help="Which application(s) to run (default: all).",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.3,
        help="Alpha threshold for convergent evolution detection (default: 0.3).",
    )
    parser.add_argument(
        "--imputation-alpha-threshold",
        type=float,
        default=0.7,
        help="Alpha threshold for species-aware imputation (default: 0.7).",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=30,
        help="Number of nearest neighbours for convergent evolution (default: 30).",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.1,
        help="kNN overlap threshold for convergent evolution (default: 0.1).",
    )
    args = parser.parse_args()

    import anndata as ad

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apps_to_run = set(args.app)
    run_all = "all" in apps_to_run

    logger.info("Loading model from %s", args.model_dir)
    # Import Chimera model loader
    from chimera.model import ChimeraModel

    model = ChimeraModel.load(args.model_dir)

    logger.info("Loading AnnData from %s", args.adata_path)
    adata = ad.read_h5ad(args.adata_path)

    # --- Application 13 / 10.4d: per-species-pair integration ---
    if run_all or "pair_integration" in apps_to_run:
        logger.info("Running per-species-pair integration (10.4d)...")
        df_pairs = per_species_pair_integration(
            model, adata, species_key=args.species_key
        )
        out_path = out_dir / "pair_integration.csv"
        df_pairs.to_csv(out_path)
        logger.info(
            "Per-species-pair integration: %d genes x %d pairs -> %s",
            df_pairs.shape[0],
            df_pairs.shape[1],
            out_path,
        )

    # --- Application 3 / 10.4a: convergent evolution ---
    if run_all or "convergent" in apps_to_run:
        logger.info("Running convergent evolution detection (10.4a)...")
        df_conv = detect_convergent_evolution(
            model,
            adata,
            label_key=args.label_key,
            species_key=args.species_key,
            knn_k=args.knn_k,
            alpha_threshold=args.alpha_threshold,
            overlap_threshold=args.overlap_threshold,
        )
        out_path = out_dir / "convergent_evolution.csv"
        df_conv.to_csv(out_path, index=False)
        n_flagged = df_conv["convergent_candidate"].sum() if len(df_conv) else 0
        logger.info(
            "Convergent evolution: %d pairs evaluated, %d flagged -> %s",
            len(df_conv),
            n_flagged,
            out_path,
        )

    # --- Application 10 / 10.4c: species-aware imputation ---
    if run_all or "imputation" in apps_to_run:
        logger.info("Running species-aware imputation (10.4c)...")
        imputed, high_alpha = species_aware_imputation(
            model, adata, alpha_threshold=args.imputation_alpha_threshold
        )
        out_path = out_dir / "imputed_expression.npy"
        np.save(out_path, imputed)
        logger.info(
            "Species-aware imputation: %d / %d genes used shared decoder -> %s",
            high_alpha.sum(),
            len(high_alpha),
            out_path,
        )

    # --- Application 5 / 10.4b: phylogenetic alpha analysis ---
    if run_all or "phylo" in apps_to_run:
        logger.info(
            "Skipping phylogenetic alpha analysis (10.4b): requires "
            "--species-tree argument (not yet implemented in CLI). "
            "Call phylogenetic_alpha_analysis() directly with a "
            "species_tree dict."
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
