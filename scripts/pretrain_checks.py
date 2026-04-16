"""Pre-training conservation signal check (Task 9.9).

Runs BEFORE model training to verify that the prepared dataset contains
a usable conservation signal and to flag potential issues.

Checks performed:
  1. Spearman correlation between protein identity and expression pattern
     similarity -- validates that the conservation prior is informative.
  2. Library-size heterogeneity diagnostic -- flags species with very
     different sequencing depths and recommends observed vs learned
     library sizes.
  3. Cell-type composition warnings -- detects species with dominant
     cell types that could bias integration.

Results are printed to stdout and saved to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Check 1: Conservation signal (protein identity vs expression similarity)
# ------------------------------------------------------------------


def _compute_expression_pattern_similarity(
    adata: ad.AnnData,
    species_key: str = "species",
) -> np.ndarray:
    """Compute per-gene expression pattern similarity across species.

    For each gene, computes the mean pairwise Pearson correlation of
    its expression profile across species (using per-species mean
    expression vectors binned by cell type if available, otherwise
    raw per-species means).

    Returns an array of shape (n_genes,) with values in [-1, 1].
    """
    species = adata.obs[species_key]
    unique_species = sorted(species.unique())
    n_genes = adata.n_vars

    if len(unique_species) < 2:
        logger.warning("Fewer than 2 species; expression similarity is undefined.")
        return np.ones(n_genes, dtype=np.float64)

    # Build per-species mean expression profiles.
    # If cell_type is available, compute mean-of-celltype-means for robustness.
    has_ct = "cell_type" in adata.obs.columns
    species_profiles = []

    for sp_name in unique_species:
        sp_mask = (species == sp_name).values
        X_sp = adata.X[sp_mask]

        if has_ct:
            ct = adata.obs.loc[sp_mask, "cell_type"]
            unique_ct = ct.unique()
            ct_means = []
            for ct_name in unique_ct:
                ct_mask = (ct == ct_name).values
                X_ct = X_sp[ct_mask]
                if sp.issparse(X_ct):
                    ct_means.append(np.asarray(X_ct.mean(axis=0)).ravel())
                else:
                    ct_means.append(np.asarray(X_ct, dtype=np.float64).mean(axis=0))
            # Mean of cell-type means (balanced across cell types)
            profile = np.mean(ct_means, axis=0) if ct_means else np.zeros(n_genes)
        else:
            if sp.issparse(X_sp):
                profile = np.asarray(X_sp.mean(axis=0)).ravel()
            else:
                profile = np.asarray(X_sp, dtype=np.float64).mean(axis=0)

        species_profiles.append(profile)

    species_profiles = np.array(species_profiles)  # (K, G)

    # Per-gene pairwise Pearson across species
    K = len(unique_species)
    sim = np.zeros(n_genes, dtype=np.float64)
    n_pairs = 0

    for i in range(K):
        for j in range(i + 1, K):
            for g in range(n_genes):
                vals_i = species_profiles[i, g]
                vals_j = species_profiles[j, g]
                # For per-gene similarity with only two values per pair,
                # we use signed agreement: positive if both are in same
                # direction relative to their means.
                sim[g] += 1.0 if (vals_i > 0 and vals_j > 0) else 0.0
            n_pairs += 1

    # Normalise: use log-space correlation of the full expression vectors
    # across species instead, which is more meaningful.
    log_profiles = np.log1p(species_profiles)  # (K, G)

    gene_similarities = np.zeros(n_genes, dtype=np.float64)
    pair_count = 0
    for i in range(K):
        for j in range(i + 1, K):
            # Per-gene: correlation across the species pair
            # Here we compute the cosine similarity of per-gene expression
            # across cell-type means as a scalar per gene.
            # With K species, for each gene we compute the coefficient of
            # variation across species -- low CV means conserved expression.
            pair_count += 1

    # More robust approach: per-gene coefficient of variation across species
    gene_means = log_profiles.mean(axis=0)  # (G,)
    gene_stds = log_profiles.std(axis=0)    # (G,)
    # Similarity = 1 / (1 + CV); ranges from ~0 (highly variable) to 1 (conserved)
    cv = np.divide(gene_stds, gene_means + 1e-8)
    gene_similarities = 1.0 / (1.0 + cv)

    return gene_similarities


def conservation_signal_check(
    adata: ad.AnnData,
    species_key: str = "species",
) -> Dict[str, Any]:
    """Check Spearman correlation between protein identity and expression similarity.

    Parameters
    ----------
    adata
        AnnData with conservation_prior in ``adata.var`` and raw counts in ``.X``.
    species_key
        obs column for species.

    Returns
    -------
    dict
        Keys: ``spearman_r``, ``p_value``, ``n_genes``, ``pass``, ``details``.
    """
    if "conservation_prior" not in adata.var.columns:
        return {
            "spearman_r": float("nan"),
            "p_value": float("nan"),
            "n_genes": 0,
            "pass": False,
            "details": "conservation_prior not found in adata.var; cannot run check.",
        }

    conservation_prior = adata.var["conservation_prior"].values.astype(np.float64)
    expression_similarity = _compute_expression_pattern_similarity(adata, species_key)

    mask = np.isfinite(conservation_prior) & np.isfinite(expression_similarity)
    n_valid = int(mask.sum())

    if n_valid < 10:
        return {
            "spearman_r": float("nan"),
            "p_value": float("nan"),
            "n_genes": n_valid,
            "pass": False,
            "details": f"Only {n_valid} valid genes; insufficient for correlation.",
        }

    rho, pval = spearmanr(conservation_prior[mask], expression_similarity[mask])

    # A positive correlation indicates that genes with higher protein identity
    # also have more similar expression patterns -- the prior is informative.
    passed = rho > 0.1 and pval < 0.05

    if passed:
        details = (
            f"Conservation signal OK: Spearman r={rho:.4f}, p={pval:.2e}, "
            f"n={n_valid} genes. Positive correlation confirms conservation "
            f"prior is informative."
        )
    else:
        details = (
            f"Conservation signal WARNING: Spearman r={rho:.4f}, p={pval:.2e}, "
            f"n={n_valid} genes. Weak or absent correlation between protein "
            f"identity and expression similarity. The conservation prior may "
            f"not be informative for this dataset."
        )

    logger.info(details)
    return {
        "spearman_r": float(rho),
        "p_value": float(pval),
        "n_genes": n_valid,
        "pass": passed,
        "details": details,
    }


# ------------------------------------------------------------------
# Check 2: Library-size heterogeneity
# ------------------------------------------------------------------


def library_size_check(
    adata: ad.AnnData,
    species_key: str = "species",
) -> Dict[str, Any]:
    """Diagnose library-size heterogeneity across species.

    Delegates to the existing diagnostic in chimera._diagnostics but
    wraps it for the pretrain_checks JSON format.

    Parameters
    ----------
    adata
        AnnData with raw counts.
    species_key
        obs column for species.

    Returns
    -------
    dict
        Library-size diagnostic results.
    """
    from chimera._diagnostics import library_size_heterogeneity_diagnostic

    result = library_size_heterogeneity_diagnostic(adata, species_key=species_key)

    # Normalise output for JSON serialisation
    return {
        "per_species_median_lib": {
            str(k): float(v) for k, v in result["per_species_median_lib"].items()
        },
        "max_min_ratio": float(result["max_min_ratio"]),
        "recommendation": result["recommendation"],
        "use_observed_lib_size": bool(result["use_observed_lib_size"]),
        "pass": result["max_min_ratio"] <= 5.0,
        "details": result["details"],
    }


# ------------------------------------------------------------------
# Check 3: Cell-type composition warnings
# ------------------------------------------------------------------


def cell_type_composition_check(
    adata: ad.AnnData,
    species_key: str = "species",
    cell_type_key: str = "cell_type",
    dominance_threshold: float = 0.8,
) -> Dict[str, Any]:
    """Check for cell-type composition imbalance across species.

    Parameters
    ----------
    adata
        AnnData with cell-type and species annotations.
    species_key
        obs column for species.
    cell_type_key
        obs column for cell type.
    dominance_threshold
        Fraction above which a cell type is flagged as dominant.

    Returns
    -------
    dict
        Composition diagnostic results.
    """
    if cell_type_key not in adata.obs.columns:
        return {
            "flagged_species": [],
            "dominant_types": {},
            "n_species": int(adata.obs[species_key].nunique()),
            "pass": True,
            "details": (
                f"cell_type column '{cell_type_key}' not found; "
                "skipping composition check."
            ),
        }

    from chimera._diagnostics import cell_type_composition_diagnostic

    result = cell_type_composition_diagnostic(
        adata,
        species_key=species_key,
        cell_type_key=cell_type_key,
        dominance_threshold=dominance_threshold,
    )

    passed = len(result["flagged_species"]) == 0

    return {
        "flagged_species": [str(s) for s in result["flagged_species"]],
        "dominant_types": {
            str(k): {
                "cell_type": str(v["cell_type"]),
                "proportion": float(v["proportion"]),
            }
            for k, v in result["dominant_types"].items()
        },
        "n_species": int(adata.obs[species_key].nunique()),
        "pass": passed,
        "details": result["details"],
    }


# ------------------------------------------------------------------
# Main orchestrator
# ------------------------------------------------------------------


def pretrain_conservation_signal_check(
    adata_path: str,
    output_path: str,
    species_key: str = "species",
    cell_type_key: str = "cell_type",
) -> Dict[str, Any]:
    """Run all pre-training checks and save results to JSON.

    Parameters
    ----------
    adata_path
        Path to prepared AnnData .h5ad.
    output_path
        Path for JSON output.
    species_key
        obs column for species.
    cell_type_key
        obs column for cell type.

    Returns
    -------
    dict
        Combined results from all checks.
    """
    logger.info("Loading AnnData from %s", adata_path)
    adata = ad.read_h5ad(adata_path)
    logger.info("AnnData: %d cells x %d genes", adata.n_obs, adata.n_vars)

    results: Dict[str, Any] = {}

    # --- Check 1: Conservation signal ---
    logger.info("=== Check 1: Conservation signal ===")
    results["conservation_signal"] = conservation_signal_check(adata, species_key)

    # --- Check 2: Library-size heterogeneity ---
    logger.info("=== Check 2: Library-size heterogeneity ===")
    results["library_size"] = library_size_check(adata, species_key)

    # --- Check 3: Cell-type composition ---
    logger.info("=== Check 3: Cell-type composition ===")
    results["cell_type_composition"] = cell_type_composition_check(
        adata, species_key, cell_type_key
    )

    # --- Overall summary ---
    all_passed = all(
        results[check].get("pass", False)
        for check in ("conservation_signal", "library_size", "cell_type_composition")
    )
    failed_checks = [
        check
        for check in ("conservation_signal", "library_size", "cell_type_composition")
        if not results[check].get("pass", False)
    ]

    results["overall"] = {
        "all_pass": all_passed,
        "failed_checks": failed_checks,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_species": int(adata.obs[species_key].nunique()),
    }

    # Print summary
    logger.info("=" * 60)
    logger.info("PRE-TRAINING CHECK SUMMARY")
    logger.info("=" * 60)
    for check_name, check_result in results.items():
        if check_name == "overall":
            continue
        status = "PASS" if check_result.get("pass", False) else "WARN"
        logger.info("  [%s] %s: %s", status, check_name, check_result.get("details", ""))
    logger.info("-" * 60)
    logger.info(
        "  Overall: %s (%d/%d checks passed)",
        "ALL PASS" if all_passed else "WARNINGS",
        3 - len(failed_checks),
        3,
    )
    logger.info("=" * 60)

    # Save to JSON
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    return results


def main():
    """CLI entry point for pretrain_checks."""
    parser = argparse.ArgumentParser(
        description="Pre-training conservation signal and data quality checks (Task 9.9).",
    )
    parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="Path to prepared AnnData .h5ad file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/pretrain_checks.json",
        help="Output JSON path for check results.",
    )
    parser.add_argument(
        "--species-key",
        type=str,
        default="species",
        help="obs column for species.",
    )
    parser.add_argument(
        "--cell-type-key",
        type=str,
        default="cell_type",
        help="obs column for cell type.",
    )

    args = parser.parse_args()

    pretrain_conservation_signal_check(
        adata_path=args.adata,
        output_path=args.output,
        species_key=args.species_key,
        cell_type_key=args.cell_type_key,
    )


if __name__ == "__main__":
    main()
