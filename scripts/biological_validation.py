"""Biological ground truth validation for Chimera v2.4 alpha values.

Validates that learned alpha (cross-species integration coefficients)
reflect known biology:
  - Known-conserved markers (hepatocyte, T-cell, epithelial, housekeeping)
    should have alpha above the gene-wide median.
  - Known-divergent families (immune receptors, MHC class II, cytokines,
    CYP enzymes) should have alpha below the median.
  - Mann-Whitney U test confirms the two distributions are significantly
    separated (p < 0.05).
  - Delta-alpha outlier enrichment reports top-20 positive/negative genes
    for downstream GO enrichment.

Task 9.7 of the Chimera integration pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Curated gene sets (extend based on species and tissue)
# ------------------------------------------------------------------

KNOWN_CONSERVED_MARKERS = {
    # Tissue-specific markers with well-documented cross-species conservation
    "hepatocyte": ["ALB", "APOA1", "APOB", "CYP2E1", "PCK1"],
    "T_cell": ["CD3E", "CD3D", "CD3G", "CD2", "LCK"],
    "epithelial": ["EPCAM", "KRT8", "KRT18", "CDH1"],
    "housekeeping": ["ACTB", "GAPDH", "RPL13A", "RPS18", "UBC"],
}

KNOWN_DIVERGENT_FAMILIES = {
    # Gene families with well-documented cross-species regulatory divergence
    "immune_receptors": ["KLRK1", "KLRD1", "NCR1", "NCR3", "FCGR3A"],
    "MHC_class_II": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1"],
    "cytokines": ["IL17A", "IL22", "IFNG", "IL4"],
    "CYP_enzymes": ["CYP3A4", "CYP2D6", "CYP1A2", "CYP2C9"],
}


# ------------------------------------------------------------------
# Core validation
# ------------------------------------------------------------------


def biological_ground_truth(model, adata, gene_table=None):
    """Validate alpha against known biology.

    IMPORTANT: Gene lists use human gene names, but unified AnnData (Task 1.11)
    uses ortholog group IDs as var_names. Must use adata.var["human_gene_name"]
    mapping (populated from Task 1.3 ortholog table) for lookup. Without this
    column, falls back to var_names directly (works only if var_names are human)."""
    alpha = model.get_alpha()
    delta = model.get_delta_alpha()
    gene_names = list(adata.var_names)

    # Build name-to-index lookup via human_gene_name column (Task 1.11 wiring)
    if "human_gene_name" in adata.var.columns:
        name_to_idx = {name: i for i, name in enumerate(adata.var["human_gene_name"].values)
                       if isinstance(name, str) and name != ""}
    else:
        # Fallback: var_names are human gene names (only works for single-species or
        # if Task 1.11 used human names as var_names instead of ortholog group IDs)
        # Run 27 fix: emit runtime warning — without human_gene_name column,
        # all gene categories will silently return empty if var_names are ortholog IDs
        warnings.warn(
            "adata.var['human_gene_name'] column not found — falling back to var_names. "
            "If var_names are ortholog group IDs (expected from Task 1.11), all gene "
            "lookups will fail silently. Add human_gene_name column via Task 1.11.",
            UserWarning,
        )
        name_to_idx = {name: i for i, name in enumerate(gene_names)}

    results = {}

    # === Known-conserved markers: expect high alpha ===
    for category, genes in KNOWN_CONSERVED_MARKERS.items():
        present = [g for g in genes if g in name_to_idx]
        if not present:
            results[f"conserved_{category}"] = {"status": "no_genes_found", "genes_checked": genes}
            continue
        indices = [name_to_idx[g] for g in present]
        cat_alpha = alpha[indices]
        cat_delta = delta[indices]
        results[f"conserved_{category}"] = {
            "genes_found": present,
            "n_found": len(present),
            "mean_alpha": float(cat_alpha.mean()),
            "mean_delta": float(cat_delta.mean()),
            "per_gene": {g: {"alpha": float(alpha[name_to_idx[g]]),
                            "delta": float(delta[name_to_idx[g]])}
                        for g in present},
            "pass": float(cat_alpha.mean()) > float(np.median(alpha)),
        }

    # === Known-divergent families: expect low alpha ===
    for category, genes in KNOWN_DIVERGENT_FAMILIES.items():
        present = [g for g in genes if g in name_to_idx]
        if not present:
            results[f"divergent_{category}"] = {"status": "no_genes_found", "genes_checked": genes}
            continue
        indices = [name_to_idx[g] for g in present]
        cat_alpha = alpha[indices]
        cat_delta = delta[indices]
        results[f"divergent_{category}"] = {
            "genes_found": present,
            "n_found": len(present),
            "mean_alpha": float(cat_alpha.mean()),
            "mean_delta": float(cat_delta.mean()),
            "per_gene": {g: {"alpha": float(alpha[name_to_idx[g]]),
                            "delta": float(delta[name_to_idx[g]])}
                        for g in present},
            "pass": float(cat_alpha.mean()) < float(np.median(alpha)),
        }

    # === Statistical test: conserved vs divergent alpha distributions ===
    conserved_indices = []
    for genes in KNOWN_CONSERVED_MARKERS.values():
        conserved_indices.extend([name_to_idx[g] for g in genes if g in name_to_idx])
    divergent_indices = []
    for genes in KNOWN_DIVERGENT_FAMILIES.values():
        divergent_indices.extend([name_to_idx[g] for g in genes if g in name_to_idx])

    if conserved_indices and divergent_indices:
        alpha_conserved = alpha[conserved_indices]
        alpha_divergent = alpha[divergent_indices]
        stat, pval = mannwhitneyu(alpha_conserved, alpha_divergent, alternative="greater")
        results["conserved_vs_divergent_test"] = {
            "mann_whitney_U": float(stat),
            "p_value": float(pval),
            "conserved_mean_alpha": float(alpha_conserved.mean()),
            "divergent_mean_alpha": float(alpha_divergent.mean()),
            "pass": pval < 0.05,  # conserved genes should have significantly higher alpha
        }

    # === Delta_alpha outlier enrichment ===
    if gene_table is not None:
        # Top-20 positive delta_alpha: expected enrichment for tissue-specific
        # conserved markers (cell-state variation conserved across species)
        top_pos = gene_table.head(20)["gene"].tolist()

        # Top-20 negative delta_alpha: expected enrichment for immune/regulatory
        # divergence genes
        top_neg = gene_table.tail(20)["gene"].tolist()

        results["delta_alpha_outliers"] = {
            "top_20_positive": top_pos,
            "top_20_negative": top_neg,
            "note": "Cross-reference with GO enrichment (gseapy) for publication",
        }

    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Biological ground truth validation (Task 9.7)",
    )
    parser.add_argument(
        "--adata", type=str, required=True,
        help="Path to unified AnnData (.h5ad) with human_gene_name in var.",
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Path to saved Chimera model directory.",
    )
    parser.add_argument(
        "--gene-table", type=str, default=None,
        help="Path to gene_table CSV (from Task 9.4 alpha analysis) for "
             "delta-alpha outlier enrichment.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write JSON results. Prints to stdout if not provided.",
    )
    args = parser.parse_args()

    # --- load AnnData ---
    import anndata as ad

    logger.info("Loading AnnData from %s", args.adata)
    adata = ad.read_h5ad(args.adata)
    logger.info("AnnData: %d cells x %d genes", adata.n_obs, adata.n_vars)

    # --- load model ---
    # Import here to avoid hard dep at module level
    from chimera import ChimeraModel

    logger.info("Loading model from %s", args.model_dir)
    model = ChimeraModel.load(args.model_dir, adata=adata)

    # --- optional gene table ---
    gene_table = None
    if args.gene_table is not None:
        logger.info("Loading gene table from %s", args.gene_table)
        gene_table = pd.read_csv(args.gene_table)

    # --- run validation ---
    logger.info("Running biological ground truth validation ...")
    results = biological_ground_truth(model, adata, gene_table=gene_table)

    # --- summarise pass/fail ---
    n_pass = sum(1 for v in results.values() if isinstance(v, dict) and v.get("pass", False))
    n_fail = sum(1 for v in results.values() if isinstance(v, dict) and v.get("pass") is False)
    n_skip = sum(1 for v in results.values()
                 if isinstance(v, dict) and v.get("status") == "no_genes_found")
    logger.info("Validation summary: %d pass, %d fail, %d skipped (no genes found)",
                n_pass, n_fail, n_skip)

    if "conserved_vs_divergent_test" in results:
        test = results["conserved_vs_divergent_test"]
        logger.info(
            "Mann-Whitney U: %.2f, p=%.4g, conserved_mean=%.4f, divergent_mean=%.4f — %s",
            test["mann_whitney_U"], test["p_value"],
            test["conserved_mean_alpha"], test["divergent_mean_alpha"],
            "PASS" if test["pass"] else "FAIL",
        )

    # --- output ---
    output_str = json.dumps(results, indent=2)
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_str)
        logger.info("Results written to %s", args.output)
    else:
        print(output_str)

    return results


if __name__ == "__main__":
    main()
