"""Train the baseline ladder for Chimera v2.4 ablation experiments.

Models trained:
  1. scVI       -- species treated as batch
  2. Harmony    -- on PCA of log-normalized counts
  3. scANVI     -- semi-supervised (if cell_type annotations present)
  4. Chimera D  -- full v2.4 without anchor (ABLATION_CONFIGS["D"])
  5. Chimera F  -- full v2.4 without adversarial (ABLATION_CONFIGS["F"])

After training, computes post-hoc per-gene scVI F-statistics and
correlates them with Chimera alpha via Spearman rank correlation.

All latent representations are saved into adata.obsm and the updated
AnnData is written back to disk.  Models are saved to subdirectories
under --output-dir.  Optionally logs to Weights & Biases.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr, f_oneway

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _try_wandb_init(project: str, config: dict):
    """Attempt to initialise a wandb run; return None on failure."""
    try:
        import wandb

        run = wandb.init(project=project, config=config, reinit=True)
        return run
    except Exception as exc:
        logger.warning("wandb unavailable or init failed (%s); skipping logging.", exc)
        return None


def _wandb_log(run, metrics: dict):
    """Log metrics to wandb if the run is active."""
    if run is not None:
        try:
            import wandb

            wandb.log(metrics)
        except Exception:
            pass


def _wandb_finish(run):
    """Finish wandb run if active."""
    if run is not None:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass


def _set_seed(seed: int):
    """Set global random seeds for reproducibility."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Baseline trainers
# ------------------------------------------------------------------


def _train_scvi(adata: ad.AnnData, output_dir: str, max_epochs: int = 200):
    """Train scVI treating species as batch variable; return latent key name."""
    import scvi as scvi_pkg

    logger.info("=== Training scVI (species-as-batch) ===")
    adata_scvi = adata.copy()

    scvi_pkg.model.SCVI.setup_anndata(adata_scvi, batch_key="species")
    model = scvi_pkg.model.SCVI(adata_scvi, n_latent=10)
    model.train(max_epochs=max_epochs, early_stopping=True)

    latent = model.get_latent_representation()
    adata.obsm["X_scVI"] = latent

    model_dir = os.path.join(output_dir, "scvi_model")
    model.save(model_dir, overwrite=True)
    logger.info("scVI model saved to %s", model_dir)

    return model


def _train_harmony(adata: ad.AnnData):
    """Run Harmony on PCA of log-normalized counts; store in adata.obsm."""
    import harmonypy

    logger.info("=== Running Harmony ===")
    adata_harmony = adata.copy()

    # Log-normalise and PCA
    sc.pp.normalize_total(adata_harmony, target_sum=1e4)
    sc.pp.log1p(adata_harmony)
    sc.pp.highly_variable_genes(adata_harmony, n_top_genes=2000, subset=True)
    sc.tl.pca(adata_harmony, n_comps=50)

    pca_matrix = adata_harmony.obsm["X_pca"]
    meta = adata_harmony.obs[["species"]].copy()

    ho = harmonypy.run_harmony(pca_matrix, meta, "species")
    z_corr = ho.Z_corr.T if ho.Z_corr.shape[0] != adata.n_obs else ho.Z_corr
    adata.obsm["X_Harmony"] = z_corr

    logger.info("Harmony complete; stored in obsm['X_Harmony'].")


def _train_scanvi(adata: ad.AnnData, output_dir: str, max_epochs: int = 200):
    """Train scANVI (semi-supervised); return model or None if no cell_type."""
    if "cell_type" not in adata.obs.columns:
        logger.info("Skipping scANVI: no cell_type column in adata.obs.")
        return None

    import scvi as scvi_pkg

    logger.info("=== Training scANVI ===")
    adata_scanvi = adata.copy()

    scvi_pkg.model.SCVI.setup_anndata(adata_scanvi, batch_key="species")
    scvi_model = scvi_pkg.model.SCVI(adata_scanvi, n_latent=10)
    scvi_model.train(max_epochs=max_epochs, early_stopping=True)

    scanvi_model = scvi_pkg.model.SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category="Unknown",
        labels_key="cell_type",
    )
    scanvi_model.train(max_epochs=max_epochs // 2, early_stopping=True)

    latent = scanvi_model.get_latent_representation()
    adata.obsm["X_scANVI"] = latent

    model_dir = os.path.join(output_dir, "scanvi_model")
    scanvi_model.save(model_dir, overwrite=True)
    logger.info("scANVI model saved to %s", model_dir)

    return scanvi_model


def _train_chimera_config(
    adata: ad.AnnData,
    config_name: str,
    output_dir: str,
    max_epochs: int = 200,
):
    """Train a Chimera model from a named ablation config; return model."""
    from chimera._model import ChimeraModel

    logger.info("=== Training Chimera Config %s ===", config_name)

    ChimeraModel.setup_anndata(adata, species_key="species", batch_key="batch")
    model = ChimeraModel.from_config(adata, config_name)
    model.train(max_epochs=max_epochs)

    latent = model.get_latent_representation()
    obsm_key = f"X_Chimera_{config_name}"
    adata.obsm[obsm_key] = latent

    model_dir = os.path.join(output_dir, f"chimera_{config_name}_model")
    model.save(model_dir, overwrite=True)
    logger.info("Chimera %s model saved to %s", config_name, model_dir)

    return model


# ------------------------------------------------------------------
# Post-hoc scVI per-gene F-statistic
# ------------------------------------------------------------------


def compute_scvi_f_statistics(adata: ad.AnnData, scvi_model) -> np.ndarray:
    """Compute per-gene F-statistics from the scVI latent space.

    For each gene, runs a one-way ANOVA across species in the scVI
    denoised expression space.  Returns an array of F-statistics of
    shape (n_genes,).
    """
    logger.info("Computing per-gene scVI F-statistics ...")

    denoised = scvi_model.get_normalized_expression(library_size=1e4)
    if hasattr(denoised, 'values'):
        denoised = denoised.values  # DataFrame -> ndarray
    species = adata.obs["species"].values
    unique_species = np.unique(species)
    n_genes = denoised.shape[1]

    f_stats = np.zeros(n_genes, dtype=np.float64)

    for g in range(n_genes):
        groups = [denoised[species == sp, g] for sp in unique_species]
        # Filter out groups with fewer than 2 observations
        groups = [gr for gr in groups if len(gr) >= 2]
        if len(groups) >= 2:
            stat, _ = f_oneway(*groups)
            f_stats[g] = stat if np.isfinite(stat) else 0.0
        else:
            f_stats[g] = 0.0

    return f_stats


def compare_fstats_with_alpha(
    f_stats: np.ndarray,
    alpha: np.ndarray,
) -> dict:
    """Spearman correlation between per-gene scVI F-stats and Chimera alpha.

    High F-stat means high between-species variance (divergent gene).
    Low alpha means the model treats the gene as divergent.
    Expectation: negative Spearman correlation.
    """
    mask = np.isfinite(f_stats) & np.isfinite(alpha) & (f_stats > 0)
    if mask.sum() < 10:
        logger.warning(
            "Only %d valid genes for Spearman correlation; result unreliable.",
            mask.sum(),
        )
        if mask.sum() < 3:
            return {"spearman_r": float("nan"), "p_value": float("nan"), "n_genes": int(mask.sum())}

    rho, pval = spearmanr(f_stats[mask], alpha[mask])
    logger.info(
        "Spearman(scVI F-stat, Chimera alpha) = %.4f (p=%.2e, n=%d genes)",
        rho,
        pval,
        mask.sum(),
    )
    return {"spearman_r": float(rho), "p_value": float(pval), "n_genes": int(mask.sum())}


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def train_baseline_ladder(
    dataset_path: str,
    output_dir: str,
    seed: int = 42,
    max_epochs: int = 200,
    wandb_project: str = "chimera-baselines",
):
    """Train the full baseline ladder and save results.

    Parameters
    ----------
    dataset_path
        Path to a prepared AnnData .h5ad file.
    output_dir
        Directory for saving models, latent representations, and metrics.
    seed
        Random seed for reproducibility.
    max_epochs
        Maximum training epochs per model.
    wandb_project
        Weights & Biases project name (logging is best-effort).
    """
    _set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading dataset from %s", dataset_path)
    adata = ad.read_h5ad(dataset_path)
    logger.info("Dataset: %d cells x %d genes", adata.n_obs, adata.n_vars)

    run = _try_wandb_init(wandb_project, {"seed": seed, "max_epochs": max_epochs})

    results = {}

    # --- 1. scVI ---
    scvi_model = _train_scvi(adata, output_dir, max_epochs=max_epochs)
    results["scVI"] = "trained"
    _wandb_log(run, {"scVI": "complete"})

    # --- 2. Harmony ---
    _train_harmony(adata)
    results["Harmony"] = "trained"
    _wandb_log(run, {"Harmony": "complete"})

    # --- 3. scANVI (optional) ---
    scanvi_model = _train_scanvi(adata, output_dir, max_epochs=max_epochs)
    results["scANVI"] = "trained" if scanvi_model is not None else "skipped"
    _wandb_log(run, {"scANVI": results["scANVI"]})

    # --- 4. Chimera Config D (no anchor) ---
    chimera_d = _train_chimera_config(adata, "D", output_dir, max_epochs=max_epochs)
    results["Chimera_D"] = "trained"
    _wandb_log(run, {"Chimera_D": "complete"})

    # --- 5. Chimera Config F (no adversarial) ---
    chimera_f = _train_chimera_config(adata, "F", output_dir, max_epochs=max_epochs)
    results["Chimera_F"] = "trained"
    _wandb_log(run, {"Chimera_F": "complete"})

    # --- Post-hoc: scVI F-stats vs Chimera alpha ---
    f_stats = compute_scvi_f_statistics(adata, scvi_model)
    alpha_d = chimera_d.get_alpha()

    comparison = compare_fstats_with_alpha(f_stats, alpha_d)
    results["fstat_alpha_comparison"] = comparison
    _wandb_log(run, comparison)

    logger.info(
        "F-stat vs alpha Spearman: r=%.4f, p=%.2e",
        comparison["spearman_r"],
        comparison["p_value"],
    )

    # --- Save outputs ---
    adata_out_path = os.path.join(output_dir, "adata_with_latents.h5ad")
    adata.write(adata_out_path)
    logger.info("AnnData with latent representations saved to %s", adata_out_path)

    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results summary saved to %s", results_path)

    _wandb_finish(run)
    return results


def main():
    """CLI entry point for train_baselines."""
    parser = argparse.ArgumentParser(
        description="Train the baseline ladder for Chimera v2.4 ablation experiments.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to prepared AnnData .h5ad file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/baselines",
        help="Directory for saving models and results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Maximum training epochs per model.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="chimera-baselines",
        help="Weights & Biases project name.",
    )

    args = parser.parse_args()

    train_baseline_ladder(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        seed=args.seed,
        max_epochs=args.max_epochs,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
