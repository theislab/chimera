"""Ablation suite runner for Chimera v2.4 (Task 9.3).

Trains ChimeraModel under each named ablation configuration, collects
latent representations, alpha values, and delta_alpha, then saves results
as a CSV summary alongside per-config model checkpoints.

Usage
-----
    python -m scripts.run_ablations \
        --dataset data/my_adata.h5ad \
        --configs A,D,F,M,J,C,H \
        --output-dir results/ablations \
        --max-epochs 200 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Priority ablation order from the v2.4 ablation plan.
# Config names reference chimera._constants.ABLATION_CONFIGS.
PRIORITY_ORDER: List[str] = [
    "A",   # 1. Full v2.4 default
    "D",   # 2. No anchor (lambda_anchor=0)
    "B_prime",  # 3. Frozen alpha = conservation prior
    "M",   # 4. Per-gene learnable alpha (no GIN)
    "J",   # 5. No FQR (lambda_fit=0)
    "B",   # 6. No shared aux (lambda_aux=0)
    "F",   # 7. No adversarial training
    "C",   # 8. No L1 sparsity (lambda_res=0)
    "C_double_prime",  # 9. Random gene features (tests GIN utility)
    "G",   # 10. No z.detach on shared path
]

DEFAULT_CONFIGS = "A,D,B_prime,M,J,B,F"


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across numpy, torch, and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_basic_metrics(
    latent: np.ndarray,
    alpha: np.ndarray,
    delta_alpha: np.ndarray,
) -> Dict[str, float]:
    """Compute lightweight summary metrics from a single ablation run.

    Parameters
    ----------
    latent
        Latent representations of shape ``(N, d)``.
    alpha
        Per-gene alpha values of shape ``(G,)``.
    delta_alpha
        Per-gene delta_alpha (alpha - prior) of shape ``(G,)``.

    Returns
    -------
    dict
        Dictionary of metric name to float value.
    """
    metrics: Dict[str, float] = {}

    # Latent space stats
    per_dim_var = latent.var(axis=0)
    metrics["latent_mean_var"] = float(per_dim_var.mean())
    metrics["latent_collapsed_dims"] = int((per_dim_var < 0.1).sum())
    metrics["latent_effective_dims"] = int((per_dim_var >= 0.1).sum())

    # Alpha distribution
    metrics["alpha_mean"] = float(alpha.mean())
    metrics["alpha_std"] = float(alpha.std())
    metrics["alpha_frac_below_0.1"] = float((alpha < 0.1).mean())
    metrics["alpha_frac_above_0.9"] = float((alpha > 0.9).mean())

    # Delta-alpha two-sidedness
    metrics["delta_alpha_frac_positive"] = float((delta_alpha > 0).mean())
    metrics["delta_alpha_frac_negative"] = float((delta_alpha < 0).mean())
    metrics["delta_alpha_abs_mean"] = float(np.abs(delta_alpha).mean())

    return metrics


def run_single_ablation(
    adata: ad.AnnData,
    config_name: str,
    output_dir: str,
    max_epochs: int,
    seed: int,
) -> Dict[str, object]:
    """Train a single ablation configuration and return results.

    Parameters
    ----------
    adata
        AnnData object already registered via ``ChimeraModel.setup_anndata``.
    config_name
        Name of the ablation config (key in ``ABLATION_CONFIGS``).
    output_dir
        Directory to save model checkpoint and artifacts.
    max_epochs
        Maximum training epochs.
    seed
        Random seed (reset before each run for reproducibility).

    Returns
    -------
    dict
        Keys: ``config``, ``status``, ``wall_time``, ``metrics``, ``error``.
    """
    from chimera._model import ChimeraModel

    _set_seed(seed)

    config_dir = os.path.join(output_dir, f"config_{config_name}")
    os.makedirs(config_dir, exist_ok=True)

    result: Dict[str, object] = {
        "config": config_name,
        "status": "pending",
        "wall_time": 0.0,
        "metrics": {},
        "error": None,
    }

    t0 = time.time()

    try:
        logger.info("=== Starting ablation config %s ===", config_name)
        model = ChimeraModel.from_config(adata, config_name)

        model.train(max_epochs=max_epochs)

        # Extract representations
        latent = model.get_latent_representation()
        alpha = model.get_alpha()
        delta_alpha = model.get_delta_alpha()

        # Save model checkpoint
        model.save(config_dir, overwrite=True)

        # Save arrays for downstream analysis
        np.save(os.path.join(config_dir, "latent.npy"), latent)
        np.save(os.path.join(config_dir, "alpha.npy"), alpha)
        np.save(os.path.join(config_dir, "delta_alpha.npy"), delta_alpha)

        metrics = _compute_basic_metrics(latent, alpha, delta_alpha)
        result["metrics"] = metrics
        result["status"] = "success"

        logger.info(
            "Config %s completed: alpha_mean=%.4f, alpha_std=%.4f, "
            "effective_dims=%d",
            config_name,
            metrics["alpha_mean"],
            metrics["alpha_std"],
            metrics["latent_effective_dims"],
        )

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        logger.error(
            "Config %s FAILED: %s\n%s",
            config_name,
            exc,
            traceback.format_exc(),
        )

    result["wall_time"] = time.time() - t0
    return result


def run_ablation_suite(
    dataset_path: str,
    configs: Optional[List[str]] = None,
    output_dir: str = "results/ablations",
    max_epochs: int = 200,
    seed: int = 42,
    species_key: str = "species",
    batch_key: str = "batch",
) -> pd.DataFrame:
    """Run the full ablation suite and return a summary DataFrame.

    Parameters
    ----------
    dataset_path
        Path to an ``.h5ad`` file containing the AnnData.
    configs
        List of config names to run. If ``None``, uses ``PRIORITY_ORDER``.
    output_dir
        Root directory for saving results.
    max_epochs
        Maximum training epochs per configuration.
    seed
        Random seed for reproducibility.
    species_key
        Column in ``adata.obs`` identifying species.
    batch_key
        Column in ``adata.obs`` identifying batches.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with one row per config and columns for all
        collected metrics. Saved to ``{output_dir}/ablation_summary.csv``.
    """
    from chimera._model import ChimeraModel

    if configs is None:
        configs = list(PRIORITY_ORDER)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Loading dataset from %s", dataset_path)
    adata = ad.read_h5ad(dataset_path)

    logger.info("Setting up AnnData for ChimeraModel")
    ChimeraModel.setup_anndata(
        adata,
        species_key=species_key,
        batch_key=batch_key,
    )

    all_results: List[Dict[str, object]] = []

    for config_name in configs:
        result = run_single_ablation(
            adata=adata,
            config_name=config_name,
            output_dir=output_dir,
            max_epochs=max_epochs,
            seed=seed,
        )
        all_results.append(result)

    # Build summary DataFrame
    rows = []
    for r in all_results:
        row = {
            "config": r["config"],
            "status": r["status"],
            "wall_time_s": r["wall_time"],
            "error": r["error"] if r["error"] else "",
        }
        if isinstance(r["metrics"], dict):
            row.update(r["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save summary CSV
    csv_path = os.path.join(output_dir, "ablation_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Ablation summary saved to %s", csv_path)

    # Save full results (including error tracebacks) as JSON
    json_path = os.path.join(output_dir, "ablation_results.json")
    serialisable = []
    for r in all_results:
        entry = dict(r)
        entry["wall_time"] = float(entry["wall_time"])
        serialisable.append(entry)
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    logger.info("Full results JSON saved to %s", json_path)

    # Print summary table
    n_success = sum(1 for r in all_results if r["status"] == "success")
    n_failed = sum(1 for r in all_results if r["status"] == "failed")
    logger.info(
        "Ablation suite complete: %d/%d succeeded, %d failed",
        n_success,
        len(all_results),
        n_failed,
    )

    return df


def main() -> None:
    """CLI entry point for the ablation runner."""
    parser = argparse.ArgumentParser(
        description="Run Chimera v2.4 ablation configurations (Task 9.3)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to AnnData .h5ad file",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=DEFAULT_CONFIGS,
        help=(
            "Comma-separated config names to run "
            f"(default: {DEFAULT_CONFIGS})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablations",
        help="Directory to save results (default: results/ablations)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Maximum training epochs per config (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--species-key",
        type=str,
        default="species",
        help="Column in adata.obs for species (default: species)",
    )
    parser.add_argument(
        "--batch-key",
        type=str,
        default="batch",
        help="Column in adata.obs for batch (default: batch)",
    )

    args = parser.parse_args()

    config_list = [c.strip() for c in args.configs.split(",") if c.strip()]

    df = run_ablation_suite(
        dataset_path=args.dataset,
        configs=config_list,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        seed=args.seed,
        species_key=args.species_key,
        batch_key=args.batch_key,
    )

    print("\n=== Ablation Summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
