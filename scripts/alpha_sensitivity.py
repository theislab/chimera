"""Alpha hyperparameter sensitivity test for Chimera v2.4 (Task 9.11).

Tests alpha ranking robustness to 2x lambda perturbations and multi-seed
stability.  Key publishability gate: Spearman r > 0.8 across perturbations.

Round2: ortholog_aware_integration_guide_initial.md, Component 12 sixth-pass
"Alpha as relative metric"; publishability criteria item 3.

Usage
-----
    python -m scripts.alpha_sensitivity \\
        --dataset data/processed/camex_pancreas.h5ad \\
        --output-dir results/sensitivity \\
        --n-seeds 3 \\
        --max-epochs 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def alpha_hyperparameter_sensitivity(
    adata,
    species_key: str = "species",
    batch_key: str = "batch",
    n_seeds: int = 3,
    base_anchor: float = 0.005,
    max_epochs: int = 200,
) -> Tuple[Dict[str, Any], Dict[Tuple[str, int], np.ndarray]]:
    """Test alpha ranking robustness to 2x lambda perturbations.

    Trains Config D with default, 2x, and 0.5x ``lambda_anchor`` across
    multiple seeds.  Compares gene rankings by alpha across settings via
    Spearman rank correlation.

    Parameters
    ----------
    adata
        Prepared multi-species AnnData.
    species_key, batch_key
        Obs column names.
    n_seeds
        Number of random seeds per setting.
    base_anchor
        Default ``lambda_anchor`` value (0.005 from v2.4 spec).
    max_epochs
        Training epochs per run.

    Returns
    -------
    results : dict
        Cross-setting Spearman r and multi-seed stability statistics.
    all_alphas : dict
        Mapping ``(setting_name, seed) -> alpha_array``.
    """
    import scvi
    from chimera import ChimeraModel

    settings = {
        "default": {"lambda_anchor": base_anchor},
        "2x_anchor": {"lambda_anchor": base_anchor * 2},
        "0.5x_anchor": {"lambda_anchor": base_anchor * 0.5},
    }

    all_alphas: Dict[Tuple[str, int], np.ndarray] = {}
    for name, overrides in settings.items():
        for seed in range(n_seeds):
            logger.info("Training: setting=%s, seed=%d", name, seed)
            adata_run = adata.copy()
            ChimeraModel.setup_anndata(
                adata_run, species_key=species_key, batch_key=batch_key,
            )
            model = ChimeraModel(adata_run, **overrides)
            scvi.settings.seed = seed
            model.train(max_epochs=max_epochs)
            all_alphas[(name, seed)] = model.get_alpha()

    results: Dict[str, Any] = {}

    # Cross-setting ranking comparison (Spearman of alpha vectors)
    for s1, s2 in [("default", "2x_anchor"), ("default", "0.5x_anchor")]:
        rs = []
        for seed in range(n_seeds):
            r, _ = spearmanr(all_alphas[(s1, seed)], all_alphas[(s2, seed)])
            rs.append(r)
        results[f"{s1}_vs_{s2}"] = {
            "mean_spearman_r": float(np.mean(rs)),
            "min_spearman_r": float(np.min(rs)),
            "robust": float(np.mean(rs)) > 0.8,
        }
        logger.info(
            "%s vs %s: mean_r=%.3f, robust=%s",
            s1, s2, np.mean(rs), np.mean(rs) > 0.8,
        )

    # Multi-seed stability (same setting, different seeds)
    for name in settings:
        seed_alphas = np.stack([all_alphas[(name, s)] for s in range(n_seeds)])
        std_per_gene = seed_alphas.std(axis=0)
        results[f"stability_{name}"] = {
            "mean_alpha_std": float(std_per_gene.mean()),
            "frac_unreliable": float((std_per_gene > 0.05).mean()),
            "n_unreliable_genes": int((std_per_gene > 0.05).sum()),
        }

    return results, all_alphas


def extended_multiseed_stability(
    adata,
    n_seeds: int = 5,
    species_key: str = "species",
    batch_key: str = "batch",
    max_epochs: int = 200,
) -> Dict[str, Any]:
    """Extended 5-seed stability for publishability gate.

    Trains Config D with ``n_seeds`` different seeds and reports per-gene
    alpha standard deviation.  Genes with std > 0.05 are flagged as
    "alpha-unreliable".

    Parameters
    ----------
    adata
        Prepared multi-species AnnData.
    n_seeds
        Number of random seeds (default 5 for publishability gate).
    species_key, batch_key
        Obs column names.
    max_epochs
        Training epochs per run.

    Returns
    -------
    dict
        Mean/median alpha std, fraction unreliable, unreliable gene indices.
    """
    import scvi
    from chimera import ChimeraModel

    alphas = []
    for seed in range(n_seeds):
        logger.info("Extended stability: seed=%d/%d", seed + 1, n_seeds)
        adata_run = adata.copy()
        ChimeraModel.setup_anndata(
            adata_run, species_key=species_key, batch_key=batch_key,
        )
        model = ChimeraModel.from_config(adata_run, "D")
        scvi.settings.seed = seed
        model.train(max_epochs=max_epochs)
        alphas.append(model.get_alpha())

    alphas_arr = np.stack(alphas)
    std_per_gene = alphas_arr.std(axis=0)

    # Pairwise Spearman between all seed pairs
    pairwise_r = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            r, _ = spearmanr(alphas[i], alphas[j])
            pairwise_r.append(r)

    result = {
        "n_seeds": n_seeds,
        "mean_std": float(std_per_gene.mean()),
        "median_std": float(np.median(std_per_gene)),
        "frac_unreliable": float((std_per_gene > 0.05).mean()),
        "n_unreliable_genes": int((std_per_gene > 0.05).sum()),
        "unreliable_gene_indices": [int(i) for i in np.where(std_per_gene > 0.05)[0]],
        "mean_pairwise_spearman_r": float(np.mean(pairwise_r)),
        "min_pairwise_spearman_r": float(np.min(pairwise_r)),
    }
    logger.info(
        "Extended stability: mean_std=%.4f, frac_unreliable=%.3f, "
        "mean_pairwise_r=%.3f",
        result["mean_std"], result["frac_unreliable"],
        result["mean_pairwise_spearman_r"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alpha hyperparameter sensitivity test (Task 9.11)",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to prepared .h5ad")
    parser.add_argument("--output-dir", type=str, default="results/sensitivity")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument(
        "--extended", action="store_true",
        help="Run extended 5-seed stability (publishability gate)",
    )
    args = parser.parse_args()

    import anndata as ad

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset: %s", args.dataset)
    adata = ad.read_h5ad(args.dataset)

    if args.extended:
        result = extended_multiseed_stability(
            adata, n_seeds=5, max_epochs=args.max_epochs,
        )
        out_path = out / "extended_stability.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
    else:
        result, all_alphas = alpha_hyperparameter_sensitivity(
            adata, n_seeds=args.n_seeds, max_epochs=args.max_epochs,
        )
        # Save alpha arrays
        alpha_dict = {
            f"{k[0]}_seed{k[1]}": v for k, v in all_alphas.items()
        }
        np.savez(out / "all_alphas.npz", **alpha_dict)
        out_path = out / "sensitivity_results.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
