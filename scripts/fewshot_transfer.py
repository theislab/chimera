"""Few-shot species transfer test for Chimera v2.4 (Task 9.10).

Holds out one species, trains on the rest, evaluates cross-species
label transfer accuracy using each training species as encoder proxy.
GIN alpha is zero-shot (gene features only); encoder is few-shot via
proxy species.

Usage
-----
    python -m scripts.fewshot_transfer \
        --dataset data/processed/camex_pancreas.h5ad \
        --output-dir results/fewshot \
        --max-epochs 200 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def fewshot_species_transfer(
    adata,
    holdout_species: str,
    species_key: str = "species",
    batch_key: str = "batch",
    label_key: str = "cell_type",
    max_epochs: int = 200,
    seed: int = 42,
    **model_kwargs: Any,
) -> Dict[str, Any]:
    """Hold out one species, train on rest, evaluate label transfer.

    GIN alpha is zero-shot (gene features only); encoder is few-shot
    via proxy species.

    Parameters
    ----------
    adata
        Full multi-species AnnData (all species).
    holdout_species
        Species name to hold out entirely.
    species_key
        Column in ``adata.obs`` with species labels.
    batch_key
        Column in ``adata.obs`` with batch labels.
    label_key
        Column in ``adata.obs`` with cell-type labels for kNN evaluation.
    max_epochs
        Training epochs for the K-1 species model.
    seed
        Random seed for reproducibility.
    **model_kwargs
        Additional keyword arguments passed to ``ChimeraModel``.

    Returns
    -------
    dict
        Holdout species, best proxy, accuracy per proxy, sensitivity.
    """
    import scvi
    from chimera import ChimeraModel

    train_mask = (adata.obs[species_key] != holdout_species).values
    adata_train = adata[train_mask].copy()
    adata_test = adata[~train_mask].copy()
    train_species = list(adata_train.obs[species_key].unique())

    logger.info(
        "Training on %d species (%d cells), holding out %s (%d cells)",
        len(train_species), adata_train.n_obs, holdout_species, adata_test.n_obs,
    )

    # Train on K-1 species
    ChimeraModel.setup_anndata(adata_train, species_key=species_key, batch_key=batch_key)
    model = ChimeraModel(adata_train, **model_kwargs)
    scvi.settings.seed = seed
    model.train(max_epochs=max_epochs)

    # GIN alpha is zero-shot — gene features only, valid for held-out species
    alpha_zeroshot = model.get_alpha()  # (G,)
    logger.info("Zero-shot alpha: mean=%.3f, std=%.3f", alpha_zeroshot.mean(), alpha_zeroshot.std())

    # Get training-set latent and labels for kNN
    z_train = model.get_latent_representation(adata_train)
    y_train = adata_train.obs[label_key].values
    y_test = adata_test.obs[label_key].values

    # Evaluate ALL training species as proxy (mandatory for publication)
    proxy_results: Dict[str, float] = {}
    for proxy_sp in train_species:
        adata_proxy = adata_test.copy()
        adata_proxy.obs[species_key] = proxy_sp
        # Use first batch of proxy species for batch encoding
        proxy_batches = adata_train.obs.loc[
            adata_train.obs[species_key] == proxy_sp, batch_key
        ].unique()
        adata_proxy.obs[batch_key] = proxy_batches[0]

        # Register proxy AnnData with the model's data registry
        adata_proxy = model._validate_anndata(adata_proxy)
        z_test = model.get_latent_representation(adata_proxy)
        knn = KNeighborsClassifier(n_neighbors=15).fit(z_train, y_train)
        acc = float((knn.predict(z_test) == y_test).mean())
        proxy_results[proxy_sp] = acc
        logger.info("  Proxy %s: accuracy=%.3f", proxy_sp, acc)

    best_proxy = max(proxy_results, key=proxy_results.get)
    acc_range = max(proxy_results.values()) - min(proxy_results.values())
    return {
        "holdout": holdout_species,
        "best_proxy": best_proxy,
        "best_accuracy": proxy_results[best_proxy],
        "all_proxy_accuracies": proxy_results,
        "proxy_sensitivity": float(acc_range),
        "fewshot_claim_strong": acc_range < 0.10,
    }


def run_fewshot_battery(
    adata,
    species_key: str = "species",
    max_epochs: int = 200,
    seed: int = 42,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Leave-one-species-out for all species (close and distant hold-outs).

    Parameters
    ----------
    adata
        Full multi-species AnnData.
    species_key
        Column in ``adata.obs`` with species labels.
    max_epochs
        Training epochs per held-out model.
    seed
        Random seed.

    Returns
    -------
    dict
        Mapping from held-out species name to transfer results.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for sp in adata.obs[species_key].unique():
        logger.info("=== Holding out species: %s ===", sp)
        results[sp] = fewshot_species_transfer(
            adata, sp, species_key=species_key,
            max_epochs=max_epochs, seed=seed, **kwargs,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Few-shot species transfer test for Chimera v2.4 (Task 9.10)",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to prepared .h5ad")
    parser.add_argument("--output-dir", type=str, default="results/fewshot")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout", type=str, default=None,
        help="Single species to hold out (default: leave-one-out for all)",
    )
    args = parser.parse_args()

    import anndata as ad
    import scvi

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scvi.settings.seed = args.seed

    logger.info("Loading dataset: %s", args.dataset)
    adata = ad.read_h5ad(args.dataset)

    if args.holdout:
        result = fewshot_species_transfer(
            adata, args.holdout, max_epochs=args.max_epochs, seed=args.seed,
        )
        results = {args.holdout: result}
    else:
        results = run_fewshot_battery(
            adata, max_epochs=args.max_epochs, seed=args.seed,
        )

    out_path = out / "fewshot_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    for sp, res in results.items():
        logger.info(
            "Species %s: best_proxy=%s (acc=%.3f), sensitivity=%.3f, strong=%s",
            sp, res["best_proxy"], res["best_accuracy"],
            res["proxy_sensitivity"], res["fewshot_claim_strong"],
        )


if __name__ == "__main__":
    main()
