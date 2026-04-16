"""Optuna hyperparameter search runner for Chimera v2.4 (Task 9.5).

Samples from the search space defined in ``configs/optuna_search_space.json``
and trains Chimera models, using the validation loss as the optimisation
objective.

Usage
-----
    python -m scripts.run_optuna \
        --dataset data/my_adata.h5ad \
        --n-trials 50 \
        --output-dir results/optuna \
        --max-epochs 200 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import anndata as ad
import numpy as np
import optuna
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_SEARCH_SPACE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "optuna_search_space.json",
)


def load_search_space(path: str = _DEFAULT_SEARCH_SPACE_PATH) -> Dict[str, Any]:
    """Load the search space configuration from a JSON file.

    Parameters
    ----------
    path
        Path to the JSON search space file.

    Returns
    -------
    dict
        The ``"parameters"`` sub-dict from the JSON file.
    """
    with open(path) as f:
        config = json.load(f)
    return config["parameters"]


def sample_hyperparameters(
    trial: optuna.Trial,
    search_space: Dict[str, Any],
) -> Dict[str, Any]:
    """Sample hyperparameters from the search space for a single Optuna trial.

    Parameters
    ----------
    trial
        Optuna trial object.
    search_space
        Parameter definitions (the ``"parameters"`` dict from the JSON config).

    Returns
    -------
    dict
        Sampled hyperparameter values keyed by parameter name.
    """
    params: Dict[str, Any] = {}

    for name, spec in search_space.items():
        ptype = spec["type"]

        if ptype == "loguniform":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=True,
            )
        elif ptype == "uniform":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
            )
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif ptype == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
            )
        else:
            raise ValueError(
                f"Unknown parameter type '{ptype}' for parameter '{name}'"
            )

    return params


def create_objective(
    adata: ad.AnnData,
    search_space: Dict[str, Any],
    max_epochs: int = 200,
    seed: int = 42,
) -> callable:
    """Create an Optuna objective function that trains Chimera and returns validation loss.

    Parameters
    ----------
    adata
        AnnData object already registered via ``ChimeraModel.setup_anndata``.
    search_space
        Parameter definitions from the JSON config.
    max_epochs
        Maximum training epochs per trial.
    seed
        Base random seed (offset by trial number for per-trial reproducibility).

    Returns
    -------
    callable
        Objective function compatible with ``optuna.Study.optimize``.
    """
    from chimera._model import ChimeraModel

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial, search_space)

        trial_seed = seed + trial.number
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)

        # Separate model constructor kwargs from training kwargs
        model_kwargs = {
            "lambda_aux": params["lambda_aux"],
            "lambda_res": params["lambda_res"],
            "lambda_anchor": params["lambda_anchor"],
            "lambda_fit": params["lambda_fit"],
            "n_latent": int(params["n_latent"]),
            "n_hidden": int(params["n_hidden"]),
        }

        # Training plan kwargs
        plan_kwargs = {}
        if "adv_weight" in params:
            plan_kwargs["adv_weight"] = params["adv_weight"]

        # Build model with the "A" (full v2.4) config as baseline,
        # then override with sampled hyperparameters
        model = ChimeraModel(adata, **model_kwargs)

        # Train with the sampled learning rate
        train_kwargs: Dict[str, Any] = {
            "max_epochs": max_epochs,
            "early_stopping": True,
        }
        if plan_kwargs:
            train_kwargs["plan_kwargs"] = plan_kwargs

        model.train(**train_kwargs)

        # Use the model's training history for the objective.
        # scvi-tools stores training history in model.history_.
        history = getattr(model, "history_", {})
        val_loss_key = None
        for candidate in ["validation_loss", "elbo_validation"]:
            if candidate in history:
                val_loss_key = candidate
                break

        if val_loss_key is not None:
            val_losses = history[val_loss_key]
            if hasattr(val_losses, "values"):
                val_losses = val_losses.values
            val_array = np.array(val_losses).ravel()
            final_val_loss = float(val_array[-1])
        else:
            # Fallback: compute reconstruction loss on the latent representation
            logger.warning(
                "Trial %d: no validation loss in history, "
                "falling back to final training loss",
                trial.number,
            )
            train_loss_key = None
            for candidate in ["train_loss_epoch", "training_loss"]:
                if candidate in history:
                    train_loss_key = candidate
                    break
            if train_loss_key is not None:
                train_losses = history[train_loss_key]
                if hasattr(train_losses, "values"):
                    train_losses = train_losses.values
                final_val_loss = float(np.array(train_losses).ravel()[-1])
            else:
                final_val_loss = float("inf")

        if math.isnan(final_val_loss) or math.isinf(final_val_loss):
            raise optuna.TrialPruned(
                f"Trial {trial.number} produced NaN/Inf validation loss"
            )

        logger.info(
            "Trial %d: val_loss=%.4f | params=%s",
            trial.number,
            final_val_loss,
            {k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in params.items()},
        )

        return final_val_loss

    return objective


def create_study(
    dataset_path: str,
    search_space_path: str = _DEFAULT_SEARCH_SPACE_PATH,
    n_trials: int = 50,
    output_dir: str = "results/optuna",
    max_epochs: int = 200,
    seed: int = 42,
    species_key: str = "species",
    batch_key: str = "batch",
    study_name: str = "chimera_v2.4_hpo",
    storage: Optional[str] = None,
) -> optuna.Study:
    """Create and run an Optuna study for Chimera hyperparameter optimisation.

    Parameters
    ----------
    dataset_path
        Path to an ``.h5ad`` file containing the AnnData.
    search_space_path
        Path to the JSON search space configuration file.
    n_trials
        Number of Optuna trials to run.
    output_dir
        Directory to save study results.
    max_epochs
        Maximum training epochs per trial.
    seed
        Random seed for reproducibility.
    species_key
        Column in ``adata.obs`` identifying species.
    batch_key
        Column in ``adata.obs`` identifying batches.
    study_name
        Name for the Optuna study.
    storage
        Optuna storage URL (e.g. ``sqlite:///optuna.db``). If ``None``,
        uses in-memory storage.

    Returns
    -------
    optuna.Study
        The completed Optuna study object.
    """
    from chimera._model import ChimeraModel

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading dataset from %s", dataset_path)
    adata = ad.read_h5ad(dataset_path)

    logger.info("Setting up AnnData for ChimeraModel")
    ChimeraModel.setup_anndata(
        adata,
        species_key=species_key,
        batch_key=batch_key,
    )

    search_space = load_search_space(search_space_path)
    logger.info(
        "Loaded search space with %d parameters from %s",
        len(search_space),
        search_space_path,
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=50,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    objective = create_objective(
        adata=adata,
        search_space=search_space,
        max_epochs=max_epochs,
        seed=seed,
    )

    logger.info("Starting Optuna study '%s' with %d trials", study_name, n_trials)
    study.optimize(objective, n_trials=n_trials)

    # Save results
    best_trial = study.best_trial
    logger.info(
        "Best trial %d: val_loss=%.4f | params=%s",
        best_trial.number,
        best_trial.value,
        best_trial.params,
    )

    # Save best params
    best_params_path = os.path.join(output_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(
            {
                "trial_number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
            },
            f,
            indent=2,
        )
    logger.info("Best params saved to %s", best_params_path)

    # Save all trial results as CSV
    trials_df = study.trials_dataframe()
    trials_csv_path = os.path.join(output_dir, "all_trials.csv")
    trials_df.to_csv(trials_csv_path, index=False)
    logger.info("All trial results saved to %s", trials_csv_path)

    return study


def main() -> None:
    """CLI entry point for the Optuna hyperparameter search."""
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for Chimera v2.4 (Task 9.5)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to AnnData .h5ad file",
    )
    parser.add_argument(
        "--search-space",
        type=str,
        default=_DEFAULT_SEARCH_SPACE_PATH,
        help="Path to search space JSON config",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/optuna",
        help="Directory to save results (default: results/optuna)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Maximum training epochs per trial (default: 200)",
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
    parser.add_argument(
        "--study-name",
        type=str,
        default="chimera_v2.4_hpo",
        help="Optuna study name (default: chimera_v2.4_hpo)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///optuna.db (default: in-memory)",
    )

    args = parser.parse_args()

    study = create_study(
        dataset_path=args.dataset,
        search_space_path=args.search_space,
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        seed=args.seed,
        species_key=args.species_key,
        batch_key=args.batch_key,
        study_name=args.study_name,
        storage=args.storage,
    )

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
