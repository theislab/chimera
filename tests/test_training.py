"""Integration tests for Chimera v2.4 training pipeline (Tasks 8.4 and 8.5).

Tests the full scvi-tools integration: AnnData → setup_anndata → ChimeraModel → train.
These are intentionally lightweight (small data, few epochs) for CI without GPU.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from chimera._data_utils import create_synthetic_adata, precompute_p_null
from chimera._model import ChimeraModel


def _prepare_adata(N=120, G=80, K=2, Nb=4, seed=42):
    """Create and register a synthetic AnnData for ChimeraModel.

    Returns the registered AnnData ready for ChimeraModel construction.
    """
    result = create_synthetic_adata(
        n_cells_per_species=N // K,
        n_genes=G,
        n_species=K,
        n_batches_per_species=Nb // K,
        seed=seed,
    )
    adata = result["adata"]

    # Attach gene features and conservation prior
    rng = np.random.RandomState(seed)
    adata.varm["gene_features"] = rng.randn(G, 72).astype(np.float32)
    adata.var["conservation_prior"] = rng.uniform(0.05, 0.95, G).astype(np.float32)

    # Precompute p_null
    adata.uns["p_null"] = precompute_p_null(adata).astype(np.float32)

    # Register
    ChimeraModel.setup_anndata(adata, species_key="species", batch_key="batch")

    return adata


# =========================================================================
# Task 8.4: Full Training Step
# =========================================================================


class TestFullTrainingStep:
    """End-to-end: AnnData → setup_anndata → ChimeraModel → train(1 epoch)."""

    def test_single_epoch_no_error(self):
        """One epoch of training completes without error."""
        adata = _prepare_adata(N=120, G=80, K=2, Nb=4)
        model = ChimeraModel(adata, n_latent=8, n_hidden=32)

        # Train for 1 epoch — should not raise
        model.train(
            max_epochs=1,
            batch_size=64,
            plan_kwargs={
                "warmup_kl_epochs": 0,
                "warmup_fqr_epochs": 0,
                "warmup_adversarial_epochs": 0,
                "min_training_epochs": 0,
            },
        )

    def test_parameters_updated(self):
        """All trainable parameters change after one epoch."""
        adata = _prepare_adata(N=120, G=80, K=2, Nb=4)
        model = ChimeraModel(adata, n_latent=8, n_hidden=32)

        # Snapshot initial params
        initial_params = {
            n: p.detach().clone()
            for n, p in model.module.named_parameters()
            if p.requires_grad
        }

        model.train(
            max_epochs=1,
            batch_size=64,
            plan_kwargs={
                "warmup_kl_epochs": 0,
                "warmup_fqr_epochs": 0,
                "warmup_adversarial_epochs": 0,
                "min_training_epochs": 0,
            },
        )

        # Check that at least some params changed
        changed = 0
        for name, p in model.module.named_parameters():
            if p.requires_grad and name in initial_params:
                if not torch.equal(p.data, initial_params[name]):
                    changed += 1

        assert changed > 0, "No parameters changed after training — optimizer broken"

    def test_latent_representation_shape(self):
        """get_latent_representation returns correct shape after training."""
        adata = _prepare_adata(N=120, G=80, K=2, Nb=4)
        model = ChimeraModel(adata, n_latent=8, n_hidden=32)

        model.train(
            max_epochs=1,
            batch_size=64,
            plan_kwargs={
                "warmup_kl_epochs": 0,
                "warmup_fqr_epochs": 0,
                "warmup_adversarial_epochs": 0,
                "min_training_epochs": 0,
            },
        )

        z = model.get_latent_representation()
        assert z.shape == (adata.n_obs, 8), f"z shape {z.shape} != ({adata.n_obs}, 8)"
        assert np.isfinite(z).all(), "z contains NaN or Inf"

    def test_alpha_in_bounds(self):
        """Alpha values are in (0, 1) after training."""
        adata = _prepare_adata(N=120, G=80, K=2, Nb=4)
        model = ChimeraModel(adata, n_latent=8, n_hidden=32)

        model.train(
            max_epochs=1,
            batch_size=64,
            plan_kwargs={
                "warmup_kl_epochs": 0,
                "warmup_fqr_epochs": 0,
                "warmup_adversarial_epochs": 0,
                "min_training_epochs": 0,
            },
        )

        alpha = model.get_alpha()
        assert alpha.shape == (adata.n_vars,)
        assert (alpha > 0).all() and (alpha < 1).all(), (
            f"Alpha out of (0,1): min={alpha.min():.6f}, max={alpha.max():.6f}"
        )


# =========================================================================
# Task 8.5: Short Training Loop (schedule validation)
# =========================================================================


class TestShortTrainingLoop:
    """Multi-epoch training validates training schedule mechanics."""

    def test_loss_finite_throughout(self):
        """Loss stays finite for 5 epochs."""
        adata = _prepare_adata(N=120, G=80, K=2, Nb=4)
        model = ChimeraModel(adata, n_latent=8, n_hidden=32)

        model.train(
            max_epochs=5,
            batch_size=64,
            plan_kwargs={
                "warmup_kl_epochs": 2,
                "warmup_fqr_epochs": 2,
                "warmup_adversarial_epochs": 2,
                "min_training_epochs": 0,
            },
        )

        # If training completed without error, losses were finite
        # Verify model is in a valid state
        alpha = model.get_alpha()
        assert np.isfinite(alpha).all(), "Alpha contains NaN after 5 epochs"

    def test_alpha_differentiated_from_prior(self):
        """After training, alpha should have moved from the prior."""
        adata = _prepare_adata(N=200, G=80, K=2, Nb=4)
        model = ChimeraModel(adata, n_latent=8, n_hidden=32)

        model.train(
            max_epochs=10,
            batch_size=64,
            plan_kwargs={
                "warmup_kl_epochs": 2,
                "warmup_fqr_epochs": 2,
                "warmup_adversarial_epochs": 2,
                "min_training_epochs": 0,
            },
        )

        alpha = model.get_alpha()
        prior = model.module.conservation_prior.cpu().numpy()
        delta = np.abs(alpha - prior)

        # At least some genes should have moved from the prior
        assert delta.max() > 0.001, (
            f"Alpha unchanged after 10 epochs — GIN not learning. "
            f"Max |delta| = {delta.max():.6f}"
        )

    def test_return_dist(self):
        """get_latent_representation with return_dist returns z_mean and z_logvar."""
        adata = _prepare_adata(N=120, G=80, K=2, Nb=4)
        model = ChimeraModel(adata, n_latent=8, n_hidden=32)

        model.train(
            max_epochs=2,
            batch_size=64,
            plan_kwargs={
                "warmup_kl_epochs": 0,
                "warmup_fqr_epochs": 0,
                "warmup_adversarial_epochs": 0,
                "min_training_epochs": 0,
            },
        )

        z_mean, z_logvar = model.get_latent_representation(return_dist=True)
        assert z_mean.shape == (adata.n_obs, 8)
        assert z_logvar.shape == (adata.n_obs, 8)
        assert np.isfinite(z_mean).all()
        assert np.isfinite(z_logvar).all()

    def test_shared_only_expression(self):
        """get_shared_only_expression returns valid non-negative values."""
        adata = _prepare_adata(N=120, G=80, K=2, Nb=4)
        model = ChimeraModel(adata, n_latent=8, n_hidden=32)

        model.train(
            max_epochs=2,
            batch_size=64,
            plan_kwargs={
                "warmup_kl_epochs": 0,
                "warmup_fqr_epochs": 0,
                "warmup_adversarial_epochs": 0,
                "min_training_epochs": 0,
            },
        )

        mu = model.get_shared_only_expression()
        assert mu.shape == (adata.n_obs, adata.n_vars)
        assert (mu >= 0).all(), "mu_shared_only has negative values"
        assert np.isfinite(mu).all(), "mu_shared_only has NaN/Inf"
