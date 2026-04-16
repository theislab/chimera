"""Task 4.8: Loss Scale Calibration Test.

The most critical test in the suite. It catches the v2.2 bug that silently
disabled the L1 and anchor by using `.mean()` instead of `.sum(-1).mean()`.

The v2.2 bug: using `.mean()` on a (B, G) tensor gives a per-element average,
which dilutes the penalty by a factor of G (n_genes). The correct reduction is
`.sum(-1).mean()` -- sum over genes per cell, then average over cells. With
G=200 genes, the buggy reduction makes the L1 and anchor terms ~200x weaker
than intended, effectively disabling them.

Three-phase test:
  Phase 1: After 1 forward pass (untrained model), weighted L1 must be
           > 0.0002 * L_recon (catches .mean() vs .sum(-1).mean();
           correct gives ~0.0007, bug gives ~0.000004).
  Phase 2: After 10 training steps, GIN correction must have drifted from
           zero, anchor must use .sum() reduction, and weighted L1 must
           remain meaningful.
  Phase 3: After 50 more steps with FQR enabled (current_epoch past warmup),
           S_g should have positive fraction > 0.05 and FQR weighted by
           LAMBDA_FIT must contribute meaningfully (> 0.0001 * L_recon).
"""

import numpy as np
import scipy.sparse as sp
import torch

from chimera._constants import (
    LAMBDA_ANCHOR,
    LAMBDA_FIT,
    LAMBDA_RES,
    WARMUP_FQR_EPOCHS,
)
from chimera._data_utils import create_synthetic_adata, precompute_p_null
from chimera._module import ChimeraModule


def _build_module_and_batch():
    """Construct a ChimeraModule and a raw tensor batch from synthetic data.

    Returns (module, batch_dict) where batch_dict has keys "X", "species",
    "batch", "library_size" -- all plain tensors, no scvi DataLoader.
    """
    result = create_synthetic_adata(
        n_cells_per_species=100,
        n_genes=200,
        n_species=3,
        n_batches_per_species=2,
        seed=42,
    )
    adata = result["adata"]
    n_genes = adata.n_vars
    n_species = len(adata.obs["species"].cat.categories)
    species_cats = list(adata.obs["species"].cat.categories)
    batch_cats = list(adata.obs["batch"].cat.categories)
    n_batch = len(batch_cats)

    # Build batch_to_species mapping
    batch_to_species = []
    for b in batch_cats:
        mask = adata.obs["batch"] == b
        s = adata.obs.loc[mask, "species"].iloc[0]
        batch_to_species.append(species_cats.index(s))
    batch_to_species_idx = torch.tensor(batch_to_species, dtype=torch.long)

    # Gene features
    gene_features = torch.tensor(result["gene_features"], dtype=torch.float32)

    # Conservation prior
    cp = result["conservation_prior"]
    if isinstance(cp, torch.Tensor):
        conservation_prior = cp.float()
    else:
        conservation_prior = torch.tensor(np.asarray(cp), dtype=torch.float32)

    # p_null
    p_null = torch.tensor(precompute_p_null(adata), dtype=torch.float32)

    # Construct module
    module = ChimeraModule(
        n_genes=n_genes,
        n_species=n_species,
        n_batch=n_batch,
        n_latent=10,
        n_hidden=128,
        gene_features=gene_features,
        conservation_prior=conservation_prior,
        p_null=p_null,
        batch_to_species_idx=batch_to_species_idx,
        lambda_aux=0.3,
        lambda_res=LAMBDA_RES,
        lambda_anchor=LAMBDA_ANCHOR,
        lambda_fit=LAMBDA_FIT,
        use_observed_lib_size=True,
    )

    # Build batch dict from raw tensors (no scvi DataLoader)
    X_raw = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    X = torch.tensor(X_raw, dtype=torch.float32)
    species = torch.tensor(
        adata.obs["species"].cat.codes.values, dtype=torch.long
    ).unsqueeze(-1)
    batch = torch.tensor(
        adata.obs["batch"].cat.codes.values, dtype=torch.long
    ).unsqueeze(-1)
    library_size = X.sum(-1, keepdim=True)

    batch_dict = {
        "X": X,
        "species": species,
        "batch": batch,
        "library_size": library_size,
    }

    return module, batch_dict


class TestLossScaleCalibration:
    """Three-phase loss scale calibration test (Task 4.8).

    Catches the v2.2 bug where `.mean()` was used instead of
    `.sum(-1).mean()`, silently disabling the L1 sparsity and anchor terms.
    """

    def test_loss_scale_calibration_three_phases(self):
        """Full three-phase calibration: untrained, 10 steps, 50 more with FQR."""
        torch.manual_seed(0)
        module, batch_dict = _build_module_and_batch()
        module.train()

        # ==================================================================
        # Phase 1: After 1 forward pass (untrained model)
        #
        # Two checks:
        # (a) Verify L_residual_sparsity uses .sum(-1).mean(), NOT .mean().
        #     With G=200 genes, the wrong reduction gives a value 200x smaller.
        # (b) Weighted L1 must be > 0.0002 * L_recon (ratio threshold).
        #     Correct reduction gives ratio ~0.0007; buggy .mean() gives
        #     ~0.000004 -- safely separated by the 0.0002 threshold.
        # ==================================================================
        _, outputs_p1 = module(batch_dict)

        # (a) Direct reduction-semantics check: the reported L_residual_sparsity
        # must match .sum(-1).mean(), NOT .mean(), on the same log_rate_residual.
        log_rate_res = outputs_p1["log_rate_residual"]
        l1_sum_mean = log_rate_res.abs().sum(-1).mean()
        l1_mean_bug = log_rate_res.abs().mean()
        reported_l1 = outputs_p1["L_residual_sparsity"]

        assert torch.allclose(reported_l1, l1_sum_mean, atol=1e-5), (
            f"Phase 1: L_residual_sparsity ({reported_l1.item():.6f}) does not "
            f"match .sum(-1).mean() ({l1_sum_mean.item():.6f}). "
            f"It may be using the wrong reduction."
        )
        # Sanity: the two reductions must differ by factor ~G
        n_genes = log_rate_res.shape[1]
        ratio_reductions = (l1_sum_mean / (l1_mean_bug + 1e-12)).item()
        assert abs(ratio_reductions - n_genes) < 2.0, (
            f"Phase 1: .sum(-1).mean() / .mean() = {ratio_reductions:.1f}, "
            f"expected ~{n_genes}. Reduction semantics are inconsistent."
        )

        # (b) Weighted L1 ratio check
        L_recon_p1 = outputs_p1["L_recon"].item()
        L_residual_sparsity_p1 = outputs_p1["L_residual_sparsity"].item()
        weighted_l1_p1 = LAMBDA_RES * L_residual_sparsity_p1

        assert L_recon_p1 > 0, (
            f"Phase 1: L_recon should be positive, got {L_recon_p1}"
        )
        ratio_l1_p1 = weighted_l1_p1 / (L_recon_p1 + 1e-12)
        assert ratio_l1_p1 > 0.0002, (
            f"Phase 1: weighted L1 ({weighted_l1_p1:.6f}) is < 0.02% of "
            f"L_recon ({L_recon_p1:.4f}). Ratio = {ratio_l1_p1:.6f}. "
            f"This suggests the wrong reduction (.mean() instead of "
            f".sum(-1).mean()) was used -- the buggy reduction would give "
            f"a ratio ~{n_genes}x smaller."
        )

        # ==================================================================
        # Phase 2: After 10 training steps
        # - GIN correction must have drifted from zero
        # - L_alpha_anchor must use .sum() reduction (not .mean())
        # - Weighted L1 must still be meaningful
        # ==================================================================
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

        for _ in range(10):
            optimizer.zero_grad()
            loss, _ = module(batch_dict)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            _, outputs_p2 = module(batch_dict)

        # GIN correction must have drifted from zero
        correction_p2 = outputs_p2["correction"]
        correction_drift = correction_p2.abs().mean().item()
        assert correction_drift > 0.01, (
            f"Phase 2: GIN correction has not drifted from zero after 10 "
            f"training steps. correction.abs().mean() = {correction_drift:.6f}, "
            f"expected > 0.01."
        )

        # Verify anchor uses .sum() reduction (not .mean()).
        # With G=200 genes, .sum() gives a value 200x larger than .mean().
        # This is the same class of bug as the L1 reduction issue.
        L_anchor_p2 = outputs_p2["L_alpha_anchor"]
        expected_anchor_sum = correction_p2.pow(2).sum()
        expected_anchor_mean = correction_p2.pow(2).mean()
        assert torch.allclose(L_anchor_p2, expected_anchor_sum, atol=1e-5), (
            f"Phase 2: L_alpha_anchor ({L_anchor_p2.item():.6f}) does not "
            f"match correction.pow(2).sum() ({expected_anchor_sum.item():.6f}). "
            f"Expected .sum(), not .mean() ({expected_anchor_mean.item():.6f})."
        )

        # Weighted L1 must remain meaningful after training
        L_recon_p2 = outputs_p2["L_recon"].item()
        weighted_l1_p2 = LAMBDA_RES * outputs_p2["L_residual_sparsity"].item()
        ratio_l1_p2 = weighted_l1_p2 / (L_recon_p2 + 1e-12)
        assert ratio_l1_p2 > 0.0002, (
            f"Phase 2: weighted L1 ({weighted_l1_p2:.6f}) collapsed below "
            f"0.02% of L_recon ({L_recon_p2:.4f}) after 10 training steps. "
            f"Ratio = {ratio_l1_p2:.6f}."
        )

        # ==================================================================
        # Phase 3: After 50 more steps with FQR enabled
        # - Set current_epoch past FQR warmup
        # - S_g should have positive fraction > 0.05
        # - FQR weighted by LAMBDA_FIT must contribute > 0.0001 * L_recon
        # - Anchor weighted by LAMBDA_ANCHOR must be > 0.0001 * L_recon
        #   (correction has had 60 total steps to drift)
        # ==================================================================
        module.current_epoch = 25  # past WARMUP_FQR_EPOCHS (20)
        assert module.current_epoch > WARMUP_FQR_EPOCHS, (
            f"current_epoch ({module.current_epoch}) should be > "
            f"WARMUP_FQR_EPOCHS ({WARMUP_FQR_EPOCHS})"
        )

        for _ in range(50):
            optimizer.zero_grad()
            loss, _ = module(batch_dict)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            _, outputs_p3 = module(batch_dict)

        L_recon_p3 = outputs_p3["L_recon"].item()
        assert L_recon_p3 > 0, (
            f"Phase 3: L_recon should be positive, got {L_recon_p3}"
        )

        # S_g should have a meaningful positive fraction
        S_g = outputs_p3["S_g"]
        positive_frac = (S_g > 0).float().mean().item()
        assert positive_frac > 0.05, (
            f"Phase 3: S_g positive fraction = {positive_frac:.4f}, "
            f"expected > 0.05. FQR signal is too sparse to be useful."
        )

        # FQR weighted by LAMBDA_FIT must contribute meaningfully
        L_fit_p3 = outputs_p3["L_fit_reward"].item()
        weighted_fqr_p3 = LAMBDA_FIT * abs(L_fit_p3)
        ratio_fqr_p3 = weighted_fqr_p3 / (L_recon_p3 + 1e-12)
        assert ratio_fqr_p3 > 0.0001, (
            f"Phase 3: weighted FQR ({weighted_fqr_p3:.6f}) is < 0.01% of "
            f"L_recon ({L_recon_p3:.4f}). Ratio = {ratio_fqr_p3:.8f}. "
            f"FQR is not contributing meaningfully to the loss."
        )

        # Anchor weighted by LAMBDA_ANCHOR must be meaningful after
        # sufficient training (correction has had 60 total steps to grow).
        weighted_anchor_p3 = LAMBDA_ANCHOR * outputs_p3["L_alpha_anchor"].item()
        ratio_anchor_p3 = weighted_anchor_p3 / (L_recon_p3 + 1e-12)
        assert ratio_anchor_p3 > 0.0001, (
            f"Phase 3: weighted anchor ({weighted_anchor_p3:.6f}) is < 0.01% "
            f"of L_recon ({L_recon_p3:.4f}). Ratio = {ratio_anchor_p3:.8f}. "
            f"Anchor term is negligible -- correction may not be growing, "
            f"or anchor uses wrong reduction (.mean() instead of .sum())."
        )
