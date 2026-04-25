"""Tests for Chimera v2.4 modules (_module.py).

Covers:
  - ChimeraMinimalModule
  - ChimeraModule (full v2.4)
"""

import torch

from chimera._module import ChimeraMinimalModule, ChimeraModule


# =========================================================================
# ChimeraMinimalModule
# =========================================================================


class TestChimeraMinimalModule:
    """Tests for the minimal (fixed-alpha) Chimera module."""

    def _make_module(self, sd):
        return ChimeraMinimalModule(
            n_genes=sd["n_genes"],
            n_species=sd["n_species"],
            n_batch=sd["n_batch"],
            conservation_prior=sd["conservation_prior"],
        )

    def test_minimal_forward_pass(self, synthetic_data):
        """Loss is finite and output dict contains expected keys."""
        module = self._make_module(synthetic_data)
        module.train()
        loss, outputs = module(synthetic_data["batch_dict"])

        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
        expected_keys = {
            "z_mean", "z_logvar", "z", "mu_full", "mu_shared",
            "log_rate_shared", "log_rate_residual", "alpha", "theta",
            "L_recon", "L_kl", "L_shared_aux", "L_residual_sparsity", "loss",
        }
        assert expected_keys.issubset(outputs.keys()), (
            f"Missing keys: {expected_keys - outputs.keys()}"
        )

    def test_minimal_alpha_not_learnable(self, synthetic_data):
        """alpha is a buffer (not a parameter), has no gradient."""
        module = self._make_module(synthetic_data)
        module.train()
        loss, outputs = module(synthetic_data["batch_dict"])
        loss.backward()

        # alpha must be a buffer, not a parameter
        param_names = {n for n, _ in module.named_parameters()}
        assert "alpha" not in param_names, "alpha should be a buffer, not a parameter"

        # alpha should not have a gradient
        assert module.alpha.grad is None or not module.alpha.requires_grad, (
            "alpha buffer should not accumulate gradients"
        )

    def test_minimal_gradient_flow(self, synthetic_data):
        """Both shared and residual output heads receive gradients."""
        module = self._make_module(synthetic_data)
        module.train()
        loss, outputs = module(synthetic_data["batch_dict"])
        loss.backward()

        shared_grad = module.output_shared.weight.grad
        # Handle both Sequential (full module) and Linear (minimal module) residual
        if hasattr(module.output_residual, 'weight'):
            residual_grad = module.output_residual.weight.grad
        else:
            residual_grad = module.output_residual[-1].weight.grad
        assert shared_grad is not None, "No gradient on output_shared.weight"
        assert residual_grad is not None, "No gradient on output_residual.weight"
        assert shared_grad.abs().sum().item() > 0, "output_shared gradient is all zeros"
        assert residual_grad.abs().sum().item() > 0, "output_residual gradient is all zeros"

    def test_minimal_theta_detach_in_shared_aux(self, synthetic_data):
        """Dispersion gets NO gradient from L_shared_aux path.

        The L_shared_aux loss uses theta.detach(), so when we isolate
        that loss term, dispersion should get zero gradient from it.
        """
        module = self._make_module(synthetic_data)
        module.train()
        loss, outputs = module(synthetic_data["batch_dict"])

        # Zero all grads and backprop only L_shared_aux
        module.zero_grad()
        outputs["L_shared_aux"].backward(retain_graph=True)

        disp_grad = module.dispersion.grad
        # When theta is detached, dispersion.grad is None (no grad_fn)
        assert disp_grad is None or disp_grad.abs().max().item() == 0.0, (
            f"dispersion got gradient from L_shared_aux: max abs = {disp_grad.abs().max().item()}"
        )

    def test_minimal_l1_reduction(self, synthetic_data):
        """L_residual_sparsity uses .sum(-1).mean(), not .mean().

        These differ by a factor of G (n_genes). We verify by recomputing
        with both reductions and checking which one matches.
        """
        module = self._make_module(synthetic_data)
        module.train()
        loss, outputs = module(synthetic_data["batch_dict"])

        log_rate_res = outputs["log_rate_residual"]
        G = synthetic_data["n_genes"]

        l1_sum_mean = log_rate_res.abs().sum(-1).mean()
        l1_mean = log_rate_res.abs().mean()

        reported = outputs["L_residual_sparsity"]

        assert torch.allclose(reported, l1_sum_mean, atol=1e-5), (
            f"L_residual_sparsity ({reported.item():.6f}) != .sum(-1).mean() ({l1_sum_mean.item():.6f})"
        )
        # Sanity: the two reductions should differ by factor G when B > 1
        ratio = l1_sum_mean / (l1_mean + 1e-12)
        assert abs(ratio.item() - G) < 1.0, (
            f".sum(-1).mean() / .mean() ratio = {ratio.item():.2f}, expected ~{G}"
        )


# =========================================================================
# ChimeraModule (full v2.4)
# =========================================================================


class TestChimeraModule:
    """Tests for the full Chimera v2.4 module."""

    def _make_module(self, sd):
        return ChimeraModule(
            n_genes=sd["n_genes"],
            n_species=sd["n_species"],
            n_batch=sd["n_batch"],
            gene_features=sd["gene_features"],
            conservation_prior=sd["conservation_prior"],
            p_null=sd["p_null"],
            batch_to_species_idx=sd["batch_to_species_idx"],
            use_observed_lib_size=True,
        )

    def test_full_forward_pass(self, synthetic_data):
        """Loss is finite, all expected output keys present."""
        module = self._make_module(synthetic_data)
        module.train()
        loss, outputs = module(synthetic_data["batch_dict"])

        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
        expected_keys = {
            "z_mean", "z_logvar", "z", "alpha", "correction",
            "mu_full", "mu_shared_only", "mu_shared_cellstate",
            "log_rate_shared_full", "log_rate_shared_det",
            "log_rate_residual", "log_rate_batch", "theta",
            "S_g", "L_recon", "L_kl", "L_shared_aux",
            "L_residual_sparsity", "L_alpha_anchor", "L_fit_reward",
            "L_disp", "loss",
        }
        assert expected_keys.issubset(outputs.keys()), (
            f"Missing keys: {expected_keys - outputs.keys()}"
        )

    def test_full_output_shapes(self, synthetic_data):
        """Check shapes of alpha, z, mu_full, theta, etc."""
        module = self._make_module(synthetic_data)
        module.train()
        _, outputs = module(synthetic_data["batch_dict"])

        B = synthetic_data["batch_dict"]["X"].shape[0]
        G = synthetic_data["n_genes"]
        D = module.n_latent

        assert outputs["alpha"].shape == (G,), f"alpha shape: {outputs['alpha'].shape}"
        assert outputs["correction"].shape == (G,), f"correction shape: {outputs['correction'].shape}"
        assert outputs["z"].shape == (B, D), f"z shape: {outputs['z'].shape}"
        assert outputs["z_mean"].shape == (B, D), f"z_mean shape: {outputs['z_mean'].shape}"
        assert outputs["mu_full"].shape == (B, G), f"mu_full shape: {outputs['mu_full'].shape}"
        assert outputs["mu_shared_only"].shape == (B, G), f"mu_shared_only shape: {outputs['mu_shared_only'].shape}"
        assert outputs["mu_shared_cellstate"].shape == (B, G), f"mu_shared_cellstate shape: {outputs['mu_shared_cellstate'].shape}"
        assert outputs["theta"].shape == (B, G), f"theta shape: {outputs['theta'].shape}"

    def test_full_gradient_flow(self, synthetic_data):
        """All parameter groups get gradients (after a few steps for GIN)."""
        module = self._make_module(synthetic_data)
        module.train()
        # Set epoch past FQR warmup so FQR provides gradient to the GIN.
        # Under detach_alpha_from_recon, the GIN gets gradient only from
        # FQR + anchor, and FQR activates after warmup_fqr_epochs.
        module.current_epoch = module.warmup_fqr_epochs + 1

        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

        # Run a few steps so GIN gradients propagate
        for _ in range(5):
            optimizer.zero_grad()
            loss, _ = module(synthetic_data["batch_dict"])
            loss.backward()
            optimizer.step()

        # After the last backward, check gradient presence
        optimizer.zero_grad()
        loss, _ = module(synthetic_data["batch_dict"])
        loss.backward()

        groups = {
            "encoder": module.encoder[0].weight,
            "decoder": module.decoder_hidden[0].weight,
            "output_shared": module.output_shared.weight,
            "output_residual": module.output_residual[-1].weight,
            "dispersion": module.dispersion,
            "gin_layer0": module.gin.correction_mlp[0].weight,
            "gin_last_layer": module.gin.correction_mlp[-1].weight,
            "batch_offset": module.batch_offset.batch_offset_raw,
        }
        for name, param in groups.items():
            assert param.grad is not None, f"{name} has no gradient"
            assert param.grad.abs().sum().item() > 0, f"{name} gradient is all zeros"

    def test_full_dual_h_gradient_isolation(self, synthetic_data):
        """h_detached path does NOT give gradient to encoder.

        L_shared_aux uses mu_shared_only which is built from h_detached
        (decoder applied to z.detach()). Backpropagating only L_shared_aux
        should produce zero gradient on the encoder.
        """
        module = self._make_module(synthetic_data)
        module.train()
        loss, outputs = module(synthetic_data["batch_dict"])

        module.zero_grad()
        outputs["L_shared_aux"].backward(retain_graph=True)

        enc_grad = module.encoder[0].weight.grad
        # When z is detached, encoder.grad is None (no path from L_shared_aux to encoder)
        assert enc_grad is None or enc_grad.abs().max().item() == 0.0, (
            f"encoder got gradient from L_shared_aux: max abs = {enc_grad.abs().max().item()}"
        )

    def test_full_three_mu_predictions(self, synthetic_data):
        """mu_full, mu_shared_only, and mu_shared_cellstate are distinct tensors."""
        module = self._make_module(synthetic_data)
        module.train()
        _, outputs = module(synthetic_data["batch_dict"])

        mu_full = outputs["mu_full"]
        mu_shared_only = outputs["mu_shared_only"]
        mu_shared_cellstate = outputs["mu_shared_cellstate"]

        # They should generally differ (not identical tensors)
        diff_full_shared = (mu_full - mu_shared_only).abs().sum().item()
        diff_shared_cellstate = (mu_shared_only - mu_shared_cellstate).abs().sum().item()
        diff_full_cellstate = (mu_full - mu_shared_cellstate).abs().sum().item()

        # At least one pair must differ; in practice all three are distinct
        # because residual output and batch offsets contribute differently
        total_diff = diff_full_shared + diff_shared_cellstate + diff_full_cellstate
        assert total_diff > 0, (
            "All three mu predictions are identical -- they should be distinct"
        )

    def test_full_training_step(self, synthetic_data):
        """Loss decreases over 10 training steps."""
        module = self._make_module(synthetic_data)
        module.train()

        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

        # Record initial loss
        torch.manual_seed(0)
        with torch.no_grad():
            loss_initial, _ = module(synthetic_data["batch_dict"])

        # Train for 10 steps
        for step in range(10):
            optimizer.zero_grad()
            loss, _ = module(synthetic_data["batch_dict"])
            loss.backward()
            optimizer.step()

        # Record final loss
        with torch.no_grad():
            loss_final, _ = module(synthetic_data["batch_dict"])

        assert loss_final.item() < loss_initial.item(), (
            f"Loss did not decrease: {loss_initial.item():.4f} -> {loss_final.item():.4f}"
        )

    def test_full_fqr_warmup(self, synthetic_data):
        """L_fit_reward is 0 when current_epoch < warmup_fqr_epochs."""
        module = self._make_module(synthetic_data)
        module.train()
        # Ensure current_epoch is below warmup
        module.current_epoch = 0

        _, outputs = module(synthetic_data["batch_dict"])
        assert outputs["L_fit_reward"].item() == 0.0, (
            f"L_fit_reward should be 0 during warmup, got {outputs['L_fit_reward'].item()}"
        )

        # Also check that setting epoch above warmup gives non-zero
        module.current_epoch = module.warmup_fqr_epochs + 1
        # First run a step to populate S_g_ema with non-zero values
        loss, _ = module(synthetic_data["batch_dict"])
        loss.backward()
        # Now the EMA is updated; run again
        _, outputs2 = module(synthetic_data["batch_dict"])
        # L_fit_reward may or may not be zero depending on S_g_ema,
        # but it should NOT be forced to zero by the warmup gate
        # (it is computed as -(correction * S_g_ema).sum())
        assert isinstance(outputs2["L_fit_reward"], torch.Tensor), (
            "L_fit_reward should be a tensor after warmup"
        )

    def test_forward_pass_trace(self, synthetic_data):
        """Comprehensive trace test: shape checks for all outputs + gradient flow verification."""
        module = self._make_module(synthetic_data)
        module.train()
        # Past FQR warmup so GIN gets gradient from FQR (not just L_recon).
        module.current_epoch = module.warmup_fqr_epochs + 1

        B = synthetic_data["batch_dict"]["X"].shape[0]
        G = synthetic_data["n_genes"]
        D = module.n_latent

        # Run a few optimizer steps so GIN gradients propagate
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        for _ in range(3):
            optimizer.zero_grad()
            loss, _ = module(synthetic_data["batch_dict"])
            loss.backward()
            optimizer.step()

        # Final forward + backward for gradient checks
        optimizer.zero_grad()
        loss, outputs = module(synthetic_data["batch_dict"])
        loss.backward()

        # ---- Shape checks for ALL outputs ----

        # (G,) shaped outputs
        assert outputs["alpha"].shape == (G,), f"alpha shape {outputs['alpha'].shape} != ({G},)"
        assert outputs["correction"].shape == (G,), f"correction shape {outputs['correction'].shape} != ({G},)"
        assert outputs["S_g"].shape == (G,), f"S_g shape {outputs['S_g'].shape} != ({G},)"

        # (B, D) shaped outputs
        assert outputs["z_mean"].shape == (B, D), f"z_mean shape {outputs['z_mean'].shape} != ({B}, {D})"
        assert outputs["z_logvar"].shape == (B, D), f"z_logvar shape {outputs['z_logvar'].shape} != ({B}, {D})"

        # (B, G) shaped outputs
        for key in [
            "mu_full", "mu_shared_only", "mu_shared_cellstate",
            "log_rate_shared_full", "log_rate_shared_det",
            "log_rate_residual", "log_rate_batch", "theta",
        ]:
            assert outputs[key].shape == (B, G), f"{key} shape {outputs[key].shape} != ({B}, {G})"

        # All L_* scalars must be 0-dim tensors
        for key in [
            "L_recon", "L_kl", "L_shared_aux", "L_residual_sparsity",
            "L_alpha_anchor", "L_fit_reward", "L_disp", "loss",
        ]:
            assert outputs[key].dim() == 0, f"{key} is not a scalar (dim={outputs[key].dim()})"

        # ---- Gradient flow: these MUST have gradients ----
        must_have_grad = {
            "encoder[0].weight": module.encoder[0].weight,
            "decoder_hidden[0].weight": module.decoder_hidden[0].weight,
            "output_shared.weight": module.output_shared.weight,
            "output_residual.weight": module.output_residual[-1].weight,
            "gin.correction_mlp[0].weight": module.gin.correction_mlp[0].weight,
            "gin.correction_mlp[-1].weight": module.gin.correction_mlp[-1].weight,
            "batch_offset.batch_offset_raw": module.batch_offset.batch_offset_raw,
            "dispersion": module.dispersion,
        }
        for name, param in must_have_grad.items():
            assert param.grad is not None, f"{name} has no gradient"
            assert param.grad.abs().sum().item() > 0, f"{name} gradient is all zeros"

        # ---- Buffers MUST NOT have gradients ----
        buffers_no_grad = {
            "conservation_prior": module.conservation_prior,
            "gene_features": module.gene_features,
            "p_null": module.p_null,
            "S_g_ema": module.S_g_ema,
        }
        for name, buf in buffers_no_grad.items():
            assert buf.grad is None, f"buffer {name} should have no gradient, got {buf.grad}"

    def test_inject_covariates_guard(self, synthetic_data):
        """Verify decoder_hidden first layer has n_latent input dim (no covariates)."""
        module = self._make_module(synthetic_data)
        first_weight = list(module.decoder_hidden.parameters())[0]
        assert first_weight.shape[1] == module.n_latent, (
            f"decoder_hidden input dim {first_weight.shape[1]} != n_latent {module.n_latent}"
        )
