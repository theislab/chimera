"""Task 2.7: Unit tests for ChimeraMinimalModule.

Tests the minimal variant (fixed alpha, no GIN/FQR/anchor) covering:
  1. Forward pass output shapes
  2. Loss finiteness and positivity
  3. Gradient flow to shared decoder from both L_recon and L_shared_aux
  4. Gradient flow to residual decoder from L_recon
  5. L1 loss scale (catches wrong reduction)
  6. theta.detach() in L_shared_aux (dispersion isolation)
  7. Alpha is fixed (non-learnable) across training steps
"""

import torch

from chimera._module import ChimeraMinimalModule


class TestChimeraMinimalVariant:
    """Comprehensive tests for ChimeraMinimalModule (Task 2.7)."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_module(sd):
        """Build a ChimeraMinimalModule from the synthetic_data fixture."""
        return ChimeraMinimalModule(
            n_genes=sd["n_genes"],
            n_species=sd["n_species"],
            n_batch=sd["n_batch"],
            conservation_prior=sd["conservation_prior"],
            lambda_aux=0.3,
            lambda_res=0.05,
        )

    @staticmethod
    def _forward(module, sd):
        """Run a forward pass and return (loss, outputs)."""
        module.train()
        return module(sd["batch_dict"])

    # ------------------------------------------------------------------
    # Test 1: Forward pass shapes
    # ------------------------------------------------------------------

    def test_forward_pass_shapes(self, synthetic_data):
        """mu_full, mu_shared, z, log_rate_shared, log_rate_residual have correct shapes."""
        torch.manual_seed(42)
        module = self._make_module(synthetic_data)
        _, outputs = self._forward(module, synthetic_data)

        B = synthetic_data["batch_dict"]["X"].shape[0]
        G = synthetic_data["n_genes"]
        D = module.n_latent

        assert outputs["mu_full"].shape == (B, G), (
            f"mu_full shape {outputs['mu_full'].shape}, expected ({B}, {G})"
        )
        assert outputs["mu_shared"].shape == (B, G), (
            f"mu_shared shape {outputs['mu_shared'].shape}, expected ({B}, {G})"
        )
        assert outputs["z"].shape == (B, D), (
            f"z shape {outputs['z'].shape}, expected ({B}, {D})"
        )
        assert outputs["z_mean"].shape == (B, D), (
            f"z_mean shape {outputs['z_mean'].shape}, expected ({B}, {D})"
        )
        assert outputs["z_logvar"].shape == (B, D), (
            f"z_logvar shape {outputs['z_logvar'].shape}, expected ({B}, {D})"
        )
        assert outputs["log_rate_shared"].shape == (B, G), (
            f"log_rate_shared shape {outputs['log_rate_shared'].shape}, expected ({B}, {G})"
        )
        assert outputs["log_rate_residual"].shape == (B, G), (
            f"log_rate_residual shape {outputs['log_rate_residual'].shape}, expected ({B}, {G})"
        )

    # ------------------------------------------------------------------
    # Test 2: Loss is finite and positive
    # ------------------------------------------------------------------

    def test_loss_finite_and_positive(self, synthetic_data):
        """L_recon > 0, L_shared_aux > 0, L_residual_sparsity > 0, total loss finite."""
        torch.manual_seed(42)
        module = self._make_module(synthetic_data)
        loss, outputs = self._forward(module, synthetic_data)

        assert torch.isfinite(loss), f"total loss is not finite: {loss.item()}"
        assert outputs["L_recon"].item() > 0, (
            f"L_recon should be positive, got {outputs['L_recon'].item()}"
        )
        assert outputs["L_shared_aux"].item() > 0, (
            f"L_shared_aux should be positive, got {outputs['L_shared_aux'].item()}"
        )
        assert outputs["L_residual_sparsity"].item() > 0, (
            f"L_residual_sparsity should be positive, got {outputs['L_residual_sparsity'].item()}"
        )

    # ------------------------------------------------------------------
    # Test 3: Gradient flows to shared decoder from BOTH L_recon AND
    #         L_shared_aux
    # ------------------------------------------------------------------

    def test_gradient_flows_to_shared_decoder(self, synthetic_data):
        """output_shared.weight gets gradient from total loss. Then, in isolation,
        L_shared_aux alone also provides gradient to shared decoder."""
        torch.manual_seed(42)
        module = self._make_module(synthetic_data)
        loss, outputs = self._forward(module, synthetic_data)

        # Part A: Full loss backward -- shared decoder must get gradient
        loss.backward()
        shared_grad = module.output_shared.weight.grad
        assert shared_grad is not None, (
            "output_shared.weight.grad is None after full loss backward"
        )
        assert shared_grad.abs().sum().item() > 0, (
            "output_shared.weight gradient is all zeros after full loss backward"
        )

        # Part B: Zero grads, then backward only L_shared_aux
        module.zero_grad()
        # Re-run forward to get a fresh graph
        loss2, outputs2 = self._forward(module, synthetic_data)
        outputs2["L_shared_aux"].backward()

        shared_grad_aux = module.output_shared.weight.grad
        assert shared_grad_aux is not None, (
            "output_shared.weight.grad is None after L_shared_aux backward"
        )
        assert shared_grad_aux.abs().sum().item() > 0, (
            "output_shared.weight gradient is all zeros from L_shared_aux alone"
        )

    # ------------------------------------------------------------------
    # Test 4: Gradient flows to residual decoder from L_recon
    # ------------------------------------------------------------------

    def test_gradient_flows_to_residual_decoder(self, synthetic_data):
        """output_residual.weight gets gradient after full loss backward."""
        torch.manual_seed(42)
        module = self._make_module(synthetic_data)
        loss, outputs = self._forward(module, synthetic_data)
        loss.backward()

        residual_grad = module.output_residual.weight.grad
        assert residual_grad is not None, (
            "output_residual.weight.grad is None after backward"
        )
        assert residual_grad.abs().sum().item() > 0, (
            "output_residual.weight gradient is all zeros"
        )

    # ------------------------------------------------------------------
    # Test 5: L1 loss scale (catches wrong reduction)
    # ------------------------------------------------------------------

    def test_l1_loss_scale(self, synthetic_data):
        """lambda_res * L_residual_sparsity > 0.0001 * L_recon.

        With the correct .sum(-1).mean() reduction the ratio is ~0.0008.
        With the buggy .mean() reduction it would be G (~100) times smaller,
        yielding ~0.000008, which fails this check. The threshold of 0.0001
        cleanly separates the two regimes.

        Additionally verifies the reduction matches .sum(-1).mean() exactly.
        """
        torch.manual_seed(42)
        module = self._make_module(synthetic_data)
        _, outputs = self._forward(module, synthetic_data)

        lambda_res = module.lambda_res
        weighted_l1 = lambda_res * outputs["L_residual_sparsity"].item()
        L_recon = outputs["L_recon"].item()

        # Ratio check: correct reduction gives ~0.0008, buggy gives ~0.000008
        assert weighted_l1 > 0.0001 * L_recon, (
            f"lambda_res * L_residual_sparsity ({weighted_l1:.6f}) <= "
            f"0.0001 * L_recon ({0.0001 * L_recon:.6f}). "
            f"This suggests the L1 term uses the wrong reduction "
            f"(.mean() instead of .sum(-1).mean())."
        )

        # Direct reduction verification: reported value must match .sum(-1).mean()
        log_rate_res = outputs["log_rate_residual"]
        l1_sum_mean = log_rate_res.abs().sum(-1).mean()
        reported = outputs["L_residual_sparsity"]
        assert torch.allclose(reported, l1_sum_mean, atol=1e-5), (
            f"L_residual_sparsity ({reported.item():.6f}) != "
            f".sum(-1).mean() ({l1_sum_mean.item():.6f})"
        )

    # ------------------------------------------------------------------
    # Test 6: theta.detach() -- dispersion isolation
    # ------------------------------------------------------------------

    def test_theta_detach_in_shared_aux(self, synthetic_data):
        """Dispersion must NOT get gradient from L_shared_aux alone (theta.detach()).
        Dispersion MUST get gradient from the full loss (via L_recon)."""
        torch.manual_seed(42)
        module = self._make_module(synthetic_data)

        # Phase A: L_shared_aux alone should not affect dispersion
        loss, outputs = self._forward(module, synthetic_data)
        module.zero_grad()
        outputs["L_shared_aux"].backward(retain_graph=True)

        disp_grad_aux = module.dispersion.grad
        assert disp_grad_aux is None or disp_grad_aux.abs().max().item() == 0.0, (
            f"dispersion received gradient from L_shared_aux: "
            f"max |grad| = {disp_grad_aux.abs().max().item() if disp_grad_aux is not None else 'N/A'}"
        )

        # Phase B: Full loss should give dispersion gradient (from L_recon)
        module.zero_grad()
        loss.backward()

        disp_grad_full = module.dispersion.grad
        assert disp_grad_full is not None, (
            "dispersion.grad is None after full loss backward"
        )
        assert disp_grad_full.abs().max().item() > 0, (
            "dispersion gradient is all zeros after full loss backward -- "
            "L_recon should provide gradient through theta"
        )

    # ------------------------------------------------------------------
    # Test 7: Alpha is fixed (non-learnable)
    # ------------------------------------------------------------------

    def test_alpha_fixed_after_training(self, synthetic_data):
        """Alpha must not change after multiple optimizer steps."""
        torch.manual_seed(42)
        module = self._make_module(synthetic_data)
        module.train()

        # Record alpha before training
        alpha_before = module.alpha.clone()

        # Confirm alpha is not in parameter list
        param_names = {name for name, _ in module.named_parameters()}
        assert "alpha" not in param_names, (
            "alpha appears in named_parameters -- it should be a buffer"
        )

        # Run several training steps
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        for _ in range(10):
            optimizer.zero_grad()
            loss, _ = module(synthetic_data["batch_dict"])
            loss.backward()
            optimizer.step()

        # Alpha must be unchanged
        alpha_after = module.alpha
        assert torch.equal(alpha_before, alpha_after), (
            f"alpha changed after training steps. "
            f"max |delta| = {(alpha_before - alpha_after).abs().max().item():.8f}"
        )
