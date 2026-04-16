"""Tests for Chimera v2.4 loss functions and loss calibration.

Tests individual loss reduction semantics, gradient isolation properties,
and the critical loss-scale calibration that ensures the weighted L1
sparsity term is not negligible relative to reconstruction loss.
"""

import torch
import torch.nn.functional as F
from scvi.distributions import NegativeBinomial

from chimera._module import ChimeraModule


# =========================================================================
# Loss reduction semantics
# =========================================================================


class TestLossReductions:
    """Verify that loss terms use the correct reduction strategy."""

    def test_l1_reduction_correct(self, synthetic_data):
        """.sum(-1).mean() differs from .mean() by factor G.

        The residual sparsity loss must use .sum(-1).mean() (sum over genes
        per cell, then mean over cells), NOT .mean() which would dilute the
        per-gene penalty by a factor of G.
        """
        G = synthetic_data["n_genes"]
        log_rate_res = torch.randn(32, G)  # (B, G)

        l1_sum_mean = log_rate_res.abs().sum(-1).mean()
        l1_mean = log_rate_res.abs().mean()

        ratio = l1_sum_mean / l1_mean
        assert abs(ratio.item() - G) < 1.0, (
            f".sum(-1).mean() / .mean() = {ratio.item():.2f}, expected ~{G}"
        )

    def test_anchor_reduction_correct(self, synthetic_data):
        """L_alpha_anchor uses .sum(), not .mean()."""
        module = _make_full_module(synthetic_data)
        module.train()
        _, outputs = module(synthetic_data["batch_dict"])

        correction = outputs["correction"]
        expected = correction.pow(2).sum()
        actual = outputs["L_alpha_anchor"]

        assert torch.allclose(actual, expected, atol=1e-5), (
            f"L_alpha_anchor ({actual.item():.6f}) != correction.pow(2).sum() ({expected.item():.6f})"
        )


# =========================================================================
# Gradient isolation
# =========================================================================


class TestGradientIsolation:
    """Verify that detach() calls correctly isolate gradients."""

    def test_theta_detach_contrast(self, synthetic_data):
        """With detach: no grad to dispersion from L_shared_aux.
        Without detach: gradient would exist.
        """
        module = _make_full_module(synthetic_data)
        module.train()
        _, outputs = module(synthetic_data["batch_dict"])

        # --- WITH detach (as implemented) ---
        module.zero_grad()
        outputs["L_shared_aux"].backward(retain_graph=True)
        # When theta is detached, dispersion.grad is None (no grad_fn connects them)
        assert module.dispersion.grad is None or module.dispersion.grad.abs().max().item() == 0.0, (
            "dispersion got gradient from L_shared_aux despite theta.detach()"
        )

        # --- WITHOUT detach (counterfactual) ---
        # Manually compute L_shared_aux without detaching theta
        module.zero_grad()
        theta_live = F.softplus(module.dispersion[
            synthetic_data["batch_dict"]["batch"].squeeze(-1).long()
        ])
        mu_so = outputs["mu_shared_only"]
        x = synthetic_data["batch_dict"]["X"]
        l_aux_no_detach = (
            -NegativeBinomial(mu=mu_so.detach(), theta=theta_live)
            .log_prob(x)
            .sum(-1)
            .mean()
        )
        l_aux_no_detach.backward()
        grad_without_detach = module.dispersion.grad
        assert grad_without_detach is not None, "Expected gradient without detach"
        assert grad_without_detach.abs().max().item() > 0, (
            "dispersion should get gradient when theta is not detached"
        )

    def test_fqr_gradient_isolation(self, synthetic_data):
        """Only correction (via GIN) gets gradient from L_fit_reward.

        L_fit_reward = -(correction * S_g_ema).sum(). S_g_ema is a buffer
        (detached), so the only learnable path is through correction.
        The encoder should get zero gradient from this term.
        """
        module = _make_full_module(synthetic_data)
        module.train()
        # Set epoch past warmup so FQR is active
        module.current_epoch = module.warmup_fqr_epochs + 1

        # Run a forward pass to populate S_g_ema
        loss, _ = module(synthetic_data["batch_dict"])
        loss.backward()
        # Run again so S_g_ema is non-trivially populated
        module.zero_grad()
        _, outputs = module(synthetic_data["batch_dict"])

        # Isolate L_fit_reward
        module.zero_grad()
        outputs["L_fit_reward"].backward(retain_graph=True)

        # GIN should get gradient
        gin_grad = module.gin.correction_mlp[2].weight.grad
        assert gin_grad is not None, "GIN last layer has no gradient from L_fit_reward"

        # Encoder should NOT get gradient from L_fit_reward
        # (grad is None because there's no path from L_fit_reward to encoder)
        enc_grad = module.encoder[0].weight.grad
        assert enc_grad is None or enc_grad.abs().max().item() == 0.0, (
            f"encoder got gradient from L_fit_reward: max abs = {enc_grad.abs().max().item()}"
        )


# =========================================================================
# NB log_prob sanity
# =========================================================================


class TestNBLogProb:
    """Basic sanity checks for Negative Binomial log_prob."""

    def test_nb_log_prob_finite(self):
        """NB log_prob is finite for typical input ranges."""
        mu = torch.tensor([1.0, 10.0, 100.0, 1000.0])
        theta = torch.tensor([0.1, 1.0, 10.0, 100.0])
        x = torch.tensor([0.0, 5.0, 50.0, 500.0])

        log_prob = NegativeBinomial(mu=mu, theta=theta).log_prob(x)
        assert torch.isfinite(log_prob).all(), (
            f"NB log_prob has non-finite values: {log_prob}"
        )


# =========================================================================
# Loss scale calibration
# =========================================================================


class TestLossScaleCalibration:
    """Verify loss terms are in a sensible relative range after training."""

    def test_loss_scale_calibration(self, synthetic_data):
        """After 10 training steps, weighted L1 > 0.1% of L_recon.

        This is the critical calibration test: if the L1 sparsity term is
        negligibly small compared to reconstruction loss, it cannot
        effectively regularize the residual decoder. Using the wrong
        reduction (.mean() instead of .sum(-1).mean()) would cause this
        to fail because the L1 term would be diluted by a factor of G.
        """
        module = _make_full_module(synthetic_data)
        module.train()

        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        for _ in range(10):
            optimizer.zero_grad()
            loss, outputs = module(synthetic_data["batch_dict"])
            loss.backward()
            optimizer.step()

        # Run one more forward pass for measurement
        with torch.no_grad():
            _, outputs = module(synthetic_data["batch_dict"])

        L_recon = outputs["L_recon"].item()
        weighted_l1 = module.lambda_res * outputs["L_residual_sparsity"].item()

        assert L_recon > 0, f"L_recon should be positive, got {L_recon}"
        ratio = weighted_l1 / (L_recon + 1e-12)
        assert ratio > 0.001, (
            f"Weighted L1 ({weighted_l1:.4f}) is < 0.1% of L_recon ({L_recon:.4f}). "
            f"Ratio = {ratio:.6f}. This suggests the wrong reduction is used."
        )


# =========================================================================
# Task 8.3: Expanded Loss tests
# =========================================================================


class TestReconReduction:
    """Verify L_recon uses NB log_prob .sum(-1).mean()."""

    def test_L_recon_reduction(self, synthetic_data):
        """L_recon matches -NB.log_prob(x).sum(-1).mean() exactly."""
        module = _make_full_module(synthetic_data)
        module.train()
        _, outputs = module(synthetic_data["batch_dict"])

        x = synthetic_data["batch_dict"]["X"]
        mu_full = outputs["mu_full"]
        theta = outputs["theta"]

        expected = (
            -NegativeBinomial(mu=mu_full, theta=theta)
            .log_prob(x)
            .sum(-1)
            .mean()
        )

        assert torch.allclose(outputs["L_recon"], expected, atol=1e-5), (
            f"L_recon ({outputs['L_recon'].item():.6f}) != "
            f"-NB.log_prob.sum(-1).mean() ({expected.item():.6f})"
        )


class TestSharedAuxThetaDetach:
    """Verify that theta.detach() blocks gradient to dispersion in L_shared_aux."""

    def test_L_shared_aux_theta_detach(self, synthetic_data):
        """theta.detach() in L_shared_aux blocks gradient to dispersion parameter."""
        module = _make_full_module(synthetic_data)
        module.train()
        _, outputs = module(synthetic_data["batch_dict"])

        # Only backward L_shared_aux
        module.zero_grad()
        outputs["L_shared_aux"].backward(retain_graph=True)

        # dispersion must get no gradient from L_shared_aux
        assert module.dispersion.grad is None or module.dispersion.grad.abs().max().item() == 0.0, (
            "dispersion got gradient from L_shared_aux despite theta.detach()"
        )


class TestFitRewardGradientIsolation:
    """Verify L_fit_reward gradient flows ONLY through correction/GIN."""

    def test_L_fit_reward_gradient_isolation(self, synthetic_data):
        """FQR gradient flows ONLY through correction/GIN, not encoder/decoder/dispersion."""
        module = _make_full_module(synthetic_data)
        module.train()
        # Set epoch past warmup so FQR is active
        module.current_epoch = module.warmup_fqr_epochs + 1

        # Forward pass to populate S_g_ema
        loss, _ = module(synthetic_data["batch_dict"])
        loss.backward()

        # Second forward to get meaningful S_g_ema
        module.zero_grad()
        _, outputs = module(synthetic_data["batch_dict"])

        # Isolate L_fit_reward
        module.zero_grad()
        outputs["L_fit_reward"].backward(retain_graph=True)

        # GIN should get gradient
        gin_last_layer = module.gin.correction_mlp[2].weight.grad
        assert gin_last_layer is not None, "GIN last layer has no gradient from L_fit_reward"

        # Encoder should NOT get gradient
        enc_grad = module.encoder[0].weight.grad
        assert enc_grad is None or enc_grad.abs().max().item() == 0.0, (
            f"encoder got gradient from L_fit_reward: max = {enc_grad.abs().max().item()}"
        )

        # Decoder should NOT get gradient
        dec_grad = module.decoder_hidden[0].weight.grad
        assert dec_grad is None or dec_grad.abs().max().item() == 0.0, (
            f"decoder got gradient from L_fit_reward: max = {dec_grad.abs().max().item()}"
        )

        # Dispersion should NOT get gradient
        disp_grad = module.dispersion.grad
        assert disp_grad is None or disp_grad.abs().max().item() == 0.0, (
            f"dispersion got gradient from L_fit_reward: max = {disp_grad.abs().max().item()}"
        )


class TestLossesWithExtremeInputs:
    """Verify no NaN/Inf with extreme inputs."""

    def test_losses_with_zero_counts(self, synthetic_data):
        """No NaN/Inf when x = 0 (all-zero counts)."""
        module = _make_full_module(synthetic_data)
        module.train()

        batch_dict = {k: v.clone() for k, v in synthetic_data["batch_dict"].items()}
        batch_dict["X"] = torch.zeros_like(batch_dict["X"])
        # Library size must be recomputed for zero counts (avoid div-by-zero)
        batch_dict["library_size"] = torch.ones_like(batch_dict["library_size"])

        loss, outputs = module(batch_dict)

        assert torch.isfinite(loss), f"Loss is not finite with x=0: {loss.item()}"
        for key in ["L_recon", "L_kl", "L_shared_aux", "L_residual_sparsity"]:
            assert torch.isfinite(outputs[key]), (
                f"{key} is not finite with x=0: {outputs[key].item()}"
            )

    def test_losses_with_large_counts(self, synthetic_data):
        """No NaN/Inf with very large counts (x = 10000)."""
        module = _make_full_module(synthetic_data)
        module.train()

        batch_dict = {k: v.clone() for k, v in synthetic_data["batch_dict"].items()}
        batch_dict["X"] = torch.full_like(batch_dict["X"], 10000.0)
        batch_dict["library_size"] = batch_dict["X"].sum(-1, keepdim=True)

        loss, outputs = module(batch_dict)

        assert torch.isfinite(loss), f"Loss is not finite with x=10000: {loss.item()}"
        for key in ["L_recon", "L_kl", "L_shared_aux", "L_residual_sparsity"]:
            assert torch.isfinite(outputs[key]), (
                f"{key} is not finite with x=10000: {outputs[key].item()}"
            )

    def test_losses_with_small_mu(self, synthetic_data):
        """NB log_prob is finite when mu is very small."""
        mu = torch.tensor([1e-8, 1e-6, 1e-4])
        theta = torch.tensor([1.0, 1.0, 1.0])
        x = torch.tensor([0.0, 0.0, 0.0])

        log_prob = NegativeBinomial(mu=mu, theta=theta).log_prob(x)
        assert torch.isfinite(log_prob).all(), (
            f"NB log_prob not finite with very small mu: {log_prob}"
        )


# =========================================================================
# Helper
# =========================================================================


def _make_full_module(sd):
    """Construct a ChimeraModule from synthetic_data dict."""
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


# =========================================================================
# Regression tests for bugs found in rigor audit (Run 7)
# =========================================================================


class TestSgReluRegression:
    """Regression: S_g must be non-negative (ReLU applied).

    Bug found: S_g = Q_g - Q_null_g was computed without ReLU clamp.
    Spec: S_g = ReLU(Q_g - Q_null_g) — negative values must be clamped
    to zero so FQR doesn't push alpha downward when shared decoder is weak.
    """

    def test_S_g_nonnegative(self, synthetic_data):
        """S_g in forward pass outputs must be >= 0 for all genes."""
        module = _make_full_module(synthetic_data)
        module.train()
        module.current_epoch = 25  # Past FQR warmup
        _, outputs = module(synthetic_data["batch_dict"])
        S_g = outputs["S_g"]
        assert (S_g >= 0).all(), (
            f"S_g has {(S_g < 0).sum()} negative values. "
            f"ReLU must be applied: S_g = max(0, Q_g - Q_null_g)"
        )

    def test_S_g_ema_nonnegative(self, synthetic_data):
        """S_g EMA buffer must remain non-negative after updates."""
        module = _make_full_module(synthetic_data)
        module.train()
        module.current_epoch = 25
        # Run several forward passes to accumulate EMA
        for _ in range(5):
            module(synthetic_data["batch_dict"])
        assert (module.S_g_ema >= 0).all(), "S_g EMA buffer has negative values"


class TestMinimalVariantFreeBits:
    """Regression: ChimeraMinimalModule must apply free_bits on KL.

    Bug found: minimal variant computed KL without per-dimension clamping.
    Both variants must use free_bits=0.5 to prevent posterior collapse.
    """

    def test_minimal_module_has_free_bits(self, synthetic_data):
        """ChimeraMinimalModule must have free_bits attribute."""
        from chimera._module import ChimeraMinimalModule
        from chimera._constants import FREE_BITS
        module = ChimeraMinimalModule(
            n_genes=synthetic_data["n_genes"],
            n_species=synthetic_data["n_species"],
            n_batch=synthetic_data["n_batch"],
            conservation_prior=synthetic_data["conservation_prior"],
        )
        assert hasattr(module, "free_bits"), "Missing free_bits attribute"
        assert module.free_bits == FREE_BITS, (
            f"free_bits={module.free_bits}, expected {FREE_BITS}"
        )

    def test_minimal_kl_floor(self, synthetic_data):
        """KL per dimension must be >= free_bits (0.5) in minimal variant."""
        from chimera._module import ChimeraMinimalModule
        module = ChimeraMinimalModule(
            n_genes=synthetic_data["n_genes"],
            n_species=synthetic_data["n_species"],
            n_batch=synthetic_data["n_batch"],
            conservation_prior=synthetic_data["conservation_prior"],
        )
        module.train()
        _, outputs = module(synthetic_data["batch_dict"])
        # With free_bits=0.5, the KL contribution per latent dim must be >= 0.5
        # Total KL = kl_per_dim.sum(-1).mean(), where each dim >= 0.5
        # So total KL >= n_latent * 0.5
        n_latent = 10  # default
        L_kl = outputs["L_kl"]
        assert L_kl.item() >= n_latent * 0.5 * 0.9, (  # 0.9 for numerical margin
            f"KL={L_kl.item():.2f} < {n_latent * 0.5}. "
            f"Free-bits floor not applied."
        )
