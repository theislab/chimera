"""Tests for Chimera v2.4 standalone components (_components.py).

Covers:
  - GeneIntegrationNetwork (GIN)
  - ConstrainedBatchOffset
  - LibrarySizeEncoder
"""

import pytest
import torch
import torch.nn.functional as F

from chimera._components import (
    ConstrainedBatchOffset,
    GeneIntegrationNetwork,
    LibrarySizeEncoder,
)


# =========================================================================
# GeneIntegrationNetwork
# =========================================================================


class TestGeneIntegrationNetwork:
    """Tests for the GIN component."""

    def test_gin_output_shapes(self, synthetic_data):
        """alpha has shape (G,) and correction has shape (G,)."""
        gin = GeneIntegrationNetwork(n_features=72, n_hidden=32)
        gf = synthetic_data["gene_features"]
        cp = synthetic_data["conservation_prior"]

        alpha, correction = gin(gf, cp)

        G = synthetic_data["n_genes"]
        assert alpha.shape == (G,), f"alpha shape {alpha.shape} != ({G},)"
        assert correction.shape == (G,), f"correction shape {correction.shape} != ({G},)"

    def test_gin_zero_initialization(self, synthetic_data):
        """At init, correction ~0 and alpha approx conservation_prior."""
        gin = GeneIntegrationNetwork(n_features=72, n_hidden=32)
        gf = synthetic_data["gene_features"]
        cp = synthetic_data["conservation_prior"]

        alpha, correction = gin(gf, cp)

        # correction should be near zero because last layer is zero-init
        assert correction.abs().max().item() < 1e-4, (
            f"correction not near zero at init: max abs = {correction.abs().max().item()}"
        )
        # alpha should approximately equal the conservation prior
        assert torch.allclose(alpha, cp, atol=1e-3), (
            f"alpha differs from conservation_prior: max diff = {(alpha - cp).abs().max().item()}"
        )

    def test_gin_param_count(self):
        """GIN with default settings: 72->64->32->1 (3-layer MLP)."""
        gin = GeneIntegrationNetwork(n_features=72, n_hidden=32)
        total = sum(p.numel() for p in gin.parameters())
        # 72*64+64 + 64*32+32 + 32*1+1 = 6785
        assert total == 6785, f"GIN param count {total} != 6785"

    def test_gin_gradient_flow(self, synthetic_data):
        """Gradients reach both layers after a few training steps."""
        gin = GeneIntegrationNetwork(n_features=72, n_hidden=32)
        gf = synthetic_data["gene_features"]
        cp = synthetic_data["conservation_prior"]

        optimizer = torch.optim.Adam(gin.parameters(), lr=1e-3)

        # Run a few steps so gradients propagate through ReLU to the first layer
        for _ in range(5):
            optimizer.zero_grad()
            alpha, correction = gin(gf, cp)
            loss = correction.pow(2).sum() + (1.0 - alpha).sum()
            loss.backward()
            optimizer.step()

        # After the last step, check that both layers have gradients
        optimizer.zero_grad()
        alpha, correction = gin(gf, cp)
        loss = correction.pow(2).sum() + (1.0 - alpha).sum()
        loss.backward()

        layer0_grad = gin.correction_mlp[0].weight.grad
        last_layer_grad = gin.correction_mlp[-1].weight.grad
        assert layer0_grad is not None, "No gradient on first layer weight"
        assert last_layer_grad is not None, "No gradient on last layer weight"
        assert layer0_grad.abs().sum().item() > 0, "First layer gradient is all zeros"
        assert last_layer_grad.abs().sum().item() > 0, "Last layer gradient is all zeros"

    def test_gin_alpha_range(self, synthetic_data):
        """alpha stays in [0, 1] even after training steps.

        Sigmoid output is analytically in (0, 1) but float32 can round
        extreme values to exactly 0.0 or 1.0. We verify the weaker but
        correct invariant: 0 <= alpha <= 1.
        """
        gin = GeneIntegrationNetwork(n_features=72, n_hidden=32)
        gf = synthetic_data["gene_features"]
        cp = synthetic_data["conservation_prior"]

        optimizer = torch.optim.Adam(gin.parameters(), lr=1e-2)
        for _ in range(20):
            optimizer.zero_grad()
            alpha, correction = gin(gf, cp)
            # Artificial loss that pushes correction hard
            loss = -correction.sum()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            alpha, _ = gin(gf, cp)
        assert (alpha >= 0).all(), f"alpha has values < 0: min = {alpha.min().item()}"
        assert (alpha <= 1).all(), f"alpha has values > 1: max = {alpha.max().item()}"

    def test_gin_dynamic_n_features(self, synthetic_data):
        """GIN works with n_features=73 (e.g. when dN/dS is included)."""
        gin = GeneIntegrationNetwork(n_features=73, n_hidden=32)
        G = synthetic_data["n_genes"]
        gf_73 = torch.randn(G, 73)
        cp = synthetic_data["conservation_prior"]

        alpha, correction = gin(gf_73, cp)
        assert alpha.shape == (G,)
        assert correction.shape == (G,)

    # --- Task 8.1: Expanded GIN tests ---

    @pytest.mark.parametrize("G", [100, 500, 15000])
    def test_output_shapes_varied_G(self, G):
        """alpha and correction shapes match (G,) for varied gene counts."""
        n_features = 72
        gin = GeneIntegrationNetwork(n_features=n_features, n_hidden=32)
        gf = torch.randn(G, n_features)
        cp = torch.rand(G).clamp(0.05, 0.95)

        alpha, correction = gin(gf, cp)

        assert alpha.shape == (G,), f"alpha shape {alpha.shape} != ({G},)"
        assert correction.shape == (G,), f"correction shape {correction.shape} != ({G},)"

    def test_alpha_near_prior_at_init(self, synthetic_data):
        """At init, alpha is within atol=0.05 of the conservation prior."""
        gin = GeneIntegrationNetwork(n_features=72, n_hidden=32)
        gf = synthetic_data["gene_features"]
        cp = synthetic_data["conservation_prior"]

        with torch.no_grad():
            alpha, _ = gin(gf, cp)

        assert torch.allclose(alpha, cp, atol=0.05), (
            f"alpha not near prior at init: max diff = {(alpha - cp).abs().max().item():.6f}"
        )

    def test_correction_near_zero_at_init(self, synthetic_data):
        """At init, correction absolute max is < 0.5."""
        gin = GeneIntegrationNetwork(n_features=72, n_hidden=32)
        gf = synthetic_data["gene_features"]
        cp = synthetic_data["conservation_prior"]

        with torch.no_grad():
            _, correction = gin(gf, cp)

        assert correction.abs().max().item() < 0.5, (
            f"correction abs max = {correction.abs().max().item():.6f}, expected < 0.5"
        )

    def test_gradient_flows_to_mlp(self, synthetic_data):
        """Backward from alpha produces nonzero grads on all correction_mlp params."""
        gin = GeneIntegrationNetwork(n_features=72, n_hidden=32)
        gf = synthetic_data["gene_features"]
        cp = synthetic_data["conservation_prior"]

        # Need a few optimizer steps so that the zero-init last layer
        # gets nonzero weights, enabling gradient flow to layer 0.
        optimizer = torch.optim.Adam(gin.parameters(), lr=1e-2)
        for _ in range(5):
            optimizer.zero_grad()
            alpha, correction = gin(gf, cp)
            loss = alpha.sum()
            loss.backward()
            optimizer.step()

        # Final check: all correction_mlp params have nonzero gradient
        optimizer.zero_grad()
        alpha, _ = gin(gf, cp)
        alpha.sum().backward()

        for name, param in gin.correction_mlp.named_parameters():
            assert param.grad is not None, (
                f"correction_mlp.{name} has no gradient from alpha"
            )
            assert param.grad.abs().sum().item() > 0, (
                f"correction_mlp.{name} gradient is all zeros from alpha"
            )

    def test_alpha_bounded_01(self, synthetic_data):
        """With extreme features (x10), alpha stays in (0, 1)."""
        gin = GeneIntegrationNetwork(n_features=72, n_hidden=32)
        gf = synthetic_data["gene_features"] * 10.0  # extreme scaling
        cp = synthetic_data["conservation_prior"]

        # Train a few steps with extreme features to push weights
        optimizer = torch.optim.Adam(gin.parameters(), lr=1e-2)
        for _ in range(20):
            optimizer.zero_grad()
            alpha, correction = gin(gf, cp)
            loss = -correction.abs().sum()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            alpha, _ = gin(gf, cp)

        assert (alpha >= 0).all(), f"alpha has values < 0: min = {alpha.min().item()}"
        assert (alpha <= 1).all(), f"alpha has values > 1: max = {alpha.max().item()}"

    def test_prior_sensitivity(self):
        """Different conservation priors produce different alphas."""
        G = 100
        n_features = 72
        gin = GeneIntegrationNetwork(n_features=n_features, n_hidden=32)
        gf = torch.randn(G, n_features)

        cp_low = torch.full((G,), 0.1)
        cp_high = torch.full((G,), 0.9)

        with torch.no_grad():
            alpha_low, _ = gin(gf, cp_low)
            alpha_high, _ = gin(gf, cp_high)

        # Higher prior should produce higher alpha (since correction starts near 0)
        assert alpha_high.mean().item() > alpha_low.mean().item(), (
            f"Higher prior did not produce higher alpha: "
            f"mean(alpha_high)={alpha_high.mean().item():.4f}, "
            f"mean(alpha_low)={alpha_low.mean().item():.4f}"
        )
        # They should not be identical
        assert not torch.allclose(alpha_low, alpha_high, atol=1e-4), (
            "Different priors produced identical alphas"
        )


# =========================================================================
# ConstrainedBatchOffset
# =========================================================================


class TestConstrainedBatchOffset:
    """Tests for the ConstrainedBatchOffset component."""

    def test_batch_offset_zero_sum(self, synthetic_data):
        """Within-species zero-sum constraint holds for every gene."""
        n_batch = synthetic_data["n_batch"]
        n_genes = synthetic_data["n_genes"]
        b2s = synthetic_data["batch_to_species_idx"]

        cbo = ConstrainedBatchOffset(n_batch, n_genes, b2s)
        # Perturb the raw parameters so the constraint is non-trivial
        with torch.no_grad():
            cbo.batch_offset_raw.normal_(std=0.5)

        constrained = cbo()  # (n_batch, n_genes)

        n_species = int(b2s.max().item()) + 1
        for s in range(n_species):
            mask = b2s == s
            species_sum = constrained[mask].sum(dim=0)  # (n_genes,)
            assert torch.allclose(species_sum, torch.zeros(n_genes), atol=1e-5), (
                f"species {s}: zero-sum violated, max abs sum = {species_sum.abs().max().item()}"
            )

    def test_batch_offset_single_batch_species(self):
        """Offset is identically zero for a species with only one batch."""
        # Species 0: batches 0,1  |  Species 1: batch 2 (single)
        b2s = torch.tensor([0, 0, 1], dtype=torch.long)
        cbo = ConstrainedBatchOffset(n_batch=3, n_genes=10, batch_to_species_idx=b2s)
        with torch.no_grad():
            cbo.batch_offset_raw.normal_(std=1.0)

        constrained = cbo()
        single_batch_offset = constrained[2]  # batch 2, the single-batch species
        assert torch.allclose(single_batch_offset, torch.zeros(10), atol=1e-6), (
            f"Single-batch species offset not zero: max abs = {single_batch_offset.abs().max().item()}"
        )

    def test_batch_offset_gradient_flow(self, synthetic_data):
        """Gradients flow to batch_offset_raw."""
        n_batch = synthetic_data["n_batch"]
        n_genes = synthetic_data["n_genes"]
        b2s = synthetic_data["batch_to_species_idx"]

        cbo = ConstrainedBatchOffset(n_batch, n_genes, b2s)
        # Initialize with random data so constrained is nonzero
        cbo.batch_offset_raw.data.normal_(0, 0.1)

        constrained = cbo()
        loss = constrained.pow(2).sum()
        loss.backward()

        assert cbo.batch_offset_raw.grad is not None, "No gradient on batch_offset_raw"
        assert cbo.batch_offset_raw.grad.abs().sum().item() > 0, (
            "batch_offset_raw gradient is all zeros"
        )

    def test_batch_offset_shape(self, synthetic_data):
        """Output shape is (n_batch, n_genes)."""
        n_batch = synthetic_data["n_batch"]
        n_genes = synthetic_data["n_genes"]
        b2s = synthetic_data["batch_to_species_idx"]

        cbo = ConstrainedBatchOffset(n_batch, n_genes, b2s)
        constrained = cbo()

        assert constrained.shape == (n_batch, n_genes), (
            f"shape {constrained.shape} != ({n_batch}, {n_genes})"
        )

    # --- Task 8.2: Expanded Batch Offset tests ---

    def test_zero_sum_per_species_perturbed(self, synthetic_data):
        """Sum of offsets per species is approximately 0 for each gene, with strong perturbation."""
        n_batch = synthetic_data["n_batch"]
        n_genes = synthetic_data["n_genes"]
        b2s = synthetic_data["batch_to_species_idx"]

        cbo = ConstrainedBatchOffset(n_batch, n_genes, b2s)
        # Strong perturbation to stress-test the constraint
        with torch.no_grad():
            cbo.batch_offset_raw.normal_(std=5.0)

        constrained = cbo()

        n_species = int(b2s.max().item()) + 1
        for s in range(n_species):
            mask = b2s == s
            species_sum = constrained[mask].sum(dim=0)  # (n_genes,)
            assert torch.allclose(species_sum, torch.zeros(n_genes), atol=1e-5), (
                f"species {s}: zero-sum violated with strong perturbation, "
                f"max abs sum = {species_sum.abs().max().item()}"
            )

    def test_single_batch_species_zero_extended(self):
        """Offset is identically zero for species with only one batch, multi-species setup."""
        # Species 0: batches 0,1,2 (3 batches)
        # Species 1: batch 3 (single)
        # Species 2: batches 4,5 (2 batches)
        b2s = torch.tensor([0, 0, 0, 1, 2, 2], dtype=torch.long)
        n_genes = 50
        cbo = ConstrainedBatchOffset(n_batch=6, n_genes=n_genes, batch_to_species_idx=b2s)
        with torch.no_grad():
            cbo.batch_offset_raw.normal_(std=2.0)

        constrained = cbo()
        single_batch_offset = constrained[3]  # batch 3, the single-batch species 1
        assert torch.allclose(single_batch_offset, torch.zeros(n_genes), atol=1e-6), (
            f"Single-batch species offset not zero: max abs = {single_batch_offset.abs().max().item()}"
        )

    def test_gradient_flows_from_offset(self, synthetic_data):
        """Backward from constrained offset reaches batch_offset_raw."""
        n_batch = synthetic_data["n_batch"]
        n_genes = synthetic_data["n_genes"]
        b2s = synthetic_data["batch_to_species_idx"]

        cbo = ConstrainedBatchOffset(n_batch, n_genes, b2s)
        with torch.no_grad():
            cbo.batch_offset_raw.normal_(std=0.5)

        constrained = cbo()
        # Use pow(2).sum() instead of .sum() because the zero-sum constraint
        # makes d(sum)/d(raw) = 0, but d(sum_of_squares)/d(raw) != 0.
        loss = constrained.pow(2).sum()
        loss.backward()

        assert cbo.batch_offset_raw.grad is not None, "No gradient on batch_offset_raw"
        assert cbo.batch_offset_raw.grad.abs().sum().item() > 0, (
            "batch_offset_raw gradient is all zeros"
        )

    def test_output_shape_varied(self):
        """Output shape is (Nb, G) for various configurations."""
        for n_batch, n_genes in [(4, 200), (10, 50), (2, 1000)]:
            b2s = torch.zeros(n_batch, dtype=torch.long)  # all same species
            cbo = ConstrainedBatchOffset(n_batch, n_genes, b2s)
            constrained = cbo()
            assert constrained.shape == (n_batch, n_genes), (
                f"shape {constrained.shape} != ({n_batch}, {n_genes})"
            )

    def test_nonzero_for_multi_batch_species(self, synthetic_data):
        """Multi-batch species has nonzero offsets when raw params are nonzero."""
        n_batch = synthetic_data["n_batch"]
        n_genes = synthetic_data["n_genes"]
        b2s = synthetic_data["batch_to_species_idx"]

        cbo = ConstrainedBatchOffset(n_batch, n_genes, b2s)
        # Set raw to nonzero values that differ between batches of the same species
        with torch.no_grad():
            cbo.batch_offset_raw.normal_(std=1.0)

        constrained = cbo()

        # Find a species with multiple batches
        n_species = int(b2s.max().item()) + 1
        for s in range(n_species):
            mask = b2s == s
            if mask.sum() > 1:
                species_offsets = constrained[mask]  # (n_batches_in_species, n_genes)
                # With different raw values per batch, constrained offsets should be nonzero
                assert species_offsets.abs().max().item() > 1e-6, (
                    f"Multi-batch species {s} has all-zero constrained offsets "
                    f"despite nonzero raw params"
                )
                break


# =========================================================================
# LibrarySizeEncoder
# =========================================================================


class TestLibrarySizeEncoder:
    """Tests for the LibrarySizeEncoder component."""

    def test_library_encoder_shapes(self, synthetic_data):
        """ql_mean and ql_var both have shape (B, 1)."""
        n_genes = synthetic_data["n_genes"]
        n_batch = synthetic_data["n_batch"]
        B = synthetic_data["batch_dict"]["X"].shape[0]

        encoder = LibrarySizeEncoder(n_input=n_genes, n_batch=n_batch)
        x = synthetic_data["batch_dict"]["X"]
        batch_idx = synthetic_data["batch_dict"]["batch"].squeeze(-1).long()
        batch_oh = F.one_hot(batch_idx, n_batch).float()

        ql_mean, ql_var = encoder(x, batch_oh)

        assert ql_mean.shape == (B, 1), f"ql_mean shape {ql_mean.shape} != ({B}, 1)"
        assert ql_var.shape == (B, 1), f"ql_var shape {ql_var.shape} != ({B}, 1)"

    def test_library_encoder_positive_var(self, synthetic_data):
        """ql_var is always strictly positive (softplus output)."""
        n_genes = synthetic_data["n_genes"]
        n_batch = synthetic_data["n_batch"]

        encoder = LibrarySizeEncoder(n_input=n_genes, n_batch=n_batch)
        x = synthetic_data["batch_dict"]["X"]
        batch_idx = synthetic_data["batch_dict"]["batch"].squeeze(-1).long()
        batch_oh = F.one_hot(batch_idx, n_batch).float()

        ql_mean, ql_var = encoder(x, batch_oh)

        assert (ql_var > 0).all(), f"ql_var has non-positive values: min = {ql_var.min().item()}"
