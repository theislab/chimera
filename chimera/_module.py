"""Chimera v2.4 -- Core PyTorch modules for cross-species integration.

Two variants:
  1. ChimeraMinimalModule: fixed alpha = conservation prior, no GIN / FQR / anchor.
     Validates the residual-decoder + dual-loss concept.
  2. ChimeraModule: full v2.4 with GIN, FQR, anchor, dual-h, batch offset,
     S_g EMA, adversarial support.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.distributions import NegativeBinomial
from scvi.module.base import LossOutput

from chimera._components import (
    ConstrainedBatchOffset,
    GeneIntegrationNetwork,
    LibrarySizeEncoder,
)
from chimera._constants import (
    DROPOUT_RATE,
    FREE_BITS,
    LAMBDA_MMD,
    MIN_CELLS_FQR,
    N_HIDDEN,
    N_LATENT,
    SG_EMA_BETA,
    WARMUP_FQR_EPOCHS,
)


# ---------------------------------------------------------------------------
# ChimeraMinimalModule
# ---------------------------------------------------------------------------


class ChimeraMinimalModule(nn.Module):
    """Minimal Chimera: fixed alpha, residual decoder, dual NB loss.

    No GIN, no FQR, no anchor, no adversarial. Serves as the simplest
    configuration validating the core residual-decoder + dual-loss idea.

    Parameters
    ----------
    n_genes
        Number of genes (output dimension).
    n_species
        Number of species.
    n_batch
        Number of batches across all species.
    n_latent
        Latent space dimensionality.
    n_hidden
        Hidden layer width for encoder / decoder MLPs.
    conservation_prior
        Per-gene conservation prior in (0, 1), shape ``(G,)``.
    lambda_aux
        Weight for L_shared_aux.
    lambda_res
        Weight for L_residual_sparsity (L1).
    """

    def __init__(
        self,
        n_genes: int,
        n_species: int,
        n_batch: int,
        n_latent: int = N_LATENT,
        n_hidden: int = N_HIDDEN,
        conservation_prior: Optional[torch.Tensor] = None,
        lambda_aux: float = 0.3,
        lambda_res: float = 0.05,
    ) -> None:
        super().__init__()

        self.n_genes = n_genes
        self.n_species = n_species
        self.n_batch = n_batch
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.lambda_aux = lambda_aux
        self.lambda_res = lambda_res
        self.free_bits: float = FREE_BITS

        # Alpha: fixed, non-learnable
        if conservation_prior is None:
            conservation_prior = torch.full((n_genes,), 0.5)
        self.register_buffer("alpha", conservation_prior.detach().clone())

        # --- Encoder ---
        enc_input_dim = n_genes + n_species + n_batch
        self.encoder = nn.Sequential(
            nn.Linear(enc_input_dim, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )
        self.z_mean_linear = nn.Linear(n_hidden, n_latent)
        self.z_logvar_linear = nn.Linear(n_hidden, n_latent)

        # --- Decoder hidden ---
        self.decoder_hidden = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
        )

        # FM-F GUARD: verify decoder_hidden does NOT inject species/batch covariates
        if hasattr(self.decoder_hidden, 'inject_covariates'):
            assert not self.decoder_hidden.inject_covariates, (
                "FATAL: decoder_hidden has inject_covariates=True. "
                "This would silently inject species into the shared pathway."
            )
        # Parameter-count guard: first layer input dim must be n_latent (no covariates)
        first_layer = list(self.decoder_hidden.parameters())[0]
        assert first_layer.shape[1] == self.n_latent, (
            f"decoder_hidden first layer input dim = {first_layer.shape[1]}, "
            f"expected {self.n_latent}. Covariates may be injected."
        )

        # --- Output heads ---
        self.output_shared = nn.Linear(n_hidden, n_genes)  # no covariates
        self.output_residual = nn.Linear(n_hidden + n_species + n_batch, n_genes)
        # Small init for residual head
        nn.init.normal_(self.output_residual.weight, std=0.01)
        nn.init.zeros_(self.output_residual.bias)

        # --- Dispersion (per-gene only in minimal) ---
        self.dispersion = nn.Parameter(torch.zeros(n_genes))

    def forward(
        self,
        tensors: Dict[str, torch.Tensor],
        compute_loss: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        tensors
            Dictionary with keys ``"X"``, ``"species"``/``"labels"``,
            ``"batch"``, and optionally ``"library_size"``.
        compute_loss
            Unused (always computes loss); kept for API compatibility.
        **kwargs
            Absorbed for compatibility with scvi-tools callers.

        Returns
        -------
        loss
            Scalar loss tensor.
        outputs
            Dictionary of all intermediate tensors and loss components.
        """
        x = tensors["X"]  # (B, G)
        # Species may come as "species" (manual dict) or "labels" (scvi DataLoader)
        species_key = "labels" if "labels" in tensors else "species"
        species_idx = tensors[species_key].squeeze(-1).long()  # (B,)
        batch_idx = tensors["batch"].squeeze(-1).long()  # (B,)
        lib_key = "library_size" if "library_size" in tensors else "X"
        if lib_key == "X":
            library_size = x.sum(-1)  # compute from counts
        else:
            library_size = tensors["library_size"].squeeze(-1)  # (B,)

        species_oh = F.one_hot(species_idx, self.n_species).float()  # (B, K)
        batch_oh = F.one_hot(batch_idx, self.n_batch).float()  # (B, n_batch)

        # ---- Encode ----
        enc_input = torch.cat([x, species_oh, batch_oh], dim=-1)
        h_enc = self.encoder(enc_input)
        z_mean = self.z_mean_linear(h_enc)
        z_logvar = self.z_logvar_linear(h_enc)
        z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_logvar)

        # ---- Decode ----
        h = self.decoder_hidden(z)
        log_rate_shared = self.output_shared(h)
        log_rate_residual = self.output_residual(
            torch.cat([h, species_oh, batch_oh], dim=-1)
        )

        # ---- Combine ----
        alpha = self.alpha  # (G,)
        log_rate_full = log_rate_shared + (1.0 - alpha) * log_rate_residual
        mu_full = torch.exp(log_rate_full) * library_size.unsqueeze(-1)
        mu_shared = torch.exp(log_rate_shared) * library_size.unsqueeze(-1)
        theta = F.softplus(self.dispersion)  # (G,)

        # ---- Losses ----
        L_recon = -NegativeBinomial(mu=mu_full, theta=theta).log_prob(x).sum(-1).mean()
        kl_per_dim = 0.5 * (
            z_mean.pow(2) + torch.exp(z_logvar) - z_logvar - 1.0
        )  # (B, d)
        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        L_kl = kl_per_dim.sum(-1).mean()
        L_shared_aux = (
            -NegativeBinomial(mu=mu_shared, theta=theta.detach())
            .log_prob(x)
            .sum(-1)
            .mean()
        )
        L_sparsity = log_rate_residual.abs().sum(-1).mean()

        loss = (
            L_recon
            + L_kl
            + self.lambda_aux * L_shared_aux
            + self.lambda_res * L_sparsity
        )

        outputs = {
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z": z,
            "mu_full": mu_full,
            "mu_shared": mu_shared,
            "log_rate_shared": log_rate_shared,
            "log_rate_residual": log_rate_residual,
            "alpha": alpha,
            "theta": theta,
            "L_recon": L_recon,
            "L_kl": L_kl,
            "L_shared_aux": L_shared_aux,
            "L_residual_sparsity": L_sparsity,
            "loss": loss,
        }

        return loss, outputs

    def loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Optional[Dict] = None,
        generative_outputs: Optional[Dict] = None,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Compatibility shim for scvi-tools TrainingPlan introspection.

        The real loss computation happens in ``forward()``.
        """
        loss_scalar, outputs = self.forward(tensors)
        return LossOutput(
            loss=loss_scalar,
            reconstruction_loss=outputs["L_recon"],
            kl_local=outputs["L_kl"],
            extra_metrics={k: v for k, v in outputs.items() if k.startswith("L_")},
        )


# ---------------------------------------------------------------------------
# ChimeraModule (full v2.4)
# ---------------------------------------------------------------------------


class ChimeraModule(nn.Module):
    """Full Chimera v2.4 with GIN, FQR, anchor, dual-h, batch offset.

    Parameters
    ----------
    n_genes
        Number of genes (output dimension).
    n_species
        Number of species.
    n_batch
        Number of batches across all species.
    n_latent
        Latent space dimensionality.
    n_hidden
        Hidden layer width for encoder / decoder MLPs.
    gene_features
        Per-gene feature matrix, shape ``(G, n_features)``.
    conservation_prior
        Per-gene conservation prior in (0, 1), shape ``(G,)``.
    p_null
        Null-model expression proportions, shape ``(K, G)``.
    batch_to_species_idx
        Maps each batch index to its species, shape ``(n_batch,)``.
    use_observed_lib_size
        If True use observed library size; if False learn it.
    lambda_aux
        Weight for L_shared_aux.
    lambda_res
        Weight for L_residual_sparsity (L1).
    lambda_anchor
        Weight for L_alpha_anchor.
    lambda_fit
        Weight for L_fit_reward (FQR).
    lambda_disp
        Weight for dispersion regularization.
    """

    def __init__(
        self,
        n_genes: int,
        n_species: int,
        n_batch: int,
        n_latent: int = N_LATENT,
        n_hidden: int = N_HIDDEN,
        gene_features: Optional[torch.Tensor] = None,
        conservation_prior: Optional[torch.Tensor] = None,
        p_null: Optional[torch.Tensor] = None,
        batch_to_species_idx: Optional[torch.Tensor] = None,
        use_observed_lib_size: bool = True,
        lambda_aux: float = 0.3,
        lambda_res: float = 0.05,
        lambda_anchor: float = 0.005,
        lambda_fit: float = 0.01,
        lambda_disp: float = 1e-4,
        lambda_mmd: float = LAMBDA_MMD,
        mask_species_in_encoder: bool = False,
        mask_batch_in_encoder: bool = False,
    ) -> None:
        super().__init__()

        self.n_genes = n_genes
        self.n_species = n_species
        self.n_batch = n_batch
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.use_observed_lib_size = use_observed_lib_size
        self.lambda_aux = lambda_aux
        self.lambda_res = lambda_res
        self.lambda_anchor = lambda_anchor
        self.lambda_fit = lambda_fit
        self.lambda_disp = lambda_disp
        self.lambda_mmd = lambda_mmd
        self.warmup_fqr_epochs = WARMUP_FQR_EPOCHS
        self.current_epoch: int = 0
        self.free_bits: float = FREE_BITS
        self.kl_weight: float = 1.0  # set by ChimeraTrainingPlan during training
        # When True, the species one-hot is zeroed before being fed to the encoder.
        # The residual decoder still receives species one-hot for species-specific fitting.
        # This is a *spec deviation* for close-species datasets where adversarial training
        # alone cannot remove species from z within a reasonable training budget.
        self.mask_species_in_encoder = bool(mask_species_in_encoder)
        # When True, the batch one-hot is *also* zeroed in the encoder input.
        # On datasets where batch→species is 1:1 (our E1.4 invariant), masking
        # species alone leaves species inferable from batch_oh; both must be
        # masked to truly prevent species leakage into z. The batch_offset
        # submodule still receives batch_idx for its constrained per-batch
        # offset, so batch correction is preserved.
        self.mask_batch_in_encoder = bool(mask_batch_in_encoder)

        # ---- Buffers ----
        if conservation_prior is None:
            conservation_prior = torch.full((n_genes,), 0.5)
        self.register_buffer(
            "conservation_prior", conservation_prior.detach().clone()
        )

        if gene_features is None:
            gene_features = torch.zeros(n_genes, 72)
        self.register_buffer("gene_features", gene_features.detach().clone())

        if p_null is None:
            p_null = torch.full((n_species, n_genes), 1e-6)
        self.register_buffer("p_null", p_null.detach().clone())

        # S_g EMA buffer for FQR
        self.register_buffer("S_g_ema", torch.zeros(n_genes))

        # ---- GIN ----
        n_features = self.gene_features.shape[1]
        self.gin = GeneIntegrationNetwork(n_features=n_features, n_hidden=32)
        # GIN last layer is already zero-initialized in GeneIntegrationNetwork.__init__.
        # First-layer gradient is zero at init (expected: last-layer weights are zero,
        # so first-layer output has no effect). Gradients appear after first optimizer step.

        # ---- Batch offset ----
        if batch_to_species_idx is None:
            batch_to_species_idx = torch.zeros(n_batch, dtype=torch.long)
        self.batch_offset = ConstrainedBatchOffset(
            n_batch, n_genes, batch_to_species_idx
        )

        # ---- Encoder ----
        enc_input_dim = n_genes + n_species + n_batch
        self.encoder = nn.Sequential(
            nn.Linear(enc_input_dim, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )
        self.z_mean_linear = nn.Linear(n_hidden, n_latent)
        self.z_logvar_linear = nn.Linear(n_hidden, n_latent)

        # ---- Decoder hidden ----
        self.decoder_hidden = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
        )

        # FM-F GUARD: verify decoder_hidden does NOT inject species/batch covariates
        if hasattr(self.decoder_hidden, 'inject_covariates'):
            assert not self.decoder_hidden.inject_covariates, (
                "FATAL: decoder_hidden has inject_covariates=True. "
                "This would silently inject species into the shared pathway."
            )
        # Parameter-count guard: first layer input dim must be n_latent (no covariates)
        first_layer = list(self.decoder_hidden.parameters())[0]
        assert first_layer.shape[1] == self.n_latent, (
            f"decoder_hidden first layer input dim = {first_layer.shape[1]}, "
            f"expected {self.n_latent}. Covariates may be injected."
        )

        # ---- Output heads ----
        self.output_shared = nn.Linear(n_hidden, n_genes)
        # WARNING (Component 15.2, critical_reexamination_modules.md):
        # The residual head is intentionally a SINGLE LINEAR layer, not an MLP.
        # The alpha gradient monotonicity proof (Component 2b's weight-level
        # telescope) depends on linearity: d(L)/d(alpha) is monotonic in the
        # residual magnitude only because the residual head has no nonlinearity.
        # If upgrading to an MLP residual, the four-failure-mode framework and
        # alpha equilibrium analysis must be re-derived.  This is a known
        # limitation accepted for v1; see Caveat 32 for consequences.
        self.output_residual = nn.Linear(n_hidden + n_species + n_batch, n_genes)
        nn.init.normal_(self.output_residual.weight, std=0.01)
        nn.init.zeros_(self.output_residual.bias)

        # ---- Per-gene-per-batch dispersion ----
        self.dispersion = nn.Parameter(torch.zeros(n_batch, n_genes))

        # ---- Optional library size encoder ----
        self.lib_encoder: Optional[LibrarySizeEncoder] = None
        if not use_observed_lib_size:
            self.lib_encoder = LibrarySizeEncoder(n_genes, n_batch)

    # ------------------------------------------------------------------
    # FQR helpers
    # ------------------------------------------------------------------

    def _compute_S_g(
        self,
        mu_shared_cellstate: torch.Tensor,
        x: torch.Tensor,
        species_idx: torch.Tensor,
        theta_detached: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-gene fit-quality signal S_g.

        S_g = mean_over_species[ mean_over_cells[ log p(x|mu_shared,theta) - log p(x|p_null*lib,theta) ] ]

        Genes with fewer than MIN_CELLS_FQR expressing cells are masked to 0.

        Parameters
        ----------
        mu_shared_cellstate
            Shared-only reconstruction without batch offset, shape ``(B, G)``.
        x
            Observed counts, shape ``(B, G)``.
        species_idx
            Species indices, shape ``(B,)``.
        theta_detached
            Detached dispersion, shape ``(B, G)``.

        Returns
        -------
        S_g
            Per-gene fit-quality signal, shape ``(G,)``.
        """
        # Shared log-prob
        ll_shared = NegativeBinomial(
            mu=mu_shared_cellstate, theta=theta_detached
        ).log_prob(x)  # (B, G)

        # Null log-prob: p_null[species] * library_size
        lib = mu_shared_cellstate.sum(-1, keepdim=True).clamp(min=1.0)  # (B, 1)
        p_null_per_cell = self.p_null[species_idx]  # (B, G)
        mu_null = p_null_per_cell * lib  # (B, G)
        mu_null = mu_null.clamp(min=1e-8)
        ll_null = NegativeBinomial(
            mu=mu_null, theta=theta_detached
        ).log_prob(x)  # (B, G)

        # Delta log-likelihood per gene
        delta_ll = ll_shared - ll_null  # (B, G)

        # Sparse-gene mask: genes with < MIN_CELLS_FQR expressing cells
        expressing_mask = (x > 0).float()  # (B, G)
        n_expressing = expressing_mask.sum(0)  # (G,)
        sparse_mask = (n_expressing >= MIN_CELLS_FQR).float()  # (G,)

        # Per-species mean, then mean across species
        S_g = torch.zeros(self.n_genes, device=x.device, dtype=x.dtype)
        n_species_present = 0
        for s in range(self.n_species):
            mask_s = (species_idx == s)
            if mask_s.sum() == 0:
                continue
            S_g_s = delta_ll[mask_s].mean(0)  # (G,)
            S_g = S_g + S_g_s
            n_species_present += 1
        if n_species_present > 0:
            S_g = S_g / n_species_present

        # Apply ReLU (mandatory: S_g = ReLU(Q_g - Q_null_g), round2 spec)
        # Without ReLU, negative S_g causes FQR to push alpha downward
        # when shared decoder hasn't yet learned — fights the architecture.
        S_g = torch.clamp(S_g, min=0.0)

        # Apply sparse-gene mask
        S_g = S_g * sparse_mask

        return S_g

    def _update_S_g_ema(self, S_g: torch.Tensor) -> None:
        """Update the S_g exponential moving average (training only)."""
        beta = SG_EMA_BETA
        self.S_g_ema.mul_(beta).add_(S_g.detach(), alpha=1.0 - beta)

    def _encoder_input(
        self,
        x: torch.Tensor,
        species_oh: torch.Tensor,
        batch_oh: torch.Tensor,
    ) -> torch.Tensor:
        """Build the encoder input ``cat([x, species_oh, batch_oh])`` with the
        mask flags applied. Used by both the forward pass and every inference-
        time method in ``ChimeraModel`` so that training and inference encoder
        inputs are identical.
        """
        sp = torch.zeros_like(species_oh) if self.mask_species_in_encoder else species_oh
        ba = torch.zeros_like(batch_oh) if self.mask_batch_in_encoder else batch_oh
        return torch.cat([x, sp, ba], dim=-1)

    # ------------------------------------------------------------------
    # MMD (Maximum Mean Discrepancy) for species-mixed z
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Squared pairwise Euclidean distances (MPS-compatible, no cdist)."""
        # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 * x_i . y_j
        x_sq = (x * x).sum(-1, keepdim=True)  # (N, 1)
        y_sq = (y * y).sum(-1, keepdim=True)  # (M, 1)
        return x_sq + y_sq.T - 2.0 * x @ y.T  # (N, M)

    @staticmethod
    def _mmd_rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Multi-scale RBF kernel MMD^2 between two sample sets.

        Uses the median heuristic for bandwidth selection with 5 scales
        around the median pairwise distance. MPS-compatible (no cdist).

        Parameters
        ----------
        x, y
            Sample matrices of shape ``(N, d)`` and ``(M, d)``.

        Returns
        -------
        torch.Tensor
            Scalar MMD^2 estimate (non-negative in expectation).
        """
        xx = ChimeraModule._pairwise_sq_dist(x, x)
        yy = ChimeraModule._pairwise_sq_dist(y, y)
        xy = ChimeraModule._pairwise_sq_dist(x, y)

        # Median heuristic for bandwidth
        with torch.no_grad():
            n_x, n_y = x.shape[0], y.shape[0]
            # Upper-triangle of xx and yy (exclude diagonal self-distances)
            triu_xx = xx[torch.triu(torch.ones(n_x, n_x, device=x.device), diagonal=1).bool()]
            triu_yy = yy[torch.triu(torch.ones(n_y, n_y, device=y.device), diagonal=1).bool()]
            all_dists = torch.cat([triu_xx, triu_yy, xy.reshape(-1)])
            median_dist = all_dists.median().clamp(min=1e-5)

        bandwidths = [median_dist * f for f in (0.25, 0.5, 1.0, 2.0, 4.0)]

        mmd_sq = torch.tensor(0.0, device=x.device)
        for bw in bandwidths:
            k_xx = torch.exp(-xx / (2 * bw))
            k_yy = torch.exp(-yy / (2 * bw))
            k_xy = torch.exp(-xy / (2 * bw))
            # Unbiased estimator: exclude diagonals for k_xx and k_yy
            mmd_sq = mmd_sq + (
                (k_xx.sum() - n_x) / max(n_x * (n_x - 1), 1)
                + (k_yy.sum() - n_y) / max(n_y * (n_y - 1), 1)
                - 2.0 * k_xy.mean()
            )
        return mmd_sq

    def _compute_mmd(self, z: torch.Tensor, species_idx: torch.Tensor) -> torch.Tensor:
        """Compute MMD between all species pairs in the minibatch.

        Parameters
        ----------
        z
            Latent representations, shape ``(B, n_latent)``.
        species_idx
            Species indices, shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Mean MMD^2 across all species pairs present in the batch.
        """
        mmd_total = torch.tensor(0.0, device=z.device)
        n_pairs = 0

        for i in range(self.n_species):
            for j in range(i + 1, self.n_species):
                mask_i = species_idx == i
                mask_j = species_idx == j
                z_i = z[mask_i]
                z_j = z[mask_j]
                # Need at least 2 samples per species for unbiased MMD
                if z_i.shape[0] < 2 or z_j.shape[0] < 2:
                    continue
                mmd_total = mmd_total + self._mmd_rbf(z_i, z_j)
                n_pairs += 1

        if n_pairs == 0:
            return torch.tensor(0.0, device=z.device)
        return mmd_total / n_pairs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tensors: Dict[str, torch.Tensor],
        compute_loss: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass for the full Chimera v2.4 model.

        Parameters
        ----------
        tensors
            Dictionary with keys ``"X"``, ``"species"``/``"labels"``,
            ``"batch"``, and optionally ``"library_size"``.
        compute_loss
            Unused (always computes loss); kept for API compatibility.
        **kwargs
            Absorbed for compatibility with scvi-tools callers.

        Returns
        -------
        loss
            Scalar loss tensor.
        outputs
            Dictionary of all intermediate tensors and loss components.
        """
        # Step 1: Extract tensors
        x = tensors["X"]  # (B, G)
        # Species may come as "species" (manual dict) or "labels" (scvi DataLoader)
        species_key = "labels" if "labels" in tensors else "species"
        species_idx = tensors[species_key].squeeze(-1).long()  # (B,)
        batch_idx = tensors["batch"].squeeze(-1).long()  # (B,)
        lib_key = "library_size" if "library_size" in tensors else "X"
        if lib_key == "X":
            library_size = x.sum(-1)  # compute from counts
        else:
            library_size = tensors["library_size"].squeeze(-1)  # (B,)

        # Step 2: One-hot encode
        species_oh = F.one_hot(species_idx, self.n_species).float()  # (B, K)
        batch_oh = F.one_hot(batch_idx, self.n_batch).float()  # (B, n_batch)

        # Step 3: Encode
        enc_input = self._encoder_input(x, species_oh, batch_oh)
        h_enc = self.encoder(enc_input)
        z_mean = self.z_mean_linear(h_enc)
        z_logvar = self.z_logvar_linear(h_enc)
        z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_logvar)

        # Optional: library size encoder
        L_kl_l = torch.tensor(0.0, device=x.device)
        if self.lib_encoder is not None:
            ql_mean, ql_var = self.lib_encoder(x, batch_oh)
            library_size = torch.exp(ql_mean).squeeze(-1)  # (B,)
            # KL(q(l|x,b) || p(l|b)) — standard scVI library KL, no lambda
            # Prior: per-batch empirical mean/var of log(lib) from setup_anndata
            if hasattr(self, "local_l_mean") and hasattr(self, "local_l_var"):
                local_mean = self.local_l_mean[batch_idx]  # (B, 1)
                local_var = self.local_l_var[batch_idx]    # (B, 1)
                L_kl_l = 0.5 * (
                    (ql_var / local_var)
                    + (ql_mean - local_mean).pow(2) / local_var
                    - 1.0
                    + torch.log(local_var / ql_var)
                ).sum(-1).mean()
            else:
                # Fallback: KL against N(0, 1) prior
                ql_logvar = torch.log(ql_var)
                L_kl_l = 0.5 * (
                    ql_mean.pow(2) + ql_var - ql_logvar - 1.0
                ).sum(-1).mean()

        # Step 4: GIN -- per-gene alpha and correction
        # Config M short-circuits the GIN: `correction` is a learnable per-gene
        # parameter (added in _model.from_config when `per_gene_alpha=True`).
        # `alpha` is then computed in the same logit-space convention so that
        # L_alpha_anchor (correction.pow(2)) and L_fit_reward (-correction * S_g)
        # drive the per-gene parameter directly. See workflow.md §"GIN-input
        # ablation" and CHANGELOG for the rationale.
        if getattr(self, "_use_per_gene_alpha", False):
            correction = self.per_gene_correction  # (G,)
            alpha = torch.sigmoid(
                torch.logit(self.conservation_prior) + correction
            )  # (G,)
        else:
            alpha, correction = self.gin(
                self.gene_features, self.conservation_prior
            )  # (G,), (G,)

        # Step 5: Dual-h (Fix 15)
        h_full = self.decoder_hidden(z)  # (B, n_hidden)
        h_detached = self.decoder_hidden(z.detach())  # (B, n_hidden) -- no grad to encoder

        # Step 6: Shared head
        log_rate_shared_full = self.output_shared(h_full)  # (B, G)
        log_rate_shared_det = self.output_shared(h_detached)  # (B, G)

        # Step 7: Residual head
        # When species is masked from the encoder, also detach z for the
        # residual pathway.  The residual head still receives species_oh
        # explicitly, so it can produce species-specific corrections.  But
        # the encoder gradient from L_recon flows ONLY through the shared
        # head, removing the incentive to encode species in z.
        h_for_residual = h_detached if self.mask_species_in_encoder else h_full
        log_rate_residual = self.output_residual(
            torch.cat([h_for_residual, species_oh, batch_oh], dim=-1)
        )  # (B, G)

        # Step 8: Batch offset
        constrained_offset = self.batch_offset()  # (n_batch, G)
        log_rate_batch = constrained_offset[batch_idx]  # (B, G)

        # Step 9: Three mu predictions
        mu_full = (
            torch.exp(
                log_rate_shared_full
                + log_rate_batch
                + (1.0 - alpha) * log_rate_residual
            )
            * library_size.unsqueeze(-1)
        )
        mu_shared_only = (
            torch.exp(log_rate_shared_det + log_rate_batch)
            * library_size.unsqueeze(-1)
        )
        mu_shared_cellstate = (
            torch.exp(log_rate_shared_det) * library_size.unsqueeze(-1)
        )  # NO batch offset

        # Step 10: Per-gene-per-batch dispersion
        theta = F.softplus(self.dispersion[batch_idx])  # (B, G)

        # Step 11: Losses

        # L_recon
        L_recon = (
            -NegativeBinomial(mu=mu_full, theta=theta)
            .log_prob(x)
            .sum(-1)
            .mean()
        )

        # L_kl (with free-bits: Task 6.8)
        kl_per_dim = 0.5 * (
            z_mean.pow(2) + torch.exp(z_logvar) - z_logvar - 1.0
        )  # (B, d)
        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        L_kl = kl_per_dim.sum(-1).mean()

        # L_shared_aux (theta detached)
        L_shared_aux = (
            -NegativeBinomial(mu=mu_shared_only, theta=theta.detach())
            .log_prob(x)
            .sum(-1)
            .mean()
        )

        # L_residual_sparsity: .sum(-1).mean()
        L_residual_sparsity = log_rate_residual.abs().sum(-1).mean()

        # L_alpha_anchor: correction.pow(2).sum()
        L_alpha_anchor = correction.pow(2).sum()

        # L_dispersion regularization
        L_disp = self.dispersion.pow(2).mean()

        # FQR: L_fit_reward
        S_g = self._compute_S_g(
            mu_shared_cellstate, x, species_idx, theta.detach()
        )

        # Update EMA (training only, sparse-gene mask already applied)
        if self.training:
            self._update_S_g_ema(S_g)

        # FQR warmup check
        if self.current_epoch >= self.warmup_fqr_epochs:
            L_fit_reward = -(correction * self.S_g_ema).sum()
        else:
            L_fit_reward = torch.tensor(0.0, device=x.device)

        # MMD loss on z for species mixing
        if self.lambda_mmd > 0:
            L_mmd = self._compute_mmd(z, species_idx)
        else:
            L_mmd = torch.tensor(0.0, device=x.device)

        # KL weight (set by training plan during warmup; defaults to 1.0)
        kl_weight = getattr(self, "kl_weight", 1.0)

        # Total loss
        loss = (
            L_recon
            + kl_weight * L_kl
            + self.lambda_aux * L_shared_aux
            + self.lambda_res * L_residual_sparsity
            + self.lambda_anchor * L_alpha_anchor
            + self.lambda_fit * L_fit_reward
            + self.lambda_disp * L_disp
            + self.lambda_mmd * L_mmd
            + L_kl_l  # Library KL (no lambda; zero when using observed lib size)
        )

        outputs = {
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z": z,
            "alpha": alpha,
            "correction": correction,
            "mu_full": mu_full,
            "mu_shared_only": mu_shared_only,
            "mu_shared_cellstate": mu_shared_cellstate,
            "log_rate_shared_full": log_rate_shared_full,
            "log_rate_shared_det": log_rate_shared_det,
            "log_rate_residual": log_rate_residual,
            "log_rate_batch": log_rate_batch,
            "theta": theta,
            "S_g": S_g,
            "L_recon": L_recon,
            "L_kl": L_kl,
            "L_shared_aux": L_shared_aux,
            "L_residual_sparsity": L_residual_sparsity,
            "L_alpha_anchor": L_alpha_anchor,
            "L_fit_reward": L_fit_reward,
            "L_disp": L_disp,
            "L_mmd": L_mmd,
            "L_KL_l": L_kl_l,
            "loss": loss,
            "L_total": loss,
        }

        return loss, outputs

    def loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Optional[Dict] = None,
        generative_outputs: Optional[Dict] = None,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Compatibility shim for scvi-tools TrainingPlan introspection.

        The real loss computation happens in ``forward()``.
        """
        loss_scalar, outputs = self.forward(tensors)
        return LossOutput(
            loss=loss_scalar,
            reconstruction_loss=outputs["L_recon"],
            kl_local=outputs["L_kl"],
            extra_metrics={k: v for k, v in outputs.items() if k.startswith("L_")},
        )
