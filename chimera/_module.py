"""Chimera v2.4 -- Core PyTorch modules for cross-species integration.

Two variants:
  1. ChimeraMinimalModule: fixed alpha = conservation prior, no GIN / FQR / anchor.
     Validates the residual-decoder + dual-loss concept.
  2. ChimeraModule: full v2.4 with GIN, FQR, anchor, dual-h, batch offset,
     S_g EMA, adversarial support.
"""

import math
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
    LAMBDA_ANCHOR,
    LAMBDA_AUX,
    LAMBDA_DISP,
    LAMBDA_FIT,
    LAMBDA_MMD,
    LAMBDA_RES,
    LAMBDA_Z_BALANCE,
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
        lambda_aux: float = LAMBDA_AUX,
        lambda_res: float = LAMBDA_RES,
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
        lambda_aux: float = LAMBDA_AUX,
        lambda_res: float = LAMBDA_RES,
        lambda_anchor: float = LAMBDA_ANCHOR,
        lambda_fit: float = LAMBDA_FIT,
        lambda_disp: float = LAMBDA_DISP,
        lambda_mmd: float = LAMBDA_MMD,
        lambda_z_balance: float = LAMBDA_Z_BALANCE,
        mask_species_in_encoder: bool = True,
        mask_batch_in_encoder: bool = True,
        detach_z_residual: bool = True,
        gin_n_hidden_layers: int = 2,
        gin_n_hidden: int = 32,
        use_celltype_conditional_mmd: bool = False,
        use_vamp_prior: bool = True,
        n_vamp_components: int = 500,
        sg_ema_beta: float | None = None,
        lambda_anchor_neg_factor: float = 1.0,
        use_per_gene_bias: bool = False,
        lambda_mmd_aux: float = 0.0,
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
        # iter121: optional dual-MMD weight on the AUXILIARY MMD term. When
        # ``use_celltype_conditional_mmd=True`` AND ``lambda_mmd_aux > 0``, the
        # global (across-all-cells) MMD is added on top of the CT-conditional
        # MMD, producing species-mixing pressure on cells in unshared CTs that
        # CT-cond MMD cannot reach (memory `feedback_ctcond_mmd_4sp.md`: "either
        # use global MMD alone, or combine global + CT-cond (dual MMD). Never
        # drop global MMD entirely."). Default 0.0 = byte-identical to all
        # current best-knowns (testis iter109/iter111/iter115 use CT-cond ONLY).
        self.lambda_mmd_aux = float(lambda_mmd_aux)
        self.lambda_z_balance = lambda_z_balance
        self.warmup_fqr_epochs = WARMUP_FQR_EPOCHS
        self.current_epoch: int = 0
        # iter116: S_g EMA decay is configurable. SG_EMA_BETA=0.9 gives effective
        # ~10-batch averaging. Larger β (e.g. 0.97 → 33-batch) tests the per-
        # minibatch S_g noise hypothesis from iter115's 3-sp brainstorm: tighter
        # S_g_ema for rare neg-S_g testis genes (~1% class) may break the
        # universal E9.2 down architectural ceiling on testis 2-sp.
        self.sg_ema_beta: float = float(sg_ema_beta) if sg_ema_beta is not None else SG_EMA_BETA
        # iter118: L_alpha_anchor asymmetry factor. When 1.0 (default), the anchor is
        # the symmetric correction.pow(2).mean() preserved across all prior iterations.
        # When < 1.0, the negative-correction tail is penalized at `factor × λ_anchor`
        # instead of full λ_anchor, allowing rare neg-S_g genes' equilibrium correction
        # to grow proportionally larger in magnitude. Targets the universal testis E9.2
        # down architectural ceiling: with symmetric squashing, the MLP's joint-loss
        # minimum is "all small-positive" (mean correction on neg-S_g genes ≈ +0.353
        # at iter109); relaxing the negative-side cost lets the MLP find a separating
        # feature dimension during training.
        self.lambda_anchor_neg_factor: float = float(lambda_anchor_neg_factor)
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
        self.detach_z_residual = bool(detach_z_residual)
        # When True, alpha is detached from L_recon in the mu_full computation.
        # The GIN then learns ONLY from FQR (biological fit signal) + anchor
        # (regularization), not from L_recon's gradient through (1-α)×residual.
        # Under strong integration (VAMP+MMD), L_recon creates a uniform
        # downward force on all corrections that overwhelms FQR, collapsing
        # Δα to one-sided negative. Detaching α breaks this feedback loop and
        # allows bidirectional Δα: conserved genes (S_g>0) get α>prior,
        # divergent genes (S_g<0) get α<prior.  Validated iter81+.
        self.detach_alpha_from_recon = True
        # When True AND ``tensors["celltype"]`` is available at forward time,
        # MMD is evaluated within each cell type (species are matched per
        # cell type) rather than globally. See ChimeraModel.setup_anndata's
        # ``celltype_key`` argument and ``_compute_mmd_celltype_conditional``
        # below for the rationale. Falls back transparently to the global
        # MMD when the cell-type field is not registered.
        self.use_celltype_conditional_mmd = bool(use_celltype_conditional_mmd)
        # VAMP prior: multimodal p(z) induced by K learnable pseudo-inputs
        # passed through the same encoder. Replaces N(0, I) when active.
        # The prior itself carries the cross-species alignment inductive
        # bias (each shared celltype pulls toward one mode regardless of
        # species), so MMD pressure can be reduced or removed without
        # losing integration.
        self.use_vamp_prior = bool(use_vamp_prior)
        self.n_vamp_components = int(n_vamp_components)
        self.pseudo_inputs: Optional[nn.Parameter] = None
        if self.use_vamp_prior:
            # Random init; the factory can overwrite with data samples.
            # Scale ~0.5 so that after ReLU the values span a sensible range.
            self.pseudo_inputs = nn.Parameter(
                torch.randn(self.n_vamp_components, n_genes) * 0.5
            )

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
        # iter119: optional per-gene bias hybrid. When use_per_gene_bias=True,
        # GeneIntegrationNetwork outputs correction = MLP(features) + per_gene_bias,
        # giving each gene its own scalar dof to bypass MLP feature-space smoothness.
        # Default False = byte-identical to all prior best-knowns.
        self.use_per_gene_bias: bool = bool(use_per_gene_bias)
        self.gin = GeneIntegrationNetwork(
            n_features=n_features,
            n_hidden=int(gin_n_hidden),
            n_hidden_layers=gin_n_hidden_layers,
            n_genes=n_genes if self.use_per_gene_bias else None,
            use_per_gene_bias=self.use_per_gene_bias,
        )
        self.gin_n_hidden = int(gin_n_hidden)
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
        # Residual head: 2-layer MLP with bottleneck for species-specific corrections.
        # Uses n_hidden//2 as bottleneck to limit capacity — prevents the residual
        # from absorbing ALL species variation (which causes bimodal alpha collapse).
        # The bottleneck + L1 penalty together keep alpha smooth and continuous.
        residual_input_dim = n_hidden + n_species + n_batch
        residual_hidden = n_hidden // 2  # bottleneck
        self.output_residual = nn.Sequential(
            nn.Linear(residual_input_dim, residual_hidden),
            nn.LayerNorm(residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden, n_genes),
        )
        nn.init.normal_(self.output_residual[-1].weight, std=0.01)
        nn.init.zeros_(self.output_residual[-1].bias)

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

        # S_g is NOT ReLU-clamped.  Negative S_g (shared decoder worse than
        # null for this gene) is a valid biological signal: the gene is
        # divergent across species, so alpha should be pushed BELOW the
        # conservation prior.  Combined with detach_alpha_from_recon (which
        # removes L_recon's overwhelming uniform downward force on corrections),
        # unrectified S_g produces clean bidirectional Δα:
        #   S_g > 0  ⟹  correction > 0  ⟹  α > prior  (gene more conserved)
        #   S_g < 0  ⟹  correction < 0  ⟹  α < prior  (gene more divergent)
        # The FQR warmup (epochs 0–20) ensures the shared decoder is trained
        # before S_g_ema influences corrections.  The EMA (β=0.9) washes out
        # any early-training noise well before epoch 20.

        # Apply sparse-gene mask
        S_g = S_g * sparse_mask

        return S_g

    def _update_S_g_ema(self, S_g: torch.Tensor) -> None:
        """Update the S_g exponential moving average (training only)."""
        beta = self.sg_ema_beta
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
    # VAMP prior
    # ------------------------------------------------------------------

    def _encode_pseudo_inputs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass the K pseudo-inputs through the shared encoder.

        The pseudo-inputs are learnable "prototype" count vectors
        (shape ``(K, G)``). They go through the SAME encoder as real
        cells, with zeroed species / batch one-hots (matching the
        ``mask_species_in_encoder`` / ``mask_batch_in_encoder`` regime).
        A ``ReLU`` ensures non-negativity (cells have counts ≥ 0).

        BatchNorm is temporarily put into eval mode during this call so
        that (a) BN uses the accumulated running stats (trained on real
        cells) rather than pseudo-input batch stats, and (b) pseudo-input
        activations don't contaminate the running stats. Gradients still
        flow back to ``pseudo_inputs`` and all encoder parameters.
        """
        assert self.pseudo_inputs is not None, "VAMP prior not initialized"
        K = self.pseudo_inputs.shape[0]
        device = self.pseudo_inputs.device
        zeros_sp = torch.zeros(K, self.n_species, device=device)
        zeros_ba = torch.zeros(K, self.n_batch, device=device)
        pseudo = F.relu(self.pseudo_inputs)
        enc_input = self._encoder_input(pseudo, zeros_sp, zeros_ba)
        was_training = self.encoder.training
        self.encoder.eval()
        try:
            h_pseudo = self.encoder(enc_input)
        finally:
            self.encoder.train(mode=was_training)
        u_mean = self.z_mean_linear(h_pseudo)     # (K, d)
        u_logvar = self.z_logvar_linear(h_pseudo) # (K, d)
        return u_mean, u_logvar

    def _vamp_kl(
        self,
        z_sample: torch.Tensor,
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """MC estimate of KL(q(z|x) || p_vamp(z)), per cell.

        ``p_vamp(z) = (1/K) Σ_k q(z | u_k)`` where ``u_k`` is the k-th
        pseudo-input passed through the encoder. Using a single MC sample
        ``z_sample ~ q(z|x)``:

            log q(z|x) - log p_vamp(z)
        """
        u_mean, u_logvar = self._encode_pseudo_inputs()
        # Clamp logvars to keep exponents numerically safe.
        u_logvar = u_logvar.clamp(min=-10.0, max=10.0)
        zz_logvar = z_logvar.clamp(min=-10.0, max=10.0)

        u_var = torch.exp(u_logvar)
        # log q(z_sample | u_k) — Gaussian log-pdf, summed over dims.
        # z: (B, d); u_mean: (K, d) → broadcast to (B, K, d)
        B, d = z_sample.shape
        K = u_mean.shape[0]
        z_b = z_sample.unsqueeze(1)                # (B, 1, d)
        u_mean_b = u_mean.unsqueeze(0)             # (1, K, d)
        u_logvar_b = u_logvar.unsqueeze(0)         # (1, K, d)
        u_var_b = u_var.unsqueeze(0)               # (1, K, d)
        log_q_z_u = -0.5 * (
            math.log(2.0 * math.pi)
            + u_logvar_b
            + (z_b - u_mean_b).pow(2) / u_var_b
        ).sum(-1)                                  # (B, K)
        # log p_vamp(z_sample) = logsumexp_k log_q_z_u − log K
        log_p_z = torch.logsumexp(log_q_z_u, dim=1) - math.log(float(K))  # (B,)

        # log q(z_sample | x) — same family, per-cell variance.
        z_var = torch.exp(zz_logvar)
        log_q_z_x = -0.5 * (
            math.log(2.0 * math.pi)
            + zz_logvar
            + (z_sample - z_mean).pow(2) / z_var
        ).sum(-1)                                  # (B,)

        return log_q_z_x - log_p_z                 # (B,)

    def _compute_mmd_celltype_conditional(
        self,
        z: torch.Tensor,
        species_idx: torch.Tensor,
        celltype_idx: torch.Tensor,
    ) -> torch.Tensor:
        """MMD between species pairs evaluated **within each cell type**.

        Rationale (derived iter44-46, 2026-04-21). Global MMD on `z` closes
        the aggregate species-distribution gap but does not enforce
        within-celltype species matching — it can be satisfied by
        species-sub-clustering that mirrors across celltypes. Under strong
        global MMD the shared decoder is also forced to predict species-
        average expression at each cell state, inflating the residual, which
        collapses `alpha` to a uniformly-low one-sided distribution and kills
        E3.4 / E4.2. Celltype-conditional MMD penalizes species mismatch
        only within each cell type, leaving the rest of `z` free to carry
        cell-state information that the shared decoder can exploit. No
        cell-type information is fed into the encoder, shared decoder, or
        residual decoder — cell type labels are used **exclusively** to
        group cells for the MMD grouping.

        Parameters
        ----------
        z
            Latent representations, shape ``(B, n_latent)``.
        species_idx
            Species indices, shape ``(B,)``.
        celltype_idx
            Cell-type indices, shape ``(B,)``. Values are the integer codes
            produced by scvi-tools ``CategoricalObsField``.

        Returns
        -------
        torch.Tensor
            Mean MMD^2 over (celltype, species-pair) groups that had enough
            cells of both species. Falls back to the global MMD when no
            group had ≥2 cells per species.
        """
        mmd_total = torch.tensor(0.0, device=z.device)
        n_groups = 0

        unique_cts = torch.unique(celltype_idx)
        for c in unique_cts.tolist():
            ct_mask = celltype_idx == c
            for i in range(self.n_species):
                for j in range(i + 1, self.n_species):
                    mask_i = ct_mask & (species_idx == i)
                    mask_j = ct_mask & (species_idx == j)
                    # Skip species-unique cell types transparently: if a
                    # cell type only exists in one species, no MMD pair is
                    # defined for it.
                    if mask_i.sum() < 2 or mask_j.sum() < 2:
                        continue
                    z_i = z[mask_i]
                    z_j = z[mask_j]
                    mmd_total = mmd_total + self._mmd_rbf(z_i, z_j)
                    n_groups += 1

        if n_groups == 0:
            # No celltype had both species in the batch — fall back so the
            # loss still exerts pressure. This happens on very small batches
            # with skewed species distributions.
            return self._compute_mmd(z, species_idx)
        return mmd_total / n_groups

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
        h_for_residual = h_detached if self.detach_z_residual else h_full
        log_rate_residual = self.output_residual(
            torch.cat([h_for_residual, species_oh, batch_oh], dim=-1)
        )  # (B, G)

        # Step 8: Batch offset
        constrained_offset = self.batch_offset()  # (n_batch, G)
        log_rate_batch = constrained_offset[batch_idx]  # (B, G)

        # Step 9: Three mu predictions
        # When detach_alpha_from_recon is True, alpha.detach() is used so that
        # L_recon gradient does NOT flow back to the GIN.  The GIN learns only
        # from FQR + anchor — a cleaner biological signal that preserves
        # bidirectional Δα under strong integration.
        alpha_for_recon = alpha.detach() if self.detach_alpha_from_recon else alpha
        mu_full = (
            torch.exp(
                log_rate_shared_full
                + log_rate_batch
                + (1.0 - alpha_for_recon) * log_rate_residual
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

        # L_kl — standard N(0, I) analytic form by default, VAMP-prior
        # Monte-Carlo form when active. Free-bits is applied per-cell in
        # the VAMP case (using n_latent × free_bits as the per-cell floor)
        # so it has a consistent interpretation in either regime.
        if self.use_vamp_prior and self.pseudo_inputs is not None:
            kl_per_cell = self._vamp_kl(z, z_mean, z_logvar)  # (B,)
            if self.free_bits > 0:
                # Per-cell floor matches the total per-cell free-bits budget
                # of the analytic N(0, I) form (n_latent × free_bits).
                kl_per_cell = torch.clamp(
                    kl_per_cell, min=float(self.n_latent) * self.free_bits
                )
            L_kl = kl_per_cell.mean()
        else:
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

        # L_alpha_anchor: per-gene mean (dataset-size invariant).
        # iter118: when lambda_anchor_neg_factor < 1.0, the negative-correction
        # half-axis is penalized at `factor × λ_anchor` (relaxed). At factor=1.0
        # this branch is byte-identical to the symmetric `correction.pow(2).mean()`
        # form preserved across all prior iterations. See module docstring for
        # mechanism (testis E9.2 down architectural-ceiling attack).
        if self.lambda_anchor_neg_factor != 1.0:
            pos_sq = F.relu(correction).pow(2)
            neg_sq = F.relu(-correction).pow(2)
            L_alpha_anchor = (pos_sq + self.lambda_anchor_neg_factor * neg_sq).mean()
        else:
            L_alpha_anchor = correction.pow(2).mean()

        # L_dispersion regularization
        L_disp = self.dispersion.pow(2).mean()

        # FQR: L_fit_reward
        S_g = self._compute_S_g(
            mu_shared_cellstate, x, species_idx, theta.detach()
        )

        # Update EMA (training only, sparse-gene mask already applied)
        if self.training:
            self._update_S_g_ema(S_g)

        # FQR warmup check (per-gene mean, dataset-size invariant)
        if self.current_epoch >= self.warmup_fqr_epochs:
            L_fit_reward = -(correction * self.S_g_ema).mean()
        else:
            L_fit_reward = torch.tensor(0.0, device=x.device)

        # MMD loss on z for species mixing. When celltype-conditional MMD
        # is enabled AND a cell-type categorical field was registered via
        # setup_anndata(celltype_key=...), match species within each cell
        # type instead of globally. This directly targets E8.4 without the
        # alpha-collapse failure mode that strong global MMD produces.
        # iter121: optional dual-MMD — when ``lambda_mmd_aux > 0`` AND CT-cond
        # is the primary mode, also add the global MMD on top. This restores
        # species-mixing pressure on cells in UNSHARED CTs (which CT-cond MMD
        # cannot reach), the failure mode that collapsed iter107 (alpha
        # frozen at prior, adv leaked).
        celltype_idx_raw = tensors.get("celltype")
        if self.lambda_mmd > 0:
            if (
                self.use_celltype_conditional_mmd
                and celltype_idx_raw is not None
            ):
                celltype_idx = celltype_idx_raw.squeeze(-1).long()
                L_mmd = self._compute_mmd_celltype_conditional(
                    z, species_idx, celltype_idx
                )
            else:
                L_mmd = self._compute_mmd(z, species_idx)
        else:
            L_mmd = torch.tensor(0.0, device=x.device)
        # iter121 auxiliary MMD term (default 0.0 → no extra term, byte-identical).
        if (
            self.lambda_mmd_aux > 0
            and self.use_celltype_conditional_mmd
            and celltype_idx_raw is not None
        ):
            L_mmd_aux = self._compute_mmd(z, species_idx)
        else:
            L_mmd_aux = torch.tensor(0.0, device=x.device)

        # L_z_balance: penalize imbalanced per-dim variance in z_mean.
        # Under VAMP prior, the per-cell KL provides no per-dim regularization
        # (the mixture is too flexible), allowing some dims to explode (var=30)
        # while others near-collapse (var=0.007). This creates thin/elongated
        # UMAP clusters and wastes latent capacity. The balance loss penalizes
        # the std of log-variances across dims, encouraging all dims to carry
        # similar information.
        if self.lambda_z_balance > 0 and z_mean.shape[0] > 1:
            var_per_dim = z_mean.var(dim=0).clamp(min=1e-6)  # (d,)
            L_z_balance = torch.log(var_per_dim).std()
        else:
            L_z_balance = torch.tensor(0.0, device=x.device)

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
            + self.lambda_mmd_aux * L_mmd_aux  # iter121 dual-MMD aux term (0 when off)
            + self.lambda_z_balance * L_z_balance
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
            "L_mmd_aux": L_mmd_aux,
            "L_z_balance": L_z_balance,
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
