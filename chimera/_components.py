"""Chimera v2.4 — Core PyTorch modules for cross-species integration.

This module contains three standalone components:

1. GeneIntegrationNetwork (GIN): computes per-gene integration coefficients
   alpha_g via a prior-plus-correction architecture.
2. ConstrainedBatchOffset: learns per-batch per-gene offsets with a zero-sum
   constraint within each species.
3. LibrarySizeEncoder: a small encoder for learned library size when
   library-size heterogeneity across species exceeds 1.5x.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneIntegrationNetwork(nn.Module):
    """Per-gene integration coefficient network (prior + learned correction).

    Computes alpha_g = sigmoid(logit(conservation_prior_g) + MLP(gene_features_g)).
    The MLP is zero-initialized so that at the start of training, alpha equals the
    conservation prior exactly.

    Parameters
    ----------
    n_features
        Dimensionality of per-gene feature vectors (default 72: 64 ESM-2 PCA + 8
        scalars). May be 73 when dN/dS is included.
    n_hidden
        Hidden layer width of the correction MLP.
    """

    def __init__(
        self,
        n_features: int = 72,
        n_hidden: int = 32,
        hidden_activation: str = "tanh",
    ) -> None:
        super().__init__()
        # Tanh is used by default instead of ReLU (v2.4 spec).
        #
        # Empirical finding (2026-04-17): with ReLU the hidden activations are
        # non-negative, so the single 1×n_hidden output weight vector W2 can
        # only ever produce one-signed corrections for a given input sign. In
        # practice the L_recon gradient drives W2 uniformly negative early in
        # training; ReLU(hidden) * W2_negative ≤ 0 puts an upper bound of b2
        # (the output bias, also negative) on the correction — and therefore
        # on delta_alpha — for *every* gene, regardless of features. This
        # architecturally blocks FM4 resolution (`delta_alpha > 0`).
        #
        # tanh outputs are signed in (-1, 1), so even all-negative W2 can
        # produce positive corrections for genes whose hidden activations are
        # negative. The zero-output-at-init property (`alpha == prior` at
        # epoch 0) still holds because W2 is zero-initialized below.
        act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}[hidden_activation.lower()]
        self.correction_mlp = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            act_cls(),
            nn.Linear(n_hidden, 1),
        )
        # Zero-init last layer so correction starts at 0, alpha starts at prior.
        nn.init.zeros_(self.correction_mlp[-1].weight)
        nn.init.zeros_(self.correction_mlp[-1].bias)

    def forward(
        self,
        gene_features: torch.Tensor,
        conservation_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-gene integration coefficients.

        Parameters
        ----------
        gene_features
            StandardScaled per-gene feature matrix, shape ``(G, n_features)``.
        conservation_prior
            Per-gene conservation prior clamped to ``[0.05, 0.95]``, shape ``(G,)``.

        Returns
        -------
        alpha
            Per-gene integration coefficient in ``(0, 1)``, shape ``(G,)``.
        correction
            Raw correction output from the MLP (before adding to prior logit),
            shape ``(G,)``. Used by ``L_alpha_anchor`` and ``L_fit_reward``.
        """
        prior_logit = torch.logit(conservation_prior)  # (G,)
        correction = self.correction_mlp(gene_features).squeeze(-1)  # (G,)
        alpha = torch.sigmoid(prior_logit + correction)  # (G,)
        return alpha, correction


class ConstrainedBatchOffset(nn.Module):
    """Learned per-batch per-gene offset with a zero-sum constraint per species.

    For every species *s* and every gene *g*, the constrained offsets satisfy:

        sum_{b in species(s)} offset_constrained[b, g] == 0

    This is achieved by subtracting the per-species mean from the raw offsets.
    For single-batch species the constrained offset is identically zero.

    Parameters
    ----------
    n_batch
        Total number of batches across all species.
    n_genes
        Number of genes (columns of the offset matrix).
    batch_to_species_idx
        Integer tensor of shape ``(n_batch,)`` mapping each batch index to its
        species index.

    Notes
    -----
    Apply weight decay of ``1e-4`` to ``batch_offset_raw`` in the optimizer to
    damp null-space drift (see ``_constants.BATCH_OFFSET_WEIGHT_DECAY``).
    """

    def __init__(
        self,
        n_batch: int,
        n_genes: int,
        batch_to_species_idx: torch.Tensor,
    ) -> None:
        super().__init__()
        self.batch_offset_raw = nn.Parameter(torch.zeros(n_batch, n_genes))
        self.register_buffer(
            "batch_to_species_idx", batch_to_species_idx.long()
        )
        # Pre-compute the number of species and batch counts per species so that
        # the forward pass is a simple scatter + index operation.
        n_species = int(batch_to_species_idx.max().item()) + 1
        self.register_buffer("n_species", torch.tensor(n_species))
        # counts_per_species: (n_species,) — number of batches in each species
        counts = torch.zeros(n_species, dtype=torch.long)
        for s in range(n_species):
            counts[s] = (batch_to_species_idx == s).sum()
        self.register_buffer("counts_per_species", counts)

    def forward(self) -> torch.Tensor:
        """Return the zero-sum-constrained batch offsets.

        Returns
        -------
        torch.Tensor
            Constrained offset matrix of shape ``(n_batch, n_genes)``. For each
            species, offsets sum to zero across the batch dimension.
        """
        raw = self.batch_offset_raw  # (n_batch, n_genes)
        species_idx = self.batch_to_species_idx  # (n_batch,)
        n_species = int(self.n_species.item())

        # Compute per-species mean via scatter_add.
        # species_sum: (n_species, n_genes)
        species_sum = torch.zeros(
            n_species,
            raw.shape[1],
            dtype=raw.dtype,
            device=raw.device,
        )
        species_sum.scatter_add_(
            0, species_idx.unsqueeze(1).expand_as(raw), raw
        )
        # counts_per_species: (n_species,) -> (n_species, 1) for broadcasting
        counts = self.counts_per_species.unsqueeze(1).to(raw.dtype)  # (n_species, 1)
        species_mean = species_sum / counts  # (n_species, n_genes)

        # Subtract the species mean from each batch's raw offset.
        constrained = raw - species_mean[species_idx]  # (n_batch, n_genes)
        return constrained


class LibrarySizeEncoder(nn.Module):
    """Small encoder for learned library size.

    Used when library-size heterogeneity exceeds 1.5x across species. The encoder
    takes concatenated gene expression and batch one-hot as input and produces
    the mean and variance of a log-normal posterior over library size.

    The encoder does **not** see species identity -- only ``x`` and batch one-hot.

    Parameters
    ----------
    n_input
        Number of input genes (columns of the expression matrix).
    n_batch
        Number of batches (width of the one-hot encoding).
    """

    def __init__(self, n_input: int, n_batch: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input + n_batch, 128),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(128, 1)
        self.var_head = nn.Linear(128, 1)

    def forward(
        self,
        x: torch.Tensor,
        batch_oh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode library size posterior parameters.

        Parameters
        ----------
        x
            Gene expression matrix, shape ``(N, n_input)``.
        batch_oh
            One-hot batch encoding, shape ``(N, n_batch)``.

        Returns
        -------
        ql_mean
            Mean of the log-normal library-size posterior, shape ``(N, 1)``.
        ql_var
            Variance (positive) of the log-normal library-size posterior,
            shape ``(N, 1)``.
        """
        h = self.encoder(torch.cat([x, batch_oh], dim=-1))  # (N, 128)
        ql_mean = self.mean_head(h)  # (N, 1)
        ql_var = F.softplus(self.var_head(h))  # (N, 1) — always positive
        return ql_mean, ql_var
