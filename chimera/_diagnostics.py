"""Chimera v2.4 -- Data diagnostics for cross-species scRNA-seq integration.

Pre-training diagnostics to inform model configuration choices:
  - Library-size heterogeneity: whether to use observed or learned library sizes
  - Cell-type composition: species with dominant cell types that may bias integration

Post-training / in-training diagnostics (Task 8.6, 20 total):
  MUST-RUN (7):
  - Loss scale sanity check
  - Alpha histogram health check
  - Delta-alpha two-sidedness check
  - Shared decoder column norms
  - Residual magnitude vs alpha correlation
  - Sigmoid saturation check
  - Nonlinear z-leakage check

  SHOULD-RUN (13):
  - Per-dimension KL check
  - Gradient norm ratio
  - z-leakage diagnostic
  - Adversarial accuracy check
  - Posterior collapse check
  - GIN correction magnitude check
  - Shared decoder constant prediction check
  - Encoder gradient ratio check
  - Batch offset cell-type leak check
  - Batch offset biology absorption check
  - Dispersion stratification check
  - Loss convexity check
  - GIN gradient health check
  - Batch offset zero-sum check

  - Combined diagnostic runner (run_all_diagnostics)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn as nn

import anndata as ad

logger = logging.getLogger(__name__)


def library_size_heterogeneity_diagnostic(
    adata: ad.AnnData,
    species_key: str = "species",
) -> Dict:
    """Diagnose library-size heterogeneity across species.

    Computes per-species median library size and recommends whether to use
    observed or learned library sizes based on the max/min ratio.

    Parameters
    ----------
    adata
        AnnData with raw UMI counts in ``.X``.
    species_key
        Column in ``.obs`` identifying species.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``'per_species_median_lib'``: dict mapping species name to median
          library size
        - ``'max_min_ratio'``: ratio of largest to smallest median library size
        - ``'recommendation'``: one of ``'observed'``, ``'learned'``,
          ``'learned_with_warning'``
        - ``'use_observed_lib_size'``: bool, True if observed is recommended
        - ``'details'``: human-readable explanation string
    """
    species = adata.obs[species_key]
    unique_species = sorted(species.unique())

    X = adata.X
    # Compute per-cell library sizes.
    if sp.issparse(X):
        lib_sizes = np.asarray(X.sum(axis=1)).ravel()
    else:
        lib_sizes = np.asarray(X, dtype=np.float64).sum(axis=1)

    per_species_median = {}
    for sp_name in unique_species:
        mask = (species == sp_name).values
        sp_lib = lib_sizes[mask]
        per_species_median[sp_name] = float(np.median(sp_lib))

    medians = np.array(list(per_species_median.values()))
    min_median = medians.min()
    max_median = medians.max()

    if min_median > 0:
        ratio = max_median / min_median
    else:
        ratio = float("inf")

    # Decision thresholds (from LibrarySizeEncoder docstring in _components.py):
    #   ratio <= 1.5  -> use observed library size
    #   1.5 < ratio <= 5.0  -> use learned library size
    #   ratio > 5.0  -> use learned, but warn about extreme heterogeneity
    if ratio <= 1.5:
        recommendation = "observed"
        use_observed = True
        details = (
            f"Library-size ratio {ratio:.2f}x is within 1.5x threshold. "
            "Using observed library sizes is recommended."
        )
    elif ratio <= 5.0:
        recommendation = "learned"
        use_observed = False
        details = (
            f"Library-size ratio {ratio:.2f}x exceeds 1.5x threshold. "
            "Using learned library sizes (LibrarySizeEncoder) is recommended."
        )
    else:
        recommendation = "learned_with_warning"
        use_observed = False
        details = (
            f"Library-size ratio {ratio:.2f}x is very high (>5x). "
            "Using learned library sizes, but extreme heterogeneity may "
            "indicate data quality issues or protocol differences. "
            "Consider investigating further."
        )

    logger.info("Library size diagnostic: %s", details)

    return {
        "per_species_median_lib": per_species_median,
        "max_min_ratio": ratio,
        "recommendation": recommendation,
        "use_observed_lib_size": use_observed,
        "details": details,
    }


def cell_type_composition_diagnostic(
    adata: ad.AnnData,
    species_key: str = "species",
    cell_type_key: str = "cell_type",
    dominance_threshold: float = 0.8,
) -> Dict:
    """Diagnose cell-type composition imbalance across species.

    Builds a cell-type proportion x species cross-tabulation and flags species
    where a single cell type exceeds the dominance threshold.

    Parameters
    ----------
    adata
        AnnData with cell-type annotations.
    species_key
        Column in ``.obs`` identifying species.
    cell_type_key
        Column in ``.obs`` identifying cell types.
    dominance_threshold
        Fraction threshold above which a single cell type is considered
        dominant (default 0.8 = 80%).

    Returns
    -------
    dict
        Dictionary with keys:
        - ``'cross_tab'``: pd.DataFrame with species as rows and cell types as
          columns, values are proportions (summing to 1 per species)
        - ``'flagged_species'``: list of species names where a single cell type
          exceeds the dominance threshold
        - ``'dominant_types'``: dict mapping flagged species to their dominant
          cell type and its proportion
        - ``'details'``: human-readable summary string
    """
    species = adata.obs[species_key]
    cell_types = adata.obs[cell_type_key]

    # Build cross-tabulation: counts.
    cross_tab_counts = pd.crosstab(species, cell_types)

    # Normalize to proportions per species (row-wise).
    cross_tab = cross_tab_counts.div(cross_tab_counts.sum(axis=1), axis=0)

    flagged_species = []
    dominant_types = {}

    for sp_name in cross_tab.index:
        row = cross_tab.loc[sp_name]
        max_prop = row.max()
        if max_prop > dominance_threshold:
            dominant_ct = row.idxmax()
            flagged_species.append(sp_name)
            dominant_types[sp_name] = {
                "cell_type": dominant_ct,
                "proportion": float(max_prop),
            }

    if flagged_species:
        flagged_str = ", ".join(
            f"{sp} ({dominant_types[sp]['cell_type']}: "
            f"{dominant_types[sp]['proportion']:.1%})"
            for sp in flagged_species
        )
        details = (
            f"Species with dominant cell type (>{dominance_threshold:.0%}): "
            f"{flagged_str}. "
            "This may bias integration -- consider whether these species "
            "have genuine biological composition differences or sampling bias."
        )
    else:
        details = (
            f"No species has a single cell type exceeding "
            f"{dominance_threshold:.0%}. Cell-type composition is balanced."
        )

    logger.info("Composition diagnostic: %s", details)

    return {
        "cross_tab": cross_tab,
        "flagged_species": flagged_species,
        "dominant_types": dominant_types,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Post-training / in-training diagnostics (Task 8.6)
# ---------------------------------------------------------------------------


def loss_scale_sanity_check(
    module: nn.Module,
    batch_dict: Dict[str, torch.Tensor],
) -> Dict:
    """Check that weighted L1 and anchor terms are non-negligible vs L_recon.

    Catches the v2.2 reduction bug where regularisation terms vanish relative
    to the reconstruction loss, leaving the model effectively unregularised.

    Parameters
    ----------
    module
        A ``ChimeraModule`` instance (full v2.4).
    batch_dict
        Dictionary with keys ``"X"``, ``"species"``, ``"batch"``,
        ``"library_size"`` -- tensors for a single batch.

    Returns
    -------
    dict
        Keys: ``L1_vs_recon``, ``anchor_vs_recon``, ``L_recon``,
        ``weighted_L1``, ``weighted_anchor``, ``pass``, ``details``.
    """
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            _, outputs = module(batch_dict)
    finally:
        if was_training:
            module.train()

    L_recon = outputs["L_recon"].item()
    L_residual_sparsity = outputs["L_residual_sparsity"].item()
    L_alpha_anchor = outputs["L_alpha_anchor"].item()

    lambda_res = getattr(module, "lambda_res", 0.0)
    lambda_anchor = getattr(module, "lambda_anchor", 0.0)

    weighted_L1 = lambda_res * L_residual_sparsity
    weighted_anchor = lambda_anchor * L_alpha_anchor

    # Avoid division by zero
    if abs(L_recon) < 1e-12:
        L1_vs_recon = float("inf") if weighted_L1 > 0 else 0.0
        anchor_vs_recon = float("inf") if weighted_anchor > 0 else 0.0
    else:
        L1_vs_recon = weighted_L1 / abs(L_recon)
        anchor_vs_recon = weighted_anchor / abs(L_recon)

    passed = L1_vs_recon > 0.001 and anchor_vs_recon > 0.001

    if passed:
        details = (
            f"Loss scale OK: L1/recon={L1_vs_recon:.4f}, "
            f"anchor/recon={anchor_vs_recon:.4f}."
        )
    else:
        details = (
            f"Loss scale WARNING: L1/recon={L1_vs_recon:.6f}, "
            f"anchor/recon={anchor_vs_recon:.6f}. "
            "One or both ratios below 0.001 threshold -- regularisation may "
            "be ineffective (v2.2 reduction bug?)."
        )

    logger.info("Loss scale sanity check: %s", details)

    return {
        "L1_vs_recon": L1_vs_recon,
        "anchor_vs_recon": anchor_vs_recon,
        "L_recon": L_recon,
        "weighted_L1": weighted_L1,
        "weighted_anchor": weighted_anchor,
        "pass": passed,
        "details": details,
    }


def alpha_histogram_health_check(alpha: np.ndarray) -> Dict:
    """Check that the learned alpha distribution is healthy.

    A healthy alpha should have meaningful variance (not collapsed to a
    constant) and should not have too many genes pinned near 1.0 (which
    would suppress the residual pathway entirely).

    Parameters
    ----------
    alpha
        Per-gene alpha values as a 1-D numpy array, shape ``(G,)``.

    Returns
    -------
    dict
        Keys: ``frac_below_0.1``, ``frac_above_0.9``, ``mean``, ``std``,
        ``pass``, ``details``.
    """
    alpha = np.asarray(alpha, dtype=np.float64).ravel()

    frac_below_01 = float((alpha < 0.1).mean())
    frac_above_09 = float((alpha > 0.9).mean())
    mean_alpha = float(alpha.mean())
    std_alpha = float(alpha.std())

    passed = std_alpha > 0.05 and frac_above_09 < 0.01

    if passed:
        details = (
            f"Alpha healthy: mean={mean_alpha:.3f}, std={std_alpha:.3f}, "
            f"frac<0.1={frac_below_01:.3f}, frac>0.9={frac_above_09:.3f}."
        )
    else:
        reasons = []
        if std_alpha <= 0.05:
            reasons.append(f"std={std_alpha:.4f} <= 0.05 (collapsed)")
        if frac_above_09 >= 0.01:
            reasons.append(
                f"frac>0.9={frac_above_09:.3f} >= 0.01 "
                "(too many genes near 1.0)"
            )
        details = (
            f"Alpha WARNING: {'; '.join(reasons)}. "
            f"mean={mean_alpha:.3f}, frac<0.1={frac_below_01:.3f}."
        )

    logger.info("Alpha histogram check: %s", details)

    return {
        "frac_below_0.1": frac_below_01,
        "frac_above_0.9": frac_above_09,
        "mean": mean_alpha,
        "std": std_alpha,
        "pass": passed,
        "details": details,
    }


def delta_alpha_two_sidedness_check(
    alpha: np.ndarray,
    conservation_prior: np.ndarray,
) -> Dict:
    """Check that GIN corrections are two-sided (FM4 check).

    The GIN should both increase and decrease alpha relative to the prior.
    If corrections are entirely one-sided (all negative), the network is
    not learning gene-specific adjustments.

    Parameters
    ----------
    alpha
        Learned per-gene alpha, shape ``(G,)``.
    conservation_prior
        Per-gene conservation prior, shape ``(G,)``.

    Returns
    -------
    dict
        Keys: ``frac_positive``, ``frac_negative``, ``mean_positive``,
        ``mean_negative``, ``pass``, ``details``.
    """
    alpha = np.asarray(alpha, dtype=np.float64).ravel()
    conservation_prior = np.asarray(conservation_prior, dtype=np.float64).ravel()

    delta = alpha - conservation_prior
    frac_positive = float((delta > 0).mean())
    frac_negative = float((delta < 0).mean())

    pos_mask = delta > 0
    neg_mask = delta < 0

    mean_positive = float(delta[pos_mask].mean()) if pos_mask.any() else 0.0
    mean_negative = float(delta[neg_mask].mean()) if neg_mask.any() else 0.0

    passed = frac_positive > 0.05

    if passed:
        details = (
            f"Delta-alpha two-sided: frac_pos={frac_positive:.3f}, "
            f"frac_neg={frac_negative:.3f}, "
            f"mean_pos={mean_positive:.4f}, mean_neg={mean_negative:.4f}."
        )
    else:
        details = (
            f"Delta-alpha WARNING: frac_pos={frac_positive:.3f} <= 0.05. "
            f"GIN corrections are predominantly one-sided "
            f"(frac_neg={frac_negative:.3f}). FM4 violated."
        )

    logger.info("Delta-alpha two-sidedness: %s", details)

    return {
        "frac_positive": frac_positive,
        "frac_negative": frac_negative,
        "mean_positive": mean_positive,
        "mean_negative": mean_negative,
        "pass": passed,
        "details": details,
    }


def w_shared_column_norms(module: nn.Module) -> Dict:
    """Check shared decoder weight columns for collapse.

    If too many hidden-unit columns of ``output_shared.weight`` have near-zero
    norms, the shared pathway has effectively collapsed to a lower-dimensional
    representation, which degrades integration quality.

    Parameters
    ----------
    module
        A ``ChimeraModule`` or ``ChimeraMinimalModule`` instance.

    Returns
    -------
    dict
        Keys: ``norms`` (1-D numpy array, per-hidden-unit), ``min_norm``,
        ``max_norm``, ``frac_near_zero``, ``pass``, ``details``.
    """
    W = module.output_shared.weight.data  # (n_genes, n_hidden)
    norms = W.norm(dim=0).cpu().numpy()  # (n_hidden,)

    min_norm = float(norms.min())
    max_norm = float(norms.max())
    # Near-zero threshold: norm < 1e-3
    frac_near_zero = float((norms < 1e-3).mean())

    passed = frac_near_zero < 0.01

    if passed:
        details = (
            f"W_shared column norms OK: min={min_norm:.4f}, "
            f"max={max_norm:.4f}, frac_near_zero={frac_near_zero:.4f}."
        )
    else:
        details = (
            f"W_shared column norms WARNING: frac_near_zero={frac_near_zero:.3f} "
            f">= 0.01 (min={min_norm:.6f}, max={max_norm:.4f}). "
            "Shared decoder may have collapsed dimensions."
        )

    logger.info("W_shared column norms: %s", details)

    return {
        "norms": norms,
        "min_norm": min_norm,
        "max_norm": max_norm,
        "frac_near_zero": frac_near_zero,
        "pass": passed,
        "details": details,
    }


def residual_magnitude_vs_alpha(
    module: nn.Module,
    batch_dict: Dict[str, torch.Tensor],
) -> Dict:
    """Check expected L-shape: high alpha should correlate with low residual.

    A well-behaved model should show negative Pearson correlation between
    per-gene alpha and the mean absolute residual log-rate.

    Parameters
    ----------
    module
        A ``ChimeraModule`` instance (full v2.4).
    batch_dict
        Dictionary with keys ``"X"``, ``"species"``, ``"batch"``,
        ``"library_size"``.

    Returns
    -------
    dict
        Keys: ``correlation``, ``alpha_mean``, ``residual_mag_mean``,
        ``pass``, ``details``.
    """
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            _, outputs = module(batch_dict)
    finally:
        if was_training:
            module.train()

    alpha_np = outputs["alpha"].detach().cpu().numpy().ravel()
    residual_mag = (
        outputs["log_rate_residual"].abs().mean(dim=0).detach().cpu().numpy().ravel()
    )

    # Pearson correlation
    alpha_centered = alpha_np - alpha_np.mean()
    resid_centered = residual_mag - residual_mag.mean()

    denom = np.sqrt((alpha_centered ** 2).sum() * (resid_centered ** 2).sum())
    if denom < 1e-12:
        correlation = 0.0
    else:
        correlation = float((alpha_centered * resid_centered).sum() / denom)

    passed = correlation < -0.2

    if passed:
        details = (
            f"Residual vs alpha L-shape OK: correlation={correlation:.3f} "
            f"(< -0.2 threshold)."
        )
    else:
        details = (
            f"Residual vs alpha WARNING: correlation={correlation:.3f} "
            f">= -0.2 threshold. Expected negative correlation (L-shape) "
            f"between alpha and residual magnitude."
        )

    logger.info("Residual vs alpha: %s", details)

    return {
        "correlation": correlation,
        "alpha_mean": float(alpha_np.mean()),
        "residual_mag_mean": float(residual_mag.mean()),
        "pass": passed,
        "details": details,
    }


def per_dimension_kl_check(
    z_mean: np.ndarray,
    z_logvar: np.ndarray,
) -> Dict:
    """Check per-latent-dimension KL for posterior collapse or species encoding.

    Dimensions where KL < 0.01 indicate posterior collapse (the dimension
    is ignored). Dimensions where KL > 5 suggest the dimension may be
    encoding species identity rather than biological variation.

    Parameters
    ----------
    z_mean
        Posterior mean, shape ``(N, d)``.
    z_logvar
        Posterior log-variance, shape ``(N, d)``.

    Returns
    -------
    dict
        Keys: ``per_dim_kl`` (1-D numpy array), ``collapsed_dims`` (list of
        int), ``species_encoding_dims`` (list of int), ``n_collapsed``,
        ``n_species_encoding``, ``pass``, ``details``.
    """
    z_mean = np.asarray(z_mean, dtype=np.float64)
    z_logvar = np.asarray(z_logvar, dtype=np.float64)

    # Per-dimension KL: 0.5 * (mu^2 + exp(logvar) - logvar - 1), mean over cells
    per_dim_kl = 0.5 * (
        z_mean ** 2 + np.exp(z_logvar) - z_logvar - 1.0
    ).mean(axis=0)  # (d,)

    collapsed_dims = [int(i) for i in np.where(per_dim_kl < 0.01)[0]]
    species_encoding_dims = [int(i) for i in np.where(per_dim_kl > 5.0)[0]]

    n_collapsed = len(collapsed_dims)
    n_species_encoding = len(species_encoding_dims)

    passed = n_collapsed == 0 and n_species_encoding == 0

    if passed:
        details = (
            f"Per-dimension KL OK: all {len(per_dim_kl)} dimensions in "
            f"[0.01, 5.0] range. min={per_dim_kl.min():.4f}, "
            f"max={per_dim_kl.max():.4f}."
        )
    else:
        issues = []
        if n_collapsed > 0:
            issues.append(
                f"{n_collapsed} collapsed (KL<0.01): dims {collapsed_dims}"
            )
        if n_species_encoding > 0:
            issues.append(
                f"{n_species_encoding} species-encoding (KL>5): "
                f"dims {species_encoding_dims}"
            )
        details = (
            f"Per-dimension KL WARNING: {'; '.join(issues)}. "
            f"min={per_dim_kl.min():.4f}, max={per_dim_kl.max():.4f}."
        )

    logger.info("Per-dimension KL check: %s", details)

    return {
        "per_dim_kl": per_dim_kl,
        "collapsed_dims": collapsed_dims,
        "species_encoding_dims": species_encoding_dims,
        "n_collapsed": n_collapsed,
        "n_species_encoding": n_species_encoding,
        "pass": passed,
        "details": details,
    }


def gradient_norm_ratio(
    module: nn.Module,
    batch_dict: Dict[str, torch.Tensor],
) -> Dict:
    """Check gradient health: all encoder params should have non-zero grads.

    Runs a forward+backward pass with the total loss and verifies that all
    encoder parameters receive gradients. Also computes the ratio of max to
    min encoder gradient norms to detect pathological imbalances.

    Parameters
    ----------
    module
        A ``ChimeraModule`` or ``ChimeraMinimalModule`` instance.
    batch_dict
        Dictionary with keys ``"X"``, ``"species"``, ``"batch"``,
        ``"library_size"``.

    Returns
    -------
    dict
        Keys: ``encoder_grad_norms`` (dict of param name to norm),
        ``all_have_grad``, ``max_norm``, ``min_nonzero_norm``,
        ``ratio``, ``pass``, ``details``.
    """
    was_training = module.training
    module.train()

    # Zero out any existing gradients
    module.zero_grad()

    try:
        loss, _ = module(batch_dict)
        loss.backward()
    except Exception as e:
        if was_training:
            module.train()
        else:
            module.eval()
        return {
            "encoder_grad_norms": {},
            "all_have_grad": False,
            "max_norm": 0.0,
            "min_nonzero_norm": 0.0,
            "ratio": float("inf"),
            "pass": False,
            "details": f"Gradient computation failed: {e}",
        }

    # Collect encoder parameter gradient norms
    encoder_grad_norms = {}

    # Collect from encoder Sequential, z_mean_linear, z_logvar_linear
    for name, param in module.named_parameters():
        if not any(
            prefix in name
            for prefix in ("encoder.", "z_mean_linear.", "z_logvar_linear.")
        ):
            continue
        if param.grad is not None:
            grad_norm = float(param.grad.data.norm(2).item())
        else:
            grad_norm = 0.0
        encoder_grad_norms[name] = grad_norm

    # Zero grads to leave module clean
    module.zero_grad()

    if not was_training:
        module.eval()

    all_norms = list(encoder_grad_norms.values())
    if not all_norms:
        return {
            "encoder_grad_norms": encoder_grad_norms,
            "all_have_grad": False,
            "max_norm": 0.0,
            "min_nonzero_norm": 0.0,
            "ratio": float("inf"),
            "pass": False,
            "details": "No encoder parameters found.",
        }

    all_have_grad = all(n > 0 for n in all_norms)
    max_norm = max(all_norms)
    nonzero_norms = [n for n in all_norms if n > 0]
    min_nonzero_norm = min(nonzero_norms) if nonzero_norms else 0.0

    if min_nonzero_norm > 0:
        ratio = max_norm / min_nonzero_norm
    else:
        ratio = float("inf")

    passed = all_have_grad and ratio < 5.0

    if passed:
        details = (
            f"Gradient norms OK: all {len(all_norms)} encoder params have "
            f"grad, ratio={ratio:.2f} (< 5.0)."
        )
    else:
        issues = []
        if not all_have_grad:
            zero_params = [
                k for k, v in encoder_grad_norms.items() if v == 0
            ]
            issues.append(f"zero-grad params: {zero_params}")
        if ratio >= 5.0:
            issues.append(f"ratio={ratio:.2f} >= 5.0")
        details = (
            f"Gradient norm WARNING: {'; '.join(issues)}. "
            f"max={max_norm:.4f}, min_nonzero={min_nonzero_norm:.6f}."
        )

    logger.info("Gradient norm ratio: %s", details)

    return {
        "encoder_grad_norms": encoder_grad_norms,
        "all_have_grad": all_have_grad,
        "max_norm": max_norm,
        "min_nonzero_norm": min_nonzero_norm,
        "ratio": ratio,
        "pass": passed,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Additional post-training diagnostics (Task 8.6 expansion: 7 -> 20)
# ---------------------------------------------------------------------------


def z_leakage_diagnostic(
    mu_shared_only: np.ndarray,
    species_codes: np.ndarray,
    cell_type_labels: np.ndarray,
) -> Dict:
    """Check species invariance of mu_shared_only for shared cell types.

    For each cell type present in more than one species, compute mean
    mu_shared_only per species and measure cosine similarity.  Shared
    representations should be species-invariant, so similarity should be
    high.

    Parameters
    ----------
    mu_shared_only
        Per-cell shared-pathway reconstruction, shape ``(N, G)``.
    species_codes
        Integer species index per cell, shape ``(N,)``.
    cell_type_labels
        Cell-type label per cell (integer or string), shape ``(N,)``.

    Returns
    -------
    dict
        Keys: ``mean_cos``, ``per_celltype_cos``, ``pass``, ``details``.
    """
    mu_shared_only = np.asarray(mu_shared_only, dtype=np.float64)
    species_codes = np.asarray(species_codes).ravel()
    cell_type_labels = np.asarray(cell_type_labels).ravel()

    unique_cts = np.unique(cell_type_labels)
    unique_species = np.unique(species_codes)
    per_celltype_cos = {}

    for ct in unique_cts:
        ct_mask = cell_type_labels == ct
        species_present = np.unique(species_codes[ct_mask])
        if len(species_present) < 2:
            continue

        # Compute per-species centroid for this cell type
        centroids = []
        for sp in species_present:
            mask = ct_mask & (species_codes == sp)
            centroid = mu_shared_only[mask].mean(axis=0)
            centroids.append(centroid)

        # Pairwise cosine similarities
        cos_vals = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                a, b = centroids[i], centroids[j]
                denom = np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())
                if denom < 1e-12:
                    cos_vals.append(0.0)
                else:
                    cos_vals.append(float((a * b).sum() / denom))
        per_celltype_cos[str(ct)] = float(np.mean(cos_vals))

    if per_celltype_cos:
        mean_cos = float(np.mean(list(per_celltype_cos.values())))
    else:
        mean_cos = 1.0  # no shared cell types -- vacuously pass

    passed = mean_cos > 0.8

    if passed:
        details = (
            f"z-leakage OK: mean_cos={mean_cos:.3f} > 0.8 across "
            f"{len(per_celltype_cos)} shared cell types."
        )
    else:
        details = (
            f"z-leakage WARNING: mean_cos={mean_cos:.3f} <= 0.8. "
            f"Shared representation is not species-invariant for shared "
            f"cell types ({len(per_celltype_cos)} tested)."
        )

    logger.info("z-leakage diagnostic: %s", details)

    return {
        "mean_cos": mean_cos,
        "per_celltype_cos": per_celltype_cos,
        "pass": passed,
        "details": details,
    }


def adversarial_accuracy_check(
    z: np.ndarray,
    species_codes: np.ndarray,
    n_species: int,
) -> Dict:
    """Check that z does not encode species identity (adversarial probe).

    Trains a lightweight MLP classifier to predict species from z.
    Accuracy should be near chance (1/n_species).

    Parameters
    ----------
    z
        Latent representations, shape ``(N, d)``.
    species_codes
        Integer species label per cell, shape ``(N,)``.
    n_species
        Number of species.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``chance``, ``pass``, ``details``.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score

    z = np.asarray(z, dtype=np.float64)
    species_codes = np.asarray(species_codes, dtype=int).ravel()

    chance = 1.0 / n_species

    # Subsample if too many cells (cap at 5000 for speed)
    if len(z) > 5000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(z), 5000, replace=False)
        z = z[idx]
        species_codes = species_codes[idx]

    clf = MLPClassifier(
        hidden_layer_sizes=(32,),
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )

    scores = cross_val_score(clf, z, species_codes, cv=3, scoring="accuracy")
    accuracy = float(scores.mean())

    passed = accuracy < chance + 0.05

    if passed:
        details = (
            f"Adversarial accuracy OK: accuracy={accuracy:.3f} < "
            f"chance+0.05={chance + 0.05:.3f} (chance={chance:.3f})."
        )
    else:
        details = (
            f"Adversarial accuracy WARNING: accuracy={accuracy:.3f} >= "
            f"chance+0.05={chance + 0.05:.3f}. z may encode species identity."
        )

    logger.info("Adversarial accuracy check: %s", details)

    return {
        "accuracy": accuracy,
        "chance": chance,
        "pass": passed,
        "details": details,
    }


def sigmoid_saturation_check(alpha: np.ndarray) -> Dict:
    """Check alpha for saturation near 0/1 boundaries (Caveat 29).

    If too many genes have alpha near 0 (suppressing shared) or near 1
    (suppressing residual), the GIN sigmoid may be saturating.

    Parameters
    ----------
    alpha
        Per-gene alpha values, shape ``(G,)``.

    Returns
    -------
    dict
        Keys: ``frac_below_0.05``, ``frac_above_0.95``, ``pass``, ``details``.
    """
    alpha = np.asarray(alpha, dtype=np.float64).ravel()

    frac_below_005 = float((alpha < 0.05).mean())
    frac_above_095 = float((alpha > 0.95).mean())

    passed = frac_below_005 < 0.05 and frac_above_095 < 0.01

    if passed:
        details = (
            f"Sigmoid saturation OK: frac<0.05={frac_below_005:.4f}, "
            f"frac>0.95={frac_above_095:.4f}."
        )
    else:
        reasons = []
        if frac_below_005 >= 0.05:
            reasons.append(
                f"frac<0.05={frac_below_005:.4f} >= 0.05 (too many near 0)"
            )
        if frac_above_095 >= 0.01:
            reasons.append(
                f"frac>0.95={frac_above_095:.4f} >= 0.01 (too many near 1)"
            )
        details = (
            f"Sigmoid saturation WARNING (Caveat 29): {'; '.join(reasons)}."
        )

    logger.info("Sigmoid saturation check: %s", details)

    return {
        "frac_below_0.05": frac_below_005,
        "frac_above_0.95": frac_above_095,
        "pass": passed,
        "details": details,
    }


def gin_correction_magnitude_check(correction: np.ndarray) -> Dict:
    """Check GIN correction distribution (Caveat 28).

    The GIN should produce non-trivial corrections.  If max absolute
    correction is too small the GIN is not learning.

    Parameters
    ----------
    correction
        Per-gene GIN correction, shape ``(G,)``.

    Returns
    -------
    dict
        Keys: ``max_abs``, ``mean_abs``, ``std``, ``pass``, ``details``.
    """
    correction = np.asarray(correction, dtype=np.float64).ravel()

    max_abs = float(np.abs(correction).max())
    mean_abs = float(np.abs(correction).mean())
    std_c = float(correction.std())

    passed = max_abs > 0.1

    if passed:
        details = (
            f"GIN correction OK: max_abs={max_abs:.4f} > 0.1, "
            f"mean_abs={mean_abs:.4f}, std={std_c:.4f}."
        )
    else:
        details = (
            f"GIN correction WARNING (Caveat 28): max_abs={max_abs:.6f} <= 0.1. "
            f"GIN corrections may be too small to be effective "
            f"(mean_abs={mean_abs:.6f}, std={std_c:.6f})."
        )

    logger.info("GIN correction magnitude check: %s", details)

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "std": std_c,
        "pass": passed,
        "details": details,
    }


def nonlinear_z_leakage_check(
    z: np.ndarray,
    species_codes: np.ndarray,
    n_species: int,
) -> Dict:
    """Post-training nonlinear z-leakage check with deeper MLP.

    Trains a deeper MLP (64, 32) with 5-fold cross-validation to detect
    residual species signal in z.  More sensitive than
    ``adversarial_accuracy_check``.

    Parameters
    ----------
    z
        Latent representations, shape ``(N, d)``.
    species_codes
        Integer species label per cell, shape ``(N,)``.
    n_species
        Number of species.

    Returns
    -------
    dict
        Keys: ``mean_accuracy``, ``fold_accuracies``, ``chance``,
        ``pass``, ``details``.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score

    z = np.asarray(z, dtype=np.float64)
    species_codes = np.asarray(species_codes, dtype=int).ravel()

    chance = 1.0 / n_species

    # Subsample if too many cells (cap at 5000 for speed)
    if len(z) > 5000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(z), 5000, replace=False)
        z = z[idx]
        species_codes = species_codes[idx]

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )

    scores = cross_val_score(clf, z, species_codes, cv=5, scoring="accuracy")
    mean_accuracy = float(scores.mean())
    fold_accuracies = [float(s) for s in scores]

    passed = mean_accuracy < chance + 0.05

    if passed:
        details = (
            f"Nonlinear z-leakage OK: mean_accuracy={mean_accuracy:.3f} < "
            f"chance+0.05={chance + 0.05:.3f} "
            f"(folds: {[f'{a:.3f}' for a in fold_accuracies]})."
        )
    else:
        details = (
            f"Nonlinear z-leakage WARNING: mean_accuracy={mean_accuracy:.3f} >= "
            f"chance+0.05={chance + 0.05:.3f}. Deeper MLP detects species "
            f"signal in z (folds: {[f'{a:.3f}' for a in fold_accuracies]})."
        )

    logger.info("Nonlinear z-leakage check: %s", details)

    return {
        "mean_accuracy": mean_accuracy,
        "fold_accuracies": fold_accuracies,
        "chance": chance,
        "pass": passed,
        "details": details,
    }


def posterior_collapse_check(z: np.ndarray) -> Dict:
    """Detect dimensional collapse in latent space.

    Per-dimension variance below 0.1 indicates a collapsed dimension.
    If more than half the dimensions are collapsed the latent space is
    degenerate.

    Parameters
    ----------
    z
        Latent representations, shape ``(N, d)``.

    Returns
    -------
    dict
        Keys: ``per_dim_var``, ``collapsed_dims``, ``frac_collapsed``,
        ``pass``, ``details``.
    """
    z = np.asarray(z, dtype=np.float64)

    per_dim_var = z.var(axis=0)  # (d,)
    collapsed_dims = [int(i) for i in np.where(per_dim_var < 0.1)[0]]
    frac_collapsed = len(collapsed_dims) / z.shape[1] if z.shape[1] > 0 else 0.0

    passed = frac_collapsed < 0.5

    if passed:
        details = (
            f"Posterior collapse OK: {len(collapsed_dims)}/{z.shape[1]} dims "
            f"collapsed (frac={frac_collapsed:.3f} < 0.5). "
            f"Var range [{per_dim_var.min():.4f}, {per_dim_var.max():.4f}]."
        )
    else:
        details = (
            f"Posterior collapse WARNING: {len(collapsed_dims)}/{z.shape[1]} dims "
            f"collapsed (frac={frac_collapsed:.3f} >= 0.5). "
            f"Dims: {collapsed_dims[:10]}{'...' if len(collapsed_dims) > 10 else ''}."
        )

    logger.info("Posterior collapse check: %s", details)

    return {
        "per_dim_var": per_dim_var,
        "collapsed_dims": collapsed_dims,
        "frac_collapsed": frac_collapsed,
        "pass": passed,
        "details": details,
    }


def shared_decoder_constant_prediction_check(module: nn.Module) -> Dict:
    """Check if shared decoder collapsed to predicting gene means.

    If output_shared weight is near-zero but bias is meaningful, the
    decoder has collapsed to a constant prediction independent of z.

    Parameters
    ----------
    module
        A ``ChimeraModule`` or ``ChimeraMinimalModule`` instance.

    Returns
    -------
    dict
        Keys: ``frac_constant``, ``weight_norms``, ``bias_abs``,
        ``pass``, ``details``.
    """
    W = module.output_shared.weight.data  # (n_genes, n_hidden)
    b = module.output_shared.bias.data  # (n_genes,)

    # Per-gene weight norm (L2 across hidden dimension)
    weight_norms = W.norm(dim=1).cpu().numpy()  # (n_genes,)
    bias_abs = b.abs().cpu().numpy()  # (n_genes,)

    # A gene is "constant" if its weight norm is near-zero relative to its bias
    # Threshold: weight norm < 1e-3 AND bias > 0.1
    constant_mask = (weight_norms < 1e-3) & (bias_abs > 0.1)
    frac_constant = float(constant_mask.mean())

    passed = frac_constant < 0.1

    if passed:
        details = (
            f"Shared decoder OK: frac_constant={frac_constant:.4f} < 0.1. "
            f"Weight norms: [{weight_norms.min():.4f}, {weight_norms.max():.4f}]."
        )
    else:
        n_constant = int(constant_mask.sum())
        details = (
            f"Shared decoder WARNING: frac_constant={frac_constant:.3f} >= 0.1 "
            f"({n_constant} genes). Decoder may have collapsed to gene means."
        )

    logger.info("Shared decoder constant prediction check: %s", details)

    return {
        "frac_constant": frac_constant,
        "weight_norms": weight_norms,
        "bias_abs": bias_abs,
        "pass": passed,
        "details": details,
    }


def encoder_gradient_ratio_check(
    module: nn.Module,
    batch_dict: Dict[str, torch.Tensor],
) -> Dict:
    """FM-5 check: compare encoder gradient norms from shared vs residual.

    Runs two separate forward passes -- one computing loss from the shared
    pathway only and one from the residual pathway only -- and compares
    the encoder gradient norms.  A large ratio suggests one pathway
    dominates encoder training.

    Parameters
    ----------
    module
        A ``ChimeraModule`` instance (full v2.4).
    batch_dict
        Dictionary with keys ``"X"``, ``"species"``, ``"batch"``,
        ``"library_size"``.

    Returns
    -------
    dict
        Keys: ``shared_grad_norm``, ``residual_grad_norm``, ``ratio``,
        ``pass``, ``details``.
    """
    from scvi.distributions import NegativeBinomial

    was_training = module.training
    module.train()

    def _encoder_grad_norm() -> float:
        total = 0.0
        for name, param in module.named_parameters():
            if any(
                prefix in name
                for prefix in ("encoder.", "z_mean_linear.", "z_logvar_linear.")
            ):
                if param.grad is not None:
                    total += float(param.grad.data.norm(2).item() ** 2)
        return float(np.sqrt(total))

    # --- Shared-pathway loss ---
    module.zero_grad()
    try:
        _, outputs = module(batch_dict)
    except Exception as e:
        if not was_training:
            module.eval()
        return {
            "shared_grad_norm": 0.0,
            "residual_grad_norm": 0.0,
            "ratio": float("inf"),
            "pass": False,
            "details": f"Forward pass failed: {e}",
        }

    # Shared reconstruction loss only
    x = batch_dict["X"]
    theta = outputs["theta"]
    mu_shared = outputs["mu_shared_only"]
    L_shared = (
        -NegativeBinomial(mu=mu_shared, theta=theta.detach())
        .log_prob(x)
        .sum(-1)
        .mean()
    )
    L_shared.backward(retain_graph=True)
    shared_grad_norm = _encoder_grad_norm()

    # --- Residual-pathway loss ---
    module.zero_grad()
    mu_full = outputs["mu_full"]
    L_full = (
        -NegativeBinomial(mu=mu_full, theta=theta)
        .log_prob(x)
        .sum(-1)
        .mean()
    )
    L_full.backward()
    residual_grad_norm = _encoder_grad_norm()

    module.zero_grad()
    if not was_training:
        module.eval()

    if shared_grad_norm < 1e-12:
        ratio = float("inf") if residual_grad_norm > 0 else 1.0
    else:
        ratio = residual_grad_norm / shared_grad_norm

    passed = ratio < 2.0

    if passed:
        details = (
            f"Encoder gradient ratio OK (FM-5): ratio={ratio:.3f} < 2.0 "
            f"(shared={shared_grad_norm:.6f}, residual={residual_grad_norm:.6f})."
        )
    else:
        details = (
            f"Encoder gradient ratio WARNING (FM-5): ratio={ratio:.3f} >= 2.0. "
            f"Residual pathway may dominate encoder training "
            f"(shared={shared_grad_norm:.6f}, residual={residual_grad_norm:.6f})."
        )

    logger.info("Encoder gradient ratio check: %s", details)

    return {
        "shared_grad_norm": shared_grad_norm,
        "residual_grad_norm": residual_grad_norm,
        "ratio": ratio,
        "pass": passed,
        "details": details,
    }


def batch_offset_celltype_leak_check(
    module: nn.Module,
    marker_gene_indices: np.ndarray,
    housekeeping_gene_indices: np.ndarray,
) -> Dict:
    """FM-H check: batch offsets should not carry cell-type signal.

    Compares batch offset variance for marker genes vs housekeeping genes.
    If marker genes have systematically higher offset variance, batch
    offsets may be absorbing biological signal.

    Parameters
    ----------
    module
        A ``ChimeraModule`` instance with ``batch_offset``.
    marker_gene_indices
        Integer indices for known marker genes, shape ``(M,)``.
    housekeeping_gene_indices
        Integer indices for housekeeping genes, shape ``(H,)``.

    Returns
    -------
    dict
        Keys: ``marker_var``, ``housekeeping_var``, ``ratio``,
        ``pass``, ``details``.
    """
    marker_gene_indices = np.asarray(marker_gene_indices, dtype=int).ravel()
    housekeeping_gene_indices = np.asarray(housekeeping_gene_indices, dtype=int).ravel()

    with torch.no_grad():
        constrained_offset = module.batch_offset()  # (n_batch, G)

    offset_np = constrained_offset.cpu().numpy()  # (n_batch, G)

    # Per-gene variance across batches
    gene_var = offset_np.var(axis=0)  # (G,)

    marker_var = float(gene_var[marker_gene_indices].mean()) if len(marker_gene_indices) > 0 else 0.0
    housekeeping_var = float(gene_var[housekeeping_gene_indices].mean()) if len(housekeeping_gene_indices) > 0 else 0.0

    if housekeeping_var < 1e-12:
        ratio = float("inf") if marker_var > 0 else 1.0
    else:
        ratio = marker_var / housekeeping_var

    passed = ratio < 3.0

    if passed:
        details = (
            f"Batch offset cell-type leak OK (FM-H): ratio={ratio:.3f} < 3.0 "
            f"(marker_var={marker_var:.6f}, housekeeping_var={housekeeping_var:.6f})."
        )
    else:
        details = (
            f"Batch offset cell-type leak WARNING (FM-H): ratio={ratio:.3f} >= 3.0. "
            f"Batch offsets may absorb cell-type signal "
            f"(marker_var={marker_var:.6f}, housekeeping_var={housekeeping_var:.6f})."
        )

    logger.info("Batch offset cell-type leak check: %s", details)

    return {
        "marker_var": marker_var,
        "housekeeping_var": housekeeping_var,
        "ratio": ratio,
        "pass": passed,
        "details": details,
    }


def batch_offset_biology_absorption_check(module: nn.Module) -> Dict:
    """FM-B check: batch offset magnitude vs shared decoder scale.

    If batch offsets dominate the shared decoder signal for too many
    genes, the model may be using batch offsets to absorb biology.

    Parameters
    ----------
    module
        A ``ChimeraModule`` instance with ``batch_offset`` and
        ``output_shared``.

    Returns
    -------
    dict
        Keys: ``frac_offset_dominant``, ``offset_scale``,
        ``shared_scale``, ``pass``, ``details``.
    """
    with torch.no_grad():
        constrained_offset = module.batch_offset()  # (n_batch, G)

    # Scale: max absolute offset across batches per gene
    offset_scale = constrained_offset.abs().max(dim=0).values.cpu().numpy()  # (G,)

    # Shared decoder scale: per-gene weight norm (L2 across hidden)
    W_shared = module.output_shared.weight.data  # (G, n_hidden)
    shared_scale = W_shared.norm(dim=1).cpu().numpy()  # (G,)

    # A gene is offset-dominant if offset_scale > shared_scale
    dominant_mask = offset_scale > shared_scale
    frac_offset_dominant = float(dominant_mask.mean())

    passed = frac_offset_dominant < 0.1

    if passed:
        details = (
            f"Batch offset biology absorption OK (FM-B): "
            f"frac_offset_dominant={frac_offset_dominant:.4f} < 0.1."
        )
    else:
        n_dominant = int(dominant_mask.sum())
        details = (
            f"Batch offset biology absorption WARNING (FM-B): "
            f"frac_offset_dominant={frac_offset_dominant:.3f} >= 0.1 "
            f"({n_dominant} genes). Batch offsets may absorb biology."
        )

    logger.info("Batch offset biology absorption check: %s", details)

    return {
        "frac_offset_dominant": frac_offset_dominant,
        "offset_scale": offset_scale,
        "shared_scale": shared_scale,
        "pass": passed,
        "details": details,
    }


def dispersion_stratification_check(
    module: nn.Module,
    alpha: np.ndarray,
) -> Dict:
    """Check if divergent genes have systematically smaller theta.

    Genes with low alpha (divergent, species-specific) should not have
    systematically smaller dispersion (larger variance), which would
    indicate the model is conflating technical noise with biological
    divergence.

    Parameters
    ----------
    module
        A ``ChimeraModule`` instance with ``dispersion`` parameter.
    alpha
        Per-gene alpha values, shape ``(G,)``.

    Returns
    -------
    dict
        Keys: ``theta_low_alpha``, ``theta_high_alpha``, ``ratio``,
        ``pass``, ``details``.
    """
    alpha = np.asarray(alpha, dtype=np.float64).ravel()

    # Get mean theta across batches (softplus of dispersion parameter)
    with torch.no_grad():
        disp = module.dispersion.data  # (n_batch, G) or (G,)
        if disp.dim() == 2:
            theta = torch.nn.functional.softplus(disp).mean(dim=0).cpu().numpy()
        else:
            theta = torch.nn.functional.softplus(disp).cpu().numpy()

    # Split genes by alpha median into low-alpha (divergent) and high-alpha (conserved)
    alpha_median = np.median(alpha)
    low_mask = alpha < alpha_median
    high_mask = alpha >= alpha_median

    theta_low = float(theta[low_mask].mean()) if low_mask.any() else 0.0
    theta_high = float(theta[high_mask].mean()) if high_mask.any() else 0.0

    if theta_high < 1e-12:
        ratio = float("inf") if theta_low > 0 else 1.0
    else:
        ratio = theta_low / theta_high

    passed = ratio > 0.5

    if passed:
        details = (
            f"Dispersion stratification OK: ratio={ratio:.3f} > 0.5 "
            f"(theta_low_alpha={theta_low:.4f}, theta_high_alpha={theta_high:.4f})."
        )
    else:
        details = (
            f"Dispersion stratification WARNING: ratio={ratio:.3f} <= 0.5. "
            f"Divergent genes have systematically smaller theta "
            f"(theta_low={theta_low:.4f}, theta_high={theta_high:.4f})."
        )

    logger.info("Dispersion stratification check: %s", details)

    return {
        "theta_low_alpha": theta_low,
        "theta_high_alpha": theta_high,
        "ratio": ratio,
        "pass": passed,
        "details": details,
    }


def loss_convexity_check(
    module: nn.Module,
    batch_dict: Dict[str, torch.Tensor],
    n_genes_to_test: int = 10,
) -> Dict:
    """Check alpha landscape convexity for random genes.

    For each tested gene, evaluates the reconstruction loss at
    alpha - epsilon, alpha, and alpha + epsilon.  The landscape is convex
    if the middle point has the lowest loss.

    Parameters
    ----------
    module
        A ``ChimeraModule`` instance (full v2.4).
    batch_dict
        Dictionary with keys ``"X"``, ``"species"``, ``"batch"``,
        ``"library_size"``.
    n_genes_to_test
        Number of random genes to probe (default 10).

    Returns
    -------
    dict
        Keys: ``n_convex``, ``n_tested``, ``frac_convex``,
        ``pass``, ``details``.
    """
    from scvi.distributions import NegativeBinomial

    was_training = module.training
    module.eval()

    try:
        with torch.no_grad():
            _, outputs = module(batch_dict)
    except Exception as e:
        if was_training:
            module.train()
        return {
            "n_convex": 0,
            "n_tested": 0,
            "frac_convex": 0.0,
            "pass": False,
            "details": f"Forward pass failed: {e}",
        }

    x = batch_dict["X"]
    alpha = outputs["alpha"].detach()  # (G,)
    log_rate_shared = outputs["log_rate_shared_full"].detach()  # (B, G)
    log_rate_residual = outputs["log_rate_residual"].detach()  # (B, G)
    log_rate_batch = outputs["log_rate_batch"].detach()  # (B, G)
    theta = outputs["theta"].detach()  # (B, G)
    if "library_size" in batch_dict:
        library = batch_dict["library_size"].squeeze(-1).detach()  # (B,)
    else:
        library = batch_dict["X"].sum(dim=-1).detach()  # (B,)

    n_genes = alpha.shape[0]
    n_test = min(n_genes_to_test, n_genes)
    rng = np.random.RandomState(42)
    gene_indices = rng.choice(n_genes, n_test, replace=False)

    epsilon = 0.05
    n_convex = 0

    for g_idx in gene_indices:
        g = int(g_idx)
        losses_at_eps = []

        for delta in [-epsilon, 0.0, epsilon]:
            alpha_g_perturbed = torch.clamp(alpha[g] + delta, 0.0, 1.0)
            log_mu_g = (
                log_rate_shared[:, g]
                + log_rate_batch[:, g]
                + (1.0 - alpha_g_perturbed) * log_rate_residual[:, g]
            )
            mu_g = torch.exp(log_mu_g) * library  # (B,)
            theta_g = theta[:, g]  # (B,)
            nll = -NegativeBinomial(mu=mu_g, theta=theta_g).log_prob(x[:, g]).mean()
            losses_at_eps.append(nll.item())

        # Convex if middle point is NOT a local maximum
        # i.e., L(alpha) <= 0.5 * (L(alpha-eps) + L(alpha+eps))
        if losses_at_eps[1] <= 0.5 * (losses_at_eps[0] + losses_at_eps[2]) + 1e-8:
            n_convex += 1

    if was_training:
        module.train()

    frac_convex = n_convex / n_test if n_test > 0 else 1.0
    passed = n_convex == n_test

    if passed:
        details = (
            f"Loss convexity OK: {n_convex}/{n_test} tested genes are convex."
        )
    else:
        details = (
            f"Loss convexity WARNING: {n_convex}/{n_test} tested genes are "
            f"convex (frac={frac_convex:.3f}). Non-convex landscape may "
            f"indicate alpha optimization issues."
        )

    logger.info("Loss convexity check: %s", details)

    return {
        "n_convex": n_convex,
        "n_tested": n_test,
        "frac_convex": frac_convex,
        "pass": passed,
        "details": details,
    }


def gin_gradient_health_check(
    module: nn.Module,
    batch_dict: Dict[str, torch.Tensor],
) -> Dict:
    """Per-gene gradient health for the GIN.

    Verifies that the GIN receives non-vanishing gradients by running a
    forward+backward pass and inspecting the maximum gradient norm of
    GIN parameters.

    Parameters
    ----------
    module
        A ``ChimeraModule`` instance with ``gin`` attribute.
    batch_dict
        Dictionary with keys ``"X"``, ``"species"``, ``"batch"``,
        ``"library_size"``.

    Returns
    -------
    dict
        Keys: ``max_grad``, ``gin_param_grad_norms``, ``pass``, ``details``.
    """
    was_training = module.training
    module.train()
    module.zero_grad()

    try:
        loss, _ = module(batch_dict)
        loss.backward()
    except Exception as e:
        if not was_training:
            module.eval()
        return {
            "max_grad": 0.0,
            "gin_param_grad_norms": {},
            "pass": False,
            "details": f"Gradient computation failed: {e}",
        }

    gin_param_grad_norms = {}
    for name, param in module.gin.named_parameters():
        if param.grad is not None:
            gin_param_grad_norms[name] = float(param.grad.data.norm(2).item())
        else:
            gin_param_grad_norms[name] = 0.0

    module.zero_grad()
    if not was_training:
        module.eval()

    all_norms = list(gin_param_grad_norms.values())
    max_grad = max(all_norms) if all_norms else 0.0

    passed = max_grad > 1e-6

    if passed:
        details = (
            f"GIN gradient health OK: max_grad={max_grad:.2e} > 1e-6 "
            f"({len(all_norms)} params)."
        )
    else:
        zero_params = [k for k, v in gin_param_grad_norms.items() if v < 1e-6]
        details = (
            f"GIN gradient health WARNING: max_grad={max_grad:.2e} <= 1e-6. "
            f"GIN may have vanishing gradients. Zero-grad params: {zero_params}."
        )

    logger.info("GIN gradient health check: %s", details)

    return {
        "max_grad": max_grad,
        "gin_param_grad_norms": gin_param_grad_norms,
        "pass": passed,
        "details": details,
    }


def batch_offset_zero_sum_check(module: nn.Module) -> Dict:
    """Verify constrained batch offset sums to zero within each species.

    The ``ConstrainedBatchOffset`` should enforce a per-species zero-sum
    constraint.  This diagnostic numerically verifies that constraint.

    Parameters
    ----------
    module
        A ``ChimeraModule`` instance with ``batch_offset`` attribute.

    Returns
    -------
    dict
        Keys: ``max_violation``, ``per_species_max_abs_sum``,
        ``pass``, ``details``.
    """
    with torch.no_grad():
        constrained_offset = module.batch_offset()  # (n_batch, G)

    batch_to_species = module.batch_offset.batch_to_species_idx  # (n_batch,)
    n_species = int(module.batch_offset.n_species.item())

    offset_np = constrained_offset.cpu().numpy()
    species_np = batch_to_species.cpu().numpy()

    per_species_max_abs_sum = {}
    max_violation = 0.0

    for s in range(n_species):
        mask = species_np == s
        if mask.sum() < 2:
            # Single-batch species: offset should be zero
            species_sum = np.abs(offset_np[mask]).max() if mask.any() else 0.0
        else:
            species_sum = float(np.abs(offset_np[mask].sum(axis=0)).max())
        per_species_max_abs_sum[s] = species_sum
        max_violation = max(max_violation, species_sum)

    passed = max_violation < 1e-5

    if passed:
        details = (
            f"Batch offset zero-sum OK: max_violation={max_violation:.2e} < 1e-5 "
            f"across {n_species} species."
        )
    else:
        details = (
            f"Batch offset zero-sum WARNING: max_violation={max_violation:.2e} >= 1e-5. "
            f"Per-species: {per_species_max_abs_sum}."
        )

    logger.info("Batch offset zero-sum check: %s", details)

    return {
        "max_violation": max_violation,
        "per_species_max_abs_sum": per_species_max_abs_sum,
        "pass": passed,
        "details": details,
    }


def run_all_diagnostics(
    module: nn.Module,
    batch_dict: Dict[str, torch.Tensor],
    alpha: Optional[np.ndarray] = None,
    conservation_prior: Optional[np.ndarray] = None,
    z_mean: Optional[np.ndarray] = None,
    z_logvar: Optional[np.ndarray] = None,
    correction: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    species_codes: Optional[np.ndarray] = None,
    n_species: Optional[int] = None,
    mu_shared_only: Optional[np.ndarray] = None,
    cell_type_labels: Optional[np.ndarray] = None,
    marker_gene_indices: Optional[np.ndarray] = None,
    housekeeping_gene_indices: Optional[np.ndarray] = None,
) -> Dict:
    """Run all 20 applicable diagnostics and return a combined report.

    Runs every diagnostic for which the required inputs are provided.
    Module-based diagnostics always run; array-based diagnostics run only
    when their respective arrays are supplied.

    Results are grouped into MUST-RUN (7) and SHOULD-RUN (13) categories.

    MUST-RUN (7):
      loss_scale_sanity_check, alpha_histogram_health_check,
      delta_alpha_two_sidedness_check, w_shared_column_norms,
      residual_magnitude_vs_alpha, sigmoid_saturation_check,
      nonlinear_z_leakage_check

    SHOULD-RUN (13):
      per_dimension_kl_check, gradient_norm_ratio, z_leakage_diagnostic,
      adversarial_accuracy_check, posterior_collapse_check,
      gin_correction_magnitude_check,
      shared_decoder_constant_prediction_check,
      encoder_gradient_ratio_check, batch_offset_celltype_leak_check,
      batch_offset_biology_absorption_check,
      dispersion_stratification_check, loss_convexity_check,
      gin_gradient_health_check, batch_offset_zero_sum_check

    Parameters
    ----------
    module
        A ``ChimeraModule`` or ``ChimeraMinimalModule`` instance.
    batch_dict
        Dictionary with keys ``"X"``, ``"species"``, ``"batch"``,
        ``"library_size"``.
    alpha
        Per-gene alpha as numpy array, shape ``(G,)``. Optional.
    conservation_prior
        Per-gene conservation prior as numpy array, shape ``(G,)``. Optional.
    z_mean
        Posterior mean as numpy array, shape ``(N, d)``. Optional.
    z_logvar
        Posterior log-variance as numpy array, shape ``(N, d)``. Optional.
    correction
        Per-gene GIN correction as numpy array, shape ``(G,)``. Optional.
    z
        Latent representations as numpy array, shape ``(N, d)``. Optional.
    species_codes
        Integer species label per cell as numpy array, shape ``(N,)``. Optional.
    n_species
        Number of species. Optional (inferred from species_codes if not given).
    mu_shared_only
        Per-cell shared-pathway reconstruction, shape ``(N, G)``. Optional.
    cell_type_labels
        Cell-type label per cell, shape ``(N,)``. Optional.
    marker_gene_indices
        Integer indices of known marker genes, shape ``(M,)``. Optional.
    housekeeping_gene_indices
        Integer indices of housekeeping genes, shape ``(H,)``. Optional.

    Returns
    -------
    dict
        Keys correspond to diagnostic names, values are the result dicts.
        Also includes ``"must_run_pass"`` (bool), ``"should_run_pass"`` (bool),
        ``"all_pass"`` (bool), and ``"summary"`` (str).
    """
    must_run_results = {}
    should_run_results = {}

    # Infer n_species from species_codes if not provided
    if n_species is None and species_codes is not None:
        n_species = int(np.unique(species_codes).shape[0])

    # =====================================================================
    # MUST-RUN diagnostics (7)
    # =====================================================================

    # M1. Loss scale sanity check (needs full module with lambda_anchor)
    if hasattr(module, "lambda_anchor"):
        must_run_results["loss_scale_sanity_check"] = loss_scale_sanity_check(
            module, batch_dict
        )

    # M2. Alpha histogram
    if alpha is not None:
        must_run_results["alpha_histogram_health_check"] = (
            alpha_histogram_health_check(alpha)
        )

    # M3. Delta-alpha two-sidedness
    if alpha is not None and conservation_prior is not None:
        must_run_results["delta_alpha_two_sidedness_check"] = (
            delta_alpha_two_sidedness_check(alpha, conservation_prior)
        )

    # M4. W_shared column norms (always available)
    must_run_results["w_shared_column_norms"] = w_shared_column_norms(module)

    # M5. Residual magnitude vs alpha (needs full module with GIN)
    if hasattr(module, "gin"):
        must_run_results["residual_magnitude_vs_alpha"] = (
            residual_magnitude_vs_alpha(module, batch_dict)
        )

    # M6. Sigmoid saturation check
    if alpha is not None:
        must_run_results["sigmoid_saturation_check"] = (
            sigmoid_saturation_check(alpha)
        )

    # M7. Nonlinear z-leakage check
    if z is not None and species_codes is not None and n_species is not None:
        must_run_results["nonlinear_z_leakage_check"] = (
            nonlinear_z_leakage_check(z, species_codes, n_species)
        )

    # =====================================================================
    # SHOULD-RUN diagnostics (13)
    # =====================================================================

    # S1. Per-dimension KL
    if z_mean is not None and z_logvar is not None:
        should_run_results["per_dimension_kl_check"] = per_dimension_kl_check(
            z_mean, z_logvar
        )

    # S2. Gradient norm ratio (always available)
    should_run_results["gradient_norm_ratio"] = gradient_norm_ratio(
        module, batch_dict
    )

    # S3. z-leakage diagnostic
    if (
        mu_shared_only is not None
        and species_codes is not None
        and cell_type_labels is not None
    ):
        should_run_results["z_leakage_diagnostic"] = z_leakage_diagnostic(
            mu_shared_only, species_codes, cell_type_labels
        )

    # S4. Adversarial accuracy check
    if z is not None and species_codes is not None and n_species is not None:
        should_run_results["adversarial_accuracy_check"] = (
            adversarial_accuracy_check(z, species_codes, n_species)
        )

    # S5. Posterior collapse check
    if z is not None:
        should_run_results["posterior_collapse_check"] = (
            posterior_collapse_check(z)
        )

    # S6. GIN correction magnitude check
    if correction is not None:
        should_run_results["gin_correction_magnitude_check"] = (
            gin_correction_magnitude_check(correction)
        )

    # S7. Shared decoder constant prediction check (always available)
    should_run_results["shared_decoder_constant_prediction_check"] = (
        shared_decoder_constant_prediction_check(module)
    )

    # S8. Encoder gradient ratio check (needs full module with GIN)
    if hasattr(module, "gin"):
        should_run_results["encoder_gradient_ratio_check"] = (
            encoder_gradient_ratio_check(module, batch_dict)
        )

    # S9. Batch offset cell-type leak check
    if (
        hasattr(module, "batch_offset")
        and marker_gene_indices is not None
        and housekeeping_gene_indices is not None
    ):
        should_run_results["batch_offset_celltype_leak_check"] = (
            batch_offset_celltype_leak_check(
                module, marker_gene_indices, housekeeping_gene_indices
            )
        )

    # S10. Batch offset biology absorption check
    if hasattr(module, "batch_offset"):
        should_run_results["batch_offset_biology_absorption_check"] = (
            batch_offset_biology_absorption_check(module)
        )

    # S11. Dispersion stratification check
    if alpha is not None:
        should_run_results["dispersion_stratification_check"] = (
            dispersion_stratification_check(module, alpha)
        )

    # S12. Loss convexity check (needs full module with GIN)
    if hasattr(module, "gin"):
        should_run_results["loss_convexity_check"] = loss_convexity_check(
            module, batch_dict
        )

    # S13. GIN gradient health check (needs full module with GIN)
    if hasattr(module, "gin"):
        should_run_results["gin_gradient_health_check"] = (
            gin_gradient_health_check(module, batch_dict)
        )

    # S14. Batch offset zero-sum check
    if hasattr(module, "batch_offset"):
        should_run_results["batch_offset_zero_sum_check"] = (
            batch_offset_zero_sum_check(module)
        )

    # =====================================================================
    # Aggregate results
    # =====================================================================
    results = {}

    must_run_pass = all(
        r.get("pass", False) for r in must_run_results.values()
    ) if must_run_results else True
    should_run_pass = all(
        r.get("pass", False) for r in should_run_results.values()
    ) if should_run_results else True
    all_pass = must_run_pass and should_run_pass

    results.update(must_run_results)
    results.update(should_run_results)

    # Summary
    n_must_total = len(must_run_results)
    n_must_passed = sum(
        1 for r in must_run_results.values() if r.get("pass", False)
    )
    n_should_total = len(should_run_results)
    n_should_passed = sum(
        1 for r in should_run_results.values() if r.get("pass", False)
    )
    n_total = n_must_total + n_should_total
    n_passed = n_must_passed + n_should_passed

    failed_must = [
        name for name, r in must_run_results.items()
        if not r.get("pass", False)
    ]
    failed_should = [
        name for name, r in should_run_results.items()
        if not r.get("pass", False)
    ]

    summary_lines = [
        f"Diagnostic suite: {n_passed}/{n_total} passed.",
        f"  MUST-RUN:   {n_must_passed}/{n_must_total} passed"
        + (f" -- FAILED: {', '.join(failed_must)}" if failed_must else ""),
        f"  SHOULD-RUN: {n_should_passed}/{n_should_total} passed"
        + (f" -- FAILED: {', '.join(failed_should)}" if failed_should else ""),
    ]
    summary = "\n".join(summary_lines)

    results["must_run_pass"] = must_run_pass
    results["should_run_pass"] = should_run_pass
    results["all_pass"] = all_pass
    results["summary"] = summary

    logger.info("Diagnostic suite:\n%s", summary)

    return results
