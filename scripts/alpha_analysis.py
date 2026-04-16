"""Alpha analysis experiments for Chimera v2.4 (Task 9.4).

The signature experiments unique to Chimera -- produces the core scientific
figures and tables that differentiate Chimera from every other integration tool.

Experiments implemented:
  A. Alpha vs conservation prior correlation
  B. Conservation-stratified integration quality (quartile cosine similarity)
  C. Residual decomposition -- the L-shape (alpha vs residual magnitude)
  D. Residualized delta_alpha with confound control (OLS residualization)
  E. Per-gene diagnostic table (top-20 positive / negative delta_alpha)
  F. Two-sidedness statistic (v2.4 signature)

Usage:
  python -m scripts.alpha_analysis \\
      --model-dir results/camex_pancreas/chimera_v24 \\
      --adata-path data/camex_pancreas.h5ad \\
      --dataset-name camex_pancreas \\
      --output-dir results/camex_pancreas/alpha_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------------


def run_alpha_analysis(model, adata, dataset_name="camex_pancreas"):
    """The signature experiment: alpha analysis. Produces all alpha-related
    scientific outputs with confound control."""
    results = {}

    alpha = model.get_alpha()                          # (G,)
    prior = model.module.conservation_prior.cpu().numpy()  # (G,)
    delta = model.get_delta_alpha()                    # (G,)
    correction = model.get_correction()                # (G,)
    S_g = model.get_S_g()                              # (G,)
    gene_names = adata.var_names.values                # (G,)

    # === Experiment A: Alpha vs conservation correlation ===
    r_spearman, p_spearman = spearmanr(alpha, prior)
    r_pearson, p_pearson = pearsonr(alpha, prior)
    results["alpha_vs_prior"] = {
        "spearman_r": float(r_spearman), "spearman_p": float(p_spearman),
        "pearson_r": float(r_pearson), "pearson_p": float(p_pearson),
        "expected_range": "Spearman r ~ 0.3-0.6",
    }

    # === Experiment B: Conservation-stratified integration quality ===
    # Split genes into alpha quartiles, compute per-quartile cross-species similarity
    quartile_edges = np.percentile(alpha, [0, 25, 50, 75, 100])
    # Run 27 fix: pass adata explicitly to ensure cell ordering matches species labels.
    # Without passing adata, get_shared_only_expression uses the model's internal
    # registered AnnData -- if adata was re-shuffled, species masking would be wrong.
    mu_shared = model.get_shared_only_expression(adata, library_size=1.0)  # (N, G) rates
    species = adata.obs["species"].values
    species_cats = np.unique(species)

    stratified_results = []
    for q_idx in range(4):
        lo, hi = quartile_edges[q_idx], quartile_edges[q_idx + 1]
        gene_mask = (alpha >= lo) & (alpha <= hi) if q_idx == 3 else (alpha >= lo) & (alpha < hi)
        gene_indices = np.where(gene_mask)[0]
        if len(gene_indices) == 0:
            continue

        # Per-species mean expression for these genes
        sp_means = {}
        for sp in species_cats:
            sp_mask = species == sp
            sp_means[sp] = mu_shared[sp_mask][:, gene_indices].mean(axis=0)

        # Mean pairwise cosine similarity between species
        vecs = np.stack(list(sp_means.values()))
        cos_mat = cosine_similarity(vecs)
        n = len(vecs)
        pairwise_cos = [cos_mat[i, j] for i in range(n) for j in range(i + 1, n)]
        stratified_results.append({
            "quartile": q_idx + 1,
            "alpha_range": f"[{lo:.3f}, {hi:.3f})",
            "n_genes": int(gene_mask.sum()),
            "mean_cross_species_cosine": float(np.mean(pairwise_cos)),
        })
    results["conservation_stratified"] = stratified_results
    # Expected: higher alpha quartiles -> higher cross-species cosine similarity

    # === Experiment C: Residual decomposition -- the L-shape ===
    # Per-gene residual magnitude (requires a forward pass)
    scdl = model._make_data_loader(adata=model.adata, batch_size=256)
    residual_accum = np.zeros(len(gene_names))
    n_batches = 0
    model.module.eval()
    with torch.no_grad():
        for tensors in scdl:
            _, out = model.module(tensors)  # forward returns (loss, outputs_dict)
            residual_accum += out["log_rate_residual"].abs().mean(dim=0).cpu().numpy()
            n_batches += 1
    residual_mag = residual_accum / max(n_batches, 1)  # (G,) mean |residual| per gene, averaged over batches

    r_lshape, _ = spearmanr(alpha, residual_mag)
    results["residual_decomposition"] = {
        "alpha_vs_residual_spearman_r": float(r_lshape),
        "expected": "r < -0.3 (L-shaped: high alpha -> low residual)",
        "residual_mag_mean": float(residual_mag.mean()),
        "residual_mag_max": float(residual_mag.max()),
    }

    # === Experiment D: Residualized delta_alpha (confound control) ===
    # Residualize against 5 confounds (ortholog_aware_integration_guide_initial.md "Confound control")
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    mean_expr = X.mean(axis=0)                          # (G,)
    var_expr = X.var(axis=0)                             # (G,)
    frac_nonzero = (X > 0).mean(axis=0)                  # (G,)

    # Build confound matrix -- start with 3 always-available confounds
    confound_names = ["log_mean_expr", "log_var_expr", "frac_nonzero"]
    confound_cols = [
        np.log1p(mean_expr),  # (a) log mean expression
        np.log1p(var_expr),   # (b) log expression variance
        frac_nonzero,         # (c) detection rate
    ]

    # (d) Gene length ratio across species (optional -- requires gene-level metadata from Task 1.4)
    if "gene_length_ratio" in adata.var.columns:
        confound_cols.append(adata.var["gene_length_ratio"].values.astype(float))
        confound_names.append("gene_length_ratio")

    # (e) Per-gene batch variance magnitude
    batch_labels = adata.obs["batch"].values
    batch_cats = np.unique(batch_labels)
    if len(batch_cats) > 1:
        batch_means = np.stack([X[batch_labels == b].mean(axis=0) for b in batch_cats])  # (Nb, G)
        per_gene_batch_var = batch_means.var(axis=0)  # (G,) variance of batch means
        confound_cols.append(np.log1p(per_gene_batch_var))
        confound_names.append("log_batch_var")

    confounds = np.column_stack(confound_cols)

    # Residualize delta_alpha against confounds via OLS
    reg = LinearRegression().fit(confounds, delta)
    delta_residualized = delta - reg.predict(confounds)

    # Re-check correlation with prior after residualization
    r_resid, p_resid = spearmanr(delta_residualized, prior)
    results["residualized_delta_alpha"] = {
        "confounds_used": confound_names,
        "n_confounds": len(confound_names),
        "r2_confounds": float(reg.score(confounds, delta)),
        "delta_vs_prior_before_residualization_r": float(spearmanr(delta, prior)[0]),
        "delta_vs_prior_after_residualization_r": float(r_resid),
        "note": "If gene_length_ratio unavailable, reports 4-confound residualization"
                if "gene_length_ratio" not in confound_names else None,
    }

    # === Experiment E: Per-gene diagnostic table ===
    gene_table = pd.DataFrame({
        "gene": gene_names,
        "alpha": alpha,
        "conservation_prior": prior,
        "delta_alpha": delta,
        "delta_alpha_residualized": delta_residualized,
        "correction": correction,
        "S_g": S_g,
        "residual_magnitude": residual_mag,
        "mean_expression": mean_expr,
        "frac_nonzero": frac_nonzero,
    }).sort_values("delta_alpha", ascending=False)

    results["top_20_delta_positive"] = gene_table.head(20).to_dict(orient="records")
    results["top_20_delta_negative"] = gene_table.tail(20).to_dict(orient="records")

    # === Experiment F: Two-sidedness statistic (v2.4 signature) ===
    frac_positive = float((delta > 0).mean())
    results["two_sidedness"] = {
        "frac_delta_positive": frac_positive,
        "frac_delta_negative": float((delta < 0).mean()),
        "asymmetry": float(abs(frac_positive - 0.5)),
        "mean_positive_delta": float(delta[delta > 0].mean()) if (delta > 0).any() else 0,
        "mean_negative_delta": float(delta[delta < 0].mean()) if (delta < 0).any() else 0,
    }

    return results, gene_table


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run Chimera v2.4 alpha analysis experiments (Task 9.4).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to saved ChimeraModel directory.",
    )
    parser.add_argument(
        "--adata-path",
        type=str,
        required=True,
        help="Path to AnnData (.h5ad) used for training.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="camex_pancreas",
        help="Dataset label attached to results (default: camex_pancreas).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write results JSON and gene table CSV. "
             "Defaults to <model-dir>/alpha_analysis.",
    )

    args = parser.parse_args()

    # ---- resolve paths ----
    model_dir = Path(args.model_dir)
    adata_path = Path(args.adata_path)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "alpha_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load data and model ----
    import anndata as ad

    logger.info("Loading AnnData from %s", adata_path)
    adata = ad.read_h5ad(adata_path)

    # Lazy import to avoid pulling heavy dependencies at module level
    from chimera.model import ChimeraModel

    logger.info("Loading ChimeraModel from %s", model_dir)
    model = ChimeraModel.load(str(model_dir), adata=adata)

    # ---- run analysis ----
    logger.info("Running alpha analysis for dataset '%s'", args.dataset_name)
    results, gene_table = run_alpha_analysis(model, adata, dataset_name=args.dataset_name)

    # ---- save outputs ----
    results_path = output_dir / f"alpha_results_{args.dataset_name}.json"
    table_path = output_dir / f"gene_table_{args.dataset_name}.csv"

    logger.info("Writing results to %s", results_path)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Writing gene table to %s", table_path)
    gene_table.to_csv(table_path, index=False)

    # ---- summary log ----
    logger.info("=== Alpha Analysis Summary (%s) ===", args.dataset_name)
    logger.info(
        "  A. alpha vs prior Spearman r = %.4f (p = %.2e)",
        results["alpha_vs_prior"]["spearman_r"],
        results["alpha_vs_prior"]["spearman_p"],
    )
    logger.info(
        "  C. alpha vs residual Spearman r = %.4f (L-shape)",
        results["residual_decomposition"]["alpha_vs_residual_spearman_r"],
    )
    logger.info(
        "  D. delta vs prior r: before=%.4f, after residualization=%.4f (R2=%.4f)",
        results["residualized_delta_alpha"]["delta_vs_prior_before_residualization_r"],
        results["residualized_delta_alpha"]["delta_vs_prior_after_residualization_r"],
        results["residualized_delta_alpha"]["r2_confounds"],
    )
    logger.info(
        "  F. two-sidedness: frac_positive=%.4f, frac_negative=%.4f",
        results["two_sidedness"]["frac_delta_positive"],
        results["two_sidedness"]["frac_delta_negative"],
    )
    logger.info("  Gene table: %d genes written to %s", len(gene_table), table_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
