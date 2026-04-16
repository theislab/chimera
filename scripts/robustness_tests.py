"""Robustness stress tests for Chimera v2.4 (Task 9.8).

Tests model behavior under data perturbation to validate that the
architecture's structural guarantees hold beyond ideal conditions:

  1. Ortholog dropout   -- drop random orthologs, retrain, check alpha
                           ranking correlation degrades gracefully
  2. Shuffled conservation -- shuffle conservation priors to confirm
                              the signal is real (not just expression)
  3. False orthologs    -- inject false ortholog assignments with
                           expression column shuffling (derangement
                           enforcement) and unequal cell count handling
  4. Batch structure    -- single-batch-per-species degenerate case
                           where batch offset should be forced to zero
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
from scipy.stats import spearmanr

from chimera._model import ChimeraModel
from scripts.compute_metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Stress tests
# ------------------------------------------------------------------


def stress_test_ortholog_dropout(adata, dropout_fracs=[0.1, 0.3, 0.5],
                                  max_epochs=150, seed=42):
    """Drop random orthologs and retrain. Alpha ranking should degrade gracefully."""
    # Baseline: full gene set
    adata_full = adata.copy()
    ChimeraModel.setup_anndata(adata_full, species_key="species", batch_key="batch")
    model_full = ChimeraModel.from_config(adata_full, "D")
    import scvi
    scvi.settings.seed = seed  # scvi-tools train() does not accept seed kwarg
    model_full.train(max_epochs=max_epochs)
    alpha_full = model_full.get_alpha()

    results = []
    rng = np.random.default_rng(seed)
    G = adata.n_vars

    for frac in dropout_fracs:
        # Randomly select genes to keep
        n_keep = int(G * (1 - frac))
        keep_idx = np.sort(rng.choice(G, size=n_keep, replace=False))
        adata_drop = adata[:, keep_idx].copy()

        # Recompute gene features and conservation prior for remaining genes
        if "gene_features" in adata.varm:
            adata_drop.varm["gene_features"] = adata.varm["gene_features"][keep_idx]
        adata_drop.var["conservation_prior"] = adata.var["conservation_prior"].values[keep_idx]
        if "p_null" in adata.uns:
            adata_drop.uns["p_null"] = adata.uns["p_null"][:, keep_idx]

        ChimeraModel.setup_anndata(adata_drop, species_key="species", batch_key="batch")
        model_drop = ChimeraModel.from_config(adata_drop, "D")
        scvi.settings.seed = seed
        model_drop.train(max_epochs=max_epochs)
        alpha_drop = model_drop.get_alpha()

        # Compare alpha ranking for shared genes
        alpha_full_shared = alpha_full[keep_idx]
        r, p = spearmanr(alpha_full_shared, alpha_drop)

        # Metrics comparison
        adata_drop.obsm["X_chimera"] = model_drop.get_latent_representation()
        metrics = compute_all_metrics(adata_drop, "X_chimera") if "cell_type" in adata_drop.obs else {}

        results.append({
            "dropout_frac": frac,
            "n_genes_remaining": n_keep,
            "alpha_ranking_spearman_r": float(r),
            "overall_score": metrics.get("overall_score", None),
            "label_transfer": metrics.get("mean_label_transfer", None),
        })

    return pd.DataFrame(results)


def stress_test_shuffled_conservation(adata, max_epochs=150, seed=42):
    """Shuffle conservation scores. If alpha-conservation correlation drops to ~0,
    the model is genuinely using conservation features (not just expression)."""
    rng = np.random.default_rng(seed)

    # Baseline
    adata_real = adata.copy()
    ChimeraModel.setup_anndata(adata_real, species_key="species", batch_key="batch")
    model_real = ChimeraModel.from_config(adata_real, "D")
    import scvi
    scvi.settings.seed = seed  # scvi-tools train() does not accept seed kwarg
    model_real.train(max_epochs=max_epochs)
    alpha_real = model_real.get_alpha()
    prior_real = model_real.module.conservation_prior.cpu().numpy()

    # Shuffled
    adata_shuf = adata.copy()
    shuffled_prior = adata.var["conservation_prior"].values.copy()
    rng.shuffle(shuffled_prior)
    adata_shuf.var["conservation_prior"] = shuffled_prior
    # Also shuffle conservation-related columns in gene_features (last 8 dims)
    if "gene_features" in adata.varm:
        gf = adata.varm["gene_features"].copy()
        # Shuffle rows of scalar features (dims 64-71) to break gene-conservation link
        scalar_features = gf[:, 64:].copy()
        rng.shuffle(scalar_features)
        gf[:, 64:] = scalar_features
        adata_shuf.varm["gene_features"] = gf

    ChimeraModel.setup_anndata(adata_shuf, species_key="species", batch_key="batch")
    model_shuf = ChimeraModel.from_config(adata_shuf, "D")
    scvi.settings.seed = seed
    model_shuf.train(max_epochs=max_epochs)
    alpha_shuf = model_shuf.get_alpha()

    r_real, _ = spearmanr(alpha_real, prior_real)
    r_shuf, _ = spearmanr(alpha_shuf, shuffled_prior)
    r_cross, _ = spearmanr(alpha_real, alpha_shuf)

    return {
        "alpha_vs_real_prior_r": float(r_real),
        "alpha_vs_shuffled_prior_r": float(r_shuf),
        "alpha_real_vs_shuffled_r": float(r_cross),
        "conservation_signal_validated": abs(r_real) > abs(r_shuf) + 0.1,
    }


def stress_test_false_orthologs(adata, false_frac=0.2, max_epochs=150, seed=42):
    """Simulate false ortholog assignments. For n_false genes, shuffle expression
    columns across species to break the true ortholog correspondence, then assign
    random conservation priors. Alpha should be robust for true orthologs; false
    orthologs should get alpha near their (random) priors with small |delta_alpha|.

    NOTE: This test has two components:
    (1) Prior perturbation: random priors for false genes (tests whether GIN is
        robust to uninformative priors).
    (2) Expression column shuffling: for each false gene, permute which species'
        expression column maps to it. This breaks the true ortholog expression
        correspondence, simulating a wrong gene pairing in the ortholog table.

    Component (2) is the more biologically realistic simulation. Without it,
    this test only measures prior robustness, not false ortholog robustness."""
    rng = np.random.default_rng(seed)
    G = adata.n_vars
    n_false = int(G * false_frac)

    adata_false = adata.copy()
    false_idx = rng.choice(G, size=n_false, replace=False)

    # (1) Shuffle conservation prior for false genes
    prior = adata_false.var["conservation_prior"].values.copy()
    prior[false_idx] = rng.uniform(0.3, 0.7, size=n_false)
    adata_false.var["conservation_prior"] = prior

    # (2) Shuffle expression columns across species for false genes
    # For each false gene, randomly permute expression values across species
    # (equivalent to swapping which species' gene is aligned to this ortholog slot)
    X = adata_false.X.toarray() if hasattr(adata_false.X, 'toarray') else np.array(adata_false.X)
    species = adata_false.obs["species"].values
    species_cats = np.unique(species)
    sp_indices = {sp: np.where(species == sp)[0] for sp in species_cats}  # precompute once
    for g in false_idx:
        # Shuffle which species' cells get which expression values
        # Enforce derangement (no species maps to itself). For K=2 species,
        # a plain shuffle produces the identity permutation 50% of the time (1/K!),
        # making the "false ortholog" gene unchanged — silently weakening the test.
        # For K=2 this loop always terminates on the first retry (only 2 permutations).
        # For K>2, probability of non-derangement is ~1/e ~ 37%, so expected retries < 2.
        shuffled_species = list(species_cats)
        while True:
            rng.shuffle(shuffled_species)
            if all(s != o for s, o in zip(shuffled_species, species_cats)):
                break  # valid derangement
        col_backup = X[:, g].copy()
        for orig_sp, shuf_sp in zip(species_cats, shuffled_species):
            if orig_sp == shuf_sp:
                continue
            # Species may have unequal cell counts — resample with
            # replacement from source to match target size, avoiding ValueError
            source_vals = col_backup[sp_indices[shuf_sp]]
            target_size = len(sp_indices[orig_sp])
            if len(source_vals) != target_size:
                source_vals = rng.choice(source_vals, size=target_size, replace=True)
            X[sp_indices[orig_sp], g] = source_vals
    # Write back (handles both dense and sparse)
    if sp_sparse.issparse(adata_false.X):
        adata_false.X = sp_sparse.csr_matrix(X)
    else:
        adata_false.X = X

    # Also shuffle gene features for false genes (breaks ESM-2 -> expression link)
    if "gene_features" in adata_false.varm:
        gf = adata_false.varm["gene_features"].copy()
        shuffled_gf_idx = false_idx.copy()
        rng.shuffle(shuffled_gf_idx)
        gf[false_idx] = gf[shuffled_gf_idx]
        adata_false.varm["gene_features"] = gf

    ChimeraModel.setup_anndata(adata_false, species_key="species", batch_key="batch")
    model = ChimeraModel.from_config(adata_false, "D")
    import scvi
    scvi.settings.seed = seed
    model.train(max_epochs=max_epochs)
    alpha = model.get_alpha()
    delta = model.get_delta_alpha()

    true_mask = np.ones(G, dtype=bool)
    true_mask[false_idx] = False

    return {
        "n_false_orthologs": n_false,
        "mean_alpha_true": float(alpha[true_mask].mean()),
        "mean_alpha_false": float(alpha[~true_mask].mean()),
        "mean_abs_delta_true": float(np.abs(delta[true_mask]).mean()),
        "mean_abs_delta_false": float(np.abs(delta[~true_mask]).mean()),
        # False orthologs should have smaller |delta_alpha| than true orthologs
        "false_orthologs_near_prior": float(np.abs(delta[~true_mask]).mean()) < float(np.abs(delta[true_mask]).mean()),
    }


def stress_test_batch_structure(adata, max_epochs=150, seed=42):
    """Test with single-batch-per-species (degenerate case where batch offset
    is forced to zero) and with minimal two-batch-per-species."""
    results = {}

    # --- Single batch per species: set batch_key = species_key ---
    adata_single = adata.copy()
    adata_single.obs["batch_as_species"] = adata_single.obs["species"].copy()
    ChimeraModel.setup_anndata(adata_single, species_key="species",
                                batch_key="batch_as_species")
    model_single = ChimeraModel.from_config(adata_single, "D")
    import scvi
    scvi.settings.seed = seed  # scvi-tools train() does not accept seed kwarg
    model_single.train(max_epochs=max_epochs)

    # Verify batch offset is zero (Component 7 degenerate case)
    offset = model_single.module.get_constrained_batch_offset()
    results["single_batch"] = {
        "max_offset": float(offset.abs().max().item()),
        "offset_is_zero": float(offset.abs().max().item()) < 1e-5,
        "alpha_mean": float(model_single.get_alpha().mean()),
        "alpha_std": float(model_single.get_alpha().std()),
    }

    # --- Multi-batch (standard): use original batch_key ---
    if adata.obs["batch"].nunique() > adata.obs["species"].nunique():
        adata_multi = adata.copy()
        ChimeraModel.setup_anndata(adata_multi, species_key="species", batch_key="batch")
        model_multi = ChimeraModel.from_config(adata_multi, "D")
        scvi.settings.seed = seed
        model_multi.train(max_epochs=max_epochs)

        # Compare alpha between single-batch and multi-batch
        r, _ = spearmanr(model_single.get_alpha(), model_multi.get_alpha())
        results["multi_vs_single_batch"] = {
            "alpha_spearman_r": float(r),
            "expected": "r > 0.7 — batch structure should not radically change alpha ranking",
        }

    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Robustness stress tests for Chimera v2.4 (Task 9.8).",
    )
    parser.add_argument(
        "--adata", type=str, required=True,
        help="Path to AnnData .h5ad file with species/batch/conservation annotations.",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=150,
        help="Max training epochs for each stress test model (default: 150).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results (default: prints to stdout).",
    )
    args = parser.parse_args()

    import anndata as ad

    logger.info("Loading AnnData from %s", args.adata)
    adata = ad.read_h5ad(args.adata)
    logger.info("Loaded %d cells x %d genes", adata.n_obs, adata.n_vars)

    # --- Ortholog dropout ---
    logger.info("Running stress_test_ortholog_dropout ...")
    df_dropout = stress_test_ortholog_dropout(
        adata, max_epochs=args.max_epochs, seed=args.seed,
    )
    logger.info("Ortholog dropout results:\n%s", df_dropout.to_string(index=False))

    # --- Shuffled conservation ---
    logger.info("Running stress_test_shuffled_conservation ...")
    shuf_results = stress_test_shuffled_conservation(
        adata, max_epochs=args.max_epochs, seed=args.seed,
    )
    logger.info("Shuffled conservation results: %s", shuf_results)

    # --- False orthologs ---
    logger.info("Running stress_test_false_orthologs ...")
    false_results = stress_test_false_orthologs(
        adata, max_epochs=args.max_epochs, seed=args.seed,
    )
    logger.info("False orthologs results: %s", false_results)

    # --- Batch structure ---
    logger.info("Running stress_test_batch_structure ...")
    batch_results = stress_test_batch_structure(
        adata, max_epochs=args.max_epochs, seed=args.seed,
    )
    logger.info("Batch structure results: %s", batch_results)

    # --- Save ---
    if args.output_dir is not None:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        df_dropout.to_csv(out / "ortholog_dropout.csv", index=False)
        pd.DataFrame([shuf_results]).to_csv(out / "shuffled_conservation.csv", index=False)
        pd.DataFrame([false_results]).to_csv(out / "false_orthologs.csv", index=False)

        import json
        with open(out / "batch_structure.json", "w") as f:
            json.dump(batch_results, f, indent=2)

        logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
