"""Compute the full scIB metric suite and Chimera-specific cross-species metrics.

Metrics computed:
  Batch correction (4):
    - batch_ASW        (average silhouette width on batch labels)
    - kBET             (k-nearest-neighbour batch effect test)
    - graph_connectivity  (fraction of cells reachable within same label)
    - iLISI            (integration LISI -- diversity of batches in local neighbourhood)

  Biology preservation (5):
    - cell_type_ASW    (average silhouette width on cell-type labels)
    - NMI              (normalised mutual information of Leiden vs cell_type)
    - ARI              (adjusted Rand index of Leiden vs cell_type)
    - isolated_label_ASW  (ASW for cell types present in a single batch)
    - cLISI            (cell-type LISI -- purity of cell types in local neighbourhood)

  Chimera-specific (1):
    - cross_species_label_transfer  (leave-one-species-out KNN accuracy)

Reads an AnnData with latent representations in obsm (as produced by
train_baselines.py) and outputs a comparison CSV.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# scIB metric wrappers
# ------------------------------------------------------------------


def _safe_scib_metric(func, *args, **kwargs):
    """Call a scib metric function; return NaN on failure."""
    try:
        result = func(*args, **kwargs)
        return float(result) if result is not None else float("nan")
    except Exception as exc:
        logger.warning("scib metric %s failed: %s", func.__name__, exc)
        return float("nan")


def _compute_scib_metrics(
    adata: ad.AnnData,
    embed_key: str,
    batch_key: str = "species",
    label_key: str = "cell_type",
) -> Dict[str, float]:
    """Compute the full scIB metric suite for one embedding.

    Parameters
    ----------
    adata
        AnnData with the embedding in ``obsm[embed_key]``.
    embed_key
        Key in ``adata.obsm`` for the latent representation.
    batch_key
        obs column for batch (species).
    label_key
        obs column for cell-type labels.

    Returns
    -------
    dict
        Metric name -> value.
    """
    import scib

    # Work on a copy so neighbour graphs do not leak across calls
    ad_eval = adata.copy()
    ad_eval.obsm["X_emb"] = ad_eval.obsm[embed_key]

    # Pre-compute neighbours for downstream metrics
    sc.pp.neighbors(ad_eval, use_rep="X_emb")

    metrics: Dict[str, float] = {}
    has_labels = label_key in ad_eval.obs.columns

    # ---- Batch correction metrics ----
    metrics["batch_ASW"] = _safe_scib_metric(
        scib.metrics.silhouette_batch,
        ad_eval,
        batch_key=batch_key,
        label_key=label_key if has_labels else batch_key,
        embed="X_emb",
    )

    metrics["kBET"] = _safe_scib_metric(
        scib.metrics.kBET,
        ad_eval,
        batch_key=batch_key,
        label_key=label_key if has_labels else batch_key,
        type_="embed",
        embed="X_emb",
    )

    metrics["graph_connectivity"] = _safe_scib_metric(
        scib.metrics.graph_connectivity,
        ad_eval,
        label_key=label_key if has_labels else batch_key,
    )

    metrics["iLISI"] = _safe_scib_metric(
        scib.metrics.ilisi_graph,
        ad_eval,
        batch_key=batch_key,
        type_="embed",
        use_rep="X_emb",
    )

    # ---- Biology preservation metrics (require cell_type) ----
    if has_labels:
        metrics["cell_type_ASW"] = _safe_scib_metric(
            scib.metrics.silhouette,
            ad_eval,
            label_key=label_key,
            embed="X_emb",
        )

        # Leiden clustering for NMI / ARI
        sc.tl.leiden(ad_eval, resolution=1.0, key_added="leiden")

        metrics["NMI"] = _safe_scib_metric(
            scib.metrics.nmi,
            ad_eval,
            cluster_key="leiden",
            label_key=label_key,
        )

        metrics["ARI"] = _safe_scib_metric(
            scib.metrics.ari,
            ad_eval,
            cluster_key="leiden",
            label_key=label_key,
        )

        metrics["isolated_label_ASW"] = _safe_scib_metric(
            scib.metrics.isolated_labels_asw,
            ad_eval,
            batch_key=batch_key,
            label_key=label_key,
            embed="X_emb",
        )

        metrics["cLISI"] = _safe_scib_metric(
            scib.metrics.clisi_graph,
            ad_eval,
            label_key=label_key,
            type_="embed",
            use_rep="X_emb",
        )
    else:
        for m in ("cell_type_ASW", "NMI", "ARI", "isolated_label_ASW", "cLISI"):
            metrics[m] = float("nan")

    return metrics


# ------------------------------------------------------------------
# Cross-species label transfer
# ------------------------------------------------------------------


def cross_species_label_transfer(
    adata: ad.AnnData,
    embed_key: str,
    species_key: str = "species",
    label_key: str = "cell_type",
    n_neighbors: int = 15,
) -> Dict[str, float]:
    """Leave-one-species-out KNN label transfer accuracy.

    For each target species, trains a KNN classifier on all other species
    and predicts cell-type labels on the held-out species.

    Parameters
    ----------
    adata
        AnnData with the embedding in ``obsm[embed_key]``.
    embed_key
        Key in ``adata.obsm`` for the latent representation.
    species_key
        obs column for species.
    label_key
        obs column for cell-type labels.
    n_neighbors
        Number of neighbours for KNN.

    Returns
    -------
    dict
        ``{species_name: accuracy, ..., "mean_accuracy": float}``.
    """
    from sklearn.neighbors import KNeighborsClassifier

    if label_key not in adata.obs.columns:
        logger.info(
            "Skipping cross-species label transfer: %s not in adata.obs.", label_key
        )
        return {"mean_accuracy": float("nan")}

    X = adata.obsm[embed_key]
    species = adata.obs[species_key].values
    labels = adata.obs[label_key].values
    unique_species = np.unique(species)

    if len(unique_species) < 2:
        logger.warning("Only one species present; cannot do leave-one-out transfer.")
        return {"mean_accuracy": float("nan")}

    per_species_acc = {}

    for target_sp in unique_species:
        target_mask = species == target_sp
        train_mask = ~target_mask

        X_train, y_train = X[train_mask], labels[train_mask]
        X_test, y_test = X[target_mask], labels[target_mask]

        # Skip species with cell types unseen in training
        train_classes = set(y_train)
        test_classes = set(y_test)
        # Only evaluate on cells whose label exists in training data
        evaluable = np.array([y in train_classes for y in y_test])

        if evaluable.sum() == 0:
            logger.warning(
                "Species %s has no cell types shared with training set; skipping.",
                target_sp,
            )
            per_species_acc[str(target_sp)] = float("nan")
            continue

        knn = KNeighborsClassifier(
            n_neighbors=min(n_neighbors, len(X_train)),
            metric="cosine",
        )
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test[evaluable])
        acc = float((preds == y_test[evaluable]).mean())
        per_species_acc[str(target_sp)] = acc
        logger.info("  %s -> accuracy %.4f (%d evaluable cells)", target_sp, acc, evaluable.sum())

    valid_accs = [v for v in per_species_acc.values() if np.isfinite(v)]
    mean_acc = float(np.mean(valid_accs)) if valid_accs else float("nan")
    per_species_acc["mean_accuracy"] = mean_acc

    logger.info("Cross-species label transfer mean accuracy: %.4f", mean_acc)
    return per_species_acc


# ------------------------------------------------------------------
# Main orchestrator
# ------------------------------------------------------------------


def compute_all_metrics(
    adata_path: str,
    latent_keys: List[str],
    output_path: str,
    batch_key: str = "species",
    label_key: str = "cell_type",
):
    """Compute all metrics for the given latent representations and save CSV.

    Parameters
    ----------
    adata_path
        Path to AnnData .h5ad with latent reps in obsm.
    latent_keys
        List of obsm keys to evaluate.
    output_path
        Path for the output CSV comparison table.
    batch_key
        obs column for batch (species).
    label_key
        obs column for cell-type labels.
    """
    logger.info("Loading AnnData from %s", adata_path)
    adata = ad.read_h5ad(adata_path)
    logger.info("AnnData: %d cells x %d genes", adata.n_obs, adata.n_vars)

    # Validate requested keys exist
    available_keys = [k for k in latent_keys if k in adata.obsm]
    missing_keys = [k for k in latent_keys if k not in adata.obsm]
    if missing_keys:
        logger.warning("Keys not found in obsm (skipping): %s", missing_keys)
    if not available_keys:
        raise ValueError(
            f"None of the requested latent keys found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )

    all_rows = []

    for key in available_keys:
        logger.info("--- Evaluating: %s ---", key)

        # scIB metrics
        scib_metrics = _compute_scib_metrics(
            adata, embed_key=key, batch_key=batch_key, label_key=label_key
        )

        # Cross-species label transfer
        xfer = cross_species_label_transfer(
            adata,
            embed_key=key,
            species_key=batch_key,
            label_key=label_key,
        )

        row = {"method": key}
        row.update(scib_metrics)
        row["cross_species_transfer_acc"] = xfer.get("mean_accuracy", float("nan"))

        # Include per-species transfer accuracies
        for sp_name, sp_acc in xfer.items():
            if sp_name != "mean_accuracy":
                row[f"transfer_{sp_name}"] = sp_acc

        all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Metric comparison table saved to %s", output_path)
    logger.info("\n%s", df.to_string(index=False))

    return df


def main():
    """CLI entry point for compute_metrics."""
    parser = argparse.ArgumentParser(
        description="Compute scIB + Chimera cross-species metrics for latent representations.",
    )
    parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="Path to AnnData .h5ad with latent representations in obsm.",
    )
    parser.add_argument(
        "--latent-keys",
        type=str,
        required=True,
        help="Comma-separated obsm keys to evaluate (e.g. X_scVI,X_Harmony,X_Chimera_D).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/metrics.csv",
        help="Output CSV path for the comparison table.",
    )
    parser.add_argument(
        "--batch-key",
        type=str,
        default="species",
        help="obs column for batch / species.",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="cell_type",
        help="obs column for cell-type labels.",
    )

    args = parser.parse_args()
    keys = [k.strip() for k in args.latent_keys.split(",") if k.strip()]

    compute_all_metrics(
        adata_path=args.adata,
        latent_keys=keys,
        output_path=args.output,
        batch_key=args.batch_key,
        label_key=args.label_key,
    )


if __name__ == "__main__":
    main()
