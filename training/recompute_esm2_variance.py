"""Recompute the ESM-2 inter-species variance feature in-place on unified AnnData.

Background
----------
During audit we discovered that ``varm['gene_features'][:, 64]`` (the ESM-2
inter-species variance feature) is all zeros in the current pipeline output.
The v2.4 spec (`critical_reexamination_modules.md` §74, 78-86) treats this
feature as a primary signal for alpha learning: it captures "how much does the
protein structure differ across species for this gene", a signal not encoded
by either the consensus ESM-2 PCA or the amino-acid identity percentage.

Root cause: `notebooks/01_data_preparation.ipynb` cell 10 looks for cached
``{species}_esm2.npy`` files that are never created by
``download_and_process_esm2()``; the raw ``.torch`` files exist but are never
loaded for the variance computation. `compute_esm2_interspecies_variance` then
receives an empty or sparse dict and returns constant values, which StandardScaler
turns into zeros.

This script fixes feature 64 in-place without rebuilding the entire unified
AnnData. It:
  1. Loads each species' per-gene-symbol ESM-2 embeddings from the raw .torch files.
  2. For each ortholog group (row of ortholog_table.csv), looks up each species'
     embedding by the per-species gene name column.
  3. Computes mean pairwise cosine distance across species → per-gene (G,) array.
  4. Standard-scales the result and writes it into column 64 of
     `varm['gene_features']` on the target unified h5ad (and saves a backup).

This keeps every other feature unchanged. After running, rebuild the subset with
`build_subset.py` so the subset inherits the corrected feature.

Usage
-----
    python training/recompute_esm2_variance.py \
        --unified data/processed/camex_pancreas_unified.h5ad \
        --ortholog data/processed/ortholog_table.csv \
        --esm2 data/raw/saturn_esm2/protein_embeddings_export/ESM2
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Ortholog-table column ↔ SATURN filename prefix map.
# Ensembl prefix → (ortholog_table column prefix, list of candidate filename stems)
_SPECIES_MAP = {
    "hsapiens": ("hsapiens", ["human"]),
    "mmulatta": ("mmulatta", ["macaca_mulatta"]),
    "mfascicularis": ("mfascicularis", ["macaca_fascicularis"]),
    "mmusculus": ("mmusculus", ["mouse"]),
    "drerio": ("drerio", ["zebrafish"]),
    "sscrofa": ("sscrofa", ["pig"]),
    "xlaevis": ("xlaevis", ["frog"]),
    "mmurinus": ("mmurinus", ["mouse_lemur"]),
}


def _load_species_embeddings(esm2_dir: Path, stems: list[str]) -> dict | None:
    """Return the {gene_symbol: embedding} dict for the first matching .torch file."""
    for stem in stems:
        for path in esm2_dir.glob(f"{stem}_embedding.torch"):
            raw = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(raw, dict):
                continue
            logger.info("Loaded %s (%d keys)", path.name, len(raw))
            return {k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v)) for k, v in raw.items()}
    return None


def compute_variance(
    ortholog_table: pd.DataFrame,
    esm2_dir: Path,
    requested_species: list[str],
) -> tuple[np.ndarray, dict]:
    """Compute mean pairwise cosine distance per ortholog group.

    Returns
    -------
    variance : np.ndarray of shape (G,), raw mean pairwise cosine distance.
    info     : dict with per-species match counts and a summary.
    """
    G = len(ortholog_table)
    species_embeds: dict[str, np.ndarray] = {}
    info: dict = {"per_species": {}, "G": G}

    for sp in requested_species:
        if sp not in _SPECIES_MAP:
            logger.warning("Species %s not in species map — skipping", sp)
            continue
        col_prefix, stems = _SPECIES_MAP[sp]
        gene_name_col = f"{col_prefix}_gene_name"
        if gene_name_col not in ortholog_table.columns:
            logger.warning("Column %s not in ortholog table — skipping %s", gene_name_col, sp)
            continue

        sp_data = _load_species_embeddings(esm2_dir, stems)
        if sp_data is None:
            logger.warning("No .torch file found for species %s (stems %s)", sp, stems)
            continue

        emb_dim = next(iter(sp_data.values())).shape[0]
        mat = np.zeros((G, emb_dim), dtype=np.float32)
        mask = np.zeros(G, dtype=bool)
        names = ortholog_table[gene_name_col].values
        for i, name in enumerate(names):
            if isinstance(name, str) and name in sp_data:
                mat[i] = sp_data[name][:emb_dim]
                mask[i] = True

        n_matched = int(mask.sum())
        species_embeds[sp] = mat
        info["per_species"][sp] = {"matched": n_matched, "total": G, "emb_dim": emb_dim, "mask": mask}
        logger.info("%s: matched %d / %d ortholog groups (emb_dim=%d)", sp, n_matched, G, emb_dim)

    if len(species_embeds) < 2:
        raise RuntimeError(f"Need at least 2 species with embeddings; got {len(species_embeds)}")

    # Compute pairwise cosine distance per gene, averaged over pairs where BOTH species
    # matched the ortholog. Pairs where one embedding is absent contribute nothing to
    # that gene's sum and are excluded from the count.
    species_names = list(species_embeds.keys())
    normed: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for sp in species_names:
        x = species_embeds[sp]
        norm = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
        normed[sp] = (x / norm).astype(np.float32)
        masks[sp] = info["per_species"][sp]["mask"]

    pair_sum = np.zeros(G, dtype=np.float64)
    pair_count = np.zeros(G, dtype=np.int32)
    for i, s1 in enumerate(species_names):
        for s2 in species_names[i + 1:]:
            both = masks[s1] & masks[s2]
            if not both.any():
                continue
            cos_sim = np.einsum("ij,ij->i", normed[s1], normed[s2])
            cos_dist = 1.0 - cos_sim
            # Only use pair where both species contributed embeddings.
            pair_sum[both] += cos_dist[both]
            pair_count[both] += 1

    variance = np.zeros(G, dtype=np.float32)
    has_signal = pair_count > 0
    variance[has_signal] = (pair_sum[has_signal] / pair_count[has_signal]).astype(np.float32)
    # Genes with no contributing pairs are set to the median of the informative genes
    # so they are not outliers after standardization. We could also exclude them, but
    # we prefer preserving G.
    if has_signal.any():
        median_val = float(np.median(variance[has_signal]))
        variance[~has_signal] = median_val
    info["n_has_signal"] = int(has_signal.sum())
    info["mean_variance"] = float(variance.mean())
    info["std_variance"] = float(variance.std())
    return variance, info


def patch_gene_features(adata: ad.AnnData, variance: np.ndarray, feature_idx: int = 64) -> None:
    """Replace feature column `feature_idx` with StandardScaled `variance`."""
    gf = np.asarray(adata.varm["gene_features"]).copy()
    if variance.shape[0] != gf.shape[0]:
        raise ValueError(f"Length mismatch: variance {variance.shape} vs gene_features {gf.shape}")
    mu = float(variance.mean())
    sd = float(variance.std())
    if sd < 1e-8:
        logger.warning("Recomputed ESM-2 variance still has std=0; something is wrong upstream.")
        return
    normed = (variance - mu) / sd
    gf[:, feature_idx] = normed.astype(np.float32)
    adata.varm["gene_features"] = gf
    logger.info(
        "Patched gene_features[:, %d]: raw mean=%.4f std=%.4f -> after scaling mean=%.4f std=%.4f",
        feature_idx, mu, sd, float(gf[:, feature_idx].mean()), float(gf[:, feature_idx].std()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified", type=Path, required=True)
    parser.add_argument("--ortholog", type=Path, required=True)
    parser.add_argument("--esm2", type=Path, required=True,
                        help="Directory containing per-species *_embedding.torch files (ESM2 subdir).")
    parser.add_argument("--feature-idx", type=int, default=64,
                        help="Index into gene_features where ESM-2 variance lives (default 64).")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    ortholog_table = pd.read_csv(args.ortholog)
    logger.info("Loaded ortholog table: %s genes, columns=%s", len(ortholog_table), list(ortholog_table.columns))

    # Species in the ortholog table = unique Ensembl prefixes detected from column names
    species_in_table = sorted({
        c.split("_gene_name")[0]
        for c in ortholog_table.columns
        if c.endswith("_gene_name")
    })
    logger.info("Species present in ortholog table: %s", species_in_table)

    variance, info = compute_variance(ortholog_table, args.esm2, species_in_table)
    logger.info("Variance summary: %s", {k: v for k, v in info.items() if k != "per_species"})

    if not args.no_backup:
        backup = args.unified.with_suffix(".h5ad.bak_pre_esm2var")
        if not backup.exists():
            shutil.copy2(args.unified, backup)
            logger.info("Backup saved: %s", backup)

    logger.info("Loading %s", args.unified)
    adata = ad.read_h5ad(args.unified)
    patch_gene_features(adata, variance, args.feature_idx)
    adata.write_h5ad(args.unified)
    logger.info("Wrote patched %s", args.unified)

    # Sanity check: re-read and confirm
    adata2 = ad.read_h5ad(args.unified)
    col = adata2.varm["gene_features"][:, args.feature_idx]
    logger.info("Post-write check: feature %d std=%.4f mean=%.4f", args.feature_idx, col.std(), col.mean())
    return 0


if __name__ == "__main__":
    sys.exit(main())
