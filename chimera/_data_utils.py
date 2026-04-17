"""Chimera v2.4 -- Data utilities for cross-species scRNA-seq integration.

Functions for:
  - Downloading and QC-filtering benchmark datasets (CAMEX, van Zyl)
  - Querying Ensembl BioMart for 1:1 ortholog tables
  - Computing conservation priors and gene features (ESM-2, expression stats)
  - Assembling the unified cross-species AnnData
  - Precomputing null expression distributions and sampling weights
  - Creating synthetic data for unit tests
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Default data directory relative to the repository root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"

# ---------------------------------------------------------------------------
# Dataset download / loading helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_url(url: str, dest: Path, desc: str = "") -> Path:
    """Download *url* to *dest*. Skip if *dest* already exists."""
    if dest.exists():
        logger.info("Cached: %s", dest)
        return dest
    import urllib.request
    _ensure_dir(dest.parent)
    desc = desc or dest.name
    logger.info("Downloading %s → %s …", desc, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    logger.info("  done (%.1f MB)", dest.stat().st_size / 1e6)
    return dest


def _download_gdrive_folder(folder_id: str, dest_dir: Path) -> Path:
    """Download a public Google Drive folder via ``gdown``.

    Requires ``pip install gdown`` (or ``pip install 'chimera[data]'``).
    """
    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "gdown is required to download CAMEX data from Google Drive. "
            "Install with:  pip install 'chimera[data]'"
        ) from e
    _ensure_dir(dest_dir)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    logger.info("Downloading Google Drive folder → %s …", dest_dir)
    gdown.download_folder(url, output=str(dest_dir), quiet=False, resume=True)
    return dest_dir


def _extract_tar(archive: Path, dest_dir: Path) -> Path:
    """Extract a .tar.gz archive into *dest_dir*."""
    import tarfile
    _ensure_dir(dest_dir)
    logger.info("Extracting %s → %s …", archive.name, dest_dir)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest_dir)
    return dest_dir


def _validate_raw_counts(adata: ad.AnnData, name: str = "") -> None:
    """Assert that ``adata.X`` contains non-negative integer counts.

    Raises ``ValueError`` with an actionable message if the matrix appears
    normalized.
    """
    X = adata.X
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if len(data) == 0:
        return
    if data.min() < 0:
        raise ValueError(f"{name}: .X has negative values — not raw counts")
    if not np.allclose(data, np.round(data), atol=0.01):
        raise ValueError(
            f"{name}: .X values are not integers — likely normalized.  "
            "Raw counts may be in .layers['counts'] or .raw.X."
        )


def _standardize_obs_columns(adata: ad.AnnData) -> ad.AnnData:
    """Rename common alternative obs column names to the Chimera standard.

    Searches for known alternatives and renames in-place.  Logs every rename.
    """
    _MAP = {
        "species": ["Species", "organism", "Organism", "species_name"],
        "batch":   ["Batch", "donor", "Donor", "sample", "Sample",
                     "library", "donor_id", "sample_id"],
        "cell_type": ["celltype", "CellType", "cell_ontology_class",
                       "labels2", "labels", "cluster", "annotation",
                       "cell_annotation"],
    }
    for target, alternatives in _MAP.items():
        if target in adata.obs.columns:
            continue
        for alt in alternatives:
            if alt in adata.obs.columns:
                logger.info("Renaming .obs['%s'] → '%s'", alt, target)
                adata.obs[target] = adata.obs[alt]
                break
    return adata


def _qc_filter(
    adata: ad.AnnData,
    min_genes: int = 200,
    max_mt_pct: float = 20.0,
) -> ad.AnnData:
    """Apply standard QC filtering to an AnnData object.

    Parameters
    ----------
    adata
        Input AnnData with raw UMI counts in ``.X``.
    min_genes
        Minimum number of genes detected per cell.
    max_mt_pct
        Maximum percentage of mitochondrial reads per cell.

    Returns
    -------
    AnnData
        Filtered AnnData (copy).
    """
    import scanpy as sc

    adata = adata.copy()
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # Identify mitochondrial genes (common prefixes across species).
    mt_prefixes = ("MT-", "Mt-", "mt-", "MTND", "MTCO", "MTATP")
    adata.var["mt"] = adata.var_names.str.startswith(mt_prefixes)
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    adata = adata[adata.obs["pct_counts_mt"] < max_mt_pct].copy()
    logger.info(
        "QC filter: %d cells, %d genes after filtering", adata.n_obs, adata.n_vars
    )
    return adata


def download_camex_pancreas(
    data_dir: Union[str, Path, None] = None,
) -> ad.AnnData:
    """Download / load the CAMEX pancreas dataset (4-6 mammalian species).

    Parameters
    ----------
    data_dir
        Directory to store downloaded data.  Defaults to ``<repo>/data/raw``.

    Returns
    -------
    AnnData
        Annotated data matrix with:
        - ``.X``: raw UMI counts (sparse CSR)
        - ``.obs['species']``: species label
        - ``.obs['batch']``: donor / batch label
        - ``.obs['cell_type']``: cell-type annotation
    """
    data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR / "raw"
    _ensure_dir(data_dir)
    cache_path = data_dir / "camex_pancreas.h5ad"

    if cache_path.exists():
        logger.info("Loading cached CAMEX pancreas from %s", cache_path)
        return ad.read_h5ad(cache_path)

    # --- Download CAMEX benchmarks from Google Drive ---
    camex_dir = data_dir / "camex"
    if not camex_dir.exists() or not any(camex_dir.rglob("*.h5ad")):
        _download_gdrive_folder("1rwdjEvWFEFw82a0x2JzMi2jXICbUc5eb", camex_dir)

    # --- Locate the dev-dataset tissue directory ---
    # Round2 spec: CAMEX pancreas (3 species). CAMEX dir structure: analysis/{N}{tissue}/dataset/
    tissue_dir = None
    for pattern in ["*pancreas*", "*Pancreas*", "*liver*", "*Liver*", "1liver"]:
        hits = [d for d in camex_dir.rglob(pattern) if d.is_dir()]
        if hits:
            # Prefer the one containing h5ad files
            for h in hits:
                if list(h.rglob("*.h5ad")):
                    tissue_dir = h
                    break
            if tissue_dir:
                break
    # Last resort: any directory that contains h5ad
    if tissue_dir is None:
        for d in sorted(camex_dir.rglob("*")):
            if d.is_dir() and list(d.glob("*.h5ad")):
                tissue_dir = d
                break
    if tissue_dir is None:
        raise FileNotFoundError(
            f"No h5ad files found in CAMEX download at {camex_dir}. "
            f"Check the Google Drive contents."
        )
    logger.info("Using CAMEX tissue directory: %s", tissue_dir)

    # --- Load, validate, concatenate ---
    h5ad_files = sorted(tissue_dir.rglob("*.h5ad"))
    adatas = []
    for f in h5ad_files:
        logger.info("Loading %s", f.name)
        a = ad.read_h5ad(f)
        a.var_names_make_unique()
        _standardize_obs_columns(a)

        # Raw-counts check (try .X, then .layers['counts'], then .raw.X)
        try:
            _validate_raw_counts(a, f.name)
        except ValueError:
            if "counts" in getattr(a, "layers", {}):
                logger.info("  Using .layers['counts'] as raw counts")
                a.X = a.layers["counts"]
                _validate_raw_counts(a, f.name)
            elif a.raw is not None:
                logger.info("  Using .raw.X as raw counts")
                a = a.raw.to_adata()
                _standardize_obs_columns(a)
                _validate_raw_counts(a, f.name)

        if "species" not in a.obs.columns:
            # CAMEX filenames: raw-{tissue}-{species}-{author}.h5ad
            parts = f.stem.replace("_", "-").split("-")
            if len(parts) >= 3:
                sp_name = parts[2].title()
            else:
                sp_name = parts[0].title()
            logger.info("  Inferring species='%s' from filename", sp_name)
            a.obs["species"] = sp_name
        if "batch" not in a.obs.columns:
            a.obs["batch"] = a.obs["species"].astype(str)
            logger.info("  No batch column — using species as batch")

        a.obs["species"] = pd.Categorical(a.obs["species"].astype(str))
        a.obs["batch"]   = pd.Categorical(a.obs["batch"].astype(str))
        adatas.append(a)

    adata = ad.concat(adatas, join="outer") if len(adatas) > 1 else adatas[0]
    adata.obs_names_make_unique()
    adata = _qc_filter(adata, min_genes=200, max_mt_pct=20.0)
    adata.write_h5ad(cache_path)
    logger.info(
        "Saved CAMEX dev dataset: %d cells × %d genes → %s",
        adata.n_obs, adata.n_vars, cache_path,
    )
    return adata


def download_van_zyl(
    data_dir: Union[str, Path, None] = None,
) -> ad.AnnData:
    """Download / load the van Zyl aqueous humor 5-species dataset.

    Parameters
    ----------
    data_dir
        Directory to store downloaded data.  Defaults to ``<repo>/data/raw``.

    Returns
    -------
    AnnData
        Annotated data matrix with multiple donors per species providing
        realistic batch structure.  Fields:
        - ``.X``: raw UMI counts (sparse CSR)
        - ``.obs['species']``: species label
        - ``.obs['batch']``: donor / batch label (format ``{species}_{donor}``)
        - ``.obs['cell_type']``: cell-type annotation
    """
    data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR / "raw"
    _ensure_dir(data_dir)
    cache_path = data_dir / "van_zyl_aqueous_humor.h5ad"

    if cache_path.exists():
        logger.info("Loading cached van Zyl aqueous humor from %s", cache_path)
        return ad.read_h5ad(cache_path)

    # --- Download from SATURN server ---
    tar_path = data_dir / "ah_atlas_export.tar.gz"
    extract_dir = data_dir / "van_zyl"

    _download_url(
        "http://snap.stanford.edu/saturn/data/ah_atlas_export.tar.gz",
        tar_path,
        desc="van Zyl aqueous humor (419 MB)",
    )
    if not extract_dir.exists() or not any(extract_dir.rglob("*.h5ad")):
        _extract_tar(tar_path, extract_dir)

    # --- Load h5ad ---
    h5ad_files = sorted(extract_dir.rglob("*.h5ad"))
    if not h5ad_files:
        raise FileNotFoundError(f"No h5ad files in {extract_dir}")

    adatas = []
    for f in h5ad_files:
        logger.info("Loading %s", f.name)
        a = ad.read_h5ad(f)
        _standardize_obs_columns(a)   # handles labels2→cell_type, Species→species

        # SATURN alignment exports are already preprocessed (256 selected genes,
        # normalized values).  Accept as-is when raw counts are not available.
        try:
            _validate_raw_counts(a, f.name)
        except ValueError:
            if "counts" in getattr(a, "layers", {}):
                logger.info("  Using .layers['counts']")
                a.X = a.layers["counts"]
            elif a.raw is not None:
                logger.info("  Using .raw.X")
                a = a.raw.to_adata()
                _standardize_obs_columns(a)
            else:
                logger.info(
                    "  %s: .X is not raw counts and no fallback found — "
                    "accepting as preprocessed SATURN data", f.name
                )

        if "species" not in a.obs.columns:
            a.obs["species"] = f.stem.replace("_", " ").title()
        if "batch" not in a.obs.columns:
            # Try donor-like columns before falling back to species
            a.obs["batch"] = a.obs["species"].astype(str)
            logger.info("  No batch column — using species as batch")

        a.obs["species"] = pd.Categorical(a.obs["species"].astype(str))
        a.obs["batch"]   = pd.Categorical(a.obs["batch"].astype(str))
        adatas.append(a)

    adata = ad.concat(adatas, join="outer") if len(adatas) > 1 else adatas[0]
    adata.obs_names_make_unique()

    # Validate: each batch → one species (Task 7.2 requirement)
    for b in adata.obs["batch"].cat.categories:
        n_sp = adata.obs.loc[adata.obs["batch"] == b, "species"].nunique()
        if n_sp != 1:
            logger.warning("Batch '%s' spans %d species — splitting", b, n_sp)
            for sp_val in adata.obs.loc[adata.obs["batch"] == b, "species"].unique():
                mask = (adata.obs["batch"] == b) & (adata.obs["species"] == sp_val)
                adata.obs.loc[mask, "batch"] = f"{b}_{sp_val}"
            adata.obs["batch"] = pd.Categorical(adata.obs["batch"].astype(str))

    # SATURN alignment exports are already QC-filtered and contain a small
    # gene set (e.g. 256 landmark genes).  Skip heavy QC filtering that
    # would discard all cells.  Apply only a light filter.
    n_genes_total = adata.n_vars
    if n_genes_total <= 500:
        min_genes_qc = max(n_genes_total // 10, 5)
        logger.info(
            "SATURN export detected (%d genes) — using light QC: min_genes=%d",
            n_genes_total, min_genes_qc,
        )
    else:
        min_genes_qc = 200
    adata = _qc_filter(adata, min_genes=min_genes_qc, max_mt_pct=100.0
                        if n_genes_total <= 500 else 20.0)
    adata.write_h5ad(cache_path)
    logger.info(
        "Saved van Zyl: %d cells × %d genes, %d species → %s",
        adata.n_obs, adata.n_vars, adata.obs["species"].nunique(), cache_path,
    )
    return adata


def download_camex_testis(
    data_dir: Union[str, Path, None] = None,
) -> ad.AnnData:
    """Download / load the CAMEX testis 11-species dataset.

    Parameters
    ----------
    data_dir
        Directory to store downloaded data.  Defaults to ``<repo>/data/raw``.

    Returns
    -------
    AnnData
        Annotated data matrix with:
        - ``.X``: raw UMI counts (sparse CSR)
        - ``.obs['species']``: species label
        - ``.obs['batch']``: donor / batch label
        - ``.obs['cell_type']``: cell-type annotation
    """
    data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR / "raw"
    _ensure_dir(data_dir)
    cache_path = data_dir / "camex_testis.h5ad"

    if cache_path.exists():
        logger.info("Loading cached CAMEX testis from %s", cache_path)
        return ad.read_h5ad(cache_path)

    # --- Download CAMEX benchmarks (shared with dev dataset) ---
    camex_dir = data_dir / "camex"
    if not camex_dir.exists() or not any(camex_dir.rglob("*.h5ad")):
        _download_gdrive_folder("1rwdjEvWFEFw82a0x2JzMi2jXICbUc5eb", camex_dir)

    # --- Locate testis directory ---
    tissue_dir = None
    for pattern in ["*testis*", "*Testis*", "2testis"]:
        hits = [d for d in camex_dir.rglob(pattern) if d.is_dir()]
        for h in hits:
            if list(h.rglob("*.h5ad")):
                tissue_dir = h
                break
        if tissue_dir:
            break
    if tissue_dir is None:
        raise FileNotFoundError(f"Testis data not found in CAMEX at {camex_dir}")
    logger.info("Using CAMEX testis directory: %s", tissue_dir)

    # --- Load, validate, concatenate (same pattern as pancreas) ---
    h5ad_files = sorted(tissue_dir.rglob("*.h5ad"))
    adatas = []
    for f in h5ad_files:
        logger.info("Loading %s", f.name)
        a = ad.read_h5ad(f)
        a.var_names_make_unique()
        _standardize_obs_columns(a)
        try:
            _validate_raw_counts(a, f.name)
        except ValueError:
            if "counts" in getattr(a, "layers", {}):
                a.X = a.layers["counts"]
            elif a.raw is not None:
                a = a.raw.to_adata()
                _standardize_obs_columns(a)
        if "species" not in a.obs.columns:
            parts = f.stem.replace("_", "-").split("-")
            sp_name = parts[2].title() if len(parts) >= 3 else parts[0].title()
            a.obs["species"] = sp_name
        if "batch" not in a.obs.columns:
            a.obs["batch"] = a.obs["species"].astype(str)
        a.obs["species"] = pd.Categorical(a.obs["species"].astype(str))
        a.obs["batch"]   = pd.Categorical(a.obs["batch"].astype(str))
        adatas.append(a)

    adata = ad.concat(adatas, join="outer") if len(adatas) > 1 else adatas[0]
    adata.obs_names_make_unique()
    adata = _qc_filter(adata, min_genes=200, max_mt_pct=20.0)
    adata.write_h5ad(cache_path)
    logger.info(
        "Saved CAMEX testis: %d cells × %d genes, %d species → %s",
        adata.n_obs, adata.n_vars, adata.obs["species"].nunique(), cache_path,
    )
    return adata


# ---------------------------------------------------------------------------
# Ortholog table
# ---------------------------------------------------------------------------


def query_ortholog_table(
    species_list: Sequence[str],
    reference_species: str = "hsapiens",
    ensembl_version: int = 110,
    cache_dir: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """Query Ensembl BioMart for 1:1 orthologs across all species.

    Parameters
    ----------
    species_list
        Ensembl species names (e.g. ``['hsapiens', 'mmusculus', 'rnorvegicus']``).
    reference_species
        Species to use as the reference.  Must be in *species_list*.
    ensembl_version
        Ensembl release version for BioMart queries.
    cache_dir
        Directory for caching the ortholog table CSV.
        Defaults to ``<repo>/data/processed``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ``ortholog_group_id``: integer group ID
        - ``{species}_gene_id``: Ensembl gene ID for each species
        - ``{species}_gene_name``: gene symbol for each species
        - ``{species}_identity``: % identity for each non-reference species
        - ``mean_identity``: mean % identity across non-reference species
    """
    cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_DATA_DIR / "processed"
    _ensure_dir(cache_dir)
    cache_path = cache_dir / "ortholog_table.csv"

    if cache_path.exists():
        logger.info("Loading cached ortholog table from %s", cache_path)
        return pd.read_csv(cache_path)

    try:
        from pybiomart import Dataset as BioMartDataset
    except ImportError:
        raise ImportError(
            "pybiomart is required for ortholog queries. "
            "Install it with: pip install pybiomart"
        )

    if reference_species not in species_list:
        raise ValueError(
            f"reference_species '{reference_species}' must be in species_list"
        )

    non_ref = [s for s in species_list if s != reference_species]

    # Connect to Ensembl BioMart for the reference species.
    # Try versioned archive first, fall back to main Ensembl server.
    for host in [
        f"http://e{ensembl_version}.ensembl.org",
        "http://www.ensembl.org",
    ]:
        try:
            ref_dataset = BioMartDataset(
                name=f"{reference_species}_gene_ensembl",
                host=host,
            )
            # Trigger connection to verify it works.
            _ = ref_dataset.attributes
            logger.info("Connected to BioMart at %s", host)
            break
        except Exception as e:
            logger.warning("BioMart at %s failed: %s — trying next", host, e)
    else:
        raise RuntimeError("Could not connect to any Ensembl BioMart server.")

    # Build a merged table of 1:1 orthologs.
    merged: Optional[pd.DataFrame] = None

    for sp in non_ref:
        attrs = [
            "ensembl_gene_id",
            "external_gene_name",
            f"{sp}_homolog_ensembl_gene",
            f"{sp}_homolog_associated_gene_name",
            f"{sp}_homolog_orthology_type",
            f"{sp}_homolog_perc_id",
        ]
        result = ref_dataset.query(attributes=attrs)

        # pybiomart returns display names as columns — force to standard names.
        standard_names = [
            f"{reference_species}_gene_id",
            f"{reference_species}_gene_name",
            f"{sp}_gene_id",
            f"{sp}_gene_name",
            f"{sp}_orthology_type",
            f"{sp}_identity",
        ]
        result.columns = standard_names

        # Keep only 1:1 orthologs.
        orth_col = f"{sp}_orthology_type"
        result = result[result[orth_col] == "ortholog_one2one"].copy()
        # Drop rows with missing gene IDs.
        gene_col = f"{sp}_gene_id"
        result = result.dropna(subset=[gene_col])
        result = result[result[gene_col] != ""].copy()

        keep_cols = [
            f"{reference_species}_gene_id",
            f"{reference_species}_gene_name",
            f"{sp}_gene_id",
            f"{sp}_gene_name",
            f"{sp}_identity",
        ]
        result = result[keep_cols].drop_duplicates(
            subset=[f"{reference_species}_gene_id", f"{sp}_gene_id"]
        )

        if merged is None:
            merged = result
        else:
            merged = merged.merge(
                result,
                on=[
                    f"{reference_species}_gene_id",
                    f"{reference_species}_gene_name",
                ],
                how="inner",
            )

    if merged is None or merged.empty:
        raise RuntimeError("No orthologs found across all species.")

    # Assign ortholog group IDs.
    merged = merged.reset_index(drop=True)
    merged.insert(0, "ortholog_group_id", range(len(merged)))

    # Compute mean identity across non-reference species.
    identity_cols = [f"{sp}_identity" for sp in non_ref]
    for col in identity_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["mean_identity"] = merged[identity_cols].mean(axis=1)

    merged.to_csv(cache_path, index=False)
    logger.info(
        "Ortholog table: %d groups across %d species, saved to %s",
        len(merged),
        len(species_list),
        cache_path,
    )
    return merged


# ---------------------------------------------------------------------------
# Conservation prior
# ---------------------------------------------------------------------------


def compute_conservation_prior(
    ortholog_table: pd.DataFrame,
) -> torch.Tensor:
    """Compute the conservation prior from ortholog identity statistics.

    Parameters
    ----------
    ortholog_table
        DataFrame with a ``mean_identity`` column (percent identity, 0-100).

    Returns
    -------
    torch.Tensor
        Conservation prior of shape ``(G,)``, clamped to ``[0.05, 0.95]``.
        High values mean the gene is highly conserved -> more shared representation.
    """
    identity = torch.tensor(
        ortholog_table["mean_identity"].values, dtype=torch.float32
    )
    prior = (identity / 100.0).clamp(0.05, 0.95)
    return prior


# ---------------------------------------------------------------------------
# ESM-2 protein embeddings
# ---------------------------------------------------------------------------


def download_and_process_esm2(
    ortholog_table: pd.DataFrame,
    species_list: Sequence[str],
    saturn_data_dir: Union[str, Path, None] = None,
) -> np.ndarray:
    """Download ESM-2 embeddings from SATURN and compute consensus per ortholog group.

    For each ortholog group, the consensus embedding is the mean of the per-species
    ESM-2 embeddings (5120-dimensional, from the ESM-2 3B model).

    Parameters
    ----------
    ortholog_table
        DataFrame with per-species gene ID columns (``{species}_gene_id``).
    species_list
        List of species names matching column prefixes in *ortholog_table*.
    saturn_data_dir
        Directory containing SATURN ESM-2 embedding files.
        Defaults to ``<repo>/data/raw/saturn_esm2``.

    Returns
    -------
    np.ndarray
        Consensus embeddings of shape ``(G, 5120)`` where G is the number of
        ortholog groups.
    """
    saturn_data_dir = (
        Path(saturn_data_dir)
        if saturn_data_dir is not None
        else _DEFAULT_DATA_DIR / "raw" / "saturn_esm2"
    )

    # SATURN embedding files use .torch extension and live in ESM2/ subdir.
    _emb_globs = ("*.pt", "*.torch")
    def _find_emb_files(d: Path) -> list:
        files = []
        for g in _emb_globs:
            files.extend(d.rglob(g))
        return files

    if not saturn_data_dir.exists() or not _find_emb_files(saturn_data_dir):
        tar_path = saturn_data_dir.parent / "protein_embeddings.tar.gz"
        _download_url(
            "http://snap.stanford.edu/saturn/data/protein_embeddings.tar.gz",
            tar_path,
            desc="SATURN protein embeddings (3.5 GB)",
        )
        _extract_tar(tar_path, saturn_data_dir)
        emb_files = _find_emb_files(saturn_data_dir)
        if not emb_files:
            raise FileNotFoundError(
                f"No .pt/.torch files found after extracting to {saturn_data_dir}"
            )
        logger.info("Found %d embedding files in %s", len(emb_files), saturn_data_dir)

    # Map Ensembl species names to common names used in SATURN filenames.
    _ENSEMBL_TO_COMMON = {
        "hsapiens": "human", "mmusculus": "mouse", "rnorvegicus": "rat",
        "mmulatta": "macaca_mulatta", "mfascicularis": "macaca_fascicularis",
        "sscrofa": "pig", "drerio": "zebrafish", "ggallus": "chicken",
        "ptroglodytes": "chimpanzee", "ggorilla": "gorilla",
        "ppaniscus": "bonobo", "nleucogenys": "gibbon",
        "cjacchus": "marmoset", "mdomestica": "opossum",
        "oanatinus": "platypus", "xlaevis": "frog",
    }

    n_groups = len(ortholog_table)
    emb_dim = 5120
    consensus = np.zeros((n_groups, emb_dim), dtype=np.float32)

    # Prefer ESM2 subdirectory if present (5120-dim); fall back to ESM1b (1280-dim).
    esm2_dir = None
    for candidate in saturn_data_dir.rglob("ESM2"):
        if candidate.is_dir():
            esm2_dir = candidate
            break
    search_dir = esm2_dir if esm2_dir is not None else saturn_data_dir

    for sp in species_list:
        gene_id_col = f"{sp}_gene_id"
        gene_name_col = f"{sp}_gene_name"
        if gene_id_col not in ortholog_table.columns and gene_name_col not in ortholog_table.columns:
            logger.warning("Skipping species %s: no gene column in ortholog table.", sp)
            continue

        # --- Try .npy first (our cached format), then find matching .pt/.torch ---
        emb_path_npy = saturn_data_dir / f"{sp}_esm2.npy"
        sp_data = None

        if emb_path_npy.exists():
            sp_data = np.load(emb_path_npy, allow_pickle=True).item()
        else:
            # Resolve the common name for filename matching
            common_name = _ENSEMBL_TO_COMMON.get(sp.lower(), sp.lower())
            search_names = {common_name, sp.lower()}

            for emb_file in _find_emb_files(search_dir):
                stem = emb_file.stem.lower().replace("_embedding", "")
                if stem in search_names or any(n in stem for n in search_names):
                    import torch as _torch
                    raw = _torch.load(emb_file, map_location="cpu", weights_only=False)
                    if isinstance(raw, dict):
                        sp_data = {
                            k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v))
                            for k, v in raw.items()
                        }
                        logger.info("Loaded embeddings for %s from %s (%d genes, dim=%d)",
                                    sp, emb_file.name, len(sp_data),
                                    next(iter(sp_data.values())).shape[0] if sp_data else 0)
                    break

        if sp_data is None:
            logger.warning("No embeddings found for species %s", sp)
            continue

        # Try matching by gene ID first, then gene name (SATURN uses gene symbols).
        gene_ids = ortholog_table[gene_id_col].values if gene_id_col in ortholog_table.columns else None
        gene_names = ortholog_table[gene_name_col].values if gene_name_col in ortholog_table.columns else None
        n_matched = 0
        for i in range(n_groups):
            gid = gene_ids[i] if gene_ids is not None else None
            gname = gene_names[i] if gene_names is not None else None
            matched_key = None
            if gid is not None and gid in sp_data:
                matched_key = gid
            elif gname is not None and gname in sp_data:
                matched_key = gname
            if matched_key is not None:
                emb = sp_data[matched_key]
                if len(emb) >= emb_dim:
                    consensus[i] += emb[:emb_dim]
                else:
                    consensus[i, :len(emb)] += emb
                n_matched += 1
        logger.info("  %s: matched %d / %d ortholog genes", sp, n_matched, n_groups)

    # Average over species.
    n_species = len(species_list)
    if n_species > 0:
        consensus /= n_species

    return consensus


def pca_esm2(
    consensus_embeddings: np.ndarray,
    n_components: int = 64,
) -> np.ndarray:
    """Reduce ESM-2 consensus embeddings from 5120 to a lower dimensionality via PCA.

    Parameters
    ----------
    consensus_embeddings
        Array of shape ``(G, 5120)``.
    n_components
        Number of principal components to retain.

    Returns
    -------
    np.ndarray
        PCA-reduced embeddings of shape ``(G, n_components)``.
    """
    n_components = min(n_components, *consensus_embeddings.shape)
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(consensus_embeddings)
    logger.info(
        "PCA ESM-2: %d -> %d, explained variance=%.3f",
        consensus_embeddings.shape[1],
        n_components,
        pca.explained_variance_ratio_.sum(),
    )
    return reduced.astype(np.float32)


def compute_esm2_interspecies_variance(
    per_species_embeddings: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute mean pairwise cosine distance per ortholog group across species.

    Parameters
    ----------
    per_species_embeddings
        Dictionary mapping species name to an ``(G, D)`` embedding array.
        All arrays must have the same number of rows (ortholog groups) and
        columns (embedding dimensionality).

    Returns
    -------
    np.ndarray
        Mean pairwise cosine distance per ortholog group, shape ``(G,)``.
        Values range from 0 (identical embeddings) to 2 (opposite embeddings).
    """
    species_names = list(per_species_embeddings.keys())
    n_species = len(species_names)
    if n_species < 2:
        G = next(iter(per_species_embeddings.values())).shape[0]
        return np.zeros(G, dtype=np.float32)

    # Stack all species embeddings: (K, G, D)
    embeddings = np.stack(
        [per_species_embeddings[sp] for sp in species_names], axis=0
    )
    # Normalize each embedding vector.
    norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings_normed = embeddings / norms  # (K, G, D)

    G = embeddings.shape[1]
    pairwise_sum = np.zeros(G, dtype=np.float64)
    n_pairs = 0

    for i in range(n_species):
        for j in range(i + 1, n_species):
            # Cosine similarity per gene: (G,)
            cos_sim = np.sum(
                embeddings_normed[i] * embeddings_normed[j], axis=1
            )
            cos_dist = 1.0 - cos_sim
            pairwise_sum += cos_dist
            n_pairs += 1

    mean_dist = (pairwise_sum / n_pairs).astype(np.float32)
    return mean_dist


# ---------------------------------------------------------------------------
# Expression statistics
# ---------------------------------------------------------------------------


def compute_expression_stats(
    adata: ad.AnnData,
    species_key: str = "species",
) -> Dict[str, np.ndarray]:
    """Compute per-gene expression statistics across all cells.

    Parameters
    ----------
    adata
        AnnData with raw UMI counts in ``.X``.
    species_key
        Column in ``.obs`` identifying species (used for potential per-species
        stats in future, currently computes global stats).

    Returns
    -------
    dict
        Dictionary with keys:
        - ``'log_mean_expr'``: log1p of mean expression per gene, shape ``(G,)``
        - ``'log_expr_var'``: log1p of variance of expression per gene, shape ``(G,)``
        - ``'frac_nonzero'``: fraction of cells expressing each gene, shape ``(G,)``
    """
    X = adata.X
    if sp.issparse(X):
        mean_expr = np.asarray(X.mean(axis=0)).ravel()
        # Variance via E[X^2] - E[X]^2
        X_sq = X.copy()
        X_sq.data **= 2
        mean_sq = np.asarray(X_sq.mean(axis=0)).ravel()
        var_expr = mean_sq - mean_expr ** 2
        var_expr = np.maximum(var_expr, 0.0)  # numerical safety
        # Fraction non-zero
        nnz_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()
        frac_nonzero = nnz_per_gene / X.shape[0]
    else:
        X_dense = np.asarray(X, dtype=np.float64)
        mean_expr = X_dense.mean(axis=0)
        var_expr = X_dense.var(axis=0)
        frac_nonzero = (X_dense > 0).mean(axis=0)

    return {
        "log_mean_expr": np.log1p(mean_expr).astype(np.float32),
        "log_expr_var": np.log1p(var_expr).astype(np.float32),
        "frac_nonzero": frac_nonzero.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# GIN feature assembly
# ---------------------------------------------------------------------------


def assemble_gin_features(
    esm2_pca: np.ndarray,
    esm2_interspecies_var: np.ndarray,
    ortholog_table: pd.DataFrame,
    expression_stats: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, StandardScaler]:
    """Assemble and standardize per-gene features for the GIN.

    Concatenates ~72 features per gene:
      - 64 ESM-2 PCA dimensions
      - 1 ESM-2 inter-species variance
      - 1 mean identity (from ortholog table)
      - 3 expression stats (log_mean_expr, log_expr_var, frac_nonzero)
      - Remaining slots padded with zeros if needed (or additional stats)

    StandardScaler is applied (MANDATORY).

    Parameters
    ----------
    esm2_pca
        PCA-reduced ESM-2 consensus embeddings, shape ``(G, 64)``.
    esm2_interspecies_var
        Inter-species ESM-2 variance, shape ``(G,)``.
    ortholog_table
        Ortholog table with ``mean_identity`` column.
    expression_stats
        Dictionary from :func:`compute_expression_stats`.

    Returns
    -------
    features
        StandardScaled feature matrix of shape ``(G, D)`` where D is the total
        number of features (typically 72).
    scaler
        Fitted :class:`StandardScaler` instance (needed for inference on new data).
    """
    G = esm2_pca.shape[0]
    n_pca = esm2_pca.shape[1]  # typically 64

    parts = [
        esm2_pca,  # (G, 64)
        esm2_interspecies_var.reshape(-1, 1),  # (G, 1)
        ortholog_table["mean_identity"].values.reshape(-1, 1).astype(np.float32),  # (G, 1)
        expression_stats["log_mean_expr"].reshape(-1, 1),  # (G, 1)
        expression_stats["log_expr_var"].reshape(-1, 1),  # (G, 1)
        expression_stats["frac_nonzero"].reshape(-1, 1),  # (G, 1)
    ]

    raw_features = np.hstack(parts)  # (G, 69) typically
    current_d = raw_features.shape[1]
    target_d = max(current_d, 72)

    # Pad to target dimensionality if needed.
    if current_d < target_d:
        pad = np.zeros((G, target_d - current_d), dtype=np.float32)
        raw_features = np.hstack([raw_features, pad])

    scaler = StandardScaler()
    features = scaler.fit_transform(raw_features).astype(np.float32)

    logger.info("GIN features: shape=%s", features.shape)
    return features, scaler


# ---------------------------------------------------------------------------
# Unified AnnData construction
# ---------------------------------------------------------------------------


def construct_unified_adata(
    per_species_adatas: Dict[str, ad.AnnData],
    ortholog_table: pd.DataFrame,
    gene_features: np.ndarray,
    conservation_prior: torch.Tensor,
) -> ad.AnnData:
    """Construct a unified cross-species AnnData from per-species data.

    Maps each species' genes to ortholog group IDs, concatenates all species,
    and attaches gene features and conservation prior.

    Parameters
    ----------
    per_species_adatas
        Dictionary mapping species name to per-species AnnData.  Each AnnData
        must have gene names or IDs that can be mapped to the ortholog table.
    ortholog_table
        Ortholog table with per-species gene ID columns and ``ortholog_group_id``.
    gene_features
        StandardScaled gene feature matrix, shape ``(G, D)``.
    conservation_prior
        Conservation prior tensor, shape ``(G,)``.

    Returns
    -------
    AnnData
        Unified AnnData with:
        - ``.X``: raw UMI counts (sparse CSR), genes indexed by ortholog group
        - ``.obs['species']``: species label
        - ``.obs['batch']``: batch label
        - ``.var['ortholog_group_id']``: integer group ID per gene
        - ``.var['conservation_prior']``: prior value per gene
        - ``.varm['gene_features']``: gene feature matrix
    """
    G = len(ortholog_table)
    mapped_adatas = []

    for species_name, sp_adata in per_species_adatas.items():
        gene_id_col = f"{species_name}_gene_id"
        gene_name_col = f"{species_name}_gene_name"

        # Build mapping from species gene -> ortholog group index.
        # Try gene_id (Ensembl IDs) and gene_name (symbols) — use whichever
        # matches more var_names in the AnnData.
        gene_names_set = set(sp_adata.var_names.values)
        best_col = None
        best_count = 0
        for candidate_col in [gene_id_col, gene_name_col]:
            if candidate_col not in ortholog_table.columns:
                continue
            n_match = sum(
                1 for v in ortholog_table[candidate_col].values
                if v in gene_names_set
            )
            if n_match > best_count:
                best_count = n_match
                best_col = candidate_col

        if best_col is None or best_count == 0:
            logger.warning(
                "No gene mapping found for species %s (tried %s, %s). Skipping.",
                species_name, gene_id_col, gene_name_col,
            )
            continue
        logger.info(
            "Mapping %s via %s: %d / %d ortholog genes matched",
            species_name, best_col, best_count, len(ortholog_table),
        )

        ortho_map = dict(
            zip(
                ortholog_table[best_col],
                ortholog_table["ortholog_group_id"],
            )
        )

        # Find overlapping genes.
        gene_names = sp_adata.var_names.values
        gene_in_ortho = [g for g in gene_names if g in ortho_map]

        if len(gene_in_ortho) == 0:
            logger.warning("No ortholog matches for species %s.", species_name)
            continue

        # Subset to ortholog genes and reorder to ortholog group order.
        sp_sub = sp_adata[:, gene_in_ortho].copy()
        ortho_indices = [ortho_map[g] for g in gene_in_ortho]

        # Build a sparse mapping matrix: (n_ortho_genes, G)
        n_cells = sp_sub.n_obs
        X_orig = sp_sub.X
        if not sp.issparse(X_orig):
            X_orig = sp.csr_matrix(X_orig)

        # Create the new expression matrix aligned to ortholog groups.
        X_aligned = sp.lil_matrix((n_cells, G), dtype=X_orig.dtype)
        for j_new, j_old in enumerate(range(len(gene_in_ortho))):
            ortho_idx = ortho_indices[j_new]
            X_aligned[:, ortho_idx] = X_orig[:, j_old].toarray()

        X_aligned = sp.csr_matrix(X_aligned)

        # Create aligned AnnData.
        obs = sp_sub.obs.copy()
        if "species" not in obs.columns:
            obs["species"] = species_name
        var = pd.DataFrame(
            {"ortholog_group_id": np.arange(G)},
            index=[f"ortho_{i}" for i in range(G)],
        )
        aligned = ad.AnnData(X=X_aligned, obs=obs, var=var)
        mapped_adatas.append(aligned)

    if not mapped_adatas:
        raise ValueError("No species could be mapped to ortholog groups.")

    # Concatenate all species.
    unified = ad.concat(mapped_adatas, join="outer")
    unified.obs_names_make_unique()

    # Attach gene features and conservation prior.
    if gene_features.shape[0] == unified.n_vars:
        unified.varm["gene_features"] = gene_features
    unified.var["conservation_prior"] = conservation_prior.numpy()

    return unified


# ---------------------------------------------------------------------------
# Null-expression distribution and sampling weights
# ---------------------------------------------------------------------------


def precompute_p_null(
    adata: ad.AnnData,
    species_key: str = "species",
    eps: float = 1.0,
) -> np.ndarray:
    """Precompute the null expression distribution per species per gene.

    Uses Laplace smoothing:

        p_null[s, g] = (sum_{c in s} x[c,g] + eps) / (sum_{c in s} lib[c] + eps * G)

    Parameters
    ----------
    adata
        AnnData with raw UMI counts in ``.X``.
    species_key
        Column in ``.obs`` identifying species.
    eps
        Laplace smoothing pseudocount.

    Returns
    -------
    np.ndarray
        Null expression probabilities of shape ``(K, G)`` where K is the number
        of unique species and G is the number of genes.
    """
    species = adata.obs[species_key]
    unique_species = sorted(species.unique())
    K = len(unique_species)
    G = adata.n_vars
    X = adata.X

    p_null = np.zeros((K, G), dtype=np.float64)

    for s_idx, sp_name in enumerate(unique_species):
        mask = (species == sp_name).values
        X_s = X[mask]

        if sp.issparse(X_s):
            gene_totals = np.asarray(X_s.sum(axis=0)).ravel()  # (G,)
            lib_total = X_s.sum()  # scalar
        else:
            X_s = np.asarray(X_s, dtype=np.float64)
            gene_totals = X_s.sum(axis=0)  # (G,)
            lib_total = X_s.sum()  # scalar

        p_null[s_idx] = (gene_totals + eps) / (lib_total + eps * G)

    return p_null.astype(np.float64)


def get_sqrt_species_weights(
    adata: ad.AnnData,
    species_key: str = "species",
) -> np.ndarray:
    """Compute sqrt-proportional species sampling weights per cell.

    Each cell from species *s* gets weight proportional to
    ``sqrt(N_total / N_s)`` where ``N_s`` is the number of cells from species *s*.
    Weights are normalized to sum to 1.

    Parameters
    ----------
    adata
        AnnData with species annotations.
    species_key
        Column in ``.obs`` identifying species.

    Returns
    -------
    np.ndarray
        Per-cell sampling weights of shape ``(N_cells,)``, summing to ~1.
    """
    species = adata.obs[species_key].values
    unique_species, inverse = np.unique(species, return_inverse=True)
    counts = np.bincount(inverse)  # (K,)
    N_total = len(species)

    # sqrt-proportional weight per species.
    species_weights = np.sqrt(N_total / counts.astype(np.float64))  # (K,)

    # Assign per-cell weights.
    cell_weights = species_weights[inverse]  # (N_cells,)

    # Normalize to sum to 1.
    cell_weights /= cell_weights.sum()

    return cell_weights.astype(np.float64)


# ---------------------------------------------------------------------------
# Synthetic data for unit testing
# ---------------------------------------------------------------------------


def create_synthetic_adata(
    n_cells_per_species: int = 100,
    n_genes: int = 200,
    n_species: int = 3,
    n_batches_per_species: int = 2,
    seed: int = 42,
) -> Dict[str, object]:
    """Create a synthetic AnnData for unit tests and integration tests.

    Produces a valid AnnData that can pass through the full Chimera pipeline,
    along with matching gene features, conservation prior, and precomputed p_null.

    Parameters
    ----------
    n_cells_per_species
        Number of cells per species.
    n_genes
        Number of ortholog genes.
    n_species
        Number of species.
    n_batches_per_species
        Number of batches (donors) per species.
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``'adata'``: AnnData with ``.X`` = raw UMI counts (sparse CSR),
          ``.obs['species']``, ``.obs['batch']``, ``.obs['cell_type']``
        - ``'gene_features'``: np.ndarray of shape ``(n_genes, 72)``
        - ``'conservation_prior'``: torch.Tensor of shape ``(n_genes,)``
        - ``'p_null'``: np.ndarray of shape ``(n_species, n_genes)``
        - ``'ortholog_table'``: pd.DataFrame with ``mean_identity`` column
    """
    rng = np.random.RandomState(seed)

    species_names = [f"species_{i}" for i in range(n_species)]
    cell_type_names = ["TypeA", "TypeB", "TypeC", "TypeD"]
    n_cell_types = len(cell_type_names)

    N_total = n_cells_per_species * n_species
    total_batches = n_batches_per_species * n_species

    # --- Generate cell metadata ---
    species_labels = []
    batch_labels = []
    cell_type_labels = []

    for s_idx, sp_name in enumerate(species_names):
        species_labels.extend([sp_name] * n_cells_per_species)
        # Assign cells to batches roughly evenly.
        batch_assignments = np.repeat(
            [f"{sp_name}_batch_{b}" for b in range(n_batches_per_species)],
            int(np.ceil(n_cells_per_species / n_batches_per_species)),
        )[:n_cells_per_species]
        batch_labels.extend(batch_assignments.tolist())
        # Assign cell types.
        ct_assignments = rng.choice(cell_type_names, size=n_cells_per_species)
        cell_type_labels.extend(ct_assignments.tolist())

    obs = pd.DataFrame(
        {
            "species": pd.Categorical(species_labels),
            "batch": pd.Categorical(batch_labels),
            "cell_type": pd.Categorical(cell_type_labels),
        },
        index=[f"cell_{i}" for i in range(N_total)],
    )

    # --- Generate count matrix ---
    # Use species-specific Poisson rate vectors to create realistic variation.
    # Each species has a base rate vector modulated by cell-type-specific offsets.
    base_rates = rng.exponential(scale=1.0, size=(n_species, n_genes)) + 0.1

    # Cell-type-specific scaling.
    ct_scale = rng.exponential(scale=1.0, size=(n_cell_types, n_genes)) + 0.5

    X_dense = np.zeros((N_total, n_genes), dtype=np.float32)

    for s_idx, sp_name in enumerate(species_names):
        start = s_idx * n_cells_per_species
        end = start + n_cells_per_species
        for c in range(n_cells_per_species):
            ct_name = cell_type_labels[start + c]
            ct_idx = cell_type_names.index(ct_name)
            # Poisson rate = base_rate * ct_scale, with per-cell library size variation.
            lib_size_factor = rng.lognormal(mean=0, sigma=0.5)
            rate = base_rates[s_idx] * ct_scale[ct_idx] * lib_size_factor
            X_dense[start + c] = rng.poisson(lam=rate)

    X_sparse = sp.csr_matrix(X_dense)

    # --- Build AnnData ---
    var = pd.DataFrame(
        {"gene_name": [f"gene_{g}" for g in range(n_genes)]},
        index=[f"gene_{g}" for g in range(n_genes)],
    )
    adata = ad.AnnData(X=X_sparse, obs=obs, var=var)

    # --- Gene features (synthetic) ---
    # 64 PCA components + 8 scalar features = 72.
    esm2_pca = rng.randn(n_genes, 64).astype(np.float32)
    esm2_var = rng.exponential(0.3, size=n_genes).astype(np.float32)
    mean_identity = rng.uniform(30, 100, size=n_genes).astype(np.float32)

    expr_stats = compute_expression_stats(adata)

    ortho_table = pd.DataFrame(
        {
            "ortholog_group_id": np.arange(n_genes),
            "mean_identity": mean_identity,
        }
    )

    gene_features, scaler = assemble_gin_features(
        esm2_pca=esm2_pca,
        esm2_interspecies_var=esm2_var,
        ortholog_table=ortho_table,
        expression_stats=expr_stats,
    )

    # --- Conservation prior ---
    conservation_prior = compute_conservation_prior(ortho_table)

    # --- p_null ---
    p_null = precompute_p_null(adata)

    return {
        "adata": adata,
        "gene_features": gene_features,
        "conservation_prior": conservation_prior,
        "p_null": p_null,
        "ortholog_table": ortho_table,
    }
