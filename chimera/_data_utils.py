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


def download_van_zyl_raw(
    data_dir: Union[str, Path, None] = None,
) -> ad.AnnData:
    """Download the *raw* van Zyl AH atlas (5 species, per-species Ensembl genes).

    This is a faithful re-implementation of SATURN's
    ``Vignettes/ah_atlas/ah_atlas_data_loader.ipynb``
    (https://github.com/snap-stanford/SATURN/blob/main/Vignettes/ah_atlas/ah_atlas_data_loader.ipynb)
    with three additions needed for Chimera's cross-species pipeline:

    1. Adds ``obs['species']`` (SATURN's loader keeps species implicit, one file
       per species). Labels match SATURN's preprocessed export exactly:
       ``{'human', 'mouse', 'pig', 'macaca_fascicularis', 'macaca_mulatta'}``.
    2. Adds ``obs['batch']`` derived from the cell-barcode prefix before the
       first ``_`` (e.g. ``H1TMS1`` for human sample 1, ``MM1LTM`` for macaque
       mulatta sample 1). Each batch is nested within exactly one species.
    3. Concatenates the 5 per-species AnnDatas with an outer gene join, so the
       returned object is a single cached file (``data/raw/van_zyl_raw.h5ad``).

    The pipeline otherwise mirrors SATURN cell-for-cell:
      - Download 5 per-species gzipped CSVs from NCBI GEO (public, no auth).
      - ``gunzip`` in-memory (skip if the decompressed ``.csv`` exists).
      - ``sc.read(csv).T`` to flip the genes × cells layout to cells × genes.
      - Align ``obs`` to the SATURN-curated metafile
        ``all_five_species_metafile.csv`` (downloaded from SATURN's repo);
        drop the SCP ``TYPE,group`` row at index 0.
      - ``obs['cell_type'] = obs['Cluster']``.

    Parameters
    ----------
    data_dir
        Directory to store raw data. Defaults to ``<repo>/data/raw``. Creates a
        ``van_zyl_raw/`` subdir with per-species ``.csv.gz`` / ``.csv`` and the
        metafile, plus a top-level ``van_zyl_raw.h5ad`` cache.

    Returns
    -------
    AnnData
        Concatenated 5-species AnnData:
          - ``.X``: raw UMI counts (sparse CSR, float32). Integer-valued.
          - ``.obs['species']``: one of the 5 lowercase labels above.
          - ``.obs['batch']``: barcode prefix (sample-level batch).
          - ``.obs['cell_type']``: SATURN's cross-species cluster label.
          - ``.obs['Cluster']``: retained for provenance / SATURN parity.
          - ``.var_names``: union of per-species gene symbols (outer join).

    Notes
    -----
    Unlike ``download_camex_*``, this function does **not** apply the Chimera
    QC filter — SATURN's loader keeps every cell in every CSV and we preserve
    that behavior so the cross-check against the cached SATURN export
    (``data/raw/van_zyl_aqueous_humor.h5ad``) can verify cell-for-cell parity.
    Apply ``_qc_filter`` downstream if you want stricter cells.
    """
    data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR / "raw"
    _ensure_dir(data_dir)
    cache_path = data_dir / "van_zyl_raw.h5ad"

    if cache_path.exists():
        logger.info("Loading cached van Zyl raw from %s", cache_path)
        return ad.read_h5ad(cache_path)

    # --- Per-species sources (species_label, GEO URL) in SATURN's order. ---
    # species_label values match the SATURN preprocessed export on disk so
    # cross-check is direct: human, mouse, pig, macaca_fascicularis,
    # macaca_mulatta. "MacaF" = Macaca Fascicularis, "MacaqueM" = Macaca
    # Mulatta (first-letter-of-species shorthand, not female/male).
    _VZ_SOURCES: list[tuple[str, str]] = [
        ("mouse",
         "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE146nnn/GSE146186/suppl/GSE146186_Mouse_count_matrix.csv.gz"),
        ("pig",
         "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE146nnn/GSE146187/suppl/GSE146187_Pig_count_matrix.csv.gz"),
        ("human",
         "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE148nnn/GSE148371/suppl/GSE148371_Human_count_matrix.csv.gz"),
        ("macaca_fascicularis",
         "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE148nnn/GSE148373/suppl/GSE148373_MacaF_count_matrix.csv.gz"),
        ("macaca_mulatta",
         "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE148nnn/GSE148374/suppl/GSE148374_MacaqueM_count_matrix.csv.gz"),
    ]
    _VZ_META_URL = (
        "https://raw.githubusercontent.com/snap-stanford/SATURN/main/"
        "Vignettes/ah_atlas/data/all_five_species_metafile.csv"
    )

    vz_dir = data_dir / "van_zyl_raw"
    _ensure_dir(vz_dir)

    # --- 1. Fetch each .csv.gz, then decompress to .csv (skip-if-exists). ---
    import gzip, shutil
    for sp_label, url in _VZ_SOURCES:
        gz_name = url.rsplit("/", 1)[-1]
        csv_name = gz_name[:-3]  # strip ".gz"
        gz_path = vz_dir / gz_name
        csv_path = vz_dir / csv_name
        if csv_path.exists():
            logger.info("Cached (decompressed): %s", csv_path.name)
            continue
        _download_url(url, gz_path, desc=f"van Zyl {sp_label}")
        logger.info("Decompressing %s → %s", gz_path.name, csv_path.name)
        # Decompress via tmp path + atomic rename so a killed process leaves no
        # partial .csv that the next run would mistake for a complete cache.
        tmp_path = csv_path.with_suffix(csv_path.suffix + ".part")
        try:
            with gzip.open(gz_path, "rb") as fin, open(tmp_path, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
            tmp_path.rename(csv_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    # --- 2. Metadata file from SATURN's repo ("NAME,Cluster" + SCP TYPE row). ---
    meta_path = vz_dir / "all_five_species_metafile.csv"
    if not meta_path.exists():
        _download_url(_VZ_META_URL, meta_path, desc="van Zyl SATURN metafile")

    meta = pd.read_csv(meta_path)
    # SCP's convention is that row 0 after the header is a TYPE,group
    # annotation row ("TYPE,group,group,..."). Drop it exactly as SATURN's
    # loader does with `meta.drop(0)`.
    meta = meta.drop(0)
    meta = meta.set_index("NAME")
    logger.info("van Zyl metafile: %d cells annotated", len(meta))

    # --- 3. Per-species: sc.read + .T + obs attachment (SATURN parity). ---
    import scanpy as sc
    from scipy.sparse import issparse, csr_matrix

    per_species_adatas: list[ad.AnnData] = []
    for sp_label, url in _VZ_SOURCES:
        gz_name = url.rsplit("/", 1)[-1]
        csv_path = vz_dir / gz_name[:-3]
        logger.info("Loading %s from %s", sp_label, csv_path.name)

        # `sc.read` on CSV returns a dense matrix of genes × cells; .T flips
        # to cells × genes. This matches SATURN's loader exactly.
        a = sc.read(csv_path).T

        # Align to metafile barcodes. SATURN's loader uses `.loc[obs_names]`
        # and assumes every barcode has metadata; we intersect defensively.
        overlap = a.obs_names.intersection(meta.index)
        if len(overlap) < a.n_obs:
            missing = a.n_obs - len(overlap)
            logger.warning("  %s: %d cells missing from metafile — dropping", sp_label, missing)
            a = a[a.obs_names.isin(overlap)].copy()

        # SATURN: `pig_ad.obs = meta.set_index("NAME").loc[pig_ad.obs_names]`
        a.obs = meta.loc[a.obs_names].copy()
        # SATURN: `pig_ad.obs["cell_type"] = pig_ad.obs["Cluster"]`
        # Then SATURN's downstream integration pipeline collapses subcluster
        # labels by stripping a leading "<digit>_" prefix ("1_Uveal" → "Uveal",
        # "3_Macrophage" → "Macrophage"). This step is not in the data loader
        # notebook but is the exact difference between the metafile's
        # `Cluster` column and the `cell_type` column of the SATURN-cached
        # h5ad on disk (verified: applying `re.sub(r'^\d+_', '', Cluster)`
        # yields 100% equality with SATURN's cached labels across all 50,112
        # cells). We apply it here so `cell_type` matches SATURN's exactly
        # while preserving the finer-grained labels in `Cluster` for
        # provenance / downstream sub-annotation use.
        import re as _re
        _prefix = _re.compile(r"^\d+_")
        a.obs["cell_type"] = a.obs["Cluster"].astype(str).map(
            lambda s: _prefix.sub("", s)
        )

        # Chimera additions: species + batch.
        a.obs["species"] = sp_label
        # Batch = barcode prefix before the first underscore (sample-level).
        a.obs["batch"] = a.obs_names.to_series().str.split("_").str[0].values

        # Convert dense float to sparse CSR int-like to save memory and match
        # the sparse storage used by every other raw cache.
        if not issparse(a.X):
            a.X = csr_matrix(a.X.astype(np.float32, copy=False))

        _validate_raw_counts(a, f"van Zyl {sp_label}")

        logger.info("  %s: %d cells × %d genes (X is sparse CSR, raw int counts)",
                    sp_label, a.n_obs, a.n_vars)
        per_species_adatas.append(a)

    # --- 4. Concatenate with outer gene join. ---
    adata = ad.concat(
        per_species_adatas,
        join="outer",
        merge="unique",
        label=None,
        index_unique=None,
    )
    adata.obs_names_make_unique()

    # Categorical dtypes (matches pancreas / testis convention).
    for col in ("species", "batch", "cell_type", "Cluster"):
        if col in adata.obs.columns:
            adata.obs[col] = pd.Categorical(adata.obs[col].astype(str))

    adata.write_h5ad(cache_path)
    logger.info(
        "Saved van Zyl raw: %d cells × %d genes, %d species → %s",
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
    # Bug-fix (2026-04-23): the pre-fix check used `any(camex_dir.rglob("*.h5ad"))`
    # which was True whenever the sibling liver tissue had already been pulled, so
    # the gdown call never ran and we would FileNotFoundError at the locate step
    # below. We now check for testis-specific h5ads. `gdown.download_folder` with
    # `resume=True` is a no-op for files already on disk, so re-invoking it here
    # only downloads what is missing (the testis h5ads).
    camex_dir = data_dir / "camex"
    def _testis_h5ads_present() -> bool:
        if not camex_dir.exists():
            return False
        for pattern in ("*testis*", "*Testis*", "2testis"):
            for d in camex_dir.rglob(pattern):
                if d.is_dir() and any(d.rglob("*.h5ad")):
                    return True
        return False
    if not _testis_h5ads_present():
        # gdown walks the whole root folder and can rate-limit mid-walk
        # (observed 2026-04-23: all 11 testis h5ads completed, then failure
        # on Margarida's bulk files with "Cannot retrieve the public link …
        # may have had many accesses"). If the walk dies AFTER the testis
        # files landed, that is success for our purposes — re-check and only
        # propagate the error if testis is still missing.
        try:
            _download_gdrive_folder("1rwdjEvWFEFw82a0x2JzMi2jXICbUc5eb", camex_dir)
        except Exception as e:
            if _testis_h5ads_present():
                logger.warning(
                    "gdown raised %s, but testis h5ads are on disk — proceeding.",
                    type(e).__name__,
                )
            else:
                raise

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
    cache_path: Union[str, Path, None] = None,
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
        Defaults to ``<repo>/data/processed``. Ignored if *cache_path* is given.
    cache_path
        Explicit path to the cache CSV. When provided, overrides *cache_dir*
        so multiple datasets can keep their own ortholog tables side-by-side
        (e.g. ``ortholog_table.csv`` for pancreas and
        ``ortholog_table_testis.csv`` for testis) without clobbering.

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
    if cache_path is not None:
        cache_path = Path(cache_path)
        _ensure_dir(cache_path.parent)
    else:
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
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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
    consensus : np.ndarray
        Consensus embeddings of shape ``(G, 5120)`` where G is the number of
        ortholog groups. Rows for unmatched ortholog groups are all-zero.
    per_species_matrices : Dict[str, np.ndarray]
        Mapping from species name (as passed in ``species_list``) to its per-
        ortholog ESM-2 matrix of shape ``(G, 5120)``. Rows where the species
        had no match in the ortholog table are all-zero (used downstream as a
        "missing" mask by :func:`compute_esm2_interspecies_variance`). Species
        whose ortholog columns are absent are omitted from this dict.
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
    per_species_matrices: Dict[str, np.ndarray] = {}

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
        sp_matrix = np.zeros((n_groups, emb_dim), dtype=np.float32)
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
                vec = np.zeros(emb_dim, dtype=np.float32)
                if len(emb) >= emb_dim:
                    vec[:] = emb[:emb_dim]
                else:
                    vec[:len(emb)] = emb
                consensus[i] += vec
                sp_matrix[i] = vec
                n_matched += 1
        per_species_matrices[sp] = sp_matrix
        logger.info("  %s: matched %d / %d ortholog genes", sp, n_matched, n_groups)

    # Average over species.
    n_species = len(species_list)
    if n_species > 0:
        consensus /= n_species

    return consensus, per_species_matrices


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

    Rows where a species has no embedding (all-zero vector) are treated as
    missing: the pair is excluded from that gene's average. Genes with no
    informative pairs (neither species matched the ortholog) are filled with
    the median variance so they do not become outliers after
    :class:`~sklearn.preprocessing.StandardScaler` is applied downstream.

    Parameters
    ----------
    per_species_embeddings
        Dictionary mapping species name to an ``(G, D)`` embedding array.
        All arrays must have the same number of rows (ortholog groups) and
        columns (embedding dimensionality). Rows with all-zero values are
        interpreted as "species had no embedding for this ortholog".

    Returns
    -------
    np.ndarray
        Mean pairwise cosine distance per ortholog group, shape ``(G,)``.
        Values range from 0 (identical embeddings) to 2 (opposite embeddings).

    Raises
    ------
    ValueError
        If fewer than 2 species are provided. Inter-species variance is
        undefined for 0 or 1 species; raising loudly prevents silently
        returning an all-zero feature that later contaminates training.
    """
    species_names = list(per_species_embeddings.keys())
    n_species = len(species_names)
    if n_species < 2:
        raise ValueError(
            f"compute_esm2_interspecies_variance requires at least 2 species, "
            f"got {n_species} ({species_names}). Check that species_list was "
            f"mapped to valid Ensembl identifiers and that the ortholog table "
            f"contains matching '{{species}}_gene_id' / '{{species}}_gene_name' "
            f"columns."
        )

    # Per-species mask: True where the gene has a non-zero embedding.
    masks = {sp: np.any(per_species_embeddings[sp] != 0, axis=1) for sp in species_names}

    # Unit-normalize each row (floor tiny norms to avoid div-by-zero on all-zero rows).
    normed: Dict[str, np.ndarray] = {}
    for sp in species_names:
        mat = per_species_embeddings[sp]
        norm = np.linalg.norm(mat, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        normed[sp] = (mat / norm).astype(np.float32)

    G = normed[species_names[0]].shape[0]
    pair_sum = np.zeros(G, dtype=np.float64)
    pair_count = np.zeros(G, dtype=np.int32)

    for i in range(n_species):
        for j in range(i + 1, n_species):
            s1, s2 = species_names[i], species_names[j]
            both = masks[s1] & masks[s2]
            if not both.any():
                continue
            cos_sim = np.einsum("ij,ij->i", normed[s1], normed[s2])
            cos_dist = 1.0 - cos_sim
            pair_sum[both] += cos_dist[both]
            pair_count[both] += 1

    mean_dist = np.zeros(G, dtype=np.float32)
    has_signal = pair_count > 0
    mean_dist[has_signal] = (pair_sum[has_signal] / pair_count[has_signal]).astype(np.float32)
    # Impute unmatched genes with the median so they don't become outliers after
    # StandardScaler. Matches the behavior of training/recompute_esm2_variance.py.
    if has_signal.any() and (~has_signal).any():
        mean_dist[~has_signal] = float(np.median(mean_dist[has_signal]))
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

    Concatenates 72 features per gene:
      - 64 ESM-2 PCA dimensions
      - 1 ESM-2 inter-species variance
      - 1 mean identity (from ortholog table)
      - 3 expression stats (log_mean_expr, log_expr_var, frac_nonzero)
      - 3 identity spread stats (std, max, min across species pairs)

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

    # Per-species identity columns (e.g. mmulatta_identity, mmusculus_identity, ...)
    # provide additional signal beyond mean_identity: the spread of pairwise
    # identities across species matters for multi-species settings.
    identity_cols = [c for c in ortholog_table.columns if c.endswith("_identity") and c != "mean_identity"]
    if identity_cols:
        id_matrix = ortholog_table[identity_cols].values.astype(np.float32)
        std_identity = np.nanstd(id_matrix, axis=1)
        max_identity = np.nanmax(id_matrix, axis=1)
        min_identity = np.nanmin(id_matrix, axis=1)
    else:
        std_identity = np.zeros(G, dtype=np.float32)
        max_identity = ortholog_table["mean_identity"].values.astype(np.float32)
        min_identity = ortholog_table["mean_identity"].values.astype(np.float32)

    parts = [
        esm2_pca,  # (G, 64)
        esm2_interspecies_var.reshape(-1, 1),  # (G, 1)
        ortholog_table["mean_identity"].values.reshape(-1, 1).astype(np.float32),  # (G, 1)
        expression_stats["log_mean_expr"].reshape(-1, 1),  # (G, 1)
        expression_stats["log_expr_var"].reshape(-1, 1),  # (G, 1)
        expression_stats["frac_nonzero"].reshape(-1, 1),  # (G, 1)
        std_identity.reshape(-1, 1),  # (G, 1) — replaces zero padding
        max_identity.reshape(-1, 1),  # (G, 1) — replaces zero padding
        min_identity.reshape(-1, 1),  # (G, 1) — replaces zero padding
    ]

    raw_features = np.hstack(parts)  # (G, 72) typically
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


def get_equal_species_weights(
    adata: ad.AnnData,
    species_key: str = "species",
) -> np.ndarray:
    """Per-cell weights giving each species equal total sampling probability.

    Each of ``S`` species contributes ``1/S`` of the total sampling probability,
    independent of cell count. A cell from species *s* gets per-cell weight
    ``(1/S) / N_s``. Weights are normalized to sum to 1.

    In a mini-batch of size ``B`` drawn with replacement, the expected number
    of cells from each species is ``B/S`` with std ``sqrt(B (1/S)(1 - 1/S))``.
    For ``B=128, S=2`` the std is ~5.65 — much tighter than sqrt-proportional
    sampling on skewed datasets (~1:2) which produces ~42/58 expected splits.
    This tightens the per-batch species distribution seen by MMD and
    reduces seed-to-seed variance in the MMD-driven integration signal
    on skewed celltypes (target: E8.5 Hepatocytes robustness).
    """
    species = adata.obs[species_key].values
    unique_species, inverse = np.unique(species, return_inverse=True)
    counts = np.bincount(inverse)  # (K,)
    n_sp = len(unique_species)

    # Total weight per species = 1/n_sp. Per-cell within species = (1/n_sp) / N_s.
    species_weights = (1.0 / n_sp) / counts.astype(np.float64)  # (K,)
    cell_weights = species_weights[inverse]  # (N_cells,)
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
