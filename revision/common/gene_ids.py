"""Gene identity for every Chimera cohort. The only gene-identity helper in revision code.

``ortho_N`` numbers are table-local: the liver, testis, Van Zyl, bone-marrow and kidney ortholog
tables assign them independently, so the same ID names different genes in different tables.
Every lookup must therefore state its cohort, and joins across tissues must use
``hsapiens_gene_id``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(os.environ.get("CHIMERA_ROOT", Path(__file__).resolve().parents[2]))
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

_LIVER = "ortholog_table.csv"
_TESTIS = "ortholog_table_testis.csv"
# Tables built during the revision live outside data/processed/; an absolute path here is
# kept as is by ``DATA_PROCESSED / name``.
_VAN_ZYL = str(REPO_ROOT / "data" / "revision" / "van_zyl" / "ortholog_table_van_zyl.csv")
_BONE_MARROW = str(REPO_ROOT / "data" / "revision" / "bone_marrow" / "ortholog_table_bone_marrow.csv")
# Kidney AKI case (plan section 8c): human-mouse 1:1, P13_kidney/build_orthologs.py.
_KIDNEY = str(REPO_ROOT / "data" / "revision" / "disease_kidney" / "ortholog_table_kidney.csv")

TABLES = {
    "liver": _LIVER,
    "liv_hmonkey": _LIVER,
    "liv_hmouse": _LIVER,
    "liv_hzeb": _LIVER,
    "liv_4sp": _LIVER,
    "testis": _TESTIS,
    "test_hmac": _TESTIS,
    "test_hmouse": _TESTIS,
    "test_3sp": _TESTIS,
    "van_zyl": _VAN_ZYL,
    "bone_marrow": _BONE_MARROW,
    "kidney": _KIDNEY,
}

H5ADS = {
    "liv_hmonkey": "camex_human_monkey",
    "liv_hmouse": "camex_human_mouse",
    "liv_hzeb": "camex_human_zebrafish",
    "liv_4sp": "camex_pancreas_unified",
    "test_hmac": "camex_testis_human_macaque",
    "test_hmouse": "camex_testis_human_mouse",
    "test_3sp": "camex_testis_unified",
}

# Cohorts rebuilt during the revision live outside data/processed/; full paths, so code
# that iterates H5ADS under DATA_PROCESSED never reads the superseded processed object.
REVISION_H5ADS = {
    "van_zyl": REPO_ROOT / "data" / "revision" / "van_zyl" / "van_zyl_unified.h5ad",
    "bone_marrow": REPO_ROOT / "data" / "revision" / "bone_marrow" / "bone_marrow_unified.h5ad",
    "kidney": REPO_ROOT / "data" / "revision" / "disease_kidney" / "kidney_healthy_train.h5ad",
}


def h5ad_path(cohort: str) -> Path:
    """Training h5ad for ``cohort``: the rebuilt object for revision cohorts, else data/processed/."""
    if cohort in REVISION_H5ADS:
        return REVISION_H5ADS[cohort]
    return DATA_PROCESSED / f"{H5ADS[cohort]}.h5ad"


_cache: dict = {}


def load_table(cohort: str):
    """Ortholog table for ``cohort``, indexed by ``ortholog_group_id``. Raises on unknown cohort."""
    import pandas as pd

    if cohort not in TABLES:
        raise KeyError(f"unknown cohort {cohort!r}; expected one of {sorted(TABLES)}")
    name = TABLES[cohort]
    if name not in _cache:
        _cache[name] = pd.read_csv(DATA_PROCESSED / name).set_index("ortholog_group_id")
    return _cache[name]


def _group_ids(var_names: Iterable[str]):
    import pandas as pd

    idx = pd.Index([str(v) for v in var_names])
    ids = idx.str.extract(r"^ortho_(\d+)$")[0]
    if ids.isna().any():
        raise ValueError(f"not ortho_N names: {idx[ids.isna()][:5].tolist()}")
    return idx, ids.astype(int)


def _lookup(cohort: str, var_names: Iterable[str], column: str):
    import pandas as pd

    table = load_table(cohort)
    idx, ids = _group_ids(var_names)
    return pd.Series(ids.map(table[column]).values, index=idx, name=column, dtype=object)


def symbols(cohort: str, var_names: Iterable[str]):
    """Human gene symbol per ``var_name``; missing symbols stay NaN."""
    return _lookup(cohort, var_names, "hsapiens_gene_name")


def ensembl(cohort: str, var_names: Iterable[str]):
    """Human Ensembl gene ID per ``var_name``: the only valid cross-tissue join key."""
    return _lookup(cohort, var_names, "hsapiens_gene_id")


def assert_table_matches(adata, table) -> float:
    """Fail unless ``table`` is the table that built ``adata``; return the Spearman used."""
    from scipy.stats import spearmanr

    idx, ids = _group_ids(adata.var_names)
    identity = ids.map(table["mean_identity"]).to_numpy(dtype=float)
    prior = adata.var["conservation_prior"].to_numpy(dtype=float)
    rho = float(spearmanr(prior, identity, nan_policy="omit")[0])
    if not rho > 0.9:
        raise AssertionError(f"ortholog table does not match adata: spearman {rho:.3f} <= 0.9")
    return rho
