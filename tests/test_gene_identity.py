"""Gene identity across cohorts.

``ortho_N`` numbers are table-local: the liver, testis and van Zyl ortholog tables
number their rows independently. These tests pin the cohort-aware lookups in
``revision/common/gene_ids.py`` and ``notebooks/_paper_utils.py`` and the
``hsapiens_gene_id`` join used for every cross-tissue comparison.

Data-dependent tests skip when ``data/processed/`` is absent (it is not tracked).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "notebooks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import _paper_utils as pu  # noqa: E402
from revision.common import gene_ids  # noqa: E402

PROCESSED = gene_ids.DATA_PROCESSED
COHORTS = list(gene_ids.H5ADS)
TESTIS = ["test_hmac", "test_hmouse", "test_3sp"]
CROSS_TISSUE = {
    ("liv_hmonkey", "test_hmac"): 8366,
    ("liv_hmouse", "test_hmouse"): 8233,
    ("liv_4sp", "test_3sp"): 8879,
}
MARKERS = {  # symbol: (testis-table ortho ID, Ensembl ID, expected peak cell type)
    "PRM2": ("ortho_3972", "ENSG00000122304", "Elongated_spermatids"),
    "TNP1": ("ortho_14566", "ENSG00000118245", "Elongated_spermatids"),
}

_tables = [PROCESSED / f for f in sorted(set(gene_ids.TABLES.values()))]
_h5ads = [PROCESSED / f"{s}.h5ad" for s in gene_ids.H5ADS.values()]
needs_data = pytest.mark.skipif(
    not all(p.exists() for p in _tables + _h5ads),
    reason="data/processed/ (ortholog tables and cohort h5ad files) not available",
)

_var: dict = {}


def _read(name: str, backed: bool = True):
    import anndata as ad

    return ad.read_h5ad(PROCESSED / f"{name}.h5ad", backed="r" if backed else None)


def _var_of(cohort: str) -> pd.DataFrame:
    if cohort not in _var:
        a = _read(gene_ids.H5ADS[cohort])
        _var[cohort] = a.var[["conservation_prior"]].copy()
        a.file.close()
    return _var[cohort]


# ---------------------------------------------------------------------------
# No default table


def test_lookups_require_a_cohort():
    with pytest.raises(TypeError):
        pu.annotate_genes_with_names(["ortho_0"])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        pu.load_ortholog_table()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        pu.annotate_gene_categories(["ortho_0"])  # type: ignore[call-arg]
    with pytest.raises(KeyError):
        pu.cohort_for("brain")
    with pytest.raises(KeyError):
        gene_ids.load_table("pancreas_unknown")


@pytest.mark.parametrize("name,tissue", [
    ("pancreas", "liver"), ("panc_4sp", "liver"), ("pancreas_hmonkey_s42", "liver"),
    ("liv_hzeb", "liver"), ("testis", "testis"), ("test_3sp", "testis"),
    ("testis_hmac_s42", "testis"), ("van_zyl", "van_zyl"),
])
def test_cohort_aliases(name, tissue):
    assert gene_ids.TABLES[pu.cohort_for(name)] == gene_ids.TABLES[tissue]


# ---------------------------------------------------------------------------
# Table identity


@needs_data
@pytest.mark.parametrize("cohort", COHORTS)
def test_assert_table_matches_every_cohort(cohort):
    a = _read(gene_ids.H5ADS[cohort])
    try:
        rho = gene_ids.assert_table_matches(a, gene_ids.load_table(cohort))
    finally:
        a.file.close()
    assert rho > 0.9


@needs_data
def test_assert_table_matches_van_zyl():
    import anndata as ad

    path = gene_ids.h5ad_path("van_zyl")
    if not (path.exists() and (PROCESSED / gene_ids.TABLES["van_zyl"]).exists()):
        pytest.skip("van Zyl h5ad or table not available")
    a = ad.read_h5ad(path, backed="r")
    try:
        assert gene_ids.assert_table_matches(a, gene_ids.load_table("van_zyl")) > 0.9
    finally:
        a.file.close()


@needs_data
def test_assert_table_matches_rejects_wrong_table():
    a = _read(gene_ids.H5ADS["test_3sp"])
    try:
        with pytest.raises(AssertionError):
            gene_ids.assert_table_matches(a, gene_ids.load_table("liver"))
    finally:
        a.file.close()


# ---------------------------------------------------------------------------
# Testis labels


@needs_data
def test_testis_symbol_coverage():
    vn = _var_of("test_3sp").index
    assert len(vn) == 15732
    assert int(gene_ids.symbols("test_3sp", vn).notna().sum()) == 15706
    sym = pu.annotate_genes_with_names(vn, "test_3sp")
    assert int(sym.str.fullmatch(r"ortho_\d+").sum()) == 26  # unnamed genes keep their ID


@needs_data
@pytest.mark.parametrize("cohort", TESTIS)
def test_testis_markers_labelled(cohort):
    vn = _var_of(cohort).index
    sym = pu.annotate_genes_with_names(vn, cohort)
    ens = pu.ensembl_ids(vn, cohort)
    for name, (ortho, ensg, _) in MARKERS.items():
        assert sym[ortho] == name
        assert ens[ortho] == ensg
        # the liver table names this ID differently: the bug this guards against
        assert pu.annotate_genes_with_names([ortho], "liver").iloc[0] != name


@needs_data
@pytest.mark.parametrize("cohort", TESTIS)
def test_testis_markers_peak_in_elongated_spermatids(cohort):
    import scipy.sparse as sp

    a = _read(gene_ids.H5ADS[cohort], backed=False)
    sym = pu.annotate_genes_with_names(a.var_names, cohort)
    ct = a.obs["cell_type"].astype(str).to_numpy()
    for name, (_, _, peak) in MARKERS.items():
        col = np.flatnonzero(sym.to_numpy() == name)
        assert len(col) == 1
        x = a.X[:, col[0]]
        detected = (x.toarray() if sp.issparse(x) else np.asarray(x)).ravel() > 0
        det = pd.Series(detected).groupby(ct).mean()
        assert det.idxmax() == peak, (name, det.sort_values().tail(3).to_dict())
        assert det.max() > 0.5


# ---------------------------------------------------------------------------
# Cross-tissue join


@needs_data
@pytest.mark.parametrize("pair", list(CROSS_TISSUE))
def test_cross_tissue_join(pair):
    liver, testis = pair
    vl, vt = _var_of(liver), _var_of(testis)
    il, it = pu.shared_genes(vl.index, liver, vt.index, testis)
    assert len(il) == CROSS_TISSUE[pair]
    assert (pu.ensembl_ids(vl.index[il], liver).to_numpy()
            == pu.ensembl_ids(vt.index[it], testis).to_numpy()).all()
    prior_l = vl["conservation_prior"].to_numpy(float)
    prior_t = vt["conservation_prior"].to_numpy(float)
    rho_true = spearmanr(prior_l[il], prior_t[it])[0]
    assert rho_true > 0.8
    # joining on the table-local var_names pairs unrelated genes
    common = vl.index.intersection(vt.index)
    rho_label = spearmanr(prior_l[vl.index.get_indexer(common)],
                          prior_t[vt.index.get_indexer(common)])[0]
    assert abs(rho_label) < 0.1


@needs_data
def test_within_tissue_join_equals_var_names():
    va, vb = _var_of("test_hmac"), _var_of("test_3sp")
    ia, ib = pu.shared_genes(va.index, "test_hmac", vb.index, "test_3sp")
    common = va.index.intersection(vb.index)
    assert (va.index[ia] == common).all()
    assert (vb.index[ib] == common).all()
