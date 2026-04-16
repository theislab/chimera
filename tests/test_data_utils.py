"""Tests for Chimera v2.4 data utility functions (_data_utils.py).

Covers in-memory / pure-compute functions only -- no network access required.
  - compute_conservation_prior
  - compute_expression_stats
  - assemble_gin_features
  - construct_unified_adata
  - precompute_p_null
  - get_sqrt_species_weights
  - create_synthetic_adata
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from chimera._data_utils import (
    assemble_gin_features,
    compute_conservation_prior,
    compute_expression_stats,
    construct_unified_adata,
    create_synthetic_adata,
    get_sqrt_species_weights,
    precompute_p_null,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ortholog_table():
    """Minimal ortholog table with known mean_identity values."""
    return pd.DataFrame(
        {
            "ortholog_group_id": range(10),
            "mean_identity": np.array(
                [20, 40, 60, 80, 95, 100, 0, 50, 70, 30], dtype=np.float64
            ),
        }
    )


@pytest.fixture
def sparse_adata():
    """Sparse AnnData with two species and four batches."""
    rng = np.random.RandomState(0)
    X = sp.random(100, 10, density=0.5, format="csr", random_state=0)
    X.data = np.round(X.data * 100).astype(np.float32)
    obs = pd.DataFrame(
        {
            "species": pd.Categorical(["sp1"] * 50 + ["sp2"] * 50),
            "batch": pd.Categorical(
                ["b1"] * 25 + ["b2"] * 25 + ["b3"] * 25 + ["b4"] * 25
            ),
        },
        index=[f"cell_{i}" for i in range(100)],
    )
    return ad.AnnData(X=X, obs=obs)


@pytest.fixture
def dense_adata():
    """Dense AnnData (same data as sparse_adata but with dense .X)."""
    rng = np.random.RandomState(0)
    X = sp.random(100, 10, density=0.5, format="csr", random_state=0)
    X_dense = X.toarray()
    X_dense = np.round(X_dense * 100).astype(np.float32)
    obs = pd.DataFrame(
        {
            "species": pd.Categorical(["sp1"] * 50 + ["sp2"] * 50),
            "batch": pd.Categorical(
                ["b1"] * 25 + ["b2"] * 25 + ["b3"] * 25 + ["b4"] * 25
            ),
        },
        index=[f"cell_{i}" for i in range(100)],
    )
    return ad.AnnData(X=X_dense, obs=obs)


# =========================================================================
# compute_conservation_prior
# =========================================================================


class TestComputeConservationPrior:
    """Tests for compute_conservation_prior."""

    def test_output_shape(self, ortholog_table):
        """Output tensor has shape (G,)."""
        prior = compute_conservation_prior(ortholog_table)
        assert prior.shape == (len(ortholog_table),)

    def test_dtype_is_float32(self, ortholog_table):
        """Output dtype is float32."""
        prior = compute_conservation_prior(ortholog_table)
        assert prior.dtype == torch.float32

    def test_clamped_lower_bound(self, ortholog_table):
        """Identity=0 should clamp to 0.05."""
        prior = compute_conservation_prior(ortholog_table)
        # Index 6 has mean_identity=0
        assert prior[6].item() == pytest.approx(0.05, abs=1e-6)

    def test_clamped_upper_bound(self, ortholog_table):
        """Identity=100 should clamp to 0.95."""
        prior = compute_conservation_prior(ortholog_table)
        # Index 5 has mean_identity=100
        assert prior[5].item() == pytest.approx(0.95, abs=1e-6)

    def test_scaling_midrange(self, ortholog_table):
        """Identity=80 should map to 0.80 (within clamp range)."""
        prior = compute_conservation_prior(ortholog_table)
        # Index 3 has mean_identity=80
        assert prior[3].item() == pytest.approx(0.80, abs=1e-6)

    def test_all_values_in_range(self, ortholog_table):
        """All values lie in [0.05, 0.95]."""
        prior = compute_conservation_prior(ortholog_table)
        assert prior.min().item() >= 0.05 - 1e-7
        assert prior.max().item() <= 0.95 + 1e-7

    def test_logit_is_finite(self, ortholog_table):
        """Logit of prior is finite for all genes (no 0 or 1 in prior)."""
        prior = compute_conservation_prior(ortholog_table)
        logit = torch.log(prior / (1.0 - prior))
        assert torch.isfinite(logit).all()


# =========================================================================
# compute_expression_stats
# =========================================================================


class TestComputeExpressionStats:
    """Tests for compute_expression_stats."""

    def test_keys_present(self, sparse_adata):
        """Returned dict has the three expected keys."""
        stats = compute_expression_stats(sparse_adata)
        assert set(stats.keys()) == {"log_mean_expr", "log_expr_var", "frac_nonzero"}

    def test_shapes(self, sparse_adata):
        """All arrays have shape (G,)."""
        stats = compute_expression_stats(sparse_adata)
        G = sparse_adata.n_vars
        for key, arr in stats.items():
            assert arr.shape == (G,), f"{key} shape {arr.shape} != ({G},)"

    def test_no_nan_sparse(self, sparse_adata):
        """No NaN values in any stat (sparse input)."""
        stats = compute_expression_stats(sparse_adata)
        for key, arr in stats.items():
            assert not np.isnan(arr).any(), f"NaN in {key} (sparse)"

    def test_no_nan_dense(self, dense_adata):
        """No NaN values in any stat (dense input)."""
        stats = compute_expression_stats(dense_adata)
        for key, arr in stats.items():
            assert not np.isnan(arr).any(), f"NaN in {key} (dense)"

    def test_frac_nonzero_range(self, sparse_adata):
        """frac_nonzero values are in [0, 1]."""
        stats = compute_expression_stats(sparse_adata)
        assert stats["frac_nonzero"].min() >= 0.0
        assert stats["frac_nonzero"].max() <= 1.0

    def test_dtype_float32(self, sparse_adata):
        """All returned arrays are float32."""
        stats = compute_expression_stats(sparse_adata)
        for key, arr in stats.items():
            assert arr.dtype == np.float32, f"{key} dtype {arr.dtype} != float32"

    def test_sparse_dense_agree(self, sparse_adata, dense_adata):
        """Sparse and dense paths produce approximately the same results."""
        stats_sp = compute_expression_stats(sparse_adata)
        stats_dn = compute_expression_stats(dense_adata)
        for key in stats_sp:
            np.testing.assert_allclose(
                stats_sp[key], stats_dn[key], rtol=1e-4, atol=1e-6,
                err_msg=f"Sparse/dense mismatch for {key}",
            )

    def test_known_values_on_trivial_input(self):
        """Verify exact values on a hand-crafted tiny matrix."""
        # 4 cells, 2 genes:
        # Gene 0: [1, 0, 0, 0]  -> mean=0.25, var=0.1875, frac_nonzero=0.25
        # Gene 1: [2, 2, 2, 2]  -> mean=2.0,  var=0.0,    frac_nonzero=1.0
        X = np.array([[1, 2], [0, 2], [0, 2], [0, 2]], dtype=np.float32)
        obs = pd.DataFrame(
            {"species": pd.Categorical(["s"] * 4)},
            index=[f"c{i}" for i in range(4)],
        )
        adata = ad.AnnData(X=X, obs=obs)
        stats = compute_expression_stats(adata)

        expected_log_mean = np.log1p(np.array([0.25, 2.0])).astype(np.float32)
        np.testing.assert_allclose(stats["log_mean_expr"], expected_log_mean, atol=1e-5)

        expected_frac = np.array([0.25, 1.0], dtype=np.float32)
        np.testing.assert_allclose(stats["frac_nonzero"], expected_frac, atol=1e-6)


# =========================================================================
# assemble_gin_features
# =========================================================================


class TestAssembleGinFeatures:
    """Tests for assemble_gin_features."""

    @pytest.fixture
    def gin_inputs(self, ortholog_table, sparse_adata):
        """Build all inputs needed for assemble_gin_features."""
        G = len(ortholog_table)
        rng = np.random.RandomState(99)
        esm2_pca = rng.randn(G, 64).astype(np.float32)
        esm2_var = rng.exponential(0.3, size=G).astype(np.float32)
        expr_stats = compute_expression_stats(sparse_adata)
        return esm2_pca, esm2_var, ortholog_table, expr_stats

    def test_output_shape_at_least_70(self, gin_inputs):
        """Output has shape (G, D) with D >= 70."""
        esm2_pca, esm2_var, orth, expr = gin_inputs
        features, scaler = assemble_gin_features(esm2_pca, esm2_var, orth, expr)
        G = len(orth)
        assert features.shape[0] == G
        assert features.shape[1] >= 70

    def test_approximately_zero_mean(self, gin_inputs):
        """StandardScaler should produce near-zero column means."""
        esm2_pca, esm2_var, orth, expr = gin_inputs
        features, _ = assemble_gin_features(esm2_pca, esm2_var, orth, expr)
        col_means = np.abs(features.mean(axis=0))
        assert col_means.max() < 0.01, f"Max column mean {col_means.max()} >= 0.01"

    def test_approximately_unit_variance(self, gin_inputs):
        """StandardScaler should produce near-unit column standard deviations."""
        esm2_pca, esm2_var, orth, expr = gin_inputs
        features, _ = assemble_gin_features(esm2_pca, esm2_var, orth, expr)
        col_stds = features.std(axis=0)
        # Padded zero-columns will have std=0 after scaling (constant -> 0).
        non_const = col_stds > 0.01
        assert np.abs(col_stds[non_const] - 1.0).max() < 0.01

    def test_scaler_is_returned(self, gin_inputs):
        """A fitted StandardScaler is returned as second element."""
        from sklearn.preprocessing import StandardScaler

        esm2_pca, esm2_var, orth, expr = gin_inputs
        _, scaler = assemble_gin_features(esm2_pca, esm2_var, orth, expr)
        assert isinstance(scaler, StandardScaler)
        # Confirm it is fitted (has mean_ attribute).
        assert hasattr(scaler, "mean_")

    def test_dtype_float32(self, gin_inputs):
        """Output features are float32."""
        esm2_pca, esm2_var, orth, expr = gin_inputs
        features, _ = assemble_gin_features(esm2_pca, esm2_var, orth, expr)
        assert features.dtype == np.float32


# =========================================================================
# construct_unified_adata
# =========================================================================


class TestConstructUnifiedAdata:
    """Tests for construct_unified_adata."""

    @pytest.fixture
    def unified_inputs(self):
        """Build minimal per-species adatas and ortholog table for unification."""
        rng = np.random.RandomState(7)
        G = 5  # ortholog groups

        orth = pd.DataFrame(
            {
                "ortholog_group_id": range(G),
                "mean_identity": rng.uniform(30, 100, size=G),
                "sp1_gene_id": [f"sp1_g{i}" for i in range(G)],
                "sp2_gene_id": [f"sp2_g{i}" for i in range(G)],
            }
        )

        # Per-species AnnData -- gene names match the ortholog table gene IDs.
        def _make_sp_adata(sp_name, n_cells):
            gene_ids = [f"{sp_name}_g{i}" for i in range(G)]
            X = sp.random(n_cells, G, density=0.6, format="csr", random_state=42)
            X.data = np.round(X.data * 50).astype(np.float32)
            obs = pd.DataFrame(
                {
                    "species": pd.Categorical([sp_name] * n_cells),
                    "batch": pd.Categorical(
                        [f"{sp_name}_b0"] * (n_cells // 2)
                        + [f"{sp_name}_b1"] * (n_cells - n_cells // 2)
                    ),
                },
                index=[f"{sp_name}_cell_{i}" for i in range(n_cells)],
            )
            var = pd.DataFrame(index=gene_ids)
            return ad.AnnData(X=X, obs=obs, var=var)

        per_sp = {
            "sp1": _make_sp_adata("sp1", 30),
            "sp2": _make_sp_adata("sp2", 20),
        }

        gene_features = rng.randn(G, 72).astype(np.float32)
        conservation_prior = compute_conservation_prior(orth)

        return per_sp, orth, gene_features, conservation_prior

    def test_total_cells(self, unified_inputs):
        """Unified AnnData has sum-of-per-species cell count."""
        per_sp, orth, gf, cp = unified_inputs
        unified = construct_unified_adata(per_sp, orth, gf, cp)
        expected_cells = sum(a.n_obs for a in per_sp.values())
        assert unified.n_obs == expected_cells

    def test_gene_count(self, unified_inputs):
        """Unified AnnData gene count equals ortholog group count."""
        per_sp, orth, gf, cp = unified_inputs
        unified = construct_unified_adata(per_sp, orth, gf, cp)
        assert unified.n_vars == len(orth)

    def test_species_column_present(self, unified_inputs):
        """obs contains 'species' column."""
        per_sp, orth, gf, cp = unified_inputs
        unified = construct_unified_adata(per_sp, orth, gf, cp)
        assert "species" in unified.obs.columns

    def test_batch_column_present(self, unified_inputs):
        """obs contains 'batch' column."""
        per_sp, orth, gf, cp = unified_inputs
        unified = construct_unified_adata(per_sp, orth, gf, cp)
        assert "batch" in unified.obs.columns

    def test_gene_features_in_varm(self, unified_inputs):
        """gene_features attached to varm."""
        per_sp, orth, gf, cp = unified_inputs
        unified = construct_unified_adata(per_sp, orth, gf, cp)
        assert "gene_features" in unified.varm
        assert unified.varm["gene_features"].shape == gf.shape

    def test_conservation_prior_in_var(self, unified_inputs):
        """conservation_prior attached to var."""
        per_sp, orth, gf, cp = unified_inputs
        unified = construct_unified_adata(per_sp, orth, gf, cp)
        assert "conservation_prior" in unified.var.columns
        assert len(unified.var["conservation_prior"]) == len(orth)


# =========================================================================
# precompute_p_null
# =========================================================================


class TestPrecomputePNull:
    """Tests for precompute_p_null."""

    def test_shape(self, sparse_adata):
        """Output shape is (K, G)."""
        p = precompute_p_null(sparse_adata)
        K = sparse_adata.obs["species"].nunique()
        G = sparse_adata.n_vars
        assert p.shape == (K, G)

    def test_all_positive(self, sparse_adata):
        """Laplace smoothing ensures all values > 0."""
        p = precompute_p_null(sparse_adata)
        assert (p > 0).all()

    def test_rows_sum_near_one(self, sparse_adata):
        """Each species row sums to approximately 1."""
        p = precompute_p_null(sparse_adata)
        row_sums = p.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=0.05)

    def test_analytical_tiny_dataset(self):
        """Verify exact values on a hand-crafted 2-cell, 2-gene dataset.

        Species A: 1 cell with counts [3, 1]  -> lib=4
          p_null[0,0] = (3+1) / (4+1*2) = 4/6
          p_null[0,1] = (1+1) / (4+1*2) = 2/6
        Species B: 1 cell with counts [0, 6]  -> lib=6
          p_null[1,0] = (0+1) / (6+1*2) = 1/8
          p_null[1,1] = (6+1) / (6+1*2) = 7/8
        """
        X = np.array([[3, 1], [0, 6]], dtype=np.float32)
        obs = pd.DataFrame(
            {"species": pd.Categorical(["A", "B"])},
            index=["c0", "c1"],
        )
        adata = ad.AnnData(X=X, obs=obs)
        p = precompute_p_null(adata, eps=1.0)

        expected = np.array(
            [[4.0 / 6.0, 2.0 / 6.0], [1.0 / 8.0, 7.0 / 8.0]], dtype=np.float64
        )
        np.testing.assert_allclose(p, expected, atol=1e-10)

    def test_dtype_float64(self, sparse_adata):
        """Output dtype is float64."""
        p = precompute_p_null(sparse_adata)
        assert p.dtype == np.float64


# =========================================================================
# get_sqrt_species_weights
# =========================================================================


class TestGetSqrtSpeciesWeights:
    """Tests for get_sqrt_species_weights."""

    def test_shape(self, sparse_adata):
        """Output has shape (N_cells,)."""
        w = get_sqrt_species_weights(sparse_adata)
        assert w.shape == (sparse_adata.n_obs,)

    def test_sums_to_one(self, sparse_adata):
        """Weights sum to approximately 1."""
        w = get_sqrt_species_weights(sparse_adata)
        assert w.sum() == pytest.approx(1.0, abs=1e-10)

    def test_equal_species_equal_weights(self):
        """When species are equal-sized, all cells get the same weight."""
        obs = pd.DataFrame(
            {"species": pd.Categorical(["A"] * 20 + ["B"] * 20)},
            index=[f"c{i}" for i in range(40)],
        )
        X = sp.random(40, 5, density=0.3, format="csr", random_state=1)
        adata = ad.AnnData(X=X, obs=obs)
        w = get_sqrt_species_weights(adata)
        np.testing.assert_allclose(w, w[0], atol=1e-12)

    def test_minority_species_gets_higher_weight(self):
        """Species with fewer cells should receive higher per-cell weight."""
        obs = pd.DataFrame(
            {"species": pd.Categorical(["A"] * 90 + ["B"] * 10)},
            index=[f"c{i}" for i in range(100)],
        )
        X = sp.random(100, 5, density=0.3, format="csr", random_state=2)
        adata = ad.AnnData(X=X, obs=obs)
        w = get_sqrt_species_weights(adata)
        # Cells from species B (index 90-99) should have higher weights
        # than cells from species A (index 0-89).
        assert w[95] > w[0]

    def test_numerical_check(self):
        """Verify exact weights on a small known dataset.

        3 species: A=10 cells, B=20 cells, C=30 cells.  N=60.
          raw_A = sqrt(60/10) = sqrt(6)
          raw_B = sqrt(60/20) = sqrt(3)
          raw_C = sqrt(60/30) = sqrt(2)
          total_raw = 10*sqrt(6) + 20*sqrt(3) + 30*sqrt(2)
          w_A_cell = sqrt(6) / total_raw
        """
        obs = pd.DataFrame(
            {
                "species": pd.Categorical(
                    ["A"] * 10 + ["B"] * 20 + ["C"] * 30
                )
            },
            index=[f"c{i}" for i in range(60)],
        )
        X = sp.random(60, 3, density=0.5, format="csr", random_state=3)
        adata = ad.AnnData(X=X, obs=obs)
        w = get_sqrt_species_weights(adata)

        raw_A = np.sqrt(60.0 / 10)
        raw_B = np.sqrt(60.0 / 20)
        raw_C = np.sqrt(60.0 / 30)
        total = 10 * raw_A + 20 * raw_B + 30 * raw_C

        np.testing.assert_allclose(w[0], raw_A / total, atol=1e-12)
        np.testing.assert_allclose(w[10], raw_B / total, atol=1e-12)
        np.testing.assert_allclose(w[30], raw_C / total, atol=1e-12)

    def test_all_positive(self, sparse_adata):
        """All weights are strictly positive."""
        w = get_sqrt_species_weights(sparse_adata)
        assert (w > 0).all()


# =========================================================================
# create_synthetic_adata
# =========================================================================


class TestCreateSyntheticAdata:
    """Tests for create_synthetic_adata."""

    def test_returned_keys(self):
        """Returned dict has all expected keys."""
        result = create_synthetic_adata(
            n_cells_per_species=20, n_genes=30, n_species=2, seed=0
        )
        expected_keys = {
            "adata",
            "gene_features",
            "conservation_prior",
            "p_null",
            "ortholog_table",
        }
        assert set(result.keys()) == expected_keys

    def test_adata_obs_columns(self):
        """AnnData obs has species, batch, cell_type columns."""
        result = create_synthetic_adata(
            n_cells_per_species=20, n_genes=30, n_species=2, seed=0
        )
        adata = result["adata"]
        for col in ("species", "batch", "cell_type"):
            assert col in adata.obs.columns, f"Missing obs column: {col}"

    def test_cell_count(self):
        """Total cells = n_cells_per_species * n_species."""
        ncs, ns = 25, 3
        result = create_synthetic_adata(
            n_cells_per_species=ncs, n_genes=40, n_species=ns, seed=1
        )
        assert result["adata"].n_obs == ncs * ns

    def test_gene_count(self):
        """Number of genes equals n_genes."""
        ng = 50
        result = create_synthetic_adata(
            n_cells_per_species=20, n_genes=ng, n_species=2, seed=2
        )
        assert result["adata"].n_vars == ng

    def test_gene_features_shape(self):
        """gene_features shape is (n_genes, 72)."""
        ng = 40
        result = create_synthetic_adata(
            n_cells_per_species=20, n_genes=ng, n_species=2, seed=3
        )
        assert result["gene_features"].shape == (ng, 72)

    def test_conservation_prior_shape(self):
        """conservation_prior shape is (n_genes,)."""
        ng = 40
        result = create_synthetic_adata(
            n_cells_per_species=20, n_genes=ng, n_species=2, seed=4
        )
        assert result["conservation_prior"].shape == (ng,)

    def test_p_null_shape(self):
        """p_null shape is (n_species, n_genes)."""
        ng, ns = 40, 3
        result = create_synthetic_adata(
            n_cells_per_species=20, n_genes=ng, n_species=ns, seed=5
        )
        assert result["p_null"].shape == (ns, ng)

    def test_counts_non_negative_integer(self):
        """Raw counts in .X are non-negative integers."""
        result = create_synthetic_adata(
            n_cells_per_species=30, n_genes=20, n_species=2, seed=6
        )
        X = result["adata"].X
        if sp.issparse(X):
            data = X.data
        else:
            data = np.asarray(X).ravel()
        assert (data >= 0).all(), "Counts contain negative values"
        np.testing.assert_array_equal(data, np.round(data))

    def test_reproducibility(self):
        """Same seed produces identical outputs."""
        r1 = create_synthetic_adata(
            n_cells_per_species=20, n_genes=30, n_species=2, seed=99
        )
        r2 = create_synthetic_adata(
            n_cells_per_species=20, n_genes=30, n_species=2, seed=99
        )
        X1 = r1["adata"].X.toarray() if sp.issparse(r1["adata"].X) else r1["adata"].X
        X2 = r2["adata"].X.toarray() if sp.issparse(r2["adata"].X) else r2["adata"].X
        np.testing.assert_array_equal(X1, X2)
