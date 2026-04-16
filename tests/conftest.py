"""Shared fixtures for Chimera v2.4 test suite."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from chimera._data_utils import create_synthetic_adata, precompute_p_null


@pytest.fixture
def synthetic_data():
    """Create a synthetic dataset matching the full Chimera pipeline expectations."""
    result = create_synthetic_adata(
        n_cells_per_species=50,
        n_genes=100,
        n_species=3,
        n_batches_per_species=2,
        seed=42,
    )
    adata = result["adata"]

    # Build batch_to_species mapping
    species_cats = list(adata.obs["species"].cat.categories)
    batch_cats = list(adata.obs["batch"].cat.categories)
    batch_to_species = []
    for b in batch_cats:
        mask = adata.obs["batch"] == b
        s = adata.obs.loc[mask, "species"].iloc[0]
        batch_to_species.append(species_cats.index(s))

    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)

    return {
        "n_genes": 100,
        "n_species": 3,
        "n_batch": 6,
        "gene_features": torch.tensor(
            result["gene_features"], dtype=torch.float32
        ),
        "conservation_prior": torch.tensor(
            result["conservation_prior"].numpy()
            if isinstance(result["conservation_prior"], torch.Tensor)
            else result["conservation_prior"],
            dtype=torch.float32,
        ),
        "p_null": torch.tensor(
            precompute_p_null(adata), dtype=torch.float32
        ),
        "batch_to_species_idx": torch.tensor(
            batch_to_species, dtype=torch.long
        ),
        "batch_dict": {
            "X": torch.tensor(X, dtype=torch.float32),
            "species": torch.tensor(
                adata.obs["species"].cat.codes.values, dtype=torch.long
            ).unsqueeze(-1),
            "batch": torch.tensor(
                adata.obs["batch"].cat.codes.values, dtype=torch.long
            ).unsqueeze(-1),
            "library_size": torch.tensor(
                X.sum(axis=1), dtype=torch.float32
            ).unsqueeze(-1),
        },
    }
