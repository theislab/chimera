"""scVI(species) reference on the same object, same settings as training/compare_vs_scvi.py, cached there; metrics by the same function.

    python analysis/tasks/C1_cohorts/scvi_ref.py <cohort>
"""
from __future__ import annotations

import json
import sys

import anndata as ad
import numpy as np

from analysis.tasks.C1_cohorts import configs as CF

sys.path.insert(0, str(CF.ROOT / "training"))
from compare_vs_scvi import compute_integration_metrics, get_or_train_scvi  # noqa: E402


def main() -> int:
    c = sys.argv[1]
    z = get_or_train_scvi(CF.h5ad(c), 42, 100, 10, 128, batch_key="species", accelerator="cpu")
    a = ad.read_h5ad(CF.h5ad(c), backed="r")
    m = compute_integration_metrics(z, a.obs["species"].astype(str).to_numpy(), a.obs["cell_ontology_class"].astype(str).to_numpy(), k=15, seed=42)["global"]
    m = {k: float(v) for k, v in m.items() if np.isscalar(v) and not isinstance(v, str)}
    d = CF.RES / c
    d.mkdir(parents=True, exist_ok=True)
    (d / "scvi_reference.json").write_text(json.dumps(m, indent=2))
    np.save(d / "scvi_latent.npy", z)
    print(c, m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
