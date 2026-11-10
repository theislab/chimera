"""scVI on the same objects: n_latent 10, n_hidden 128, 100 epochs, the batch key of the Chimera run. Saved so it can encode new cells; nothing is written to training/runs.
python -m analysis.tasks.Z2_embedding.scvi_train <cohort> <seed>"""
from __future__ import annotations

import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.tasks.Z2_embedding import rules as RU

MANIFEST = [("bone_marrow", 42)] + [(c, s) for c in ("kidney", "islet", "test_hmouse", "test_3sp", "test_hmac", "liv_hmonkey", "bone_marrow") for s in (42, 43, 44) if (c, s) != ("bone_marrow", 42)]


def chimera_batch_key(c: str) -> str:
    r = pd.read_csv(RU.REPO / "results/analysis/H1/runs.tsv", sep="\t")
    run = r[(r.cohort == c) & r.kept].run.iloc[0]
    return json.loads((RU.RUNS / run / "config.json").read_text()).get("batch_key") or "batch"


def main():
    import scvi

    c, seed = sys.argv[1], int(sys.argv[2])
    d = RU.OUT / "scvi" / f"{c}_s{seed}"
    if (d / "z_mean.npy").exists():
        return
    scvi.settings.seed = seed
    a = ad.read_h5ad(RU.REPO / RU.H5AD[c])
    a.X = sp.csr_matrix(a.X)
    key = chimera_batch_key(c)
    scvi.model.SCVI.setup_anndata(a, batch_key=key)
    m = scvi.model.SCVI(a, n_latent=10, n_hidden=128)
    m.train(max_epochs=100, accelerator="cpu", enable_progress_bar=False)
    m.save(str(d), overwrite=True, save_anndata=False)
    np.save(d / "z_mean.npy", m.get_latent_representation())
    (d / "batch_key.txt").write_text(key + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "row":
        c, s = MANIFEST[int(sys.argv[2])]
        sys.argv = [sys.argv[0], c, str(s)]
    main()
