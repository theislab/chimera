"""Per-species inputs of SATURN and CAMEX for the integration benchmark (1:1 orthologs), from the cohort's object as in H1_harmonise. Run in chimera_env.  python -m ...integration.prep <cohort>"""

from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from analysis.common import gene_ids
from analysis.tasks.B2_benchmark.integration import rules as RI

ESM2 = RI.REPO / "data" / "raw" / "saturn_esm2" / "protein_embeddings_export" / "ESM2"


def leiden_within(x: ad.AnnData) -> np.ndarray:
    """T5_2_adapters rule."""
    b = x.copy()
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    sc.pp.highly_variable_genes(b, n_top_genes=2000, flavor="seurat")
    b = b[:, b.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(b, max_value=10)
    sc.tl.pca(b, n_comps=30, svd_solver="arpack", random_state=0)
    sc.pp.neighbors(b, n_neighbors=15, n_pcs=30, transformer="sklearn", random_state=0)
    sc.tl.leiden(b, resolution=1.0, flavor="igraph", n_iterations=2, directed=False, random_state=0, key_added="leiden")
    return ("L" + b.obs["leiden"].astype(str)).to_numpy()


def main(c: str) -> None:
    d = RI.COHORTS[c]
    out = RI.INPUTS / c
    (out / "saturn").mkdir(parents=True, exist_ok=True)
    (out / "camex").mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(RI.REPO / d["h5ad"])
    tab = gene_ids.load_table(d["table"])
    ids = pd.Index(a.var_names.astype(str)).str.replace("ortho_", "", regex=False).astype(int)
    spc = a.obs["species"].astype(str).to_numpy()
    syms = {s: tab.loc[ids, col].astype(object).values for s, (_, col) in d["species"].items()}
    keep = np.ones(a.n_vars, bool)
    for v in syms.values():
        keep &= pd.notna(v) & ~pd.Series(v).duplicated().to_numpy()
    rows = []
    human = next(s for s in d["species"] if s.lower() == "human")
    for s, (sat, _) in d["species"].items():
        m = spc == s
        X = sp.csr_matrix(a.X[m][:, keep], dtype=np.float32)
        obs = pd.DataFrame(dict(cell_type=a.obs[d["label"]].astype(str).to_numpy()[m]), index=a.obs_names.astype(str)[m])
        obs["cell_ontology_class"] = obs["cell_type"]
        obs["species"] = sat
        x = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(syms[s][keep].astype(str))))
        nz = np.asarray((x.X > 0).sum(0)).ravel() >= 3
        y = x[:, nz].copy()
        y.obs["leiden_within"] = leiden_within(y)
        y.write_h5ad(out / "saturn" / f"{sat}.h5ad")
        x.write_h5ad(out / "camex" / f"raw-b2-{sat.replace('_', '')}-x.h5ad")
        rows.append(dict(path=str(out / "saturn" / f"{sat}.h5ad"), species=sat, embedding_path=str(ESM2 / f"{sat}_embedding.torch")))
        if s != human:
            n = sat.replace("_", "")
            pd.DataFrame({"raw-b2-human-x.h5ad": syms[human][keep], f"raw-b2-{n}-x.h5ad": syms[s][keep]}).to_csv(out / "camex" / f"gene_matches_human2{n}.csv", index=False)
    pd.DataFrame(rows).sort_values("species").to_csv(out / "saturn" / "run.csv", index=False)
    print(c, int(keep.sum()), "genes", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
