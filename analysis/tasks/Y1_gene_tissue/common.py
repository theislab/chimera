"""Tissue registry and readers. Gene identity through analysis.common.gene_ids only."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.Y1_gene_tissue import rules as RU

ROOT = gene_ids.REPO_ROOT
RES = ROOT / "results" / "analysis"
CACHE = RU.OUT / "cache"
FW = RU.OUT / "forward"

_MM = {42: "xds-hmouse-iter90-150ep-s42", 43: "xds-hmouse-iter90-150ep-s43", 44: "xds-hmouse-iter90-150ep-s44",
       **{s: f"rev-liv_hmouse-s{s}" for s in range(45, 52)}}
_TT = {42: "testis-hmouse-iter111-pclamp-coc-200ep-s42", **{s: f"rev-test_hmouse-s{s}" for s in range(43, 52)}}
TISSUES = {
    "liver": dict(h5ad=gene_ids.DATA_PROCESSED / "camex_human_mouse.h5ad", table="liv_hmouse", group="cell_type", unit="batch", runs=_MM),
    "testis": dict(h5ad=gene_ids.DATA_PROCESSED / "camex_testis_human_mouse.h5ad", table="test_hmouse", group="cell_type", unit="sample_batch",
                   runs=_TT, coarse=T42.TESTIS_COARSE),
    "kidney": dict(h5ad=gene_ids.h5ad_path("kidney"), table="kidney", group="cell_ontology_class", unit="sample_batch",
                   runs={42: "rev-kid-grid3-s42", **{s: f"rev-kid-s{s}" for s in range(43, 47)}}),
    "islet": dict(h5ad=ROOT / "data/analysis/disease_islet/islet_healthy_train.h5ad", table="testis", group="cell_ontology_class", unit="sample_batch",
                  runs={42: "rev-isl-grid14-s42", **{s: f"rev-isl-s{s}" for s in range(43, 47)}}),
    "bone_marrow": dict(h5ad=gene_ids.h5ad_path("bone_marrow"), table="bone_marrow", group="cell_ontology_class", unit="batch",
                        runs={42: "rev-bm2-grid5-s42", 46: "rev-bm2-s46"}),
    "eye": dict(h5ad=ROOT / "data/analysis/van_zyl/van_zyl_hmouse.h5ad", table="van_zyl", group="cell_ontology_class", unit="batch",
                runs={42: "rev-vz-hmouse-grid1-s42", 43: "rev-vz-hmouse-s43"}),
}
ORDER = RU.TISSUES
LABEL = {"liver": "Liver", "testis": "Testis", "kidney": "Kidney", "islet": "Islet", "bone_marrow": "Bone marrow", "eye": "Eye"}


def species_names(obs_species) -> tuple[str, str]:
    u = sorted(set(map(str, obs_species)))
    h = next(s for s in u if s.lower() == "human")
    m = next(s for s in u if s.lower() == "mouse")
    return h, m


def read(t: str):
    """Training object of tissue t with obs['y1_group'] = the T4_2_gene_baselines groups (testis coarse map)."""
    d = TISSUES[t]
    a = ad.read_h5ad(d["h5ad"])
    lab = a.obs[d["group"]].astype(str)
    if "coarse" in d:
        lab = lab.map(d["coarse"])
    a.obs["y1_group"] = lab.astype(str).values
    return a


def hm_view(a):
    """Human and mouse cells only (bone marrow also has lemur)."""
    h, m = species_names(a.obs["species"])
    return a[a.obs["species"].astype(str).isin([h, m]).to_numpy()].copy(), h, m


def csr(X):
    return (X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)).astype(np.float64)


def tissue_table(t: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / f"tissue__{t}.parquet")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
