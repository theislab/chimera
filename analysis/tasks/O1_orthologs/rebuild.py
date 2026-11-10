"""O1 part C: liver H+Mouse through the steps of notebooks/01_data_preparation.ipynb (sections 2-6, then build_subset) with a given ortholog table.

Same functions, same order, same arguments as the notebook. ``--tag canonical`` keeps the whole table and is the control: it must reproduce the canonical object.
``--tag t3`` drops every group flagged T3 on human or mouse, then renumbers groups from 0 (the unified object needs them dense); the canonical ID stays in ``canonical_group_id``.
Writes under data/analysis/O1/rebuild_<tag>/ only.

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.rebuild --tag canonical
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from chimera._data_utils import (assemble_gin_features, compute_conservation_prior, compute_esm2_interspecies_variance, compute_expression_stats, construct_unified_adata,
                                 download_and_process_esm2, pca_esm2, precompute_p_null)
from analysis.tasks.O1_orthologs import rules as RU

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sys.path.insert(0, str(RU.REPO / "training"))
from build_subset import build_subset  # noqa: E402

ENS = {"Human": "hsapiens", "Monkey": "mmulatta", "Mouse": "mmusculus", "Zebrafish": "drerio"}
PROC = RU.REPO / "data/processed"


def flagged_groups() -> set[int]:
    """Groups flagged T3 on human or mouse, table-wide (also groups the two-species subset would drop)."""
    P = pd.read_parquet(RU.RES / "table_pairs_flags.parquet")
    M = pd.read_parquet(RU.RES / "mapping_rows.parquet")
    a = P[(P.table == "liver") & (P.species == "mmusculus") & P.T2].ortholog_group_id
    b = M[(M.table == "liver") & M.obj_species.isin(["Human", "Mouse"]) & (M.M_wrong_row | M.M_false_zero)].ortholog_group_id
    return set(a) | set(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["canonical", "t3"], required=True)
    a = ap.parse_args()
    out = RU.DATA / f"rebuild_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(RU.TABLES["liver"])
    rep = dict(tag=a.tag, n_table=len(table))
    if a.tag == "t3":
        fl = flagged_groups()
        table = table[~table.ortholog_group_id.isin(fl)].reset_index(drop=True)
        table.insert(1, "canonical_group_id", table.ortholog_group_id)
        table["ortholog_group_id"] = np.arange(len(table))
        rep.update(n_flagged=len(fl), n_kept=len(table))
    table.to_csv(out / "ortholog_table.csv", index=False)

    adata = ad.read_h5ad(RU.REPO / "data/raw/camex_pancreas.h5ad")
    species_list = sorted(adata.obs["species"].unique().tolist())
    ens = [ENS[s] for s in species_list]
    prior = compute_conservation_prior(table)
    cons, per = download_and_process_esm2(table, species_list=ens, saturn_data_dir=RU.REPO / "data/raw/saturn_esm2")
    pca = pca_esm2(cons, n_components=64)
    var = compute_esm2_interspecies_variance(per)
    per_sp = {ENS[s]: adata[adata.obs["species"] == s].copy() for s in species_list}
    uni = construct_unified_adata(per_species_adatas=per_sp, ortholog_table=table, gene_features=np.zeros((len(table), 72), dtype=np.float32), conservation_prior=prior)
    stats = compute_expression_stats(uni)
    feats, scaler = assemble_gin_features(esm2_pca=pca, esm2_interspecies_var=var, ortholog_table=table, expression_stats=stats)
    uni.varm["gene_features"] = feats
    uni.uns["p_null"] = precompute_p_null(uni)
    with open(out / "gin_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    p4 = out / "liver_4sp.h5ad"
    uni.write_h5ad(p4)
    sub = build_subset(p4, ("Human", "Mouse"), out / "liver_hmouse.h5ad")
    rep["subset_shape"] = list(sub.shape)
    if a.tag == "canonical":
        c = ad.read_h5ad(PROC / "camex_human_mouse.h5ad")
        c4 = ad.read_h5ad(PROC / "camex_pancreas_unified.h5ad")
        rep["same_var"] = bool((c.var_names == sub.var_names).all()) if c.n_vars == sub.n_vars else False
        if rep["same_var"]:
            rep["X_equal"] = int((sp.csr_matrix(c.X) != sp.csr_matrix(sub.X)).nnz) == 0
            rep["prior_max_abs"] = float(np.abs(c.var.conservation_prior.to_numpy() - sub.var.conservation_prior.to_numpy()).max())
            d = np.abs(np.asarray(c.varm["gene_features"]) - np.asarray(sub.varm["gene_features"]))
            rep["features_max_abs_by_block"] = {k: float(d[:, v].max()) for k, v in dict(esm64=list(range(64)), esm_var=[64], mean_identity=[65], expr3=[66, 67, 68], identity3=[69, 70, 71]).items()}
            rep["features_max_abs_sign_free_pca"] = float(np.minimum(np.abs(np.asarray(c.varm["gene_features"])[:, :64] - np.asarray(sub.varm["gene_features"])[:, :64]),
                                                                     np.abs(np.asarray(c.varm["gene_features"])[:, :64] + np.asarray(sub.varm["gene_features"])[:, :64])).max())
            rep["p_null_max_rel"] = float(np.abs(c.uns["p_null"] / sub.uns["p_null"] - 1).max())
        rep["unified_X_equal"] = int((sp.csr_matrix(c4.X) != sp.csr_matrix(uni.X)).nnz) == 0
        rep["unified_sha_note"] = hashlib.sha256(np.asarray(uni.varm["gene_features"]).tobytes()).hexdigest()[:16] + " vs " + hashlib.sha256(np.asarray(c4.varm["gene_features"]).tobytes()).hexdigest()[:16]
    (out / "build_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
