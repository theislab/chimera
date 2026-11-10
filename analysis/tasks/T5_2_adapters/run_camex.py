"""Adapter C runner: CAMEX integration, CPU. Run in camex_env.

Loads the authors' `analysis/<tissue>/params.py` from the pinned clone and
changes only: `path` (cohort input folder), the dataset rows (the cohort's
species pairs, human as labelled source), `device` -> 'cpu', `log_path`.
All other values stay at the authors' defaults (epoch_integration=10,
dim_hidden=128, gnn_layer_num=2, encoder=GCN). Runs `Trainer.integration()`
only, never `annotation()`. Seeds random, numpy, torch and dgl after the
CAMEX import (CAMEX.base calls np.random.seed(42) at import and has no seed
argument). Aligns `obsm['X_CAMEX_Integration']` to the processed h5ad by
obs_names.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cohorts as C  # noqa: E402

NT = C.n_threads()
C.set_thread_env(NT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
sys.path.insert(0, str(C.CAMEX_DIR))
import dgl  # noqa: E402
from CAMEX.base import Dataset  # noqa: E402
from CAMEX.trainer import Trainer  # noqa: E402


def authors_params(tissue: str) -> dict:
    sub = {"liver": "1liver", "testis": "2testis", "kidney": "1liver"}[tissue]  # kidney: the authors' liver defaults
    spec = importlib.util.spec_from_file_location(f"camex_params_{tissue}", C.CAMEX_DIR / "analysis" / sub / "params.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return copy.deepcopy(m.PARAMS)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stem", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inputs", default=None)
    p.add_argument("--work", required=True)
    p.add_argument("--dest", default=None)
    p.add_argument("--epochs", type=int, default=None, help="smoke only; default = authors' epoch_integration")
    a = p.parse_args(argv)

    inputs = (Path(a.inputs) if a.inputs else C.INPUTS / a.stem / "camex").resolve()
    work = Path(a.work).resolve()
    if (work / "gene_integration_hidden.h5ad").exists():
        raise SystemExit(f"refusing to overwrite {work}")
    work.mkdir(parents=True, exist_ok=True)
    dest = (Path(a.dest) if a.dest else work / "final").resolve()
    tissue = C.COHORTS[a.stem][1]
    smap = C.species_map(a.stem)

    P = authors_params(tissue)
    authors_train = copy.deepcopy(P["train"])
    rows = P["preprocess"]["dataset_file"]
    if tissue == "kidney":
        rows = pd.DataFrame([["raw-kidney-human-KPMP.h5ad", True, "gene_matches_human2mouse.csv", "raw-kidney-mouse-KPMP.h5ad", False]],
                            columns=rows.columns)
    files = {Path(v[0]).name for v in smap.values()}
    rows = rows[rows["source"].isin(files) & rows["destination"].isin(files)].reset_index(drop=True)
    if len(rows) != len(smap) - 1:
        raise SystemExit(f"dataset rows {rows} do not cover {files}")
    P["preprocess"]["dataset_file"] = rows
    P["preprocess"]["path"] = str(inputs) + "/"
    P["train"]["device"] = "cpu"
    if a.epochs is not None:
        P["train"]["epoch_integration"] = a.epochs
    for v in P.values():
        v["log_path"] = str(work) + "/"
        v["time_start"] = time.strftime("%Y-%m-%d-%H-%M-%S")
    changed = {k: (authors_train.get(k), P["train"][k]) for k in P["train"]
               if authors_train.get(k) != P["train"][k] and k not in ("log_path", "time_start")}
    print("dataset rows:\n", rows, "\ntrain params changed from authors':", changed, flush=True)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dgl.seed(a.seed)

    t0 = time.time()
    dataset = Dataset(**P["preprocess"])
    t_pre = time.time() - t0
    adata = dataset.adata_whole
    trainer = Trainer(adata, dataset.dgl_data, **P["train"])
    trainer.integration()
    wall = time.time() - t0

    emb = np.asarray(adata.obsm["X_CAMEX_Integration"])
    names = pd.Index(adata.obs_names.astype(str))
    obs, n_vars = C.processed_obs(a.stem)
    if inputs != (C.INPUTS / a.stem / "camex").resolve():
        man = json.loads((inputs / "manifest.json").read_text())
        import anndata as ad
        keep = []
        for f in man["files"]:
            if f.endswith(".h5ad"):
                keep += list(ad.read_h5ad(inputs / f, backed="r").obs_names)
        obs = obs.loc[obs.index.isin(keep)]
    if names.duplicated().any() or set(names) != set(obs.index):
        raise SystemExit("CAMEX output cells != processed cells")
    pos = names.get_indexer(obs.index)
    z = emb[pos]
    # label agreement after alignment (CAMEX carries cell_ontology_class from the raw files)
    agree = float(np.mean(adata.obs["cell_ontology_class"].astype(str).values[pos] == obs["cell_ontology_class"].values))
    print(f"label agreement after alignment: {agree:.6f}", flush=True)
    if agree != 1.0:
        raise SystemExit("label agreement != 1.0; refusing to save")

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(inputs / "manifest.json", dest / "input_manifest.json")
    gene_h5ad = work / "gene_integration_hidden.h5ad"
    cfg = C.base_config(
        stem=a.stem, n_obs=int(len(obs)), n_vars=n_vars, method="camex", seed=a.seed,
        labels_key=None, n_latent=int(z.shape[1]),
        extra={
            "pkg_versions": C.pkg_versions(["torch", "dgl", "scanpy", "anndata", "numpy", "pandas", "numba",
                                            "harmonypy", "scikit-learn", "scipy", "leidenalg"]),
            "env": C.env_info(),
            "tool": "CAMEX", "tool_repo": "zhanglabtools/CAMEX", "tool_state": C.tool_state(C.CAMEX_DIR),
            "params_source": "CAMEX/analysis/" + ("1liver" if tissue == "liver" else "2testis") + "/params.py",
            "dataset_rows": rows.values.tolist(),
            "train_params": {k: v for k, v in P["train"].items() if k not in ("log_path", "time_start")},
            "train_params_changed_from_authors": changed,
            "labels_used_in_training": "none (Trainer.integration only; cell_ontology_class read but unused)",
            "label_agreement_after_alignment": agree,
            "gene_input": "tool native, many-to-many homology; HVG 2000 + top-50 DEG per Leiden cluster",
            "latent": "obsm['X_CAMEX_Integration']",
            "seeds_set": ["random", "numpy", "torch", "dgl"],
            "threads": NT,
            "epoch_integration": P["train"]["epoch_integration"],
            "gene_embedding_h5ad": str(gene_h5ad) if gene_h5ad.exists() else None,
            "input_dir": str(inputs),
            "wall_time_preprocess_s": round(t_pre, 1),
            "wall_time_s": round(wall, 1),
            "peak_rss_gb": C.peak_rss_gb(),
        })
    C.write_latent(dest, z, int(len(obs)), cfg)
    print(f"WALL_S={wall:.1f} PEAK_RSS_GB={C.peak_rss_gb()} DEST={dest}", flush=True)


if __name__ == "__main__":
    main()
