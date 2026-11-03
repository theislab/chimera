"""CAMEX on a setting's object: authors' parameters (liver file; CPU; integration only, labels unused in training), the 1:1 orthologs as homology, then C_cos = cos(h_human, h_mouse)
of the gene embeddings, the frozen primary. Run in camex_env. Writes results/revision/B2/tool_scores/CAMEX/<run>.csv
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "T5_2_adapters"))
import cohorts as C  # noqa: E402

NT = C.n_threads()
C.set_thread_env(NT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NT)
sys.path.insert(0, str(C.CAMEX_DIR))
import dgl  # noqa: E402
import anndata as ad  # noqa: E402
from CAMEX.base import Dataset  # noqa: E402
from CAMEX.trainer import Trainer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--root", default=None, help="input folder; default data/revision/B2/tool_inputs/<run>")
    p.add_argument("--latent-out", default=None, help="integration benchmark: write the cell latent there and stop; no gene score")
    a = p.parse_args()
    root = Path(a.root).resolve() if a.root else C.REPO / "data/revision/B2/tool_inputs" / a.run  # absolute: train-saturn.py runs from its own folder
    out = C.REPO / "results/revision/B2/tool_scores/CAMEX" / f"{a.run}.csv"
    if out.exists() or (a.latent_out and (Path(a.latent_out) / "z_mean.npy").exists()):
        return
    work = C.REPO / "data/revision/B2/tool_work" / f"camex_{a.run}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("camex_params", C.CAMEX_DIR / "analysis" / "1liver" / "params.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    P = copy.deepcopy(m.PARAMS)
    others = sorted(f.stem.split("-")[2] for f in (root / "camex").glob("raw-b2-*-x.h5ad") if "-human-" not in f.name)
    P["preprocess"]["dataset_file"] = pd.DataFrame([["raw-b2-human-x.h5ad", True, f"gene_matches_human2{o}.csv", f"raw-b2-{o}-x.h5ad", False] for o in others],
                                                   columns=["source", "source label", "relationship", "destination", "destination label"])
    P["preprocess"]["path"] = str(root / "camex") + "/"
    P["train"]["device"] = "cpu"
    for v in P.values():
        v["log_path"] = str(work) + "/"
        v["time_start"] = time.strftime("%Y-%m-%d-%H-%M-%S")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dgl.seed(a.seed)
    t0 = time.time()
    dataset = Dataset(**P["preprocess"])
    trainer = Trainer(dataset.adata_whole, dataset.dgl_data, **P["train"])
    trainer.integration()
    if a.latent_out:
        d = Path(a.latent_out)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "z_mean.npy", np.asarray(dataset.adata_whole.obsm["X_CAMEX_Integration"]))
        pd.Series(np.asarray(dataset.adata_whole.obs_names).astype(str)).to_csv(d / "obs_names.csv", index=False, header=False)
        (d / "config.json").write_text(json.dumps(dict(run=a.run, seed=a.seed, wall_s=time.time() - t0, peak_rss_gb=C.peak_rss_gb(),
                                                       adapter="1:1 orthologs, authors' liver parameters", tool_state=C.tool_state(C.CAMEX_DIR)), indent=1))
        shutil.rmtree(work)
        return
    g = ad.read_h5ad(work / "gene_integration_hidden.h5ad")
    g.obs_names = g.obs_names.astype(str)
    gene = np.asarray(g.obs_names)
    b = g.obs["batch"].astype(str).to_numpy()
    X = np.asarray(g.X, dtype=np.float64)
    dh, dm = "raw-b2-human-x", "raw-b2-mouse-x"
    pos_h = {k: i for i, k in zip(np.where(b == dh)[0], gene[b == dh])}
    pos_m = {k: i for i, k in zip(np.where(b == dm)[0], gene[b == dm])}
    if not pos_h or not pos_m:
        raise SystemExit(f"dataset names not found in {sorted(set(b))}")
    gm = pd.read_csv(root / "gene_map.csv")
    ih, im = gm.symbol_h.astype(str).map(pos_h), gm.symbol_m.astype(str).map(pos_m)
    ok = (ih.notna() & im.notna()).to_numpy()
    ih, im = ih[ok].astype(int).to_numpy(), im[ok].astype(int).to_numpy()
    co = (X[ih] * X[im]).sum(1) / np.maximum(np.linalg.norm(X[ih], axis=1) * np.linalg.norm(X[im], axis=1), 1e-300)
    d = pd.DataFrame(dict(var_name=gm.var_name.to_numpy()[ok], score=co))
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    Path(str(out) + ".json").write_text(json.dumps(dict(run=a.run, seed=a.seed, n_pairs=int(ok.sum()), wall_s=time.time() - t0, peak_rss_gb=C.peak_rss_gb(), tool_state=C.tool_state(C.CAMEX_DIR)), indent=1))
    shutil.rmtree(work)


if __name__ == "__main__":
    main()
