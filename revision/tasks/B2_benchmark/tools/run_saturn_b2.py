"""SATURN on a setting's object, the T5.2/T5.6 Adapter B call (unmodified train-saturn.py, published defaults plus pe_sim_penalty 1.0, every 1:1 gene), then S_deploy.
Run in saturn_env. S_deploy (the frozen primary): r_m = Pearson across cell types with >= 30 cells in both species of human and mouse type means of macrogene m; weights wbar = mean of
the two genes' final macrogene weights; S_deploy = sum wbar r / sum wbar over macrogenes with r defined. Writes results/revision/B2/tool_scores/SATURN/<run>.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import runpy
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
import sklearn  # noqa: E402,F401
import anndata as ad  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

torch.set_num_threads(NT)
D = dict(num_macrogenes=2000, hv_genes=40000, pe_sim_penalty=1.0, model_dim=256, hidden_dim=256, pretrain_epochs=200, epochs=50)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--root", default=None, help="input folder; default data/revision/B2/tool_inputs/<run>")
    p.add_argument("--label-col", default="cell_type")
    p.add_argument("--latent-out", default=None, help="integration benchmark: write the latent there and stop; no gene score")
    a = p.parse_args()
    root = Path(a.root).resolve() if a.root else C.REPO / "data/revision/B2/tool_inputs" / a.run  # absolute: train-saturn.py runs from its own folder
    out = C.REPO / "results/revision/B2/tool_scores/SATURN" / f"{a.run}.csv"
    if out.exists() or (a.latent_out and (Path(a.latent_out) / "z_mean.npy").exists()):
        return
    work = C.REPO / "data/revision/B2/tool_work" / f"saturn_{a.run}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cli = [f"--in_data={root / 'saturn/run.csv'}", f"--in_label_col={a.label_col}", "--ref_label_col=cell_type", "--embedding_model=ESM2",
           f"--num_macrogenes={D['num_macrogenes']}", f"--hv_genes={D['hv_genes']}", f"--pe_sim_penalty={D['pe_sim_penalty']}", f"--model_dim={D['model_dim']}",
           f"--hidden_dim={D['hidden_dim']}", f"--pretrain_epochs={D['pretrain_epochs']}", f"--epochs={D['epochs']}", f"--polling_freq={max(D['epochs'], D['pretrain_epochs']) + 1}",
           f"--centroids_init_path={work / f'centroids_s{a.seed}.pkl'}", f"--work_dir={work}/", f"--seed={a.seed}", "--device=cpu", "--device_num=-1"]
    t0 = time.time()
    old = os.getcwd()
    sys.argv = [str(C.SATURN_DIR / "train-saturn.py")] + cli
    sys.path.insert(0, str(C.SATURN_DIR))
    os.chdir(C.SATURN_DIR)
    with threadpool_limits(limits=NT):
        runpy.run_path(str(C.SATURN_DIR / "train-saturn.py"), run_name="__main__")
    os.chdir(old)
    res = work / "saturn_results"
    f = [x for x in glob.glob(str(res / "*.h5ad")) if not x.endswith("_pretrain.h5ad") and "_ep_" not in Path(x).name]
    if len(f) != 1:
        raise SystemExit(f"expected one final h5ad, found {f}")
    A = ad.read_h5ad(f[0])
    if a.latent_out:
        d = Path(a.latent_out)
        d.mkdir(parents=True, exist_ok=True)
        X = np.asarray(A.X, dtype=np.float64)
        np.save(d / "z_raw.npy", X)
        pd.Series(np.asarray(A.obs_names).astype(str)).to_csv(d / "obs_names.csv", index=False, header=False)
        np.save(d / "z_mean.npy", X / np.linalg.norm(X, axis=1, keepdims=True))  # T5.2: SATURN enters L2-normalised
        (d / "config.json").write_text(json.dumps(dict(run=a.run, seed=a.seed, label_col=a.label_col, wall_s=time.time() - t0, peak_rss_gb=C.peak_rss_gb(),
                                                       adapter="1:1 orthologs (T5.6 Adapter B)", tool_state=C.tool_state(C.SATURN_DIR)), indent=1))
        shutil.rmtree(work)
        return
    w = pickle.load(open(glob.glob(str(res / "*_genes_to_macrogenes.pkl"))[0], "rb"))
    gm = pd.read_csv(root / "gene_map.csv")
    kh, km = "human_" + gm.symbol_h.astype(str), "mouse_" + gm.symbol_m.astype(str)
    ok = np.array([x in w and y in w for x, y in zip(kh, km)])
    Wh = np.vstack([w[k] for k in kh[ok]]).astype(np.float64)
    Wm = np.vstack([w[k] for k in km[ok]]).astype(np.float64)
    # cell labels from the inputs (SATURN writes species_label)
    lab = {}
    for s in ("human", "mouse"):
        x = ad.read_h5ad(root / "saturn" / f"{s}.h5ad", backed="r")
        lab.update(dict(zip(x.obs_names, x.obs["cell_type"].astype(str))))
    labs = pd.Series(A.obs_names).map(lab).to_numpy()
    spc = A.obs["species"].astype(str).to_numpy()
    X = np.asarray(A.obsm["macrogenes"])
    t = pd.crosstab(labs, spc)
    types = t.index[(t >= 30).all(1)].tolist()
    if len(types) >= 2:
        H = np.column_stack([X[(spc == "human") & (labs == k)].mean(0) for k in types])
        M = np.column_stack([X[(spc == "mouse") & (labs == k)].mean(0) for k in types])
        H0, M0 = H - H.mean(1, keepdims=True), M - M.mean(1, keepdims=True)
        den = np.sqrt((H0 ** 2).sum(1) * (M0 ** 2).sum(1))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(den > 0, (H0 * M0).sum(1) / den, np.nan)
        good = np.isfinite(r)
        wbar = (Wh + Wm) / 2
        dep = (wbar[:, good] * r[good]).sum(1) / wbar[:, good].sum(1)
    else:
        dep = np.full(int(ok.sum()), np.nan)
    d = pd.DataFrame(dict(var_name=gm.var_name.to_numpy()[ok], score=dep))
    d["S_cos"] = (Wh * Wm).sum(1) / np.maximum(np.linalg.norm(Wh, axis=1) * np.linalg.norm(Wm, axis=1), 1e-300)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    Path(str(out) + ".json").write_text(json.dumps(dict(run=a.run, seed=a.seed, n_pairs=int(ok.sum()), k_types=len(types), wall_s=time.time() - t0, peak_rss_gb=C.peak_rss_gb(),
                                                        tool_state=C.tool_state(C.SATURN_DIR)), indent=1))
    shutil.rmtree(work)


if __name__ == "__main__":
    main()
