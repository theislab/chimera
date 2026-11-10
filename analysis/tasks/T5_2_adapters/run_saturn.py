"""Adapter A runner: SATURN native, CPU. Run in saturn_env.

Runs the unmodified `train-saturn.py` of the pinned clone in this process
(runpy), with published defaults plus `--pe_sim_penalty=1.0`, then aligns the
output to the processed h5ad by obs_names and writes z_mean.npy (L2-normalized
`.X`), z_raw.npy (raw `.X`) and config.json.

Thread control: train-saturn.py sets OMP_NUM_THREADS=4 / MKL_NUM_THREADS=6 at
import time. Numpy, torch and sklearn are imported here first with the thread
count of the allocation, so those assignments have no effect on the already
initialized thread pools. torch.set_num_threads is set explicitly.
"""
from __future__ import annotations

import argparse
import glob
import os
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
import sklearn  # noqa: E402,F401
import anndata as ad  # noqa: E402
from threadpoolctl import threadpool_limits, threadpool_info  # noqa: E402

torch.set_num_threads(NT)

DEFAULTS = dict(embedding_model="ESM2", num_macrogenes=2000, hv_genes=8000, pe_sim_penalty=1.0,
                model_dim=256, hidden_dim=256, pretrain_epochs=200, epochs=50)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stem", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labels", choices=["cell_type", "leiden_within"], default="cell_type")
    p.add_argument("--inputs", default=None, help="folder with run.csv (default data/analysis/T5_inputs/<stem>/saturn)")
    p.add_argument("--work", required=True, help="tool work dir")
    p.add_argument("--dest", default=None, help="final dir for z_mean.npy + config.json (default: --work/final)")
    p.add_argument("--pretrain-epochs", type=int, default=DEFAULTS["pretrain_epochs"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--centroids", default=None, help="centroids pickle to reuse or create")
    p.add_argument("--method-name", default=None)
    p.add_argument("--hv-genes", type=int, default=DEFAULTS["hv_genes"],
                   help="Adapter B sets this above the gene count so every 1:1 gene stays")
    a = p.parse_args(argv)

    inputs = (Path(a.inputs) if a.inputs else C.INPUTS / a.stem / "saturn").resolve()
    work = Path(a.work).resolve()
    if (work / "saturn_results").exists():
        raise SystemExit(f"refusing to overwrite {work}/saturn_results")
    work.mkdir(parents=True, exist_ok=True)
    dest = (Path(a.dest) if a.dest else work / "final").resolve()
    centroids = (Path(a.centroids) if a.centroids else work / f"centroids_s{a.seed}.pkl").resolve()
    method = a.method_name or ("saturn" if a.labels == "cell_type" else "saturn_leiden")

    cli = [
        f"--in_data={inputs / 'run.csv'}",
        f"--in_label_col={a.labels}",
        "--ref_label_col=cell_type",
        f"--embedding_model={DEFAULTS['embedding_model']}",
        f"--num_macrogenes={DEFAULTS['num_macrogenes']}",
        f"--hv_genes={a.hv_genes}",
        f"--pe_sim_penalty={DEFAULTS['pe_sim_penalty']}",
        f"--model_dim={DEFAULTS['model_dim']}",
        f"--hidden_dim={DEFAULTS['hidden_dim']}",
        f"--pretrain_epochs={a.pretrain_epochs}",
        f"--epochs={a.epochs}",
        f"--polling_freq={max(a.epochs, a.pretrain_epochs) + 1}",
        f"--centroids_init_path={centroids}",
        f"--work_dir={work}/",
        f"--seed={a.seed}",
        "--device=cpu",
        "--device_num=-1",
    ]
    print("train-saturn.py", " ".join(cli), flush=True)
    t0 = time.time()
    old_argv, old_cwd = sys.argv, os.getcwd()
    sys.argv = [str(C.SATURN_DIR / "train-saturn.py")] + cli
    sys.path.insert(0, str(C.SATURN_DIR))
    os.chdir(C.SATURN_DIR)
    import runpy
    with threadpool_limits(limits=NT):
        runpy.run_path(str(C.SATURN_DIR / "train-saturn.py"), run_name="__main__")
    os.chdir(old_cwd)
    sys.argv = old_argv
    wall = time.time() - t0
    pools = [{k: d.get(k) for k in ("internal_api", "num_threads")} for d in threadpool_info()]

    # ---- align ----
    res = work / "saturn_results"
    finals = [f for f in glob.glob(str(res / "*.h5ad"))
              if not f.endswith("_pretrain.h5ad") and "_ep_" not in Path(f).name]
    if len(finals) != 1:
        raise SystemExit(f"expected one final h5ad, found {finals}")
    out = ad.read_h5ad(finals[0])
    obs, n_vars = C.processed_obs(a.stem)
    if inputs != (C.INPUTS / a.stem / "saturn").resolve():  # subsample (smoke): restrict to input cells
        keep = []
        for f in pd.read_csv(inputs / "run.csv")["path"]:
            keep += list(ad.read_h5ad(f, backed="r").obs_names)
        obs = obs.loc[obs.index.isin(keep)]
    if out.obs_names.duplicated().any():
        raise SystemExit("duplicate obs names in SATURN output")
    if set(out.obs_names) != set(obs.index):
        raise SystemExit("SATURN output cells != processed cells")
    out = out[obs.index].copy()
    smap = {v[1]: k for k, v in C.species_map(a.stem).items()}
    sat_species = out.obs["species"].astype(str).values
    if not (pd.Series(sat_species).map(smap).values == obs["species"].values).all():
        raise SystemExit("species mismatch after alignment")
    lab = out.obs["labels"].astype(str).values
    stripped = np.array([l[len(s) + 1:] if l.startswith(s + "_") else None for l, s in zip(lab, sat_species)],
                        dtype=object)
    if a.labels == "cell_type":
        expected = obs["cell_type"].values
    else:
        exp = {}
        for f in pd.read_csv(inputs / "run.csv")["path"]:
            x = ad.read_h5ad(f, backed="r")
            exp.update(dict(zip(x.obs_names, x.obs["leiden_within"].astype(str))))
            x.file.close()
        expected = obs.index.map(exp).values
    agree = float(np.mean(stripped == expected))
    print(f"label agreement after alignment: {agree:.6f}", flush=True)
    if agree != 1.0:
        raise SystemExit(f"label agreement {agree} != 1.0; refusing to save")

    X = np.asarray(out.X, dtype=np.float64)
    z = X / np.linalg.norm(X, axis=1, keepdims=True)
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "z_raw.npy", X.astype(np.float32))
    shutil.copy(inputs / "manifest.json", dest / "input_manifest.json")
    cfg = C.base_config(
        stem=a.stem, n_obs=int(len(obs)), n_vars=n_vars, method=method, seed=a.seed,
        labels_key=a.labels, n_latent=int(z.shape[1]),
        extra={
            "pkg_versions": C.pkg_versions(["torch", "torchvision", "scanpy", "anndata", "scvi-tools",
                                            "scikit-learn", "numpy", "pandas", "scipy", "scikit-misc"]),
            "env": C.env_info(),
            "tool": "SATURN", "tool_repo": "snap-stanford/SATURN", "tool_state": C.tool_state(C.SATURN_DIR),
            "tool_cli": ["train-saturn.py"] + cli,
            "labels_used_in_training": f"{a.labels}, within species only (SATURN species-prefixed labels)",
            "label_agreement_after_alignment": agree,
            "gene_input": ("1:1 orthologs of the processed h5ad (Adapter B), ESM2-filtered, all kept"
                           if method.endswith("1to1") else
                           "tool native symbols, ESM2-filtered, seurat_v3 HVG 8000 per species"),
            "latent": "L2-normalised .X of the final SATURN h5ad; raw .X in z_raw.npy",
            "set_detect_anomaly": "True (unmodified clone)" if C.tool_state(C.SATURN_DIR)["local_diff"] is None
            else "see tool_state.local_diff",
            "kmeans_n_init": "sklearn default ('auto' -> 1 for k-means++) in scikit-learn "
                             + C.pkg_versions(["scikit-learn"]).get("scikit-learn", "?"),
            "threads": NT, "threadpools": pools,
            "pretrain_epochs": a.pretrain_epochs, "epochs": a.epochs,
            "centroids_file": str(centroids),
            "tool_output_h5ad": finals[0],
            "input_dir": str(inputs),
            "wall_time_s": round(wall, 1),
            "peak_rss_gb": C.peak_rss_gb(),
        })
    C.write_latent(dest, z, int(len(obs)), cfg)
    print(f"WALL_S={wall:.1f} PEAK_RSS_GB={C.peak_rss_gb()} DEST={dest}", flush=True)


if __name__ == "__main__":
    main()
