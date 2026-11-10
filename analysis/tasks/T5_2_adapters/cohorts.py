"""Cohort table and shared helpers for the SATURN and CAMEX adapters.

Imports only the standard library, numpy and pandas, so it loads in every
tool env (saturn_env, camex_env, revision_de_env). Never imports chimera.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WORKSPACE = REPO.parent
TOOLS = WORKSPACE / "external_tools"
SATURN_DIR = TOOLS / "SATURN"
CAMEX_DIR = TOOLS / "CAMEX"
SATURN_COMMIT = "6906abf23728aa8ebf400cedf4ce38e4821a0a49"
CAMEX_COMMIT = "17053c956de528aec88878d4493d94838a668d6e"

PROCESSED = REPO / "data" / "processed"
RAW_CAMEX = REPO / "data" / "raw" / "camex"
ESM2_DIR = REPO / "data" / "raw" / "saturn_esm2" / "protein_embeddings_export" / "ESM2"
INPUTS = REPO / "data" / "analysis" / "T5_inputs"
RUNS = REPO / "training" / "runs"
RESULTS = REPO / "results" / "analysis"

# Processed `species` value -> (raw CAMEX file, SATURN species name).
LIVER = {
    "Human": ("1liver/dataset/raw-liver-human-Martin.h5ad", "human"),
    "Monkey": ("1liver/dataset/raw-liver-monkey-Martin.h5ad", "macaca_mulatta"),
    "Mouse": ("1liver/dataset/raw-liver-mouse-Martin.h5ad", "mouse"),
    "Zebrafish": ("1liver/dataset/raw-liver-zebrafish-ggj5.h5ad", "zebrafish"),
}
TESTIS = {
    "Human": ("2testis/dataset/raw-testis-human-Murat.h5ad", "human"),
    "Macaque": ("2testis/dataset/raw-testis-macaque-Murat.h5ad", "macaca_mulatta"),
    "Mouse": ("2testis/dataset/raw-testis-mouse-Murat.h5ad", "mouse"),
}
# Kidney H+Mouse (P13 healthy training object). No authors' raw files: B1 builds tool inputs from the object itself.
KIDNEY = {
    "Human": ("kidney/raw-kidney-human-KPMP.h5ad", "human"),
    "Mouse": ("kidney/raw-kidney-mouse-KPMP.h5ad", "mouse"),
}
PROCESSED_OVERRIDE = {"kidney_healthy_train": REPO / "data" / "analysis" / "disease_kidney" / "kidney_healthy_train.h5ad"}
# CAMEX homology files, relative to the tissue's dataset folder (human hub).
CAMEX_REL = {
    ("liver", "Monkey"): "gene_matches_human2monkey.csv",
    ("liver", "Mouse"): "gene_matches_human2mouse.csv",
    ("liver", "Zebrafish"): "gene_matches_human2zebrafish.csv",
    ("testis", "Macaque"): "gene_matches_human2macaque.csv",
    ("testis", "Mouse"): "gene_matches_human2mouse.csv",
}

# stem -> (cohort key, tissue, species in processed h5ad)
COHORTS = {
    "camex_human_monkey": ("liv_hmonkey", "liver", ["Human", "Monkey"]),
    "camex_human_mouse": ("liv_hmouse", "liver", ["Human", "Mouse"]),
    "camex_human_zebrafish": ("liv_hzeb", "liver", ["Human", "Zebrafish"]),
    "camex_pancreas_unified": ("liv_4sp", "liver", ["Human", "Monkey", "Mouse", "Zebrafish"]),
    "camex_testis_human_macaque": ("test_hmac", "testis", ["Human", "Macaque"]),
    "camex_testis_human_mouse": ("test_hmouse", "testis", ["Human", "Mouse"]),
    "camex_testis_unified": ("test_3sp", "testis", ["Human", "Macaque", "Mouse"]),
    "kidney_healthy_train": ("kidney", "kidney", ["Human", "Mouse"]),
}
KEY_TO_STEM = {v[0]: k for k, v in COHORTS.items()}


def species_map(stem: str) -> dict:
    tissue = COHORTS[stem][1]
    table = {"liver": LIVER, "testis": TESTIS, "kidney": KIDNEY}[tissue]
    return {sp: table[sp] for sp in COHORTS[stem][2]}


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_sha(repo: Path = REPO) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def git_dirty(repo: Path = REPO, paths=("analysis/tasks/T5_2_adapters", "analysis/tasks/T5_3_runs")) -> bool | None:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--", *paths], cwd=str(repo),
                                      stderr=subprocess.DEVNULL).decode().strip()
        return bool(out)
    except Exception:
        return None


def tool_state(tool_dir: Path) -> dict:
    """Commit and local modifications of a tool clone."""
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(tool_dir)).decode().strip()
        diff = subprocess.check_output(["git", "diff", "--stat"], cwd=str(tool_dir)).decode().strip()
    except Exception:
        head, diff = None, None
    return {"commit": head, "local_diff": diff or None}


def env_info() -> dict:
    prefix = Path(sys.prefix)
    freeze = REPO / "analysis" / "envs" / f"{prefix.name}.pip_freeze.txt"
    return {
        "env_name": prefix.name,
        "python": sys.version.split()[0],
        "pip_freeze_file": str(freeze.relative_to(REPO)) if freeze.exists() else None,
        "pip_freeze_sha256": sha256(freeze) if freeze.exists() else None,
    }


def pkg_versions(names) -> dict:
    from importlib.metadata import version
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except Exception:
            pass
    return out


def peak_rss_gb() -> float:
    """Peak RSS of this process and its waited-for children, in GB."""
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(max(self_kb, child_kb) / 1024 ** 2, 3)


def n_threads() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("OMP_NUM_THREADS") or 1)


def set_thread_env(n: int) -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = str(n)


def write_latent(out_dir: Path, z, n_obs: int, config: dict) -> Path:
    """Write z_mean.npy and config.json in baseline_runner's schema."""
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] != n_obs:
        raise ValueError(f"latent shape {z.shape} != ({n_obs}, d)")
    if not np.isfinite(z).all():
        raise ValueError("latent has NaN/Inf")
    np.save(out_dir / "z_mean.npy", z)
    cfg = dict(config)
    cfg["latent_shape"] = list(z.shape)
    cfg["latent_dtype"] = str(z.dtype)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    return out_dir / "z_mean.npy"


def base_config(*, stem: str, n_obs: int, n_vars: int, method: str, seed: int,
                labels_key, n_latent: int, extra: dict) -> dict:
    """Same leading fields as training/baseline_runner.py::_base_config."""
    import datetime
    cfg = {
        "adata": str((PROCESSED / f"{stem}.h5ad").resolve()),
        "adata_stem": stem,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "method": method,
        "seed": seed,
        "species_key": "species",
        "batch_key": None,
        "labels_key": labels_key,
        "n_latent": n_latent,
        "n_latent_actual": n_latent,
        "n_latent_override": False,
        "n_latent_skip_reason": "method_default",
        "n_hidden": None,
        "max_epochs": None,
        "max_epochs_override": False,
        "device": "cpu",
        "torch_num_threads": n_threads(),
        "torch_num_interop_threads": None,
        "git_sha": git_sha(),
        "git_dirty_task_code": git_dirty(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": os.uname().nodename,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    cfg.update(extra)
    return cfg


def processed_obs(stem: str):
    """obs of the processed h5ad (cell order of record), read in backed mode."""
    import anndata as ad
    a = ad.read_h5ad(PROCESSED_OVERRIDE.get(stem, PROCESSED / f"{stem}.h5ad"), backed="r")
    obs = a.obs[["species", "cell_type", "cell_ontology_class"]].copy()
    n_vars = a.n_vars
    a.file.close()
    for c in obs.columns:
        obs[c] = obs[c].astype(str)
    return obs, n_vars
