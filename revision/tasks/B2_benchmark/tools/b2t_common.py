"""Shared helpers of the B2 gene-level tool adapters.

Standard library, numpy, pandas, scipy and h5py only, so it loads in every tool env. The h5ad is
read with h5py, because the tool envs carry anndata versions that cannot read every recent file.
Gene identity goes through revision/common/gene_ids.py, loaded by path.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TOOLS = REPO.parent / "external_tools"
SPECIES = ("Human", "Mouse")
TABLE_ID = {"Human": "hsapiens_gene_id", "Mouse": "mmusculus_gene_id"}
TABLE_SYMBOL = {"Human": "hsapiens_gene_name", "Mouse": "mmusculus_gene_name"}


def gene_ids_module():
    spec = importlib.util.spec_from_file_location("gene_ids", REPO / "revision" / "common" / "gene_ids.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gene_ids"] = mod
    spec.loader.exec_module(mod)
    return mod


def _read_col(group, key):
    """One obs/var column as an object array of str, with None for missing."""
    node = group[key]
    if isinstance(node, h5py.Group):  # categorical, or nullable with mask
        if "categories" in node:
            cats = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in node["categories"][()]], dtype=object)
            codes = node["codes"][()]
            out = np.empty(codes.shape, dtype=object)
            ok = codes >= 0
            out[ok] = cats[codes[ok]]
            out[~ok] = None
            return out
        vals = node["values"][()]
        mask = node["mask"][()]
        out = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in vals], dtype=object)
        out[mask] = None
        return out
    vals = node[()]
    if "categories" in node.attrs:  # legacy categorical
        cats = group[node.attrs["categories"]][()]
        cats = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in cats], dtype=object)
        out = np.empty(vals.shape, dtype=object)
        ok = vals >= 0
        out[ok] = cats[vals[ok]]
        out[~ok] = None
        return out
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in vals], dtype=object)


def _index(group):
    key = group.attrs.get("_index", "_index")
    return _read_col(group, key).astype(str)


def read_h5ad(path, label_key):
    """Return X (csr, float64 counts), var_names, species, labels (None if missing)."""
    with h5py.File(path, "r") as f:
        obs, var = f["obs"], f["var"]
        species = _read_col(obs, "species")
        labels = _read_col(obs, label_key)
        var_names = _index(var)
        x = f["X"]
        n_obs, n_var = species.size, var_names.size
        if isinstance(x, h5py.Group):
            enc = x.attrs.get("encoding-type", x.attrs.get("h5sparse_format", "csr_matrix"))
            enc = enc.decode() if isinstance(enc, bytes) else str(enc)
            cls = sp.csc_matrix if enc.startswith("csc") else sp.csr_matrix
            X = cls((x["data"][()], x["indices"][()], x["indptr"][()]), shape=(n_obs, n_var))
        else:
            X = sp.csr_matrix(x[()])
    X = sp.csr_matrix(X, dtype=np.float64)
    X.eliminate_zeros()
    if X.nnz and not np.allclose(X.data, np.round(X.data)):
        raise SystemExit(f"{path}: X is not integer counts")
    if not pd.Index(var_names).str.fullmatch(r"ortho_\d+").all():
        raise SystemExit(f"{path}: var_names are not ortho_N")
    if not pd.Index(var_names).is_unique:
        raise SystemExit(f"{path}: var_names are not unique")
    found = set(species[pd.notna(species)])
    if found != set(SPECIES):
        raise SystemExit(f"{path}: species must be exactly {SPECIES}, found {sorted(found)}")
    missing = pd.isna(labels) | np.isin(labels.astype(str), ["nan", "None", "NA", ""])
    labels = labels.copy()
    labels[missing] = None
    return X, var_names, species, labels


def ortholog_map(cohort, var_names):
    """Per var_name the Ensembl gene id and symbol of each species, from the cohort's table."""
    gi = gene_ids_module()
    tab = gi.load_table(cohort)
    ids = pd.Index(var_names).str.replace("ortho_", "", regex=False).astype(int)
    out = pd.DataFrame({"var_name": var_names})
    for sp_name in SPECIES:
        out[f"id_{sp_name.lower()}"] = pd.Series(ids.map(tab[TABLE_ID[sp_name]]).values, dtype=object).values
        out[f"symbol_{sp_name.lower()}"] = pd.Series(ids.map(tab[TABLE_SYMBOL[sp_name]]).values, dtype=object).values
    return out


def n_cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)


def set_threads(n):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = str(n)


def peak_rss_gb():
    a = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    b = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(max(a, b) / 1024 ** 2, 3)


def tool_state(folder):
    def run(*cmd):
        return subprocess.check_output(cmd, cwd=str(folder), stderr=subprocess.DEVNULL).decode().strip()
    try:
        return {"commit": run("git", "rev-parse", "HEAD"), "local_diff": run("git", "diff", "--stat") or None}
    except Exception:
        return {"commit": None, "local_diff": None}


def label_table(species, labels):
    """Cells per label per species, as a nested dict."""
    df = pd.DataFrame({"species": species, "label": labels}).dropna()
    return {s: {k: int(v) for k, v in g["label"].value_counts().sort_index().items()} for s, g in df.groupby("species")}


def write_outputs(out, df, *, tool, tool_info, primary, params, args, t0, n_input_genes, extra=None):
    """Write the csv (var_name, score, secondary columns) and the sidecar <out>.json."""
    import time
    if list(df.columns[:2]) != ["var_name", "score"]:
        raise SystemExit("first columns must be var_name, score")
    if not df["var_name"].is_unique:
        raise SystemExit("duplicate var_name in the output")
    if not np.isfinite(df["score"].to_numpy(dtype=float)).all():
        raise SystemExit("non-finite score in the output")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    side = {
        "tool": tool,
        "tool_info": tool_info,
        "primary_scalar": primary,
        "parameters": params,
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_input_genes": int(n_input_genes),
        "n_genes_covered": int(len(df)),
        "score_min": float(df["score"].min()) if len(df) else None,
        "score_max": float(df["score"].max()) if len(df) else None,
        "n_distinct_scores": int(df["score"].nunique()),
        "runtime_s": round(time.time() - t0, 1),
        "peak_rss_gb": peak_rss_gb(),
        "cpus": n_cpus(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "env": Path(sys.prefix).name,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    side.update(extra or {})
    Path(str(out) + ".json").write_text(json.dumps(side, indent=2, default=str))
    print(f"wrote {out}: {len(df)} genes, {side['runtime_s']} s, peak {side['peak_rss_gb']} GB", flush=True)
