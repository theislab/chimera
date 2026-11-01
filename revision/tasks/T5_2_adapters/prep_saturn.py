"""Adapter A input builder (SATURN native genes). Run in saturn_env.

For one cohort, per species: read the raw CAMEX h5ad, keep exactly the cells of
the processed cohort (processed order), copy `cell_type`, `species`,
`cell_ontology_class` from the processed obs, keep raw counts and native gene
symbols, drop duplicate symbols (first kept) and genes with zero counts in these
cells, and add `leiden_within` (within-species Leiden: normalize_total 1e4,
log1p, 2000 HVG, PCA 50, k=15, resolution 1.0, random_state 0).

Writes <inputs>/<stem>/saturn/{<species>.h5ad, run.csv, manifest.json}.
run.csv rows are sorted alphabetically by SATURN species name (SATURN issue
#83: obs names follow CSV order, matrices follow sorted species).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cohorts as C  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
import scipy.sparse as sp  # noqa: E402


def leiden_within(a: ad.AnnData) -> pd.Series:
    b = a.copy()
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    sc.pp.highly_variable_genes(b, n_top_genes=min(2000, b.n_vars), subset=True)
    sc.tl.pca(b, n_comps=min(50, b.n_obs - 1, b.n_vars - 1), random_state=0)
    sc.pp.neighbors(b, n_neighbors=15, random_state=0)
    sc.tl.leiden(b, resolution=1.0, random_state=0, key_added="leiden")
    return ("L" + b.obs["leiden"].astype(str)).reindex(a.obs_names)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stem", required=True)
    p.add_argument("--out-root", default=str(C.INPUTS))
    p.add_argument("--subsample", type=int, default=0, help="total cells, smoke test only")
    a = p.parse_args(argv)

    obs, _ = C.processed_obs(a.stem)
    if a.subsample:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(obs), size=a.subsample, replace=False)
        obs = obs.iloc[np.sort(keep)]
    out = Path(a.out_root) / a.stem / "saturn"
    out.mkdir(parents=True, exist_ok=True)

    rows, man = [], {"stem": a.stem, "n_obs": int(len(obs)), "subsample": a.subsample, "species": {}}
    for sp_proc, (raw_rel, sat_sp) in C.species_map(a.stem).items():
        o = obs[obs["species"] == sp_proc]
        raw_path = C.RAW_CAMEX / raw_rel
        r = ad.read_h5ad(raw_path)
        missing = (~o.index.isin(r.obs_names)).sum()
        if missing:
            raise SystemExit(f"{a.stem}/{sp_proc}: {missing} processed cells absent from {raw_path}")
        r = r[o.index].copy()
        r.var_names = r.var_names.astype(str)
        n_dup = int(r.var_names.duplicated().sum())
        r = r[:, ~r.var_names.duplicated(keep="first")].copy()
        X = sp.csr_matrix(r.X, dtype=np.float32)
        if not np.allclose(X.data, np.round(X.data)):
            raise SystemExit(f"{raw_path} is not integer counts")
        nz = np.asarray((X > 0).sum(axis=0)).ravel() > 0
        x = ad.AnnData(X=X[:, nz], obs=pd.DataFrame(index=r.obs_names.astype(str)),
                       var=pd.DataFrame(index=r.var_names[nz]))
        x.obs["cell_type"] = o["cell_type"].values
        x.obs["cell_ontology_class"] = o["cell_ontology_class"].values
        x.obs["species_processed"] = sp_proc
        x.obs["leiden_within"] = leiden_within(x).values
        f = out / f"{sat_sp}.h5ad"
        x.write_h5ad(f)
        emb = C.ESM2_DIR / f"{sat_sp}_embedding.torch"
        rows.append({"path": str(f), "species": sat_sp, "embedding_path": str(emb)})
        man["species"][sat_sp] = {
            "species_processed": sp_proc, "n_cells": int(x.n_obs), "n_genes": int(x.n_vars),
            "dropped_duplicate_symbols": n_dup, "dropped_zero_genes": int((~nz).sum()),
            "n_cell_type": int(x.obs["cell_type"].nunique()),
            "n_leiden_within": int(x.obs["leiden_within"].nunique()),
            "raw_file": str(raw_path), "raw_sha256": C.sha256(raw_path),
            "embedding_file": str(emb), "embedding_sha256": C.sha256(emb),
            "input_sha256": C.sha256(f),
        }
        print(sat_sp, man["species"][sat_sp], flush=True)
    df = pd.DataFrame(rows).sort_values("species")
    df.to_csv(out / "run.csv", index=False)
    man["run_csv_order"] = df["species"].tolist()
    man["git_sha"] = C.git_sha()
    man["env"] = C.env_info()
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
