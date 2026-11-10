"""Adapter C input builder (CAMEX). Run in camex_env.

Per species: the raw CAMEX h5ad subset to the processed cohort's cells
(processed order), same filename, obs and var untouched. The authors'
`gene_matches_human2*.csv` files are copied unchanged. Dataset names keep the
`raw-<tissue>-<species>-<study>` form (no underscores: CAMEX splits graph keys
on '_').

Writes <inputs>/<stem>/camex/{raw-*.h5ad, gene_matches_*.csv, manifest.json}.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cohorts as C  # noqa: E402

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stem", required=True)
    p.add_argument("--out-root", default=str(C.INPUTS))
    p.add_argument("--subsample", type=int, default=0, help="total cells, smoke test only")
    a = p.parse_args(argv)

    obs, _ = C.processed_obs(a.stem)
    if a.subsample:
        rng = np.random.default_rng(0)
        obs = obs.iloc[np.sort(rng.choice(len(obs), size=a.subsample, replace=False))]
    tissue = C.COHORTS[a.stem][1]
    out = Path(a.out_root) / a.stem / "camex"
    out.mkdir(parents=True, exist_ok=True)
    man = {"stem": a.stem, "n_obs": int(len(obs)), "subsample": a.subsample, "files": {}}
    for sp_proc, (raw_rel, _) in C.species_map(a.stem).items():
        o = obs[obs["species"] == sp_proc]
        raw_path = C.RAW_CAMEX / raw_rel
        r = ad.read_h5ad(raw_path)
        if (~o.index.isin(r.obs_names)).sum():
            raise SystemExit(f"{a.stem}/{sp_proc}: processed cells missing from raw")
        r = r[o.index].copy()
        if "cell_ontology_class" not in r.obs:
            raise SystemExit(f"{raw_path}: no cell_ontology_class")
        f = out / raw_path.name
        r.write_h5ad(f)
        man["files"][f.name] = {"species_processed": sp_proc, "n_cells": int(r.n_obs), "n_genes": int(r.n_vars),
                                "raw_file": str(raw_path), "raw_sha256": C.sha256(raw_path),
                                "input_sha256": C.sha256(f)}
        if sp_proc != "Human":
            rel = raw_path.parent / C.CAMEX_REL[(tissue, sp_proc)]
            shutil.copy(rel, out / rel.name)
            man["files"][rel.name] = {"sha256": C.sha256(out / rel.name)}
        print(f.name, man["files"][f.name], flush=True)
    man["git_sha"] = C.git_sha()
    man["env"] = C.env_info()
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
