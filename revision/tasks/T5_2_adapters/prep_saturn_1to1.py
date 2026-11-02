"""Adapter B input builder (SATURN on the cohort's 1:1 orthologs). Run in saturn_env.

From the processed h5ad, one AnnData per species: same cells and order, raw counts of `.X`,
`var_names` = that species' symbol through `ortho_N` and the ortholog table. Drops genes with no
symbol, repeated symbols (first kept) and genes detected in fewer than 3 cells of that species
(same rule as Adapter A). `--hv-genes` is set above G in the run, so every remaining gene stays.

Writes <inputs>/<stem>/saturn_1to1/{<species>.h5ad, run.csv, manifest.json, gene_map.csv}.
gene_map.csv: var_name, hsapiens_gene_id and the symbol of each species, the join used in T5.6.
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
import scipy.sparse as sp  # noqa: E402

MIN_CELLS = 3
TABLE = {"liver": "ortholog_table.csv", "testis": "ortholog_table_testis.csv"}
SYMBOL = {"human": "hsapiens_gene_name", "mouse": "mmusculus_gene_name", "macaca_mulatta": "mmulatta_gene_name",
          "zebrafish": "drerio_gene_name"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stem", required=True)
    a = p.parse_args(argv)
    tissue = C.COHORTS[a.stem][1]
    tab = pd.read_csv(C.PROCESSED / TABLE[tissue]).set_index("ortholog_group_id")
    full = ad.read_h5ad(C.PROCESSED / f"{a.stem}.h5ad")
    ids = pd.Index(full.var_names.astype(str)).str.replace("ortho_", "", regex=False).astype(int)
    genes = pd.DataFrame({"var_name": full.var_names.astype(str), "hsapiens_gene_id": tab.loc[ids, "hsapiens_gene_id"].values})
    out = (C.INPUTS / a.stem / "saturn_1to1").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, man = [], {"stem": a.stem, "n_obs": int(full.n_obs), "species": {}}
    for sp_proc, (_, sat_sp) in C.species_map(a.stem).items():
        m = (full.obs["species"].astype(str) == sp_proc).to_numpy()
        X = sp.csr_matrix(full.X[m], dtype=np.float32)
        if not np.allclose(X.data, np.round(X.data)):
            raise SystemExit("processed .X is not integer counts")
        sym = pd.Series(tab.loc[ids, SYMBOL[sat_sp]].values, index=full.var_names.astype(str), dtype=object)
        genes[sat_sp] = sym.values
        keep = sym.notna().to_numpy() & ~sym.duplicated(keep="first").to_numpy()
        keep &= np.asarray((X > 0).sum(axis=0)).ravel() >= MIN_CELLS
        x = ad.AnnData(X=X[:, keep], obs=pd.DataFrame(index=full.obs_names[m].astype(str)),
                       var=pd.DataFrame(index=sym[keep].astype(str).values))
        x.obs["cell_type"] = full.obs["cell_type"].astype(str).values[m]
        x.obs["cell_ontology_class"] = full.obs["cell_ontology_class"].astype(str).values[m]
        x.obs["species_processed"] = sp_proc
        f = out / f"{sat_sp}.h5ad"
        x.write_h5ad(f)
        emb = C.ESM2_DIR / f"{sat_sp}_embedding.torch"
        rows.append({"path": str(f), "species": sat_sp, "embedding_path": str(emb)})
        man["species"][sat_sp] = {"species_processed": sp_proc, "n_cells": int(x.n_obs), "n_genes": int(x.n_vars),
                                  "n_dropped": int((~keep).sum()), "min_cells": MIN_CELLS,
                                  "embedding_sha256": C.sha256(emb), "input_sha256": C.sha256(f)}
        print(sat_sp, man["species"][sat_sp], flush=True)
    df = pd.DataFrame(rows).sort_values("species")
    df.to_csv(out / "run.csv", index=False)
    genes.to_csv(out / "gene_map.csv", index=False)
    man.update(run_csv_order=df["species"].tolist(), processed_sha256=C.sha256(C.PROCESSED / f"{a.stem}.h5ad"),
               git_sha=C.git_sha(), env=C.env_info())
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
