"""Build one tool row's input in chimera_env: the setting's object, and per-species files for the T5 pipelines on the 1:1 orthologs (Adapter B of T5.6).
    data/revision/B2/tool_inputs/<run>/object.h5ad            obs b2_label, b2_library, species; X counts
    .../saturn/{human,mouse}.h5ad, run.csv                    symbols as var_names, obs cell_type = b2_label; min 3 cells per gene (Adapter A rule)
    .../camex/raw-b2-{human,mouse}-x.h5ad, gene_matches_human2mouse.csv   obs cell_ontology_class = b2_label
    .../gene_map.csv                                          var_name, symbol_h, symbol_m
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

from revision.common import gene_ids
from revision.tasks.B2_benchmark import perturb as P
from revision.tasks.B2_benchmark import rules as RU

ROOT = RU.REPO / "data" / "revision" / "B2" / "tool_inputs"
ESM2 = RU.REPO / "data" / "raw" / "saturn_esm2" / "protein_embeddings_export" / "ESM2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    row = pd.read_csv(a.manifest, sep="\t", keep_default_na=False).iloc[a.index].to_dict()
    out = ROOT / row["run"]
    (out / "saturn").mkdir(parents=True, exist_ok=True)
    (out / "camex").mkdir(parents=True, exist_ok=True)
    if (out / "saturn/run.csv").exists() and (out / "camex/gene_matches_human2mouse.csv").exists() and (out / "gene_map.csv").exists():
        return
    b, info = P.build(row, check=False)
    tab = gene_ids.load_table(RU.ALL_COHORTS[row["cohort"]]["gene_table"])
    ids = pd.Index(b.var_names.astype(str)).str.replace("ortho_", "", regex=False).astype(int)
    sh, sm = tab.loc[ids, "hsapiens_gene_name"].astype(object).values, tab.loc[ids, "mmusculus_gene_name"].astype(object).values
    b.write_h5ad(out / "object.tmp.h5ad")
    (out / "object.tmp.h5ad").replace(out / "object.h5ad")
    keep_pair = pd.notna(sh) & pd.notna(sm)
    # repeated symbols: first kept, as Adapter B
    dup = pd.Series(sh).duplicated().to_numpy() | pd.Series(sm).duplicated().to_numpy()
    keep_pair &= ~dup
    spc = b.obs["species"].astype(str).to_numpy()
    for s, sym, sat, cam in (("Human", sh, "human", "human"), ("Mouse", sm, "mouse", "mouse")):
        m = spc == s
        x = b[m][:, keep_pair].copy()
        x.var_names = pd.Index(sym[keep_pair].astype(str))
        x.var = x.var[[]]
        x.obs["cell_type"] = x.obs["b2_label"].astype(str)
        x.obs["cell_ontology_class"] = x.obs["b2_label"].astype(str)
        x.obs["species"] = sat
        for k in list(x.uns):
            del x.uns[k]
        for k in list(x.varm):
            del x.varm[k]
        x.X = sp.csr_matrix(x.X, dtype=np.float32)
        y = x[:, np.asarray((x.X > 0).sum(0)).ravel() >= 3].copy()
        y.write_h5ad(out / "saturn" / f"{sat}.h5ad")
        x.write_h5ad(out / "camex" / f"raw-b2-{cam}-x.h5ad")
    pd.DataFrame([dict(path=str(out / "saturn" / f"{k}.h5ad"), species=k, embedding_path=str(ESM2 / f"{k}_embedding.torch")) for k in ("human", "mouse")]).to_csv(out / "saturn" / "run.csv", index=False)
    pd.DataFrame({"raw-b2-human-x.h5ad": sh[keep_pair], "raw-b2-mouse-x.h5ad": sm[keep_pair]}).to_csv(out / "camex" / "gene_matches_human2mouse.csv", index=False)
    pd.DataFrame(dict(var_name=b.var_names.astype(str)[keep_pair], symbol_h=sh[keep_pair], symbol_m=sm[keep_pair])).to_csv(out / "gene_map.csv", index=False)
    print(info, flush=True)


if __name__ == "__main__":
    main()
