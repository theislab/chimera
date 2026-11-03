"""SAMap adapter (atarashansky/SAMap, sc-samap). Run in samap_env.

PRIMARY SCALAR, declared before any comparison:
    score = weight of the human-mouse edge of the ortholog pair in the final refined homology graph
    (SAMAP.gnnm_refined, the "correlation" value of SAMAP.query_gene_pair). Range 0-1, higher = more
    conserved. It is the Pearson correlation of the pair's expression on the stitched manifold,
    negative values set to 0; bit scores select the candidate edges but do not enter the weight.
    Reason: it is the gene-pair score the tool's vignette offers for the expression of a homolog pair.

A gene is covered if both genes of the pair are in the dataset and the pair has an edge in the BLAST
graph (reciprocal hit, e-value 1e-6, within 25% of the best hit of a gene). A covered pair whose
refined edge is absent (correlation <= 0, or SAM weight 0 in a species) scores 0, with
in_refined_graph = 0. Secondary: blast_score (bit score of the edge), SAM gene weights.

Workflow: authors' defaults (SAM preprocessing, 3 iterations, Leiden neighborhoods, no labels).
Differences that do not change the graph: umap=False, save_processed=False, the BLAST graph comes
ready through the `gnnm` argument. Genes enter under their Ensembl gene ids.

Prebuilt BLAST: --blast-dir holds graph/gnnm.npz and graph/gns.npy, written once by
build_samap_graph.py from the reciprocal blastp of blast_samap.sbatch. The graph is over all Ensembl
110 human and mouse genes and does not depend on the cells, so every Human+Mouse cohort reuses it.
--label-key is checked and recorded only: this workflow does not use labels. --seed seeds numpy, but
SAM.run and the Leiden step use their fixed default seed 0, so it does not reach the tool.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b2t_common as C  # noqa: E402

CPUS = C.n_cpus()
C.set_threads(CPUS)

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from samap import SAMAP  # noqa: E402

BLAST_DIR = C.REPO / "data" / "revision" / "B2" / "tools" / "samap"
TOOL_DIR = C.TOOLS / "SAMap"
PREFIX = {"Human": "hs", "Mouse": "mm"}
PARAMS = dict(n_iterations=3, cross_species_k=20, neighborhood_sizes=3, hom_edge_thr=0, hom_edge_mode="pearson",
              scale_edges_by_corr=True, pairwise=True, neigh_from_keys=False, keys="leiden_clusters", leiden_resolution=3.0,
              eval_thr=1e-6, filter_thr=0.25, umap=False, save_processed=False, backend="cpu")


def load_graph(blast_dir):
    graph = Path(blast_dir) / "graph"
    if not (graph / "gnnm.npz").exists():
        raise SystemExit(f"no prebuilt graph in {graph}: run blast_samap.sbatch first")
    gnnm = sp.load_npz(graph / "gnnm.npz").tocsr()
    gns = np.load(graph / "gns.npy").astype("object")
    prefix = np.array([g.split("_")[0] for g in gns])
    gns_dict = {sid: gns[prefix == sid] for sid in PREFIX.values()}
    return gnnm, gns, gns_dict


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--label-key", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", default=None, help="scratch folder for the per-species h5ad files")
    p.add_argument("--blast-dir", default=str(BLAST_DIR), help="folder with the prebuilt graph/ of build_samap_graph.py")
    a = p.parse_args(argv)
    t0 = time.time()
    logging.getLogger("samap").setLevel(logging.INFO)
    np.random.seed(a.seed)

    X, var_names, species, labels = C.read_h5ad(a.h5ad, a.label_key)
    genes = C.ortholog_map(a.cohort, var_names)
    keep_cells = np.ones(species.size, dtype=bool)
    tmp_root = Path(a.work) if a.work else Path(a.out).resolve().parent / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="samap_", dir=tmp_root))
    try:
        files = {}
        per_species = {}
        for s, sid in PREFIX.items():
            ids = genes[f"id_{s.lower()}"]
            usable = (ids.notna() & ~ids.duplicated(keep="first")).to_numpy()
            m = (species == s) & keep_cells
            per_species[s] = int(usable.sum())
            x = ad.AnnData(X=sp.csr_matrix(X[m][:, usable], dtype=np.float32),
                           obs=pd.DataFrame(index=[f"{s}_{i}" for i in range(int(m.sum()))]),
                           var=pd.DataFrame(index=ids[usable].astype(str).values))
            f = work / f"{sid}.h5ad"
            x.write_h5ad(f)
            files[sid] = str(f)
        sm = SAMAP(files, gnnm=load_graph(a.blast_dir), save_processed=False, backend="cpu")
        sm.run(umap=False, ncpus=CPUS)

        gns = pd.Series(np.arange(sm.gns.size), index=sm.gns)
        hs = ("hs_" + genes["id_human"].astype(str)).map(gns)
        mm = ("mm_" + genes["id_mouse"].astype(str)).map(gns)
        ok = (hs.notna() & mm.notna()).to_numpy()
        hi, mi = hs[ok].astype(int).to_numpy(), mm[ok].astype(int).to_numpy()
        blast = np.asarray(sm.gnnm[hi, mi]).ravel()
        refined = np.asarray(sm.gnnm_refined[hi, mi]).ravel()
        w_h = sm.sams["hs"].adata.var["weights"]
        w_m = sm.sams["mm"].adata.var["weights"]
        sub = genes[ok]
        out = pd.DataFrame({
            "var_name": sub["var_name"].values,
            "score": refined.astype(float),
            "blast_score": blast.astype(float),
            "in_refined_graph": (refined != 0).astype(int),
            "sam_weight_human": ("hs_" + sub["id_human"].astype(str)).map(w_h).values,
            "sam_weight_mouse": ("mm_" + sub["id_mouse"].astype(str)).map(w_m).values,
        })
        out = out[out["blast_score"] > 0].reset_index(drop=True)
        ver = C.tool_state(TOOL_DIR)
        C.write_outputs(
            a.out, out, tool="SAMap", tool_info=dict(ver, repo="atarashansky/SAMap", version=C_version()),
            primary="score = edge weight of the ortholog pair in SAMAP.gnnm_refined (query_gene_pair 'correlation')",
            params=PARAMS, args=a, t0=t0, n_input_genes=var_names.size,
            extra={"genes_with_id": per_species, "n_pairs_in_dataset_graph": int(ok.sum()),
                   "n_in_refined_graph": int(out["in_refined_graph"].sum()), "labels_used": False, "seed_reaches_tool": False,
                   "blast_graph": str(Path(a.blast_dir) / "graph"), "label_cells": C.label_table(species, labels)})
    finally:
        shutil.rmtree(work, ignore_errors=True)


def C_version():
    from importlib.metadata import version
    return version("sc-samap")


if __name__ == "__main__":
    main()
