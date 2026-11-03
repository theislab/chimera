"""SAMap gene-level homology graph from the BLAST tables, once per species pair. Run in samap_env.

The same calls and defaults as SAMAP.__init__ (reciprocal hits, e-value 1e-6, transcript to gene by
maximum, 25% filter), done once because every run on a perturbed object would repeat them; the
result goes to the documented `gnnm` argument of SAMAP. Transcripts map to Ensembl gene ids through
the `gene:` field of the FASTA headers.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from samap._constants import DEFAULT_EVAL_THRESHOLD, DEFAULT_FILTER_THRESHOLD
from samap.core.homology import _calculate_blast_graph, _coarsen_blast_graph, _filter_gnnm

REPO = Path(__file__).resolve().parents[4]
D = REPO / "data" / "revision" / "B2" / "tools" / "samap"
FASTA = {"hs": "Homo_sapiens.GRCh38.pep.all.fa.gz", "mm": "Mus_musculus.GRCm39.pep.all.fa.gz"}


def names_from_fasta(path):
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].split()
                gene = next(x for x in parts if x.startswith("gene:"))[5:].split(".")[0]
                out.append((parts[0], gene))
    return out


def main():
    t0 = time.time()
    ids = ["hs", "mm"]
    gnnm, gns, _ = _calculate_blast_graph(ids, f_maps=str(D / "maps") + "/", reciprocate=True, eval_thr=DEFAULT_EVAL_THRESHOLD)
    names = {k: names_from_fasta(D / "fasta" / v) for k, v in FASTA.items()}
    gnnm, gns_dict, gns = _coarsen_blast_graph(gnnm, gns, names)
    gnnm = _filter_gnnm(gnnm, thr=DEFAULT_FILTER_THRESHOLD)
    dest = D / "graph"
    dest.mkdir(exist_ok=True)
    sp.save_npz(dest / "gnnm.npz", sp.csr_matrix(gnnm))
    np.save(dest / "gns.npy", np.asarray(gns, dtype=str))
    (dest / "graph.json").write_text(json.dumps({
        "eval_thr": DEFAULT_EVAL_THRESHOLD, "filter_thr": DEFAULT_FILTER_THRESHOLD, "reciprocate": True,
        "n_genes": {k: int(v.size) for k, v in gns_dict.items()}, "n_edges": int(gnnm.nnz),
        "seconds": round(time.time() - t0, 1)}, indent=2))
    print("graph", gnnm.shape, gnnm.nnz, {k: v.size for k, v in gns_dict.items()}, flush=True)


if __name__ == "__main__":
    main()
