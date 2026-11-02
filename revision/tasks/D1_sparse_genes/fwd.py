"""S_fwd on the canonical fits: forward margin at the posterior mean, no 20-cell mask, mean over up to three kept fits that still hold a model. Reads no gold.
    python -m revision.tasks.D1_sparse_genes.fwd --cohort c
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("SLURM_CPUS_PER_TASK", "4"))

import anndata as ad
import numpy as np
import pandas as pd

from revision.common import cohorts as CO
from revision.common import gene_ids
from revision.tasks.B2_benchmark import score_run as SR
from revision.tasks.D1_sparse_genes import rules as RU
from revision.tasks.H1_harmonise import rules as H1R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    a = ap.parse_args()
    c = a.cohort
    d = H1R.COHORTS[c]
    rt = pd.read_csv(RU.REPO / "results/revision/H1/runs.tsv", sep="\t")
    rt = rt[(rt.cohort == c) & rt.kept]
    runs = [r for r in rt.run.tolist() if (CO.RUNS_DIR / r / "model").exists()][:3]
    A = ad.read_h5ad(d["h5ad"])
    import torch

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    S = []
    for r in runs:
        cfg = json.loads((CO.RUNS_DIR / r / "config.json").read_text())
        S.append(SR.forward_margin(CO.RUNS_DIR / r, A, cfg))
        print(r, flush=True)
    idx = pd.Index(np.asarray(gene_ids.ensembl(d["table"], A.var_names)), name="hsapiens_gene_id")
    out = pd.DataFrame(dict(S_fwd=np.mean(S, 0) if S else np.nan, n_fwd=len(S)), index=idx)
    (RU.RES / "fwd").mkdir(parents=True, exist_ok=True)
    out.to_parquet(RU.RES / "fwd" / f"{c}.parquet")


if __name__ == "__main__":
    main()
