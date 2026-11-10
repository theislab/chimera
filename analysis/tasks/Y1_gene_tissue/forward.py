"""Y1: T2.5 checkpoint forward pass (code unchanged), grouped by label x random cell half.

Gives per seed the species x type margin and two cell halves for the technical reliability of S on the fixed model.
Stratified 50/50 within species x group, seeded. Nothing is written to run directories.

    python -I launch.py analysis.tasks.Y1_gene_tissue.forward <tissue> <seed> [--threads 8]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from analysis.common import cohorts as C
from analysis.tasks.Y1_gene_tissue import common as Y
from analysis.tasks.Y1_gene_tissue import rules as RU

sys.path.insert(0, str(Y.ROOT / "analysis" / "tasks" / "T2_5_celltype_margin"))
import run as T25RUN  # noqa: E402


def half_labels(a, rng) -> np.ndarray:
    lab = a.obs["y1_group"].astype(str).to_numpy()
    spc = a.obs["species"].astype(str).to_numpy()
    h = np.zeros(a.n_obs, dtype=int)
    for k in sorted({(s, l) for s, l in zip(spc, lab)}):
        i = np.flatnonzero((spc == k[0]) & (lab == k[1]))
        h[rng.permutation(i)[: len(i) // 2]] = 1
    return np.array([f"{l}|h{x}" for l, x in zip(lab, h)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tissue")
    p.add_argument("seed", type=int)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    a = p.parse_args()
    d = Y.TISSUES[a.tissue]
    ad_ = Y.read(a.tissue)
    ad_.obs["y1half"] = half_labels(ad_, np.random.default_rng(RU.SEED))
    C.read_adata = lambda cohort, backed=False: ad_
    T25RUN.GROUP_KEYS = {"liver": ["y1half"], "testis": ["y1half"]}
    fw = T25RUN.forward_sums(f"{a.tissue}_s{a.seed}", d["table"], d["runs"][a.seed], a.threads, a.batch_size)
    fw["run"] = np.array(d["runs"][a.seed])
    Y.FW.mkdir(parents=True, exist_ok=True)
    tmp = Y.FW / f"fw_{a.tissue}_s{a.seed}.tmp.npz"
    np.savez_compressed(tmp, **fw)
    tmp.replace(Y.FW / f"fw_{a.tissue}_s{a.seed}.npz")


if __name__ == "__main__":
    main()
