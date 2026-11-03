"""E3: checkpoint-only forward passes of the per-gene (free) gate fits kept by Z4, so the count
margin S_g of the per-gene read-out decomposes by species and cell type exactly as the canonical
forward passes (T2.5 code unchanged: ``forward_sums``). Nothing is written to run dirs.

    PYTHONPATH=$PWD python revision/tasks/E3_cells/forward_free.py manifest
    PYTHONPATH=$PWD python revision/tasks/E3_cells/forward_free.py run --row I --threads 4

Output: ``results/revision/E3/forward_free/<cohort>/fw_s<seed>.npz`` (grouping = the cohort's
registry cell-type key, as H1).
"""
from __future__ import annotations

import argparse
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd

from revision.common import cohorts as C
from revision.common import gene_ids
from revision.tasks.H1_harmonise import rules as RU

OUT = RU.RES / "E3" / "forward_free"
MANIFEST = RU.RES / "E3" / "forward_free_manifest.tsv"


def manifest() -> None:
    kept = json.loads((RU.RES / "Z4" / "kept.json").read_text())
    rows = [dict(cohort=c, seed=int(s), run=r) for c, d in kept.items() for s, r in sorted(d.items())]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(MANIFEST, sep="\t", index=False)
    print(len(rows), "rows ->", MANIFEST)


def run(row: int, threads: int, batch: int) -> None:
    m = pd.read_csv(MANIFEST, sep="\t").iloc[row]
    c, seed, rn = str(m.cohort), int(m.seed), str(m.run)
    reg = RU.cohorts()[c]
    dst = OUT / c / f"fw_s{seed}.npz"
    if dst.exists():
        print("exists", dst)
        return
    sys.path.insert(0, str(gene_ids.REPO_ROOT / "revision" / "tasks" / "T2_5_celltype_margin"))
    import run as T25RUN  # noqa: PLC0415

    C.read_adata = lambda _c, backed=False: ad.read_h5ad(reg["h5ad"], backed="r" if backed else None)
    T25RUN.C.read_adata = C.read_adata
    T25RUN.GROUP_KEYS["liver"] = [reg["ct"]]
    T25RUN.GROUP_KEYS["testis"] = [reg["ct"]]
    # landmark genes only feed T2.5's bootstrap, not used here; skip the symbol lookup (no table key for some cohorts)
    T25RUN.gene_ids.symbols = lambda _c, vn: pd.Series([""] * len(vn))
    fw = T25RUN.forward_sums(f"{c}_s{seed}", c, rn, threads, batch)
    fw["run"] = np.array(rn)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez_compressed(tmp, **fw)
    tmp.replace(dst)
    print("wrote", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "run"])
    ap.add_argument("--row", type=int)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    a = ap.parse_args()
    manifest() if a.cmd == "manifest" else run(a.row, a.threads, a.batch_size)
