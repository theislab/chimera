"""T2_5_celltype_margin checkpoint-only forward pass on one Van Zyl seed run (code unchanged).

``T2_5_celltype_margin/run.py::forward_sums`` with the Van Zyl dataset as data, gene identity
through the ``van_zyl`` table, and two groupings: the harmonized fine types
(``cell_type_harmonised``, for the per-gene cell-type drivers of V3) and the harmonized classes
(``cell_ontology_class``, the conditional-MMD key). Batch codes come from the run's own
``batch_key`` (``source_batch``). Output ``results/analysis/T11/drivers/fw_<dataset>_s<seed>.npz``;
nothing is written to the run directory.

    PYTHONPATH=<chimera repo> python analysis/tasks/T11_5_vz_genes/forward.py --dataset full --seed 42
    PYTHONPATH=<chimera repo> python analysis/tasks/T11_5_vz_genes/forward.py --missing  # every D seed without output
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from analysis.common import cohorts as C
from analysis.tasks.T11_5_vz_genes import vz

sys.path.insert(0, str(vz.ROOT / "analysis" / "tasks" / "T2_5_celltype_margin"))
import run as T25RUN  # noqa: E402

KEYS = [vz.LABEL_KEY, vz.CLASS_KEY]


def one(ds: str, seed: int, threads: int, batch_size: int) -> None:
    run = vz.seed_runs(ds)[seed]
    out = vz.OUT / "drivers"
    out.mkdir(parents=True, exist_ok=True)
    C.read_adata = lambda cohort, backed=False: vz.read_h5ad(ds, backed)  # data for forward_sums
    T25RUN.GROUP_KEYS = {"liver": KEYS, "testis": KEYS}
    fw = T25RUN.forward_sums(f"{ds}_s{seed}", vz.TABLE_COHORT, run, threads, batch_size)
    fw["run"] = np.array(run)
    tmp = out / f"fw_{ds}_s{seed}.tmp.npz"
    np.savez_compressed(tmp, **fw)
    tmp.replace(out / f"fw_{ds}_s{seed}.npz")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(vz.DATASETS))
    p.add_argument("--seed", type=int)
    p.add_argument("--missing", action="store_true")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    a = p.parse_args()
    if a.missing:
        for ds in vz.usable_datasets():
            for s in vz.seeds_in_D(ds):
                if not (vz.OUT / "drivers" / f"fw_{ds}_s{s}.npz").exists():
                    one(ds, s, a.threads, a.batch_size)
    else:
        one(a.dataset, a.seed, a.threads, a.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
