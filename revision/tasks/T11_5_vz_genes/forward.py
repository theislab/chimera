"""T11.5: T2.5 checkpoint-only forward pass on one Van Zyl seed run (code unchanged).

``T2_5_celltype_margin/run.py::forward_sums`` with the Van Zyl dataset as data, gene identity
through the ``van_zyl`` table, and two groupings: the harmonised fine types
(``cell_type_harmonised``, for the per-gene cell-type drivers of V3) and the harmonised classes
(``cell_ontology_class``, the conditional-MMD key). Batch codes come from the run's own
``batch_key`` (``source_batch``). Output ``results/revision/T11/drivers/fw_<dataset>_s<seed>.npz``;
nothing is written to the run directory.

    PYTHONPATH=<chimera repo> python revision/tasks/T11_5_vz_genes/forward.py --dataset full --seed 42
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from revision.common import cohorts as C
from revision.tasks.T11_5_vz_genes import vz

sys.path.insert(0, str(vz.ROOT / "revision" / "tasks" / "T2_5_celltype_margin"))
import run as T25RUN  # noqa: E402

KEYS = [vz.LABEL_KEY, vz.CLASS_KEY]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(vz.DATASETS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    a = p.parse_args()
    run = vz.seed_runs(a.dataset)[a.seed]
    out = vz.OUT / "drivers"
    out.mkdir(parents=True, exist_ok=True)
    C.read_adata = lambda cohort, backed=False: vz.read_h5ad(a.dataset, backed)  # data for forward_sums
    T25RUN.GROUP_KEYS = {"liver": KEYS, "testis": KEYS}
    fw = T25RUN.forward_sums(f"{a.dataset}_s{a.seed}", vz.TABLE_COHORT, run, a.threads, a.batch_size)
    fw["run"] = np.array(run)
    tmp = out / f"fw_{a.dataset}_s{a.seed}.tmp.npz"
    np.savez_compressed(tmp, **fw)
    tmp.replace(out / f"fw_{a.dataset}_s{a.seed}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
