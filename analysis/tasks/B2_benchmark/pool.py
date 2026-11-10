"""Score many units in one job with a process pool, so that scoring takes few queue slots.
    python -m analysis.tasks.B2_benchmark.pool b     Track B, every (cohort, setting) with scores
    python -m analysis.tasks.B2_benchmark.pool c     Track C units, lists and Y4 replication
"""

from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

from concurrent.futures import ProcessPoolExecutor

import pandas as pd


def _b(i):
    from analysis.tasks.B2_benchmark import score_b as SB

    c, s = SB.task_list()[i]
    try:
        SB.score_task(c, s)
        return (c, s, "ok")
    except BaseException as e:  # noqa: BLE001
        return (c, s, repr(e)[:200])


def _u(i):
    from analysis.tasks.B2_benchmark import rules as RU
    from analysis.tasks.B2_benchmark import track_c as TC

    u = pd.read_csv(RU.TASK_DIR / "track_c_units.tsv", sep="\t").iloc[i].to_dict()
    try:
        TC.score_unit(u)
        return (u["cohort"], u["kind"], u["stratum"], "ok")
    except BaseException as e:  # noqa: BLE001
        return (u["cohort"], u["kind"], u["stratum"], repr(e)[:200])


def _l(c):
    from analysis.tasks.B2_benchmark import track_c as TC

    try:
        TC.score_lists(c)
        return (c, "ok")
    except BaseException as e:  # noqa: BLE001
        return (c, repr(e)[:200])


def main():
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    if sys.argv[1] == "b":
        from analysis.tasks.B2_benchmark import score_b as SB

        with ProcessPoolExecutor(n) as ex:
            for r in ex.map(_b, range(len(SB.task_list()))):
                print(r, flush=True)
    else:
        from analysis.tasks.B2_benchmark import rules as RU
        from analysis.tasks.B2_benchmark import rules_c as RC
        from analysis.tasks.B2_benchmark import track_c as TC

        u = TC.unit_list()
        pd.DataFrame(u).to_csv(RU.TASK_DIR / "track_c_units.tsv", sep="\t", index=False)
        with ProcessPoolExecutor(n) as ex:
            for r in ex.map(_u, range(len(u))):
                print(r, flush=True)
            for r in ex.map(_l, list(RC.COHORTS_C)):
                print(r, flush=True)
        TC.y4()


if __name__ == "__main__":
    main()
