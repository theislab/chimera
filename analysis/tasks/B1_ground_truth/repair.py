"""Rows whose fit ended without scores (tasks submitted before the direct readouts were separated from the fit). Lists them in repair.tsv; no fit is rerun.
A task in the queue is not a row to repair. Then: sbatch --array=0-N analysis/tasks/B1_ground_truth/repair.sbatch"""

from __future__ import annotations

import pandas as pd

from analysis.common import cohorts as CO
from analysis.tasks.B1_ground_truth import rules as RU
from analysis.tasks.B1_ground_truth.completion import active_runs


def main():
    g = pd.read_csv(RU.TASK_DIR / "grid_all.tsv", sep="\t", keep_default_na=False)
    g = g[(g.alt != "none") & (g.duplicate_of == "")]
    act = active_runs()
    fail = {p.name.rsplit("_", 1)[0] for p in (CO.RUNS_DIR / "_failed").glob("b1-*")} if (CO.RUNS_DIR / "_failed").exists() else set()
    r = g[g.run.map(lambda x: (x not in act) and ((CO.RUNS_DIR / x).exists() or x in fail) and not (RU.RES / "scores" / f"{x}.parquet").exists())]
    r.to_csv(RU.TASK_DIR / "repair.tsv", sep="\t", index=False)
    print(len(r), "rows to repair", list(r.run)[:10])


if __name__ == "__main__":
    main()
