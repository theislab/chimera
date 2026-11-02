"""G1 step 1: table per cohort from step1_runs.csv and the training trace of the non-finite bone-marrow seeds. Adds the lower bound of the largest z_logvar at any logged step, log(var_max): the largest
exp(z_logvar) in a batch is at least the batch mean.    python -m revision.tasks.G1_guard.step1_table
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from revision.tasks.G1_guard.step1_scan import REPO, training_trace

OUT = REPO / "results" / "revision" / "G1"


def main():
    d = pd.read_csv(OUT / "step1_runs.csv")
    d["trace_logvar_lower"] = np.log(d["var_max"])
    d.to_csv(OUT / "step1_runs.csv", index=False)
    cols = dict(n_seeds=("seed", "size"), eval_max=("eval_max", "max"), eval_min=("eval_min", "min"), eval_runs_above=("eval_n_above", lambda x: int((x > 0).sum())),
                eval_share_above_max=("eval_share_above", "max"), eval_runs_below=("eval_n_below", lambda x: int((x > 0).sum())), eval_share_below_max=("eval_share_below", "max"),
                train_max=("train_max", "max"), train_min=("train_min", "min"), train_share_above_max=("train_share_above", "max"), train_share_below_max=("train_share_below", "max"),
                trace_lower_max=("trace_logvar_lower", "max"), trace_above_runs=("trace_above_10", lambda x: int((x == "above").sum())),
                trace_indet_runs=("trace_above_10", lambda x: int((x == "indeterminate").sum())), trace_none_runs=("trace_above_10", lambda x: int((x == "none").sum())))
    t = d.groupby(["role", "cohort"], sort=False).agg(**cols).reset_index()
    t.to_csv(OUT / "step1_table.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 40)
    print(t.round(6).to_string())
    print(d[d.eval_n_above > 0][["cohort", "seed", "run", "eval_max", "eval_n_above", "eval_cells_above", "n_cells", "eval_share_above"]].to_string())
    print(d[d.role == "canonical"].query("train_max > 10")[["cohort", "seed", "train_max", "train_share_above", "train_cells_above", "n_cells"]].to_string())
    r = []
    for run in ["rev-bm2-s44", "rev-bm2-s45", "rev-bm2-grid2-s42", "rev-bm2-grid7-s42", "rev-bm2-s43"]:
        tr = training_trace(run)
        tr["run"] = run
        r.append(tr)
    print(pd.DataFrame(r).to_string())
    pd.DataFrame(r).to_csv(OUT / "step1_nan_runs_trace.csv", index=False)


if __name__ == "__main__":
    main()
