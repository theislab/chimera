"""Print the training arguments of one grid row as shell words."""

from __future__ import annotations

import argparse
import shlex

import pandas as pd

from revision.tasks.D1_sparse_genes import rules as RU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    RU.register()
    r = pd.read_csv(RU.TASK / "grid.tsv", sep="\t", keep_default_na=False).iloc[a.index]
    args = RU.B2.COHORTS[r["cohort"]]["recipe"]
    if r["arm"]:
        args = (args + " " + RU.ARMS[r["arm"]]).strip()
    print(f"RUN={shlex.quote(r['run'])}; SEED={int(r['seed'])}; ARGS={shlex.quote(args)}")


if __name__ == "__main__":
    main()
