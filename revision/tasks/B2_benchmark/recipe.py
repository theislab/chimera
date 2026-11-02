"""Print the training arguments of one grid row as shell words: run seed fit args."""

from __future__ import annotations

import argparse
import shlex

import pandas as pd

from revision.tasks.B2_benchmark import rules as RU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    r = pd.read_csv(a.manifest, sep="\t", keep_default_na=False).iloc[a.index]
    cfg = RU.COHORTS[r["cohort"]]
    nolabel = RU.SETTINGS[r["setting"]]["kind"] == "nolabel" and cfg["nolabel_recipe"]
    args = cfg["nolabel_recipe"] if nolabel else cfg["recipe"]
    print(f"RUN={shlex.quote(r['run'])}; SEED={int(r['seed'])}; FIT={int(r['fit'])}; ARGS={shlex.quote(args)}")


if __name__ == "__main__":
    main()
