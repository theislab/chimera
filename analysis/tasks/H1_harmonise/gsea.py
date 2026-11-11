"""Hallmark GSEA per cohort, the call of notebooks/_pathway_builder.py: gseapy.prerank, 1,000 permutations, seed 0, sizes 10-500, paralog
collisions keep the largest |value|. Ranked on D over all genes of the cohort, as that notebook ranks Delta-alpha.

    PYTHONPATH=<chimera repo> python analysis/tasks/H1_harmonise/gsea.py [--jobs 6]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from analysis.tasks.H1_harmonise import rules as RU

OUT = RU.OUT / "gsea"


def one(c: str) -> str:
    import gseapy

    f = OUT / f"{c}.csv"
    g = pd.read_parquet(RU.OUT / "genes" / f"{c}.parquet").dropna(subset=["symbol"])
    d = g.assign(sym=g.symbol.str.upper(), a=g.D.abs()).sort_values("a", ascending=False).drop_duplicates("sym")
    rnk = d[["sym", "D"]].sort_values("D", ascending=False)
    rnk.columns = ["Gene", "Score"]
    res = gseapy.prerank(rnk=rnk, gene_sets="MSigDB_Hallmark_2020", organism="human", outdir=None, permutation_num=1000, min_size=10, max_size=500, seed=0, verbose=False, no_plot=True, threads=2)
    t = res.res2d.copy()
    for col in ("ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"):
        if col in t:
            t[col] = pd.to_numeric(t[col], errors="coerce")
    t.insert(0, "cohort", c)
    t.to_csv(f, index=False)
    return f"{c} {len(t)} {int((t['FDR q-val'] < 0.05).sum())}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(one, list(RU.cohorts())):
            print(r, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
