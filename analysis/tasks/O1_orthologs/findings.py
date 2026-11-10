"""Recurrence of H1_harmonise and the statistics of X4_where_differs and X6_adds_beyond that the paper cites, recomputed without flagged genes. Same code as the owning tasks, frames filtered at the loader, outputs under results/analysis/O1/.

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.findings h1 --tier T3
    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.findings x6 --tier T3
    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.findings x4 --tier T3
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import pandas as pd

from analysis.tasks.O1_orthologs import rules as RU


def flags(tier: str) -> dict:
    G = pd.read_parquet(RU.RES / "flags_by_cohort.parquet")
    G["MAP"] = G.M_wrong_row | G.M_false_zero
    return {c: x for c, x in G.groupby("cohort")}, tier


def h1(tier: str, jobs: int) -> None:
    F, _ = flags(tier)
    out = RU.RES / f"h1_{tier}"
    (out / "genes").mkdir(parents=True, exist_ok=True)
    for c, x in F.items():
        src = RU.REPO / f"results/analysis/H1/genes/{c}.parquet"
        if not src.exists():  # the two eye fits that are not in H1_harmonise's own tables
            continue
        g = pd.read_parquet(src)
        g = g[~g.hsapiens_gene_id.isin(set(x.hsapiens_gene_id[x[tier]]))]
        g.to_parquet(out / "genes" / f"{c}.parquet")
    for f in ("runs.tsv",):
        shutil.copy(RU.REPO / f"results/analysis/H1/{f}", out / f)
    env = dict(os.environ, H1_OUT=str(out), PYTHONPATH=str(RU.REPO), H1_EXTRA=str(out / "no_extra_runs.tsv"))  # the 14 cohorts of the H1_harmonise tables
    subprocess.run([sys.executable, str(RU.REPO / "analysis/tasks/H1_harmonise/recurrence.py"), "--jobs", str(jobs)], env=env, check=True)


def _filter_load(CM, tier):
    F, _ = flags(tier)
    orig = CM.load

    def load(c):
        g = orig(c)
        return g[~g.hsapiens_gene_id.isin(set(F[c].hsapiens_gene_id[F[c][tier]]))].reset_index(drop=True)

    CM.load = load


def x6(tier: str, jobs: int) -> None:
    from analysis.tasks.X4_where_differs import common as CM
    from analysis.tasks.X6_adds_beyond import analyses as A

    _filter_load(CM, tier)
    A.OUT = RU.RES / f"x6_{tier}"
    A.OUT.mkdir(parents=True, exist_ok=True)
    CM.N_JOBS = jobs
    A.incremental()
    A.degradation()


def x4(tier: str, jobs: int) -> None:
    from analysis.tasks.X4_where_differs import analyses as A
    from analysis.tasks.X4_where_differs import common as CM

    _filter_load(CM, tier)
    out = RU.RES / f"x4_{tier}"
    out.mkdir(parents=True, exist_ok=True)
    A.OUT = out
    CM.N_JOBS = jobs
    A.N_JOBS = jobs
    G = {c: A.prepare(c) for c in A.RU.COHORTS}
    for s in ("adds", "x1", "x2"):
        print("==", s, flush=True)
        A.SECTIONS[s](G)
    signature(A, CM, G, out)


def signature(A, CM, G, out) -> None:
    """The X2_liver_biology program block of X4_where_differs ``mechanism``, alone: the full section needs every gene in run order."""
    lst = pd.read_csv(CM.RES / "X2/divergent_conserved.csv")
    prog = sorted(set(lst[lst.cohort == "liv_4sp"].symbol) - {"MT-CO1"})
    sets = {"prog63_noMT": prog, "core15": A.PROG_CORE, "fasting6": A.FASTING}
    rows = []
    for c in ("liv_4sp", "liv_hmouse"):
        g = A.gdrv(A.atlas_columns(G[c].copy()))
        d = pd.read_parquet(CM.CACHE / f"Sdir__{c}.parquet").set_index("var_name")
        for k in ("Sdir", "Sdir_pattern", "Sdir_level"):
            g[k] = g.var_name.map(d[k])
        for k, names in sets.items():
            f = g.symbol.isin(names).to_numpy()
            pool = ~g.symbol.isin(prog).to_numpy()
            for v in ("S", "Sdir", "Sdir_pattern", "Sdir_level", "B1b_raw"):
                rows.append(dict(cohort=c, set=k, value=v, matching="C_hep", **CM.matched(g, f, pool, v, A.CP + ["hep_enrich"], key=f"sig{c}{k}{v}")))
    pd.DataFrame(rows).to_csv(out / "mechanism_x2_signature.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["h1", "x6", "x4"])
    ap.add_argument("--tier", default="T3")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    dict(h1=h1, x6=x6, x4=x4)[a.what](a.tier, a.jobs)


if __name__ == "__main__":
    main()
