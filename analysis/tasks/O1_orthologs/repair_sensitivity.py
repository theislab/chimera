"""After the gate of repair_compare.py only: the tests of T4_3_scoreboard on D of the identifier-joined ("repaired") objects, beside the frozen row. Frozen folders are read, never written.

Arms, each a whole scoreboard run with the frozen draws, seeds and golds:
  can3        canonical D of seeds 42, 43, 44, frozen covariates and baselines: the like-for-like comparator for a 3-seed D;
  rep3        repaired D of seeds 42, 43, 44, covariates, tau, ESM blocks and baselines recomputed on the repaired object;
  rep3_shared rep3 on the genes of the canonical object;
  rep3_norep  rep3 without the repaired genes: their own contribution;
  rep3_int    post hoc: rep3 with liver H+Mouse on its integrated repaired seeds only (42, 44), since can3 is all integrated;
  can_int     post hoc: can3 with liver H+Mouse on the same seeds (42, 44).
Testis 3-sp is not rebuilt: its rows stay canonical in every arm. Baseline split halves and ceilings are not computed here (no status uses them).

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.repair_sensitivity baselines
    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.repair_sensitivity arm --arm rep3
    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.repair_sensitivity subsets
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.tasks.O1_orthologs import rules as RU
from analysis.tasks.O1_orthologs.repair_compare import COH, OUT, SEEDS, obj, repaired_set
from analysis.tasks.T4_3_scoreboard import inputs as I
from analysis.tasks.T4_3_scoreboard import run as R
from analysis.tasks.T4_3_scoreboard import stats as S

REP = [c for c in ("liv_hmouse", "test_hmouse", "liv_4sp")]
T42 = OUT / "T4_2"
TOOLS = RU.REPO / "results/analysis/T5_6/tools_primary.csv"


def gate_open() -> None:
    """Outcome of the gate (amendment log of the pre-specified analysis plan, 2026-11-02): H+Mouse and testis H+Mouse passed; 4-sp beyond, marginal, opened as a labeled exception. H+Monkey stays closed here."""
    g = json.loads((OUT / "gate.json").read_text())
    for c in ("liv_hmouse", "test_hmouse"):
        if g[c]["all"]["beyond"] or g[c]["unrepaired"]["beyond"]:
            raise SystemExit(f"{c} beyond seed range: stop, no gold is read")


def patch_objects() -> None:
    o = C.read_adata

    def read(cohort, backed=False):
        if cohort in REP:
            import anndata as ad

            return ad.read_h5ad(obj(cohort), backed="r" if backed else None)
        return o(cohort, backed=backed)

    C.read_adata = read
    for c in REP:
        C.CANONICAL[c] = f"o2-{c}-s42"


def baselines() -> None:
    gate_open()
    patch_objects()
    from analysis.tasks.T4_2_gene_baselines import run as T

    T.main(["--cohorts", ",".join(REP), "--outdir", str(T42), "--threads", "8", "--force"])


def shim() -> Path:
    """R.RES stand-in: golds, T2_5_celltype_margin and DONE as frozen, no consensus-D B3b check, no baseline halves."""
    d = OUT / "shim"
    (d / "T4_3").mkdir(parents=True, exist_ok=True)
    (d / "T3_3").mkdir(exist_ok=True)
    real = RU.REPO / "results/analysis"
    for src, dst in ((real / "T4_3/golds", d / "T4_3/golds"), (real / "T2_5", d / "T2_5"), (real / "T4_3/cache", d / "T4_3/cache")):
        if not dst.exists():
            dst.symlink_to(src)
    (d / "T3_3/DONE").touch()
    return d


def patch_arm(arm: str) -> None:
    repaired = arm.startswith("rep3")
    R.INTEGRATED_SEEDS = {}  # its unit needs seeds 47, 48, 51; Family 3 only, not in a 3-seed D
    runs = {}
    for c in REP:
        can = {int(s): r for s, r in I.seed_runs_used(c).items()}
        use = (42, 44) if (arm in ("rep3_int", "can_int") and c == "liv_hmouse") else SEEDS  # post hoc: repaired H+Mouse s43 is poorly mixed, canonical 42-44 are not
        runs[c] = {s: (f"o2-{c}-s{s}" if repaired else can[s]) for s in use}
    o_load, o_used = I.load, I.seed_runs_available
    o_u, o_su = R.universe, R.set_universe
    I.seed_runs_available = lambda cohort: runs[cohort]
    can_vars = {}
    for c in REP:
        a = C.read_adata(c, backed=True)
        can_vars[c] = set(a.var_names.astype(str))
        a.file.close()
    drop = {c: set(repaired_set(c).var_name) for c in REP}
    if repaired:
        patch_objects()
        I.T42_DIR = T42
    real_t42 = RU.REPO / "results/analysis/T4_2"
    cache = OUT / f"cache_{'rep' if repaired else 'can'}"

    def load(cohort, cache_dir, consensus=True):
        I.T42_DIR = T42 if (repaired and cohort in REP) else real_t42
        if cohort in REP:
            d = o_load(cohort, cache, consensus=False)
            mean_da = d["M"].mean(1)
            d["tab"]["D"] = mean_da - np.median(mean_da)
        else:
            d = o_load(cohort, cache_dir, consensus=True)
        t = d["tab"]
        keep = np.ones(len(t), bool)
        if cohort in REP and arm == "rep3_shared":
            keep = t["var_name"].isin(can_vars[cohort]).to_numpy()
        if cohort in REP and arm == "rep3_norep":
            keep = t["var_name"].isin(can_vars[cohort]).to_numpy() & ~t["var_name"].isin(drop[cohort]).to_numpy()
        t["o1_keep"] = keep
        return d

    I.load = load
    R.universe = lambda tab, gold, cov, scores, subset=None: o_u(tab, gold, cov, scores, subset) & tab["o1_keep"].to_numpy()
    R.set_universe = lambda tab, subset=None: o_su(tab, subset) & tab["o1_keep"].to_numpy()
    R.RES = shim()


def arm(name: str, workers: int) -> None:
    gate_open()
    patch_arm(name)
    out = OUT / f"scoreboard_{name}"
    argv = ["--mode", "real", "--outdir", str(out), "--workers", str(workers)]
    if TOOLS.exists() and name not in ("rep3_int", "can_int"):  # the tool unit needs seeds 42, 43, 44
        argv += ["--tools", str(TOOLS)]
    R.main(argv)


def subsets() -> None:
    """All 3-seed subsets of canonical seeds on the frozen inputs: the range of rho_p for 3 seeds."""
    gate_open()
    R.CFG.update(draws=1, n_perm=1, n_boot_sets=1, n_perm_sets=1, n_calib=1, n_perm_calib=1, mode="real")
    R.prepare("real", 1)
    rows = []
    for h, v in R.PRIMARY.items():
        D = R.DATA[v["cohort"]]
        tab, M = D["tab"], D["M"]
        m = R.universe(tab, v["gold"], v["cov"], ["D"])
        T = tab.loc[m]
        Z = S.design(T, R.COVSETS[v["cov"]])
        g = S.midrank(T[v["gold"]].to_numpy(float))
        vals = []
        for p in itertools.combinations(range(M.shape[1]), 3):
            x = M[m][:, list(p)].mean(1)
            vals.append(float(S.rho_p_matrix(np.column_stack([g, S.midrank(x)]), Z)[1]))
        vals = np.array(vals)
        rows.append(dict(hyp=h, n_subsets=len(vals), min=vals.min(), q05=float(np.quantile(vals, 0.05)), median=float(np.median(vals)), q95=float(np.quantile(vals, 0.95)), max=vals.max()))
        if v["cohort"] == "liv_hmouse":  # post hoc: all pairs of integrated canonical seeds
            ii = [D["seeds"].index(q) for q in (42, 43, 44, 47, 48, 51)]
            w = np.array([float(S.rho_p_matrix(np.column_stack([g, S.midrank(M[m][:, list(p)].mean(1))]), Z)[1]) for p in itertools.combinations(ii, 2)])
            rows.append(dict(hyp=f"{h} integrated pairs", n_subsets=len(w), min=w.min(), q05=float(np.quantile(w, 0.05)), median=float(np.median(w)), q95=float(np.quantile(w, 0.95)), max=w.max()))
    pd.DataFrame(rows).to_csv(OUT / "three_seed_range.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["baselines", "arm", "subsets"])
    ap.add_argument("--arm", choices=["can3", "rep3", "rep3_shared", "rep3_norep", "rep3_int", "can_int"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.what == "baselines":
        baselines()
    elif a.what == "subsets":
        subsets()
    else:
        arm(a.arm, a.workers)


if __name__ == "__main__":
    main()
