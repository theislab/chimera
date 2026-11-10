"""Analysis of the control that keeps the sequence features and destroys the counts. Run once the 12 control runs have finished.

For each control run ``rev-ctl-<cohort>-perm<A|B|C>-s<seed>`` (``manifest.tsv``; permutation
from ``data/analysis/controls/<control>.perm.npz``), against the cohort consensus of D
excluding the run's own seed (``analysis/common/controls.py``, seeds 42-46):

* slot rho, ``rho(D_arm[g], D_ref[g])``: the sequence-driven component;
* moved rho, ``rho(D_arm[g], D_ref[perm[g]])``: the count-driven component;

each on the Spearman scale and on the covariate-residual scale (each score residualized on
its own covariates; the control's count covariates move with the counts), and the same for
S_g_ema. Calls (pre-specified analysis plan, section 9):

* slot: ``high`` only if above the 97.5th percentile of the 1,000-permutation offline ridge
  projection null for the same arm and reference (``offline_null.py``); ``low`` if below the
  2.5th; ``within null`` otherwise. Arm c is the primary answer to m1.
* moved, against the floor F (the five leave-one-seed-out rho on the same scale):
  ``preserved`` rho >= min F, ``lost`` rho < 0.5 * median F, ``partial`` otherwise; residual
  scale decides. On S the floor is about 0.9.

Also reported per run: last epoch, ``expectation_summary.fail``, ``failed_expectations``, tail
fraction |D| > 0.05. ``test_hmouse`` arm c and ``liv_4sp`` arm a are single runs (descriptive).

Usage::

    python analysis/tasks/T3_5_count_perm/analyze.py [--n-perm 1000] [--drop liv_hmouse:43 ...]
        [--allow-missing] [--outdir DIR] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.common.outputs import TaskOutput, add_common_args

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offline_null as ON  # noqa: E402
from make_control_h5ad import CONTROLS_TSV  # noqa: E402

TAIL = 0.05


def run_name(row) -> str:
    return f"rev-ctl-{row['cohort']}-perm{row['arm'].upper()}-s{int(row['train_seed'])}"


def last_epoch(run: str) -> int | None:
    f = C.RUNS_DIR / run / "lightning" / "metrics.csv"
    if not f.exists():
        return None
    e = pd.read_csv(f, usecols=["epoch"])["epoch"].dropna()
    return int(e.max()) if len(e) else None


def parse_drop(items) -> dict[str, tuple[int, ...]]:
    out: dict[str, list[int]] = {}
    for it in items or []:
        c, s = it.split(":")
        out.setdefault(c, []).append(int(s))
    return {c: tuple(v) for c, v in out.items()}


def floor_table(cohort: str, M: pd.DataFrame, cov: pd.DataFrame, key: str) -> pd.DataFrame:
    U = cov[K.COVARIATE_COLUMNS].notna().all(1).to_numpy()
    Z = K.design(cov[U])
    f = K.floor(M, Z, rows=np.where(U)[0])
    return f.assign(cohort=cohort, readout=key)


def analyse_cohort(cohort: str, rows: pd.DataFrame, drop: tuple[int, ...], n_perm: int, seed: int,
                   allow_missing: bool):
    runs = K.seed_runs(cohort, drop, require_all=not allow_missing)
    inp = ON.cohort_inputs(cohort)
    cov = inp["cov"]
    MD = K.seed_matrix(cohort, runs, "D")
    MS = K.seed_matrix(cohort, runs, "S_g_ema")
    fl = pd.concat([floor_table(cohort, MD, cov, "D"), floor_table(cohort, MS, cov, "S")], ignore_index=True)
    out, nulls = [], []
    for i, (_, row) in enumerate(rows.iterrows()):
        name = run_name(row)
        if not K.run_finished(name):
            if allow_missing:
                out.append(dict(run=name, cohort=cohort, arm=row["arm"], status="not finished"))
                continue
            raise FileNotFoundError(f"{name} has not finished")
        perm = np.load(ON.CONTROL_DIR / f"{row['control']}.perm.npz")["perm"]
        r = K.read_run(name)
        ts = int(row["train_seed"])
        ref_seeds = [s for s in MD.columns if s != ts] if len(MD.columns) > 1 else list(MD.columns)
        D_ref = MD[ref_seeds].mean(1).to_numpy()
        S_ref = MS[ref_seeds].mean(1).to_numpy()
        c_ref = K.seed_matrix(cohort, {s: runs[s] for s in ref_seeds}, "correction").mean(1).to_numpy()
        prior = K.read_run(runs[ref_seeds[0]])["prior"]
        if not np.allclose(prior, r["prior"]):
            raise AssertionError(f"{name}: prior differs from the reference runs")
        acov = ON.arm_covariates(cov, perm)
        sD = ON.slot_moved(r["D"], acov, D_ref, cov, perm)
        sS = ON.slot_moved(r["S_g_ema"], acov, S_ref, cov, perm)
        proj = ON.Projection(inp["F"], S_ref, c_ref, prior)
        d = ON.null_draws(row["arm"], proj, S_ref, c_ref, D_ref, cov, inp["bins"], n_perm, seed + i)
        ns = ON.summarise(d)
        nulls.append(dict(run=name, cohort=cohort, arm=row["arm"], ref_seeds=";".join(map(str, ref_seeds)), **ns))
        Fd = fl[(fl.readout == "D")]
        Fs = fl[(fl.readout == "S")]
        summ = r["summary"]
        rec = dict(run=name, control=row["control"], cohort=cohort, arm=row["arm"], train_seed=ts,
                   perm_seed=int(row["perm_seed"]), status="finished", ref_seeds=";".join(map(str, ref_seeds)),
                   last_epoch=last_epoch(name), expectation_fail=summ["expectation_summary"]["fail"],
                   failed_expectations=";".join(summ["failed_expectations"]),
                   tail_frac=float((np.abs(r["D"]) > TAIL).mean()),
                   tail_frac_ref=float((np.abs(D_ref) > TAIL).mean()),
                   sd_D=float(r["D"].std()), sd_D_ref=float(D_ref.std()))
        rec.update({f"D_{k}": v for k, v in sD.items()})
        rec.update({f"S_{k}": v for k, v in sS.items()})
        for scale in ("spearman", "residual"):
            lo, hi = ns[f"{scale}_slot__p2_5"], ns[f"{scale}_slot__p97_5"]
            v = sD[f"{scale}_slot"]
            rec[f"null_{scale}_slot_p2_5"], rec[f"null_{scale}_slot_p97_5"] = lo, hi
            rec[f"null_{scale}_moved_p2_5"] = ns[f"{scale}_moved__p2_5"]
            rec[f"null_{scale}_moved_p97_5"] = ns[f"{scale}_moved__p97_5"]
            rec[f"call_slot_{scale}"] = "high" if v > hi else ("low" if v < lo else "within null")
            rec[f"floor_D_{scale}_min"] = float(Fd[scale].min())
            rec[f"floor_D_{scale}_median"] = float(Fd[scale].median())
            rec[f"call_moved_D_{scale}"] = K.call(sD[f"{scale}_moved"], Fd[scale].to_numpy())
            rec[f"call_moved_S_{scale}"] = K.call(sS[f"{scale}_moved"], Fs[scale].to_numpy())
            rec[f"call_slot_D_floor_{scale}"] = K.call(v, Fd[scale].to_numpy())
        out.append(rec)
    return out, nulls, fl


def arm_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    fin = per_run[per_run.status == "finished"]
    rows = []
    for (cohort, arm), g in fin.groupby(["cohort", "arm"]):
        r = dict(cohort=cohort, arm=arm, n_runs=len(g))
        for col in ["D_spearman_slot", "D_spearman_moved", "D_residual_slot", "D_residual_moved",
                    "S_spearman_slot", "S_spearman_moved", "S_residual_slot", "S_residual_moved", "tail_frac"]:
            r[f"{col}__mean"] = float(g[col].mean())
            r[f"{col}__min"] = float(g[col].min())
            r[f"{col}__max"] = float(g[col].max())
        for scale in ("spearman", "residual"):
            r[f"n_slot_high_{scale}"] = int((g[f"call_slot_{scale}"] == "high").sum())
            r[f"moved_D_calls_{scale}"] = json.dumps(g[f"call_moved_D_{scale}"].value_counts().to_dict())
        r["primary"] = bool(cohort == "liv_hmouse" and arm == "c")
        r["descriptive_only"] = bool(len(g) == 1)
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3500)
    ap.add_argument("--drop", nargs="*", default=[], help="cohort:seed pairs left out of D (T3.1 rule)")
    ap.add_argument("--allow-missing", action="store_true", help="skip unfinished runs and seeds (pre-check)")
    add_common_args(ap)
    args = ap.parse_args()
    outdir = args.outdir
    if args.allow_missing and not outdir:  # a pre-check never writes the official DONE
        outdir = str(gene_ids.REPO_ROOT / "results" / "analysis" / "T3_5" / "_precheck")
    out = TaskOutput("T3_5", outdir, prerun=args.prerun or args.allow_missing, force=args.force)
    drop = parse_drop(args.drop)
    tab = pd.read_csv(CONTROLS_TSV, sep="\t")
    per_run, nulls, floors = [], [], []
    for ci, (cohort, rows) in enumerate(tab.groupby("cohort", sort=False)):
        print("==", cohort, flush=True)
        a, b, f = analyse_cohort(cohort, rows, drop.get(cohort, ()), args.n_perm, args.seed + 100 * ci,
                                 args.allow_missing)
        per_run += a
        nulls += b
        floors.append(f)
    per_run = pd.DataFrame(per_run)
    out.csv(per_run, "per_run.csv")
    out.csv(pd.DataFrame(nulls), "null_per_run.csv")
    out.csv(pd.concat(floors, ignore_index=True), "floor.csv")
    summ = arm_summary(per_run)
    out.csv(summ, "arm_summary.csv")
    pd.set_option("display.width", 250, "display.max_columns", 60)
    cols = ["run", "last_epoch", "expectation_fail", "D_spearman_slot", "null_spearman_slot_p97_5",
            "D_residual_slot", "null_residual_slot_p97_5", "call_slot_residual", "D_residual_moved",
            "floor_D_residual_min", "call_moved_D_residual", "S_spearman_moved"]
    print(per_run[[c for c in cols if c in per_run.columns]].round(3).to_string())
    print(summ.set_index(["cohort", "arm"]).T.to_string())
    out.finish(dict(n_perm=args.n_perm, drop=args.drop, allow_missing=args.allow_missing,
                    n_finished=int((per_run.status == "finished").sum()) if "status" in per_run else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
