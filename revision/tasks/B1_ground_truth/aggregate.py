"""B1 tidy tables, in the schema of the benchmark overview: track, task, setting, method, metric, value, ci_lo, ci_hi, n, rank_direction.

task: the alteration (pattern, level, loss) or null (sham_swap, thin_both, placebo). setting: 'cohort=..;effect=..;band=..;share=..;identity=..;fit=..',
'all' where marginal. A failed fit removes only Chimera rows. A value is the mean over the fits of the setting; the CI is the 2.5-97.5 percentile of the mean of the fits' pair-bootstrap
draws (fit-wise draws share an index, independent across fits). n sums the altered genes. Discarded fits leave.
rank_direction: higher for detection metrics on real alterations; closer_to_0.5 for AUROC and dir_acc on nulls; closer_to_0.05 for reject05 and power_05 on nulls.

Files: results/revision/B1/B1_tidy.csv (every setting), B1_overview.csv (the rows the overview needs: per task x cohort, and per task, for each primary metric; plus Track D),
per_cell.parquet (one row per method x fit x subset x metric), failed_runs.csv. Only tiers in rules.RUN_TIERS are read; tiers 4-6 are planned, not run,
so the six liver seed-43 fits of the first pass (tier 5) are not scored.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from revision.tasks.B1_ground_truth import metrics as MT
from revision.tasks.B1_ground_truth import rules as RU

METRICS = ["AUROC", "AUROC_cov", "AUPRC_match", "AUPRC_genome", "topk_genome", "topk_band", "power_05", "dir_acc", "dir_raw",
           "effect_sd", "reject05", "coverage", "frac_zero"]
NULL_TASKS = {"sham_swap", "thin_both"}
PRIMARY = ["AUROC", "AUPRC_genome", "topk_genome", "power_05", "dir_acc", "reject05"]


def direction(task: str, metric: str) -> str:
    if task in NULL_TASKS or task == "placebo":
        if metric in ("reject05", "power_05"):
            return "closer_to_0.05"
        if metric in ("AUROC", "AUROC_cov", "dir_acc", "dir_raw"):
            return "closer_to_0.5"
        return "none"
    if metric in ("coverage", "frac_zero"):
        return "none"
    return "higher"


def load_cells() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, failed = [], []
    g = pd.read_csv(RU.TASK_DIR / "grid_all.tsv", sep="\t", keep_default_na=False)
    g = g[(g.alt != "none") & (g.duplicate_of == "") & g["tier"].astype(int).isin(RU.RUN_TIERS)]
    if (RU.TASK_DIR / "grid_arms.tsv").exists():
        ga = pd.read_csv(RU.TASK_DIR / "grid_arms.tsv", sep="\t", keep_default_na=False)
        g = pd.concat([g, ga[ga.alt != "none"]], ignore_index=True)
    for _, r in g.iterrows():
        run = r["run"]
        h, m = RU.RES / "health" / f"{run}.json", RU.RES / "metrics" / f"{run}.parquet"
        if not h.exists() or not m.exists():
            failed.append(dict(run=run, cohort=r["cohort"], reason="no output"))
            continue
        hh = json.loads(h.read_text())
        t = pd.read_parquet(m)
        if hh.get("discard", True):  # a failed fit removes Chimera's rows only; direct, sequence and tool readouts of the object stay
            failed.append(dict(run=run, cohort=r["cohort"], reason=f"finite={hh.get('finite')} active_dims={hh.get('active_dims')}"))
            t = t[~t["group"].astype(str).str.startswith("chimera")]
        rows.append(t)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(failed, columns=["run", "cohort", "reason"])


def draws_for(runs: list[str], method: str, cache: dict) -> np.ndarray | None:
    arr = []
    for r in runs:
        if r not in cache:
            z = np.load(RU.RES / "draws" / f"{r}.npz")
            cache[r] = {k: z[k] for k in z.files}
        if method in cache[r]:
            arr.append(cache[r][method])
    return np.mean(arr, 0) if arr else None


def ci(runs, method, metric, cache) -> tuple[float, float]:
    if metric not in MT.BOOT_METRICS:
        return np.nan, np.nan
    d = draws_for(runs, method, cache)
    if d is None:
        return np.nan, np.nan
    v = d[:, MT.BOOT_METRICS.index(metric)]
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def main():
    cells, failed = load_cells()
    failed.to_csv(RU.RES / "failed_runs.csv", index=False)
    cells.to_parquet(RU.RES / "per_cell.parquet")
    cells["effect"] = cells["effect"].astype(float).map(lambda v: f"{v:g}")
    out = []
    cache: dict = {}
    # settings: full cells and all marginals over cohort, effect, band, share
    dims = ["cohort", "effect", "band", "share"]
    base = cells[cells.subset == "all"]
    for task, dt in base.groupby("alt"):
        for keep in itertools.chain.from_iterable(itertools.combinations(dims, k) for k in range(len(dims) + 1)):
            keys = list(keep) + ["method", "metric"]
            for vals, d in dt[dt.metric.isin(METRICS)].groupby(keys, sort=False):
                vals = vals if isinstance(vals, tuple) else (vals,)
                kv = dict(zip(keys, vals))
                setting = ";".join(f"{x}={kv.get(x, 'all')}" for x in dims) + ";identity=all"
                lo, hi = ci(sorted(d["run"].unique()), kv["method"], kv["metric"], cache)
                out.append(dict(track="A", task=task, setting=setting, method=kv["method"], metric=kv["metric"], value=d["value"].mean(),
                                ci_lo=lo, ci_hi=hi, n=int(d.drop_duplicates("run")["n"].sum()), rank_direction=direction(task, kv["metric"])))
    # subsets: identity and masking, at the marginal per task x cohort x band
    for sub in RU.IDENTITY_BANDS + ["ref_unmasked", "ref_masked"]:
        d0 = cells[(cells.subset == sub) & cells.metric.isin(METRICS)]
        for (task, c, b, m, k), d in d0.groupby(["alt", "cohort", "band", "method", "metric"]):
            out.append(dict(track="A", task=task, setting=f"cohort={c};effect=all;band={b};share=all;identity={sub}", method=m, metric=k,
                            value=d["value"].mean(), ci_lo=np.nan, ci_hi=np.nan, n=int(d.drop_duplicates("run")["n"].sum()),
                            rank_direction=direction(task, k)))
    # placebo
    pl = []
    for c in RU.COHORTS:
        for f in RU.RES.glob(f"placebo/{c}*.parquet"):
            t = pd.read_parquet(f)
            t["cohort"] = c
            pl.append(t)
    if pl:
        t = pd.concat(pl, ignore_index=True)
        for (c, m), d in t.groupby(["cohort", "method"]):
            for k in ("AUROC", "reject05", "power_05"):
                out.append(dict(track="A", task="placebo", setting=f"cohort={c};effect=all;band=all;share=all;identity=all", method=m, metric=k,
                                value=d[k].mean(), ci_lo=d[k].quantile(0.025), ci_hi=d[k].quantile(0.975), n=int(len(d)),
                                rank_direction=direction("placebo", k)))
        for (c, b, m), d in t.groupby(["cohort", "band", "method"]):
            for k in ("AUROC", "reject05", "power_05"):
                out.append(dict(track="A", task="placebo", setting=f"cohort={c};effect=all;band={b};share=all;identity=all", method=m, metric=k,
                                value=d[k].mean(), ci_lo=d[k].quantile(0.025), ci_hi=d[k].quantile(0.975), n=int(len(d)),
                                rank_direction=direction("placebo", k)))
    tidy = pd.DataFrame(out)
    tidy.to_csv(RU.RES / "B1_tidy.csv", index=False)
    is_cohort = tidy["setting"].str.contains(r"effect=all;band=all;share=all;identity=all$", regex=True)
    ov = tidy[is_cohort & tidy["metric"].isin(PRIMARY)]
    d = RU.RES / "B1_trackD_tidy.csv"
    if d.exists():  # Track D rows in the same schema (track_d/collect.py)
        ov = pd.concat([ov, pd.read_csv(d)], ignore_index=True)
    ov.to_csv(RU.RES / "B1_overview.csv", index=False)
    print(len(tidy), "rows", flush=True)


if __name__ == "__main__":
    main()
