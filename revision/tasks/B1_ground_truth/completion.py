"""Track A: runs tried and finished, per method and setting, in the way of B2_benchmark/completion.py. Failures stay out of the scores and are shown here.

Chimera readouts (S_g, Delta_alpha, B1z): tried = a health record or a run folder for the grid row; finished = healthy under the discard rule of rules.py.
Direct, sequence and control readouts are computed on the same altered object: tried = the row's score file exists; finished = the column has a finite value.
Tools: tried = a score file or a recorded attempt (results/revision/B1/attempts/<tool>/<run>.rc); finished = a score file with at least one finite score.
Not counted as tried: a row still in the queue or not reached, and a job lost to a bad node (it stops before a run folder or an attempt record exists).

Writes results/revision/B1/completion.csv: track, task, setting, method, tried, finished; setting as in B1_tidy.csv, full cells and the marginals per cohort,
per task and per task x band.   python -m revision.tasks.B1_ground_truth.completion
"""

from __future__ import annotations

import csv
import json
import os
import subprocess

import pandas as pd

from revision.common import cohorts as CO
from revision.tasks.B1_ground_truth import metrics as MT
from revision.tasks.B1_ground_truth import rules as RU

CHIMERA = ["S_g", "Delta_alpha", "B1z"]
TOOL_COL = {"saturn": ["S_deploy", "S_cos"], "camex": ["C_cos"], "genespectra": ["GeneSpectra"], "cacimar": ["CACIMAR"], "samap": ["SAMap"]}
SEQ_METHODS = {"pi", "dNdS", "LOEUF"}


def active_runs() -> set[str]:
    """Runs with a task still in the queue, so a run folder in progress is not read as a failure."""
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-r", "-h", "-o", "%i|%j"], capture_output=True, text=True).stdout
    grids, act = {}, set()
    for line in out.split("\n"):
        if "|" not in line:
            continue
        jid, name = line.split("|")
        if "_" not in jid or not name.startswith("b1_"):
            continue
        man = RU.TASK_DIR / f"grid_{name[3:]}.tsv"
        if man.exists():
            grids.setdefault(man, list(csv.DictReader(open(man), delimiter="\t")))
            act.add(grids[man][int(jid.split("_")[1])]["run"])
    return act


def setting(r) -> str:
    return f"cohort={r['cohort']};effect={float(r['effect']):g};band={r['band']};share={r['share']};identity=all"


def table() -> pd.DataFrame:
    g = pd.read_csv(RU.TASK_DIR / "grid_all.tsv", sep="\t", keep_default_na=False)
    g = g[(g.alt != "none") & (g.duplicate_of == "")]
    active = active_runs()
    rows = []
    for _, r in g.iterrows():
        run, st = r["run"], setting(r)
        base = dict(task=r["alt"], setting=st, cohort=r["cohort"], effect=f"{float(r['effect']):g}", band=r["band"], share=r["share"])
        h, f = RU.RES / "health" / f"{run}.json", RU.RES / "scores" / f"{run}.parquet"
        tried = h.exists() or ((CO.RUNS_DIR / run).exists() and run not in active)
        fin = h.exists() and not json.loads(h.read_text())["discard"]
        for m in CHIMERA:
            rows.append(dict(base, method=m, tried=int(tried), finished=int(fin)))
        if f.exists():
            d = pd.read_parquet(f)
            for m, (col, _, grp) in MT.METHODS.items():
                if m in CHIMERA or grp == "chimera_exploratory" or col not in d:
                    continue
                rows.append(dict(base, method=m, tried=1, finished=int(d[col].notna().any())))
            for m in SEQ_METHODS:
                rows.append(dict(base, method=m, tried=1, finished=1))
    t = pd.read_csv(RU.TASK_DIR / "grid_tools_v2.tsv", sep="\t")
    t = t[t.alt != "none"]
    for _, r in t.iterrows():
        base = dict(task=r["alt"], setting=setting(r), cohort=r["cohort"], effect=f"{float(r['effect']):g}", band=r["band"], share=r["share"])
        for tool, cols in TOOL_COL.items():
            f, a = RU.RES / "tool_scores" / f"{tool}__{r['run']}.parquet", RU.RES / "attempts" / tool / f"{r['run']}.rc"
            d = pd.read_parquet(f) if f.exists() else None
            for c in cols:
                fin = d is not None and c in d and bool(d[c].notna().any())
                rows.append(dict(base, method=c, tried=int(f.exists() or a.exists()), finished=int(fin)))
    d = pd.DataFrame(rows)
    d = d[d.tried > 0]
    cell = d.groupby(["task", "setting", "method"], as_index=False)[["tried", "finished"]].sum()
    out = [cell]
    for keys, fmt in ((["task", "cohort", "method"], lambda k: f"cohort={k[1]};effect=all;band=all;share=all;identity=all"),
                      (["task", "method"], lambda k: "cohort=all;effect=all;band=all;share=all;identity=all"),
                      (["task", "cohort", "band", "method"], lambda k: f"cohort={k[1]};effect=all;band={k[2]};share=all;identity=all")):
        x = d.groupby(keys, as_index=False)[["tried", "finished"]].sum()
        x["setting"] = [fmt(k) for k in x[keys].itertuples(index=False)]
        out.append(x[["task", "setting", "method", "tried", "finished"]])
    res = pd.concat(out, ignore_index=True)
    res.insert(0, "track", "A")
    return res


def main():
    t = table()
    t.to_csv(RU.RES / "completion.csv", index=False)
    print(len(t), "rows")
    print(t[t.tried > t.finished].to_string())


if __name__ == "__main__":
    main()
