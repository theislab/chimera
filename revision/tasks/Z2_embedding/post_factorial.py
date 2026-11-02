"""Measure and encode every finished factorial run, one process per pass, skipping what is done. python -m revision.tasks.Z2_embedding.post_factorial"""
import subprocess, sys, time
import pandas as pd
from revision.tasks.Z2_embedding import rules as RU

fac = pd.read_csv(RU.REPO / "revision/tasks/Z2_embedding/factorial.tsv", sep="\t")
base = {c: list(RU.base_seed_runs(c).values()) for c in ("liv_hmouse", "test_hmouse")}
while True:
    n = 0
    for r in list(fac.itertuples()) + [type("B", (), dict(run=x, cohort=c)) for c, v in base.items() for x in v]:
        d = RU.RUNS / r.run
        if not (d / "summary.json").exists():
            continue
        for kind, f in (("measure", RU.OUT / "measure" / f"{r.cohort}__{r.run}.parquet"), ("encode", RU.OUT / "encoded" / f"{r.cohort}__{r.run}.parquet")):
            if f.exists():
                continue
            cmd = [sys.executable, "-m", f"revision.tasks.Z2_embedding.{kind}"] + (["run", r.cohort, r.run] if kind == "measure" else ["model", r.cohort, r.run])
            if not (RU.OUT / "cache" / f"{r.cohort}_targets.npz").exists():
                continue
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode != 0:
                print("FAILED", kind, r.run, out.stderr[-400:], flush=True)
            n += 1
    done = sum((RU.RUNS / x / "summary.json").exists() for x in fac.run)
    print(time.strftime("%H:%M"), "factorial finished", done, "of", len(fac), "new", n, flush=True)
    if done == len(fac) and n == 0:
        break
    time.sleep(120)
