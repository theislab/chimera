"""Stop a depth fit once its loss is NaN (it cannot recover; the remaining epochs are wasted CPU) and record a discarded health stub, so completion counts it as tried, not finished.
Only tasks of the D1 arrays, only depth settings. Run in a loop by the session.    python3 nan_watch.py
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path("results/analysis/D1")
HEALTH = ROOT / "health"
grid = [l.rstrip("\n").split("\t") for l in open("analysis/tasks/D1_sparse_genes/grid.tsv")]
head, grid = grid[0], grid[1:]
q = subprocess.run(["squeue", "--me", "-h", "-t", "R", "-o", "%i|%j|%A"], capture_output=True, text=True).stdout.split()
for line in q:
    jid, name, _ = line.split("|")
    if name not in ("d1_test", "d1_4sp", "d1_liv") or "_" not in jid:
        continue
    arr, task = jid.split("_")
    row = dict(zip(head, grid[int(task)]))
    if not row["setting"].startswith("depth"):
        continue
    logs = list((ROOT / "logs").glob(f"{name}_{arr}_{task}.out"))
    if not logs:
        continue
    m = re.search(r"Epoch (\d+)/\d+[^\r\n]*train_loss_epoch=nan", logs[0].read_text(errors="ignore"))
    if m:
        h = HEALTH / f"{row['run']}.json"
        if not h.exists():
            h.write_text(json.dumps(dict(run=row["run"], cohort=row["cohort"], setting=row["setting"], finite=False, discard=True, note=f"loss NaN at epoch {m.group(1)}, cancelled there"), indent=1))
        subprocess.run(["scancel", jid])
        print(f"NaN {row['run']} epoch {m.group(1)}, cancelled {jid}", flush=True)
