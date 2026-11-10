"""G1: stop a guard-arm fit once its training loss is NaN at an epoch end (it does not recover), after the probe has written the first non-finite batch; then write its health record.
Only tasks of G1 arrays (job name g1_arm). Run repeatedly while the array runs, from the repository root with mamba on PATH:    python3 analysis/tasks/G1_guard/nan_watch.py
"""
import re
import subprocess
from pathlib import Path

LOG = Path("results/analysis/G1/logs")
q = subprocess.run(["squeue", "--me", "-h", "-t", "R", "-n", "g1_arm", "-o", "%i"], capture_output=True, text=True).stdout.split()
for jid in q:
    if "_" not in jid:
        continue
    arr, task = jid.split("_")
    log = LOG / f"g1_arm_{arr}_{task}.out"
    if not log.exists():
        continue
    m = re.search(r"Epoch (\d+)/\d+[^\r\n]*train_loss_epoch=nan", log.read_text(errors="ignore"))
    if m:
        subprocess.run(["scancel", jid])
        with open(LOG / "nan_watch.log", "a") as f:
            f.write(f"{jid} loss NaN at epoch {m.group(1)}, cancelled\n")
        subprocess.run(["bash", "-c", f"mamba run -n chimera_env python -m analysis.tasks.G1_guard.step3_health --index {task} >> {LOG}/nan_watch.log 2>&1"])
        print(f"NaN {jid} epoch {m.group(1)}", flush=True)
