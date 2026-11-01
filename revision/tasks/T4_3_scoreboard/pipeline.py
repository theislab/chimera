"""One command from MANIFEST.tsv to all T4.3 outputs: checks and checksums every input, runs
the steps in order (a step whose outputs exist is skipped unless --redo), then the scoreboard.

Usage (from chimera/, in chimera_env, PYTHONPATH=.):
    python -m revision.tasks.T4_3_scoreboard.pipeline [--redo golds,baseline_splits]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

from revision.common import gene_ids

HERE = Path(__file__).resolve().parent
REPO = gene_ids.REPO_ROOT
COHORTS = ["liv_hmouse", "test_hmouse", "liv_4sp", "test_3sp"]
SKIP_IF = {"golds": "results/revision/T4_3/golds/gold_meta.json",
           "baseline_splits": "results/revision/T4_3/baseline_splits/baseline_splits__test_hmouse.npz",
           "b3b_consensus": "results/revision/T4_3/baselines_consensusD/DONE"}


def expand(v: str) -> list[Path]:
    v = os.path.expanduser(v)
    out = []
    subs = [{}]
    if "{cohort}" in v:
        subs = [{"cohort": c} for c in COHORTS]
    if "{h5ad}" in v:
        subs = [{"h5ad": gene_ids.H5ADS[c]} for c in COHORTS]
    if "{species}" in v:
        subs = [{"species": s} for s in ["bonobo", "chimpanzee", "gorilla", "gibbon", "marmoset", "opossum",
                                         "platypus", "chicken"]]
    for s in subs:
        p = v.format(**s)
        out.append(Path(p) if p.startswith("/") else REPO / p)
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=HERE / "MANIFEST.tsv")
    ap.add_argument("--redo", default="")
    a = ap.parse_args()
    m = pd.read_csv(a.manifest, sep="\t")
    rows = []
    for r in m[m.kind == "input"].itertuples():
        for p in expand(r.value):
            if not p.exists():
                raise SystemExit(f"missing input {r.key}: {p}")
            rows.append(dict(key=r.key, path=str(p), bytes=p.stat().st_size, sha256=sha256(p)))
            print(f"input {r.key}: {p.name} ok", flush=True)
    out = REPO / "results/revision/T4_3"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "inputs_sha256.tsv.tmp", sep="\t", index=False)
    (out / "inputs_sha256.tsv.tmp").replace(out / "inputs_sha256.tsv")
    redo = set(filter(None, a.redo.split(",")))
    env = dict(os.environ, PYTHONPATH=str(REPO))
    for r in m[m.kind == "step"].itertuples():
        skip = SKIP_IF.get(r.key)
        if skip and (REPO / skip).exists() and r.key not in redo:
            print(f"step {r.key}: outputs exist, skipped", flush=True)
            continue
        cmd = shlex.split(r.value)
        cmd[0] = sys.executable if cmd[0] == "python" else cmd[0]
        print(f"step {r.key}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
