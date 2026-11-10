"""Z4: write manifest.tsv, one row per free-gate fit to train. Recipe = argv of the cohort's first kept H1 run with provenance,
`--config A` -> `--config M`, seed replaced; nothing else changes. Memory = 1.3 x the peak RSS of the cohort's canonical jobs + 1 GB,
at least 3 GB (sacct MaxRSS of the jobs named in _provenance.json; cohorts without provenance use the values in MEM_FALLBACK).

    PYTHONPATH=$PWD python analysis/tasks/Z4_free_gate_all/make_manifest.py [--seeds 42,43]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess

import pandas as pd

from analysis.tasks.H1_harmonise import rules as H1R
from analysis.tasks.Z4_free_gate_all import rules as RU

MEM_FALLBACK = {"liv_hmonkey": 1.5, "liv_hzeb": 1.5, "test_hmac": 4.0}  # GB; sizes as liver H+Mouse and testis H+Mouse (no sacct record)


def kept_runs() -> pd.DataFrame:
    t = pd.read_csv(H1R.OUT / "runs.tsv", sep="\t")
    return t[t.kept.astype(bool)]


def peak_gb(job_ids: list[str]) -> tuple[float | None, float]:
    """Peak MaxRSS (GB) and longest elapsed time (h) over the jobs."""
    if not job_ids:
        return None, 1.0
    out = subprocess.run(["sacct", "-j", ",".join(job_ids), "--format=JobID%30,MaxRSS,ElapsedRaw", "-n", "-P"], capture_output=True, text=True).stdout
    vals, hours = [], [1.0]
    for line in out.splitlines():
        jid, rss, el = (line.split("|") + ["", ""])[:3]
        if el.isdigit():
            hours.append(int(el) / 3600)
        if not rss:
            continue
        m = re.match(r"([\d.]+)([KMG]?)", rss)
        if m:
            vals.append(float(m.group(1)) / {"K": 1024**2, "M": 1024, "G": 1, "": 1024**3}[m.group(2)])
    return (max(vals) if vals else None), max(hours)


def recipe(cohort: str, runs: list[str]) -> tuple[list[str], list[str], str]:
    jobs, argv, src = [], None, None
    for r in runs:
        p = RU.RUNS / r / "_provenance.json"
        if p.exists():
            j = json.loads(p.read_text())
            jobs.append(str(j["job_id"]))
            if argv is None:
                argv, src = list(j["argv"]), r
    if argv is None:
        argv, src = ["--config", "A"] + shlex.split(RU.NO_PROVENANCE[cohort]) + ["--seed", "0"], "configs/canonical_recipes.tsv"
    return argv, jobs, src


def free_args(argv: list[str], seed: int) -> str:
    a = list(argv)
    i = a.index("--config")
    assert a[i + 1] == "A", a
    a[i + 1] = "M"
    j = a.index("--seed")
    a[j + 1] = str(seed)
    assert "--clamp-z-logvar" not in a  # no chosen recipe of H1 uses the guard; G1's guard is only a labelled fallback
    return " ".join(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(map(str, RU.SEEDS)))
    ap.add_argument("--cohorts", default="")
    ap.add_argument("--out", default=str(RU.TASK / "manifest.tsv"))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    k = kept_runs()
    want = [c for c in a.cohorts.split(",") if c] or [c for c in dict.fromkeys(k.cohort) if c not in RU.Z3_FREE]
    rows = []
    for c in want:
        runs = list(k[k.cohort == c].sort_values("seed").run)
        argv, jobs, src = recipe(c, runs)
        pk, hrs = peak_gb(jobs)
        pk = pk or MEM_FALLBACK[c]
        mem = max(3, math.ceil(1.3 * pk + 1))
        for s in seeds:
            rows.append(dict(run_name=f"{RU.PREFIX}-{c}-m-s{s}", cohort=c, seed=s, mem=f"{mem}G", peak_canonical_gb=round(pk, 2), canonical_hours=round(hrs, 2), recipe_from=src,
                             args=free_args(argv, s)))
    t = pd.DataFrame(rows).sort_values(["canonical_hours", "cohort", "seed"], ascending=[False, True, True], kind="stable")  # slow fits first
    t.to_csv(a.out, sep="\t", index=False)
    print(t[["run_name", "mem", "peak_canonical_gb", "canonical_hours", "recipe_from"]].reset_index(drop=True).to_string())


if __name__ == "__main__":
    main()
