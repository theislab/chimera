"""Checks of the liver disease data from GEO sample metadata only: donors per condition, G-D1 countable parts.

Reads the SOFT sample text written by fetch_disease.sh and the fixed hold-out table.
No expression value is read.

Usage: python -I sample_counts.py <disease_raw_dir> <holdouts.tsv> <out_dir>
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def soft_samples(path: Path) -> list[dict]:
    recs, cur = [], None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("^SAMPLE"):
            cur = {"gsm": line.split("=", 1)[1].strip(), "files": []}
            recs.append(cur)
        elif cur is None:
            continue
        elif line.startswith("!Sample_title"):
            cur["title"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_platform_id"):
            cur["platform"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_characteristics_ch1"):
            k, _, v = line.split("=", 1)[1].strip().partition(": ")
            cur[k.strip()] = v.strip()
        elif line.startswith("!Sample_supplementary_file"):
            cur["files"].append(line.split("=", 1)[1].strip())
    return recs


def main(raw: Path, holdouts: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    # GSE202379 human: donors and samples per disease status
    h = soft_samples(raw / "GSE202379/GSE202379_samples_brief.soft.txt")
    donors, samples, sex = defaultdict(set), Counter(), defaultdict(Counter)
    for r in h:
        st = r.get("disease status", "?")
        donors[st].add(r.get("patient id"))
        samples[st] += 1
    for st in donors:
        for d in donors[st]:
            sx = {r.get("gender") for r in h if r.get("patient id") == d}
            sex[st]["/".join(sorted(x for x in sx if x))] += 1
    say("GSE202379 samples", len(h))
    for st in sorted(donors):
        say(f"  {st}: donors {len(donors[st])}, samples {samples[st]}, donor sex {dict(sex[st])}")
    healthy = [k for k in donors if k.lower().startswith("healthy")]
    nash = [k for k in donors if k.lower() == "nash w/o cirrhosis"]
    n_h = sum(len(donors[k]) for k in healthy)
    n_n = sum(len(donors[k]) for k in nash)
    # mouse training mice from the fixed hold-out table
    with holdouts.open() as f:
        ho = list(csv.DictReader(f, delimiter="\t"))
    tr = Counter(r["diet"] for r in ho if r["role"] == "training")
    say("G-D1 countable now:")
    say(f"  healthy donors (>=3): {n_h} -> {'pass' if n_h >= 3 else 'FAIL'}  [{', '.join(healthy)}]")
    say(f"  MASH donors, NASH w/o cirrhosis (>=15): {n_n} -> {'pass' if n_n >= 15 else 'FAIL'}")
    say(f"  training mice SD, WD (>=4 each): {tr.get('SD', 0)}, {tr.get('WD', 0)} -> "
        f"{'pass' if tr.get('SD', 0) >= 4 and tr.get('WD', 0) >= 4 else 'FAIL'}")
    say("  >=4 of 5 types at >=20 nuclei per sample: not countable before annotation (T7.2)")
    # GSE129516 condition split
    x = soft_samples(raw / "GSE129516/GSE129516_samples_brief.soft.txt")
    say("GSE129516 samples", len(x))
    for r in x:
        say(f"  {r['gsm']} {r.get('title')} | {r.get('phenotype')} | {r.get('cell type')} | {r.get('platform')}")
    ph = Counter(r.get("phenotype") for r in x)
    say("  condition split:", dict(ph))
    # GSE136103 species and condition
    y = soft_samples(raw / "GSE136103/GSE136103_samples_brief.soft.txt")
    keys = sorted({k for r in y for k in r if k not in ("gsm", "files", "title")})
    say("GSE136103 samples", len(y), "characteristics", keys)
    for r in y:
        say("  " + " | ".join([r["gsm"], r.get("title", "")] + [f"{k}={r[k]}" for k in keys if k in r]))
    # GSE135251 bulk
    z = soft_samples(raw / "GSE135251/GSE135251_samples_brief.soft.txt")
    zk = sorted({k for r in z for k in r if k not in ("gsm", "files", "title")})
    say("GSE135251 samples", len(z), "characteristics", zk)
    for k in zk:
        c = Counter(r.get(k) for r in z)
        if len(c) <= 12:
            say(f"  {k}: {dict(c)}")
    (out / "sample_counts.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
