"""Fix the GSE192742 mouse hold-out before any expression is read.

Rule (pre-specified analysis plan, section 8): hold out 1 Western-diet (WD) and 1
standard-diet (SD) mouse at each of 24 and 36 weeks. Candidates are the nuclei diet
series CS184-CS197. The draw uses only GEO sample metadata (series matrix titles and
characteristics), never a count.

Each mouse is one GEO sample here. Within each (diet, weeks) stratum, candidates are
sorted by GSM accession and one is drawn with ``random.Random(SEED)`` in the fixed
stratum order SD-24, WD-24, SD-36, WD-36. Rerunning gives the same IDs.

Usage: python -I fix_holdouts.py <GSE192742-GPL24247_series_matrix.txt.gz> <out.tsv>
"""

from __future__ import annotations

import csv
import gzip
import random
import re
import sys

SEED = 20261101
CANDIDATES = [f"CS{n}" for n in range(184, 198)]
STRATA = [("SD", 24), ("WD", 24), ("SD", 36), ("WD", 36)]


def read_matrix(path: str) -> list[dict]:
    """Per-sample metadata from a GEO series matrix (header block only)."""
    fields: dict[str, list[list[str]]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            key, *vals = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
            fields.setdefault(key, []).append(vals)
    n = len(fields["!Sample_geo_accession"][0])
    out = []
    for i in range(n):
        rec = {"gsm": fields["!Sample_geo_accession"][0][i], "title": fields["!Sample_title"][0][i]}
        for vals in fields.get("!Sample_characteristics_ch1", []):
            k, _, v = vals[i].partition(": ")
            rec[k.strip()] = v.strip()
        out.append(rec)
    return out


def diet_weeks(title: str) -> tuple[str, int]:
    m = re.match(r"(Standard|Western) diet fed \((\d+)\s*weeks\)", title)
    if not m:
        raise ValueError(f"unexpected title: {title!r}")
    return ("SD" if m.group(1) == "Standard" else "WD"), int(m.group(2))


def main(matrix: str, out: str) -> None:
    recs = [r for r in read_matrix(matrix) if r.get("shortfilename") in CANDIDATES]
    found = sorted(r["shortfilename"] for r in recs)
    missing = sorted(set(CANDIDATES) - set(found) - {"CS186", "CS187"})
    if missing:
        raise SystemExit(f"candidates missing from the matrix: {missing}")
    for r in recs:
        r["diet"], r["weeks"] = diet_weeks(r["title"])
        if "nuclei" not in r["title"].lower():
            raise SystemExit(f"not a nuclei sample: {r['title']}")
    rng = random.Random(SEED)
    held = {}
    for diet, weeks in STRATA:
        pool = sorted((r for r in recs if r["diet"] == diet and r["weeks"] == weeks), key=lambda r: r["gsm"])
        if len(pool) < 2:
            raise SystemExit(f"stratum {diet}-{weeks} has {len(pool)} mice; a hold-out needs at least 2")
        pick = rng.choice(pool)
        held[pick["gsm"]] = f"{diet}-{weeks}"
    cols = ["gsm", "shortfilename", "diet", "weeks", "role", "stratum_size", "geo_number_of_cells", "platform", "title"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols + ["seed"])
        for r in sorted(recs, key=lambda r: (r["weeks"], r["diet"], r["gsm"])):
            size = sum(1 for x in recs if x["diet"] == r["diet"] and x["weeks"] == r["weeks"])
            w.writerow([r["gsm"], r["shortfilename"], r["diet"], r["weeks"],
                        "holdout" if r["gsm"] in held else "training", size,
                        r.get("number of cells", ""), r.get("platform", ""), r["title"], SEED])
    for r in sorted(recs, key=lambda r: (r["weeks"], r["diet"], r["gsm"])):
        print(r["gsm"], r["shortfilename"], r["diet"], r["weeks"], "HOLDOUT" if r["gsm"] in held else "training",
              r.get("number of cells", ""))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
