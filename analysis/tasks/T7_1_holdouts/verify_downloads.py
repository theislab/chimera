"""Check the disease downloads against GEO's own file lists. Metadata only.

For each RAW.tar, every member name and size must match ``filelist.txt``. The GSE192742
per-sample h5 files are checked against that series' file list. For GSE129516 the header
line of each per-sample matrix is read to count barcodes (no count value is parsed).

Usage: python -I verify_downloads.py <disease_raw_dir> <out_dir>
"""

from __future__ import annotations

import gzip
import sys
import tarfile
from pathlib import Path


def filelist(path: Path) -> dict[str, int]:
    out = {}
    for line in path.read_text().splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 4 and p[0] == "File":
            out[p[1]] = int(p[3])
    return out


def main(raw: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        lines.append(s)

    ok_all = True
    for gse in ("GSE202379", "GSE135251", "GSE129516", "GSE136103"):
        fl = filelist(raw / gse / "filelist.txt")
        with tarfile.open(raw / gse / f"{gse}_RAW.tar") as t:
            mem = {m.name: m.size for m in t.getmembers() if m.isfile()}
        same = mem == fl
        ok_all &= same
        say(f"{gse}: tar members {len(mem)}, filelist {len(fl)}, names and sizes identical: {same}")
        if not same:
            say("  differences:", sorted(set(mem.items()) ^ set(fl.items()))[:10])
    fl = filelist(raw / "GSE192742" / "filelist.txt")
    h5 = sorted((raw / "GSE192742_diet").glob("*.h5"))
    bad = [f.name for f in h5 if fl.get(f.name) != f.stat().st_size]
    ok_all &= not bad
    say(f"GSE192742 diet h5: {len(h5)} files, size matches filelist: {not bad} {bad}")
    with tarfile.open(raw / "GSE129516" / "GSE129516_RAW.tar") as t:
        for m in sorted(t.getmembers(), key=lambda m: m.name):
            with gzip.open(t.extractfile(m), "rt") as f:
                header = f.readline().rstrip("\n").split(",")
            say(f"GSE129516 {m.name}: {len(header) - 1} barcodes")
    say("ALL_MATCH" if ok_all else "MISMATCH")
    (out / "verify_downloads.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
