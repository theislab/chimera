"""Read one worksheet of an .xlsx file with the standard library only.

The analysis environments carry no Excel reader. Cells are placed by their column
reference, so empty cells keep their position.
"""

from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

_M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def sheet_names(path: Path) -> list[str]:
    z = zipfile.ZipFile(path)
    return [s.get("name") for s in ET.fromstring(z.read("xl/workbook.xml")).iter(f"{_M}sheet")]


def read_sheet(path: Path, name: str | None = None) -> list[list[str]]:
    """Rows of sheet ``name`` (default: the first sheet) as lists of strings."""
    z = zipfile.ZipFile(path)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = ["".join(t.text or "" for t in si.iter(f"{_M}t"))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{_M}si")]
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    sheets = list(ET.fromstring(z.read("xl/workbook.xml")).iter(f"{_M}sheet"))
    sh = sheets[0] if name is None else next(s for s in sheets if s.get("name") == name)
    target = rels[sh.get(f"{_R}id")].lstrip("/")
    target = target if target.startswith("xl/") else f"xl/{target}"
    rows = []
    for r in ET.fromstring(z.read(target)).iter(f"{_M}row"):
        vals: dict[int, str] = {}
        for i, c in enumerate(r.iter(f"{_M}c")):
            j = _col(c.get("r")) if c.get("r") else i
            v = c.find(f"{_M}v")
            if v is None:
                is_ = c.find(f"{_M}is")
                vals[j] = "".join(t.text or "" for t in is_.iter(f"{_M}t")) if is_ is not None else ""
            else:
                vals[j] = ss[int(v.text)] if c.get("t") == "s" else v.text
        rows.append([vals.get(k, "") for k in range(max(vals) + 1)] if vals else [])
    return rows
