"""Minimal .xlsx reader (standard library only): sheet name -> list of rows of cell strings.

Used for the Cippa et al. 2018 supplementary tables, so no spreadsheet package is added to
any environment. Reads shared strings, inline strings and numbers; formulas give their cached
value. Run with ``python -I`` on downloaded files.

    python -I revision/tasks/P13_kidney/xlsx_min.py <file.xlsx>      # prints sheet sizes
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
import zipfile

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "pr": "http://schemas.openxmlformats.org/package/2006/relationships"}


def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx(path: str) -> dict[str, list[list[str]]]:
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = {r.get("Id"): r.get("Target") for r in rels.findall("pr:Relationship", NS)}
        out: dict[str, list[list[str]]] = {}
        for sh in wb.find("m:sheets", NS).findall("m:sheet", NS):
            rid = sh.get(f"{{{NS['r']}}}id")
            t = target[rid].lstrip("/")
            t = t if t.startswith("xl/") else "xl/" + t
            root = ET.fromstring(z.read(t))
            rows: list[list[str]] = []
            for row in root.iter(f"{{{NS['m']}}}row"):
                cells: dict[int, str] = {}
                for c in row.findall("m:c", NS):
                    typ, v = c.get("t"), c.find("m:v", NS)
                    if typ == "s" and v is not None:
                        val = shared[int(v.text)]
                    elif typ == "inlineStr":
                        val = "".join(x.text or "" for x in c.iter(f"{{{NS['m']}}}t"))
                    else:
                        val = v.text if v is not None and v.text is not None else ""
                    cells[_col_index(c.get("r"))] = val
                if cells:
                    width = max(cells) + 1
                    rows.append([cells.get(i, "") for i in range(width)])
                else:
                    rows.append([])
            out[sh.get("name")] = rows
    return out


if __name__ == "__main__":
    for name, rows in read_xlsx(sys.argv[1]).items():
        print(f"{name}\t{len(rows)} rows\t{max((len(r) for r in rows), default=0)} cols")
