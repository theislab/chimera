"""Convert every sheet of an .xlsx file to TSV with the standard library only.

Usage: python -I xlsx_to_tsv.py <file.xlsx> <outdir>
Writes <outdir>/<stem>__<sheet>.tsv, one per sheet. Values as stored (shared strings resolved,
numbers as written in the XML, inline strings kept). Formulas are not evaluated: the cached value is used.
"""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "pr": "http://schemas.openxmlformats.org/package/2006/relationships"}


def col_index(ref):
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def text_of(el):
    return "".join(t.text or "" for t in el.iter("{%s}t" % NS["m"]))


def main(path, outdir):
    path, outdir = Path(path), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        shared = [text_of(si) for si in root.findall("m:si", NS)]
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels.findall("pr:Relationship", NS)}
    for sh in wb.find("m:sheets", NS).findall("m:sheet", NS):
        name = sh.get("name")
        rid = sh.get("{%s}id" % NS["r"])
        t = target[rid].lstrip("/")
        t = t if t.startswith("xl/") else "xl/" + t
        root = ET.fromstring(z.read(t))
        rows = []
        for row in root.iter("{%s}row" % NS["m"]):
            vals = {}
            for c in row.findall("m:c", NS):
                ref, typ = c.get("r"), c.get("t")
                v = c.find("m:v", NS)
                if typ == "s" and v is not None:
                    val = shared[int(v.text)]
                elif typ == "inlineStr":
                    val = text_of(c)
                else:
                    val = v.text if v is not None else ""
                vals[col_index(ref)] = (val or "").replace("\t", " ").replace("\n", " ").replace("\r", " ")
            if vals:
                width = max(vals) + 1
                rows.append([vals.get(i, "") for i in range(width)])
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        out = outdir / f"{path.stem}__{safe}.tsv"
        with open(out, "w") as fh:
            for r in rows:
                fh.write("\t".join(r) + "\n")
        print(f"{out}\t{len(rows)} rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
