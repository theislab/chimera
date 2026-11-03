"""Fig 1 page text in the binding vocabulary until the owner re-exports the page from Keynote (decision D12).

The overview figure is page 2 of figures_planning_cropped.pdf (Keynote export, Roboto). Phrases that the paper no longer
supports ("expression conservation", "cis-regulatory divergence", "conserved regulatory programs", "tissue-specific sign
shifts", "evolutionary divergence", "Rhesus" for the liver macaque) are deleted from the page's content stream, so no old
text stays under the new, and new text is set at the same places in Helvetica (the near-match available here).

    pdfseparate -f 2 -l 2 figures_planning_cropped.pdf p2.pdf
    python fig1_overlay.py p2.pdf <out_dir>          # writes fig1_overview_overlay.pdf
"""
import re
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])

# Text objects to delete, by their decoded text, in page order (verified: each occurs once).
DELETE = ["Rhesus", "macaque", ": Expression conservation is per", "gene#title",
          "expression#app", "conservation#app", "pathway", "evolutionary", "divergence",
          "tissue", "-#ts", "specific#ts", "sign shifts",
          "expression more divergent", "than sequence predicts;", "cis ", "-#cis", "regulatory divergence;",
          "signed expression", "-#s1", "vs", "-#s2", "sequence conservation residual",
          "expression more", "conserved than sequence", "predicts; conserved", "regulatory programs",
          "c#a", "c#da", "reports", "expression  conservation  relative  to  protein  sequence."]
# Block indices on the page (decoded order), checked against DELETE below.
INDEX = [103, 104, 85, 87, 8, 9, 12, 13, 14, 15, 16, 17, 18, 39, 40, 41, 42, 43, 49, 50, 51, 52, 53,
         94, 95, 96, 97, 201, 204, 205, 206]


def text_of(b: bytes) -> str:
    out, i = [], 0
    while i < len(b):
        if b[i:i + 1] == b"(":
            depth, j, s = 1, i + 1, bytearray()
            while depth:
                ch = b[j:j + 1]
                if ch == b"\\":
                    s += b[j + 1:j + 2]; j += 2; continue
                if ch == b"(": depth += 1
                elif ch == b")":
                    depth -= 1
                    if depth == 0: break
                s += ch; j += 1
            out.append(bytes(s)); i = j + 1
        else:
            i += 1
    return b"".join(out).decode("latin-1")


d = src.read_bytes()
objs = {int(m.group(1)): m.group(2) for m in re.finditer(rb"(?s)(\d+) 0 obj(.*?)endobj", d)}
page = [b for b in objs.values() if re.search(rb"/Type /Page\s", b)][0]
cn = int(re.search(rb"/Contents (\d+) 0 R", page).group(1))
o = objs[cn]
c = zlib.decompress(o[o.index(b"stream") + 6:o.rindex(b"endstream")].lstrip(b"\r\n"))
blocks = list(re.finditer(rb"(?s)BT.*?ET", c))
assert len(blocks) == 210, len(blocks)
assert len(DELETE) == len(INDEX) + 0 or True
for k, name in zip(INDEX, DELETE):
    want = name.split("#")[0]
    assert text_of(blocks[k].group(0)) == want, (k, text_of(blocks[k].group(0)), want)
# also unit 86 is the "-" of "per-gene" in the title; delete it with the title
INDEX = sorted(set(INDEX + [86]))
assert text_of(blocks[86].group(0)) == "-"
cut = bytearray()
last = 0
for k in INDEX:
    m = blocks[k]
    cut += c[last:m.start()]
    last = m.end()
cut += c[last:]
z = zlib.compress(bytes(cut), 9)
objs[cn] = b"\n<</Filter /FlateDecode /Length %d >>\nstream\n" % len(z) + z + b"\nendstream\n"
tr = d[d.rfind(b"trailer"):]
root = re.search(rb"/Root (\d+) 0 R", tr).group(1)
info = re.search(rb"/Info (\d+) 0 R", tr).group(1)
size = max(objs) + 1
buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
off = {}
for n in sorted(objs):
    off[n] = len(buf)
    buf += b"%d 0 obj" % n + objs[n] + b"endobj\n"
x = len(buf)
buf += b"xref\n0 %d\n" % size + b"0000000000 65535 f \n"
for n in range(1, size):
    buf += (b"%010d 00000 n \n" % off[n]) if n in off else b"0000000000 65535 f \n"
buf += b"trailer\n<< /Size %d /Root %s 0 R /Info %s 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, root, info, x)

W, H = 1017.0, 545.0
S, T = 11.5, 13.0  # body and title sizes (Roboto boxes 13.5 and 15.2 bp high)
GRAY = "0.10"


def line(cx: float, y_max: float, s: str, size: float = S, align: str = "c", bold: bool = False) -> str:
    base = H - (y_max - 0.24 * size)
    box = {"c": "[b]", "l": "[bl]"}[align]
    f = r"\bfseries" if bold else ""
    return (rf"\put({cx:.1f},{base:.1f}){{\makebox(0,0){box}{{\fontsize{{{size}}}{{{size}}}\selectfont{f}"
            rf"\color[gray]{{{GRAY}}}{s}}}}}")


puts = [
    line(112.6, 92.3, "Macaque"),
    line(78.0, 28.0, ": Deployment is per gene", T, "l"),
    line(937.3, 493.5, "deployment"), line(937.3, 506.6, "vs protein"),
    line(683.4, 380.8, "pathway"), line(683.4, 393.5, "-level"), line(683.4, 406.6, "patterns"),
    line(863.2, 325.3, "cross-tissue", align="l"), line(863.2, 338.4, "comparison", align="l"),
    line(126.5, 478.0, "count evidence below"), line(126.5, 492.4, "the protein prior:"),
    line(126.5, 505.2, "pattern or level parts;"),
    line(494.2, 481.4, "count evidence above"), line(494.2, 494.5, "the protein prior:"),
    line(494.2, 507.3, "pattern shared"), line(494.2, 520.4, "across species"),
    line(314.2, 540.7, "signed place of count evidence against the protein prior"),
    line(918.7, 528.0, "places", align="l"),
    line(680.6, 539.3, "shared cell-state pattern relative to the protein prior.", align="l"),
]
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "stripped.pdf").write_bytes(buf)
    tex = (r"\documentclass{article}\usepackage[T1]{fontenc}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}"
           r"\usepackage{graphicx,xcolor}\paperwidth=%sbp\paperheight=%sbp\pdfpagewidth=%sbp\pdfpageheight=%sbp\hoffset=-1in\voffset=-1in"
           r"\oddsidemargin=0pt\topmargin=0pt\headheight=0pt\headsep=0pt\topskip=0pt\textwidth=%sbp\textheight=%sbp"
           r"\parindent=0pt\pagestyle{empty}\setlength{\unitlength}{1bp}\begin{document}\begin{picture}(%s,%s)(0,0)"
           r"\put(0,0){\includegraphics{stripped.pdf}}" "\n%s\n" r"\end{picture}\end{document}"
           % (W, H, W, H, W, H, W, H, "\n".join(puts)))
    (td / "f.tex").write_text(tex)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "f.tex"], cwd=td, capture_output=True, text=True)
    assert (td / "f.pdf").exists(), r.stdout[-2000:]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fig1_overview_overlay.pdf").write_bytes((td / "f.pdf").read_bytes())
print("written", out_dir / "fig1_overview_overlay.pdf")
