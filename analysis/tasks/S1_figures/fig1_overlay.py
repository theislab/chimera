"""Fig 1 page text in the binding vocabulary until the owner re-exports the page from Keynote (decision D12).

The overview figure is page 2 of figures_planning_cropped.pdf (Keynote export, Roboto 2.138). Text the paper no longer
supports, and text the export garbled, is deleted from the page's content stream, so no old text stays under the new;
the new text is set at the same places in Roboto itself (the same font files as the export: glyph widths match), at the
export's sizes (11.49 bp body, 12.93 bp titles, 7.18 bp subscripts), in black like the original.

Deleted and re-set:
  - phrases: "expression conservation", "cis-regulatory divergence", "conserved regulatory programs", "tissue-specific
    sign shifts", "evolutionary divergence", "Rhesus" for the liver macaque; the old closing sentence of (e)
  - (c) the gate equation, exported as "α_g = o(logit π_π + c_κ)", now α_g = σ(logit π_g + c_g); the prior, exported
    twice as "π_α", now π_g; "α_g (per-gene gate)" under the gauges, now "α_g (protein-feature gate)" (the paper calls
    the default gate the protein-feature gate; "per-gene gate" names the companion read-out)
  - stray spaces of the export: "ESM -2" (twice), "multi -species", "1: 1 orthologs", "α -gating"
  - (d) the histogram's y label, exported in three pieces ("ge", "ne", "s"), now one rotated word
  - (e) "Δa" in the panel title (a Latin a), now "Δα"
Also removed: three small white image boxes that the export drew under α in (c) and (e) (visible as box glyphs), and
the faint grey patches that the export left in the background picture where text boxes had been: background pixels
that are a uniform shade at most 7 levels darker than the background (254, 254, 250) are set to the background.
Nothing else on the page changes.

    pdfseparate -f 2 -l 2 figures_planning_cropped.pdf p2.pdf
    python fig1_overlay.py p2.pdf <out_dir> [--roboto DIR]     # writes <out_dir>/fig1_overview_overlay.pdf

Roboto: DIR (or $ROBOTO_DIR) holding Roboto-{Regular,Italic,Bold,BoldItalic}.ttf of Roboto 2.138; without it the
release zip is fetched from github.com/googlefonts/roboto (Apache 2.0) into results/analysis/S1/roboto and checked
against its SHA-256. Needs lualatex with fontspec.
"""
import argparse
import hashlib
import io
import os
import re
import subprocess
import tempfile
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("out_dir", type=Path)
ap.add_argument("--roboto", type=Path, default=os.environ.get("ROBOTO_DIR"))
args = ap.parse_args()
src, out_dir = args.src, args.out_dir

ROBOTO_URL = "https://github.com/googlefonts/roboto/releases/download/v2.138/roboto-unhinted.zip"
ROBOTO_ZIP_SHA = "70f64c718510a601fbcf752aafe644314dacaeb85474dc689c89787c4a72a728"
FACES = ["Roboto-Regular.ttf", "Roboto-Italic.ttf", "Roboto-Bold.ttf", "Roboto-BoldItalic.ttf"]


def roboto_dir() -> Path:
    if args.roboto:
        d = Path(args.roboto)
    else:
        d = Path(__file__).resolve().parents[3] / "results" / "analysis" / "S1" / "roboto"
        if not all((d / f).exists() for f in FACES):
            data = urllib.request.urlopen(ROBOTO_URL, timeout=120).read()
            assert hashlib.sha256(data).hexdigest() == ROBOTO_ZIP_SHA, "Roboto zip checksum"
            d.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for f in FACES:
                    (d / f).write_bytes(z.read(f))
    assert all((d / f).exists() for f in FACES), d
    return d


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


def stream(o: bytes) -> bytes:
    return zlib.decompress(o[o.index(b"stream") + 6:o.rindex(b"endstream")].lstrip(b"\r\n"))


d = src.read_bytes()
objs = {int(m.group(1)): m.group(2) for m in re.finditer(rb"(?s)(\d+) 0 obj(.*?)endobj", d)}
page = [b for b in objs.values() if re.search(rb"/Type /Page\s", b)][0]
assert b"/MediaBox [450 268 1467 813 ]" in page
cn = int(re.search(rb"/Contents (\d+) 0 R", page).group(1))
c = stream(objs[cn])
blocks = list(re.finditer(rb"(?s)BT.*?ET", c))
assert len(blocks) == 210, len(blocks)

W, H = 1017.0, 545.0
X0, Y0 = 450.0, 268.0  # MediaBox origin


def anchor(k: int) -> tuple:
    """Start of text block k on the page: x from the left, baseline from the top (bp); from its cm and Tm."""
    pre = c[:blocks[k].start()]
    a, b, cc, dd, e, f = [float(v) for v in list(re.finditer(rb"([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)\s+cm", pre))[-1].groups()]
    tx, ty = [float(v) for v in re.search(rb"[-\d.]+ [-\d.]+ [-\d.]+ [-\d.]+ ([-\d.]+) ([-\d.]+)\s+Tm", blocks[k].group(0)).groups()]
    x, y = a * tx + cc * ty + e, b * tx + dd * ty + f
    return x - X0, H - (y - Y0)


# Text blocks to delete, by their decoded text (each checked). "#" separates a tag where the same text recurs.
DELETE = {
    # phrases outside the binding vocabulary (first overlay)
    103: "Rhesus", 104: "macaque", 85: ": Expression conservation is per", 86: "-", 87: "gene",
    8: "expression", 9: "conservation", 12: "pathway", 13: "evolutionary", 14: "divergence",
    15: "tissue", 16: "-", 17: "specific", 18: "sign shifts",
    39: "expression more divergent", 40: "than sequence predicts;", 41: "cis ", 42: "-", 43: "regulatory divergence;",
    49: "signed expression", 50: "-", 51: "vs", 52: "-", 53: "sequence conservation residual",
    94: "expression more", 95: "conserved than sequence", 96: "predicts; conserved", 97: "regulatory programs",
    # (e) closing sentence, whole: "α_c identifies genes supporting alignment; Δα_c reports expression conservation ..."
    200: '"', 201: "c", 202: "identifies  genes  supporting  alignment;  ", 203: '!"', 204: "c", 205: "reports",
    206: "expression  conservation  relative  to  protein  sequence.",
    # (c) the equation "α_g = o(logit π_π + c_κ)"
    191: "!", 192: "g", 193: "= o(logit  ", 194: "#", 195: "!", 196: "+  ", 197: "c", 198: '"', 199: ")",
    # the prior exported as "π_α", twice
    3: "!", 4: "!", 133: "!", 134: "!",
    # "α_g (per-gene gate)" under the gauges
    186: "!", 187: "g", 188: "(per", 189: "-", 190: "gene  gate)",
    # stray spaces: "ESM -2" (b), "ESM -2 features" (c), "multi -species", "1: 1 orthologs", "α -gating"
    88: "ESM ", 89: "-", 90: "2", 110: "ESM ", 111: "-", 112: "2 features", 118: "multi ", 119: "-", 120: "species",
    155: "1: 1 orthologs", 59: "!", 60: " ", 61: "-", 62: "gating",
    # (d) histogram y label in three pieces
    180: "ge", 181: "ne", 182: "s",
    # (e) title "Δa"
    47: "!", 48: "a",
}
for k, want in DELETE.items():
    assert text_of(blocks[k].group(0)) == want, (k, text_of(blocks[k].group(0)), want)
POS = {k: anchor(k) for k in DELETE}

# Small white image boxes drawn under α in (c) and (e): "q <matrix> cm /Im2 Do Q" (twice) and the /Im3 one.
BOXES = [rb"q 32\.39867 0 0 29\.986 1227\.592 541\.389 cm /Im2 Do Q", rb"q 32\.39867 0 0 29\.986 1216\.622 596\.3919\s+cm /Im2 Do Q",
         rb"q 30\.67533 0 0 30\.67533 1116\.758 273\.0339 cm /Im3 Do\s+Q"]

cut, last = bytearray(), 0
for k in sorted(DELETE):
    m = blocks[k]
    cut += c[last:m.start()]
    last = m.end()
cut += c[last:]
cut = bytes(cut)
for pat in BOXES:
    cut, n = re.subn(pat, b"", cut)
    assert n == 1, pat
z = zlib.compress(cut, 9)
objs[cn] = b"\n<</Filter /FlateDecode /Length %d >>\nstream\n" % len(z) + z + b"\nendstream\n"

# Background picture (Im1, JPEG): the export's pale text-box patches (near-neutral, 244-249) to the background (251).
res = objs[int(re.search(rb"/Resources (\d+) 0 R", page).group(1))]
im1 = int(re.search(rb"/Im1 (\d+) 0 R", res).group(1))
o = objs[im1]
head = o[:o.index(b"stream")]
assert b"/DCTDecode" in head and b"/Width 3000 /Height 1674" in head
jpg = o[o.index(b"stream") + 6:o.rindex(b"endstream")].lstrip(b"\r\n")
a = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB")).astype(np.int16)
BG = np.array([254, 254, 250], dtype=np.int16)  # the picture's background (median over empty areas)
dev = BG[None, None, :] - a  # how much darker than the background, per channel
patch = (dev.min(axis=2) >= -1) & (dev.max(axis=2) <= 7) & (dev.max(axis=2) - dev.min(axis=2) <= 3)
a[patch] = BG
# Raster copies of old labels that the removed white boxes hid (α_g of the equation, the subscript under the gauge
# label, α of the (e) sentence) and a stray "#" beside the histogram label: those areas (page bp, x0 x1 y0 y1 from
# the top; the boxes' own extents, the first clipped above the gauges) are set to the background.
IMG_X, IMG_TOP, IMG_W, IMG_H = 432.2805 - X0, H - (268.5653 - Y0 + 576.972), 1034.0, 576.972
for x0, x1, y0, y1 in [(766.6, 799.0, 186.6, 210.0), (777.6, 810.0, 241.6, 271.6), (666.8, 697.4, 509.3, 540.0),
                       (207.0, 221.0, 473.0, 485.5)]:
    c0, c1 = [int(round((v - IMG_X) * 3000 / IMG_W)) for v in (x0, x1)]
    r0, r1 = [int(round((v - IMG_TOP) * 1674 / IMG_H)) for v in (y0, y1)]
    ring = np.concatenate([a[r0 - 8:r0, c0:c1].reshape(-1, 3), a[r1:r1 + 8, c0:c1].reshape(-1, 3),
                           a[r0:r1, c0 - 8:c0].reshape(-1, 3), a[r0:r1, c1:c1 + 8].reshape(-1, 3)])
    ring = ring[ring.min(axis=1) >= 240]  # the background around the area, not artwork
    a[r0:r1, c0:c1] = np.median(ring, axis=0).round().astype(np.int16)
buf_j = io.BytesIO()
Image.fromarray(a.astype(np.uint8)).save(buf_j, format="JPEG", quality=95, subsampling=0, optimize=True)
jpg2 = buf_j.getvalue()
head2 = re.sub(rb"/Length \d+", b"/Length %d" % len(jpg2), head)
objs[im1] = head2 + b"stream\n" + jpg2 + b"\nendstream\n"

tr = d[d.rfind(b"trailer"):]
root = re.search(rb"/Root (\d+) 0 R", tr).group(1)
info = re.search(rb"/Info (\d+) 0 R", tr).group(1)
size = max(objs) + 1
buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
off = {}
for n in sorted(objs):
    off[n] = len(buf)
    buf += b"%d 0 obj" % n + objs[n] + b"endobj\n"
xref = len(buf)
buf += b"xref\n0 %d\n" % size + b"0000000000 65535 f \n"
for n in range(1, size):
    buf += (b"%010d 00000 n \n" % off[n]) if n in off else b"0000000000 65535 f \n"
buf += b"trailer\n<< /Size %d /Root %s 0 R /Info %s 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, root, info, xref)

S, T, SUB, SMALL = 11.489, 12.925, 7.180, 8.617  # export sizes: 33.33, 37.5, 20.83, 25 x 0.3446667


def put(x: float, base: float, s: str, size: float = S, align: str = "l", rot: int = 0) -> str:
    """Text with its baseline at `base` (bp from the top); x is its left end, centre or right end."""
    box = {"l": r"\rlap{%s}", "c": r"\hbox to 0pt{\hss %s\hss}", "r": r"\llap{%s}"}[align]
    body = box % (r"\fontsize{%.3f}{%.3f}\selectfont %s" % (size, size, s))
    if rot:
        body = r"\rotatebox{%d}{%s}" % (rot, body)
    return r"\put(%.2f,%.2f){%s}" % (x, H - base, body)


def line(cx: float, y_max: float, s: str, size: float = S, align: str = "c") -> str:
    """First overlay's placement: centre (or left end) and the bottom of the old word box (pdftotext yMax)."""
    return put(cx, y_max - 0.244 * size, s, size, {"c": "c", "l": "l"}[align])


def sub(s: str) -> str:
    return r"\raisebox{-2.76bp}{\fontsize{%.3f}{%.3f}\selectfont %s}" % (SUB, SUB, s)


eq_x, eq_b = POS[191]
puts = [
    # phrases (first overlay), now in Roboto
    line(112.6, 92.3, "Macaque"),
    put(POS[85][0], POS[85][1], ": Deployment is per gene", T),
    line(937.3, 493.5, "deployment"), line(937.3, 506.6, "vs protein"),
    line(683.4, 380.8, "pathway"), line(683.4, 393.5, "-level"), line(683.4, 406.6, "patterns"),
    line(863.2, 325.3, "cross-tissue", align="l"), line(863.2, 338.4, "comparison", align="l"),
    line(126.5, 478.0, "count evidence below"), line(126.5, 492.4, "the protein prior:"),
    line(126.5, 505.2, "pattern or level parts;"),
    line(494.2, 481.4, "count evidence above"), line(494.2, 494.5, "the protein prior:"),
    line(494.2, 507.3, "pattern shared"), line(494.2, 520.4, "across species"),
    line(314.2, 540.7, "signed place of count evidence against the protein prior"),
    # (e) closing sentence, two lines, left edge and baselines of the export's lines
    put(POS[200][0], POS[200][1], r"\textbf{\textit{α}} identifies genes supporting alignment; \textbf{Δα} places"),
    put(POS[200][0], POS[206][1], "shared cell-state pattern relative to the protein prior."),
    # (c) the gate equation and the prior
    put(eq_x, eq_b, r"\textit{α}%s = σ(logit \textit{π}%s + \textit{c}%s)" % (sub(r"\textit{g}"), sub(r"\textit{g}"), sub(r"\textit{g}"))),
    put(POS[3][0], POS[3][1], "π" + r"\kern0.4bp" + sub("g")),
    put(POS[133][0], POS[133][1], "π" + r"\kern0.4bp" + sub("g")),
    put(833.4, POS[186][1], r"\textit{α}%s (protein-feature gate)" % sub(r"\textit{g}"), align="c"),
    # stray spaces
    put((378.56 + 413.38) / 2, POS[88][1], "ESM-2", align="c"),
    put(601.05, POS[110][1], "ESM-2 features", align="r"),
    put((332.46 + 401.59) / 2, POS[118][1], "multi-species", align="c"),
    put((333.34 + 401.77) / 2, POS[155][1], "1:1 orthologs", align="c"),
    put(POS[59][0], POS[59][1], r"\textit{α}-gating", T),
    # (d) histogram y label, one rotated word, centred on the three pieces
    put(225.0, 480.2 + 11.0, "genes", SMALL, rot=90),
    # (e) title
    put(POS[47][0], POS[47][1], "Δα", T),
]

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "stripped.pdf").write_bytes(buf)
    rd = roboto_dir()
    tex = (r"\documentclass{article}\usepackage{fontspec}\usepackage{graphicx}"
           r"\setmainfont{Roboto}[Path=%s/,UprightFont=Roboto-Regular.ttf,ItalicFont=Roboto-Italic.ttf,"
           r"BoldFont=Roboto-Bold.ttf,BoldItalicFont=Roboto-BoldItalic.ttf]"
           r"\paperwidth=%sbp\paperheight=%sbp\pagewidth=%sbp\pageheight=%sbp\hoffset=-1in\voffset=-1in"
           r"\oddsidemargin=0pt\topmargin=0pt\headheight=0pt\headsep=0pt\topskip=0pt\textwidth=%sbp\textheight=%sbp"
           r"\parindent=0pt\pagestyle{empty}\setlength{\unitlength}{1bp}\begin{document}\begin{picture}(%s,%s)(0,0)"
           r"\put(0,0){\includegraphics{stripped.pdf}}" "\n%s\n" r"\end{picture}\end{document}"
           % (rd, W, H, W, H, W, H, W, H, "\n".join(puts)))
    (td / "f.tex").write_text(tex, encoding="utf-8")
    r = subprocess.run(["lualatex", "-interaction=nonstopmode", "f.tex"], cwd=td, capture_output=True, text=True)
    assert (td / "f.pdf").exists(), r.stdout[-3000:]
    assert "Missing character" not in (td / "f.log").read_text(errors="replace"), "glyph missing in Roboto"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fig1_overview_overlay.pdf").write_bytes((td / "f.pdf").read_bytes())
print("written", out_dir / "fig1_overview_overlay.pdf")
