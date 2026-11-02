"""Compose a figure from cropped panels on a grid, in the style of ``figures_planning_cropped.pdf``.

A spec is a tree of ``Row``/``Col``/``Panel``. Each panel is a PDF (or a page of a PDF, optionally a
region of it). The tool finds its ink box by rendering at 300 dpi, crops with vector-preserving
``\\includegraphics`` trim and clip, scales it to the grid, and writes one PDF through pdflatex, so
panel text stays text and panel fonts stay embedded:

    ``<manuscript>/figures/revision/<fig>_composed.pdf``   the figure, at its size on the page
    ``results/revision/F1/<fig>_composed.png``             preview at 200 dpi
    ``results/revision/F1/report.csv``                     one row per figure (updated in place)

Layout. Every node has height ``h = a * w + b`` at width ``w``, so a tree is solved exactly:

    Row(a, b, ...)             children get the same height and fill the width (aligned top, bottom, edges)
    Row(a, b, ..., same_scale=True)   leaf panels share one scale, bottoms and labels on one baseline
    Col(a, b, ...)             children get the same width and stack
    Row(..., width=300)        a row narrower than the figure, centred

Text check, at the final size. ``pdftotext -bbox`` gives the height of each word; DejaVu Sans has a
height of 1.164 x font size. A figure fails when a word of two or more characters is under ``floor``
(default 5 pt, about a 4.3 pt font; the originals in print have 5.5 pt at the 7 pt font of page 16).
One-character words are subscripts, stars and fragments of oblique labels: they have their own
floor (0.7 x ``floor``), read from ghostscript's font sizes, because ``pdftotext`` boxes of oblique
text are unreliable. ``exempt_font=`` on a panel skips text at or below that source font size; use it
only for an unchanged page of the originals; it is reported. Also fails on unembedded or Type 3 fonts.

Exact grids. A panel saved at its exact size (``common.save(fig, name, exact=True)``) and placed with
``Panel(..., ink=False)`` keeps its page, so axes drawn at the same place line up to the point.

Run from ``chimera/``:

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env \\
        python -m revision.tasks.F1_figures.compose fig4 fig5        # names in specs.py
    ... compose --spec my_specs.py myfig                             # FIGS in another file
    ... compose --all
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO = Path(__file__).resolve().parents[3]
MS = Path(os.environ.get("CHIMERA_MANUSCRIPT", REPO.parent / "chimera-manuscript"))
FIG_DIR = MS / "figures" / "revision"
ORIGINALS = MS / "figures" / "figures_planning_cropped.pdf"
OUT = REPO / "results" / "revision" / "F1"
POPPLER = Path(os.environ.get("POPPLER_BIN", Path.home() / "tools/apps/mamba/envs/tardis_env/bin"))

TEXTWIDTH = 486.0  # bp, 6.75 in, \textwidth of the manuscript
LABEL_SIZE = 7.0  # bp, bold (a) under each panel; the originals print at about 6 to 7
LABEL_H = 9.5  # bp, band under a panel that holds its label
GAP = 7.0  # bp, between panels in a row
VGAP = 6.0  # bp, between rows
FLOOR = 5.0  # bp, smallest pdftotext -bbox word height
DEJAVU_BBOX = 1.164  # pdftotext word height per unit of font size, DejaVu Sans
INK_DPI = 300
INK_THRESH = 250
INK_MARGIN = 0.4  # bp left around the ink


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def ink_box(pdf: Path, page: int = 1, region: tuple | None = None) -> tuple:
    """Ink box (x0, y0, x1, y1, page_w, page_h) in bp from the page's top left, inside ``region`` if given."""
    with tempfile.TemporaryDirectory() as td:
        stem = Path(td) / "p"
        r = _run([POPPLER / "pdftoppm", "-r", INK_DPI, "-cropbox", "-f", page, "-l", page, "-singlefile", "-png",
                  pdf, stem])
        if r.returncode:
            raise RuntimeError(f"pdftoppm failed on {pdf}: {r.stderr}")
        a = np.asarray(Image.open(f"{stem}.png").convert("RGB"))
    u = 72.0 / INK_DPI
    if region is not None:
        x0, y0, x1, y1 = [int(round(v / u)) for v in region]
        sub = a[y0:y1, x0:x1]
    else:
        x0, y0 = 0, 0
        sub = a
    mask = (sub < INK_THRESH).any(axis=2)
    if not mask.any():
        raise RuntimeError(f"no ink in {pdf} page {page} region {region}")
    ys, xs = np.where(mask)
    H, W = a.shape[:2]
    return (max(0.0, (xs.min() + x0) * u - INK_MARGIN), max(0.0, (ys.min() + y0) * u - INK_MARGIN),
            min(W * u, (xs.max() + 1 + x0) * u + INK_MARGIN), min(H * u, (ys.max() + 1 + y0) * u + INK_MARGIN),
            W * u, H * u)


def bbox_words(pdf: Path, page: int) -> list:
    """Words on a page: (x0, y0, x1, y1, text), bp from the page's top left."""
    o = _run([POPPLER / "pdftotext", "-bbox", "-f", page, "-l", page, pdf, "-"]).stdout
    return [(float(m[0]), float(m[1]), float(m[2]), float(m[3]), html.unescape(m[4])) for m in
            re.findall(r'<word xMin="([\d.-]+)" yMin="([\d.-]+)" xMax="([\d.-]+)" yMax="([\d.-]+)">(.*?)</word>', o)]


def gs_spans(pdf: Path, page: int) -> list:
    """Font-size spans from ghostscript txtwrite: [x0, y0, x1, y1, size, n_chars, oblique]."""
    o = _run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dUseCropBox", f"-dFirstPage={page}", f"-dLastPage={page}",
              "-sDEVICE=txtwrite", "-dTextFormat=0", "-sOutputFile=-", pdf]).stdout
    out = []
    for m in re.finditer(r'<span bbox="([\d.-]+) ([\d.-]+) ([\d.-]+) ([\d.-]+)" font="[^"]*" size="([\d.]+)">(.*?)</span>',
                         o, re.S):
        x0, y0, x1, y1, size = [float(v) for v in m.groups()[:5]]
        out.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), size, m.group(6).count("<char"), False])
    # Oblique text comes one glyph per span, each glyph moving in both x and y: mark such runs.
    for a, b in zip(out, out[1:]):
        if a[5] == 1 and b[5] == 1 and a[4] == b[4]:
            dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
            if 0.15 * a[4] < dx < 2 * a[4] and 0.15 * a[4] < dy < 2 * a[4]:
                a[6] = b[6] = True
    return out


@dataclass
class Panel:
    src: str | Path
    label: str | dict | None = None  # "a", or {"a": x, "b": x} with x in bp on the source page, for a region
    # that holds several panels with no gap between them
    page: int = 1
    region: tuple | None = None  # (x0, y0, x1, y1) in bp from the page's top left; ink is found inside it
    pad: tuple = (0, 0, 0, 0)  # white added (left, bottom, right, top), bp at the panel's natural size
    exempt_font: float | None = None  # unchanged original page only: text at or below this source font size
    # is reported but not held to the floor
    ink: bool = True  # False: keep the whole page (a panel saved at its exact size, for exact grid alignment)
    box: tuple = field(default=None, init=False)
    place_: tuple = field(default=None, init=False)
    scale: float = field(default=None, init=False)

    def path(self) -> Path:
        p = Path(self.src)
        if p.is_absolute() and p.exists():
            return p
        q = FIG_DIR / (p.name if p.suffix else f"{p}.pdf")
        if not q.exists():
            raise FileNotFoundError(q)
        return q

    def load(self) -> None:
        self.box = ink_box(self.path(), self.page, self.region)
        if not self.ink:
            x0, y0, x1, y1, w, h = self.box
            self.clipped = x0 < 0.3 or y0 < 0.3 or x1 > w - 0.3 or y1 > h - 0.3
            self.box = (0.0, 0.0, w, h, w, h)

    def natural(self) -> tuple:
        x0, y0, x1, y1 = self.box[:4]
        l, b, r, t = self.pad
        return x1 - x0 + l + r, y1 - y0 + t + b

    def coeffs(self) -> tuple:
        w, h = self.natural()
        return h / w, LABEL_H if self.label else 0.0

    def place(self, x: float, y: float, w: float) -> None:
        nw, nh = self.natural()
        s = w / nw
        l, b, r, t = self.pad
        self.scale = s
        iw, ih = (self.box[2] - self.box[0]) * s, (self.box[3] - self.box[1]) * s
        self.place_ = (x + l * s, y + t * s, iw, ih)
        self.labels_at = []
        if isinstance(self.label, dict):
            self.labels_at = [(k, x + l * s + (v - self.box[0]) * s, y + nh * s) for k, v in self.label.items()]
        elif self.label:
            self.labels_at = [(self.label, x + l * s + iw / 2, y + nh * s)]

    def leaves(self) -> list:
        return [self]


@dataclass
class _Group:
    children: tuple
    gap: float | None = None

    def leaves(self) -> list:
        return [p for c in self.children for p in c.leaves()]

    def load(self) -> None:
        for c in self.children:
            c.load()


class Col(_Group):
    def __init__(self, *children, gap: float | None = None):
        super().__init__(children, VGAP if gap is None else gap)

    def coeffs(self) -> tuple:
        cs = [c.coeffs() for c in self.children]
        return sum(a for a, _ in cs), sum(b for _, b in cs) + self.gap * (len(cs) - 1)

    def place(self, x: float, y: float, w: float) -> None:
        for c in self.children:
            a, b = c.coeffs()
            c.place(x, y, w)
            y += a * w + b + self.gap


class Row(_Group):
    def __init__(self, *children, gap: float | None = None, same_scale: bool = False, width: float | None = None):
        super().__init__(children, GAP if gap is None else gap)
        self.same_scale = same_scale
        self.width = width

    def _free(self) -> tuple:
        G = self.gap * (len(self.children) - 1)
        if self.same_scale:
            nat = [c.natural() for c in self.children]
            lab = max(LABEL_H if c.label else 0.0 for c in self.children)
            a = max(h for _, h in nat) / sum(w for w, _ in nat)
            return a, lab - a * G
        cs = [c.coeffs() for c in self.children]
        A = sum(1 / a for a, _ in cs)
        B = sum(b / a for a, b in cs)
        return 1 / A, (B - G) / A

    def coeffs(self) -> tuple:
        a, b = self._free()
        return (0.0, a * self.width + b) if self.width else (a, b)

    def place(self, x: float, y: float, w: float) -> None:
        if self.width:
            x += (w - self.width) / 2
            w = self.width
        G = self.gap
        if self.same_scale:
            nat = [c.natural() for c in self.children]
            s = (w - G * (len(nat) - 1)) / sum(n[0] for n in nat)
            top = max(n[1] for n in nat)
            for c, (nw, nh) in zip(self.children, nat):
                c.place(x, y + (top - nh) * s, nw * s)
                x += nw * s + G
            return
        a, b = self._free()
        H = a * w + b
        for c in self.children:
            ca, cb = c.coeffs()
            cw = (H - cb) / ca
            c.place(x, y, cw)
            x += cw + G


@dataclass
class Figure:
    tree: object  # Row, Col or Panel
    width: float = TEXTWIDTH
    max_height: float | None = None  # bp; fail above this
    floor: float = FLOOR
    note: str = ""


def _tex(fig: Figure, W: float, H: float) -> str:
    def lab(s: str) -> str:
        return rf"\fontsize{{{LABEL_SIZE}}}{{{LABEL_SIZE}}}\fontfamily{{phv}}\fontseries{{b}}\selectfont {s}"

    body = []
    for p in fig.tree.leaves():
        x, y, w, h = p.place_
        x0, y0, x1, y1, pw, ph = p.box
        trim = f"{x0:.3f}bp {ph - y1:.3f}bp {pw - x1:.3f}bp {y0:.3f}bp"
        body.append(rf"\put({x:.3f},{H - y - h:.3f}){{\includegraphics[page={p.page},trim={trim},clip,width={w:.3f}bp]"
                    rf"{{{p.path()}}}}}")
        for k, lx, ly in p.labels_at:
            body.append(rf"\put({lx:.3f},{H - ly - LABEL_H + 1.2:.3f}){{\makebox(0,0)[b]{{{lab('(' + k + ')')}}}}}")
    lines = "\n".join(body)
    return rf"""\documentclass{{article}}
\pdfminorversion=7
\usepackage[T1]{{fontenc}}
\usepackage{{graphicx}}
\pdfsuppresswarningpagegroup=1
\paperwidth={W:.3f}bp
\paperheight={H:.3f}bp
\pdfpagewidth={W:.3f}bp
\pdfpageheight={H:.3f}bp
\hoffset=-1in \voffset=-1in \oddsidemargin=0pt \topmargin=0pt \headheight=0pt \headsep=0pt \topskip=0pt \footskip=0pt
\textwidth={W:.3f}bp
\textheight={H:.3f}bp
\parindent=0pt
\pagestyle{{empty}}
\setlength{{\unitlength}}{{1bp}}
\begin{{document}}
\begin{{picture}}({W:.3f},{H:.3f})(0,0)
{lines}
\end{{picture}}
\end{{document}}
"""


def text_stats(fig: Figure) -> dict:
    """Word heights and font sizes at the final size, panel by panel, so hidden cropped text is not counted."""
    body, single, gs_multi, gs_single, exempted = [], [], [], [], []
    for p in fig.tree.leaves():
        x0, y0, x1, y1 = p.box[:4]
        ex_h = p.exempt_font * DEJAVU_BBOX + 0.01 if p.exempt_font else 0.0
        name = p.path().stem + (f"[p{p.page}]" if p.page != 1 else "")
        spans = gs_spans(p.path(), p.page)
        for wx0, wy0, wx1, wy1, t in bbox_words(p.path(), p.page):
            cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
            h = wy1 - wy0
            if any(sp[0] <= wx1 and wx0 <= sp[2] and sp[1] <= wy1 and wy0 <= sp[3] and h < 0.9 * sp[4] * DEJAVU_BBOX
                   for sp in spans):
                continue  # a fragment of oblique text: its box is not its height; ghostscript reads its size below
            if h <= ex_h:
                exempted.append((round(h * p.scale, 2), t, name))
            elif len(t) >= 2:
                body.append((round(h * p.scale, 2), t, name))
            else:
                single.append((round(h * p.scale, 2), t, name))
        for sx0, sy0, sx1, sy1, size, n, obl in spans:
            if not (x0 <= (sx0 + sx1) / 2 <= x1 and y0 <= (sy0 + sy1) / 2 <= y1):
                continue
            if p.exempt_font and size <= p.exempt_font + 0.01:
                continue
            (gs_multi if (n >= 2 or obl) else gs_single).append((round(size * p.scale, 2), name))
    return dict(body=sorted(body), single=sorted(single), gs_multi=sorted(gs_multi), gs_single=sorted(gs_single),
                exempted=sorted(exempted))


def check_fonts(pdf: Path) -> list:
    """Problems reported by pdffonts: Type 3 or not embedded."""
    rows = _run([POPPLER / "pdffonts", pdf]).stdout.splitlines()[2:]
    bad = []
    for ln in rows:
        if " Type 3 " in ln:
            bad.append("Type 3: " + ln[:40].strip())
        parts = ln[36:].split()
        if len(parts) >= 5 and parts[-5] != "yes":
            bad.append("not embedded: " + ln[:40].strip())
    return bad


def compose(name: str, fig: Figure, verbose: bool = True) -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tree.load()
    a, b = fig.tree.coeffs()
    W = fig.width
    H = a * W + b
    fig.tree.place(0.0, 0.0, W)
    pdf = FIG_DIR / f"{name}_composed.pdf"
    with tempfile.TemporaryDirectory() as td:
        tex = Path(td) / "c.tex"
        tex.write_text(_tex(fig, W, H))
        r = _run(["pdflatex", "-interaction=nonstopmode", "-no-shell-escape", "c.tex"], cwd=td)
        built = Path(td) / "c.pdf"
        if not built.exists() or re.search(r"^!", r.stdout, re.M):
            raise RuntimeError(f"pdflatex failed for {name}:\n" + "\n".join(re.findall(r"^!.*", r.stdout, re.M)[:5]))
        pdf.write_bytes(built.read_bytes())
    _run([POPPLER / "pdftoppm", "-r", 200, "-singlefile", "-png", pdf, OUT / f"{name}_composed"])

    s = text_stats(fig)
    problems = check_fonts(pdf)
    info = _run([POPPLER / "pdfinfo", pdf]).stdout
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    pw, ph = [float(v) for v in re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups()]
    if pages != 1:
        problems.append(f"{pages} pages")
    if abs(pw - W) > 1 or abs(ph - H) > 1:
        problems.append(f"page {pw:.0f} x {ph:.0f}, expected {W:.0f} x {H:.0f}")
    problems += [f"ink reaches the page edge of {p.path().name}" for p in fig.tree.leaves()
                 if getattr(p, "clipped", False)]
    if fig.max_height and H > fig.max_height:
        problems.append(f"height {H:.0f} above {fig.max_height:.0f} bp")
    low = [w for w in s["body"] if w[0] < fig.floor]
    if low:
        problems.append(f"{len(low)} words under {fig.floor} pt: " + ", ".join(f"{h} {t!r} ({n})" for h, t, n in low[:6]))
    font_floor = fig.floor / DEJAVU_BBOX
    low = [g for g in s["gs_multi"] if g[0] < font_floor - 0.02]
    if low:
        problems.append(f"{len(low)} text spans under font size {font_floor:.2f}: {low[:3]}")
    low = [g for g in s["gs_single"] if g[0] < 0.7 * font_floor - 0.02]
    if low:
        problems.append(f"{len(low)} single characters under font size {0.7 * font_floor:.2f}: {low[:3]}")
    scales = [p.scale for p in fig.tree.leaves()]
    row = dict(figure=name, width_bp=round(W, 1), height_bp=round(H, 1), n_panels=len(scales),
               scale_min=round(min(scales), 3), scale_max=round(max(scales), 3),
               min_word_height=s["body"][0][0] if s["body"] else "", min_word=s["body"][0][1] if s["body"] else "",
               min_word_panel=s["body"][0][2] if s["body"] else "",
               min_single_font=s["gs_single"][0][0] if s["gs_single"] else "",
               min_font_size=s["gs_multi"][0][0] if s["gs_multi"] else "",
               n_exempted=len(s["exempted"]),
               min_exempted_height=s["exempted"][0][0] if s["exempted"] else "",
               ok=not problems, problems=" | ".join(problems), note=fig.note)
    _report(row)
    if verbose:
        print(f"{name}: {W:.0f} x {H:.0f} bp, {len(scales)} panels, scale {row['scale_min']}-{row['scale_max']}, "
              f"smallest word {row['min_word_height']} pt ({row['min_word']!r}, {row['min_word_panel']}), "
              f"smallest font {row['min_font_size']}, exempted {row['min_exempted_height']} "
              f"{'OK' if not problems else 'FAIL'}")
        for p in fig.tree.leaves():
            print(f"   {str(p.label or '-')[:12]:>12} {p.path().name:<36} scale {p.scale:.3f}  "
                  f"{p.place_[2]:.0f} x {p.place_[3]:.0f}")
        for q in problems:
            print("   !", q)
    return row


def _report(row: dict) -> None:
    path = OUT / "report.csv"
    rows = {}
    if path.exists():
        with path.open() as f:
            rows = {r["figure"]: r for r in csv.DictReader(f)}
    rows[row["figure"]] = row
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        for k in sorted(rows):
            w.writerow({c: rows[k].get(c, "") for c in row})


def _load_specs(path: Path | None) -> dict:
    if path is None:
        from revision.tasks.F1_figures import specs

        return specs.FIGS
    sp = importlib.util.spec_from_file_location("f1_user_spec", path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod.FIGS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--spec", type=Path, default=None, help="Python file defining FIGS = {name: Figure(...)}")
    a = ap.parse_args()
    figs = _load_specs(a.spec)
    names = list(figs) if a.all else a.figures
    if not names:
        print("figures:", ", ".join(figs))
        return 2
    fails = 0
    for n in names:
        fails += not compose(n, figs[n])["ok"]
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
