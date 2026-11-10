"""Compose a figure from panel files on a grid, at the size it has on the page of the paper.

A spec is a tree of ``Row``/``Col``/``Panel``. Each panel is a PDF (or a page of a PDF, optionally a region of it). The
tool finds its ink box by rendering at 300 dpi, crops with vector-preserving ``\\includegraphics`` trim and clip, scales
it to the grid, and writes one PDF through pdflatex, so panel text stays text and panel fonts stay embedded. The page of
that PDF is the figure: it is included in the paper at its natural size.

    figures/composed/<key>.pdf     the figure
    figures/preview/<key>.png      preview at 150 dpi

Layout. Every node has height ``h = a * w + b`` at width ``w``, so a tree is solved exactly:

    Row(a, b, ...)                    children get the same height and fill the width (aligned top, bottom, edges)
    Row(a, b, ..., same_scale=True)   leaf panels share one scale, bottoms and labels on one baseline
    Col(a, b, ...)                    children get the same width and stack
    Row(..., width=300)               a row narrower than the figure, centered

Panels. ``P("family/name", "a")`` is ``figures/panels/family/name.pdf`` cropped to its ink; ``X(...)`` keeps the whole
page of a panel that was saved at its exact size, so axes drawn at the same place line up to the point; ``NB("notebook/name")``
is a panel in ``notebooks/figures/``; ``SRC("name")`` is a file in ``figures/sources/``. Panel letters are drawn by this tool,
bold, under the panel; a panel file carries no letter of its own.

Run from the repository root:

    PYTHONPATH=$PWD python -m figures.compose main_validation        # one or more keys
    PYTHONPATH=$PWD python -m figures.compose --list

The build (compose all, merge, crop, manifest, checks) is ``figures/build.sh``.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from figures.style import blank_file_names, machine_paths

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent  # .../figures
REPO = Path(os.environ.get("CHIMERA_ROOT", ROOT.parent))
OUT = Path(os.environ.get("CHIMERA_FIGURES_OUT", ROOT))  # outputs; set it for a dry run, inputs do not move
PANELS = ROOT / "panels"
SOURCES = ROOT / "sources"
NOTEBOOK_PANELS = REPO / "notebooks" / "figures"
COMPOSED = OUT / "composed"
PREVIEW = OUT / "preview"

TEXTWIDTH = 486.0  # bp, 6.75 in, \textwidth of the manuscript
COLUMNWIDTH = 234.0  # bp, 3.25 in
LANDSCAPE = 648.0  # bp, 9.0 in, \linewidth of a landscape page
LABEL_SIZE = 7.0  # bp, bold (a) under each panel
LABEL_H = 9.5  # bp, band under a panel that holds its label
GAP = 7.0  # bp, between panels in a row
VGAP = 6.0  # bp, between rows
DEJAVU_BBOX = 1.164  # pdftotext word height per unit of font size, DejaVu Sans
INK_DPI = 300
INK_THRESH = 250
INK_MARGIN = 0.4  # bp left around the ink
PREVIEW_DPI = 150


def tool(name: str) -> str:
    """Path of a poppler tool or ghostscript: ``POPPLER_BIN`` if set, else the first on ``PATH``."""
    d = os.environ.get("POPPLER_BIN")
    if d and name != "gs" and (Path(d) / name).exists():
        return str(Path(d) / name)
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"{name} not found on PATH; activate chimera_figures_env (figures/environment.yml)")
    return p


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def ink_box(pdf: Path, page: int = 1, region: tuple | None = None) -> tuple:
    """Ink box (x0, y0, x1, y1, page_w, page_h) in bp from the page's top left, inside ``region`` if given."""
    with tempfile.TemporaryDirectory() as td:
        stem = Path(td) / "p"
        r = _run([tool("pdftoppm"), "-r", INK_DPI, "-cropbox", "-f", page, "-l", page, "-singlefile", "-png", pdf, stem])
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
    o = _run([tool("pdftotext"), "-bbox", "-f", page, "-l", page, pdf, "-"]).stdout
    return [(float(m[0]), float(m[1]), float(m[2]), float(m[3]), html.unescape(m[4])) for m in
            re.findall(r'<word xMin="([\d.-]+)" yMin="([\d.-]+)" xMax="([\d.-]+)" yMax="([\d.-]+)">(.*?)</word>', o)]


def gs_spans(pdf: Path, page: int) -> list:
    """Text spans from ghostscript txtwrite: [x0, y0, x1, y1, font size, text, oblique, font name]."""
    o = _run([tool("gs"), "-q", "-dNOPAUSE", "-dBATCH", "-dUseCropBox", f"-dFirstPage={page}", f"-dLastPage={page}",
              "-sDEVICE=txtwrite", "-dTextFormat=0", "-sOutputFile=-", pdf]).stdout
    out = []
    for m in re.finditer(r'<span bbox="([\d.-]+) ([\d.-]+) ([\d.-]+) ([\d.-]+)" font="([^"]*)" size="([\d.]+)">(.*?)</span>',
                         o, re.S):
        x0, y0, x1, y1 = [float(v) for v in m.groups()[:4]]
        text = html.unescape("".join(re.findall(r'c="(.*?)"/>', m.group(7))))
        out.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), float(m.group(6)), text, False, m.group(5)])
    # Oblique text comes one glyph per span, each glyph moving in both x and y: mark such runs.
    for a, b in zip(out, out[1:]):
        if len(a[5]) == 1 and len(b[5]) == 1 and a[4] == b[4]:
            dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
            if 0.15 * a[4] < dx < 2 * a[4] and 0.15 * a[4] < dy < 2 * a[4]:
                a[6] = b[6] = True
    return out


@dataclass
class Panel:
    src: str | Path  # "family/name" in figures/panels, "nb:notebook/name", "src:name", or an absolute path
    label: str | dict | None = None  # "a", or {"a": x, "b": x} with x in bp on the source page, for a region
    # that holds several panels with no gap between them
    page: int = 1
    region: tuple | None = None  # (x0, y0, x1, y1) in bp from the page's top left; ink is found inside it
    pad: tuple = (0, 0, 0, 0)  # white added (left, bottom, right, top), bp at the panel's natural size
    ink: bool = True  # False: keep the whole page (a panel saved at its exact size, for exact grid alignment)
    box: tuple = field(default=None, init=False)
    place_: tuple = field(default=None, init=False)
    scale: float = field(default=None, init=False)

    def path(self) -> Path:
        s = str(self.src)
        if s.startswith("nb:"):
            q = NOTEBOOK_PANELS / (s[3:] + ".pdf")
        elif s.startswith("src:"):
            q = SOURCES / (s[4:] + ".pdf")
        elif Path(s).is_absolute():
            q = Path(s)
        else:
            q = PANELS / (s if s.endswith(".pdf") else s + ".pdf")
        if not q.exists():
            raise FileNotFoundError(q)
        return q

    def name(self) -> str:
        """Short name for reports and the manifest: the path relative to the repository."""
        p = self.path()
        for base in (ROOT.parent, REPO):
            try:
                return p.relative_to(base).as_posix()
            except ValueError:
                continue
        return p.as_posix()

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

    def letters(self) -> list:
        return list(self.label) if isinstance(self.label, dict) else ([self.label] if self.label else [])

    def leaves(self) -> list:
        return [self]


class LabelledPage(Panel):
    """A whole page cropped to its ink and padded back to the full page (plus the label band), with letters at given
    positions on the source page instead of under the panel: ``at = {"d": (center x, bottom of the letter box)}`` in bp
    from the page's top left. The page is embedded once."""

    def __init__(self, src, at: dict, bottom: float, **kw):
        super().__init__(src, label=None, **kw)
        self.at, self.bottom = at, bottom

    def load(self) -> None:
        super().load()
        x0, y0, x1, y1, w, h = self.box
        self.pad = (x0, self.bottom - y1, w - x1, y0)

    def place(self, x: float, y: float, w: float) -> None:
        super().place(x, y, w)
        s, (l, b, r, t) = self.scale, self.pad
        x0, y0 = self.box[:2]
        self.labels_at = [(k, x + (l + cx - x0) * s, y + (t + yb - 1.5 - y0) * s - LABEL_H + 1.2)
                          for k, (cx, yb) in self.at.items()]

    def letters(self) -> list:
        return list(self.at)


def P(src: str, label=None, **kw) -> Panel:  # noqa: N802
    """A panel in figures/panels, cropped to its ink."""
    return Panel(src, label=label, **kw)


def X(src: str, label=None, **kw) -> Panel:  # noqa: N802
    """A panel saved at its exact size: the page is kept, so axes share a baseline in a same-scale row."""
    return Panel(src, label=label, ink=False, **kw)


def NB(src: str, label=None, **kw) -> Panel:  # noqa: N802
    """A panel rendered by a notebook: notebooks/figures/<notebook>/<name>.pdf."""
    return Panel("nb:" + src, label=label, **kw)


def SRC(src: str, label=None, **kw) -> Panel:  # noqa: N802
    """A file in figures/sources (a page that is not composed from panels)."""
    return Panel("src:" + src, label=label, **kw)


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
    width: float = TEXTWIDTH  # bp at print size; LANDSCAPE for a figure on a landscape page
    shrink: float = 1.0  # the whole laid-out figure, letters and gaps included, scaled by this factor. Only to reproduce a
    # figure that used to be scaled in LaTeX; the checks list every figure that uses it
    label: str = ""  # the \label of the figure in the paper; a continuation float shares it
    part: int = 1  # 1 for the float that carries the label, 2, 3, ... for its continuation floats
    title: str = ""  # short neutral title for the index
    shows: str = ""  # what the figure shows, in plain language, for the index
    note: str = ""  # layout note for maintainers


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
    s = fig.shrink
    open_, close_ = (rf"\scalebox{{{s:.6f}}}{{", "}") if s != 1.0 else ("", "")
    PW, PH = W * s, H * s
    return rf"""\documentclass{{article}}
\pdfminorversion=7
\pdfinfoomitdate=1
\pdftrailerid{{}}
\pdfsuppressptexinfo=-1
\usepackage[T1]{{fontenc}}
\usepackage{{graphicx}}
\pdfsuppresswarningpagegroup=1
\paperwidth={PW:.3f}bp
\paperheight={PH:.3f}bp
\pdfpagewidth={PW:.3f}bp
\pdfpageheight={PH:.3f}bp
\hoffset=-1in \voffset=-1in \oddsidemargin=0pt \topmargin=0pt \headheight=0pt \headsep=0pt \topskip=0pt \footskip=0pt
\textwidth={PW:.3f}bp
\textheight={PH:.3f}bp
\parindent=0pt
\pagestyle{{empty}}
\setlength{{\unitlength}}{{1bp}}
\begin{{document}}
{open_}\begin{{picture}}({W:.3f},{H:.3f})(0,0)
{lines}
\end{{picture}}{close_}
\end{{document}}
"""


def layout(fig: Figure) -> tuple:
    """Load the panels, solve the tree, place everything. Returns (width, height) of the layout in bp, before ``shrink``."""
    fig.tree.load()
    a, b = fig.tree.coeffs()
    W = fig.width
    H = a * W + b
    fig.tree.place(0.0, 0.0, W)
    return W, H


def text_spans(fig: Figure) -> tuple:
    """Visible text of a placed figure, panel by panel, so text hidden by a crop is not counted.

    Returns (spans, words). A span is a run of glyphs in one font size as ghostscript reports it: a dict with size (font
    size in pt at print size), text, n (characters), panel, font and single. ``single`` marks a character that stands
    alone (a subscript, a star, a dagger): the span belongs to a word of one character, where words are those of
    ``pdftotext``. Words are the visible words of each panel, for the checks on names and spelling."""
    spans, words = [], []
    for p in fig.tree.leaves():
        x0, y0, x1, y1 = p.box[:4]
        ws = [w for w in bbox_words(p.path(), p.page) if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
        words += [w[4] for w in ws]
        wb = np.array([w[:4] for w in ws], dtype=float).reshape(-1, 4)
        wlen = np.array([len(w[4]) for w in ws], dtype=int)
        warea = (wb[:, 2] - wb[:, 0]) * (wb[:, 3] - wb[:, 1]) if len(ws) else np.zeros(0)
        for sx0, sy0, sx1, sy1, size, text, obl, font in gs_spans(p.path(), p.page):
            cx, cy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
            n = len(text)
            single = n < 2 and not obl
            if single and len(ws):
                hit = np.where((wb[:, 0] - 0.5 <= cx) & (cx <= wb[:, 2] + 0.5) & (wb[:, 1] - 0.5 <= cy) & (cy <= wb[:, 3] + 0.5))[0]
                if len(hit):
                    single = bool(wlen[hit[np.argmin(warea[hit])]] < 2)
            spans.append(dict(size=round(size * p.scale * fig.shrink, 2), text=text, n=max(n, 1), panel=p.name(), page=p.page,
                              single=single, font=re.sub(r"^[A-Z]{6}\+", "", font)))
    return spans, words


def font_problems(pdf: Path) -> list:
    """Problems reported by pdffonts: Type 3 or not embedded."""
    rows = _run([tool("pdffonts"), pdf]).stdout.splitlines()[2:]
    bad = []
    for ln in rows:
        if " Type 3 " in ln:
            bad.append("Type 3: " + ln[:40].strip())
        parts = ln[36:].split()
        if len(parts) >= 5 and parts[-5] != "yes":
            bad.append("not embedded: " + ln[:40].strip())
    return bad


def compose(key: str, fig: Figure, verbose: bool = True) -> dict:
    """Write composed/<key>.pdf and preview/<key>.png; return the measured facts of the figure."""
    COMPOSED.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    W, H = layout(fig)
    pdf = COMPOSED / f"{key}.pdf"
    env = dict(os.environ, SOURCE_DATE_EPOCH="0", FORCE_SOURCE_DATE="1")
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "c.tex").write_text(_tex(fig, W, H))
        r = _run(["pdflatex", "-interaction=nonstopmode", "-no-shell-escape", "c.tex"], cwd=td, env=env)
        built = Path(td) / "c.pdf"
        if not built.exists() or re.search(r"^!", r.stdout, re.M):
            raise RuntimeError(f"pdflatex failed for {key}:\n" + "\n".join(re.findall(r"^!.*", r.stdout, re.M)[:5]))
        pdf.write_bytes(built.read_bytes())
    blank_file_names(pdf)
    _run([tool("pdftoppm"), "-r", PREVIEW_DPI, "-singlefile", "-png", pdf, PREVIEW / key])

    info = _run([tool("pdfinfo"), pdf]).stdout
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    pw, ph = [float(v) for v in re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups()]
    leaves = fig.tree.leaves()
    k = fig.shrink
    row = dict(key=key, width_bp=round(W * k, 3), height_bp=round(H * k, 3), shrink=k, pages=pages, page_w=pw, page_h=ph,
               panels=[dict(letters=p.letters(), file=p.name(), page=p.page, region=p.region, scale=round(p.scale * k, 4),
                            x=round(p.place_[0], 2), y=round(p.place_[1], 2), w=round(p.place_[2], 2), h=round(p.place_[3], 2),
                            exact=not p.ink, clipped=bool(getattr(p, "clipped", False))) for p in leaves],
               font_problems=font_problems(pdf), absolute_paths=machine_paths(pdf, deep=False))
    row["spans"], row["words"] = text_spans(fig)
    if verbose:
        sizes = sorted(s["size"] for s in row["spans"] if not s["single"])
        print(f"{key}: {W * k:.0f} x {H * k:.0f} bp, {len(leaves)} panels, scale {min(p.scale for p in leaves) * k:.3f}-"
              f"{max(p.scale for p in leaves) * k:.3f}, smallest text {sizes[0] if sizes else '-'} pt")
    return row


def main() -> int:
    from figures import registry

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    figs = registry.load(in_paper_only=False)
    if a.list or not a.keys:
        for k, f in figs.items():
            print(f"{k:<40} {f.label:<42} part {f.part}  {f.title}")
        return 0 if a.list else 2
    for k in a.keys:
        compose(k, figs[k])
    return 0


if __name__ == "__main__":
    sys.exit(main())
