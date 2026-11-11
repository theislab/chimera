"""Checks of the figures: the house style on every composed figure, and pixel comparison with a reference build.

House style (run by the build on every figure):

    text      no text under 5.0 pt font size at print size; single characters (subscripts, stars) not under 3.5 pt
    fonts     all embedded, none of Type 3
    paths     no trace of a working machine (path, user name) in any panel, composed figure, asset or in figures.pdf
    page      one page of the computed size, not wider than the text (or the landscape page)
    scale     no panel placed below 0.9 of its drawn size; no figure shrunk as a whole
    letters   the letters of a figure and its continuation floats run a, b, c ... in reading order
    words     no internal name and no British spelling in the text of the artwork

A figure that does not pass yet is listed in ``specs/_allow.py`` with the reason; the build prints the list every time
and ``--strict`` fails on it. A figure that is not listed must pass.

Comparison with a reference (``python -m figures.check --against reference.json``): every page is rendered so that one
pixel is 1/150 inch at print size, cut to its ink and compared.

    identical    ink boxes equal within 1 pixel and no pixel differs by more than 2 gray levels
    equivalent   for a page that is placed by another mechanism than in the reference (scaled by the layout tool where
                 LaTeX scaled it before), so that the two renders can differ by a fraction of a pixel. Both pages are
                 rendered four times finer, cut to their ink there and averaged down. Then: ink boxes equal within
                 1 pixel, and no more than 10 "orphan" pixels, none of them in a group of more than 6. A pixel is an
                 orphan when no pixel of the other page within 1 pixel has a value within 96 gray levels. A shift by a
                 fraction of a pixel makes no orphans; a missing label, an added mark or a part moved by 2 pixels does.
                 Calibrated on the 28 pages that were placed both ways: 27 have no orphan; one has a group of 5 at the
                 end of one word whose box agrees to 0.004 bp in the two files (glyph rasterisation). Damage for
                 comparison: 16 orphans for a few erased strokes, 105 for a 12 x 8 pixel mark, 736 for a block moved
                 by 2 pixels.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from figures import compose as C

FLOOR = 5.0  # pt, font size at print size
FLOOR_SINGLE = 3.5  # pt, single characters: subscripts, superscripts, stars, daggers
MIN_SCALE = 0.9
DPI = 150

INTERNAL = [r"panc_\w*", r"liv_h\w*", r"test_h\w*", r"\w*_4sp\b", r"\bB1b\b", r"\bB1c\b", r"\bB2b\b", r"\bB3a\b", r"\bB1z\b", r"\bSdir\b",
            r"\bLfc\b", r"\b[a-z]{2,}(?:_[a-z0-9]+)+\b", r"\bcanonical\b", r"\bfree gate\b", r"\bgolds?\b", r"\bown table\b",
            r"\bstage_\w+", r"\brevision\b"]
BRITISH = ["centred", "programme", "specialised", "signalling", "labelled", "tumour", "neighbour", "modelled", "colour", "normalised",
           "harmonised", "analysed", "behaviour", "favour", "ischaemi", "haem", "fibre", "in brackets", "read-out"]


# ----------------------------------------------------------------------------------------------------- house style
def style_problems(row: dict, fig: C.Figure) -> dict:
    """The checks on one composed figure (``row`` is what ``compose.compose`` returned). name -> message."""
    out = {}
    spans = row["spans"]
    body = [s for s in spans if not s["single"]]
    single = [s for s in spans if s["single"]]
    low = [s for s in body if s["size"] < FLOOR - 0.005]
    n_all = sum(s["n"] for s in body) or 1
    if low:
        ex = sorted(low, key=lambda s: s["size"])[:3]
        out["text"] = (f"{sum(s['n'] for s in low) / n_all:.0%} of the characters under {FLOOR} pt, smallest {min(s['size'] for s in low)} pt ("
                       + "; ".join(f"{s['size']} {s['text'][:18]!r}" for s in ex) + ")")
    low1 = [s for s in single if s["size"] < FLOOR_SINGLE - 0.005]
    if low1:
        out["text_single"] = f"{len(low1)} single characters under {FLOOR_SINGLE} pt, smallest {min(s['size'] for s in low1)} pt"
    if row["font_problems"]:
        out["fonts"] = "; ".join(sorted(set(row["font_problems"]))[:4])
    if row.get("absolute_paths"):
        out["paths"] = "the file carries traces of a working machine: " + ", ".join(row["absolute_paths"])
    families = {re.sub(r"[-,].*$", "", s["font"]) for s in spans if s["font"]} - {"DejaVuSans", "DejaVuSansMono", "DejaVuSerif"}
    families = {f for f in families if not f.startswith("DejaVu")}
    if families:
        out["font_family"] = "fonts other than DejaVu in the panels: " + ", ".join(sorted(families)[:5])
    if row["pages"] != 1:
        out["page"] = f"{row['pages']} pages"
    elif abs(row["page_w"] - row["width_bp"]) > 1 or abs(row["page_h"] - row["height_bp"]) > 1:
        out["page"] = f"page {row['page_w']:.0f} x {row['page_h']:.0f}, expected {row['width_bp']:.0f} x {row['height_bp']:.0f}"
    if row["width_bp"] > C.LANDSCAPE + 0.5 or (row["width_bp"] > C.TEXTWIDTH + 0.5 and abs(row["width_bp"] - C.LANDSCAPE) > 0.5):
        out["width"] = f"{row['width_bp']:.0f} bp wide"
    small = [p for p in row["panels"] if p["scale"] < MIN_SCALE - 0.0005]
    if row.get("shrink", 1.0) != 1.0:
        out["scale"] = f"the whole figure is shrunk to {row['shrink']}"
    elif small:
        out["scale"] = f"{len(small)} of {len(row['panels'])} panels below scale {MIN_SCALE}, smallest {min(p['scale'] for p in small):.3f}"
    clipped = [p["file"] for p in row["panels"] if p["clipped"]]
    if clipped:
        out["edge"] = "ink reaches the page edge of " + ", ".join(sorted({Path(c).name for c in clipped})[:4])
    words = " ".join(row.get("words", []))
    hits = sorted({m.group(0) for pat in INTERNAL for m in re.finditer(pat, words)})
    if hits:
        out["words"] = "internal names: " + ", ".join(hits[:12]) + (" ..." if len(hits) > 12 else "")
    brit = sorted({b for b in BRITISH if b in words.lower()})
    if brit:
        out["spelling"] = "British spelling: " + ", ".join(brit)
    undecodable = sum(1 for s in body if s["text"] and not re.search(r"[A-Za-z0-9]", s["text"]))
    if body and undecodable > 0.5 * len(body):
        out["unreadable_text_layer"] = "the text layer cannot be decoded (hand-assembled page): words and spelling are not checked"
    return out


def machine_path_scan(root: Path | None = None, jobs: int = 4) -> list:
    """Every file that is kept with the figures (panels, composed figures, assets of the redraw scripts, the single PDF,
    the tables) searched for traces of a working machine; PDFs also in their uncompressed form. Returns (file, hits)."""
    from concurrent.futures import ThreadPoolExecutor

    from figures import style

    out_root = C.OUT
    root = root or C.ROOT
    files = [p for p in sorted((root / "panels").rglob("*")) if p.is_file()]
    files += [p for p in sorted(C.COMPOSED.glob("*.pdf"))]
    files += [p for d in sorted((root / "redraw").rglob("assets")) for p in sorted(d.rglob("*")) if p.is_file()]
    files += [p for p in (out_root / "figures.pdf", out_root / "manifest.tsv", out_root / "manifest.json", out_root / "INDEX.md",
                          root / "panel_names.tsv", root / "producers.tsv", root / "order.txt", root / "numbers.tsv") if p.exists()]
    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(style.machine_paths, files))
    return [(str(p.relative_to(root.parent)) if root.parent in p.parents else str(p), h) for p, h in zip(files, res) if h]


def letter_problems(rows: dict, figs: dict) -> dict:
    """Letters over a figure and its continuation floats: a, b, c ... in reading order. key -> message.

    Reading order on one float: the next letter is never on a panel that lies wholly above the panel before, and never on
    one to its left in the same band (two panels are in one band when they overlap in height) unless that panel starts
    and ends lower, which begins a new row. A tall panel beside a column of two reads tall panel, upper, lower; two
    columns of unequal panels read row by row. Several letters on one panel run left to right."""
    out, by_label = {}, {}
    for k, f in figs.items():
        if k in rows:
            by_label.setdefault(f.label, []).append(k)
    for label, keys in by_label.items():
        seq = []                                    # (key, letter, x, top, bottom) in the order of the letters
        for k in keys:
            on_float = [(L, p["x"] + i * 1e-3, p["y"], p["y"] + p["h"]) for p in rows[k]["panels"] for i, L in enumerate(p["letters"])]
            seq += [(k, *t) for t in sorted(on_float)]
        letters = [L for _, L, *_ in seq]
        if not letters:
            continue
        want = [chr(ord("a") + i) for i in range(len(letters))]
        if letters != want:
            out[keys[0]] = f"letters {''.join(letters)}: not a, b, c ... without gap or repeat"
            continue
        tol = 2.0                                   # bp
        for (k1, a, x1, t1, b1), (k2, b, x2, t2, b2) in zip(seq, seq[1:]):
            if k1 != k2:
                continue                            # the next float starts a new page
            above = b2 <= t1 + tol
            same_band = t2 < b1 - tol and t1 < b2 - tol
            new_row = t2 > t1 + tol and b2 > b1 + tol
            if above or (same_band and x2 < x1 and not new_row):
                out[k1] = f"letters not in reading order: ({b}) is {'above' if above else 'left of'} ({a})"
                break
    return out


# ----------------------------------------------------------------------------------------------------- pixels
def render(pdf: Path, page: int = 1, dpi: float = DPI) -> np.ndarray:
    """A page as a gray image (uint8), rendered with ghostscript."""
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "p.png"
        r = subprocess.run([C.tool("gs"), "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pnggray", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                            "-dUseCropBox", f"-r{dpi:.5f}", f"-dFirstPage={page}", f"-dLastPage={page}", f"-sOutputFile={png}", str(pdf)],
                           capture_output=True, text=True)
        if not png.exists():
            raise RuntimeError(f"ghostscript failed on {pdf} page {page}: {r.stderr[:300]}")
        return np.asarray(Image.open(png).convert("L"))


def ink(a: np.ndarray, thresh: int = 250) -> tuple:
    ys, xs = np.where(a < thresh)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else (0, 0, a.shape[1], a.shape[0])


SUPER = 4  # the tolerant comparison renders this many times finer
ORPHAN_LEVELS, ORPHAN_MAX, ORPHAN_GROUP = 96, 10, 6


def _cut(a: np.ndarray) -> np.ndarray:
    b = ink(a)
    return a[b[1]:b[3], b[0]:b[2]]


def _down(a: np.ndarray, n: int) -> np.ndarray:
    h, w = (a.shape[0] // n) * n, (a.shape[1] // n) * n
    return a[:h, :w].reshape(h // n, n, w // n, n).mean(axis=(1, 3))


def _orphans(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pixels of A with no pixel of B within 1 pixel whose value is within ORPHAN_LEVELS."""
    Bp = np.pad(B, 1, mode="edge")
    h, w = A.shape
    best = np.full(A.shape, 255.0)
    for dy in range(3):
        for dx in range(3):
            best = np.minimum(best, np.abs(A - Bp[dy:dy + h, dx:dx + w]))
    return best > ORPHAN_LEVELS


def _largest_group(mask: np.ndarray) -> int:
    pts = {(int(y), int(x)) for y, x in zip(*np.where(mask))}
    best = 0
    while pts:
        stack, n = [pts.pop()], 0
        while stack:
            y, x = stack.pop()
            n += 1
            for q in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1), (y + 1, x + 1), (y - 1, x - 1), (y + 1, x - 1), (y - 1, x + 1)):
                if q in pts:
                    pts.remove(q)
                    stack.append(q)
        best = max(best, n)
    return best


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact comparison of two renders after cutting each to its ink box."""
    A, B = _cut(a).astype(np.int16), _cut(b).astype(np.int16)
    dsize = max(abs(A.shape[1] - B.shape[1]), abs(A.shape[0] - B.shape[0]))
    h, w = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
    d = np.abs(A[:h, :w] - B[:h, :w])
    res = dict(size_a=(A.shape[1], A.shape[0]), size_b=(B.shape[1], B.shape[0]), size_diff_px=int(dsize),
               max_diff=int(d.max()) if d.size else 0, share_over_2=float((d > 2).mean()) if d.size else 0.0)
    res["identical"] = bool(dsize <= 1 and res["max_diff"] <= 2)
    res["verdict"] = "identical" if res["identical"] else "not identical"
    return res


def compare_tolerant(a: np.ndarray, b: np.ndarray) -> dict:
    """Comparison that forgives a shift by a fraction of a pixel; ``a`` and ``b`` are rendered SUPER times finer."""
    A, B = _down(_cut(a), SUPER), _down(_cut(b), SUPER)
    dsize = max(abs(A.shape[1] - B.shape[1]), abs(A.shape[0] - B.shape[0]))
    h, w = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
    A, B = A[:h, :w], B[:h, :w]
    m = _orphans(A, B) | _orphans(B, A)
    n = int(m.sum())
    group = _largest_group(m) if 0 < n <= 20000 else n
    ok = bool(dsize <= 1 and n <= ORPHAN_MAX and group <= ORPHAN_GROUP)
    return dict(size_a=(A.shape[1], A.shape[0]), size_b=(B.shape[1], B.shape[0]), size_diff_px=int(dsize), orphans=n, largest_group=int(group),
                mean_diff=float(np.abs(A - B).mean()), equivalent=ok, verdict="equivalent" if ok else "DIFFERENT")


def against(reference: Path, keys: list | None, new_dir: Path, out_tsv: Path | None, diff_dir: Path | None) -> int:
    """Compare ``new_dir/<key>.pdf`` with the pages of a reference table: a JSON list of dicts with key, file (or
    source with root), page and scale (print width over page width)."""
    ref = json.loads(Path(reference).read_text())
    root = Path(reference).parent
    rows, bad = [], 0
    for r in ref:
        k = r["key"]
        if keys and k not in keys:
            continue
        new = new_dir / f"{k}.pdf"
        if not new.exists():
            rows.append(dict(key=k, verdict="MISSING"))
            bad += 1
            continue
        src = Path(r["file"]) if "file" in r else root / r["root"] / r["source"]
        scale, page, new_page = r.get("scale", 1.0), r.get("page", 1), r.get("new_page", 1)
        a = render(src, page, DPI * scale)
        b = render(new, new_page, DPI)
        c = compare(a, b)
        row = dict(key=k, expect=r.get("expect", ""), verdict=c["verdict"], size_px=f"{c['size_a'][0]}x{c['size_a'][1]}",
                   size_new_px=f"{c['size_b'][0]}x{c['size_b'][1]}", max_diff=c["max_diff"], share_over_2=f"{c['share_over_2']:.5f}", orphans="", largest_group="")
        if not c["identical"]:
            t = compare_tolerant(render(src, page, DPI * scale * SUPER), render(new, new_page, DPI * SUPER))
            row.update(verdict=t["verdict"], orphans=t["orphans"], largest_group=t["largest_group"])
            if diff_dir is not None:
                diff_dir.mkdir(parents=True, exist_ok=True)
                A, B = _cut(a), _cut(b)
                h, w = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
                D = (255 - np.clip(np.abs(A[:h, :w].astype(np.int16) - B[:h, :w].astype(np.int16)) * 4, 0, 255)).astype(np.uint8)
                (diff_dir / f"{k}.png").parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.concatenate([A[:h, :w], B[:h, :w], D], axis=1)).save(diff_dir / f"{k}.png")
        rows.append(row)
        if row["verdict"] == "DIFFERENT" or (r.get("expect") == "identical" and row["verdict"] != "identical"):
            bad += 1
        print(f"{k:<40} {row['verdict']:<10} {row['size_px']:>10} vs {row['size_new_px']:<10} max diff {row['max_diff']:>3}  "
              f">2: {float(row['share_over_2']):.3%}  orphans {row['orphans']}  group {row['largest_group']}", flush=True)
    if out_tsv:
        cols = ["key", "expect", "verdict", "size_px", "size_new_px", "max_diff", "share_over_2", "orphans", "largest_group"]
        out_tsv.write_text("\t".join(cols) + "\n" + "\n".join("\t".join(str(r.get(c, "")) for c in cols) for r in rows) + "\n")
    n = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("identical", "equivalent", "DIFFERENT", "MISSING")}
    print("compared", len(rows), n)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--against", type=Path, required=True, help="reference table (JSON)")
    ap.add_argument("--new", type=Path, default=C.COMPOSED, help="folder with <key>.pdf (default: figures/composed)")
    ap.add_argument("--tsv", type=Path, default=None)
    ap.add_argument("--diff", type=Path, default=None, help="folder for side-by-side images of pages that are not identical")
    ap.add_argument("keys", nargs="*")
    a = ap.parse_args()
    return against(a.against, a.keys or None, a.new, a.tsv, a.diff)


if __name__ == "__main__":
    sys.exit(main())
