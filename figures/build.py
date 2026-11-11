"""Build the figures of the paper from the panel files: compose, merge, crop, manifest, checks.

    PYTHONPATH=$PWD python -m figures.build                  # everything
    PYTHONPATH=$PWD python -m figures.build main_validation  # compose these keys again, then merge, crop, manifest
    PYTHONPATH=$PWD python -m figures.build --check          # write nothing: is everything current? exit 1 if not
    PYTHONPATH=$PWD python -m figures.build --strict         # also fail on the figures listed in specs/_allow.py

Steps: (1) every figure of ``figures/specs`` is composed into ``composed/<key>.pdf`` at print size and checked against
the house style; (2) the composed figures are merged in the order of ``order.txt``; (3) the merged file is cropped with
``pdf-crop-margins -v -p 0`` into ``figures.pdf``; (4) the crop is verified page by page (no ink lost, scale unchanged);
(5) ``manifest.tsv``, ``manifest.json`` and ``INDEX.md`` are written; (6) every panel, composed figure, asset and the
single PDF is searched for traces of the machine it was made on (paths, user name), and the build fails on a hit;
(7) every panel of a figure must have its script recorded in ``producers.tsv`` (named by the build, an error with
``--strict``). Nothing is recomputed: the build reads panel files only.

The crop. ``pdf-crop-margins -v -p 0`` with its defaults finds the content of a page on a 150 dpi MuPDF render and
treats every pixel lighter than 191 of 255 as background. On these figures that cuts into light ink (panel frames,
light gray labels and bars at a page edge): on 76 of 85 pages by more than 0.25 bp, by up to 10 bp. Four options make
the same command lossless: ``-t 250`` (only near-white is background), ``-c gr -x 300 -y 300`` (content found on a
300 dpi Ghostscript render) and ``-a -0.5`` (half a point is left around the content). With them no page loses ink and
at most 0.6 bp of white is left on a side. Each page keeps its scale: the tool changes page boxes only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from figures import check as K
from figures import compose as C
from figures import guide, registry

WORK = C.OUT / "_work"
ROWS = WORK / "rows"
SINGLE = C.OUT / "figures.pdf"
MANIFEST_TSV = C.OUT / "manifest.tsv"
MANIFEST_JSON = C.OUT / "manifest.json"
CROP_ARGS = ["-v", "-p", "0", "-t", "250", "-c", "gr", "-x", "300", "-y", "300", "-a", "-0.5"]
CROP_MAX_CUT, CROP_MAX_WHITE = 0.1, 1.5  # bp: how far a new page box may cut into the ink, and how much white it may leave


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def allow() -> dict:
    try:
        from figures.specs import _allow

        return _allow.ALLOW
    except ImportError:
        return {}


def compose_one(key: str, fig: C.Figure) -> dict:
    signature = registry.signature_hash(fig)
    row = C.compose(key, fig, verbose=False)
    row["signature"] = signature
    row["sha256"] = sha(C.COMPOSED / f"{key}.pdf")
    ROWS.mkdir(parents=True, exist_ok=True)
    (ROWS / f"{key}.json").write_text(json.dumps(row))
    return row


def load_row(key: str) -> dict | None:
    p = ROWS / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def merge(keys: list) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "figures_uncropped.pdf"
    cmd = [C.tool("qpdf"), "--deterministic-id", "--empty", "--pages"] + [str(C.COMPOSED / f"{k}.pdf") for k in keys] + ["--", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode not in (0, 3) or not out.exists():  # 3: warnings
        raise RuntimeError("qpdf merge failed: " + r.stderr[:500])
    return out


def crop(src: Path, dst: Path) -> None:
    """pdf-crop-margins -v -p 0 with the options that make it lossless (module docstring): each page is cropped to its
    own content; only the page boxes change."""
    tmp = WORK / "figures_cropped.pdf"
    tmp.unlink(missing_ok=True)
    env = dict(os.environ, PATH=str(Path(C.tool("gs")).parent) + os.pathsep + os.environ.get("PATH", ""))
    r = subprocess.run([C.tool("pdf-crop-margins")] + CROP_ARGS + [str(src), "-o", str(tmp)], capture_output=True, text=True, env=env)
    (WORK / "crop.log").write_text(r.stdout + "\n" + r.stderr)
    if not tmp.exists():
        raise RuntimeError("pdf-crop-margins failed, see " + str(WORK / "crop.log"))
    # Remove the date and tool stamps so that an unchanged build is the same file byte for byte.
    r = subprocess.run([C.tool("qpdf"), "--deterministic-id", "--remove-info", "--remove-metadata", str(tmp), str(dst)], capture_output=True, text=True)
    if not dst.exists() or r.returncode not in (0, 3):
        raise RuntimeError("qpdf normalisation failed: " + r.stderr[:500])


def verify_crop(uncropped: Path, cropped: Path, keys: list, rows: dict) -> list:
    """Each new page box holds all the ink of the page before the crop (found at 300 dpi, any pixel under 250), leaves
    little white around it, and the page count and order are unchanged."""
    from pypdf import PdfReader

    un, cr = PdfReader(str(uncropped)), PdfReader(str(cropped))
    if len(un.pages) != len(keys) or len(cr.pages) != len(keys):
        return [f"page count: {len(un.pages)} before the crop, {len(cr.pages)} after, {len(keys)} figures"]

    def one(i):
        a = K.render(uncropped, i + 1, 300)
        b, u = K.ink(a), 72.0 / 300
        H = a.shape[0] * u
        return i, (b[0] * u, H - b[3] * u, b[2] * u, H - b[1] * u)  # left, bottom, right, top, from the lower left of the page

    bad = []
    with ThreadPoolExecutor(4) as ex:
        for i, ink in ex.map(one, range(len(keys))):
            k = keys[i]
            o = un.pages[i].mediabox
            m, c = cr.pages[i].mediabox, cr.pages[i].cropbox
            box = (max(float(m.left), float(c.left)) - float(o.left), max(float(m.bottom), float(c.bottom)) - float(o.bottom),
                   min(float(m.right), float(c.right)) - float(o.left), min(float(m.top), float(c.top)) - float(o.bottom))
            cut = max(box[0] - ink[0], box[1] - ink[1], ink[2] - box[2], ink[3] - box[3])
            white = max(ink[0] - box[0], ink[1] - box[1], box[2] - ink[2], box[3] - ink[3])
            rows[k]["cropped_w"], rows[k]["cropped_h"] = round(box[2] - box[0], 2), round(box[3] - box[1], 2)
            rows[k]["crop_cut_bp"], rows[k]["crop_white_bp"] = round(max(cut, 0.0), 3), round(white, 3)
            if cut > CROP_MAX_CUT:
                bad.append(f"page {i + 1} ({k}): the crop cuts {cut:.2f} bp into the ink")
            if white > CROP_MAX_WHITE:
                bad.append(f"page {i + 1} ({k}): {white:.2f} bp of white left on a side")
    return bad


def write_manifest(figs: dict, rows: dict, problems: dict, allowed: dict) -> None:
    cols = ["page", "key", "label", "part", "number", "title", "width_bp", "height_bp", "spec", "n_panels", "panels",
            "smallest_text_pt", "share_under_5pt", "checks", "sha256"]
    numbers = guide.numbers()  # printed figure numbers, when the manuscript's sync script has written them
    scripts = guide.producers()  # the script that draws each panel file
    table, full = [], []
    for i, (k, f) in enumerate(figs.items(), 1):
        r = rows[k]
        body = [s for s in r["spans"] if not s["single"]]
        n = sum(s["n"] for s in body) or 1
        under = sum(s["n"] for s in body if s["size"] < K.FLOOR - 0.005) / n
        chk = "ok"
        if problems.get(k):
            chk = "FAIL: " + ", ".join(sorted(problems[k]))
        elif allowed.get(k):
            chk = "listed: " + ", ".join(sorted(allowed[k]))
        table.append(dict(page=i, key=k, label=f.label, part=f.part, number=numbers.get(k, ""), title=f.title,
                          width_bp=r.get("cropped_w", r["width_bp"]), height_bp=r.get("cropped_h", r["height_bp"]),
                          spec=f"specs/{f.module}.py", n_panels=len(r["panels"]),
                          panels="; ".join((("".join(p["letters"]) + ": ") if p["letters"] else "") + p["file"] + (f" p.{p['page']}" if p["page"] != 1 else "")
                                           for p in r["panels"]),
                          smallest_text_pt=min((s["size"] for s in body), default=""), share_under_5pt=f"{under:.2f}", checks=chk, sha256=r["sha256"]))
        panels = [dict(p, script=(guide.producer(p["file"], scripts) or ("", ""))[0]) for p in r["panels"]]
        full.append(dict(table[-1], shows=f.shows, signature=r["signature"], shrink=r.get("shrink", 1.0), composed_size=[r["width_bp"], r["height_bp"]],
                         panels=panels, problems=problems.get(k, {}), listed=allowed.get(k, {})))
    with MANIFEST_TSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    MANIFEST_JSON.write_text(json.dumps(dict(figures_pdf_sha256=sha(SINGLE) if SINGLE.exists() else "", crop=" ".join(["pdf-crop-margins"] + CROP_ARGS),
                                             figures=full), indent=1, ensure_ascii=False) + "\n")


def run_checks(figs: dict, rows: dict) -> tuple:
    """(problems, allowed): key -> {check: message}. A check listed for the key in specs/_allow.py is allowed."""
    al = allow()
    problems, allowed = {}, {}
    letters = K.letter_problems(rows, figs)
    for k, f in figs.items():
        p = K.style_problems(rows[k], f)
        if k in letters:
            p["letters"] = letters[k]
        for name, msg in p.items():
            if name in al.get(k, {}):
                allowed.setdefault(k, {})[name] = f"{msg} [{al[k][name]}]"
            else:
                problems.setdefault(k, {})[name] = msg
        for name in al.get(k, {}):
            if name not in p:
                problems.setdefault(k, {})["stale_allow"] = f"specs/_allow.py lists '{name}' for this figure but the check passes: remove the entry"
    return problems, allowed


def missing_panel(fig: C.Figure) -> str:
    """The first panel file of a figure that does not exist, or ''."""
    for p in fig.tree.leaves():
        try:
            p.path()
        except FileNotFoundError as e:
            return str(e)
    return ""


def signature_or_none(fig: C.Figure) -> str | None:
    return None if missing_panel(fig) else registry.signature_hash(fig)


def current(figs: dict) -> list:
    """What is not up to date, without writing anything."""
    out = []
    man = json.loads(MANIFEST_JSON.read_text()) if MANIFEST_JSON.exists() else {"figures": []}
    by = {f["key"]: f for f in man["figures"]}
    if [f["key"] for f in man["figures"]] != list(figs):
        out.append("manifest.json: the list or order of figures differs from order.txt")
    for k, f in figs.items():
        pdf = C.COMPOSED / f"{k}.pdf"
        if k not in by:
            out.append(f"{k}: not in the manifest")
        elif not pdf.exists():
            out.append(f"{k}: composed/{k}.pdf is missing")
        elif sha(pdf) != by[k]["sha256"]:
            out.append(f"{k}: composed/{k}.pdf differs from the manifest")
        elif signature_or_none(f) is None:
            out.append(f"{k}: a panel file of its spec is missing ({missing_panel(f)})")
        elif signature_or_none(f) != by[k]["signature"]:
            out.append(f"{k}: the spec or a panel file changed since it was composed")
    if not SINGLE.exists():
        out.append("figures.pdf is missing")
    elif man.get("figures_pdf_sha256") != sha(SINGLE):
        out.append("figures.pdf differs from the manifest")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="*", help="compose only these figures again (default: all)")
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 if anything is not current")
    ap.add_argument("--strict", action="store_true", help="fail on the figures listed in specs/_allow.py as well")
    ap.add_argument("--no-single", action="store_true", help="compose and check only; no merge, crop and manifest")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    figs = registry.load(in_paper_only=True)
    extra = {k: f for k, f in registry.load(in_paper_only=False).items() if k not in figs}
    pending = [k for k, f in extra.items() if getattr(f, "pending", False)]
    if a.check:
        stale = current(figs)
        print("\n".join(stale) if stale else f"current: {len(figs)} figures, figures.pdf, manifest")
        if pending:
            print("pending (specified, not yet a float of the paper): " + ", ".join(pending))
        return 1 if stale else 0

    unknown = [k for k in a.keys if k not in figs and k not in extra]
    if unknown:
        print("unknown keys:", ", ".join(unknown))
        return 2
    both = dict(figs, **{k: extra[k] for k in pending})  # pending figures are composed too, so that their owners see them
    # A figure with a missing panel file (its panels are being redrawn) keeps its last composed version and is named.
    broken = {k: missing_panel(f) for k, f in dict(both, **extra).items() if (not a.keys or k in a.keys) and missing_panel(f)}
    todo = [k for k in (a.keys or both) if k not in broken and (
        bool(a.keys) or load_row(k) is None or not (C.COMPOSED / f"{k}.pdf").exists()
        or load_row(k)["signature"] != registry.signature_hash(both[k]))]
    if not a.keys and len(todo) < len(both):
        print(f"{len(both) - len(todo) - len(broken)} figures are current and are not composed again")
    for k, m in broken.items():
        print(f"MISSING PANEL  {k}: {m}" + ("; its last composed version is kept" if (C.COMPOSED / f"{k}.pdf").exists() else ""))
    unusable = [k for k in broken if k in figs and (load_row(k) is None or not (C.COMPOSED / f"{k}.pdf").exists())]
    if unusable and not a.no_single:
        print("cannot build figures.pdf: no composed version of " + ", ".join(unusable))
        return 1
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = {k: ex.submit(compose_one, k, (figs.get(k) or extra[k])) for k in todo}
        failed = []
        for k, fu in futs.items():
            try:
                r = fu.result()
                body = [s["size"] for s in r["spans"] if not s["single"]]
                print(f"composed {k:<38} {r['width_bp']:>6.1f} x {r['height_bp']:>6.1f} bp  {len(r['panels']):>2} panels  smallest text {min(body) if body else '-'} pt", flush=True)
            except Exception as e:  # noqa: BLE001
                failed.append(k)
                print(f"FAILED   {k}: {e}", flush=True)
    if failed:
        return 1
    if a.no_single:
        return 0

    rows = {k: load_row(k) for k in figs}
    problems, allowed = run_checks(figs, rows)
    uncropped = merge(list(figs))
    crop(uncropped, SINGLE)
    bad = verify_crop(uncropped, SINGLE, list(figs), rows)
    write_manifest(figs, rows, problems, allowed)
    guide.write_index()
    leaks = K.machine_path_scan(jobs=a.jobs)
    unrecorded = guide.unrecorded([p["file"] for k in figs for p in rows[k]["panels"]])

    n_listed = sum(len(v) for v in allowed.values())
    print(f"\nfigures.pdf: {len(figs)} pages, {SINGLE.stat().st_size / 1e6:.1f} MB")
    if allowed:
        print(f"\nListed in specs/_allow.py ({len(allowed)} figures, {n_listed} checks): they do not pass the house style yet")
        for k, v in allowed.items():
            for name, msg in v.items():
                print(f"  {k:<38} {name:<12} {msg[:150]}")
    if problems:
        print(f"\nFAILED checks ({len(problems)} figures):")
        for k, v in problems.items():
            for name, msg in v.items():
                print(f"  {k:<38} {name:<12} {msg[:170]}")
    for m in bad:
        print("CROP:", m)
    if pending:
        print("\nPending (specified, not yet a float of the paper; composed, not in figures.pdf): " + ", ".join(pending))
    for m in unrecorded:
        print("PANEL SCRIPT:", m)
    for f, hits in leaks:
        print(f"MACHINE PATH: {f}: {', '.join(hits)}")
    print(f"no trace of a working machine in the panels, composed figures, assets and figures.pdf" if not leaks
          else f"{len(leaks)} files carry traces of a working machine: the build fails")
    return 1 if (problems or bad or leaks or broken or (a.strict and (allowed or pending or unrecorded))) else 0


if __name__ == "__main__":
    sys.exit(main())
