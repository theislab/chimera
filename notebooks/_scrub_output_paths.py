"""Remove machine-specific path prefixes from the stored outputs of the workflow notebooks.

A notebook that was executed on some machine prints absolute paths of that machine
(``figures will be written to: /home/<user>/.../chimera/notebooks/figures/...``). This tool
rewrites them so the tracked notebooks read the same on every machine:

- a path below the checkout becomes repository-relative (``notebooks/figures/...``); the
  checkout itself becomes ``.``; a path below it that the display cut short (``/home/<user>/pro...``)
  becomes ``./...``;
- a path below the home folder becomes ``~/...``;
- any other prefix given with ``--map OLD=NEW`` becomes ``NEW``.

Only the text of code-cell outputs is touched (stream text, ``text/plain``, ``text/html``,
tracebacks). Cell sources, cell order, execution counts, metadata and images stay byte for
byte; the file keeps its JSON layout, so ``git diff`` shows the changed output lines only.
Nothing is executed.

Usage, from the repository root:

    python notebooks/_scrub_output_paths.py                 # rewrite notebooks/*.ipynb in place
    python notebooks/_scrub_output_paths.py --check         # change nothing; exit 1 if a path is left
    python notebooks/_scrub_output_paths.py --root /old/place/of/chimera --home /Users/name \
        --map /opt/mamba='$MAMBA_ROOT_PREFIX' notebooks/03_alpha_master.ipynb

``--root`` and ``--home`` add locations of other machines on which a notebook was executed;
the current checkout and the current home folder are always included.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TEXT_MIMES = ("text/plain", "text/html", "text/markdown", "text/latex", "application/javascript")
# Absolute paths of a user account on Linux or macOS; what ``--check`` looks for besides the given roots.
USER_PATH = re.compile(r"(?<![\w.~-])/(?:Users|home)/[^/\s\"'<>]+/")
# Shortest piece of a root that still identifies a path the display cut short ("/home/name/proj...").
MIN_CUT = 12


def _variants(path: str) -> list[str]:
    """The path as given and with symbolic links resolved, longest first, without trailing slash."""
    out = {path.rstrip("/")}
    with contextlib.suppress(OSError):
        out.add(str(Path(path).resolve()).rstrip("/"))
    return sorted((p for p in out if p), key=len, reverse=True)


class Scrubber:
    def __init__(self, roots: list[str], homes: list[str], maps: list[tuple[str, str]]):
        self.roots = sorted({v for r in roots for v in _variants(r)}, key=len, reverse=True)
        self.homes = sorted({v for h in homes for v in _variants(h)}, key=len, reverse=True)
        self.maps = sorted(((old.rstrip("/"), new) for old, new in maps), key=lambda m: len(m[0]), reverse=True)
        cut = []
        for root in self.roots:  # "/home/name/proj..." : a proper prefix of the root followed by an ellipsis
            cut += [re.escape(root[:k]) for k in range(len(root), MIN_CUT - 1, -1)]
        self._cut = re.compile("(?:" + "|".join(cut) + r")\.\.\.") if cut else None

    def text(self, s: str, pad: bool) -> str:
        for old, new in self.maps:
            s = s.replace(old, new)
        if self._cut is not None:
            s = self._cut.sub(lambda m: "./...".rjust(len(m.group(0))) if pad else "./...", s)
        for root in self.roots:
            s = s.replace(root + "/", "")
            s = re.sub(re.escape(root) + r"(?![\w.-])", ".", s)
        for home in self.homes:
            s = re.sub(re.escape(home) + r"(?![\w.-])", "~", s)
        return s

    def left(self, s: str) -> list[str]:
        hits = [r for r in self.roots + self.homes + [m[0] for m in self.maps] if r in s]
        return hits + USER_PATH.findall(s)

    def _field(self, value, pad: bool, fn):
        if isinstance(value, list):
            return [fn(v, pad) if isinstance(v, str) else v for v in value]
        return fn(value, pad) if isinstance(value, str) else value

    def outputs(self, cell: dict, fn) -> None:
        """Apply ``fn(text, pad)`` to every text field of the outputs of one code cell, in place."""
        for out in cell.get("outputs", []):
            if "text" in out:
                out["text"] = self._field(out["text"], True, fn)
            if "traceback" in out:
                out["traceback"] = self._field(out["traceback"], False, fn)
            for key in ("evalue",):
                if key in out:
                    out[key] = self._field(out[key], False, fn)
            for mime, value in out.get("data", {}).items():
                if mime in TEXT_MIMES:
                    out["data"][mime] = self._field(value, mime == "text/plain", fn)


def fingerprint(nb: dict) -> str:
    """Hash of everything this tool must not change: cell types, sources, execution counts, metadata."""
    keep = [[c["cell_type"], c["source"], c.get("execution_count"), c.get("metadata"),
             [[o.get("output_type"), o.get("execution_count"), sorted(o.get("data", {}))] for o in c.get("outputs", [])]]
            for c in nb["cells"]]
    blob = json.dumps([keep, nb.get("metadata"), nb.get("nbformat"), nb.get("nbformat_minor")], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def process(path: Path, scrubber: Scrubber, check: bool) -> tuple[int, int]:
    """Return (number of changed text fields, number of paths left) for one notebook."""
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)
    newline = "\n" if raw.endswith("\n") else ""
    if json.dumps(nb, indent=1, ensure_ascii=False) + newline != raw:
        raise SystemExit(f"{path}: not in the usual notebook JSON layout; refusing to rewrite it")
    before = fingerprint(nb)
    changed = left = 0

    def scrub(s: str, pad: bool) -> str:
        nonlocal changed
        new = scrubber.text(s, pad)
        changed += new != s
        return new

    def count(s: str, pad: bool) -> str:
        nonlocal left
        left += len(scrubber.left(s))
        return s

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            if not check:
                scrubber.outputs(cell, scrub)
            scrubber.outputs(cell, count)
    if fingerprint(nb) != before:
        raise SystemExit(f"{path}: sources or metadata would change; nothing written")
    if changed and not check:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + newline, encoding="utf-8")
    return changed, left


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("notebooks", nargs="*", type=Path, help="notebooks to process (default: notebooks/*.ipynb)")
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 if an output holds a machine path")
    ap.add_argument("--root", action="append", default=[], help="location of the checkout on another machine")
    ap.add_argument("--home", action="append", default=[], help="home folder on another machine")
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW", help="replace prefix OLD by NEW")
    a = ap.parse_args()
    maps = [tuple(m.split("=", 1)) for m in a.map]
    here = [str(REPO_ROOT), str(Path(__file__).absolute().parent.parent), os.environ.get("CHIMERA_ROOT", "")]
    scrubber = Scrubber([r for r in here if r] + a.root, [str(Path.home()), *a.home], maps)
    total_left = 0
    for path in a.notebooks or sorted(HERE.glob("*.ipynb")):
        changed, left = process(path, scrubber, a.check)
        total_left += left
        print(f"{path.name}: {left} machine paths in outputs" if a.check
              else f"{path.name}: {changed} output fields rewritten, {left} machine paths left")
    return 1 if total_left else 0


if __name__ == "__main__":
    sys.exit(main())
