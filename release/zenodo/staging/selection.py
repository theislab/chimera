"""Turn the member selectors of a bundle into the list of files to pack.

Nothing here reads file contents or writes anything: a selection is computed from file names,
sizes and modification times only, so it is cheap enough for ``--dry-run`` and ``--check``.
"""
from __future__ import annotations

import csv
import fnmatch
import json
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .declaration import Bundle, Declaration, Exclusion, RunView, SkipRule

RUNS_FOLDER = "training/runs"
SELECTOR_KEYS = {
    "from": {"from", "take", "skip", "into", "others"},
    "file": {"file", "as"},
    "runs": {"runs", "view"},
    "tasks_in": {"tasks_in", "only", "except", "complete"},
    "obs_table_of": {"obs_table_of", "as"},
}


class SelectionError(RuntimeError):
    """A selector cannot be satisfied by the current repository and storage."""


@dataclass(frozen=True)
class Member:
    """One file of a bundle."""

    arc: str                 # path inside the tarball = path below the repository root after unpacking
    source: Path             # file the content is read from
    origin: str              # MANIFEST text: repository-relative source path, or "derived: ..."
    size: int                # bytes of the source file
    mtime_ns: int            # modification time of the source file
    derive: str = ""         # "" = packed as it is; "obs_table" = cell annotation table of an AnnData file


@dataclass
class Selection:
    """The resolved content of one bundle."""

    members: list[Member] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (path, reason)

    @property
    def n_bytes(self) -> int:
        """Bytes of the members packed as they are (before any column is dropped, before compression)."""
        return sum(m.size for m in self.members if not m.derive)

    @property
    def n_derived(self) -> int:
        """Number of members that are derived from their source (their size is known after packing)."""
        return sum(1 for m in self.members if m.derive)


def brace_expand(pattern: str) -> list[str]:
    """Expand ``a/{b,c}/d`` into ``['a/b/d', 'a/c/d']`` (several groups allowed, no nesting)."""
    match = re.search(r"\{([^{}]*)\}", pattern)
    if not match:
        return [pattern]
    head, tail = pattern[:match.start()], pattern[match.end():]
    return [p for alt in match.group(1).split(",") for p in brace_expand(head + alt + tail)]


def pattern_regex(pattern: str) -> re.Pattern[str]:
    """Compile a path pattern: ``*`` stays inside one folder level, ``**`` crosses levels."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append(r"(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(r".*")
            i += 2
        elif pattern[i] == "*":
            out.append(r"[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append(r"[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def has_wildcard(pattern: str) -> bool:
    """Return True when the pattern can match more than one name."""
    return any(c in pattern for c in "*?")


def _string_leaves(obj: Any) -> Iterator[str]:
    """Yield every string value of a nested JSON object."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _string_leaves(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _string_leaves(value)


class Resolver:
    """Resolve bundles of one declaration against one repository checkout."""

    def __init__(self, repo: Path, declaration: Declaration) -> None:
        self.repo = repo
        self.decl = declaration
        self._run_names: list[str] | None = None

    # ------------------------------------------------------------------ helpers
    def rel(self, path: Path) -> str:
        """Repository-relative path with forward slashes."""
        return path.relative_to(self.repo).as_posix()

    def walk(self, folder: Path) -> Iterator[Path]:
        """Yield the files below ``folder`` in a stable order, without the never-packed ones.

        Links to folders are not followed; a link to a file is packed as the file it points to.
        """
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames.sort()
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if not self.decl.never_packed.matches(name) and path.is_file():
                    yield path

    def member(self, source: Path, arc: str | None = None, origin: str | None = None,
               derive: str = "") -> Member:
        """Build the member for a source file."""
        stat = source.stat()
        rel = self.rel(source)
        return Member(arc=arc or rel, source=source, origin=origin or rel, size=stat.st_size,
                      mtime_ns=stat.st_mtime_ns, derive=derive)

    def _level(self, folders: list[Path], part: str, want_folders: bool) -> list[Path]:
        """Match one path segment in each of ``folders``; return folders or files."""
        out: list[Path] = []
        for folder in folders:
            if not has_wildcard(part):
                path = folder / part
                if path.is_dir() if want_folders else path.is_file():
                    out.append(path)
                continue
            for entry in sorted(os.scandir(folder), key=lambda e: e.name):
                if not fnmatch.fnmatchcase(entry.name, part):
                    continue
                is_wanted = (entry.is_dir(follow_symlinks=False) if want_folders
                             else entry.is_file() and not self.decl.never_packed.matches(entry.name))
                if is_wanted:
                    out.append(Path(entry.path))
        return out

    def expand(self, base: Path, pattern: str) -> list[Path]:
        """Return the files below ``base`` that match ``pattern`` (relative to ``base``).

        Folders are listed level by level, so only the part of the tree that the pattern can
        reach is read; from a ``**`` segment on, the whole subtree is walked.
        """
        parts = pattern.split("/")
        if any("**" in part and part != "**" for part in parts):
            raise SelectionError(f"pattern {pattern!r}: '**' must be a whole path segment")
        deep = parts.index("**") if "**" in parts else None
        folders = [base]
        for part in parts[:deep] if deep is not None else parts[:-1]:
            folders = self._level(folders, part, want_folders=True)
        if deep is None:
            return self._level(folders, parts[-1], want_folders=False)
        regex = pattern_regex("/".join(parts[deep:]))
        return [path for folder in folders for path in self.walk(folder)
                if regex.match(path.relative_to(folder).as_posix())]

    def run_names(self) -> list[str]:
        """Names of the entries of ``training/runs`` (listed once)."""
        if self._run_names is None:
            root = self.repo / RUNS_FOLDER
            self._run_names = sorted(e.name for e in os.scandir(root)) if root.is_dir() else []
        return self._run_names

    # ------------------------------------------------------------------ selectors
    def _from(self, sel: dict[str, Any], out: Selection) -> None:
        base = self.repo / sel["from"]
        if not base.is_dir():
            raise SelectionError(f"folder {sel['from']} does not exist")
        skip = [pattern_regex(p) for p in sel.get("skip") or []]
        chosen: list[Path] = []
        for pattern in sel.get("take") or ["**"]:
            found = self.expand(base, pattern)
            if not found:
                raise SelectionError(f"{sel['from']}: {pattern!r} matches no file")
            chosen += [p for p in found
                       if not any(rx.match(p.relative_to(base).as_posix()) for rx in skip)]
        target = (sel.get("into") or sel["from"]).rstrip("/")
        for path in chosen:
            out.members.append(self.member(path, arc=f"{target}/{path.relative_to(base).as_posix()}"))
        if sel.get("others"):
            taken = {p.relative_to(base).parts[0] for p in chosen}
            for child in sorted(base.iterdir()):
                if child.name not in taken and not self.decl.never_packed.matches(child.name):
                    out.skipped.append((self.rel(child), sel["others"]))

    def _file(self, sel: dict[str, Any], out: Selection) -> None:
        path = self.repo / sel["file"]
        if not path.is_file():
            raise SelectionError(f"file {sel['file']} does not exist")
        out.members.append(self.member(path, arc=sel.get("as")))

    def _obs_table(self, sel: dict[str, Any], out: Selection) -> None:
        path = self.repo / sel["obs_table_of"]
        if not path.is_file():
            raise SelectionError(f"file {sel['obs_table_of']} does not exist")
        arc = sel.get("as") or re.sub(r"\.h5ad$", ".obs.csv.gz", sel["obs_table_of"])
        origin = f"derived: obs of {self.rel(path)} (expression matrix not deposited)"
        out.members.append(self.member(path, arc=arc, origin=origin, derive="obs_table"))

    def _selected_runs(self, spec: dict[str, Any]) -> list[str]:
        unknown = set(spec) - {"recipes", "names", "listed_in", "except_listed_in"}
        if unknown:
            raise SelectionError(f"runs selector: unknown key(s) {sorted(unknown)}")
        names: set[str] = set()
        if spec.get("recipes"):
            with open(self.repo / spec["recipes"], newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    names |= {row["run_name"].replace("{seed}", s.strip())
                              for s in row["seeds"].split(",")}
        for pattern in spec.get("names") or []:
            names |= set(fnmatch.filter(self.run_names(), pattern))
        if spec.get("listed_in"):
            names |= set(_string_leaves(json.loads((self.repo / spec["listed_in"]).read_text())))
        if spec.get("except_listed_in"):
            names -= set(_string_leaves(json.loads((self.repo / spec["except_listed_in"]).read_text())))
        return sorted(names)

    def _run_files(self, folder: Path, view: RunView) -> Iterator[Path]:
        if view.everything:
            yield from self.walk(folder)
            return
        for path in sorted(folder.iterdir()):
            if not path.is_file() or self.decl.never_packed.matches(path.name):
                continue
            small = path.suffix in view.small_suffixes and path.stat().st_size < view.small_below_bytes
            if path.name in view.top_level_names or small:
                yield path
        for subtree in view.subtrees:
            if (folder / subtree).is_dir():
                yield from self.walk(folder / subtree)
        for name in view.files:
            if (folder / name).is_file():
                yield folder / name

    def _runs(self, sel: dict[str, Any], out: Selection) -> None:
        view = self.decl.run_views.get(sel.get("view", ""))
        if view is None:
            raise SelectionError(f"runs selector: unknown view {sel.get('view')!r}")
        for run in self._selected_runs(sel["runs"]):
            folder = self.repo / RUNS_FOLDER / run
            if not folder.is_dir():
                if folder.is_file():      # a stray file whose name matches a run pattern
                    continue
                raise SelectionError(f"run {run} is named by the declaration but {RUNS_FOLDER}/{run} does not exist")
            if not all((folder / req).is_file() for req in view.requires):
                message = f"run not finished (no {' or '.join(view.requires)})"
                if view.if_unfinished == "error":
                    raise SelectionError(f"{RUNS_FOLDER}/{run}: {message}")
                out.skipped.append((self.rel(folder), message))
                continue
            out.members += [self.member(p) for p in self._run_files(folder, view)]

    @staticmethod
    def _matches(rule: SkipRule, rel_parts: tuple[str, ...], name: str) -> bool:
        if rule.path_part_contains and any(rule.path_part_contains in part.lower() for part in rel_parts):
            return True
        if rule.name_regex and re.search(rule.name_regex, name):
            return True
        return bool(rule.path_regex and re.search(rule.path_regex, "/" + "/".join(rel_parts)))

    def _task(self, top: Path, base: Path, compact: bool, out: Selection) -> None:
        """Select the files of one task folder (or one top-level file)."""
        rules = self.decl.task_folders
        counted = [0] * len(rules.counted_skips)
        n_compact = 0
        for path in ([top] if top.is_file() else self.walk(top)):
            parts = path.relative_to(base).parts
            listed = next((r for r in rules.listed_skips if self._matches(r, parts, path.name)), None)
            if listed:
                out.skipped.append((self.rel(path), listed.reason))
                continue
            hit = next((i for i, r in enumerate(rules.counted_skips)
                        if self._matches(r, parts, path.name)), None)
            if hit is not None:
                counted[hit] += 1
            elif compact and (path.suffix.lower() in rules.compact_drop_suffixes
                              or set(parts[1:-1]) & rules.compact_drop_folders):
                n_compact += 1
            else:
                out.members.append(self.member(path))
        for rule, n in zip(rules.counted_skips, counted, strict=True):
            if n:
                out.skipped.append((f"{self.rel(top)}/{rule.label}", f"{n} {rule.reason}"))
        if n_compact:
            out.skipped.append((self.rel(top), f"{n_compact} {rules.compact_reason}"))

    def _tasks(self, sel: dict[str, Any], out: Selection) -> None:
        base = self.repo / sel["tasks_in"]
        if not base.is_dir():
            raise SelectionError(f"folder {sel['tasks_in']} does not exist")
        if sel.get("only") and sel.get("except"):
            raise SelectionError("tasks_in selector: give 'only' or 'except', not both")
        rules = self.decl.task_folders
        only, excluded = sel.get("only"), set(sel.get("except") or [])
        complete = set(sel.get("complete") or [])
        missing = [t for t in only or [] if not (base / t).exists()]
        if missing:
            raise SelectionError(f"{sel['tasks_in']}: task folder(s) {missing} do not exist")
        for top in sorted(base.iterdir()):
            name = top.name
            if self.decl.never_packed.matches(name):
                continue
            if name in rules.not_results_names or any(c in name.lower() for c in rules.not_results_name_contains):
                if only is None:
                    out.skipped.append((self.rel(top), rules.not_results_reason))
                continue
            if (name in only) if only is not None else (name not in excluded):
                self._task(top, base, compact=name not in complete, out=out)

    # ------------------------------------------------------------------ public
    def resolve(self, bundle: Bundle) -> Selection:
        """Return the members of ``bundle`` (sorted by path) and the list of what was left out."""
        out = Selection()
        handlers = {"from": self._from, "file": self._file, "runs": self._runs,
                    "tasks_in": self._tasks, "obs_table_of": self._obs_table}
        for selector in bundle.members:
            kind = next((k for k in handlers if k in selector), None)
            if kind is None or set(selector) - SELECTOR_KEYS[kind]:
                raise SelectionError(f"bundle {bundle.name}: cannot read selector {selector}")
            try:
                handlers[kind](selector, out)
            except SelectionError as err:
                raise SelectionError(f"bundle {bundle.name}: {err}") from None
            except OSError as err:   # a file vanished between listing and reading its size
                raise SelectionError(f"bundle {bundle.name}: the storage changed while it was read "
                                     f"({err.filename or err}); run again when nothing writes to it") from None
        out.skipped += [(e.path, e.reason) for e in bundle.not_included]
        out.members = _unique_sorted(out.members, bundle.name)
        return out


def _unique_sorted(members: list[Member], bundle: str) -> list[Member]:
    """Drop exact repeats, refuse two different files under one member path, sort by path."""
    by_arc: dict[str, Member] = {}
    for m in members:
        seen = by_arc.setdefault(m.arc, m)
        if seen.source != m.source:
            raise SelectionError(f"bundle {bundle}: {m.arc} is selected from two different files")
    return [by_arc[arc] for arc in sorted(by_arc)]


def _exclusion_regexes(exclusions: Iterable[Exclusion | tuple[str, str]]) -> list[re.Pattern[str]]:
    paths = (e.path if isinstance(e, Exclusion) else e[0] for e in exclusions)
    return [pattern_regex(p) for path in paths for p in brace_expand(path)]


def coverage_gaps(repo: Path, declaration: Declaration, selections: dict[str, Selection]) -> list[str]:
    """Return the entries of the coverage roots that no bundle packs and no exclusion names."""
    packed: set[str] = set()
    for selection in selections.values():
        for m in selection.members:
            rel = m.source.relative_to(repo)
            packed.add(rel.as_posix())
            packed.update(parent.as_posix() for parent in rel.parents)
    named = _exclusion_regexes(declaration.not_deposited)
    for selection in selections.values():
        named += _exclusion_regexes(selection.skipped)
    gaps = []
    for root in declaration.coverage_roots:
        folder = repo / root
        if not folder.is_dir():
            continue
        for entry in sorted(os.scandir(folder), key=lambda e: e.name):
            rel = f"{root}/{entry.name}"
            if declaration.never_packed.matches(entry.name) or rel in declaration.coverage_roots:
                continue
            if rel not in packed and not any(rx.match(rel) for rx in named):
                gaps.append(rel)
    return gaps
