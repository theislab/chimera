"""Compare a staged bundle with the current repository and storage.

A bundle is *up to date* when its fingerprint (member list with source sizes and times, plus
the rules that alter content) equals the one recorded when it was staged, every upload file
is present with its recorded size, and the generated texts are current. ``deep=True`` also
recomputes the sha256 of the upload files and of every unaltered source file.
"""
from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .declaration import Declaration
from .packing import PACK_FORMAT, fingerprint, sha256_file
from .selection import Selection

UP_TO_DATE, STALE, NOT_STAGED, DAMAGED = "up to date", "stale", "not staged", "damaged"
MAX_LISTED = 8


@dataclass
class BundleStatus:
    """Result of checking one bundle."""

    name: str
    state: str
    reasons: list[str] = field(default_factory=list)


def read_info(out: Path, name: str) -> dict | None:
    """Staging information of a bundle, or None when it was never staged."""
    path = out / name / "staging_info.json"
    return json.loads(path.read_text()) if path.is_file() else None


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Rows of a tab-separated file written by the packer."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))


def _listed(items: list[str]) -> str:
    more = f" and {len(items) - MAX_LISTED} more" if len(items) > MAX_LISTED else ""
    return ", ".join(items[:MAX_LISTED]) + more


def _member_changes(selection: Selection, staged_members: Path) -> list[str]:
    """Describe how the current member list differs from the staged one."""
    old = {r["path"]: (r["bytes"], r["mtime_ns"]) for r in read_tsv(staged_members)}
    new = {m.arc: (str(m.size), str(m.mtime_ns)) for m in selection.members}
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(arc for arc in set(old) & set(new) if old[arc] != new[arc])
    reasons = []
    if added:
        reasons.append(f"{len(added)} new member(s): {_listed(added)}")
    if removed:
        reasons.append(f"{len(removed)} member(s) no longer selected: {_listed(removed)}")
    if changed:
        reasons.append(f"{len(changed)} member(s) changed on disk: {_listed(changed)}")
    return reasons


def _upload_problems(upload: Path, info: dict) -> list[str]:
    problems = []
    for name, size in info["upload_files"].items():
        path = upload / name
        if not path.is_file():
            problems.append(f"upload file {name} is missing")
        elif path.stat().st_size != size:
            problems.append(f"upload file {name} has {path.stat().st_size} bytes, {size} recorded")
    return problems


def _deep_problems(upload: Path, info: dict, selection: Selection, jobs: int) -> list[str]:
    """Recompute checksums of the upload files and of the unaltered source files."""
    problems = [f"upload file {name} does not match its recorded sha256"
                for name, digest in info["sha256"].items()
                if (upload / name).is_file() and sha256_file(upload / name) != digest]
    recorded = {r["path"]: r for r in read_tsv(upload / "MANIFEST.tsv")}
    plain = [m for m in selection.members
             if m.arc in recorded and not recorded[m.arc]["origin"].startswith("derived:")]
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        digests = pool.map(lambda m: sha256_file(m.source), plain)
        differing = [m.arc for m, digest in zip(plain, digests, strict=True) if digest != recorded[m.arc]["sha256"]]
    if differing:
        problems.append(f"{len(differing)} source file(s) differ from the packed content: {_listed(differing)}")
    return problems


def bundle_status(name: str, declaration: Declaration, selection: Selection, out: Path,
                  expected_texts: dict[str, str] | None = None, deep: bool = False,
                  jobs: int = 1) -> BundleStatus:
    """Return the state of the staged bundle ``name`` against ``selection``.

    ``expected_texts`` maps a path below the bundle folder to the text it should hold now
    (README.md, LICENSES.md, zenodo_metadata.json).
    """
    info = read_info(out, name)
    if info is None:
        return BundleStatus(name, NOT_STAGED)
    if not _same_layout(info):
        return BundleStatus(name, STALE, ["staged by an earlier version of the tool (other layout); it will be rebuilt"])
    folder = out / name
    problems = _upload_problems(folder / "upload", info)
    if problems:
        return BundleStatus(name, DAMAGED, problems)
    reasons: list[str] = []
    if info.get("fingerprint") != fingerprint(declaration, selection):
        reasons = _member_changes(selection, folder / "members.tsv")
        if not reasons:
            reasons.append("the rules that alter content changed (dropped columns, text rules or private rules)")
    for rel, text in (expected_texts or {}).items():
        path = folder / rel
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            reasons.append(f"{rel} is not current (declaration, licenses or citation file changed)")
    if deep and not reasons:
        problems = _deep_problems(folder / "upload", info, selection, jobs)
        if problems:
            return BundleStatus(name, DAMAGED, problems)
    return BundleStatus(name, STALE if reasons else UP_TO_DATE, reasons)


def _same_layout(info: dict) -> bool:
    """Return True when the staging information was written by this version of the packer."""
    return info.get("pack_format") == PACK_FORMAT and "upload_files" in info and "sha256" in info


def tarball_is_current(name: str, declaration: Declaration, selection: Selection, out: Path) -> bool:
    """Return True when the staged tarball of ``name`` can be kept as it is."""
    info = read_info(out, name)
    return (info is not None and _same_layout(info)
            and info.get("fingerprint") == fingerprint(declaration, selection)
            and not _upload_problems(out / name / "upload", info))


def undeclared_folders(declaration: Declaration, out: Path) -> list[str]:
    """Bundle folders in the staging area that the declaration no longer names."""
    if not out.is_dir():
        return []
    return sorted(p.name for p in out.iterdir()
                  if p.is_dir() and (p / "staging_info.json").is_file() and p.name not in declaration.bundles)


def docs_missing_bundles(repo: Path, declaration: Declaration) -> list[str]:
    """Bundle names that a documentation file should mention and does not."""
    missing = []
    for doc in declaration.deposit.docs_naming_every_bundle:
        path = repo / doc
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        missing += [f"{doc} does not name bundle {name}" for name in declaration.bundles if f"`{name}`" not in text]
    return missing
