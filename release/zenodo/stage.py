#!/usr/bin/env python
"""Build the Zenodo staging area of the Chimera data deposit.

The deposit is declared in ``bundles.yaml`` beside this script: which files of the working
tree go into which bundle, what is left out, and why. This script turns the declaration and
the current state of the repository and its storage (``data/``, ``results/``,
``training/runs/``, ``notebooks/cache/``) into one folder per bundle, ready to upload as one
Zenodo record each. It uploads nothing.

Run it inside the project environment, from anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from staging import declaration as decl_mod  # noqa: E402
from staging import packing, records, status  # noqa: E402
from staging.declaration import Bundle, Declaration  # noqa: E402
from staging.scrub import BlockedText  # noqa: E402
from staging.selection import Resolver, Selection, SelectionError, coverage_gaps  # noqa: E402

ENV_ROOT = "CHIMERA_ROOT"
ENV_OUT = "CHIMERA_ZENODO_STAGING"
ENV_PRIVATE_RULES = "CHIMERA_ZENODO_PRIVATE_RULES"
DEFAULT_OUT_NAME = "_zenodo_staging"
EXIT_OK, EXIT_STALE, EXIT_ERROR = 0, 1, 2

EPILOG = f"""\
examples:
  python release/zenodo/stage.py --list                    the bundles of the declaration
  python release/zenodo/stage.py --dry-run                 what would be packed, with sizes; writes nothing
  python release/zenodo/stage.py                           build every bundle that is not up to date
  python release/zenodo/stage.py --bundles gene_tables     build one bundle (comma-separated list)
  python release/zenodo/stage.py --force                   rebuild even what is up to date
  python release/zenodo/stage.py --check                   compare the staging area with repository and storage
  python release/zenodo/stage.py --check --deep            ... and recompute every checksum

where things are:
  repository   the checkout this script is in, or ${ENV_ROOT} when set
  output       --out, else ${ENV_OUT}, else ../{DEFAULT_OUT_NAME} beside the checkout
  declaration  release/zenodo/bundles.yaml (bundles, exclusions, rules, release tag and DOIs)
  licenses     release/zenodo/LICENSES.md (copied into every record)

A bundle is rebuilt only when its fingerprint changed: the list of its members with the size
and modification time of each source file, and the rules that alter content. A bundle is
written to a temporary folder and moved into place when complete, so an interrupted run
never leaves a half-written bundle that looks finished.

exit code: 0 = done, or --check found everything up to date; 1 = --check found something stale,
missing, damaged or not assigned to a bundle; 2 = error (a bundle that cannot be built is
reported and skipped, the others are still built).
"""


def repo_root() -> Path:
    """The repository: ``$CHIMERA_ROOT`` when set, else the checkout that holds this script."""
    return Path(os.environ[ENV_ROOT]).resolve() if os.environ.get(ENV_ROOT) else HERE.parents[1]


def repo_state(repo: Path) -> dict[str, object]:
    """Commit of the repository and whether tracked files are modified."""
    def git(*args: str) -> str | None:
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return {"repo_commit": commit or "unknown", "repo_dirty": bool(dirty) if dirty is not None else None}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help=f"staging folder (default: ${ENV_OUT}, else ../{DEFAULT_OUT_NAME})")
    parser.add_argument("--bundles", help="comma-separated bundle names (default: all)")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1),
                        help="threads for compression and checksums (default: %(default)s)")
    parser.add_argument("--force", action="store_true", help="rebuild the selected bundles even when up to date")
    parser.add_argument("--private-rules", type=Path,
                        help=f"YAML with further text rules: replace, review, blocked (default: ${ENV_PRIVATE_RULES})")
    parser.add_argument("--declaration", type=Path, default=HERE / "bundles.yaml", help=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="list the bundles of the declaration and exit")
    mode.add_argument("--dry-run", action="store_true",
                      help="show what would be packed, with sizes; write nothing (-v lists every member)")
    mode.add_argument("--check", action="store_true",
                      help="verify the staging area against repository and storage; exit 1 when anything is stale")
    mode.add_argument("--bundle-table", action="store_true",
                      help="print the bundle table for REPRODUCE.md (Markdown) and exit")
    mode.add_argument("--prune", action="store_true",
                      help="remove staged bundle folders that the declaration no longer names, then exit")
    parser.add_argument("--deep", action="store_true",
                        help="with --check: recompute the sha256 of upload files and source files")
    parser.add_argument("-v", "--verbose", action="store_true", help="list every member (with --dry-run)")
    return parser.parse_args(argv)


class Stager:
    """One run of the tool: a declaration, a repository and a staging folder."""

    def __init__(self, repo: Path, out: Path, declaration: Declaration, jobs: int) -> None:
        self.repo, self.out, self.decl, self.jobs = repo, out, declaration, jobs
        self.resolver = Resolver(repo, declaration)
        self.licenses = (repo / declaration.deposit.licenses_file).read_text(encoding="utf-8")
        self.creators, self.keywords = records.creators_and_keywords(repo, declaration.deposit.citation_file)

    # ------------------------------------------------------------------ texts
    def upload_texts(self, bundle: Bundle, manifest: list[tuple[str, int]], parts: list[str]) -> dict[str, str]:
        """The text files of a record: ``{file name: text}``."""
        return {"README.md": records.record_readme(bundle, self.decl, manifest, parts),
                "LICENSES.md": self.licenses}

    def metadata(self, bundle: Bundle) -> dict:
        return records.zenodo_metadata(bundle, self.decl, self.creators, self.keywords)

    def expected_texts(self, bundle: Bundle) -> dict[str, str]:
        """Generated texts of a staged bundle as they should be now, keyed by path in the bundle folder."""
        info = status.read_info(self.out, bundle.name) or {}
        rows = status.read_tsv(self.out / bundle.name / "upload" / "MANIFEST.tsv")
        manifest = [(r["path"], int(r["bytes"])) for r in rows]
        texts = {f"upload/{name}": text
                 for name, text in self.upload_texts(bundle, manifest, info.get("tarball_parts", [])).items()}
        texts["zenodo_metadata.json"] = json.dumps(self.metadata(bundle), indent=2) + "\n"
        return texts

    # ------------------------------------------------------------------ actions
    def build(self, bundle: Bundle, selection: Selection) -> dict:
        """Pack the bundle from scratch."""
        def texts(manifest: list[tuple[str, int, str, str]]) -> dict[str, str]:
            return self.upload_texts(bundle, [(row[0], row[1]) for row in manifest],
                                     [f"chimera_{bundle.name}.tar.gz"])

        return packing.build_bundle(bundle, self.decl, selection, self.out, texts, self.metadata(bundle),
                                    self.jobs, repo_state(self.repo))

    def refresh_texts(self, bundle: Bundle) -> bool:
        """Rewrite the generated texts of an unchanged bundle; return True when one was rewritten."""
        folder = self.out / bundle.name
        stale = {rel: text for rel, text in self.expected_texts(bundle).items()
                 if not (folder / rel).is_file() or (folder / rel).read_text(encoding="utf-8") != text}
        if not stale:
            return False
        for rel, text in stale.items():
            (folder / rel).write_text(text, encoding="utf-8")
        info = status.read_info(self.out, bundle.name)
        keep = {name: digest for name, digest in info["sha256"].items() if f"upload/{name}" not in stale}
        info["sha256"] = packing.write_checksums(folder / "upload", known=keep)
        info["upload_files"] = {p.name: p.stat().st_size for p in sorted((folder / "upload").iterdir())}
        info["record_bytes"] = sum(info["upload_files"].values())
        (folder / "staging_info.json").write_text(json.dumps(info, indent=2) + "\n")
        return True

    def write_top_level(self) -> None:
        """Write README.md, LICENSES.md and size_table.tsv of the staging area."""
        infos = {name: info for name in self.decl.bundles
                 if (info := status.read_info(self.out, name)) and "upload_files" in info}
        (self.out / "README.md").write_text(records.staging_readme(self.decl, infos), encoding="utf-8")
        (self.out / "LICENSES.md").write_text(self.licenses, encoding="utf-8")
        (self.out / "size_table.tsv").write_text(records.size_table(self.decl, infos), encoding="utf-8")


def resolve(stager: Stager, name: str) -> Selection | None:
    """Resolve one bundle; print the reason and return None when the storage cannot satisfy it."""
    try:
        return stager.resolver.resolve(stager.decl.bundles[name])
    except SelectionError as err:
        print(f"{name}: cannot be resolved: {err}")
        return None


def cmd_dry_run(stager: Stager, names: list[str], verbose: bool) -> int:
    """Print what would be packed; write nothing."""
    total_members = total_bytes = failed = 0
    for name in names:
        selection = resolve(stager, name)
        if selection is None:
            failed += 1
            continue
        current = status.tarball_is_current(name, stager.decl, selection, stager.out)
        derived = f" (+ {selection.n_derived} derived tables)" if selection.n_derived else ""
        print(f"{name}: {len(selection.members)} members, {records.gigabytes(selection.n_bytes)} before compression"
              f"{derived}, {len(selection.skipped)} exclusion rows; {'up to date' if current else 'would be built'}")
        if verbose:
            for m in selection.members:
                size = f"{m.size:>13,d}" if not m.derive else f"{'derived':>13s}"
                print(f"  {size}  {m.arc}" + ("" if m.origin == m.arc else f"    <- {m.origin}"))
        total_members += len(selection.members)
        total_bytes += selection.n_bytes
    print(f"total: {total_members} members, {records.gigabytes(total_bytes)} before compression")
    return EXIT_ERROR if failed else EXIT_OK


def cmd_check(stager: Stager, names: list[str], deep: bool) -> int:
    """Report what is stale; exit code 1 when anything is."""
    resolved = {name: resolve(stager, name) for name in names}
    selections = {name: selection for name, selection in resolved.items() if selection is not None}
    problems = len(resolved) - len(selections)
    for name in selections:
        bundle = stager.decl.bundles[name]
        staged = (stager.out / name / "upload" / "MANIFEST.tsv").is_file()
        result = status.bundle_status(name, stager.decl, selections[name], stager.out,
                                      stager.expected_texts(bundle) if staged else None, deep, stager.jobs)
        print(f"{name}: {result.state}")
        for reason in result.reasons:
            print(f"    {reason}")
        problems += result.state != status.UP_TO_DATE
    notes = [f"staged folder {n}/ is not a bundle of the declaration (remove it with --prune)"
             for n in status.undeclared_folders(stager.decl, stager.out)]
    notes += status.docs_missing_bundles(stager.repo, stager.decl)
    notes += status.citation_mismatches(stager.repo, stager.decl)
    if set(selections) == set(stager.decl.bundles):
        notes += [f"{gap} is in no bundle and has no stated reason for being left out"
                  for gap in coverage_gaps(stager.repo, stager.decl, selections)]
    else:
        print("storage coverage not checked (it needs every bundle resolved; run --check without --bundles)")
    for note in notes:
        print(f"note: {note}")
    if not stager.decl.text_rules.private_rules_used:
        print("note: no private rules file was given; only the public text rules were considered")
    problems += len(notes)
    print("staging area is up to date" if not problems else f"{problems} thing(s) need attention")
    return EXIT_OK if not problems else EXIT_STALE


def cmd_stage(stager: Stager, names: list[str], force: bool) -> int:
    """Build every selected bundle that is not up to date, then the top-level files."""
    os.umask(0o077)                      # the staging area is private until its files are uploaded
    stager.out.mkdir(parents=True, exist_ok=True)
    os.chmod(stager.out, 0o700)
    lock = stager.out / ".lock"
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        print(f"{lock} exists: another run is using this staging folder (delete the file if that is not so)",
              file=sys.stderr)
        return EXIT_ERROR
    failed = []
    try:
        packing.remove_leftovers(stager.out)
        for name in names:
            try:
                stage_bundle(stager, name, force)
            except (SelectionError, packing.PackError, BlockedText, OSError) as err:
                # The staged copy of this bundle, if any, is left as it was; the other bundles go on.
                # (OSError: a source file changed or vanished while it was packed.)
                print(f"{name}: FAILED: {err}", flush=True)
                failed.append(name)
        stager.write_top_level()
    finally:
        lock.unlink(missing_ok=True)
    print(f"staging area: {stager.out}")
    if failed:
        print(f"not built: {', '.join(failed)}", file=sys.stderr)
    return EXIT_ERROR if failed else EXIT_OK


def stage_bundle(stager: Stager, name: str, force: bool) -> None:
    """Build one bundle, or only refresh its generated texts when its tarball is current."""
    bundle = stager.decl.bundles[name]
    selection = stager.resolver.resolve(bundle)
    if not force and status.tarball_is_current(name, stager.decl, selection, stager.out):
        refreshed = stager.refresh_texts(bundle)
        print(f"{name}: up to date" + (", texts refreshed" if refreshed else ""), flush=True)
        return
    print(f"{name}: packing {len(selection.members)} members, "
          f"{records.gigabytes(selection.n_bytes)} before compression ...", flush=True)
    info = stager.build(bundle, selection)
    print(f"{name}: done, {records.gigabytes(info['record_bytes'])} in the record, "
          f"{info['n_scrubbed']} altered, {info['n_review_hits']} review hits", flush=True)


def cmd_prune(stager: Stager) -> int:
    """Remove staged bundle folders that the declaration no longer names."""
    for name in status.undeclared_folders(stager.decl, stager.out):
        shutil.rmtree(stager.out / name)
        print(f"removed {name}/")
    if stager.out.is_dir():
        stager.write_top_level()
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = repo_root()
    private = args.private_rules or (Path(os.environ[ENV_PRIVATE_RULES]) if os.environ.get(ENV_PRIVATE_RULES) else None)
    try:
        declaration = decl_mod.load(args.declaration, private)
        names = list(declaration.bundles) if not args.bundles else [n.strip() for n in args.bundles.split(",")]
        unknown = [n for n in names if n not in declaration.bundles]
        if unknown:
            raise decl_mod.DeclarationError(f"unknown bundle(s) {unknown}; declared: {', '.join(declaration.bundles)}")
        if args.list:
            for name, bundle in declaration.bundles.items():
                print(f"{name:32s} {bundle.title}")
            return EXIT_OK
        if args.bundle_table:
            print(records.bundle_table(declaration), end="")
            return EXIT_OK
        out = (args.out or Path(os.environ.get(ENV_OUT) or repo.parent / DEFAULT_OUT_NAME)).resolve()
        if out == repo or repo in out.parents:
            raise decl_mod.DeclarationError(f"the staging folder {out} must be outside the repository {repo}")
        stager = Stager(repo, out, declaration, args.jobs)
        if args.dry_run:
            return cmd_dry_run(stager, names, args.verbose)
        if args.check:
            return cmd_check(stager, names, args.deep)
        if args.prune:
            return cmd_prune(stager)
        return cmd_stage(stager, names, args.force)
    except (decl_mod.DeclarationError, SelectionError, packing.PackError, BlockedText, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
