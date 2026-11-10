#!/usr/bin/env python3
"""Fetch the third-party tools that Chimera is compared with, at their pinned commits.

The tools are the rows of ``tools.tsv`` beside this script (name, upstream URL, pinned
commit, license, environment, how the clone is used, which scripts use it). This script is
the only producer of the clones::

    python3 analysis/external_tools/fetch.py                 # clone and pin every tool a script uses, write README.md
    python3 analysis/external_tools/fetch.py SATURN CAMEX    # only these
    python3 analysis/external_tools/fetch.py --check         # change nothing; exit 1 unless every clone
                                                             # exists, is at its pinned commit and unmodified

A row whose ``used_by`` is ``-`` is a reference clone: no script reads it. It is cloned only
when it is named on the command line, and when it is absent ``--check`` does not count it.

The clones go to ``$CHIMERA_EXTERNAL_TOOLS`` when set, else to ``external_tools`` beside the
repository checkout (see ``location.py``). A clone that already exists is never reset: when
it is at another commit and has no local modification, the pinned commit is checked out;
when tracked files are modified, the script stops and reports it.

The environments of the tools are built afterwards by
``analysis/tasks/T5_1_envs/build_envs.sh`` (SATURN, CAMEX) and
``analysis/tasks/B2_benchmark/tools/build_envs.sh`` (GeneSpectra, CACIMAR, SAMap) from the
files in ``analysis/envs/``. Needs git and the standard library only.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import location  # noqa: E402

README_NAME = "README.md"


def git(folder: Path, *args: str) -> str:
    """Run git in ``folder`` and return its output; raise on failure."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}      # reading the state of a clone must not write into it
    done = subprocess.run(["git", "-C", str(folder), *args], capture_output=True, text=True, env=env)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {folder}: {done.stderr.strip()}")
    return done.stdout.strip()


def state(folder: Path) -> tuple[str | None, bool]:
    """Checked-out commit of a clone (None when it does not exist) and whether tracked files are modified."""
    if not (folder / ".git").exists():
        return None, False
    return git(folder, "rev-parse", "HEAD"), bool(git(folder, "status", "--porcelain", "--untracked-files=no"))


def problem(name: str, row: dict[str, str]) -> str | None:
    """Return what is wrong with the clone of ``name``, or None when it is as pinned."""
    commit, modified = state(location.tool_dir(name))
    if commit is None:
        return "not fetched"
    if commit != row["commit"]:
        return f"at {commit[:12]}, pinned to {row['commit'][:12]}"
    return "tracked files are modified" if modified else None


def fetch(name: str, row: dict[str, str]) -> None:
    """Clone ``name`` if needed and check out its pinned commit."""
    folder = location.tool_dir(name)
    commit, modified = state(folder)
    if commit is None:
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f"{folder} exists and is not a git clone; move it away first")
        folder.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", row["url"], str(folder)], check=True)
        commit = git(folder, "rev-parse", "HEAD")
    if commit == row["commit"]:
        return
    if modified:
        raise RuntimeError(f"{folder} has modified tracked files; commit or discard them before pinning")
    if subprocess.run(["git", "-C", str(folder), "cat-file", "-e", row["commit"] + "^{commit}"],
                      capture_output=True).returncode != 0:
        git(folder, "fetch", "origin")
    git(folder, "checkout", "--quiet", "--detach", row["commit"])


def is_reference(row: dict[str, str]) -> bool:
    """Return True for a clone that no script reads (kept for reading only)."""
    return row["used_by"].strip() == "-"


def readme(rows: dict[str, dict[str, str]]) -> str:
    """Text of the README.md written into the folder of the clones."""
    lines = [
        "# Third-party tools (clones)",
        "",
        "This folder is generated. It holds git clones of the methods that Chimera is compared with, each",
        "at the commit pinned in `analysis/external_tools/tools.tsv` of the Chimera repository.",
        "",
        "- **Producer:** `python3 analysis/external_tools/fetch.py` (run from the Chimera repository). It",
        "  clones, pins and verifies the tools and writes this file. Do not edit the clones or this file.",
        "- **Location:** `$CHIMERA_EXTERNAL_TOOLS` when set, else `external_tools/` beside the Chimera",
        "  checkout; defined once in `analysis/external_tools/location.py`, which every consumer asks.",
        "- **Environments:** built from `analysis/envs/<env>.yml` by `analysis/tasks/T5_1_envs/build_envs.sh`",
        "  (SATURN, CAMEX) and `analysis/tasks/B2_benchmark/tools/build_envs.sh` (GeneSpectra, CACIMAR, SAMap).",
        "- A tool with no environment and no consumer is a reference clone: it is kept for reading and no",
        "  script reads it.",
        *other_entries_note(rows),
        "",
        "| Tool | Upstream | Pinned commit | State | License | Environment | How the clone is used | Used by |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in rows.items():
        used_by = "<br>".join(f"`{p.strip()}`" for p in row["used_by"].split(";")) if row["used_by"] != "-" else "-"
        env = f"`{row['env']}`" if row["env"] != "-" else "-"
        lines.append(f"| `{name}/` | <{row['url']}> | `{row['commit'][:12]}` | {problem(name, row) or 'as pinned'} | "
                     f"{row['license']} | {env} | {row['use']} | {used_by} |")
    return "\n".join(lines) + "\n"


def other_entries_note(rows: dict[str, dict[str, str]]) -> list[str]:
    """One README line about folders beside the clones that this script did not make."""
    folder = location.tools_dir()
    others = sorted(p.name for p in folder.iterdir() if p.is_dir() and p.name not in rows) if folder.is_dir() else []
    if not others:
        return []
    return [f"- Also in this folder, not made by `fetch.py` and not read by any script: {', '.join(f'`{n}/`' for n in others)}."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tools", nargs="*", help="tool names (default: every row of tools.tsv)")
    parser.add_argument("--check", action="store_true", help="verify only; exit 1 when a clone is not as pinned")
    args = parser.parse_args()
    rows = location.manifest()
    unknown = [t for t in args.tools if t not in rows]
    if unknown:
        parser.error(f"unknown tool(s) {unknown}; known: {', '.join(rows)}")
    # Without names: every tool a script uses, plus the reference clones that are already there.
    chosen = {name: rows[name] for name in (args.tools or rows)
              if args.tools or not is_reference(rows[name]) or state(location.tool_dir(name))[0] is not None}
    if not args.check:
        for name, row in chosen.items():
            fetch(name, row)
        folder = location.tools_dir()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / README_NAME).write_text(readme(rows), encoding="utf-8")
    bad = 0
    for name, row in chosen.items():
        issue = problem(name, row)
        print(f"{name:28s} {row['commit'][:12]}  {issue or 'as pinned'}")
        bad += issue is not None
    print(f"folder: {location.tools_dir()}")
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as err:
        print(f"error: {err}", file=sys.stderr)
        sys.exit(2)
