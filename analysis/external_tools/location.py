"""Where the clones of the third-party tools live, and which commit each is pinned to.

This module is the one definition of both. Every script that needs a clone asks it instead
of building the path itself:

* the folder is ``$CHIMERA_EXTERNAL_TOOLS`` when set, else ``external_tools`` beside the
  repository checkout (outside the repository, so that git never sees the clones);
* the tools, their upstream URL and their pinned commit are the rows of ``tools.tsv`` beside
  this file.

It uses the standard library only, because it is loaded inside the tools' own environments.
From Python, in a script that must not put the repository on ``sys.path``::

    import runpy
    tools = runpy.run_path(str(REPO / "analysis" / "external_tools" / "location.py"))
    saturn = tools["tool_dir"]("SATURN")

From a shell script::

    TOOLS="$(python3 analysis/external_tools/location.py)"            # the folder
    SAMAP="$(python3 analysis/external_tools/location.py SAMap)"      # one clone
    python3 analysis/external_tools/location.py --commit SAMap        # its pinned commit
    python3 analysis/external_tools/location.py --url SAMap           # its upstream URL

The clones are created by ``fetch.py`` in this folder.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MANIFEST = HERE / "tools.tsv"
ENV_VAR = "CHIMERA_EXTERNAL_TOOLS"
DEFAULT_FOLDER_NAME = "external_tools"


def tools_dir() -> Path:
    """Folder that holds the clones."""
    override = os.environ.get(ENV_VAR)
    return Path(override).expanduser().resolve() if override else REPO.parent / DEFAULT_FOLDER_NAME


def manifest() -> dict[str, dict[str, str]]:
    """Rows of ``tools.tsv`` keyed by tool name (columns: url, commit, license, env, use, used_by)."""
    with open(MANIFEST, newline="", encoding="utf-8") as handle:
        return {row["name"]: row for row in csv.DictReader(handle, delimiter="\t")}


def tool_dir(name: str) -> Path:
    """Folder of the clone of ``name``; the name must be a row of ``tools.tsv``."""
    if name not in manifest():
        raise KeyError(f"{name!r} is not in {MANIFEST.name}; known tools: {', '.join(manifest())}")
    return tools_dir() / name


def pinned_commit(name: str) -> str:
    """Commit that ``name`` is pinned to."""
    return manifest()[name]["commit"]


def main(argv: list[str]) -> str:
    """Answer one command-line question: the folder, a clone, or the commit or URL of a tool."""
    if not argv:
        return str(tools_dir())
    if argv[0] in ("--commit", "--url") and len(argv) == 2:
        tool_dir(argv[1])                      # raises for an unknown tool
        return manifest()[argv[1]][argv[0].lstrip("-")]
    if len(argv) == 1 and not argv[0].startswith("-"):
        return str(tool_dir(argv[0]))
    raise SystemExit("usage: location.py [TOOL | --commit TOOL | --url TOOL]")


if __name__ == "__main__":
    print(main(sys.argv[1:]))
