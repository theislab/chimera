"""Atomic task outputs: every file is written as ``*.tmp`` and renamed; ``DONE`` comes last."""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

from revision.common import gene_ids


def git_sha(short: bool = False) -> str:
    try:
        args = ["git", "-C", str(gene_ids.REPO_ROOT), "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance must not crash the analysis
        return "unknown"


def git_dirty(paths: list[str]) -> str:
    try:
        r = subprocess.run(["git", "-C", str(gene_ids.REPO_ROOT), "status", "--porcelain", "--", *paths],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


class TaskOutput:
    """Output folder of one task. ``prerun`` marks a check run outside ``chimera_env``."""

    def __init__(self, task: str, outdir: str | os.PathLike | None, prerun: bool = False,
                 force: bool = False):
        self.task = task
        self.dir = Path(outdir) if outdir else gene_ids.REPO_ROOT / "results" / "revision" / task
        self.prerun = prerun
        self.dir.mkdir(parents=True, exist_ok=True)
        self.done = self.dir / "DONE"
        if self.done.exists() and not force:
            print(f"{self.done} exists; nothing to do (use --force to recompute)")
            sys.exit(0)
        if self.done.exists():
            self.done.unlink()
        self.files: list[str] = []
        self.started = _dt.datetime.now().isoformat(timespec="seconds")

    def _tmp(self, name: str) -> tuple[Path, Path]:
        final = self.dir / name
        final.parent.mkdir(parents=True, exist_ok=True)
        return final, final.with_name(final.name + ".tmp")

    def _commit(self, tmp: Path, final: Path) -> Path:
        os.replace(tmp, final)
        self.files.append(str(final.relative_to(self.dir)))
        return final

    def csv(self, df, name: str, index: bool = False) -> Path:
        final, tmp = self._tmp(name)
        df.to_csv(tmp, index=index)
        return self._commit(tmp, final)

    def parquet(self, df, name: str) -> Path:
        final, tmp = self._tmp(name)
        df.to_parquet(tmp, index=False)
        return self._commit(tmp, final)

    def json(self, obj, name: str) -> Path:
        final, tmp = self._tmp(name)
        tmp.write_text(json.dumps(obj, indent=2, default=float))
        return self._commit(tmp, final)

    def figure(self, fig, name: str) -> Path:
        final, tmp = self._tmp(name)
        fmt = final.suffix.lstrip(".")
        fig.savefig(tmp, format=fmt)
        return self._commit(tmp, final)

    def finish(self, extra: dict | None = None) -> None:
        prov = dict(task=self.task, prerun=self.prerun,
                    label=("PRE-RUN outside chimera_env; not the official result" if self.prerun
                           else "official run"),
                    git_sha=git_sha(), python=sys.version.split()[0],
                    conda_env=os.environ.get("CONDA_DEFAULT_ENV", ""), argv=sys.argv,
                    started=self.started, finished=_dt.datetime.now().isoformat(timespec="seconds"),
                    files=self.files)
        if extra:
            prov.update(extra)
        final, tmp = self._tmp("_provenance.json")
        tmp.write_text(json.dumps(prov, indent=2, default=str))
        os.replace(tmp, final)
        tmp = self.done.with_name("DONE.tmp")
        tmp.write_text(prov["finished"] + "\n")
        os.replace(tmp, self.done)
        print(f"wrote {len(self.files)} files to {self.dir}; DONE")


def add_common_args(parser) -> None:
    parser.add_argument("--outdir", default=None,
                        help="output folder (default results/revision/<task>)")
    parser.add_argument("--prerun", action="store_true",
                        help="label outputs as a pre-run outside chimera_env")
    parser.add_argument("--force", action="store_true", help="recompute even if DONE exists")
