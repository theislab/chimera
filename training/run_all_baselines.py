"""Cross-method integration baselines — orchestrator.

Single source of truth for the (dataset × method × seed × batch_key)
matrix. Loops `baseline_runner.py` over the matrix, sequential CPU
subprocesses, dataset-major (one dataset at a time, all five Tier-1
methods in `[pca, harmony, scvi, scanvi, sysvi]` order before moving
to the next dataset). Two-phase: Tier-1 first across every dataset,
then Tier-2 (camex, saturn) which is currently NotImplementedError
stubs in the runner.

**Importable from a notebook**: the orchestrator is *primarily* an
importable Python module — `from chimera.training.run_all_baselines
import DATASETS, run_one, run_phase, write_summary` — and the CLI is
a thin argparse wrapper that calls the same functions. The notebook
(`chimera/notebooks/02_baselines.ipynb`) drives the matrix cell-by-cell
using the importable API, not via subprocess to this script.

Operational rules: CPU only with 5 threads (Rule A), max_epochs ≤ 150
default 100 (Rule B), sequential subprocesses (Rule C). The runner
enforces the per-process budget; the orchestrator enforces sequential
ordering and skip-if-cached.

Module import is intentionally cheap: importing this file does NOT
import torch / scvi-tools / lightning. The DatasetSpec list and
helper functions are loaded; the model machinery is invoked only by
the runner subprocess. A notebook can `from … import DATASETS` in
under one second.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("run_all_baselines")

REPO_ROOT = Path(__file__).resolve().parents[1]
# Honor CHIMERA_RUNS_ROOT (same hook as baseline_runner) so tests
# redirecting one redirect both.
RUNS_ROOT = Path(os.environ["CHIMERA_RUNS_ROOT"]) if os.environ.get(
    "CHIMERA_RUNS_ROOT"
) else REPO_ROOT / "training" / "runs"
LOG_ROOT = RUNS_ROOT / "baselines_logs"
RUNNER_SCRIPT = REPO_ROOT / "training" / "baseline_runner.py"

TIER1_METHODS: List[str] = ["pca", "harmony", "scvi", "scanvi", "sysvi"]
TIER2_METHODS: List[str] = ["camex", "saturn"]
ALL_METHODS: List[str] = TIER1_METHODS + TIER2_METHODS


@dataclass
class DatasetSpec:
    """Single source of truth for a dataset's covariate routing.

    `path` — repo-relative or absolute path to the .h5ad.
    `species_col` — `.obs` column carrying species identity.
    `batch_col` — `.obs` column carrying within-species batch
        (or 1-to-1 with species in pancreas; the runner detects this
        and adjusts SysVI's `categorical_covariate_keys` accordingly).
    `labels_col` — `.obs` column with cell-type labels (scANVI only).
    `note` — short human-readable rationale for this row.
    """

    path: str
    species_col: str
    batch_col: str
    labels_col: str
    note: str = ""

    @property
    def path_resolved(self) -> Path:
        p = Path(self.path)
        if not p.is_absolute():
            p = (REPO_ROOT / self.path).resolve()
        return p

    @property
    def stem(self) -> str:
        return self.path_resolved.stem


# ----------------------------------------------------------------------
# Dataset matrix — single source of truth.
#
# Verified 2026-04-26: every entry has the listed .obs columns.
# Routing rules:
#   - Pancreas: batch_col=batch (≡ species 1-to-1 except H+Mo where
#     Monkey has 2 within-species batches; the runner detects this and
#     handles SysVI's covariate routing accordingly).
#   - Testis: batch_col=sample_batch (donor-level, within-species).
#   - labels_col=cell_type for every dataset (auto-fallback to
#     cell_ontology_class is in the runner; populated explicitly here
#     because every adata in scope has cell_type populated).
#   - Not included: van_zyl_unified.h5ad. The processed object exists but no
#     Chimera recipe was evaluated on it, so it is not part of this matrix.
# ----------------------------------------------------------------------
DATASETS: List[DatasetSpec] = [
    # Pancreas (priority — do these first per the prompt).
    DatasetSpec(
        path="data/processed/camex_human_monkey.h5ad",
        species_col="species",
        batch_col="batch",
        labels_col="cell_type",
        note="2-sp pancreas; H+Mo. batch carries 1 within-species dof in Monkey.",
    ),
    DatasetSpec(
        path="data/processed/camex_human_mouse.h5ad",
        species_col="species",
        batch_col="batch",
        labels_col="cell_type",
        note="2-sp pancreas; H+Mouse. batch ≡ species 1-to-1.",
    ),
    DatasetSpec(
        path="data/processed/camex_human_zebrafish.h5ad",
        species_col="species",
        batch_col="batch",
        labels_col="cell_type",
        note="2-sp pancreas; H+Zebrafish. batch ≡ species 1-to-1.",
    ),
    DatasetSpec(
        path="data/processed/camex_pancreas_unified.h5ad",
        species_col="species",
        batch_col="batch",
        labels_col="cell_type",
        note="4-sp pancreas. batch ≡ species 1-to-1.",
    ),
    # Testis.
    DatasetSpec(
        path="data/processed/camex_testis_human_macaque.h5ad",
        species_col="species",
        batch_col="sample_batch",
        labels_col="cell_type",
        note="2-sp testis; H+Mac. sample_batch is donor-level, within-species.",
    ),
    DatasetSpec(
        path="data/processed/camex_testis_human_mouse.h5ad",
        species_col="species",
        batch_col="sample_batch",
        labels_col="cell_type",
        note="2-sp testis; H+Mouse. sample_batch is donor-level, within-species.",
    ),
    DatasetSpec(
        path="data/processed/camex_testis_unified.h5ad",
        species_col="species",
        batch_col="sample_batch",
        labels_col="cell_type",
        note="3-sp testis; H+Mac+Mouse. sample_batch is donor-level.",
    ),
]


@dataclass
class RunResult:
    """One (dataset, method, seed) outcome.

    `status` is one of:
      - OK         — runner exited 0, latent saved.
      - SKIPPED    — cache hit; subprocess not launched.
      - FAILED     — runner raised; see error_msg + log_path.
      - NOT_IMPL   — Tier-2 stub raised NotImplementedError.
    """

    dataset_stem: str
    method: str
    seed: int
    status: str
    cache_path: Optional[str] = None
    cache_dir: Optional[str] = None
    wall_time_s: float = 0.0
    error_msg: Optional[str] = None
    log_path: Optional[str] = None
    latent_shape: Optional[Tuple[int, int]] = None
    batch_key: Optional[str] = None
    species_key: Optional[str] = None
    labels_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        if self.latent_shape is not None:
            d["latent_shape"] = list(self.latent_shape)
        return d


# ----------------------------------------------------------------------
# Read-only inspection helpers (no training, no torch import).
# ----------------------------------------------------------------------
def list_datasets() -> List[DatasetSpec]:
    return list(DATASETS)


def list_methods(tier: Optional[int] = None) -> List[str]:
    if tier == 1:
        return list(TIER1_METHODS)
    if tier == 2:
        return list(TIER2_METHODS)
    if tier is None:
        return list(ALL_METHODS)
    raise ValueError(f"tier must be 1, 2, or None; got {tier}")


def expected_cache_path(spec: DatasetSpec, method: str, seed: int = 42) -> Path:
    """Best-guess cache path for default-config runs.

    For runs with overrides (--max-epochs / --n-latent), the cache
    directory name picks up additional suffixes; this helper covers
    the orchestrator's default-only path. Used for skip-if-cached.
    """
    stem = spec.stem
    bk = spec.batch_col
    if method == "pca":
        return RUNS_ROOT / "baselines_pca" / stem / "z_mean.npy"
    if method == "harmony":
        suffix = f"-bk_{bk}" if bk != "species" else ""
        return RUNS_ROOT / "baselines_harmony" / f"{stem}-s{seed}{suffix}" / "z_mean.npy"
    if method == "scvi":
        bk_suffix = f"-bk_{bk}" if bk != "species" else ""
        return (
            RUNS_ROOT
            / "baselines_scvi"
            / f"{stem}-s{seed}-e100-nl10{bk_suffix}"
            / "z_mean.npy"
        )
    if method == "scanvi":
        suffix = f"-bk_{bk}" if bk != "species" else ""
        return RUNS_ROOT / "baselines_scanvi" / f"{stem}-s{seed}{suffix}" / "z_mean.npy"
    if method == "sysvi":
        suffix = f"-bk_{bk}" if bk != "species" else ""
        return RUNS_ROOT / "baselines_sysvi" / f"{stem}-s{seed}{suffix}" / "z_mean.npy"
    if method in TIER2_METHODS:
        return RUNS_ROOT / f"baselines_{method}" / f"{stem}-s{seed}" / "z_mean.npy"
    raise ValueError(f"Unknown method '{method}'.")


def is_cached(spec: DatasetSpec, method: str, seed: int = 42) -> bool:
    return expected_cache_path(spec, method, seed).exists()


# ----------------------------------------------------------------------
# Subprocess invocation.
# ----------------------------------------------------------------------
_LATENT_RE = re.compile(r"^LATENT_PATH=(.+)$")
_CACHE_RE = re.compile(r"^CACHE_DIR=(.+)$")
_OK_RE = re.compile(r"^METHOD_OK=(.+)$")
_FAIL_RE = re.compile(r"^METHOD_FAILED=(\w+):\s*(.*)$")


def _python_prefix() -> List[str]:
    """Decide how to invoke the runner subprocess.

    - If the orchestrator is itself running inside `chimera_env` (the
      common case when launched from a notebook with the chimera_env
      kernel, or via `mamba run -n chimera_env ...`), we use the same
      `sys.executable` directly. Skipping the wrapper avoids ~1s of
      mamba startup overhead per invocation, which compounds across
      the matrix.
    - Otherwise we wrap with `mamba run -n chimera_env python` so the
      runner still hits the right env.

    Override with the `CHIMERA_BASELINE_PYTHON` env var (e.g. tests
    that want to use `sys.executable` deterministically, or a remote
    runner that uses a different invocation prefix).
    """
    override = os.environ.get("CHIMERA_BASELINE_PYTHON")
    if override:
        return shlex.split(override)
    if os.environ.get("CONDA_DEFAULT_ENV") == "chimera_env":
        return [sys.executable]
    return ["mamba", "run", "-n", "chimera_env", "python"]


def _build_runner_cmd(
    spec: DatasetSpec,
    method: str,
    seed: int,
    *,
    force: bool,
    max_epochs: Optional[int],
) -> List[str]:
    """Build the argv for a single runner subprocess."""
    cmd = [
        *_python_prefix(),
        str(RUNNER_SCRIPT),
        "--adata",
        str(spec.path_resolved),
        "--method",
        method,
        "--seed",
        str(seed),
        "--species-key",
        spec.species_col,
    ]
    if method != "pca":
        cmd += ["--batch-key", spec.batch_col]
    if method == "scanvi":
        cmd += ["--labels-key", spec.labels_col]
    if max_epochs is not None:
        cmd += ["--max-epochs", str(max_epochs)]
    if force:
        cmd += ["--force"]
    return cmd


def _parse_runner_stdout(stdout: str) -> Dict[str, Any]:
    """Extract LATENT_PATH / CACHE_DIR / status markers from stdout."""
    info: Dict[str, Any] = {"latent_path": None, "cache_dir": None, "ok": [], "failed": []}
    for line in stdout.splitlines():
        m = _LATENT_RE.match(line.strip())
        if m:
            info["latent_path"] = m.group(1).strip()
            continue
        m = _CACHE_RE.match(line.strip())
        if m:
            info["cache_dir"] = m.group(1).strip()
            continue
        m = _OK_RE.match(line.strip())
        if m:
            info["ok"].append(m.group(1).strip())
            continue
        m = _FAIL_RE.match(line.strip())
        if m:
            info["failed"].append((m.group(1).strip(), m.group(2).strip()))
    return info


def _read_latent_shape(latent_path: Optional[str]) -> Optional[Tuple[int, int]]:
    if latent_path is None:
        return None
    p = Path(latent_path)
    if not p.exists():
        return None
    try:
        import numpy as np

        z = np.load(p, mmap_mode="r")
        return tuple(int(s) for s in z.shape)
    except Exception:
        return None


def _read_runner_config(cache_dir: Optional[str]) -> Dict[str, Any]:
    if cache_dir is None:
        return {}
    p = Path(cache_dir) / "config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def run_one(
    spec: DatasetSpec,
    method: str,
    seed: int = 42,
    *,
    force: bool = False,
    max_epochs: Optional[int] = None,
    log_dir: Optional[Path] = None,
    verbose: bool = True,
) -> RunResult:
    """Run one (dataset, method, seed). Skip-if-cached unless force=True.

    Subprocess isolation prevents one method's torch/Lightning state
    from leaking into the next; matches how compare_vs_scvi is invoked.
    """
    log_dir = Path(log_dir) if log_dir is not None else LOG_ROOT
    log_dir.mkdir(parents=True, exist_ok=True)

    # Skip-if-cached at the orchestrator level so we don't even pay
    # subprocess-spawn overhead.
    cache_path = expected_cache_path(spec, method, seed)
    if not force and cache_path.exists():
        latent_shape = _read_latent_shape(str(cache_path))
        cfg = _read_runner_config(str(cache_path.parent))
        if verbose:
            logger.info(
                "SKIP cached: %s / %s / s%d → %s",
                spec.stem,
                method,
                seed,
                cache_path,
            )
        return RunResult(
            dataset_stem=spec.stem,
            method=method,
            seed=seed,
            status="SKIPPED",
            cache_path=str(cache_path),
            cache_dir=str(cache_path.parent),
            latent_shape=latent_shape,
            batch_key=cfg.get("batch_key", spec.batch_col),
            species_key=cfg.get("species_key", spec.species_col),
            labels_key=cfg.get("labels_key"),
        )

    cmd = _build_runner_cmd(spec, method, seed, force=force, max_epochs=max_epochs)
    log_path = (
        log_dir
        / f"{method}-{spec.stem}-s{seed}-bk_{spec.batch_col}.log"
    )

    if verbose:
        logger.info(
            "RUN  %s / %s / s%d → log=%s",
            spec.stem,
            method,
            seed,
            log_path,
        )
        logger.info("    cmd: %s", " ".join(shlex.quote(c) for c in cmd))

    t0 = time.time()
    with open(log_path, "w") as fh:
        fh.write(
            f"# {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        )
        fh.write(f"# cmd: {' '.join(shlex.quote(c) for c in cmd)}\n\n")
        fh.flush()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                check=False,
            )
        except FileNotFoundError as e:
            elapsed = time.time() - t0
            err = f"Failed to launch subprocess: {e}"
            fh.write(err + "\n")
            return RunResult(
                dataset_stem=spec.stem,
                method=method,
                seed=seed,
                status="FAILED",
                wall_time_s=elapsed,
                error_msg=err,
                log_path=str(log_path),
                batch_key=spec.batch_col,
                species_key=spec.species_col,
                labels_key=spec.labels_col if method == "scanvi" else None,
            )
        elapsed = time.time() - t0
        stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
        fh.write(stdout)
        fh.write(f"\n# returncode={proc.returncode}, elapsed={elapsed:.1f}s\n")

    info = _parse_runner_stdout(stdout)
    latent_shape = _read_latent_shape(info["latent_path"])
    cfg = _read_runner_config(info["cache_dir"])

    if proc.returncode != 0 or info["failed"]:
        # NotImplementedError from Tier-2 stubs gets a distinct status
        # so the summary table can render those differently than real failures.
        is_not_impl = any("NotImplementedError" in fmsg for _, fmsg in info["failed"])
        status = "NOT_IMPL" if is_not_impl else "FAILED"
        err_msg = "; ".join(f"{m}: {f}" for m, f in info["failed"]) or "see log"
        if verbose:
            logger.error(
                "FAIL %s / %s / s%d (rc=%d): %s",
                spec.stem,
                method,
                seed,
                proc.returncode,
                err_msg,
            )
        return RunResult(
            dataset_stem=spec.stem,
            method=method,
            seed=seed,
            status=status,
            wall_time_s=elapsed,
            error_msg=err_msg,
            log_path=str(log_path),
            cache_path=info["latent_path"],
            cache_dir=info["cache_dir"],
            latent_shape=latent_shape,
            batch_key=cfg.get("batch_key", spec.batch_col),
            species_key=cfg.get("species_key", spec.species_col),
            labels_key=cfg.get("labels_key"),
        )

    if verbose:
        logger.info(
            "OK   %s / %s / s%d in %.1fs → %s",
            spec.stem,
            method,
            seed,
            elapsed,
            info["latent_path"],
        )
    return RunResult(
        dataset_stem=spec.stem,
        method=method,
        seed=seed,
        status="OK",
        cache_path=info["latent_path"],
        cache_dir=info["cache_dir"],
        wall_time_s=elapsed,
        log_path=str(log_path),
        latent_shape=latent_shape,
        batch_key=cfg.get("batch_key", spec.batch_col),
        species_key=cfg.get("species_key", spec.species_col),
        labels_key=cfg.get("labels_key"),
    )


def run_phase(
    methods: List[str],
    datasets: Optional[List[DatasetSpec]] = None,
    seeds: Iterable[int] = (42,),
    *,
    force: bool = False,
    max_epochs: Optional[int] = None,
    log_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Iterator[RunResult]:
    """Iterate (dataset × method × seed) sequentially, dataset-major.

    Yields after each call so a notebook can render progress live.
    Methods within a dataset run in the order given (typically
    `pca → harmony → scvi → scanvi → sysvi`, cheapest first, so a
    partial session still produces maximum coverage).
    """
    datasets = list(datasets) if datasets is not None else list(DATASETS)
    seeds = list(seeds)
    for spec in datasets:
        for method in methods:
            for seed in seeds:
                yield run_one(
                    spec,
                    method,
                    seed=seed,
                    force=force,
                    max_epochs=max_epochs,
                    log_dir=log_dir,
                    verbose=verbose,
                )


def run_all(
    *,
    datasets: Optional[List[DatasetSpec]] = None,
    seeds: Iterable[int] = (42,),
    include_tier2: bool = False,
    force: bool = False,
    max_epochs: Optional[int] = None,
    log_dir: Optional[Path] = None,
    verbose: bool = True,
) -> List[RunResult]:
    """Run Phase 1 (Tier-1) across every dataset, then optionally Phase 2.

    Returns the full list of RunResult.
    """
    results: List[RunResult] = []
    for r in run_phase(
        TIER1_METHODS,
        datasets=datasets,
        seeds=seeds,
        force=force,
        max_epochs=max_epochs,
        log_dir=log_dir,
        verbose=verbose,
    ):
        results.append(r)
    if include_tier2:
        for r in run_phase(
            TIER2_METHODS,
            datasets=datasets,
            seeds=seeds,
            force=force,
            max_epochs=max_epochs,
            log_dir=log_dir,
            verbose=verbose,
        ):
            results.append(r)
    return results


# ----------------------------------------------------------------------
# Summary persistence.
# ----------------------------------------------------------------------
def write_summary(results: List[RunResult], path: Optional[Path] = None) -> Path:
    path = path if path is not None else (LOG_ROOT / "_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_results": len(results),
        "n_ok": sum(1 for r in results if r.status == "OK"),
        "n_skipped": sum(1 for r in results if r.status == "SKIPPED"),
        "n_failed": sum(1 for r in results if r.status == "FAILED"),
        "n_not_impl": sum(1 for r in results if r.status == "NOT_IMPL"),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Summary written to %s (%d results)", path, len(results))
    return path


def load_summary(path: Optional[Path] = None) -> List[RunResult]:
    path = path if path is not None else (LOG_ROOT / "_summary.json")
    payload = json.loads(path.read_text())
    out: List[RunResult] = []
    for d in payload.get("results", []):
        ls = d.get("latent_shape")
        if ls is not None:
            ls = tuple(int(x) for x in ls)
        out.append(
            RunResult(
                dataset_stem=d["dataset_stem"],
                method=d["method"],
                seed=int(d["seed"]),
                status=d["status"],
                cache_path=d.get("cache_path"),
                cache_dir=d.get("cache_dir"),
                wall_time_s=float(d.get("wall_time_s") or 0.0),
                error_msg=d.get("error_msg"),
                log_path=d.get("log_path"),
                latent_shape=ls,
                batch_key=d.get("batch_key"),
                species_key=d.get("species_key"),
                labels_key=d.get("labels_key"),
            )
        )
    return out


def format_summary_table(results: List[RunResult]) -> str:
    """Plain-text summary grid. Used by the CLI's final print and the
    notebook can use the same to keep notebook+CLI output aligned."""
    headers = ["dataset", "method", "seed", "bk", "status", "wall_s", "shape", "cache"]
    rows = []
    for r in results:
        rows.append(
            [
                r.dataset_stem,
                r.method,
                str(r.seed),
                r.batch_key or "-",
                r.status,
                f"{r.wall_time_s:.0f}",
                f"{r.latent_shape}" if r.latent_shape else "-",
                r.cache_path or "-",
            ]
        )
    widths = [max(len(h), *(len(row[i]) for row in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    for row in rows:
        lines.append(fmt.format(*row))
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI — thin argparse wrapper around the importable API.
# ----------------------------------------------------------------------
def _select_datasets(filter_strings: Optional[List[str]]) -> List[DatasetSpec]:
    if not filter_strings:
        return list(DATASETS)
    keep: List[DatasetSpec] = []
    for spec in DATASETS:
        if any(s in spec.stem or s == str(spec.path_resolved) for s in filter_strings):
            keep.append(spec)
    if not keep:
        raise ValueError(
            f"No datasets matched filters {filter_strings}. "
            f"Available stems: {[s.stem for s in DATASETS]}"
        )
    return keep


def _select_methods(filter_string: Optional[str], include_tier2: bool) -> List[str]:
    if filter_string:
        methods = [m.strip() for m in filter_string.split(",") if m.strip()]
        for m in methods:
            if m not in ALL_METHODS:
                raise ValueError(f"Unknown method '{m}'. Choices: {ALL_METHODS}")
        return methods
    return TIER1_METHODS + (TIER2_METHODS if include_tier2 else [])


def _parse_seeds(seed_string: str) -> List[int]:
    return [int(s.strip()) for s in seed_string.split(",") if s.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--methods",
        type=str,
        default=None,
        help=f"Comma-sep; default = Tier-1 ({','.join(TIER1_METHODS)}). "
        f"Tier-2: {','.join(TIER2_METHODS)} (currently NotImplementedError stubs).",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="One or more dataset stems / paths (substring match). "
        "Default = full DATASETS matrix.",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-sep seeds; default 42.",
    )
    ap.add_argument(
        "--include-tier2",
        action="store_true",
        help="Include camex / saturn (currently raise NotImplementedError).",
    )
    ap.add_argument("--force", action="store_true", help="Re-run even if cached.")
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override default max_epochs cap (≤150).",
    )
    ap.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=f"Default {LOG_ROOT}.",
    )
    ap.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help=f"Default {LOG_ROOT / '_summary.json'}.",
    )
    args = ap.parse_args(argv)

    datasets = _select_datasets(args.datasets)
    methods = _select_methods(args.methods, args.include_tier2)
    seeds = _parse_seeds(args.seeds)
    log_dir = Path(args.log_dir) if args.log_dir else LOG_ROOT
    summary_path = Path(args.summary_path) if args.summary_path else log_dir / "_summary.json"

    logger.info(
        "Orchestrator: datasets=%s, methods=%s, seeds=%s, force=%s, max_epochs=%s",
        [s.stem for s in datasets],
        methods,
        seeds,
        args.force,
        args.max_epochs,
    )

    results: List[RunResult] = []
    for r in run_phase(
        methods,
        datasets=datasets,
        seeds=seeds,
        force=args.force,
        max_epochs=args.max_epochs,
        log_dir=log_dir,
        verbose=True,
    ):
        results.append(r)

    write_summary(results, path=summary_path)
    print()
    print(format_summary_table(results))
    print()
    n_failed = sum(1 for r in results if r.status == "FAILED")
    n_not_impl = sum(1 for r in results if r.status == "NOT_IMPL")
    n_ok = sum(1 for r in results if r.status == "OK")
    n_skipped = sum(1 for r in results if r.status == "SKIPPED")
    print(
        f"OK={n_ok}  SKIPPED={n_skipped}  FAILED={n_failed}  NOT_IMPL={n_not_impl}  "
        f"of {len(results)} total."
    )
    print(f"Summary: {summary_path}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
