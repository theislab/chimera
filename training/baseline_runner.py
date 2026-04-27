"""Cross-method integration baselines — single-dataset runner.

Trains (or loads cached) latent embeddings for one method on one adata,
mirrors the caching layout of `compare_vs_scvi.py::get_or_train_scvi`,
and writes a per-method `config.json` next to a `z_mean.npy` so that a
downstream metrics step can pair the baselines against Chimera runs.

**Latents only — no metric computation, no UMAPs, no reports.**

Methods (Tier-1 wired end-to-end):
    pca       — log-norm + PCA on the full gene set, no HVG, no scaling.
                Canonical recipe: scanpy PCA tutorial.
    harmony   — log-norm + PCA + harmonypy.run_harmony on the same input.
                Canonical recipe: harmonypy v2.0.0 (slowkow/harmonypy).
    scvi      — vanilla scVI; reuses `compare_vs_scvi.get_or_train_scvi`
                so the same (adata, seed, batch_key, defaults) tuple
                produces a byte-identical latent and shares cache.
                Canonical recipe: scvi-tools 1.4.x SCVI quick-start.
    scanvi    — scvi-tools two-step recipe: SCVI pretrain →
                SCANVI.from_scvi_model → SCANVI.train.
                Canonical recipe: scvi-tools harmonization tutorial
                (https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/harmonization.html).
                Note: tutorial uses `unlabeled_category="Unknown"`; the
                prompt's `"UNK"` is overridden in favor of canonical.
    sysvi     — scvi.external.SysVI (cross-species VampPrior cVAE,
                Hrovatin et al. 2023). Now lives in scvi-tools — the
                standalone `theislab/cross_system_integration` repo is
                unmaintained per the Theislab team. Canonical recipe:
                scvi-tools sysVI tutorial
                (https://docs.scvi-tools.org/en/1.3.3/tutorials/notebooks/scrna/sysVI.html).

Tier-2 (stubs):
    camex, saturn — clean NotImplementedError naming the gating step.

Operational rules (overrides any inferred default):
    - Rule A: CPU only, 5 torch threads. No MPS, no CUDA. Enforced at
      module import via env vars + torch.set_num_threads(5), and at
      training-call time via `accelerator="cpu"`. Refinement.md's
      "MPS sequentiality" guidance applies to *Chimera* training; the
      baseline pipeline runs on CPU.
    - Rule B: `max_epochs ≤ 150`, default 100. scANVI two-step's
      `max_epochs_scvi + max_epochs_scanvi ≤ 150` (default 100 + 20).
      Library-default `max_epochs` for scvi-tools models scales with
      dataset size and can be hundreds-to-thousands of epochs on small
      datasets — the cap is a fairness ceiling vs Chimera's budget.
    - Rule C: orchestrator runs methods sequentially. The runner trains
      one thing per invocation and exits.

**Fair-playing-field**: no HVG selection in any method (Chimera doesn't
HVG either), no `sc.pp.scale` unless method-essential (PCA / Harmony do
NOT need it on our curated gene set), no per-method gene subsetting,
no post-hoc latent surgery. Each method's published preprocessing is
otherwise honored.

**Dependencies**: any new package goes through
`chimera/pyproject.toml`'s `[project.optional-dependencies] baselines`
group; install via the "Force re-sync" cell in
`chimera/notebooks/00_environment.ipynb`. Never pip-install ad-hoc.

Usage (CLI, mirrors `compare_vs_scvi.py` style):

    python training/baseline_runner.py \\
        --adata data/processed/camex_human_monkey.h5ad \\
        --method scanvi \\
        --species-key species \\
        --batch-key batch \\
        --seed 42

The orchestrator (`run_all_baselines.py`) shells out to this script
once per (dataset, method, seed) and parses `LATENT_PATH=<path>` from
stdout.
"""
from __future__ import annotations

# --- Rule A enforcement: thread cap setup BEFORE any torch import ---
# scvi-tools, harmonypy, lightning, and SysVI all transitively import
# torch; they honor `torch.set_num_threads` and the BLAS env vars below
# only if those are set before their own torch initialization.
import os as _os

_os.environ.setdefault("OMP_NUM_THREADS", "5")
_os.environ.setdefault("MKL_NUM_THREADS", "5")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "5")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "5")
_os.environ.setdefault("PYTHONHASHSEED", _os.environ.get("PYTHONHASHSEED", "0"))

import torch as _torch  # noqa: E402  (after env vars by design)

_torch.set_num_threads(5)
try:
    _torch.set_num_interop_threads(2)
except RuntimeError:
    # set_num_interop_threads must be called before any parallel work
    # has happened. If a re-import in a long-lived process triggers
    # this, the existing setting is fine; ignore.
    pass
# --- end Rule A enforcement ---

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("baseline_runner")

REPO_ROOT = Path(__file__).resolve().parents[1]
# `CHIMERA_RUNS_ROOT` lets tests redirect the cache to tmp_path. In
# normal operation the env var is unset and the cache lives at
# chimera/training/runs alongside the existing baselines_scvi cache.
RUNS_ROOT = Path(os.environ["CHIMERA_RUNS_ROOT"]) if os.environ.get(
    "CHIMERA_RUNS_ROOT"
) else REPO_ROOT / "training" / "runs"

# Defaults applied by the orchestrator. These are the "no override"
# values; the corresponding `--*` CLI flags exist but the orchestrator
# does NOT pass them. When unset, the runner records the resolved
# value the library actually used in `config.json`.
DEFAULT_N_LATENT = 10
DEFAULT_N_HIDDEN = 128  # scVI-only; matches existing cache convention.
DEFAULT_MAX_EPOCHS = 100  # Rule B: ≤ 150, prefer 100. Cap for trainable methods.
DEFAULT_SCANVI_HEAD_EPOCHS = 20  # X+Y ≤ 150; tutorial uses 20.
DEFAULT_SCVI_PRETRAIN_EPOCHS = 100  # scANVI's SCVI pretrain step.
DEVICE = "cpu"  # Rule A.
TORCH_NUM_THREADS = 5
TORCH_NUM_INTEROP_THREADS = 2

UNLABELED_CATEGORY = "Unknown"  # canonical scvi-tools tutorial convention.

# Methods whose computation is deterministic given (data, n_latent), not
# (data, seed). PCA falls in this bucket — its cache directory therefore
# omits the seed suffix.
DETERMINISTIC_METHODS = {"pca"}

# Methods that consume raw counts (NB / ZINB / Poisson likelihoods).
# These get the raw-counts safety check; everything else log-normalizes
# inside the runner on a copy of the adata.
RAW_COUNTS_METHODS = {"scvi", "scanvi"}
LOG_NORM_METHODS = {"pca", "harmony", "sysvi"}

KNOWN_METHODS = ["pca", "harmony", "scvi", "scanvi", "sysvi", "camex", "saturn"]


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) — same as compare_vs_scvi."""
    random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)


def _git_sha() -> Optional[str]:
    """Return the current git commit SHA, or None if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _pkg_versions(method: str) -> Dict[str, str]:
    """Collect package versions relevant to the given method."""
    versions: Dict[str, str] = {}
    try:
        from importlib.metadata import version as _v

        for pkg in ("scvi-tools", "anndata", "scanpy", "torch", "lightning"):
            try:
                versions[pkg] = _v(pkg)
            except Exception:
                pass
        if method == "harmony":
            try:
                versions["harmonypy"] = _v("harmonypy")
            except Exception:
                pass
    except Exception:
        pass
    return versions


def _resolve_labels_key(adata, requested: Optional[str] = None) -> str:
    """Prefer `cell_type`, fall back to `cell_ontology_class`. Error otherwise.

    Mirrors the fallback used in compare_vs_scvi.py main() (lines 581-586).
    """
    if requested is not None:
        if requested in adata.obs.columns:
            return requested
        raise KeyError(
            f"Requested labels_key '{requested}' not in adata.obs.columns "
            f"({list(adata.obs.columns)[:20]}…)."
        )
    if "cell_type" in adata.obs.columns:
        return "cell_type"
    if "cell_ontology_class" in adata.obs.columns:
        return "cell_ontology_class"
    raise KeyError(
        "Neither 'cell_type' nor 'cell_ontology_class' present in "
        f"adata.obs.columns: {list(adata.obs.columns)[:20]}…"
    )


def _assert_obs_keys(adata, *keys: str) -> None:
    """Pre-flight check that every requested .obs key actually exists.

    A KeyError at setup_anndata time is a wiring bug; an early assert
    makes the diagnostic 100x faster.
    """
    missing = [k for k in keys if k is not None and k not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"adata.obs is missing required keys {missing}. "
            f"Available: {sorted(adata.obs.columns)[:30]}"
        )


def _assert_raw_counts(adata) -> None:
    """Warn if .X looks log-normalized when raw counts were expected.

    Mirrors compare_vs_scvi.py's safety block (lines 238-243). Does NOT
    error — some real data has float counts that look fractional.
    """
    X = adata.X
    if hasattr(X, "data"):
        sample_max = float(X.data.max()) if X.data.size > 0 else 0.0
    else:
        sample_max = float(np.asarray(X).max())
    if sample_max < 50:
        # Could be log-normalized (max ~10) or genuinely sparse.
        # Cross-check by sampling a few rows and seeing if values are
        # near-integer (raw) or fractional (log-normed).
        head = adata.X[:5]
        head_arr = head.toarray() if hasattr(head, "toarray") else np.asarray(head)
        is_integer = np.allclose(head_arr, np.round(head_arr))
        if not is_integer:
            logger.warning(
                "adata.X max=%.3f and head looks fractional — verify these "
                "are raw counts before training a NB/ZINB model.",
                sample_max,
            )


def _get_n_obs_from_X(adata) -> int:
    return int(adata.n_obs)


def _detect_within_species_batch(adata, species_col: str, batch_col: str) -> bool:
    """True iff `batch_col` carries within-species variation.

    Pancreas H+Mo has 2 within-species batches in Monkey (the only
    pancreas dataset where batch ≢ species); other pancreas datasets
    have batch == species 1-to-1 and the column is redundant. Testis
    `sample_batch` is donor-level and always carries genuine variation.

    The detector is purely data-driven so the runner does not need a
    hard-coded list of (dataset, has-within-species-batch) pairs.
    """
    if batch_col == species_col:
        return False
    if batch_col not in adata.obs.columns or species_col not in adata.obs.columns:
        return False
    counts_per_species = adata.obs.groupby(species_col, observed=True)[batch_col].nunique()
    return bool((counts_per_species > 1).any())


def _build_cache_dir_name(
    adata_stem: str,
    method: str,
    seed: int,
    *,
    batch_key: Optional[str] = None,
    n_latent: Optional[int] = None,
    n_hidden: Optional[int] = None,
    max_epochs: Optional[int] = None,
    n_latent_default: Optional[int] = None,
    n_hidden_default: Optional[int] = None,
    max_epochs_default: Optional[int] = None,
) -> str:
    """Compose the cache directory name per method-specific convention.

    PCA: `<stem>` (deterministic; no seed, no batch_key).
    scVI: `<stem>-s<seed>-e<epochs>-nl<n_latent>[-bk_<bk>]` — matches the
        existing `compare_vs_scvi.py` cache exactly.
    Harmony / scANVI / SysVI: `<stem>-s<seed>[-bk_<bk>][-e<eps>][-nl<nl>]`.
        The `e<eps>`/`nl<nl>` suffixes are appended only when the
        corresponding CLI override differs from the default, so default
        runs collide cleanly across reruns and overridden runs cache
        separately.
    """
    if method == "pca":
        return adata_stem
    parts = [adata_stem, f"s{seed}"]
    if method == "scvi":
        # Match existing compare_vs_scvi cache shape exactly so the
        # runner can read those latents with no migration.
        parts.append(f"e{max_epochs if max_epochs is not None else DEFAULT_MAX_EPOCHS}")
        parts.append(f"nl{n_latent if n_latent is not None else DEFAULT_N_LATENT}")
        if batch_key is not None and batch_key != "species":
            parts.append(f"bk_{batch_key}")
    else:
        if batch_key is not None and batch_key != "species":
            parts.append(f"bk_{batch_key}")
        if n_latent is not None and n_latent_default is not None and n_latent != n_latent_default:
            parts.append(f"nl{n_latent}")
        if (
            max_epochs is not None
            and max_epochs_default is not None
            and max_epochs != max_epochs_default
        ):
            parts.append(f"e{max_epochs}")
    return "-".join(parts)


def _save_latent_and_config(
    z: np.ndarray,
    adata,
    cache_dir: Path,
    config: Dict[str, Any],
) -> Path:
    """Persist (z_mean.npy, config.json) with shape/dtype assertions.

    Returns the absolute path of `z_mean.npy`. Float32 to halve disk
    use without precision loss for downstream metrics.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_obs = int(adata.n_obs)
    if z.shape[0] != n_obs:
        raise ValueError(
            f"Latent has shape {z.shape} but adata has {n_obs} cells. "
            "A row reordering is silent corruption — refusing to save."
        )
    if not np.isfinite(z).all():
        raise ValueError("Latent contains NaN/Inf — refusing to save.")
    z = np.asarray(z, dtype=np.float32)
    z_path = cache_dir / "z_mean.npy"
    np.save(z_path, z)
    config = dict(config)  # don't mutate caller's dict
    config["latent_shape"] = list(z.shape)
    config["latent_dtype"] = str(z.dtype)
    (cache_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    return z_path


def _base_config(
    *,
    adata_path: Path,
    adata,
    method: str,
    seed: int,
    species_key: Optional[str],
    batch_key: Optional[str],
    labels_key: Optional[str],
    n_latent: Optional[int],
    n_hidden: Optional[int],
    max_epochs: Optional[int],
    max_epochs_override: bool,
    n_latent_override: bool,
    n_latent_skip_reason: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Common config.json fields. Method-specific keys go in `extra`."""
    cfg: Dict[str, Any] = {
        "adata": str(adata_path.resolve()),
        "adata_stem": adata_path.stem,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "method": method,
        "seed": seed,
        "species_key": species_key,
        "batch_key": batch_key,
        "labels_key": labels_key,
        "n_latent": n_latent,
        "n_latent_actual": n_latent,
        "n_latent_override": n_latent_override,
        **(
            {
                "n_latent_override_reason": "comparability_with_scVI_baseline",
            }
            if n_latent_override
            else {
                "n_latent_skip_reason": n_latent_skip_reason or "method_default",
            }
        ),
        "n_hidden": n_hidden,
        "max_epochs": max_epochs,
        "max_epochs_override": max_epochs_override,
        **(
            {"max_epochs_override_reason": "fairness_cap_vs_chimera"}
            if max_epochs_override
            else {}
        ),
        "device": DEVICE,
        "torch_num_threads": TORCH_NUM_THREADS,
        "torch_num_interop_threads": TORCH_NUM_INTEROP_THREADS,
        "git_sha": _git_sha(),
        "pkg_versions": _pkg_versions(method),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        cfg.update(extra)
    return cfg


# ----------------------------------------------------------------------
# Method runners — each returns (z_mean.npy path, cache dir path).
# ----------------------------------------------------------------------
def run_pca(
    adata_path: Path,
    *,
    n_latent: int = DEFAULT_N_LATENT,
    force: bool = False,
    **_unused: Any,
) -> Tuple[Path, Path]:
    """Log-norm + PCA on the full gene set, no HVG, no scaling.

    Reference: scanpy PCA tutorial.
    No integration — the no-integration reference for the metrics step.
    Deterministic given the data, so the cache dir omits the seed.
    """
    cache_dir = RUNS_ROOT / "baselines_pca" / adata_path.stem
    z_path = cache_dir / "z_mean.npy"
    if z_path.exists() and not force:
        logger.info("PCA: cache hit at %s", z_path)
        return z_path, cache_dir

    import anndata as ad
    import scanpy as sc

    adata = ad.read_h5ad(adata_path)
    a = adata.copy()  # never mutate input — preprocessing is method-internal
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    # No HVG. No sc.pp.scale. Chimera doesn't HVG, so PCA doesn't either.
    sc.tl.pca(a, n_comps=n_latent)
    z = np.asarray(a.obsm["X_pca"])
    if z.shape != (a.n_obs, n_latent):
        raise ValueError(f"PCA produced shape {z.shape}; expected ({a.n_obs}, {n_latent}).")

    cfg = _base_config(
        adata_path=adata_path,
        adata=adata,
        method="pca",
        seed=0,  # PCA is deterministic; "seed" is recorded as 0 for schema parity
        species_key=None,
        batch_key=None,
        labels_key=None,
        n_latent=n_latent,
        n_hidden=None,
        max_epochs=None,
        max_epochs_override=False,
        n_latent_override=(n_latent == DEFAULT_N_LATENT),
        extra={
            "preprocessing": ["normalize_total(target_sum=1e4)", "log1p", f"pca(n_comps={n_latent})"],
            "hvg": None,
            "scale": None,
            "deterministic_by_data": True,
            "canonical_reference": (
                "scanpy PCA tutorial — sc.pp.normalize_total + sc.pp.log1p + "
                "sc.tl.pca(n_comps=...). No HVG / no scale per fair-playing-field rule."
            ),
        },
    )
    return _save_latent_and_config(z, adata, cache_dir, cfg), cache_dir


def run_harmony(
    adata_path: Path,
    *,
    batch_key: str,
    n_latent: int = DEFAULT_N_LATENT,
    seed: int = 42,
    force: bool = False,
    **_unused: Any,
) -> Tuple[Path, Path]:
    """log-norm + PCA + harmonypy.run_harmony.

    Reference: harmonypy v2.0.0 (slowkow/harmonypy). API:
        out = hm.run_harmony(data_mat, meta_data, vars_use=[batch_key], ...)
        out.Z_corr  # shape (n_cells, n_components) in v2.0.0+
                   # (older releases stored as (n_components, n_cells); we
                   #  defensively check shape and transpose if needed.)
    All theta/lamb/sigma/nclust at library defaults — fair-playing-field rule.
    """
    suffix = _build_cache_dir_name(
        adata_path.stem, "harmony", seed,
        batch_key=batch_key, n_latent=n_latent, n_latent_default=DEFAULT_N_LATENT,
    )
    cache_dir = RUNS_ROOT / "baselines_harmony" / suffix
    z_path = cache_dir / "z_mean.npy"
    if z_path.exists() and not force:
        logger.info("Harmony: cache hit at %s", z_path)
        return z_path, cache_dir

    import anndata as ad
    import harmonypy as hm
    import scanpy as sc

    _set_seed(seed)
    adata = ad.read_h5ad(adata_path)
    _assert_obs_keys(adata, batch_key)
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.tl.pca(a, n_comps=n_latent)
    pcs = np.asarray(a.obsm["X_pca"])
    obs_df = a.obs.copy()
    out = hm.run_harmony(
        pcs, obs_df, vars_use=[batch_key], random_state=seed
    )
    z = np.asarray(out.Z_corr)
    # Defend against API churn: v2.0.0 returns (n_cells, n_components);
    # older v0.x returned (n_components, n_cells). If the shape disagrees
    # with the input, transpose.
    if z.shape == (n_latent, a.n_obs):
        logger.warning(
            "harmonypy returned Z_corr in (n_components, n_cells) shape; "
            "transposing for downstream consumption."
        )
        z = z.T
    if z.shape != (a.n_obs, n_latent):
        raise ValueError(
            f"Harmony Z_corr shape {z.shape} ≠ ({a.n_obs}, {n_latent}). "
            "Refusing to save a mis-shaped latent."
        )

    cfg = _base_config(
        adata_path=adata_path,
        adata=adata,
        method="harmony",
        seed=seed,
        species_key=None,
        batch_key=batch_key,
        labels_key=None,
        n_latent=n_latent,
        n_hidden=None,
        max_epochs=None,
        max_epochs_override=False,
        n_latent_override=(n_latent == DEFAULT_N_LATENT),
        extra={
            "preprocessing": ["normalize_total(target_sum=1e4)", "log1p", f"pca(n_comps={n_latent})"],
            "hvg": None,
            "scale": None,
            "vars_use": [batch_key],
            "harmony_defaults": "theta/lamb/sigma/nclust at harmonypy defaults",
            "harmonypy_random_state": seed,
            "canonical_reference": (
                "harmonypy v2.0.0 (slowkow/harmonypy) — run_harmony(pcs, "
                "obs_df, vars_use=[batch_key]). No HVG / no scale per "
                "fair-playing-field rule."
            ),
        },
    )
    return _save_latent_and_config(z, adata, cache_dir, cfg), cache_dir


def run_scvi(
    adata_path: Path,
    *,
    batch_key: str,
    seed: int = 42,
    n_latent: int = DEFAULT_N_LATENT,
    n_hidden: int = DEFAULT_N_HIDDEN,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    force: bool = False,
    **_unused: Any,
) -> Tuple[Path, Path]:
    """Vanilla scVI baseline.

    Reuses `compare_vs_scvi.get_or_train_scvi` (single source of truth)
    so a fresh `--method scvi` run produces a byte-identical latent on
    the same (adata, seed, batch_key, defaults) tuple. The existing
    `runs/baselines_scvi/<stem>-s<seed>-e<epochs>-nl<n_latent>[-bk_<bk>]`
    cache is therefore reused without retraining.

    Per the prompt's exception note: scVI keeps `max_epochs=100,
    n_latent=10, n_hidden=128, batch_key=batch (or sample_batch)` to
    stay consistent with what `compare_vs_scvi.py` produces (the Chimera
    iteration loop's E8 gates are calibrated against this).
    """
    # Mirror the existing cache directory naming exactly.
    bk_suffix = f"-bk_{batch_key}" if batch_key != "species" else ""
    cache_dir = (
        RUNS_ROOT
        / "baselines_scvi"
        / f"{adata_path.stem}-s{seed}-e{max_epochs}-nl{n_latent}{bk_suffix}"
    )
    z_path = cache_dir / "z_mean.npy"
    if z_path.exists() and not force:
        logger.info("scVI: cache hit at %s", z_path)
        # Augment config.json with the runner-side fields if missing,
        # but preserve existing fields (the original training metadata).
        _augment_existing_config(cache_dir, adata_path, "scvi", seed,
                                 batch_key=batch_key, n_latent=n_latent,
                                 n_hidden=n_hidden, max_epochs=max_epochs)
        return z_path, cache_dir

    # Cache miss → train fresh on CPU. Import the shared helper from
    # the sibling script via sys.path injection — `chimera/training/`
    # is a script directory, not a Python package, so the standard
    # `from chimera.training.compare_vs_scvi import ...` form does not
    # work.
    _TRAINING_DIR = str(Path(__file__).resolve().parent)
    if _TRAINING_DIR not in sys.path:
        sys.path.insert(0, _TRAINING_DIR)
    from compare_vs_scvi import get_or_train_scvi  # type: ignore[import-not-found]

    if force and z_path.exists():
        z_path.unlink()
    z = get_or_train_scvi(
        adata_path=adata_path,
        seed=seed,
        max_epochs=max_epochs,
        n_latent=n_latent,
        n_hidden=n_hidden,
        batch_key=batch_key,
        accelerator=DEVICE,
    )
    # `get_or_train_scvi` writes a minimal config.json + z_mean.npy.
    # Replace the config with the runner's richer schema so downstream
    # tools see the same shape across all methods.
    import anndata as ad

    adata = ad.read_h5ad(adata_path, backed="r")
    cfg = _base_config(
        adata_path=adata_path,
        adata=adata,
        method="scvi",
        seed=seed,
        species_key=None,
        batch_key=batch_key,
        labels_key=None,
        n_latent=n_latent,
        n_hidden=n_hidden,
        max_epochs=max_epochs,
        max_epochs_override=(max_epochs != DEFAULT_MAX_EPOCHS or DEFAULT_MAX_EPOCHS != 100),
        n_latent_override=(n_latent == DEFAULT_N_LATENT),
        extra={
            "preprocessing": ["raw counts in .X"],
            "hvg": None,
            "scale": None,
            "early_stopping": True,
            "accelerator": DEVICE,
            "canonical_reference": (
                "scvi-tools 1.4.x SCVI quick-start; this is the same baseline "
                "compare_vs_scvi.py:get_or_train_scvi already trains. n_hidden=128 "
                "kept for consistency with existing E8 gates calibration."
            ),
        },
    )
    cfg["latent_shape"] = list(z.shape)
    cfg["latent_dtype"] = str(np.asarray(z).dtype)
    (cache_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    return z_path, cache_dir


def _augment_existing_config(
    cache_dir: Path,
    adata_path: Path,
    method: str,
    seed: int,
    **fields: Any,
) -> None:
    """Add runner-side fields to a pre-existing minimal cache config.

    Existing scVI cache configs (written by compare_vs_scvi.py before
    this runner existed) lack `device`, `torch_num_threads`, etc. We
    merge the runner's schema in non-destructively so a downstream
    metrics step sees a uniform field set across all methods.

    The existing fields are preserved verbatim — they are the truth
    about how the original training was wired. We only ADD the
    missing schema keys, with `cache_provenance="legacy_scvi"` to
    flag that the device/threads info reflects the runner environment,
    not the original training.
    """
    cfg_path = cache_dir / "config.json"
    if not cfg_path.exists():
        return
    try:
        existing = json.loads(cfg_path.read_text())
    except Exception:
        return
    if "cache_provenance" in existing:
        return  # already augmented
    additions: Dict[str, Any] = {
        "cache_provenance": "legacy_compare_vs_scvi" if "device" not in existing else "current",
        "git_sha": existing.get("git_sha", _git_sha()),
        "pkg_versions": existing.get("pkg_versions", _pkg_versions(method)),
        "timestamp_runner_view": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # Only add device/threads if the existing config doesn't have them
    # (don't overwrite genuine training-time records).
    additions.setdefault("device", existing.get("device", DEVICE))
    additions.setdefault(
        "torch_num_threads", existing.get("torch_num_threads", TORCH_NUM_THREADS)
    )
    merged = {**additions, **existing}  # existing wins on overlap
    merged.update(
        {
            k: v
            for k, v in additions.items()
            if k not in existing
        }
    )
    cfg_path.write_text(json.dumps(merged, indent=2, default=str))


def run_scanvi(
    adata_path: Path,
    *,
    batch_key: str,
    labels_key: Optional[str] = None,
    seed: int = 42,
    n_latent: int = DEFAULT_N_LATENT,
    max_epochs: int = DEFAULT_SCVI_PRETRAIN_EPOCHS,
    scanvi_max_epochs: int = DEFAULT_SCANVI_HEAD_EPOCHS,
    force: bool = False,
    **_unused: Any,
) -> Tuple[Path, Path]:
    """scvi-tools two-step recipe: SCVI pretrain → SCANVI.from_scvi_model → SCANVI.train.

    Reference: scvi-tools harmonization tutorial. Tutorial uses
    `unlabeled_category="Unknown"` (we follow this — the prompt's `"UNK"`
    is overridden in favor of the canonical reference, per the prompt's
    own "trust the canonical reference, surface the conflict in a comment"
    rule). All cells in our datasets are labeled, so the placeholder is
    a no-op category.

    Cap (Rule B): pretrain ≤ 100, head ≤ 30; sum ≤ 150. Defaults are
    100 + 20 = 120, comfortably under.
    """
    suffix = _build_cache_dir_name(
        adata_path.stem, "scanvi", seed,
        batch_key=batch_key, n_latent=n_latent, n_latent_default=DEFAULT_N_LATENT,
        max_epochs=max_epochs + scanvi_max_epochs,
        max_epochs_default=DEFAULT_SCVI_PRETRAIN_EPOCHS + DEFAULT_SCANVI_HEAD_EPOCHS,
    )
    cache_dir = RUNS_ROOT / "baselines_scanvi" / suffix
    z_path = cache_dir / "z_mean.npy"
    if z_path.exists() and not force:
        logger.info("scANVI: cache hit at %s", z_path)
        return z_path, cache_dir

    import anndata as ad
    import scvi

    if max_epochs + scanvi_max_epochs > 150:
        raise ValueError(
            f"scANVI cap violated: pretrain {max_epochs} + head "
            f"{scanvi_max_epochs} = {max_epochs + scanvi_max_epochs} > 150."
        )

    _set_seed(seed)
    adata = ad.read_h5ad(adata_path)
    resolved_labels = _resolve_labels_key(adata, requested=labels_key)
    _assert_obs_keys(adata, batch_key, resolved_labels)
    _assert_raw_counts(adata)

    a = adata.copy()  # scvi-tools mutates .obs (adds _scvi_batch / _scvi_labels)
    scvi.model.SCVI.setup_anndata(a, batch_key=batch_key, labels_key=resolved_labels)
    ref = scvi.model.SCVI(a, n_latent=n_latent)
    t0 = time.time()
    ref.train(
        max_epochs=max_epochs,
        accelerator=DEVICE,
        early_stopping=True,
    )
    t_scvi = time.time() - t0
    model = scvi.model.SCANVI.from_scvi_model(
        ref,
        unlabeled_category=UNLABELED_CATEGORY,
        labels_key=resolved_labels,
    )
    t1 = time.time()
    model.train(
        max_epochs=scanvi_max_epochs,
        n_samples_per_label=100,
        accelerator=DEVICE,
    )
    t_scanvi = time.time() - t1
    z = np.asarray(model.get_latent_representation())

    if z.shape != (a.n_obs, n_latent):
        raise ValueError(
            f"scANVI latent shape {z.shape} ≠ ({a.n_obs}, {n_latent})."
        )

    cfg = _base_config(
        adata_path=adata_path,
        adata=adata,
        method="scanvi",
        seed=seed,
        species_key=None,
        batch_key=batch_key,
        labels_key=resolved_labels,
        n_latent=n_latent,
        n_hidden=None,
        max_epochs=max_epochs + scanvi_max_epochs,
        max_epochs_override=True,
        n_latent_override=(n_latent == DEFAULT_N_LATENT),
        extra={
            "preprocessing": ["raw counts in .X"],
            "hvg": None,
            "scale": None,
            "max_epochs_scvi_pretrain": max_epochs,
            "max_epochs_scanvi_head": scanvi_max_epochs,
            "max_epochs_total": max_epochs + scanvi_max_epochs,
            "max_epochs_total_le_150": True,
            "n_samples_per_label": 100,
            "early_stopping": True,
            "unlabeled_category": UNLABELED_CATEGORY,
            "accelerator": DEVICE,
            "wall_time_pretrain_s": t_scvi,
            "wall_time_head_s": t_scanvi,
            "wall_time_total_s": t_scvi + t_scanvi,
            "canonical_reference": (
                "scvi-tools harmonization tutorial — "
                "https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/harmonization.html. "
                "Two-step recipe; unlabeled_category='Unknown' per tutorial."
            ),
        },
    )
    return _save_latent_and_config(z, a, cache_dir, cfg), cache_dir


def run_sysvi(
    adata_path: Path,
    *,
    batch_key: str,
    species_key: str,
    seed: int = 42,
    n_latent: int = DEFAULT_N_LATENT,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    force: bool = False,
    **_unused: Any,
) -> Tuple[Path, Path]:
    """scvi.external.SysVI — cross-species VampPrior cVAE with cycle loss.

    Reference: scvi-tools sysVI tutorial
    (https://docs.scvi-tools.org/en/1.3.3/tutorials/notebooks/scrna/sysVI.html).
    SysVI now lives in scvi-tools — the standalone
    theislab/cross_system_integration repo is unmaintained per the
    Theislab team.

    Routing (per prompt §Covariate routing):
      - SysVI's `batch_key` is the "system" to mix → species_key.
      - SysVI's `categorical_covariate_keys` is the within-system batch
        to correct → [batch_key], BUT only when batch_key actually
        carries within-species variation. Otherwise it would double-
        count the species effect (in pancreas where batch ≡ species
        1-to-1, except H+Mo, the column is redundant).

    SysVI's docstring requires log-normalized expression in `.X`.
    Preprocessing (method-essential, on a copy of the input):
      normalize_total(target_sum=1e4) + log1p. NO HVG, NO scaling
      (fair-playing-field rule).

    n_latent override: SysVI's library default is 15 (in SysVAE module).
    Forwarding `n_latent=10` through `**model_kwargs` is supported by
    the public constructor. Comparability with the rest of the matrix
    wins; record the override.
    """
    suffix = _build_cache_dir_name(
        adata_path.stem, "sysvi", seed,
        batch_key=batch_key, n_latent=n_latent, n_latent_default=DEFAULT_N_LATENT,
        max_epochs=max_epochs, max_epochs_default=DEFAULT_MAX_EPOCHS,
    )
    cache_dir = RUNS_ROOT / "baselines_sysvi" / suffix
    z_path = cache_dir / "z_mean.npy"
    if z_path.exists() and not force:
        logger.info("SysVI: cache hit at %s", z_path)
        return z_path, cache_dir

    if max_epochs > 150:
        raise ValueError(f"SysVI max_epochs={max_epochs} > 150 cap.")

    import anndata as ad
    import scanpy as sc
    from scvi.external import SysVI

    _set_seed(seed)
    adata = ad.read_h5ad(adata_path)
    _assert_obs_keys(adata, species_key, batch_key)

    has_within_species_batch = _detect_within_species_batch(adata, species_key, batch_key)

    a = adata.copy()
    # SysVI consumes log-normalized expression (per setup_anndata docstring).
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    setup_kwargs: Dict[str, Any] = {"batch_key": species_key}
    if has_within_species_batch:
        setup_kwargs["categorical_covariate_keys"] = [batch_key]
    SysVI.setup_anndata(a, **setup_kwargs)
    model = SysVI(a, n_latent=n_latent)
    t0 = time.time()
    model.train(
        max_epochs=max_epochs,
        accelerator=DEVICE,
        plan_kwargs={"z_distance_cycle_weight": 5},
    )
    t_train = time.time() - t0
    z = np.asarray(model.get_latent_representation(adata=a))

    if z.shape != (a.n_obs, n_latent):
        raise ValueError(f"SysVI latent shape {z.shape} ≠ ({a.n_obs}, {n_latent}).")

    cfg = _base_config(
        adata_path=adata_path,
        adata=adata,
        method="sysvi",
        seed=seed,
        species_key=species_key,
        batch_key=batch_key,
        labels_key=None,
        n_latent=n_latent,
        n_hidden=None,
        max_epochs=max_epochs,
        max_epochs_override=True,
        n_latent_override=(n_latent == DEFAULT_N_LATENT),
        extra={
            "preprocessing": ["normalize_total(target_sum=1e4)", "log1p"],
            "hvg": None,
            "scale": None,
            "sysvi_system_key": species_key,
            "sysvi_categorical_covariate_keys": (
                [batch_key] if has_within_species_batch else None
            ),
            "categorical_covariate_dropped_reason": (
                None
                if has_within_species_batch
                else "batch_key 1-to-1 with species_key — would double-count"
            ),
            "prior": "vamp",
            "n_prior_components": 5,
            "z_distance_cycle_weight": 5,
            "early_stopping": False,
            "accelerator": DEVICE,
            "wall_time_train_s": t_train,
            "canonical_reference": (
                "scvi-tools sysVI tutorial — "
                "https://docs.scvi-tools.org/en/1.3.3/tutorials/notebooks/scrna/sysVI.html. "
                "VampPrior + latent cycle-consistency. Now in scvi-tools as "
                "scvi.external.SysVI; the standalone "
                "theislab/cross_system_integration repo is unmaintained."
            ),
        },
    )
    return _save_latent_and_config(z, a, cache_dir, cfg), cache_dir


def run_camex(
    adata_path: Path, **_unused: Any
) -> Tuple[Path, Path]:
    raise NotImplementedError(
        "CAMEX baseline is not wired in this runner. Gating step: clone "
        "CAMEX from its source repository and document its preprocessing + "
        "training entry-point in `chimera-private/baselines_prompt.md` Tier-2 "
        "section. Until that is done, this stub deliberately raises so the "
        "matrix records `FAILED` for camex rather than silently skipping."
    )


def run_saturn(
    adata_path: Path, **_unused: Any
) -> Tuple[Path, Path]:
    raise NotImplementedError(
        "SATURN baseline is not wired in this runner. Gating step: install "
        "snap-stanford/SATURN and precompute ESM-2 protein embeddings per "
        "species. SATURN expects per-species adatas + species-specific protein "
        "embedding files, which is incompatible with our concatenated "
        "cross-species adata layout. Document the split + embedding pipeline "
        "in `chimera-private/baselines_prompt.md` Tier-2 before stubbing this "
        "out."
    )


METHOD_DISPATCH = {
    "pca": run_pca,
    "harmony": run_harmony,
    "scvi": run_scvi,
    "scanvi": run_scanvi,
    "sysvi": run_sysvi,
    "camex": run_camex,
    "saturn": run_saturn,
}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train (or load cached) one baseline integration "
        "method on one adata. Latents only.",
    )
    ap.add_argument("--adata", required=True, type=str)
    ap.add_argument(
        "--method",
        required=True,
        type=str,
        help=f"Comma-separated; choices: {','.join(KNOWN_METHODS)}",
    )
    ap.add_argument("--species-key", type=str, default="species")
    ap.add_argument("--batch-key", type=str, default=None,
                    help="Required for harmony/scvi/scanvi/sysvi.")
    ap.add_argument("--labels-key", type=str, default=None,
                    help="scANVI only; auto-resolves to cell_type / cell_ontology_class.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-latent", type=int, default=None,
                    help="Override; default = method library default (10 where applicable).")
    ap.add_argument("--n-hidden", type=int, default=None,
                    help="Override; default = scVI's 128, others use library default.")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help=f"Override; default = {DEFAULT_MAX_EPOCHS}. Cap is 150.")
    ap.add_argument("--scanvi-max-epochs", type=int, default=None,
                    help=f"Override; default = {DEFAULT_SCANVI_HEAD_EPOCHS}.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if cache exists.")
    return ap.parse_args(argv)


def _resolve_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Map CLI overrides to runner kwargs, applying Rule B caps."""
    n_latent = args.n_latent if args.n_latent is not None else DEFAULT_N_LATENT
    n_hidden = args.n_hidden if args.n_hidden is not None else DEFAULT_N_HIDDEN
    max_epochs = args.max_epochs if args.max_epochs is not None else DEFAULT_MAX_EPOCHS
    scanvi_max_epochs = (
        args.scanvi_max_epochs
        if args.scanvi_max_epochs is not None
        else DEFAULT_SCANVI_HEAD_EPOCHS
    )
    if max_epochs > 150:
        raise ValueError(f"--max-epochs {max_epochs} exceeds Rule B cap of 150.")
    return {
        "n_latent": n_latent,
        "n_hidden": n_hidden,
        "max_epochs": max_epochs,
        "scanvi_max_epochs": scanvi_max_epochs,
    }


def _run_one_method(
    method: str, adata_path: Path, args: argparse.Namespace, overrides: Dict[str, Any]
) -> Tuple[Path, Path]:
    """Dispatch one method invocation. Returns (latent path, cache dir)."""
    if method not in METHOD_DISPATCH:
        raise ValueError(f"Unknown method '{method}'. Choices: {KNOWN_METHODS}")
    fn = METHOD_DISPATCH[method]
    common_kw = dict(
        seed=args.seed,
        force=args.force,
    )
    if method == "pca":
        return fn(
            adata_path,
            n_latent=overrides["n_latent"],
            force=args.force,
        )
    if method == "harmony":
        if args.batch_key is None:
            raise ValueError("--batch-key is required for harmony.")
        return fn(
            adata_path,
            batch_key=args.batch_key,
            n_latent=overrides["n_latent"],
            **common_kw,
        )
    if method == "scvi":
        if args.batch_key is None:
            raise ValueError("--batch-key is required for scvi.")
        return fn(
            adata_path,
            batch_key=args.batch_key,
            n_latent=overrides["n_latent"],
            n_hidden=overrides["n_hidden"],
            max_epochs=overrides["max_epochs"],
            **common_kw,
        )
    if method == "scanvi":
        if args.batch_key is None:
            raise ValueError("--batch-key is required for scanvi.")
        return fn(
            adata_path,
            batch_key=args.batch_key,
            labels_key=args.labels_key,
            n_latent=overrides["n_latent"],
            max_epochs=overrides["max_epochs"],
            scanvi_max_epochs=overrides["scanvi_max_epochs"],
            **common_kw,
        )
    if method == "sysvi":
        if args.batch_key is None:
            raise ValueError("--batch-key is required for sysvi.")
        return fn(
            adata_path,
            batch_key=args.batch_key,
            species_key=args.species_key,
            n_latent=overrides["n_latent"],
            max_epochs=overrides["max_epochs"],
            **common_kw,
        )
    # Tier-2 stubs.
    return fn(adata_path)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    methods = [m.strip() for m in args.method.split(",") if m.strip()]
    unknown = [m for m in methods if m not in KNOWN_METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s) {unknown}. Choices: {KNOWN_METHODS}")

    # Resolve adata path: accept absolute or repo-relative.
    if os.path.isabs(args.adata):
        adata_path = Path(args.adata)
    else:
        adata_path = (REPO_ROOT / args.adata).resolve()
    if not adata_path.exists():
        raise FileNotFoundError(f"adata not found: {adata_path}")

    overrides = _resolve_overrides(args)
    logger.info(
        "Runner startup: device=%s, torch.get_num_threads()=%d, "
        "PYTHONHASHSEED=%s",
        DEVICE,
        _torch.get_num_threads(),
        _os.environ.get("PYTHONHASHSEED"),
    )
    logger.info("Methods: %s", methods)
    logger.info("Adata: %s (resolved)", adata_path)

    failures = []
    for method in methods:
        try:
            latent_path, cache_dir = _run_one_method(method, adata_path, args, overrides)
            print(f"LATENT_PATH={latent_path}")
            print(f"CACHE_DIR={cache_dir}")
            print(f"METHOD_OK={method}")
            sys.stdout.flush()
        except NotImplementedError as e:
            logger.error("Method %s is not implemented: %s", method, e)
            print(f"METHOD_FAILED={method}: NotImplementedError: {e}")
            sys.stdout.flush()
            failures.append((method, "NotImplementedError", str(e)))
        except Exception as e:  # noqa: BLE001 — surface every runner failure
            logger.exception("Method %s failed", method)
            print(f"METHOD_FAILED={method}: {type(e).__name__}: {e}")
            sys.stdout.flush()
            failures.append((method, type(e).__name__, str(e)))

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
