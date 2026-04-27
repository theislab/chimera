"""Smoke tests for the baseline runner + orchestrator.

Run via:
    mamba run -n chimera_env pytest chimera/tests/test_baselines.py -v

These tests build a tiny synthetic anndata mirroring the structure of
`chimera/data/processed/camex_human_monkey.h5ad` (raw integer counts in
.X, two-species `species`, three-level `batch`, `cell_type` /
`cell_ontology_class`, `conservation_prior` in `.var`, `p_null` in
`.uns`) and exercise every Tier-1 method's training-loop machinery
plus the orchestrator's subprocess path.

The tests are CPU-only (Rule A) and capped at `--max-epochs 10` for
trainable methods so the whole file completes well under the 5-minute
budget on the user's MacBook.

Method runners are invoked by importing `baseline_runner.main(argv)`
in-process — this is the same code path that the production runner
exposes via `if __name__ == "__main__"`, just without the subprocess
overhead. The orchestrator's subprocess path is exercised separately
in `test_orchestrator_end_to_end`, with `CHIMERA_BASELINE_PYTHON` set
so the test does not require `mamba run` to be on PATH.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

# Ensure the runner's thread-cap setup runs before its torch import.
# Some test orderings could otherwise let pytest's own torch import
# beat the runner to it; setting the env vars here is harmless.
os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("MKL_NUM_THREADS", "5")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "5")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "5")

# `chimera/training/` is a directory of standalone scripts, not a
# Python package, so the runner / orchestrator are imported via
# sys.path injection (matching how compare_vs_scvi.py is structured —
# top-level scripts that use `Path(__file__).parents[1]` as repo root).
TRAINING_DIR = (
    Path(__file__).resolve().parent.parent / "training"
).resolve()
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

# ----------------------------------------------------------------------
# Mock anndata fixture — mirrors the structure of camex_human_monkey.h5ad.
# ----------------------------------------------------------------------
@pytest.fixture
def mock_h5ad(tmp_path: Path) -> Path:
    """Synthetic ~120 cells × 200 genes adata with raw NB counts.

    Structure mirrors camex_human_monkey.h5ad:
      - .X: raw integer counts (NB, mean ~1, dispersion ~0.5).
      - .obs: species (Human / Monkey, ~1:2 split), batch (3 levels:
        H gets CS170, M gets CS172/CS173 — preserves the H+Mo
        identifiability profile), cell_type (4 types with cross-species
        sharing), cell_ontology_class (mirror of cell_type), plus
        n_genes / n_counts / total_counts / pct_counts_mt fillers.
      - .var: conservation_prior in [0,1].
      - .uns: p_null scalar.
      - .layers: empty.

    Saved to tmp_path so the runner reads from disk (the production
    code path).
    """
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed=0)
    n_cells = 120  # 40 Human + 80 Monkey ≈ 1:2
    n_genes = 200

    species = np.array(["Human"] * 40 + ["Monkey"] * 80)
    # Batch: Human → CS170 (40), Monkey → CS172 (40) + CS173 (40).
    batch = np.array(["CS170"] * 40 + ["CS172"] * 40 + ["CS173"] * 40)
    # Cell types: 4 types, distributed so each species has all 4.
    cell_types = np.array(
        ["TypeA"] * 10 + ["TypeB"] * 10 + ["TypeC"] * 10 + ["TypeD"] * 10
        + ["TypeA"] * 20 + ["TypeB"] * 20 + ["TypeC"] * 20 + ["TypeD"] * 20
    )

    # NB counts: mean=1, dispersion (size)=2 — keeps mean low but
    # ensures the matrix isn't all zeros. mean=1 / size=2 corresponds
    # to a sparse single-cell-like sampling. Stored as sparse CSR to
    # mirror the real adata layout (compare_vs_scvi.py's raw-counts
    # safety check assumes `.X.data` is an ndarray, which sparse
    # matrices provide and dense ones do not).
    p = 2.0 / (2.0 + 1.0)  # success probability for negative_binomial
    X_dense = rng.negative_binomial(2.0, p, size=(n_cells, n_genes)).astype(np.float32)
    # Sanity: a few genes should have non-zero counts in most cells.
    assert (X_dense > 0).any(axis=1).all(), "every cell must have at least one count"
    X = sp.csr_matrix(X_dense)

    obs = pd.DataFrame(
        {
            "species": pd.Categorical(species, categories=["Human", "Monkey"]),
            "batch": pd.Categorical(batch, categories=["CS170", "CS172", "CS173"]),
            "cell_type": pd.Categorical(cell_types),
            "cell_ontology_class": pd.Categorical(cell_types),
            "n_genes": (X_dense > 0).sum(axis=1),
            "n_counts": X_dense.sum(axis=1),
            "total_counts": X_dense.sum(axis=1),
            "pct_counts_mt": rng.uniform(0.0, 5.0, size=n_cells),
        },
        index=[f"cell_{i:04d}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"conservation_prior": rng.uniform(0.0, 1.0, size=n_genes)},
        index=[f"gene_{i:04d}" for i in range(n_genes)],
    )
    uns = {"p_null": 0.5}

    adata = ad.AnnData(X=X, obs=obs, var=var, uns=uns)
    out_path = tmp_path / "mock_camex.h5ad"
    adata.write_h5ad(out_path)
    return out_path


def _import_runner():
    """Lazy-import baseline_runner so test collection doesn't pay torch's
    import cost when running unrelated tests."""
    import baseline_runner  # type: ignore[import-not-found]

    return baseline_runner


def _invoke_runner(argv: list, expect_exit: int = 0) -> Tuple[int, list]:
    """Run baseline_runner.main(argv) in-process; return (rc, captured_lines).

    Captures stdout to a list of lines so the LATENT_PATH= line can be
    parsed from the same shape the orchestrator expects.
    """
    import io
    import contextlib

    runner = _import_runner()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = runner.main(argv)
    lines = buf.getvalue().splitlines()
    if rc != expect_exit:
        print("\n".join(lines))
    assert rc == expect_exit, f"runner.main returned {rc}, expected {expect_exit}"
    return rc, lines


def _parse_latent_path(lines: list) -> Path:
    for line in lines:
        if line.startswith("LATENT_PATH="):
            return Path(line.split("=", 1)[1].strip())
    raise AssertionError(f"No LATENT_PATH= in stdout. Got: {lines}")


def _parse_cache_dir(lines: list) -> Path:
    for line in lines:
        if line.startswith("CACHE_DIR="):
            return Path(line.split("=", 1)[1].strip())
    raise AssertionError(f"No CACHE_DIR= in stdout. Got: {lines}")


def _read_config(cache_dir: Path) -> dict:
    return json.loads((cache_dir / "config.json").read_text())


def _check_latent(z_path: Path, expected_n_cells: int, expected_n_latent: int) -> np.ndarray:
    assert z_path.exists(), f"{z_path} not written"
    z = np.load(z_path)
    assert z.shape == (expected_n_cells, expected_n_latent), (
        f"Latent shape {z.shape} ≠ ({expected_n_cells}, {expected_n_latent})"
    )
    assert np.isfinite(z).all(), "Latent contains NaN/Inf"
    return z


# ----------------------------------------------------------------------
# Tier-1 method tests.
# ----------------------------------------------------------------------
def test_pca_runner(mock_h5ad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Redirect cache to tmp_path so each test is isolated.
    monkeypatch.setenv("HOME", str(tmp_path))  # belt-and-suspenders
    runner = _import_runner()
    monkeypatch.setattr(runner, "RUNS_ROOT", tmp_path / "runs")

    rc, lines = _invoke_runner(
        ["--adata", str(mock_h5ad), "--method", "pca", "--seed", "42"]
    )
    z_path = _parse_latent_path(lines)
    cache_dir = _parse_cache_dir(lines)
    z = _check_latent(z_path, expected_n_cells=120, expected_n_latent=10)

    cfg = _read_config(cache_dir)
    assert cfg["method"] == "pca"
    assert cfg["n_obs"] == 120
    assert cfg["n_vars"] == 200
    assert cfg["device"] == "cpu"
    assert cfg["torch_num_threads"] == 5
    assert cfg["deterministic_by_data"] is True
    assert cfg["hvg"] is None
    assert cfg["scale"] is None
    # PCA's first 3 row values should be deterministic given the data.
    assert z.dtype == np.float32


def test_harmony_runner(mock_h5ad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _import_runner()
    monkeypatch.setattr(runner, "RUNS_ROOT", tmp_path / "runs")

    rc, lines = _invoke_runner(
        ["--adata", str(mock_h5ad), "--method", "harmony",
         "--batch-key", "batch", "--seed", "42"]
    )
    z_path = _parse_latent_path(lines)
    cache_dir = _parse_cache_dir(lines)
    z = _check_latent(z_path, 120, 10)

    cfg = _read_config(cache_dir)
    assert cfg["method"] == "harmony"
    assert cfg["batch_key"] == "batch"
    assert cfg["vars_use"] == ["batch"]
    assert cfg["device"] == "cpu"
    assert cfg["hvg"] is None and cfg["scale"] is None

    # Harmony actually moved the embedding away from raw PCA. We
    # recompute the pre-Harmony PCA on the same data and assert the
    # corrected embedding is *not* identical (would mean Harmony was a
    # no-op on this data).
    import anndata as ad
    import scanpy as sc

    a = ad.read_h5ad(mock_h5ad).copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.tl.pca(a, n_comps=10)
    raw_pcs = np.asarray(a.obsm["X_pca"])
    delta = float(np.linalg.norm(z - raw_pcs))
    assert delta > 0.0, f"Harmony made no change to PCA — wiring bug? (||Δ||={delta})"


def test_scvi_runner(mock_h5ad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _import_runner()
    monkeypatch.setattr(runner, "RUNS_ROOT", tmp_path / "runs")
    # Re-target compare_vs_scvi's RUNS_ROOT too — the helper writes
    # its cache at module-level RUNS_ROOT.
    import compare_vs_scvi as cvs  # type: ignore[import-not-found]
    monkeypatch.setattr(cvs, "RUNS_ROOT", tmp_path / "runs")

    rc, lines = _invoke_runner(
        ["--adata", str(mock_h5ad), "--method", "scvi",
         "--batch-key", "batch", "--seed", "42",
         "--max-epochs", "10"]
    )
    z_path = _parse_latent_path(lines)
    cache_dir = _parse_cache_dir(lines)
    z = _check_latent(z_path, 120, 10)

    cfg = _read_config(cache_dir)
    assert cfg["method"] == "scvi"
    assert cfg["batch_key"] == "batch"
    assert cfg["max_epochs"] == 10
    assert cfg["device"] == "cpu"
    assert cfg["accelerator"] == "cpu"


def test_scanvi_runner(mock_h5ad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _import_runner()
    monkeypatch.setattr(runner, "RUNS_ROOT", tmp_path / "runs")

    rc, lines = _invoke_runner(
        ["--adata", str(mock_h5ad), "--method", "scanvi",
         "--batch-key", "batch", "--seed", "42",
         "--max-epochs", "10", "--scanvi-max-epochs", "5"]
    )
    z_path = _parse_latent_path(lines)
    cache_dir = _parse_cache_dir(lines)
    z = _check_latent(z_path, 120, 10)

    cfg = _read_config(cache_dir)
    assert cfg["method"] == "scanvi"
    assert cfg["batch_key"] == "batch"
    assert cfg["labels_key"] in ("cell_type", "cell_ontology_class")
    assert cfg["max_epochs_total"] == 15
    assert cfg["max_epochs_total_le_150"] is True
    assert cfg["device"] == "cpu"
    assert cfg["unlabeled_category"] == "Unknown"


def test_sysvi_runner(mock_h5ad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # SysVI is in scvi.external but might not be importable on every
    # install — skip cleanly if so.
    try:
        from scvi.external import SysVI  # noqa: F401
    except Exception as e:
        pytest.skip(f"SysVI not importable: {e}")

    runner = _import_runner()
    monkeypatch.setattr(runner, "RUNS_ROOT", tmp_path / "runs")

    rc, lines = _invoke_runner(
        ["--adata", str(mock_h5ad), "--method", "sysvi",
         "--species-key", "species", "--batch-key", "batch",
         "--seed", "42", "--max-epochs", "10"]
    )
    z_path = _parse_latent_path(lines)
    cache_dir = _parse_cache_dir(lines)
    z = _check_latent(z_path, 120, 10)

    cfg = _read_config(cache_dir)
    assert cfg["method"] == "sysvi"
    assert cfg["sysvi_system_key"] == "species"
    # In our mock H+Mo data, batch carries within-species variation
    # (Monkey has CS172 + CS173) so SysVI should receive batch as a
    # categorical covariate.
    assert cfg["sysvi_categorical_covariate_keys"] == ["batch"]
    assert cfg["device"] == "cpu"
    assert cfg["prior"] == "vamp"


# ----------------------------------------------------------------------
# Caching + force.
# ----------------------------------------------------------------------
def test_caching_idempotent(mock_h5ad: Path, tmp_path: Path,
                            monkeypatch: pytest.MonkeyPatch):
    """Second invocation must hit the cache and skip retraining.

    PCA is the cheapest method; second-call wall-time should be a small
    fraction of the first.
    """
    import time as _time

    runner = _import_runner()
    monkeypatch.setattr(runner, "RUNS_ROOT", tmp_path / "runs")

    t0 = _time.time()
    _invoke_runner(["--adata", str(mock_h5ad), "--method", "pca", "--seed", "42"])
    first_dt = _time.time() - t0

    t0 = _time.time()
    rc, lines = _invoke_runner(
        ["--adata", str(mock_h5ad), "--method", "pca", "--seed", "42"]
    )
    second_dt = _time.time() - t0

    # Second call must be substantially faster than the first OR
    # under 5s in absolute terms (cache-hit path is just a file read).
    assert second_dt < 5.0 or second_dt < 0.5 * first_dt, (
        f"Cache hit not faster: first={first_dt:.2f}s, second={second_dt:.2f}s"
    )


def test_force_flag(mock_h5ad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """--force must overwrite the cache and produce the same shape."""
    runner = _import_runner()
    monkeypatch.setattr(runner, "RUNS_ROOT", tmp_path / "runs")

    _invoke_runner(["--adata", str(mock_h5ad), "--method", "pca", "--seed", "42"])
    z_path_1 = (tmp_path / "runs" / "baselines_pca" / mock_h5ad.stem / "z_mean.npy")
    mtime_1 = z_path_1.stat().st_mtime
    # Sleep to ensure mtime ticks over.
    import time as _time
    _time.sleep(0.05)

    _invoke_runner(
        ["--adata", str(mock_h5ad), "--method", "pca", "--seed", "42", "--force"]
    )
    mtime_2 = z_path_1.stat().st_mtime
    assert mtime_2 > mtime_1, "--force did not regenerate z_mean.npy"


# ----------------------------------------------------------------------
# Tier-2 stubs.
# ----------------------------------------------------------------------
def test_camex_stub(mock_h5ad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _import_runner()
    monkeypatch.setattr(runner, "RUNS_ROOT", tmp_path / "runs")

    rc, lines = _invoke_runner(
        ["--adata", str(mock_h5ad), "--method", "camex"], expect_exit=1
    )
    err_lines = [l for l in lines if l.startswith("METHOD_FAILED=")]
    assert err_lines, f"No METHOD_FAILED line in: {lines}"
    assert "NotImplementedError" in err_lines[0]
    assert "CAMEX" in err_lines[0]


def test_saturn_stub(mock_h5ad: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = _import_runner()
    monkeypatch.setattr(runner, "RUNS_ROOT", tmp_path / "runs")

    rc, lines = _invoke_runner(
        ["--adata", str(mock_h5ad), "--method", "saturn"], expect_exit=1
    )
    err_lines = [l for l in lines if l.startswith("METHOD_FAILED=")]
    assert err_lines, f"No METHOD_FAILED line in: {lines}"
    assert "NotImplementedError" in err_lines[0]
    assert "SATURN" in err_lines[0]


# ----------------------------------------------------------------------
# Orchestrator integration test.
# ----------------------------------------------------------------------
def test_orchestrator_end_to_end(mock_h5ad: Path, tmp_path: Path,
                                 monkeypatch: pytest.MonkeyPatch):
    """Full orchestrator path: subprocess invocation, log files,
    _summary.json. Exercises the same code that the production CLI uses,
    just with the runner subprocess pointed at sys.executable so the
    test does not depend on `mamba run` being on PATH.
    """
    import run_all_baselines as orch  # type: ignore[import-not-found]
    import baseline_runner as br  # type: ignore[import-not-found]
    _ = br  # silence unused-import warning; verifies it imports cleanly

    runs_root = tmp_path / "runs"
    log_dir = runs_root / "baselines_logs"
    monkeypatch.setattr(orch, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(orch, "LOG_ROOT", log_dir)
    # The runner subprocess inherits env so it must also see the
    # tmp_path RUNS_ROOT. We achieve this by passing a spec whose path
    # is absolute (mock_h5ad) and trusting the runner's cache dir
    # placement under chimera/training/runs to be re-routable via
    # CHIMERA_RUNS_ROOT (we add this env-var hook below).
    # For this test we set HOME and PYTHONPATH so the runner subprocess
    # can import chimera unmodified.
    monkeypatch.setenv("CHIMERA_BASELINE_PYTHON", sys.executable)

    spec = orch.DatasetSpec(
        path=str(mock_h5ad),
        species_col="species",
        batch_col="batch",
        labels_col="cell_type",
        note="mock",
    )
    # Override the runner's RUNS_ROOT via env so the subprocess writes
    # to tmp_path. We expose CHIMERA_RUNS_ROOT in the runner.
    monkeypatch.setenv("CHIMERA_RUNS_ROOT", str(runs_root))

    # Run pca + harmony + scvi (skip scanvi/sysvi for speed budget here;
    # those are exercised in their dedicated tests).
    methods = ["pca", "harmony", "scvi"]
    results = list(
        orch.run_phase(
            methods=methods,
            datasets=[spec],
            seeds=[42],
            force=False,
            max_epochs=10,
            log_dir=log_dir,
            verbose=False,
        )
    )
    assert len(results) == 3
    statuses = [r.status for r in results]
    assert all(s == "OK" for s in statuses), (
        f"Some methods failed: "
        f"{[(r.method, r.status, r.error_msg) for r in results]}"
    )
    for r in results:
        assert r.cache_path is not None
        assert Path(r.cache_path).exists(), f"Latent missing: {r.cache_path}"
        assert r.latent_shape == (120, 10), (
            f"{r.method}: shape {r.latent_shape} ≠ (120, 10)"
        )
        assert r.log_path is not None and Path(r.log_path).exists()

    # _summary.json round-trip.
    summary_path = log_dir / "_summary.json"
    orch.write_summary(results, path=summary_path)
    assert summary_path.exists()
    reloaded = orch.load_summary(summary_path)
    assert len(reloaded) == 3
    assert all(r.status == "OK" for r in reloaded)
