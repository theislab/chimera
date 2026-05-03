"""Signature-based caching for notebook 10's heavy compute + figure renders.

Used by ``10_gin_attribution_protein_structure.ipynb`` to make warm reruns
lightning fast: every expensive step (captum IG, per-residue signals,
real-GIN-attention projection, per-gene 2D/3D figure rendering) is
guarded by a content fingerprint of its inputs.  When the fingerprint
matches the on-disk signature *and* the expected output files exist,
the step is skipped.

Two patterns:

- ``cache_pickle(name, sig_inputs, compute_fn, cache_dir)`` — caches
  Python objects (dicts of numpy arrays, etc.).  Returns the cached
  value or computes + saves.
- ``render_fresh(name, sig_inputs, output_paths, cache_dir)`` +
  ``stamp_render(name, sig_inputs, cache_dir)`` — guards a *render*
  loop where the artifacts live as PDF / HTML / PNG files.  Use the
  pair: skip the loop if ``render_fresh`` returns True; call
  ``stamp_render`` after the loop completes.

The fingerprint is stable across runs as long as the ndarray content,
dict keys, and primitive values are unchanged.  Editing anything in
``sig_inputs`` invalidates the cache automatically.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


__all__ = ["fingerprint", "cache_pickle", "render_fresh", "stamp_render"]


def fingerprint(obj) -> str:
    """Stable 16-char content hash of a (possibly nested) Python object."""
    h = hashlib.sha256()

    def _emit(x):
        if isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            h.update(b"nda"); h.update(str(a.shape).encode())
            h.update(str(a.dtype).encode())
            if a.dtype == object:
                # Object arrays serialize pointer bytes via tobytes() — that's
                # not stable across runs.  Walk elements explicitly so the
                # underlying string / scalar contents drive the hash.
                for v in a.ravel():
                    _emit(v)
            else:
                h.update(a.tobytes())
        elif isinstance(x, (list, tuple)):
            h.update(f"seq{len(x)}".encode())
            for v in x:
                _emit(v)
        elif isinstance(x, dict):
            h.update(f"map{len(x)}".encode())
            for k in sorted(x, key=lambda kk: repr(kk)):
                h.update(repr(k).encode()); _emit(x[k])
        elif isinstance(x, (str, bytes)):
            h.update(b"str"); h.update(x.encode() if isinstance(x, str) else x)
        elif isinstance(x, (int, float, bool, type(None))):
            h.update(repr(x).encode())
        else:
            h.update(repr(x).encode())

    _emit(obj)
    return h.hexdigest()[:16]


def cache_pickle(
    name: str,
    sig_inputs,
    compute_fn: Callable,
    cache_dir: Path,
    *,
    force: bool = False,
    label: str = "",
):
    """Load ``<cache_dir>/<name>.pkl`` if its signature matches; else compute + save.

    The companion ``<name>.sig`` file holds the content fingerprint of
    ``sig_inputs`` from the run that built the pickle.  A mismatch — caused
    by editing the inputs, swapping the model checkpoint, or extending the
    gene set — forces recompute automatically.
    """
    cache_dir = Path(cache_dir)
    pkl_path = cache_dir / f"{name}.pkl"
    sig_path = cache_dir / f"{name}.sig"
    sig = fingerprint(sig_inputs)
    suffix = f"  · {label}" if label else ""
    if (not force and pkl_path.exists() and sig_path.exists()
            and sig_path.read_text().strip() == sig):
        with pkl_path.open("rb") as f:
            obj = pickle.load(f)
        kb = pkl_path.stat().st_size / 1024
        print(f"  [{name}] cache hit  ({kb:.0f} KB){suffix}")
        return obj
    print(f"  [{name}] cache miss — computing…{suffix}")
    obj = compute_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as f:
        pickle.dump(obj, f)
    sig_path.write_text(sig)
    kb = pkl_path.stat().st_size / 1024
    print(f"  [{name}] saved → {pkl_path.name}  ({kb:.0f} KB){suffix}")
    return obj


def render_fresh(
    name: str,
    sig_inputs,
    output_paths: Iterable[Path],
    cache_dir: Path,
) -> bool:
    """True iff every output_path exists and the cached signature matches."""
    sig_path = Path(cache_dir) / f"{name}.render_sig"
    if not sig_path.exists():
        return False
    paths = [Path(p) for p in output_paths]
    if not paths or not all(p.exists() for p in paths):
        return False
    return sig_path.read_text().strip() == fingerprint(sig_inputs)


def stamp_render(name: str, sig_inputs, cache_dir: Path) -> None:
    """Record current signature so subsequent runs can detect 'render is fresh'."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{name}.render_sig").write_text(fingerprint(sig_inputs))
