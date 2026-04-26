"""iter119 per-gene-bias post-hoc extraction.

The running iter119 process imported the legacy ``dump_final_state`` (before the
mlp_correction + per_gene_bias fields were added). This helper loads a finished
iter119 checkpoint, recovers MLP-only output and per-gene bias separately, and
either:

1. Writes a sidecar ``bias_decomp.npz`` alongside ``final_state.npz``.
2. Updates the in-place ``final_state.npz`` to add the two new fields (default).

Usage
-----
    python training/iter119_bias_extract.py <run-dir> [--sidecar]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch


def extract(run_dir: Path, sidecar: bool = False) -> dict:
    cfg_path = run_dir / "config.json"
    fs_path = run_dir / "final_state.npz"
    if not fs_path.exists():
        raise FileNotFoundError(f"missing {fs_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing {cfg_path}")

    import json
    cfg = json.loads(cfg_path.read_text())
    if not cfg.get("use_per_gene_bias"):
        print(f"  use_per_gene_bias=False — skipping (correction = MLP only).")
        return {}

    adata_path = Path("/Users/kemalinecik/git_nosync/master_chimera/chimera") / cfg["adata_path"]
    adata = ad.read_h5ad(adata_path)
    # Re-apply the prior clamp the same way train_and_diagnose did, so the
    # buffer matches the training-time conservation_prior.
    if cfg.get("prior_clamp_max") is not None and cfg["prior_clamp_max"] < 0.95:
        clamp = float(cfg["prior_clamp_max"])
        raw = adata.var["conservation_prior"].values.astype(np.float64).copy()
        adata.var["conservation_prior"] = np.clip(raw, 0.05, clamp).astype(
            adata.var["conservation_prior"].dtype
        )

    setup_kwargs: dict = {"batch_key": cfg.get("batch_key", "batch"), "species_key": "species"}
    if cfg.get("use_celltype_conditional_mmd") and cfg.get("celltype_key"):
        setup_kwargs["celltype_key"] = cfg["celltype_key"]
    elif cfg.get("use_celltype_conditional_mmd"):
        setup_kwargs["celltype_key"] = "cell_type"

    from chimera import ChimeraModel
    ChimeraModel.setup_anndata(adata, **setup_kwargs)
    m = ChimeraModel.load(str(run_dir / "model"), adata, accelerator="cpu")
    m.module.eval()

    gin = m.module.gin
    if not getattr(gin, "use_per_gene_bias", False) or gin.per_gene_bias is None:
        raise RuntimeError(
            "Loaded model does not have per_gene_bias — config and model disagree?"
        )

    # IMPORTANT: copy() before any in-place ops on the underlying tensor.
    # Without copy, `bias` shares storage with gin.per_gene_bias.data; a later
    # zero_() then re-assignment via `.data = saved` leaves `bias` pointing at
    # the OLD storage that was zeroed — silently writing zeros to the npz even
    # when the trained values were non-zero (the bug that caused iter119's
    # mis-conclusion that "bias never moved").
    bias = gin.per_gene_bias.detach().cpu().float().numpy().copy()
    # MLP-only output: zero the bias temporarily, run forward, restore.
    saved = gin.per_gene_bias.data.clone()
    with torch.no_grad():
        gin.per_gene_bias.data.zero_()
        _, mlp_corr_t = gin(m.module.gene_features, m.module.conservation_prior)
        gin.per_gene_bias.data = saved
    mlp_correction = mlp_corr_t.detach().cpu().float().numpy()

    fs = dict(np.load(fs_path))
    correction_total = fs["correction"]
    reconstructed = mlp_correction + bias
    rmse = float(np.sqrt(((reconstructed - correction_total) ** 2).mean()))
    print(f"  reconstruction rmse (mlp+bias - saved correction) = {rmse:.2e}")
    if rmse > 1e-3:
        print("  WARNING: reconstruction deviates from saved correction; module state mismatch?")

    out = {
        "mlp_correction": mlp_correction,
        "per_gene_bias": bias,
    }

    if sidecar:
        sidecar_path = run_dir / "bias_decomp.npz"
        np.savez_compressed(sidecar_path, **out)
        print(f"  → sidecar {sidecar_path}")
    else:
        # In-place: re-save final_state.npz with the new fields.
        new_fs = dict(fs)
        new_fs.update(out)
        np.savez_compressed(fs_path, **new_fs)
        print(f"  → updated {fs_path} (added mlp_correction, per_gene_bias)")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--sidecar", action="store_true",
                   help="Write bias_decomp.npz instead of patching final_state.npz")
    args = p.parse_args()
    print(f"=== {args.run_dir.name} ===")
    extract(args.run_dir, sidecar=args.sidecar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
