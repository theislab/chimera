"""UMAP diagnostic — generates a 2-panel UMAP and reports the cloud aspect ratio.

Reads ``final_state.npz['z_mean']`` from a run dir, fits scanpy UMAP, plots
species + celltype panels, and computes a robust aspect ratio (range[u1] /
range[u0]) so a thin/elongated embedding triggers automatically.

Aspect ratio is computed on UMAP1 vs UMAP2 ranges (1st-99th percentile, robust
to outliers). aspect > 1.5 → thin/elongated; this is a known regression mode
for testis 200ep schedules and a falsifying probe in the iter118 template.

Output: ``<run-dir>/plots/umap.png`` and ``<run-dir>/umap_aspect.json``.

Usage
-----
    python training/umap_check.py <run-dir> --adata data/processed/<file>.h5ad
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--adata", type=Path, required=True)
    p.add_argument("--celltype-col", default="cell_type")
    p.add_argument("--species-col", default="species")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not (args.run_dir / "final_state.npz").exists():
        print(f"missing final_state.npz in {args.run_dir}", file=sys.stderr)
        return 1

    fs = np.load(args.run_dir / "final_state.npz")
    if "z_mean" not in fs.files:
        print(f"final_state.npz has no z_mean", file=sys.stderr)
        return 1
    z = fs["z_mean"]

    import anndata as ad
    import scanpy as sc

    adata = ad.read_h5ad(args.adata)
    if z.shape[0] != adata.n_obs:
        print(f"shape mismatch z={z.shape} vs adata={adata.shape}", file=sys.stderr)
        return 1
    adata.obsm["X_z"] = z

    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep="X_z")
    sc.tl.umap(adata, random_state=args.seed)

    u = adata.obsm["X_umap"]
    # robust range via percentile band
    u_lo = np.percentile(u, 1, axis=0)
    u_hi = np.percentile(u, 99, axis=0)
    extents = u_hi - u_lo
    aspect = float(extents.max() / max(extents.min(), 1e-9))

    plots_dir = args.run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if args.species_col in adata.obs.columns:
        sc.pl.umap(adata, color=args.species_col, ax=axes[0], show=False, frameon=False)
        axes[0].set_title(f"species — UMAP aspect={aspect:.2f}")
    if args.celltype_col in adata.obs.columns:
        sc.pl.umap(
            adata,
            color=args.celltype_col,
            ax=axes[1],
            show=False,
            frameon=False,
            legend_loc="on data",
            legend_fontsize=6,
        )
        axes[1].set_title(f"{args.celltype_col} — UMAP aspect={aspect:.2f}")
    fig.tight_layout()
    out_path = plots_dir / "umap.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    summary = {
        "run_name": args.run_dir.name,
        "umap_aspect_p1_p99": aspect,
        "extent_u0": float(extents[0]),
        "extent_u1": float(extents[1]),
        "n_cells": int(adata.n_obs),
        "n_neighbors": args.n_neighbors,
        "umap_seed": args.seed,
        "plot_path": str(out_path),
    }
    (args.run_dir / "umap_aspect.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
