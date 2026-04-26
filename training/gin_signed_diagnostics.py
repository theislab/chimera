"""GIN per-gene signed-correction diagnostics.

Reads ``final_state.npz`` from a run dir and produces the per-gene metrics that
matter for the testis E9.2 down architectural ceiling:

  - Spearman rho(S_g_ema, correction) — overall and on the non-zero S_g subset.
  - Mean correction on the rare neg-S_g class (S_g_ema < -0.05) — co-primary
    metric per the iter117 lesson (rho can be misleading; class-mean is
    what determines E9.2 push<-0.05).
  - Mean correction on the abundant pos-S_g class (S_g_ema > +0.05).
  - Fraction of each class where the correction matches the expected sign.
  - alpha distribution (mean, std, sat fractions).
  - delta-alpha = alpha - prior bidirectional fractions (E9.2 up/down).
  - correction range/std.
  - z PC1/PC2 SVD aspect ratio — linear-projection counterpart of the UMAP
    aspect (catches encoder over-fitting to a thin manifold even when the
    nonlinear UMAP embedding looks round).

Output: prints a JSON line + writes ``<run-dir>/gin_signed_diagnostics.json``.

Usage
-----
    python training/gin_signed_diagnostics.py <run-dir> [<run-dir> ...]

Idempotent — overwrites the per-run JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def compute(run_dir: Path) -> dict:
    fs = np.load(run_dir / "final_state.npz")
    alpha = fs["alpha"]
    prior = fs["prior"]
    correction = fs["correction"]
    sg_ema = fs["S_g_ema"]
    delta = alpha - prior

    pos_mask = sg_ema > 0.05
    neg_mask = sg_ema < -0.05
    nz_mask = pos_mask | neg_mask

    out: dict = {}

    if nz_mask.sum() >= 5:
        rho_all = spearmanr(sg_ema, correction).statistic
        rho_nz = spearmanr(sg_ema[nz_mask], correction[nz_mask]).statistic
    else:
        rho_all = float("nan")
        rho_nz = float("nan")

    out["rho_S_g_corr_all"] = float(rho_all)
    out["rho_S_g_corr_nz"] = float(rho_nz)

    out["n_neg_S_g"] = int(neg_mask.sum())
    out["n_pos_S_g"] = int(pos_mask.sum())
    out["mean_corr_on_neg_S_g"] = (
        float(correction[neg_mask].mean()) if neg_mask.sum() > 0 else float("nan")
    )
    out["mean_corr_on_pos_S_g"] = (
        float(correction[pos_mask].mean()) if pos_mask.sum() > 0 else float("nan")
    )
    out["frac_corr_neg_on_neg_S_g"] = (
        float((correction[neg_mask] < 0).mean()) if neg_mask.sum() > 0 else float("nan")
    )
    out["frac_corr_pos_on_pos_S_g"] = (
        float((correction[pos_mask] > 0).mean()) if pos_mask.sum() > 0 else float("nan")
    )

    out["alpha_mean"] = float(alpha.mean())
    out["alpha_std"] = float(alpha.std())
    out["frac_alpha_gt_095"] = float((alpha > 0.95).mean())
    out["frac_alpha_lt_005"] = float((alpha < 0.05).mean())

    out["frac_delta_gt_005_E9_2_up"] = float((delta > 0.05).mean())
    out["frac_delta_lt_neg005_E9_2_down"] = float((delta < -0.05).mean())
    out["frac_delta_gt_002"] = float((delta > 0.02).mean())
    out["frac_delta_lt_neg002"] = float((delta < -0.02).mean())

    out["corr_min"] = float(correction.min())
    out["corr_max"] = float(correction.max())
    out["corr_std"] = float(correction.std())

    # iter119: per-gene-bias hybrid decomposition diagnostics. When the run was
    # trained with --use-per-gene-bias the dump saves mlp_correction and
    # per_gene_bias; their sum equals correction. The bias / MLP magnitude
    # ratio is the iter119 falsifying probe — if bias dominates >85% the hybrid
    # has degenerated to config M; if bias contributes <5% the per-gene dof
    # never engaged and the mechanism is inert.
    if "mlp_correction" in fs.files and "per_gene_bias" in fs.files:
        mlp_only = fs["mlp_correction"]
        bias = fs["per_gene_bias"]
        mlp_abs = float(np.abs(mlp_only).mean())
        bias_abs = float(np.abs(bias).mean())
        denom = mlp_abs + bias_abs
        out["per_gene_bias_active"] = bool(bias_abs > 1e-6)
        out["mlp_correction_abs_mean"] = mlp_abs
        out["per_gene_bias_abs_mean"] = bias_abs
        out["bias_share_of_total_magnitude"] = (
            float(bias_abs / denom) if denom > 0 else 0.0
        )
        # Per-class breakdown — the headline metric for the hybrid: does the
        # per-gene bias actually rescue the rare neg-S_g class?
        if neg_mask.sum() > 0:
            out["mean_mlp_correction_on_neg_S_g"] = float(mlp_only[neg_mask].mean())
            out["mean_per_gene_bias_on_neg_S_g"] = float(bias[neg_mask].mean())
        if pos_mask.sum() > 0:
            out["mean_mlp_correction_on_pos_S_g"] = float(mlp_only[pos_mask].mean())
            out["mean_per_gene_bias_on_pos_S_g"] = float(bias[pos_mask].mean())

    if "z_mean" in fs.files:
        zm = fs["z_mean"]
        zm_centered = zm - zm.mean(axis=0, keepdims=True)
        s = np.linalg.svd(zm_centered, compute_uv=False)
        if len(s) >= 2:
            out["z_pc1_pc2_aspect"] = float(s[0] / max(s[1], 1e-9))

    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+", type=Path)
    args = p.parse_args()

    for run_dir in args.run_dirs:
        if not run_dir.is_dir():
            print(f"SKIP (not a directory): {run_dir}", file=sys.stderr)
            continue
        if not (run_dir / "final_state.npz").exists():
            print(f"SKIP (missing final_state.npz): {run_dir}", file=sys.stderr)
            continue
        result = compute(run_dir)
        out_path = run_dir / "gin_signed_diagnostics.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"=== {run_dir.name} ===")
        print(json.dumps(result, indent=2))
        print(f"  → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
