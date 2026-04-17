"""Produce fast, canonical diagnostic plots for one Chimera run.

Six plots per run, saved to `<run_dir>/plots/`:
  1. Training loss trajectories (L_recon, L_kl, L_shared_aux, L_total)
  2. Alpha dynamics (mean, std, frac<0.1, frac>0.9, mean_delta over epochs)
  3. Adversarial trajectory (fool_loss, classifier_loss, adv_accuracy)
  4. Alpha histogram (final) vs prior (overlaid)
  5. L-shape scatter (alpha vs mean|residual|) colored by sg_ema
  6. z-dimension variance profile (how many dims are "alive")

No scib metrics are computed — these are FAST plots intended for iteration loops.

Usage
-----
    python training/plot_diagnostics.py --run-name iter10-low-res
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "training" / "runs"


def _find_metrics_csv(run_dir: Path) -> Path | None:
    candidates = list((run_dir / "lightning").rglob("metrics.csv"))
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0] if candidates else None


def _epoch_series(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    if col not in df.columns or "epoch" not in df.columns:
        return np.array([]), np.array([])
    tmp = df[["epoch", col]].dropna().groupby("epoch")[col].max().reset_index().sort_values("epoch")
    return tmp["epoch"].values, tmp[col].values


def plot_loss_trajectories(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for i, col in enumerate(["train/L_recon", "train/L_kl", "train/L_shared_aux", "train/L_total"]):
        r, c = i // 2, i % 2
        ep, val = _epoch_series(df, col)
        if len(ep):
            ax[r, c].plot(ep, val, marker=".", ms=3)
            ax[r, c].set_yscale("log")
        ax[r, c].set_title(col)
        ax[r, c].set_xlabel("epoch")
        ax[r, c].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_alpha_dynamics(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for i, col in enumerate(["alpha/mean", "alpha/std", "alpha/mean_delta", "alpha/frac_delta_positive"]):
        r, c = i // 2, i % 2
        ep, val = _epoch_series(df, col)
        if len(ep):
            ax[r, c].plot(ep, val, marker=".", ms=3)
        ax[r, c].set_title(col)
        ax[r, c].set_xlabel("epoch")
        ax[r, c].grid(alpha=0.3)
        ax[r, c].axhline(0, color="k", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_adversarial(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    # fool loss and classifier loss
    for col, lbl in [("adv/fool_loss", "fool"), ("adv/classifier_loss", "classifier")]:
        ep, val = _epoch_series(df, col)
        if len(ep):
            ax[0].plot(ep, val, marker=".", ms=3, label=lbl)
    ax[0].legend()
    ax[0].set_title("adversarial losses")
    ax[0].set_xlabel("epoch")
    ax[0].grid(alpha=0.3)

    # adversarial accuracy over time
    ep, val = _epoch_series(df, "adversarial_accuracy")
    if len(ep):
        ax[1].plot(ep, val, marker=".", ms=3)
    ax[1].set_title("adv accuracy vs epoch")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylim(0, 1)
    ax[1].axhline(0.5, color="r", linestyle="--", label="K=2 chance", alpha=0.5)
    ax[1].axhline(0.25, color="g", linestyle="--", label="K=4 chance", alpha=0.5)
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    # KL weight / adv active schedule
    ep, val = _epoch_series(df, "schedule/kl_weight")
    if len(ep):
        ax[2].plot(ep, val, marker=".", ms=3, label="kl_weight")
    ep, val = _epoch_series(df, "schedule/adv_active")
    if len(ep):
        ax[2].plot(ep, val, marker=".", ms=3, label="adv_active")
    ep, val = _epoch_series(df, "schedule/fqr_active")
    if len(ep):
        ax[2].plot(ep, val, marker=".", ms=3, label="fqr_active")
    ax[2].legend()
    ax[2].set_title("training schedule")
    ax[2].set_xlabel("epoch")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_alpha_histogram(alpha: np.ndarray, prior: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 51)
    ax[0].hist(prior, bins=bins, alpha=0.5, label=f"prior (n={len(prior)})")
    ax[0].hist(alpha, bins=bins, alpha=0.5, label=f"alpha final (mean={alpha.mean():.3f})")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("n genes")
    ax[0].legend()
    ax[0].set_title("Alpha distribution vs Prior")
    ax[0].grid(alpha=0.3)

    delta = alpha - prior
    ax[1].hist(delta, bins=np.linspace(-0.5, 0.5, 101))
    ax[1].axvline(0, color="k", lw=1)
    ax[1].axvline(0.02, color="g", lw=1, linestyle="--", label="+0.02")
    ax[1].axvline(-0.02, color="r", lw=1, linestyle="--", label="-0.02")
    frac_pos = float((delta > 0.02).mean())
    frac_neg = float((delta < -0.02).mean())
    ax[1].set_title(f"delta_alpha: +%={frac_pos:.2%}, -%={frac_neg:.2%}")
    ax[1].set_xlabel("alpha - prior")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_l_shape(alpha: np.ndarray, residual_mean_abs: np.ndarray, sg_ema: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    scatter = ax.scatter(alpha, residual_mean_abs, c=sg_ema, cmap="viridis", s=2, alpha=0.5)
    from scipy.stats import pearsonr
    if np.std(alpha) > 0 and np.std(residual_mean_abs) > 0:
        r, _ = pearsonr(alpha, residual_mean_abs)
        ax.set_title(f"L-shape: Pearson r = {r:.3f}")
    ax.set_xlabel("alpha (GIN output)")
    ax.set_ylabel("mean |residual log-rate|")
    plt.colorbar(scatter, ax=ax, label="S_g EMA")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_z_variance_profile(z: np.ndarray, out_path: Path) -> None:
    dim_var = z.var(axis=0)
    order = np.argsort(dim_var)[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(range(len(dim_var)), dim_var[order])
    ax[0].axhline(0.01, color="r", linestyle="--", label="E6.3 threshold")
    ax[0].set_xlabel("z-dim (ranked)")
    ax[0].set_ylabel("Var(z_mean)")
    ax[0].legend()
    ax[0].set_title("Per-dim variance of z_mean")
    ax[0].grid(alpha=0.3)

    # correlation matrix across z-dims
    if z.shape[1] <= 30:
        corr = np.corrcoef(z.T)
        im = ax[1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax[1])
        ax[1].set_title("z-dim correlation matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()

    run_dir = RUNS_ROOT / args.run_name
    if not run_dir.exists():
        print(f"Run not found: {run_dir}", file=sys.stderr)
        return 2

    out = run_dir / "plots"
    out.mkdir(exist_ok=True)

    metrics_path = _find_metrics_csv(run_dir)
    if metrics_path is not None:
        df = pd.read_csv(metrics_path)
        plot_loss_trajectories(df, out / "01_loss_trajectories.png")
        plot_alpha_dynamics(df, out / "02_alpha_dynamics.png")
        plot_adversarial(df, out / "03_adversarial.png")

    fs_path = run_dir / "final_state.npz"
    if fs_path.exists():
        fs = np.load(fs_path)
        alpha = fs["alpha"].astype(np.float64)
        prior = fs["prior"].astype(np.float64)
        residual = fs["residual_mean_abs"].astype(np.float64)
        sg = fs["S_g_ema"].astype(np.float64)
        z = fs["z_mean"].astype(np.float64)
        plot_alpha_histogram(alpha, prior, out / "04_alpha_histogram.png")
        plot_l_shape(alpha, residual, sg, out / "05_l_shape.png")
        plot_z_variance_profile(z, out / "06_z_variance.png")

    print(f"Plots written to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
