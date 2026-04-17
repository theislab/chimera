"""Quick inspection of the adv_accuracy trajectory for a single run.

Prints the post-warmup evolution + the E6.2 status (strict + trajectory)
without re-running ``check_expectations.py``. Useful for peeking at runs
while training and for multi-species diagnosis.

Usage:
    python training/quick_adv_trajectory.py <run-name>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(run: str) -> int:
    runs_root = Path(__file__).resolve().parents[1] / "training" / "runs"
    csvs = list((runs_root / run).rglob("metrics.csv"))
    if not csvs:
        print(f"no metrics.csv under {runs_root / run}")
        return 1
    df = pd.read_csv(csvs[0])
    if "adversarial_accuracy" not in df.columns:
        print("no adversarial_accuracy column")
        return 1
    adv = (
        df[["epoch", "adversarial_accuracy"]]
        .dropna()
        .groupby("epoch")
        .max()
        .sort_index()["adversarial_accuracy"]
    )
    # Infer n_species from final_state.npz if present.
    fs_path = runs_root / run / "final_state.npz"
    n_sp = 2
    if fs_path.exists():
        fs = np.load(fs_path)
        if "n_species" in fs:
            n_sp = int(fs["n_species"].item())
    chance = 1.0 / n_sp
    midpoint = chance + (1.0 - chance) * 0.5

    warm = 5  # default warmup_adversarial_epochs (not in CSV, assume 5)
    post = adv[adv.index >= warm]
    peak_val = float(post.max())
    peak_ep = int(post.idxmax())
    last_val = float(adv.iloc[-1])
    tail10 = adv.tail(10)
    below_mid = int((tail10 < midpoint).sum())

    strict = (chance - 0.15) <= last_val <= (chance + 0.20)
    traj = (last_val < peak_val - 0.05) and (below_mid >= 5)

    print(f"[{run}] K={n_sp}  chance={chance:.3f}  midpoint={midpoint:.3f}")
    print(f"  peak:   {peak_val:.3f} @ epoch {peak_ep}")
    print(f"  final:  {last_val:.3f}")
    print(f"  tail10: {tail10.values.round(3).tolist()}")
    print(f"  <midpt: {below_mid}/10")
    print(f"  E6.2 strict={strict}  trajectory={traj}  overall_pass={strict or traj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
