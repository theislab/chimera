"""Print training curves of runs at selected epochs (health quantities only; no gold).

    python analysis/tasks/Z1_retune/curves.py run [run ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
EPOCHS = [5, 10, 20, 25, 30, 40, 50, 60, 80, 100, 120, 149, 160, 199]
COLS = {"lr/encoder": "lr", "validation_loss": "val", "train/L_kl": "kl", "train/L_mmd": "mmd",
        "adversarial_accuracy": "adv", "z/mean_learned_var": "lvar", "z/min_var_dim": "zmin",
        "alpha/std": "astd", "fqr/sg_frac_positive": "S+", "train/L_recon": "recon"}


def show(run: str) -> None:
    d = REPO / "training" / "runs" / run
    m = pd.read_csv(sorted(d.glob("lightning/**/metrics.csv"))[-1])
    t = {}
    for c, k in COLS.items():
        if c in m:
            t[k] = m[["epoch", c]].dropna().groupby("epoch")[c].last()
    t = pd.DataFrame(t).reindex([e for e in EPOCHS if e <= m["epoch"].max()])
    # snapshots: rho(D, S_ema) over training on unmasked genes, rho with final alpha
    rows = {}
    files = sorted((d / "alpha_snapshots").glob("alpha_epoch_*.npz"))
    if files:
        final = np.load(files[-1])
        for f in files:
            e = int(f.stem.split("_")[-1])
            if e in t.index or e - 1 in t.index:
                s = np.load(f)
                D, S = s["alpha"] - s["prior"], s["S_g_ema"]
                un = np.abs(S) >= 1e-6
                if un.sum() > 10 and np.isfinite(D).all() and D.std() > 0:
                    rows[e] = dict(rhoDS=spearmanr(D[un], S[un])[0], rhoDf=spearmanr(D, final["alpha"] - final["prior"])[0],
                                   rhoSf=spearmanr(S, final["S_g_ema"])[0], sdD=D.std(), sat=(s["alpha"] > 0.95).mean())
    t = t.join(pd.DataFrame(rows).T.reindex(t.index))
    print(f"== {run} last_epoch {int(m['epoch'].max()) + 1}")
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", lambda x: f"{x:.4g}"):
        print(t)


for r in sys.argv[1:]:
    show(r)
