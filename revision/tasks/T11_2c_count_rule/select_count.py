"""T11.2c: full-cohort recipe by count dependence (README.md). Reads alpha, prior and S_g_ema only.

Writes ``results/revision/T11/count_rule/selection.{tsv,json}`` and, for ``vz.DATASETS['full']``,
``revision/tasks/T11_2c_count_rule/seed_map_winner.tsv``.

    PYTHONPATH=<chimera repo> python revision/tasks/T11_2c_count_rule/select_count.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.tasks.T11_5_vz_genes import vz

CANDIDATES = {
    1: {42: "rev-vz-grid1-s42", 43: "rev-vz-s43", 44: "rev-vz-s44", 45: "rev-vz-s45", 46: "rev-vz-s46"},
    2: {42: "rev-vz-grid2-s42", 43: "rev-vz-grid2-s43", 44: "rev-vz-grid2-s44"},
    6: {42: "rev-vz-grid6-s42", 43: "rev-vz-grid6-s43", 44: "rev-vz-grid6-s44"},
}
MASK_EPS = 1e-6
TIE = 0.02
MIN_SEEDS = 2
OUT = vz.OUT / "count_rule"
TASK = vz.ROOT / "revision" / "tasks" / "T11_2c_count_rule"


def row(k: int, s: int, run: str) -> dict:
    d = vz.RUNS / run
    g = vz.gate(d) if d.exists() else dict(finished=False)
    r = dict(grid=k, seed=s, run=run, finished=bool(g.get("finished")), first_nan_snapshot=vz.first_nan_snapshot(run) if d.exists() else "")
    if r["finished"]:
        up, dn = vz.raw_tails(run) if g.get("no_nan") else (0.0, 0.0)
        r.update({q: g.get(q) for q in ("no_nan", "sat_high", "sat_low", "n_active_z", "last_epoch")}, tail_up_raw=up, tail_dn_raw=dn)
        r["healthy"] = bool(g.get("no_nan") and g.get("n_active_z") == 10 and g["sat_high"] <= vz.SAT_MAX_SEED
                            and g["sat_low"] <= vz.SAT_MAX_SEED and up >= 0.02 and dn >= 0.02)
        if r["healthy"]:
            with np.load(d / "final_state.npz") as z:
                da, S = z["alpha"].astype(float) - z["prior"].astype(float), z["S_g_ema"].astype(float)
            m = np.abs(S) >= MASK_EPS
            r["rho"], r["n_count_support"] = float(spearmanr(da[m], S[m])[0]), int(m.sum())
    else:
        r["healthy"] = False
    return r


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame([row(k, s, run) for k, seeds in CANDIDATES.items() for s, run in seeds.items()])
    t.to_csv(OUT / "selection.tsv", sep="\t", index=False)
    means = {int(k): float(x["rho"].mean()) for k, x in t[t["healthy"]].groupby("grid") if len(x) >= MIN_SEEDS}
    best = max(means.values())
    winner = min(k for k, v in means.items() if v > best - TIE)
    keep = t[(t["grid"] == winner) & t["healthy"]]
    (TASK / "seed_map_winner.tsv").write_text("run_name\tseed\tsource\n" + "".join(
        f"rev-vz-s{r.seed}\t{r.seed}\t{r.run} (grid {winner})\n" for r in keep.itertuples()))
    vz.write_json(OUT / "selection.json", dict(mean_rho=means, winner_grid=winner, seeds=keep["run"].tolist()))
    print(t.round(4).to_string(index=False))
    print(json.dumps(dict(mean_rho=means, winner_grid=winner), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
