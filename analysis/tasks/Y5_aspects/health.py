"""G-V health gate on every seed of the seven canonical cohorts. Training-health quantities only; no readout.

Usage: python -m analysis.tasks.Y5_aspects.health
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.common import gene_ids
REPO = gene_ids.REPO_ROOT
OUT = REPO / "results" / "analysis" / "Y5"
sys.path.insert(0, str(REPO / "analysis" / "tasks" / "T12_2b_recipe"))
from select_recipe import gate  # noqa: E402
RUNS = REPO / "training" / "runs"


def main():
    t = pd.read_csv(REPO / "results/analysis/X3/seed_sets.csv")
    rows = []
    for r in t.itertuples():
        for name in str(r.runs).split():
            g = gate(RUNS / name)
            rows.append(dict(cohort=r.cohort, seed=int(name.rsplit("-s", 1)[-1]) if "-s" in name else -1, run=name,
                             **{k: g.get(k) for k in ("finished", "no_nan", "sat_high", "sat_low", "n_active_z", "z_var_min",
                                                      "tail_up", "tail_dn", "pass_GV", "last_epoch")}))
    h = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    h.to_csv(OUT / "seed_health.csv", index=False)
    print(h.groupby("cohort").pass_GV.agg(["sum", "count"]))
    print(h[~h.pass_GV.astype(bool)].to_string())


if __name__ == "__main__":
    main()


def raw_tails():
    """Tail fractions against the raw prior (health-only quantity), beside the literal gate."""
    from analysis.common import cohorts as C
    import anndata as ad
    h = pd.read_csv(OUT / "seed_health.csv")
    up, dn = [], []
    for r in h.itertuples():
        fs = np.load(RUNS / r.run / "final_state.npz", allow_pickle=True)
        a = ad.read_h5ad(gene_ids.DATA_PROCESSED / f"{gene_ids.H5ADS[r.cohort]}.h5ad", backed="r")
        d = np.asarray(fs["alpha"], float) - np.asarray(a.var["conservation_prior"], float)
        a.file.close()
        up.append(float((d > .05).mean()))
        dn.append(float((d < -.05).mean()))
    h["tail_up_raw"], h["tail_dn_raw"] = up, dn
    sat = (h.sat_high <= .05) & (h.sat_low <= .05)
    act = h.n_active_z == 10
    h["pass_GV_rawtail"] = h.no_nan.astype(bool) & sat & act & (h.tail_up_raw >= .02) & (h.tail_dn_raw >= .02)
    h.to_csv(OUT / "seed_health.csv", index=False)
    print(h.groupby("cohort")[["pass_GV", "pass_GV_rawtail"]].sum())
    print(h[["cohort", "seed", "sat_high", "tail_up", "tail_dn", "tail_up_raw", "tail_dn_raw", "pass_GV", "pass_GV_rawtail"]].round(3).to_string())
