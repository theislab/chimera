"""Per-run health report for every islet grid and seed run. Descriptive; selects nothing.

For each finished run: Delta-alpha sign profile against the prior the run anchored on
(``final_state`` ``prior``, clamped on clamped recipes) and against the raw prior
(``var['conservation_prior']``): fractions positive and negative, fractions > +0.05 and < -0.05,
SD, median. Species mixing: ``knn_mixing_entropy`` (``compute_integration_metrics``, k = 15,
seed 42) and the final adversarial accuracy.

Flag, fixed before any seed run: a seed whose ``knn_mixing_entropy`` is below 0.75 x that of the
selected grid run is listed as poorly mixed and reported separately; the primary D stays the mean
over seeds (rule of the pre-specified analysis plan, section 2).

Gate G-D2 (``select_recipe.gate``, thresholds unchanged) is applied to every run, the seeds
included, for disclosure only: the gate selected the s42 grid run and selects nothing here.

    python -I analysis/tasks/T7_9_recipe_b/run_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training.compare_vs_scvi import compute_integration_metrics  # noqa: E402
from analysis.tasks.T7_9_recipe_b.select_recipe import gate  # noqa: E402

TASK = Path(__file__).resolve().parent
RUNS = REPO / "training" / "runs"
OUT = REPO / "results" / "analysis" / "T7_9" / "recipe_b"
ADATA = REPO / "data" / "analysis" / "disease_islet" / "islet_healthy_train.h5ad"
SEED_RUNS = [f"rev-isl-s{s}" for s in (43, 44, 45, 46)]
FLAG_RATIO = 0.75


def profile(d: np.ndarray, tag: str) -> dict:
    return {f"frac_pos_{tag}": float((d > 0).mean()), f"frac_neg_{tag}": float((d < 0).mean()),
            f"frac_gt_005_{tag}": float((d > 0.05).mean()), f"frac_lt_m005_{tag}": float((d < -0.05).mean()),
            f"sd_{tag}": float(d.std()), f"median_{tag}": float(np.median(d))}


def main() -> int:
    import anndata as ad

    a = ad.read_h5ad(ADATA, backed="r")
    raw = np.asarray(a.var["conservation_prior"], dtype=float)
    species = a.obs["species"].astype(str).to_numpy()
    labels = a.obs["cell_ontology_class"].astype(str).to_numpy()
    grid = pd.read_csv(TASK / "grid.tsv", sep="\t")["run_name"].tolist()
    rows = []
    for name in grid + SEED_RUNS:
        p = RUNS / name / "final_state.npz"
        if not p.exists():
            rows.append(dict(run_name=name, finished=False))
            continue
        fs = np.load(p, allow_pickle=True)
        alpha, prior = np.asarray(fs["alpha"], float), np.asarray(fs["prior"], float)
        gd2 = {f"gd2_{k}": v for k, v in gate(RUNS / name).items() if k != "finished"}
        if not gd2.get("gd2_no_nan", False):
            rows.append(dict(run_name=name, finished=True, **gd2))
            continue
        m = compute_integration_metrics(np.asarray(fs["z_mean"]), species, labels, k=15, seed=42)["global"]
        rows.append(dict(run_name=name, finished=True, **gd2, **profile(alpha - prior, "clamped"),
                         **profile(alpha - raw, "raw"), knn_mixing_entropy=m["knn_mixing_entropy"],
                         knn_species_purity=m["knn_species_purity"],
                         adversarial_accuracy=float(fs["adversarial_accuracy_final"])))
    tab = pd.DataFrame(rows)
    sel = OUT / "selection.json"
    tab["flag_poor_mixing"] = False
    if sel.exists():
        s = json.loads(sel.read_text()).get("selected")
        if s:
            ref = float(tab.loc[tab.run_name == s["run_name"], "knn_mixing_entropy"].iloc[0])
            tab["flag_poor_mixing"] = tab["run_name"].isin(SEED_RUNS) & (tab["knn_mixing_entropy"] < FLAG_RATIO * ref)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "run_report.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(tab.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
