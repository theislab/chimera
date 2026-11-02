"""T12.3 step 1: per-seed health of the bone-marrow seed runs, before any readout (plan section 12).

Runs: ``revision/tasks/T12_2b_recipe/seed_map.tsv`` (s42 = the selected grid run
``rev-bm2-grid5-s42``, s43-s46). Per run:

* gate G-V with the selection thresholds, unchanged (``T12_2b_recipe/select_recipe.py:gate``):
  no NaN (and no E2.1 failure); saturation <= 5% in each alpha tail; 10/10 active z dims
  (Var(z_mean) >= 0.01); both tails of alpha - prior (> +0.05 and < -0.05) in >= 2% of genes;
* Delta-alpha sign profile against the prior the run anchored on (clamped at 0.85) and against
  the raw prior (``var['conservation_prior']``): fractions positive / negative, > +0.05 / < -0.05,
  SD, median (coordinator note, ``ledgers/T12_2.md``);
* species mixing: ``knn_mixing_entropy`` and ``knn_species_purity`` from
  ``training/compare_vs_scvi.py::compute_integration_metrics`` (k = 15, seed 42, labels
  ``cell_ontology_class``), final adversary accuracy.

Flag, fixed before the seed runs were read: a seed with ``knn_mixing_entropy`` below 0.75 x that of
s42 is listed as poorly mixed and reported separately (the P13 rule); D stays the mean over the
seeds that the section 2 rule admits. A non-finite run cannot enter D and is reported, not dropped
silently.

No D, S_g, COE, baseline or biology readout is computed here.

    PYTHONPATH=<chimera repo> python revision/tasks/T12_3_tests/seed_health.py
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

RECIPE = REPO / "revision" / "tasks" / "T12_2b_recipe"
sys.path.insert(0, str(RECIPE))
from select_recipe import gate  # noqa: E402

RUNS = REPO / "training" / "runs"
OUT = REPO / "results" / "revision" / "P12"
ADATA = REPO / "data" / "revision" / "bone_marrow" / "v2" / "bone_marrow_unified.h5ad"
FLAG_RATIO = 0.75


def seed_runs() -> dict[int, str]:
    m = pd.read_csv(RECIPE / "seed_map.tsv", sep="\t")
    return {int(r["seed"]): str(r["source"]).split()[0].replace("training/runs/", "") for _, r in m.iterrows()}


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
    rows = []
    for seed, name in sorted(seed_runs().items()):
        d = RUNS / name
        r = dict(seed=seed, run_name=name, **gate(d))
        if r.get("finished"):
            fs = np.load(d / "final_state.npz", allow_pickle=True)
            alpha, prior = np.asarray(fs["alpha"], float), np.asarray(fs["prior"], float)
            r["adversarial_accuracy"] = float(fs["adversarial_accuracy_final"]) if "adversarial_accuracy_final" in fs.files else np.nan
            if r.get("no_nan"):
                r.update(profile(alpha - prior, "clamped"))
                r.update(profile(alpha - raw, "raw"))
                m = compute_integration_metrics(np.asarray(fs["z_mean"]), species, labels, k=15, seed=42)["global"]
                r.update(knn_mixing_entropy=m["knn_mixing_entropy"], knn_species_purity=m["knn_species_purity"],
                         label_transfer_acc=m["label_transfer_acc"])
        rows.append(r)
    tab = pd.DataFrame(rows)
    ref = float(tab.loc[tab.seed == 42, "knn_mixing_entropy"].iloc[0])
    tab["flag_poor_mixing"] = (tab.seed != 42) & (tab["knn_mixing_entropy"] < FLAG_RATIO * ref)
    tab["enters_D"] = tab["finished"].astype(bool) & tab["no_nan"].fillna(False).astype(bool)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "seed_health.tsv", sep="\t", index=False)
    (OUT / "seed_health.json").write_text(json.dumps(dict(
        flag_rule=f"knn_mixing_entropy < {FLAG_RATIO} x s42 ({FLAG_RATIO * ref:.4f})",
        enters_D_rule="finished and finite (all CPU: the section 2 MPS clause does not apply)",
        seeds_enter_D=[int(s) for s in tab.loc[tab.enters_D, "seed"]]), indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(tab.T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
