"""Integration gate per seed: E8.1-E8.6 of ``training/compare_vs_scvi.py`` for every seed run.

Each seed's ``z_mean`` is scored with ``compute_integration_metrics`` (k=15, seed=42) and compared by
``evaluate_e8_gates`` with the cohort's stored scVI row in the canonical run's
``integration_vs_scvi.json``. No Delta-alpha, gold or disease value is read.

Usage: python analysis/tasks/T3_1b_integration_gate/run.py [--outdir results/analysis/T3_1b]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common import cohorts as C
from analysis.common import gene_ids

sys.path.insert(0, str(gene_ids.REPO_ROOT / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "T3_3_consensus"))
from compare_vs_scvi import compute_integration_metrics, evaluate_e8_gates  # noqa: E402
from run import CPU_SEEDS, MPS, PLATFORM, cpu_runs  # noqa: E402


def score(job):
    cohort, seed, run = job
    a = C.read_adata(cohort, backed=True)
    species = a.obs["species"].to_numpy()
    cts = a.obs["cell_type"].to_numpy()
    a.file.close()
    z = np.load(C.RUNS_DIR / run / "final_state.npz")["z_mean"]
    m = compute_integration_metrics(z, species, cts, k=15, seed=42)
    ref = json.loads((C.RUNS_DIR / C.CANONICAL[cohort] / "integration_vs_scvi.json").read_text())
    sc = [r for r in ref["rows"] if r["name"].startswith("vanilla scVI")][0]
    e8 = evaluate_e8_gates(m, sc)
    g = m["global"]
    row = dict(cohort=cohort, seed=seed, run=run, adv=float(np.load(C.RUNS_DIR / run / "final_state.npz")["adversarial_accuracy_final"]),
               knn_species_purity=g["knn_species_purity"], knn_mixing_entropy=g["knn_mixing_entropy"],
               sil_species=g["sil_species"], sil_celltype=g["sil_celltype"], label_transfer_acc=g["label_transfer_acc"],
               worst_excess=m.get("worst_purity_vs_chance_excess"))
    row.update({e["id"]: e["pass"] for e in e8})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=gene_ids.REPO_ROOT / "results/analysis/T3_1b")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for c in MPS:
        for s, r in MPS[c].items():
            jobs.append((c, f"mps{s}", r))
        for s, r in cpu_runs(c).items():
            jobs.append((c, f"cpu{s}", r))
        if c in PLATFORM:
            jobs.append((c, "cpu42", PLATFORM[c]))
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(score, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(a.outdir / "seed_integration.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
