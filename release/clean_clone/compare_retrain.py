"""Compare a retrained run with the archived runs of the same cohort and gate.

Run from the root of a checkout that holds the unpacked bundles, inside the environment::

    python -I release/clean_clone/compare_retrain.py <new_run> --cohort test_hmouse --gate A [--out table.csv]

``<new_run>`` is a folder under ``training/runs`` made by ``analysis/retrain_from_recipe.sh``.
``--gate A`` is the protein-feature gate (the canonical recipes), ``--gate M`` the per-gene gate.

Training on CPU is not deterministic at a fixed seed, so a retrain is not expected to repeat an
archived run value by value. The question is whether it falls inside the spread of the archived
seeds. The script prints, for the retrain against every archived run of the cohort, the Spearman
correlation of delta-alpha (alpha minus prior) and of the count evidence S_g; the correlation
with the archived seed-consensus readout D; and, as the reference, the range of the same
correlation among the archived runs themselves. It also prints the training state of the
retrain (final adversary accuracy, failed expectations), which must be looked at first: runs
that end in different training states are not comparable.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RUNS = Path("training/runs")
RECIPES = Path("configs/canonical_recipes.tsv")
KEPT_PER_GENE_GATE = Path("results/analysis/Z4/kept.json")
CONSENSUS = {"A": "results/analysis/H1/genes/{cohort}.parquet", "M": "results/analysis/Z4/h1/genes/{cohort}.parquet"}


def archived_runs(cohort: str, gate: str) -> dict[str, str]:
    """Archived runs of a cohort and gate: ``{label: run folder name}``, finished runs only."""
    runs: dict[str, str] = {}
    if gate == "A":
        with open(RECIPES, newline="") as handle:
            recipe = next(r for r in csv.DictReader(handle, delimiter="\t") if r["cohort"] == cohort)
        runs.update({f"canonical s{s}": recipe["run_name"].replace("{seed}", s) for s in recipe["seeds"].split(",")})
        runs.update({f"seed s{m.group(1)}": p.name for p in sorted(RUNS.glob(f"rev-{cohort}-s*"))
                     if (m := re.fullmatch(rf"rev-{re.escape(cohort)}-s(\d+)", p.name))})
        runs.update({f"CPU platform s{m.group(1)}": p.name for p in sorted(RUNS.glob(f"rev-cpu-{cohort}-s*"))
                     if (m := re.fullmatch(rf"rev-cpu-{re.escape(cohort)}-s(\d+)", p.name))})
    else:
        kept = json.loads(KEPT_PER_GENE_GATE.read_text()).get(cohort, {})
        runs.update({f"per-gene gate s{seed}": name for seed, name in sorted(kept.items())})
        runs.update({f"per-gene gate s{m.group(1)}": p.name for p in sorted(RUNS.glob(f"z3-{cohort}-m-s*"))
                     if (m := re.fullmatch(rf"z3-{re.escape(cohort)}-m-s(\d+)", p.name))})
    return {label: name for label, name in runs.items() if (RUNS / name / "final_state.npz").is_file()}


def load(run: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """delta-alpha, S_g and the key metrics of a run."""
    with np.load(RUNS / run / "final_state.npz", allow_pickle=False) as state:
        delta = np.asarray(state["alpha"], float) - np.asarray(state["prior"], float)
        evidence = np.asarray(state["S_g_ema"], float)
    summary = json.loads((RUNS / run / "summary.json").read_text())
    return delta, evidence, summary


def print_pair_range(label: str, deltas: list[np.ndarray]) -> None:
    """Print median and range of the pairwise Spearman correlation of delta-alpha among runs."""
    pairs = [spearmanr(a, b)[0] for i, a in enumerate(deltas) for b in deltas[i + 1:]]
    if pairs:
        print(f"{label}: pairwise Spearman median {np.median(pairs):.3f}, range {min(pairs):.3f} to {max(pairs):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("new_run", help="folder name of the retrained run under training/runs")
    parser.add_argument("--cohort", required=True, help="recipe key, e.g. liv_hmouse or test_hmouse")
    parser.add_argument("--gate", choices=["A", "M"], default="A", help="A protein-feature gate, M per-gene gate")
    parser.add_argument("--out", type=Path, help="write the comparison table as csv")
    args = parser.parse_args()

    delta_new, evidence_new, summary = load(args.new_run)
    metrics = summary.get("key_metrics", {})
    print(f"retrain {args.new_run} (cohort {args.cohort}, gate {args.gate}): "
          f"adv_acc_final {metrics.get('adv_acc_final')}, failed expectations {summary.get('failed_expectations')}")

    archived = archived_runs(args.cohort, args.gate)
    loaded = {label: load(name) for label, name in archived.items()}
    rows = [{"archived_run": archived[label], "what": label,
             "rho_delta_alpha": spearmanr(delta_new, delta)[0], "rho_S": spearmanr(evidence_new, evidence)[0],
             "adv_acc_final": s.get("key_metrics", {}).get("adv_acc_final")}
            for label, (delta, evidence, s) in loaded.items()]
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    consensus_path = Path(CONSENSUS[args.gate].format(cohort=args.cohort))
    if consensus_path.is_file():
        consensus = pd.read_parquet(consensus_path)
        centered = delta_new - np.median(delta_new)
        print(f"retrain against the archived consensus ({consensus_path}): "
              f"Spearman D {spearmanr(centered, consensus['D'])[0]:.3f}, S {spearmanr(evidence_new, consensus['S_ema'])[0]:.3f}")
    seeds_only = [delta for label, (delta, _, _) in loaded.items() if not label.startswith("CPU platform")]
    print_pair_range("reference, archived runs, one per seed", seeds_only)
    if len(seeds_only) < len(loaded):
        print_pair_range("reference, all archived runs (with the CPU platform repeats)",
                         [delta for delta, _, _ in loaded.values()])
    if args.out:
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
