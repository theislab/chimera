"""Post-hoc z-leakage probe — fits linear and non-linear classifiers on z to
predict species. Refinement.md requires BOTH linear AND non-linear probes;
the in-training adversary only covers the GRL-trained classifier, which can
under-estimate residual species structure in z (cf. iter123 audit, where
in-training adv_acc=0.765 but post-hoc MLP-64-32 reaches 0.990 on iter101 z).

Usage
-----
    python training/z_leakage_probe.py --run-name xds-4sp-iter123-...

Reads ``runs/<name>/final_state.npz`` (loads `z_mean` and `species_labels`
when present, else loads the species column from the supplied --adata path).

Writes ``runs/<name>/z_leakage_probe.json`` with linear / kNN / MLP accuracies
and 3-fold cross-validated standard errors.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "training" / "runs"


def fit_probes(z: np.ndarray, species: np.ndarray, seed: int = 42, n_subsample: int = 8000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score

    rng = np.random.RandomState(seed)
    n = z.shape[0]
    if n > n_subsample:
        idx = rng.choice(n, size=n_subsample, replace=False)
        z = z[idx]
        species = species[idx]

    n_classes = int(np.unique(species).size)
    chance = 1.0 / n_classes

    probes = [
        (LogisticRegression(max_iter=300), "linear_LR"),
        (KNeighborsClassifier(n_neighbors=5), "kNN5"),
        (MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=seed), "MLP_32"),
        (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed), "MLP_64_32"),
    ]
    out = {"chance": chance, "n_classes": n_classes, "n_cells_used": int(z.shape[0])}
    for clf, name in probes:
        scores = cross_val_score(clf, z, species, cv=3, n_jobs=-1)
        out[name] = {
            "acc_mean": float(scores.mean()),
            "acc_std": float(scores.std()),
            "excess_over_chance": float(scores.mean() - chance),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--adata", default=None,
                    help="Optional path to adata for species labels if final_state.npz lacks them.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-subsample", type=int, default=8000)
    args = ap.parse_args()

    run_dir = RUNS_ROOT / args.run_name
    fs = np.load(run_dir / "final_state.npz")
    z = np.asarray(fs["z_mean"])

    if "species_labels" in fs.files:
        species = np.asarray(fs["species_labels"]).astype(str)
    else:
        if args.adata is None:
            raise SystemExit("species_labels not in final_state.npz; pass --adata to load species column.")
        import anndata as ad
        a = ad.read_h5ad(args.adata)
        species = np.asarray(a.obs["species"].values).astype(str)

    if z.shape[0] != species.shape[0]:
        raise SystemExit(f"z ({z.shape}) and species ({species.shape}) length mismatch")

    out = fit_probes(z, species, seed=args.seed, n_subsample=args.n_subsample)
    out["run_name"] = args.run_name

    out_path = run_dir / "z_leakage_probe.json"
    out_path.write_text(json.dumps(out, indent=2))

    # Print headline
    print(f"# z-leakage probes — {args.run_name}")
    print(f"  chance: {out['chance']:.3f}, n_classes: {out['n_classes']}, n_cells_used: {out['n_cells_used']}")
    for k in ["linear_LR", "kNN5", "MLP_32", "MLP_64_32"]:
        v = out[k]
        print(f"  {k}: acc={v['acc_mean']:.3f} ± {v['acc_std']:.3f}  (excess={v['excess_over_chance']:+.3f})")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
