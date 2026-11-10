"""Per-run report for the Van Zyl grid and seeds: Delta-alpha sign profile and species mixing.

Descriptive only, and never an input to the recipe choice: it refuses to run until
``select_recipe.py`` has written its selection, which is run once every grid row has ended
(finished, or cancelled after a non-finite loss). Writes
``results/analysis/T11_2/run_profile.tsv``.

For each finished run, against the training prior (clamped where the recipe clamps) and
the raw prior: fractions positive and negative, fractions beyond +0.05 and -0.05, SD,
median of Delta-alpha; plus knn_mixing_entropy (k=15, seed 42, as in selection), the same
entropy over mouse cells only (most mouse cells have no partner class for conditional MMD),
final adversary accuracy, the first snapshot with a non-finite alpha and Gate G-V. A seed is flagged poorly mixed if its knn_mixing_entropy
is below FLAG_RATIO times the selected run's (bar fixed before any seed was trained;
Liver H+Mouse integrated seeds had 0.25-0.50, poorly mixed 0.06-0.19, results of T3_3_consensus).
Flagged seeds are reported separately; the primary D stays the mean over seeds (pre-specified analysis plan, section 2).

    python analysis/tasks/T11_2_recipe/run_profile.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from select_recipe import ADATA, LABEL_KEY, OUT, RUNS, SEEDS_T113, TASK, gate  # noqa: E402
from training.compare_vs_scvi import compute_integration_metrics  # noqa: E402

FLAG_RATIO = 0.5
TAIL = 0.05
MOUSE = "mouse"


def sign_profile(delta: np.ndarray, tag: str) -> dict:
    return {
        f"frac_pos_{tag}": float((delta > 0).mean()),
        f"frac_neg_{tag}": float((delta < 0).mean()),
        f"frac_gt_p05_{tag}": float((delta > TAIL).mean()),
        f"frac_lt_m05_{tag}": float((delta < -TAIL).mean()),
        f"sd_{tag}": float(delta.std()),
        f"median_{tag}": float(np.median(delta)),
    }


def species_entropy(z: np.ndarray, species: np.ndarray, which: str, k: int = 15) -> float:
    """Mean normalized species entropy of the k nearest neighbors, over cells of one species."""
    from sklearn.neighbors import NearestNeighbors

    codes, names = pd.factorize(species, sort=True)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z[species == which])[1][:, 1:]
    p = np.stack([(codes[idx] == c).mean(axis=1) for c in range(len(names))], axis=1)
    h = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1) / np.log(len(names))
    return float(h.mean())


def first_nan_snapshot(run_dir: Path) -> str:
    for f in sorted((run_dir / "alpha_snapshots").glob("*.npz")):
        if not np.isfinite(np.load(f)["alpha"]).all():
            return f.stem
    return ""


def main() -> int:
    import anndata as ad

    grid = pd.read_csv(TASK / "grid.tsv", sep="\t")
    sel_path = OUT / "selection.json"
    if not sel_path.exists():
        raise SystemExit("run select_recipe.py first")
    sel = json.loads(sel_path.read_text())  # written only once every grid row has ended
    selected = (sel.get("selected") or {}).get("run_name")

    a = ad.read_h5ad(ADATA, backed="r")
    species = a.obs["species"].astype(str).to_numpy()
    labels = a.obs[LABEL_KEY].astype(str).to_numpy()
    raw_var = np.asarray(a.var["conservation_prior"], dtype=np.float64)

    names = list(grid.run_name) + [f"rev-vz-s{s}" for s in SEEDS_T113]
    rows = []
    for name in names:
        d = RUNS / name
        if not (d / "summary.json").exists() or not (d / "final_state.npz").exists():
            continue
        g = gate(d)
        fs = np.load(d / "final_state.npz", allow_pickle=True)
        alpha = np.asarray(fs["alpha"], dtype=np.float64)
        prior = np.asarray(fs["prior"], dtype=np.float64)
        raw_f = d / "conservation_prior_raw.npy"
        raw = np.load(raw_f).astype(np.float64) if raw_f.exists() else raw_var
        z = np.asarray(fs["z_mean"])
        finite = bool(np.isfinite(z).all() and np.isfinite(alpha).all())
        m = compute_integration_metrics(z, species, labels, k=15, seed=42)["global"] if finite else {}
        summ = json.loads((d / "summary.json").read_text())
        rows.append({
            "run_name": name,
            "selected": name == selected,
            "pass_GV": g["pass_GV"],
            "first_nan_snapshot": first_nan_snapshot(d),
            "last_epoch": g.get("last_epoch"),
            **sign_profile(alpha - prior, "train"),
            **sign_profile(alpha - raw, "raw"),
            "knn_mixing_entropy": m.get("knn_mixing_entropy", np.nan),
            "mixing_entropy_mouse": species_entropy(z, species, MOUSE) if finite else np.nan,
            "knn_species_purity": m.get("knn_species_purity", np.nan),
            "adv_acc_final": summ["key_metrics"].get("adv_acc_final"),
        })
    tab = pd.DataFrame(rows)
    ref = tab.loc[tab["selected"], "knn_mixing_entropy"]
    tab["flag_poorly_mixed"] = False
    if len(ref):
        is_seed = tab["run_name"].str.fullmatch(r"rev-vz-s\d+")
        tab.loc[is_seed, "flag_poorly_mixed"] = tab.loc[is_seed, "knn_mixing_entropy"] < FLAG_RATIO * float(ref.iloc[0])
    tab.to_csv(OUT / "run_profile.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        print(tab.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
