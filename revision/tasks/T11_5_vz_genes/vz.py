"""Van Zyl datasets for T11.4-T11.7: paths, seed runs, seed health and integration references.

Datasets with a selected recipe (T11.2, T11.2b): the full five-species cohort and the subsets
human + mouse, human + M. mulatta and primates. Human + M. fascicularis and human + pig had no
recipe passing Gate G-V (failed training) and are not analysed.

Seed rule for new cohorts (amendment, private a82eaad): a seed failing G-V does not enter D or
S_ema and is reported with its health row; sensitivity D uses every seed with a finite alpha; a
dataset with fewer than 2 passing seeds is reported descriptively, without reliability or ceiling;
healthy but poorly mixed seeds stay in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from revision.common import gene_ids

ROOT = gene_ids.REPO_ROOT
VZ = ROOT / "data" / "revision" / "van_zyl"
RUNS = ROOT / "training" / "runs"
OUT = ROOT / "results" / "revision" / "T11"
TABLE_COHORT = "van_zyl"  # every subset keeps the build-3 ortho_N names, so the van_zyl table applies
LABEL_KEY = "cell_type_harmonised"  # integration labels, as in recipe selection
CLASS_KEY = "cell_ontology_class"  # conditional-MMD classes; pseudobulk groups

DATASETS = {
    "full": dict(h5ad=VZ / "van_zyl_unified.h5ad", seed_map=ROOT / "revision/tasks/T11_3_vz_seeds/seed_map.tsv",
                 label="Van Zyl, 5 species"),
    "hmouse": dict(h5ad=VZ / "van_zyl_hmouse.h5ad", seed_map=ROOT / "revision/tasks/T11_2b_vz_subsets/seed_map_hmouse.tsv",
                   label="Van Zyl, human + mouse"),
    "hmmul": dict(h5ad=VZ / "van_zyl_hmmul.h5ad", seed_map=ROOT / "revision/tasks/T11_2b_vz_subsets/seed_map_hmmul.tsv",
                  label="Van Zyl, human + M. mulatta"),
    "primates": dict(h5ad=VZ / "van_zyl_primates.h5ad", seed_map=ROOT / "revision/tasks/T11_2b_vz_subsets/seed_map_primates.tsv",
                     label="Van Zyl, primates"),
}
FAILED_TRAINING = {"hmfas": "human + M. fascicularis", "hpig": "human + pig"}

sys.path.insert(0, str(ROOT / "revision" / "tasks" / "T11_2_recipe"))
from select_recipe import gate  # noqa: E402  (Gate G-V, unchanged)


def seed_runs(ds: str) -> dict[int, str]:
    """seed -> run name from the committed seed map (s42 is the selected grid run)."""
    m = pd.read_csv(DATASETS[ds]["seed_map"], sep="\t")
    return {int(r.seed): str(r.source).split()[0].replace("training/runs/", "") for r in m.itertuples()}


def read_h5ad(ds: str, backed: bool = False):
    import anndata as ad

    return ad.read_h5ad(DATASETS[ds]["h5ad"], backed="r" if backed else None)


def first_nan_snapshot(run: str) -> str:
    for f in sorted((RUNS / run / "alpha_snapshots").glob("*.npz")):
        if not np.isfinite(np.load(f)["alpha"]).all():
            return f.stem
    return ""


def seed_health(ds: str) -> pd.DataFrame:
    rows = []
    for s, run in sorted(seed_runs(ds).items()):
        d = RUNS / run
        g = gate(d)
        finite = False
        if g["finished"]:
            with np.load(d / "final_state.npz") as z:
                finite = bool(np.isfinite(z["alpha"]).all())
        rows.append(dict(dataset=ds, seed=s, run_name=run, first_nan_snapshot=first_nan_snapshot(run) if d.exists() else "",
                         finite_alpha=finite, **{k: g.get(k) for k in (
                             "finished", "no_nan", "sat_high", "sat_low", "n_active_z", "z_var_min",
                             "tail_up", "tail_dn", "pass_GV", "last_epoch")}))
    t = pd.DataFrame(rows)
    t["enters_D"] = t["pass_GV"].fillna(False).astype(bool)
    t["enters_D_sensitivity"] = t["finite_alpha"].astype(bool)
    return t


def seeds_in_D(ds: str, sensitivity: bool = False) -> dict[int, str]:
    t = seed_health(ds)
    col = "enters_D_sensitivity" if sensitivity else "enters_D"
    return {int(r.seed): r.run_name for r in t.itertuples() if getattr(r, col)}


def integration_reference(obs: pd.DataFrame, k: int = 15, n: int = 20000, seed: int = 0) -> dict:
    """Chance references: majority-class rate (adversary) and the expected knn_mixing_entropy
    if the k neighbours were drawn at random from the species proportions."""
    p = obs["species"].astype(str).value_counts(normalize=True).sort_index().to_numpy()
    K = len(p)
    draws = np.random.default_rng(seed).multinomial(k, p, size=n) / k
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(draws > 0, draws * np.log(draws), 0).sum(1) / np.log(K)
    return dict(n_species=K, majority_rate=float(p.max()), one_over_K=1 / K, entropy_random=float(H.mean()))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)
