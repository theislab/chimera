"""Van Zyl datasets for T11.4-T11.7: paths, seed runs, seed health and integration references.

Registry: ``DATASETS`` below plus optional rows in ``datasets_extra.tsv`` (columns dataset, h5ad,
seed_map, label), so a retuned dataset is added with one row and one more pass of the scripts.

Seed rule for new cohorts (amendment log, latest rows): a seed enters D if it finished, has no NaN, no
latent collapse (10/10 active z dims) and both Delta-alpha tails (not frozen), and saturation is at
most 10% (above 5% its saturated genes are flagged). Other seeds are discarded and not analysed. A
dataset needs 2 or more such seeds.
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

SAT_MAX_SEED = 0.10  # saturation above 5% and up to 10% keeps the seed, genes flagged
MIN_SEEDS = 2
DATASETS = {
    "full": dict(h5ad=VZ / "van_zyl_unified.h5ad", seed_map=ROOT / "revision/tasks/T11_3_vz_seeds/seed_map.tsv",
                 label="5 species"),
    "hmouse": dict(h5ad=VZ / "van_zyl_hmouse.h5ad", seed_map=ROOT / "revision/tasks/T11_2b_vz_subsets/seed_map_hmouse.tsv",
                   label="human + mouse"),
    "hmmul": dict(h5ad=VZ / "van_zyl_hmmul.h5ad", seed_map=ROOT / "revision/tasks/T11_2b_vz_subsets/seed_map_hmmul.tsv",
                  label="human + M. mulatta"),
    "primates": dict(h5ad=VZ / "van_zyl_primates.h5ad", seed_map=ROOT / "revision/tasks/T11_2b_vz_subsets/seed_map_primates.tsv",
                     label="primates"),
}
_extra = Path(__file__).resolve().parent / "datasets_extra.tsv"
if _extra.exists():
    for _r in pd.read_csv(_extra, sep="\t").itertuples():
        DATASETS[_r.dataset] = dict(h5ad=ROOT / _r.h5ad, seed_map=ROOT / _r.seed_map, label=_r.label)

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
        rows.append(dict(dataset=ds, seed=s, run_name=run, first_nan_snapshot=first_nan_snapshot(run) if d.exists() else "",
                         **{k: g.get(k) for k in (
                             "finished", "no_nan", "sat_high", "sat_low", "n_active_z", "z_var_min",
                             "tail_up", "tail_dn", "tails_ok", "pass_GV", "last_epoch")}))
    t = pd.DataFrame(rows)
    ok = t["finished"].fillna(False).astype(bool) & t["no_nan"].fillna(False).astype(bool) & (t["n_active_z"] == 10)
    ok &= (t["sat_high"] <= SAT_MAX_SEED) & (t["sat_low"] <= SAT_MAX_SEED) & t["tails_ok"].fillna(False).astype(bool)
    t["enters_D"] = ok
    t["saturation_flagged"] = ok & ((t["sat_high"] > 0.05) | (t["sat_low"] > 0.05))
    return t


def seeds_in_D(ds: str) -> dict[int, str]:
    t = seed_health(ds)
    return {int(r.seed): r.run_name for r in t.itertuples() if r.enters_D}


def usable_datasets() -> list[str]:
    """Datasets in the registry with at least MIN_SEEDS seeds that pass health."""
    return [ds for ds in DATASETS if len(seeds_in_D(ds)) >= MIN_SEEDS]


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
