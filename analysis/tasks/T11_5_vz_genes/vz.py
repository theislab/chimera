"""Van Zyl datasets for T11_4_vz_integration-T11_7_vz_figure: paths, seed runs, seed health and integration references.

Registry: ``DATASETS`` below plus optional rows in ``datasets_extra.tsv`` (columns dataset, h5ad,
seed_map, label, beside), so a retuned dataset is added with one row and one more pass of the
scripts; ``beside`` marks a dataset reported next to the main ones (not in the figure).

Seed rule for the further cohorts (amendment log of the pre-specified analysis plan): a seed enters D if it finished, has no NaN, no
latent collapse (10/10 active z dims) and both Delta-alpha tails against the raw prior (not frozen),
and saturation is at most 10% (above 5% its saturated genes are flagged). Other seeds are discarded and not analyzed. A
dataset needs 2 or more such seeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common import gene_ids

ROOT = gene_ids.REPO_ROOT
VZ = ROOT / "data" / "analysis" / "van_zyl"
RUNS = ROOT / "training" / "runs"
OUT = ROOT / "results" / "analysis" / "T11"
TABLE_COHORT = "van_zyl"  # every subset keeps the build-3 ortho_N names, so the van_zyl table applies
LABEL_KEY = "cell_type_harmonised"  # integration labels, as in recipe selection
CLASS_KEY = "cell_ontology_class"  # conditional-MMD classes; pseudobulk groups

SAT_MAX_SEED = 0.10  # saturation above 5% and up to 10% keeps the seed, genes flagged
MIN_SEEDS = 2
DATASETS = {
    "full": dict(h5ad=VZ / "van_zyl_unified.h5ad", seed_map=ROOT / "analysis/tasks/T11_3_vz_seeds/seed_map.tsv",
                 label="5 species"),
    "hmouse": dict(h5ad=VZ / "van_zyl_hmouse.h5ad", seed_map=ROOT / "analysis/tasks/T11_2b_vz_subsets/seed_map_hmouse.tsv",
                   label="human + mouse"),
    "hmmul": dict(h5ad=VZ / "van_zyl_hmmul.h5ad", seed_map=ROOT / "analysis/tasks/T11_2b_vz_subsets/seed_map_hmmul.tsv",
                  label="human + M. mulatta"),
    "primates": dict(h5ad=VZ / "van_zyl_primates.h5ad", seed_map=ROOT / "analysis/tasks/T11_2b_vz_subsets/seed_map_primates.tsv",
                     label="primates"),
}
_extra = Path(__file__).resolve().parent / "datasets_extra.tsv"
if _extra.exists():
    for _r in pd.read_csv(_extra, sep="\t").itertuples():
        DATASETS[_r.dataset] = dict(h5ad=ROOT / _r.h5ad, seed_map=ROOT / _r.seed_map, label=_r.label, beside=bool(_r.beside))
_winner = ROOT / "analysis/tasks/T11_2c_count_rule/seed_map_winner.tsv"  # written by select_count.py
if _winner.exists():
    DATASETS["full"]["seed_map"] = _winner

sys.path.insert(0, str(ROOT / "analysis" / "tasks" / "T11_2_recipe"))
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
    ok = t["finished"].astype("boolean").fillna(False).astype(bool) & t["no_nan"].astype("boolean").fillna(False).astype(bool) & (t["n_active_z"] == 10)
    fin = ok.to_numpy()
    tails = [raw_tails(r.run_name) if f else (np.nan, np.nan) for r, f in zip(t.itertuples(), fin)]
    t["tail_up_raw"], t["tail_dn_raw"] = [x[0] for x in tails], [x[1] for x in tails]
    ok &= (t["sat_high"] <= SAT_MAX_SEED) & (t["sat_low"] <= SAT_MAX_SEED) & (t["tail_up_raw"] >= 0.02) & (t["tail_dn_raw"] >= 0.02)
    t["enters_D"] = ok
    t["saturation_flagged"] = ok & ((t["sat_high"] > 0.05) | (t["sat_low"] > 0.05))
    return t


def seeds_in_D(ds: str) -> dict[int, str]:
    t = seed_health(ds)
    return {int(r.seed): r.run_name for r in t.itertuples() if r.enters_D}


def raw_tails(run: str) -> tuple[float, float]:
    """Fractions of genes with alpha - raw prior above +0.05 and below -0.05."""
    d = RUNS / run
    with np.load(d / "final_state.npz") as z:
        alpha, prior = z["alpha"].astype(float), z["prior"].astype(float)
    f = d / "conservation_prior_raw.npy"
    da = alpha - (np.load(f).astype(float) if f.exists() else prior)
    return float((da > 0.05).mean()), float((da < -0.05).mean())


def main_datasets() -> list[str]:
    return [ds for ds in usable_datasets() if not DATASETS[ds].get("beside")]


def usable_datasets() -> list[str]:
    """Datasets in the registry with at least MIN_SEEDS seeds that pass health."""
    return [ds for ds in DATASETS if len(seeds_in_D(ds)) >= MIN_SEEDS]


def integration_reference(obs: pd.DataFrame, k: int = 15, n: int = 20000, seed: int = 0) -> dict:
    """Chance references: majority-class rate (adversary) and the expected knn_mixing_entropy
    if the k neighbors were drawn at random from the species proportions."""
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
