"""T11.4: Van Zyl integration benchmark, a separate row excluded from the 7-cohort mean.

For every usable dataset (``vz.usable_datasets``): Chimera (the selected s42
grid run; every seed in D beside it) against the canonical baseline panel (PCA, Harmony, scVI with
species as batch, scVI / scANVI / sysVI with the sample as batch; ``baselines.tsv``), all scored with
``training/compare_vs_scvi.py::compute_integration_metrics`` unchanged (k = 15, seed 42, labels
``cell_type_harmonised``). Composite = the notebooks/07_baselines_master formula: six metrics
direction-corrected, min-max across the models of the dataset, 0.4 mixing + 0.6 biology.
Chance references (majority-class rate, knn_mixing_entropy under random mixing) are reported beside.
SATURN and CAMEX are not run on Van Zyl (their adapters, T5.2, are bound to the CAMEX raw files).

    PYTHONPATH=<chimera repo> python revision/tasks/T11_4_vz_integration/score.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from revision.tasks.T11_5_vz_genes import vz
from training.compare_vs_scvi import compute_integration_metrics

sys.path.insert(0, str(vz.ROOT / "revision" / "tasks" / "T11_2_recipe"))
from select_recipe import BATCH_KEYS, BIO_KEYS, correct, minmax  # noqa: E402

OUT = vz.OUT / "integration"
B = vz.RUNS
BASELINES = {  # model -> cache dir relative to training/runs, given the h5ad stem
    "PCA": "baselines_pca/{stem}",
    "Harmony": "baselines_harmony/{stem}-s42-bk_batch",
    "scVI-sp": "baselines_scvi/{stem}-s42-e100-nl10",
    "scVI-donor": "baselines_scvi/{stem}-s42-e100-nl10-bk_batch",
    "scANVI": "baselines_scanvi/{stem}-s42-bk_batch",
    "sysVI": "baselines_sysvi/{stem}-s42-bk_batch",
}
METRICS = BATCH_KEYS + BIO_KEYS


def score(z: np.ndarray, obs: pd.DataFrame) -> dict:
    m = compute_integration_metrics(z, obs["species"].astype(str).to_numpy(), obs[vz.LABEL_KEY].astype(str).to_numpy(),
                                    k=15, seed=42)
    return {k: m["global"][k] for k in METRICS} | {"n_shared_celltypes": m["global"]["n_shared_celltypes"]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, seeds_rows = [], []
    for ds in vz.usable_datasets():
        a = vz.read_h5ad(ds, backed=True)
        obs = a.obs.copy()
        stem = vz.DATASETS[ds]["h5ad"].stem
        ref = vz.integration_reference(obs)
        runs = vz.seeds_in_D(ds)
        for s, run in sorted(runs.items()):
            fs = np.load(vz.RUNS / run / "final_state.npz")
            summ = json.loads((vz.RUNS / run / "summary.json").read_text())
            r = score(np.asarray(fs["z_mean"]), obs) | dict(dataset=ds, seed=s, run=run,
                                                            adv_acc_final=summ["key_metrics"].get("adv_acc_final"))
            seeds_rows.append(r)
            if s == 42:
                rows.append(dict(dataset=ds, model="Chimera", source=run) | {k: r[k] for k in METRICS + ["n_shared_celltypes"]})
        for model, rel in BASELINES.items():
            d = B / rel.format(stem=stem)
            zf = d / "z_mean.npy"
            if not zf.exists():
                rows.append(dict(dataset=ds, model=model, source=str(d.relative_to(vz.ROOT)), note="latent missing"))
                continue
            z = np.load(zf)
            if z.shape[0] != len(obs):
                raise AssertionError(f"{ds} {model}: {z.shape[0]} rows vs {len(obs)} cells")
            rows.append(dict(dataset=ds, model=model, source=str(d.relative_to(vz.ROOT))) | score(z, obs))
        for k, v in ref.items():
            for r in rows:
                if r["dataset"] == ds:
                    r[f"ref_{k}"] = v
    tab = pd.DataFrame(rows)
    out = []
    for ds, t in tab.groupby("dataset", sort=False):
        t = t.copy()
        ok = t[METRICS].notna().all(1)
        corr = pd.DataFrame({k: correct(k, t.loc[ok, k].astype(float)) for k in METRICS})
        norm = corr.apply(minmax)
        t.loc[ok, "mixing"] = norm[BATCH_KEYS].mean(1)
        t.loc[ok, "biology"] = norm[BIO_KEYS].mean(1)
        if ok.all():  # min-max composites compare only on the full panel of 7 models
            t.loc[ok, "composite"] = 0.4 * t.loc[ok, "mixing"] + 0.6 * t.loc[ok, "biology"]
            t.loc[ok, "rank"] = t.loc[ok, "composite"].rank(ascending=False, method="min")
        else:
            t[["mixing", "biology"]] = np.nan
            t["composite"], t["rank"] = np.nan, np.nan
        out.append(t)
    tab = pd.concat(out)
    tab.to_csv(OUT / "integration_table.csv", index=False)
    sd = pd.DataFrame(seeds_rows)
    sd.to_csv(OUT / "chimera_seeds.csv", index=False)
    summ = sd.groupby("dataset")[METRICS + ["adv_acc_final"]].agg(["mean", "std", "count"])
    summ.to_csv(OUT / "chimera_seed_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab[["dataset", "model"] + METRICS + ["composite", "rank"]].round(3).to_string(index=False))
        print(summ.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
