"""Score Track I: new cohorts on the 10-model panel, same scoring as T5.5; extended table; checks that no old value moved. Writes results/revision/B2/integration/:
master_table_extended.csv (the seven rows of master_table_ext.csv byte-identical in value, then the new cohorts), sensitivity_sample_key.csv, mean_scores_both.csv, checks.json, tidy.parquet (overview rows, Track I).
Re-runnable as latents arrive; a cohort with a missing model is listed with its available models and a blank composite.    python -m revision.tasks.B2_benchmark.integration.score
"""

from __future__ import annotations

import json
import sys

import anndata as ad
import numpy as np
import pandas as pd

from revision.tasks.B2_benchmark.integration import rules as RI

REPO = RI.REPO
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "notebooks"))
sys.path.insert(0, str(REPO / "revision/tasks/T5_5_benchmark"))
import score_external as SE  # noqa: E402

M7 = REPO / "notebooks/cache/baselines_master_ext/master_table_ext.csv"
MS7 = REPO / "notebooks/cache/baselines_master_ext/mean_scores.csv"
CACHE = RI.OUT / "cache"
METRICS = SE.METRICS


def metrics_for(c: str, model: str, z: np.ndarray, obs: pd.DataFrame, tag: str) -> dict:
    f = CACHE / f"{c}__{model}__{tag}.json"
    if f.exists():
        return json.loads(f.read_text())
    m = SE.compute_integration_metrics(z, obs["species"].to_numpy(), obs["label"].to_numpy(), k=SE.KNN_K, seed=SE.SEED)["global"]
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {k: float(m.get(k, np.nan)) for k in METRICS} | {"_n_latent": int(z.shape[1])}
    f.write_text(json.dumps(out, indent=1))
    return out


def load_tool(c: str, m: str, order: pd.Index):
    d = RI.OUT / "latents" / c / m
    if not (d / "z_mean.npy").exists():
        return None
    names = pd.read_csv(d / "obs_names.csv", header=None)[0].astype(str)
    z = np.load(d / "z_mean.npy")
    pos = pd.Index(names).get_indexer(order)
    if (pos < 0).any() or len(names) != len(order):
        raise SystemExit(f"{c} {m}: cells differ from the object")
    return z[pos]


def cohort_rows(c: str, tag: str) -> tuple[pd.DataFrame, dict]:
    d = RI.COHORTS[c]
    a = ad.read_h5ad(REPO / d["h5ad"], backed="r")
    obs = pd.DataFrame(dict(species=a.obs["species"].astype(str).to_numpy(), label=a.obs[d["label"]].astype(str).to_numpy()), index=a.obs_names.astype(str))
    a.file.close()
    res, src = {}, {}
    st = np.load(REPO / "training/runs" / d["run"] / "final_state.npz")
    res["Chimera"] = metrics_for(c, "Chimera", np.asarray(st["z_mean"]), obs, "x")
    src["Chimera"] = d["run"]
    if d.get("reuse_t11"):
        t = pd.read_csv(REPO / "results/revision/T11/integration/integration_table.csv")
        t = t[t.dataset == "hmouse"] if (t.dataset == "hmouse").any() else t[t.dataset.str.contains("mouse")]
        for r in t.itertuples():
            if r.model == "Chimera":
                continue
            res[r.model] = {k: float(getattr(r, k)) for k in METRICS}
            src[r.model] = f"T11.4 {r.source}"
    else:
        for m, p in RI.latent_paths(c, tag).items():
            if (p / "z_mean.npy").exists():
                z = np.load(p / "z_mean.npy")
                if z.shape[0] != len(obs):
                    raise SystemExit(f"{c} {m}: {z.shape[0]} vs {len(obs)} cells")
                res[m] = metrics_for(c, m, z, obs, tag)
                src[m] = str(p.relative_to(REPO))
    for m in ("SATURN", "SATURN-Leiden", "CAMEX"):
        z = load_tool(c, m, obs.index)
        if z is not None:
            res[m] = metrics_for(c, m, z, obs, "x")
            src[m] = f"B2 latents; 1:1 adapter"
    raw = pd.DataFrame({m: {k: float(res[m].get(k, np.nan)) for k in METRICS} for m in RI.MODELS_ORDER if m in res}).T
    sc = SE.score_table(raw).rename_axis("model").reset_index()
    if len(raw) < len(RI.MODELS_ORDER):
        sc[["total_score", "mixing", "biology"]] = sc[["total_score", "mixing", "biology"]].where(False)  # min-max composites compare only on the full panel
    sc.insert(0, "dataset", c)
    sc.insert(1, "dataset_label", d["dataset_label"])
    sc.insert(2, "tissue", d["tissue"])
    sc.insert(4, "n_cells", len(obs))
    sc["source"] = sc["model"].map(src)
    sc["batch_tag"] = tag
    return sc, dict(missing=[m for m in RI.MODELS_ORDER if m not in res])


def main():
    RI.OUT.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(M7, float_precision="round_trip")
    new, sens, miss = [], [], {}
    for c, d in RI.COHORTS.items():
        t, info = cohort_rows(c, "cfg")
        new.append(t)
        miss[c] = info["missing"]
        if d["sample_key"]:
            sens.append(cohort_rows(c, "sample")[0])
    N = pd.concat(new, ignore_index=True)
    T = pd.concat([old.assign(batch_tag="original", source="T5.5"), N], ignore_index=True)
    T.to_csv(RI.OUT / "master_table_extended.csv", index=False)
    if sens:
        pd.concat(sens, ignore_index=True).to_csv(RI.OUT / "sensitivity_sample_key.csv", index=False)
    # no old value moved: scaling is per cohort, the seven rows are read from the frozen table and never rescaled
    again = pd.read_csv(RI.OUT / "master_table_extended.csv", float_precision="round_trip")
    o2 = again[again.dataset.isin(old.dataset.unique())]
    cols = [k for k in old.columns if k in o2 and old[k].dtype != object]
    same = bool(np.allclose(old[cols].to_numpy(dtype=float), o2[cols].to_numpy(dtype=float), equal_nan=True, rtol=0, atol=1e-15))
    rescale = []
    for ds, g in old.groupby("dataset"):
        raw = g.set_index("model")[[f"raw__{m}" for m in METRICS]].rename(columns=lambda s: s.replace("raw__", ""))
        rescale.append(float((SE.score_table(raw)["total_score"] - g.set_index("model")["total_score"]).abs().max()))
    ms = pd.read_csv(MS7).set_index("model")[["mean_score_all"]].rename(columns={"mean_score_all": "mean_seven_cohorts_as_reported"})
    comp = T[T.total_score.notna()]
    n_ds = comp.groupby("model").dataset.nunique()
    ms["mean_extended"] = comp.groupby("model").total_score.mean()
    ms["n_cohorts_extended"] = n_ds
    ms.sort_values("mean_extended", ascending=False).to_csv(RI.OUT / "mean_scores_both.csv")
    checks = dict(old_rows_identical=same, old_composite_rebuilt_max_abs=max(rescale), scaling="per cohort: min-max across the models of each cohort; adding a cohort cannot change another cohort's value",
                  missing_models=miss)
    (RI.OUT / "checks.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1), flush=True)
    print(ms.round(3).to_string(), flush=True)
    if not same or max(rescale) > 1e-12:
        raise SystemExit("an old value moved")
    # tidy rows
    rows = []
    for r in T.itertuples():
        for col, metric in (("mixing", "mixing_score"), ("biology", "conservation_score"), ("total_score", "composite")):
            v = getattr(r, col)
            if pd.notna(v):
                rows.append(dict(track="I", task=f"I|{metric}|{r.dataset}", setting=r.dataset, cohort=r.dataset, family=metric, method=f"I:{r.model}", metric=metric, value=float(v), ci_lo=np.nan, ci_hi=np.nan, n=int(r.n_cells), rank_direction="higher", gold="integration", flag=str(getattr(r, "source", ""))))
    pd.DataFrame(rows).to_parquet(RI.OUT / "tidy.parquet")


if __name__ == "__main__":
    main()
