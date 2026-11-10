"""Integration scorecard of SATURN, SATURN-Leiden and CAMEX beside the seven-method panel.

Imports ``compute_integration_metrics`` unmodified from ``training/compare_vs_scvi.py``, with the
same k=15, seed=42, ``species`` and ``cell_type`` as the master table. The 49 cached JSONs are read,
never rewritten. Direction correction, per-dataset min-max and the 0.4/0.6 composite are the
builder's (``notebooks/_baselines_master_builder.py``, sections C.1 and C.2).

Checks that stop the run:
  * raw columns of the original seven equal ``master_table.csv`` exactly;
  * the 7-model composite rebuilt here equals the CSV within 1e-12;
  * one cached latent per dataset recomputed in this env reproduces its cached raw metrics.

Panel: 10 models, composite recomputed with no frozen bounds. SATURN enters L2-normalized, its raw
latent is scored for the Supplement only. CAMEX seed 42 enters the table; seeds 42-44 give the SD.

Outputs: notebooks/cache/baselines_master_ext/{<ds>__<model>.json, master_table_ext.csv,
mean_scores.csv, liver_only_ranking.csv, supplement_saturn_raw_latent.csv, supplement_camex_seeds.csv,
checks.json}; notebooks/figures/baselines_master_ext/*.pdf. ``master_table.csv`` is not touched.

    PYTHONPATH=. mamba run --no-capture-output -n chimera_env python analysis/tasks/T5_5_benchmark/score_external.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "notebooks"))
from compare_vs_scvi import compute_integration_metrics  # noqa: E402
import _figure_design as fd  # noqa: E402

KNN_K, SEED = 15, 42
RUNS = REPO / "training" / "runs"
DATA = REPO / "data" / "processed"
CACHE = REPO / "notebooks" / "cache" / "baselines_master"
OUT = REPO / "notebooks" / "cache" / "baselines_master_ext"
FIG = REPO / "notebooks" / "figures" / "baselines_master_ext"
DATASETS = {
    "panc_hmonkey": ("camex_human_monkey", "liver H+Monkey", "liver"),
    "panc_hmouse": ("camex_human_mouse", "liver H+Mouse", "liver"),
    "panc_hzeb": ("camex_human_zebrafish", "liver H+Zeb", "liver"),
    "panc_4sp": ("camex_pancreas_unified", "liver 4-sp", "liver"),
    "test_hmac": ("camex_testis_human_macaque", "testis H+Mac", "testis"),
    "test_hmouse": ("camex_testis_human_mouse", "testis H+Mouse", "testis"),
    "test_3sp": ("camex_testis_unified", "testis 3-sp", "testis"),
}
ORIG = ["PCA", "scVI-sp", "scVI-donor", "scANVI", "Harmony", "sysVI", "Chimera"]
NEW = {"SATURN": "baselines_saturn", "SATURN-Leiden": "baselines_saturn_leiden", "CAMEX": "baselines_camex"}
MODELS = ORIG + list(NEW)
# Figure order: label use in training, then method; Chimera's label use differs by tissue.
TIERS = [("no labels", ["PCA", "Harmony", "scVI-sp", "scVI-donor", "sysVI", "CAMEX"]),
         ("within-species\nclusters", ["SATURN-Leiden"]),
         ("cell-type labels", ["scANVI", "SATURN"]),
         ("liver none,\ntestis labels", ["Chimera"])]
METRICS = ["knn_species_purity", "knn_mixing_entropy", "sil_species", "knn_celltype_purity", "sil_celltype",
           "label_transfer_acc"]
BATCH = METRICS[:3]
BIO = METRICS[3:]


def direction_correct(s: pd.Series, metric: str) -> pd.Series:
    if metric == "knn_species_purity":
        return 1.0 - s
    if metric == "sil_species":
        return 1.0 - s.clip(lower=0.0)
    if metric == "sil_celltype":
        return (s + 1.0) / 2.0
    return s


def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return s.where(s.isna(), 0.5)
    return (s - mn) / (mx - mn)


def score_table(raw: pd.DataFrame) -> pd.DataFrame:
    """raw: models x metrics for one dataset. Same formulas as the builder."""
    corr = pd.DataFrame({m: direction_correct(raw[m], m) for m in METRICS})
    norm = pd.DataFrame({m: minmax(corr[m]) for m in METRICS})
    mix, bio = norm[BATCH].mean(1), norm[BIO].mean(1)
    wins = corr.gt(corr.loc["PCA"], axis=1).sum(1).astype(int)
    out = pd.DataFrame({f"raw__{m}": raw[m] for m in METRICS})
    for m in METRICS:
        out[f"norm__{m}"] = norm[m]
    out["mixing"], out["biology"] = mix, bio
    out["total_score"] = 0.4 * mix + 0.6 * bio
    out["wins_vs_pca"] = wins
    return out


def latent_dir(tool_dir: str, stem: str, seed: int = 42) -> Path:
    return RUNS / tool_dir / f"{stem}-s{seed}"


def scorecard(z, obs) -> dict:
    t0 = time.time()
    out = compute_integration_metrics(z, np.asarray(obs["species"].values), np.asarray(obs["cell_type"].values),
                                      k=KNN_K, seed=SEED)
    out["_runtime_sec"] = float(time.time() - t0)
    out["_z_shape"] = list(z.shape)
    return out


def cached(ds: str, model: str) -> dict:
    return json.loads((CACHE / f"{ds}__{model}.json").read_text())


def regression_check(ds: str, stem: str, obs) -> dict:
    """Harmony's stored latent, rescored in this env, against its cached metrics."""
    d = RUNS / "baselines_harmony"
    p = next(d.glob(f"{stem}-s42-bk_*"))
    new = scorecard(np.load(p / "z_mean.npy"), obs)["global"]
    old = cached(ds, "Harmony")["global"]
    return {m: float(abs(new[m] - old[m])) for m in METRICS}


def heatmap(M: pd.DataFrame, name: str, *, cmap, vmin, vmax, fmt, cbar, mark_max, dark_above=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(fd.DOUBLE_W, 0.34 * len(M) + 1.5), layout="constrained")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    im = ax.imshow(M.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    best = np.nanargmax(M.values, axis=1) if mark_max else np.full(len(M), -1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            col = "#222222" if (dark_above is None or v >= dark_above) else "#f5f5f5"
            ax.text(j, i, fmt.format(v) + ("*" if j == best[i] else ""), ha="center", va="center", fontsize=7,
                    color=col, fontweight="bold" if j == best[i] else "normal")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(M.index, fontsize=7)
    x = 0
    for lab, ms in TIERS:
        k = len(ms)
        ax.annotate("", xy=(x - 0.45, -0.62), xytext=(x + k - 0.55, -0.62), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#555555"))
        ax.text(x + (k - 1) / 2, -0.72, lab, ha="center", va="bottom", fontsize=6, color="#333333")
        if x:
            ax.axvline(x - 0.5, color="#ffffff", lw=1.5)
        x += k
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(cbar, fontsize=7)
    return fd.save_figure(fig, name, close=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fd.apply_design()
    fd.set_figure_dir(FIG)
    master = pd.read_csv(CACHE / "master_table.csv", float_precision="round_trip")  # exact comparison below
    for tool_dir in NEW.values():
        for ds, (stem, _, _) in DATASETS.items():
            if not (latent_dir(tool_dir, stem) / "z_mean.npy").exists():
                raise SystemExit(f"missing {tool_dir}/{stem}-s42")
    rows, raw_sat, cam, regress = [], [], [], {}
    t7 = []
    for ds, (stem, label, tissue) in DATASETS.items():
        a = ad.read_h5ad(DATA / f"{stem}.h5ad", backed="r")
        obs = a.obs[["species", "cell_type"]].copy()
        n = a.n_obs
        a.file.close()
        regress[ds] = regression_check(ds, stem, obs)
        res = {m: cached(ds, m) for m in ORIG}
        lat = {m: res[m]["_z_shape"][1] for m in ORIG}
        for m, tool_dir in NEW.items():
            d = latent_dir(tool_dir, stem)
            z = np.load(d / "z_mean.npy")
            cfg = json.loads((d / "config.json").read_text())
            if z.shape[0] != n or cfg["adata_stem"] != stem:
                raise SystemExit(f"{m}/{stem}: latent does not match the cohort")
            res[m] = scorecard(z, obs)
            res[m]["_latent"] = cfg["latent"]
            res[m]["_git_sha"] = cfg["git_sha"]
            (OUT / f"{ds}__{m}.json").write_text(json.dumps(res[m], default=str, indent=2))
            lat[m] = z.shape[1]
            print(ds, m, {k: round(res[m]["global"][k], 3) for k in METRICS}, flush=True)
        raw = pd.DataFrame({m: {k: float(res[m]["global"].get(k, np.nan)) for k in METRICS} for m in MODELS}).T
        # 7-model composite first: the acceptance check
        t7.append(score_table(raw.loc[ORIG]).assign(dataset=ds).rename_axis("model").reset_index())
        sc = score_table(raw).rename_axis("model").reset_index()
        sc.insert(0, "dataset", ds)
        sc.insert(1, "dataset_label", label)
        sc.insert(2, "tissue", tissue)
        sc.insert(4, "n_cells", n)
        sc.insert(5, "n_latent", sc["model"].map(lat))
        rows.append(sc)
        # Supplement: SATURN raw latent, scored alone
        for m, tool_dir in (("SATURN", NEW["SATURN"]), ("SATURN-Leiden", NEW["SATURN-Leiden"])):
            g = scorecard(np.load(latent_dir(tool_dir, stem) / "z_raw.npy"), obs)["global"]
            raw_sat.append(dict(dataset=ds, dataset_label=label, model=m, latent="raw X",
                                **{k: float(g[k]) for k in METRICS}))
        for s in (42, 43, 44):
            d = latent_dir(NEW["CAMEX"], stem, s)
            g = res["CAMEX"]["global"] if s == 42 else scorecard(np.load(d / "z_mean.npy"), obs)["global"]
            cam.append(dict(dataset=ds, dataset_label=label, seed=s, **{k: float(g[k]) for k in METRICS}))
    T = pd.concat(rows, ignore_index=True)
    # Acceptance
    a7 = pd.concat(t7, ignore_index=True).set_index(["dataset", "model"])
    mm = master.set_index(["dataset", "model"])
    raw_equal = {m: bool((a7[f"raw__{m}"].sub(mm[f"raw__{m}"].reindex(a7.index)).abs().fillna(0) == 0).all()
                         and (a7[f"raw__{m}"].isna() == mm[f"raw__{m}"].reindex(a7.index).isna()).all()) for m in METRICS}
    max_composite = float((a7["total_score"] - mm["total_score"].reindex(a7.index)).abs().max())
    max_wins = int((a7["wins_vs_pca"] - mm["wins_vs_pca"].reindex(a7.index)).abs().max())
    ext = T[T.model.isin(ORIG)].set_index(["dataset", "model"])
    raw_equal_ext = all(bool((ext[f"raw__{m}"] == mm[f"raw__{m}"].reindex(ext.index)).all()) for m in METRICS)
    worst_regress = max(max(v.values()) for v in regress.values())
    checks = dict(raw_columns_original_seven_equal=raw_equal, raw_equal_in_extended_table=raw_equal_ext,
                  composite_7model_max_abs_diff=max_composite, wins_max_abs_diff=max_wins,
                  regression_harmony_max_abs=worst_regress, regression_by_dataset=regress)
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps({k: v for k, v in checks.items() if k != "regression_by_dataset"}, indent=2), flush=True)
    if not (all(raw_equal.values()) and raw_equal_ext and max_composite <= 1e-12 and max_wins == 0):
        raise SystemExit("acceptance failed: the original seven do not reproduce master_table.csv")
    T.to_csv(OUT / "master_table_ext.csv", index=False)
    labels = {ds: v[1] for ds, v in DATASETS.items()}
    order = [m for _, ms in TIERS for m in ms]
    S = T.pivot(index="dataset", columns="model", values="total_score").loc[list(DATASETS), order].rename(index=labels)
    W = T.pivot(index="dataset", columns="model", values="wins_vs_pca").loc[list(DATASETS), order].rename(index=labels)
    use = pd.Series({m: lab.replace("\n", " ") for lab, ms in TIERS for m in ms}, name="labels_in_training")
    liver = [v[1] for v in DATASETS.values() if v[2] == "liver"]
    testis = [v[1] for v in DATASETS.values() if v[2] == "testis"]
    summ = pd.DataFrame({"labels_in_training": use, "mean_score_all": S.mean(0), "mean_score_liver": S.loc[liver].mean(0),
                         "mean_score_testis": S.loc[testis].mean(0), "rows_won": S.idxmax(1).value_counts()})
    summ["rows_won"] = summ["rows_won"].fillna(0).astype(int)
    summ.sort_values("mean_score_all", ascending=False).rename_axis("model").to_csv(OUT / "mean_scores.csv")
    liv = summ[["labels_in_training", "mean_score_liver"]].sort_values("mean_score_liver", ascending=False)
    liv.insert(0, "rank", np.arange(1, len(liv) + 1))
    liv.rename_axis("model").to_csv(OUT / "liver_only_ranking.csv")
    pd.DataFrame(raw_sat).to_csv(OUT / "supplement_saturn_raw_latent.csv", index=False)
    C = pd.DataFrame(cam)
    C.to_csv(OUT / "supplement_camex_seeds.csv", index=False)
    C.groupby(["dataset", "dataset_label"])[METRICS].agg(["mean", "std"]).to_csv(OUT / "supplement_camex_seed_sd.csv")
    print(summ.sort_values("mean_score_all", ascending=False).round(3).to_string(), flush=True)
    print(liv.round(3).to_string(), flush=True)
    heatmap(S, "composite_score_10models", cmap="Blues", vmin=0, vmax=1, fmt="{:.2f}", cbar="Composite score",
            mark_max=True)
    heatmap(W, "wins_vs_pca_10models", cmap="RdYlGn", vmin=0, vmax=6, fmt="{:.0f}", cbar="Metrics beating PCA (of 6)",
            mark_max=False)
    # raw metrics, one panel per metric
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(fd.DOUBLE_W, 3.6), layout="constrained", sharey=True)
    names = {"knn_species_purity": "kNN species purity (lower better)", "knn_mixing_entropy": "mixing entropy",
             "sil_species": "silhouette species (lower better)", "knn_celltype_purity": "kNN cell-type purity",
             "sil_celltype": "silhouette cell type", "label_transfer_acc": "label transfer accuracy"}
    for ax, m in zip(axes.ravel(), METRICS):
        R = T.pivot(index="dataset", columns="model", values=f"raw__{m}").loc[list(DATASETS), order].rename(index=labels)
        im = ax.imshow(R.values, aspect="auto", cmap="viridis")
        ax.set_title(names[m], fontsize=7)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=60, ha="right", fontsize=5)
        ax.set_yticks(range(len(R)))
        ax.set_yticklabels(R.index, fontsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
        cb.ax.tick_params(labelsize=5)
    fd.save_figure(fig, "raw_metrics_10models", close=True)
    print("DONE", OUT)
    (OUT / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")


if __name__ == "__main__":
    main()
