"""T5.5 views beside the composite (composite unchanged, as frozen).

1. ``composite_mixing_bio_10models``: four panels on the same rows and columns, columns grouped by
   label use: raw kNN species purity (lower = more species mixing), Mixing score, Biology score
   (normalised means, as in the composite), Composite. Reads ``master_table_ext.csv``.
2. Supplementary sensitivity: SATURN, SATURN-Leiden and CAMEX latents reduced to 10 dimensions by PCA
   (sklearn, random_state 0; the panel's other methods have 10), rescored with the unmodified
   ``compute_integration_metrics`` (k=15, seed=42), and the 10-model composite recomputed with these
   columns in place of the native ones. Writes ``supplement_pca10_sensitivity.csv``.

    PYTHONPATH=. mamba run --no-capture-output -n chimera_env python revision/tasks/T5_5_benchmark/views.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_external as SE  # noqa: E402

TIERS = [("no labels", ["PCA", "Harmony", "scVI-sp", "scVI-donor", "sysVI", "CAMEX"]),
         ("within-sp.\nclusters", ["SATURN-Leiden"]),
         ("cell-type\nlabels", ["scANVI", "SATURN"]),
         ("liver none;\ntestis cond.\nMMD, labels", ["Chimera"])]
ORDER = [m for _, ms in TIERS for m in ms]


def panel(ax, M, title, cmap, vmin, vmax, mark, fd, lower_better=False):
    im = ax.imshow(M.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    vals = -M.values if lower_better else M.values
    best = np.nanargmax(vals, axis=1) if mark else np.full(len(M), -1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            dark = (v - vmin) / (vmax - vmin) < 0.6
            ax.text(j, i, f"{v:.2f}" + ("*" if j == best[i] else ""), ha="center", va="center", fontsize=5.5,
                    color="#222222" if dark else "#f5f5f5", fontweight="bold" if j == best[i] else "normal")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, rotation=55, ha="right", fontsize=6)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(M.index, fontsize=6)
    ax.set_title(title, fontsize=7, pad=24)
    x = 0
    for name, ms in TIERS:
        if x:
            ax.axvline(x - 0.5, color="#ffffff", lw=1.5)
        last = ms == TIERS[-1][1]
        ax.text(x - 0.45 if last else x + (len(ms) - 1) / 2, -0.6, name, ha="left" if last else "center",
                va="bottom", fontsize=4.6, color="#333333")
        x += len(ms)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return im


def figure(T: pd.DataFrame, fd) -> None:
    import matplotlib.pyplot as plt

    lab = {ds: v[1] for ds, v in SE.DATASETS.items()}
    piv = lambda c: T.pivot(index="dataset", columns="model", values=c).loc[list(SE.DATASETS), ORDER].rename(index=lab)
    fig, axes = plt.subplots(1, 4, figsize=(fd.DOUBLE_W * 1.45, 3.3), layout="constrained", sharey=True)
    specs = [("raw__knn_species_purity", "kNN species purity (raw; lower = more mixing)", "Greys", 0.5, 1.0, True),
             ("mixing", "Species mixing score (normalised)", "Blues", 0, 1, False),
             ("biology", "Cell-type conservation score (normalised)", "Blues", 0, 1, False),
             ("total_score", "Composite = 0.4 mixing + 0.6 conservation", "Blues", 0, 1, False)]
    for ax, (c, t, cm, lo, hi, lb) in zip(axes, specs):
        im = panel(ax, piv(c), t, cm, lo, hi, True, fd, lower_better=lb)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.01, location="bottom")
        cb.ax.tick_params(labelsize=5)
    fd.save_figure(fig, "composite_mixing_bio_10models", close=True)


def pca10(master_ext: pd.DataFrame) -> pd.DataFrame:
    rows, panels = [], []
    for ds, (stem, label, tissue) in SE.DATASETS.items():
        a = ad.read_h5ad(SE.DATA / f"{stem}.h5ad", backed="r")
        obs = a.obs[["species", "cell_type"]].copy()
        a.file.close()
        raw = master_ext[master_ext.dataset == ds].set_index("model")[[f"raw__{m}" for m in SE.METRICS]]
        raw.columns = SE.METRICS
        sub = raw.copy()
        for m, tool_dir in SE.NEW.items():
            z = np.load(SE.latent_dir(tool_dir, stem) / "z_mean.npy")
            p = PCA(n_components=10, random_state=0)
            z10 = p.fit_transform(z)
            g = SE.scorecard(z10, obs)["global"]
            sub.loc[m] = [float(g[k]) for k in SE.METRICS]
            rows.append(dict(dataset=ds, dataset_label=label, model=m, native_dim=z.shape[1],
                             pca10_var_explained=float(p.explained_variance_ratio_.sum()),
                             **{f"native__{k}": float(raw.loc[m, k]) for k in SE.METRICS},
                             **{f"pca10__{k}": float(g[k]) for k in SE.METRICS}))
            print(ds, m, {k: round(g[k], 3) for k in SE.METRICS}, flush=True)
        sc = SE.score_table(sub.loc[SE.MODELS])
        nat = master_ext[master_ext.dataset == ds].set_index("model")
        for m in SE.MODELS:
            panels.append(dict(dataset=ds, model=m, composite_native=float(nat.loc[m, "total_score"]),
                               composite_pca10=float(sc.loc[m, "total_score"]), mixing_pca10=float(sc.loc[m, "mixing"]),
                               biology_pca10=float(sc.loc[m, "biology"])))
    R = pd.DataFrame(rows)
    P = pd.DataFrame(panels)
    for r in R.itertuples():
        pass
    R = R.merge(P[P.model.isin(list(SE.NEW))], on=["dataset", "model"])
    return R, P


def main():
    import _figure_design as fd

    fd.apply_design()
    fd.set_figure_dir(SE.FIG)
    T = pd.read_csv(SE.OUT / "master_table_ext.csv")
    figure(T, fd)
    if "--figure-only" in sys.argv:
        return
    R, P = pca10(T)
    R.to_csv(SE.OUT / "supplement_pca10_sensitivity.csv", index=False)
    P.to_csv(SE.OUT / "supplement_pca10_composite_all_models.csv", index=False)
    m = P.groupby("model")[["composite_native", "composite_pca10"]].mean().sort_values("composite_native", ascending=False)
    print(m.round(3).to_string())
    print(R.groupby("model")[[c for c in R.columns if c.startswith(("native__", "pca10__"))]].mean().T.round(3).to_string())


if __name__ == "__main__":
    main()
