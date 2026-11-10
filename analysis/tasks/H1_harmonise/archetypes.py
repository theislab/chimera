"""H1: gene archetypes per cohort, the first submission's method (notebooks/_gene_archetypes_builder.py, Supplementary Methods): GIN last hidden layer, dead and saturated
neurons dropped, rows L2-normalised, KMeans k in 3-12 chosen by the joint z-score of silhouette and cross-seed ARI, relabelled by ascending mean Delta-alpha. One run per cohort,
the first kept seed, as the first submission used one run. Exploratory, model-internal.

    PYTHONPATH=<chimera repo> python analysis/tasks/H1_harmonise/archetypes.py [--jobs 6]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from analysis.tasks.H1_harmonise import rules as RU

OUT = RU.OUT / "archetypes"
K_RANGE = list(range(3, 13))
N_SEEDS = 10  # first submission 30; ten keep the selection rule at a fraction of the cost, and k is compared with it on liver 4-sp
SIL_SAMPLE = 2000


def one(c: str) -> dict:
    torch.set_num_threads(2)
    runs = pd.read_csv(RU.OUT / "runs.tsv", sep="\t")
    r = runs[(runs.cohort == c) & runs.kept].sort_values("seed").iloc[0]
    d = RU.cohorts()[c]
    a = ad.read_h5ad(d["h5ad"], backed="r")
    F = np.asarray(a.varm["gene_features"], dtype=np.float32)
    vn = np.asarray(a.var_names)
    a.file.close()
    ck = torch.load(RU.RUNS / r.run / "model" / "model.pt", map_location="cpu", weights_only=False)
    sd = {k.removeprefix("gin.correction_mlp."): v.cpu().numpy() for k, v in ck["model_state_dict"].items() if k.startswith("gin.correction_mlp.")}
    x = torch.from_numpy(F)
    w = {k: torch.from_numpy(v) for k, v in sd.items()}
    with torch.no_grad():
        h1 = torch.tanh(x @ w["0.weight"].T + w["0.bias"])
        h2 = torch.tanh(h1 @ w["2.weight"].T + w["2.bias"])
        out = (h2 @ w["4.weight"].T + w["4.bias"]).squeeze(-1).numpy()
    H2 = h2.numpy()
    with np.load(RU.RUNS / r.run / "final_state.npz", allow_pickle=True) as z:
        corr = np.asarray(z["correction"], float)
    rt = float(np.max(np.abs(out - corr)))
    keep = (H2.var(0) > 0.01) & (np.abs(H2).mean(0) < 0.95)
    H = normalize(H2[:, keep], norm="l2", axis=1)
    rng = np.random.default_rng(42)
    seeds = list(map(int, rng.integers(0, 2**31, size=N_SEEDS)))
    sil, ari = [], []
    for k in K_RANGE:
        labs = [KMeans(n_clusters=k, n_init=5, random_state=s).fit_predict(H) for s in seeds]
        sil.append(float(silhouette_score(H, labs[0], sample_size=min(SIL_SAMPLE, len(H)), random_state=42)))
        ari.append(float(np.mean([adjusted_rand_score(labs[i], labs[j]) for i in range(N_SEEDS) for j in range(i + 1, N_SEEDS)])))
    sa, aa = np.array(sil), np.array(ari)
    zc = (sa - sa.mean()) / (sa.std() or 1) + (aa - aa.mean()) / (aa.std() or 1)
    k = K_RANGE[int(np.argmax(zc))]
    lab = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(H)
    g = pd.read_parquet(RU.OUT / "genes" / f"{c}.parquet").reindex(vn)
    m = pd.Series(g.D.to_numpy()).groupby(lab).mean().sort_values()
    remap = {o: i for i, o in enumerate(m.index)}
    arch = np.array([remap[x] for x in lab])
    t = pd.DataFrame(dict(var_name=vn, hsapiens_gene_id=g.hsapiens_gene_id.to_numpy(), symbol=g.symbol.to_numpy(), archetype=arch, D=g.D.to_numpy(), S_ema=g.S_ema.to_numpy(), count_support=g.count_support.to_numpy()))
    t.to_csv(OUT / f"{c}.csv", index=False)
    s = t.groupby("archetype").agg(n=("D", "size"), mean_D=("D", "mean"), median_D=("D", "median"), mean_S_supported=("S_ema", lambda v: v[t.loc[v.index, "count_support"]].mean()))
    s.insert(0, "cohort", c)
    s["k"], s["run"], s["silhouette"], s["ari"], s["roundtrip"], s["n_neurons_kept"] = k, r.run, sil[K_RANGE.index(k)], ari[K_RANGE.index(k)], rt, int(keep.sum())
    return s.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        t = pd.concat(list(ex.map(one, list(RU.cohorts()))), ignore_index=True)
    t.to_csv(OUT / "summary.csv", index=False)
    print(t.groupby("cohort", sort=False).agg(k=("k", "first"), sil=("silhouette", "first"), ari=("ari", "first"), rt=("roundtrip", "first"), lo=("mean_D", "first"), hi=("mean_D", "last")).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
