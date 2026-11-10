"""z_mean PC spectrum of every finished run on disk, no training: effective count exp(entropy of PC shares), against recipe columns. python -m analysis.tasks.Z2_embedding.spectra  Output: results/analysis/Z2/spectra.csv"""
import numpy as np, pandas as pd
from analysis.tasks.Z2_embedding import rules as RU
t = pd.read_csv(RU.OUT / "existing_runs.csv")
rows = []
for r in t.itertuples():
    try:
        with np.load(RU.RUNS / r.run / "final_state.npz", allow_pickle=True) as d:
            z = np.asarray(d["z_mean"], float)
    except Exception:
        continue
    if not np.isfinite(z).all() or len(z) < 100:
        continue
    z = z[np.random.default_rng(0).choice(len(z), min(5000, len(z)), replace=False)]
    s = np.linalg.svd(z - z.mean(0), compute_uv=False) ** 2
    p = s / s.sum()
    rows.append(dict(run=r.run, pc1=p[0], pc2=p[1], pc3=p[2], neff=float(np.exp(-(p * np.log(p.clip(min=1e-12))).sum()))))
d = t.merge(pd.DataFrame(rows), on="run")
d.to_csv(RU.OUT / "spectra.csv", index=False)
pd.set_option("display.width", 220, "display.max_rows", 200)
v = ["adata", "ep", "adv", "mmd", "fb", "nl", "cond", "vamp"]
x = d[d.adata.isin(["camex_human_mouse", "camex_human_monkey", "camex_testis_human_mouse", "kidney_healthy_train", "islet_healthy_train", "bone_marrow_unified", "camex_pancreas_unified"])].fillna("-")
print(x.astype({"adv": str, "mmd": str, "fb": str, "vamp": str, "cond": str}).groupby(v).agg(n=("run", "size"), pc1=("pc1", "mean"), pc2=("pc2", "mean"), pc3=("pc3", "mean"), neff=("neff", "mean"), eg=("run", "first")).round(3).reset_index().to_string())
