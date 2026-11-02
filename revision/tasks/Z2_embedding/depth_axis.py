"""How much of each latent is sequencing depth. Per latent on every cohort with targets: share of z variance on PC1; R2 of log library from z (linear, all cells); within (species, label) the same, weighted by cells;
share of total z variance linearly explained by log library within (species, label). Also the kNN lib_r2 of measure. python -m revision.tasks.Z2_embedding.depth_axis  Output: results/revision/Z2/depth_axis.csv"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from revision.tasks.Z2_embedding import common as K, rules as RU


from revision.tasks.Z2_embedding.depth_axis_fn import one

rows = []
for f in sorted((RU.OUT / "cache").glob("*_targets.npz")):
    c = f.name.replace("_targets.npz", "")
    T = K.targets(c)
    for name, (m, s, p) in K.latents(c).items():
        z = K.load_z(p)
        if z.shape[0] != len(T["spc"]):
            continue
        rows.append(dict(cohort=c, latent=name, method=m, seed=s, **one(z, T)))
d = pd.DataFrame(rows)
d.to_csv(RU.OUT / "depth_axis.csv", index=False)
pd.set_option("display.width", 200)
print(d.groupby(["cohort", "method"])[["pc1_share", "r2_lib_from_z", "within_r2_lib_from_z", "within_z_var_by_lib"]].mean().round(3).loc[lambda t: t.index.get_level_values(1).isin(["Chimera", "scVI", "scVI-library", "CAMEX", "sysVI", "PCA"])].to_string())
