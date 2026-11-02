"""How much of each latent is sequencing depth. Per latent on every cohort with targets: share of z variance on PC1; R2 of log library from z (linear, all cells); within (species, label) the same, weighted by cells;
share of total z variance linearly explained by log library within (species, label). Also the kNN lib_r2 of measure. python -m revision.tasks.Z2_embedding.depth_axis  Output: results/revision/Z2/depth_axis.csv"""
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from revision.tasks.Z2_embedding import common as K, rules as RU


def one(z, T):
    spc, lab, y = T["spc"], T["lab"], T["lib_size"]
    zc = z - z.mean(0)
    sv = np.linalg.svd(zc, compute_uv=False) ** 2
    r = dict(pc1_share=sv[0] / sv.sum(), r2_lib_from_z=LinearRegression().fit(zc, y).score(zc, y))
    keep, tot, expl, wts, r2s = np.zeros(len(y), bool), 0.0, 0.0, [], []
    for s in np.unique(spc):
        for l in np.unique(lab[spc == s]):
            q = (spc == s) & (lab == l)
            if q.sum() < RU.MIN_LABEL_CELLS:
                continue
            Z = z[q] - z[q].mean(0)
            Y = (y[q] - y[q].mean())[:, None]
            b = np.linalg.lstsq(Y, Z, rcond=None)[0]
            tot += (Z ** 2).sum()
            expl += ((Y @ b) ** 2).sum()
            r2s.append(LinearRegression().fit(Z, Y.ravel()).score(Z, Y.ravel()))
            wts.append(q.sum())
    r.update(within_r2_lib_from_z=float(np.average(r2s, weights=wts)), within_z_var_by_lib=expl / tot)
    return r


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
