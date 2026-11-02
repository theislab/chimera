"""Fair capacity across latents: sum over z_mean PCs of 0.5 log(1 + 1 / noise ratio), PCs with noise ratio < 1, joined to species separability. python -m revision.tasks.Z2_embedding.capacity  Output: results/revision/Z2/capacity.csv"""
import glob
import numpy as np, pandas as pd
from revision.tasks.Z2_embedding import rules as RU
p = pd.concat(pd.read_csv(f) for f in glob.glob(str(RU.OUT / "snr_pcs_*.csv")) + glob.glob(str(RU.OUT / "snr_pcs.csv")))
p = p.drop_duplicates(["cohort", "latent", "pc"])
p["cap"] = 0.5 * np.log1p(1 / p.noise_ratio)
c = p.groupby(["cohort", "latent", "method", "seed"]).agg(capacity=("cap", "sum"), axes=("noise_ratio", lambda x: int((x < 1).sum()))).reset_index()
s = pd.concat(pd.read_csv(f) for f in glob.glob(str(RU.OUT / "snr_*.csv")) if "pcs" not in f and "runs" not in f).drop_duplicates(["cohort", "latent"])
t = c.merge(s[["cohort", "latent", "sp_auc_all", "sp_auc_biglabel"]], on=["cohort", "latent"], how="left")
t.to_csv(RU.OUT / "capacity.csv", index=False)
pd.set_option("display.width", 180)
print(t.groupby(["cohort", "method"])[["capacity", "axes", "sp_auc_all", "sp_auc_biglabel"]].mean().round(2).to_string())
print(t.groupby(["cohort", "method"]).size().to_string())
