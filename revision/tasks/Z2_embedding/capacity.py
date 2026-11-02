"""Fair capacity across latents, noise-whitened (snr.py), joined to species separability. python -m revision.tasks.Z2_embedding.capacity  Output: results/revision/Z2/capacity.csv"""
import glob
import pandas as pd
from revision.tasks.Z2_embedding import rules as RU
s = pd.concat(pd.read_csv(f) for f in glob.glob(str(RU.OUT / "snr_*.csv")) if "pcs" not in f and "runs" not in f).drop_duplicates(["cohort", "latent"])
s.to_csv(RU.OUT / "capacity.csv", index=False)
pd.set_option("display.width", 180)
print(s.groupby(["cohort", "method"]).agg(n=("latent", "size"), capacity=("capacity", "mean"), cmin=("capacity", "min"), cmax=("capacity", "max"), axes=("axes", "mean"), sp_all=("sp_auc_all", "mean"), sp_big=("sp_auc_biglabel", "mean")).round(2).to_string())
