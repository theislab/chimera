"""Liver H+Mouse per seed: mixing against zone, in the model and through the encoder; what differs between seeds. python -m revision.tasks.Z2_embedding.perseed"""
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from revision.tasks.Z2_embedding import common as K, rules as RU
pd.set_option("display.width", 220, "display.max_columns", 30)
c = "liv_hmouse"
runs = K.chimera_runs(c)
e = pd.read_csv(RU.OUT / "encoded.csv")
rows = []
for s, run in runs.items():
    m = pd.read_parquet(RU.OUT / f"measure/{c}__chimera_s{s}.parquet")
    g = lambda t, k: m[(m.task == t) & (m.metric == k)].value.mean()
    q = e[e.model == run]
    h = lambda st, k, d, mk: q[(q.set == st) & (q.metric == k) & (q.depth == d) & (q.marker == mk)].value.mean()
    d = RU.RUNS / run
    mt = pd.read_csv(sorted(d.glob("lightning/**/metrics.csv"))[-1])
    lr = mt[["epoch", "lr/encoder"]].dropna() if "lr/encoder" in mt else pd.DataFrame()
    fr = lr[(lr["lr/encoder"] < 1e-5) & (lr.epoch >= 5)].epoch.min() if len(lr) else np.nan
    z = np.load(d / "final_state.npz", allow_pickle=True)["z_mean"]
    sv = np.linalg.svd(z - z.mean(0), compute_uv=False) ** 2
    rows.append(dict(seed=s, mixing=g("global", "knn_mixing_entropy"), lt=g("global", "label_transfer_acc"), rho_self=g("hep_zone", "rho_self"), rho_cross=g("hep_zone", "rho_cross"), auroc=g("hep_zone", "auroc_lr"),
                     lib=m[(m.task == "labelfree") & (m.metric == "lib_r2")].pipe(lambda x: np.average(x.value, weights=x.n)), wv=m[(m.task == "labelfree") & (m.metric == "within_var_r2")].pipe(lambda x: np.average(x.value, weights=x.n)),
                     tms=h("tms", "tms_auroc_lr", "asis", "author"), tms_depth=h("tms", "tms_auroc_lr", "depth", "author"), tms_self=h("tms", "rho_self", "asis", "all"), ts_self=h("ts", "rho_self", "asis", "all"), freeze=fr, pc1=sv[0] / sv.sum(), device=json.loads((d / "config.json").read_text()).get("accelerator")))
t = pd.DataFrame(rows).round(3)
print(t.to_string())
print(t.drop(columns=["device"]).corr(method="spearman").round(2)[["mixing", "rho_self", "tms", "lib"]].to_string())
