"""Existing runs that differ in MMD conditioning: same metrics and depth share, no training. python -m revision.tasks.Z2_embedding.mmd_pairs  Output: results/revision/Z2/mmd_pairs.csv"""
import numpy as np, pandas as pd
from revision.tasks.Z2_embedding import common as K, rules as RU
from revision.tasks.Z2_embedding.depth_axis_fn import one
e = pd.read_csv(RU.OUT / "existing_runs.csv")
rows = []
sel = {"test_hmouse": ("camex_testis_human_mouse", {"ep": [200, 150]}), "kidney": ("kidney_healthy_train", {}), "islet": ("islet_healthy_train", {})}
for c, (a, _) in sel.items():
    T = K.targets(c)
    q = e[(e.adata == a)]
    for r in q.itertuples():
        if not (RU.RUNS / r.run / "final_state.npz").exists():
            continue
        z = K.load_z(RU.RUNS / r.run / "final_state.npz")
        if not np.isfinite(z).all() or len(z) != len(T["spc"]):
            continue
        d = K.all_rows(c, z, with_global=True)
        g = lambda t, k: d[(d.task == t) & (d.metric == k)].value.mean()
        x = dict(cohort=c, run=r.run, cond=r.cond, adv=r.adv, mmd=r.mmd, fb=r.fb, ep=r.ep, seed=r.seed, mix=g("global", "knn_mixing_entropy"), lt=g("global", "label_transfer_acc"), wv=np.average(d[(d.metric == "within_var_r2") & (d.kind == "plain")].value, weights=d[(d.metric == "within_var_r2") & (d.kind == "plain")].n))
        x.update(one(z, T))
        if c == "test_hmouse":
            x["auc"] = np.nanmean([g(t, "auroc_lr") for t in ("stage_SG", "stage_SC", "stage_SD")]); x["rself"] = np.nanmean([g(t, "rho_self") for t in ("stage_SG", "stage_SC", "stage_SD")])
        elif c == "kidney":
            x["auc"] = g("pt_segment", "auroc_lr"); x["rself"] = g("pt_segment", "rho_self")
        else:
            x["auc"] = g("beta_stress", "auroc_lr"); x["rself"] = g("beta_stress", "rho_self")
        rows.append(x)
        print(c, r.run, flush=True)
        pd.DataFrame(rows).to_csv(RU.OUT / "mmd_pairs.csv", index=False)
t = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_rows", 100)
print(t.groupby(["cohort", "cond"])[["mix", "lt", "auc", "rself", "wv", "pc1_share", "within_z_var_by_lib"]].agg("mean").round(3).assign(n=t.groupby(["cohort", "cond"]).size()).to_string())
print(t[t.cond != True][["cohort", "run", "mix", "lt", "auc", "rself", "wv", "within_z_var_by_lib"]].round(3).to_string())
