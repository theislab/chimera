"""Factorial so far, level means with seed ranges; depth share of each run; post-hoc views. python -m revision.tasks.Z2_embedding.quick"""
import glob, json
import numpy as np, pandas as pd
from revision.tasks.Z2_embedding import common as K, rules as RU
from revision.tasks.Z2_embedding.depth_axis_fn import one
pd.set_option("display.width", 220, "display.max_rows", 300, "display.max_columns", 30)
fac = pd.read_csv(RU.REPO / "revision/tasks/Z2_embedding/factorial.tsv", sep="\t")
rows = []
for c in ("liv_hmouse", "test_hmouse"):
    if not (RU.OUT / f"cache/{c}_targets.npz").exists():
        continue
    T = K.targets(c)
    lv = {r.run: r.level for r in fac[fac.cohort == c].itertuples()} | {x: "base" for x in RU.base_seed_runs(c).values()}
    for run, l in lv.items():
        fm, fe = RU.OUT / f"measure/{c}__{run}.parquet", RU.OUT / f"encoded/{c}__{run}.parquet"
        if not fm.exists():
            continue
        m = pd.read_parquet(fm)
        g = lambda t, k: m[(m.task == t) & (m.metric == k)].value.mean()
        lfw = lambda k: m[(m.task == "labelfree") & (m.metric == k) & (m.kind == "plain")].pipe(lambda x: np.average(x.value, weights=x.n))
        r = dict(cohort=c, level=l, run=run, mix=g("global", "knn_mixing_entropy"), lt=g("global", "label_transfer_acc"), wv=lfw("within_var_r2"), lib=lfw("lib_r2"))
        r.update(one(K.load_z(RU.RUNS / run / "final_state.npz"), T))
        if c == "liv_hmouse":
            r.update(rself=g("hep_zone", "rho_self"), rcross=g("hep_zone", "rho_cross"), auc=g("hep_zone", "auroc_lr"))
        else:
            r.update(rself=np.nanmean([g(t, "rho_self") for t in ("stage_SG", "stage_SC", "stage_SD")]), auc=np.nanmean([g(t, "auroc_lr") for t in ("stage_SG", "stage_SC", "stage_SD")]))
        if fe.exists():
            e = pd.read_parquet(fe)
            if c == "liv_hmouse":
                q = e[(e.metric == "tms_auroc_lr") & (e.depth == "asis")].value
                r["held_auc"] = q.mean()
                r["held_auc_depth"] = e[(e.metric == "tms_auroc_lr") & (e.depth == "depth")].value.mean()
            else:
                r["held_auc"] = e[(e.metric == "stage_auroc_lr") & (e.depth == "asis")].value.mean()
        rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(RU.OUT / "quick.csv", index=False)
print(t.groupby(["cohort", "level"]).agg(n=("run", "size"), mix=("mix", "mean"), lt=("lt", "mean"), rself=("rself", "mean"), rself_max=("rself", "max"), rcross=("rcross", "mean"), auc=("auc", "mean"), held=("held_auc", "mean"), held_max=("held_auc", "max"), wv=("wv", "mean"), pc1=("pc1_share", "mean"), zlib=("within_z_var_by_lib", "mean")).round(3).to_string())
