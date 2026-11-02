"""Z2 tables: within-label vs scVI by cohort; encoder tests; factorial trade-off with the decision rule; tidy rows for the benchmark overview.
python -m revision.tasks.Z2_embedding.tables   Output: results/revision/Z2/{within_label.csv, encoded.csv, factorial.csv, factorial_seeds.csv, decision.csv, Z2_tidy.csv}"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from revision.tasks.Z2_embedding import rules as RU

pd.set_option("display.width", 240, "display.max_rows", 400, "display.max_columns", 40, "display.max_colwidth", 28)
METRICS = ["auroc_lr", "auroc_knn", "rho_self", "rho_cross", "clisi", "graph_conn"]


def load_measure():
    d = pd.concat(pd.read_parquet(f) for f in glob.glob(str(RU.OUT / "measure/*.parquet")))
    return d


def seedvals(d, task_filter):
    """One value per latent for task and metric: mean over species (self, AUROC) or ordered pairs (cross)."""
    x = d[task_filter]
    return x.groupby(["cohort", "task", "metric", "method", "latent"]).value.mean().reset_index()


def within(d):
    base = d[~d.latent.str.startswith("z2-") & ~d.latent.str.startswith("rev-") | d.method.eq("Chimera")]
    base = base[~base.task.isin(["labelfree", "global"]) & ~base.task.str.contains("__")]
    v = seedvals(base, base.metric.isin(METRICS))
    out = []
    for (c, t, m), g in v.groupby(["cohort", "task", "metric"]):
        r = dict(cohort=c, task=t, metric=m)
        for me in g.method.unique():
            q = g[g.method == me].value
            r[me] = q.mean()
            r[f"{me}_lo"], r[f"{me}_hi"], r[f"{me}_n"] = q.min(), q.max(), len(q)
        if "Chimera" in r and "scVI" in r:
            r["gap"] = r["Chimera"] - r["scVI"]
        out.append(r)
    return pd.DataFrame(out)


def labelfree(d):
    x = d[d.task == "labelfree"]
    w = x.groupby(["cohort", "method", "latent", "kind", "metric"]).apply(lambda q: np.average(q.value, weights=q.n), include_groups=False).rename("value").reset_index()
    return w


def glob_metrics(d):
    x = d[(d.task == "global")]
    return x.groupby(["cohort", "method", "latent", "metric"]).value.mean().reset_index()


def encoded():
    fs = glob.glob(str(RU.OUT / "encoded/*.parquet"))
    return pd.concat(pd.read_parquet(f) for f in fs) if fs else pd.DataFrame()


def fac_table(d, enc, reading, c):
    fac = pd.read_csv(RU.REPO / "revision/tasks/Z2_embedding/factorial.tsv", sep="\t")
    runs = {r.run: r.level for r in fac[fac.cohort == c].itertuples()}
    runs.update({x: "base" for x in RU.base_seed_runs(c).values()})
    x = d[(d.cohort == c)]
    # base seeds are stored under chimera_s<seed>; map them
    seed_of = {f"chimera_s{r.rsplit('-s', 1)[1]}": r for r in RU.base_seed_runs(c).values()}
    x = x.assign(run=x.latent.map(lambda l: seed_of.get(l, l)))
    x = x[x.run.isin(runs)]
    rows = []
    for run, g in x.groupby("run"):
        r = dict(run=run, level=runs[run])
        def get(task, metric):
            q = g[(g.task == task) & (g.metric == metric)]
            return q.value.mean() if len(q) else np.nan
        if c == "liv_hmouse":
            r.update(hep_rho_self=get("hep_zone", "rho_self"), hep_rho_cross=get("hep_zone", "rho_cross"), hep_auroc=get("hep_zone", "auroc_lr"))
        else:
            for t in ("stage_SG", "stage_SC", "stage_SD"):
                r[t + "_auroc"] = get(t, "auroc_lr")
            r["stage_auroc"] = np.nanmean([r[t + "_auroc"] for t in ("stage_SG", "stage_SC", "stage_SD")])
            r["stage_rho_self"] = np.nanmean([get(t, "rho_self") for t in ("stage_SG", "stage_SC", "stage_SD")])
            r["round_elong_cross"] = get("round_vs_elongated", "rho_cross")
        lf = g[(g.task == "labelfree") & (g.metric == "within_var_r2") & (g.kind == "plain")]
        r["within_var"] = np.average(lf.value, weights=lf.n) if len(lf) else np.nan
        lb = g[(g.task == "labelfree") & (g.metric == "lib_r2")]
        r["lib_r2"] = np.average(lb.value, weights=lb.n) if len(lb) else np.nan
        for m, k in (("knn_mixing_entropy", "mixing"), ("label_transfer_acc", "label_transfer"), ("sil_celltype", "sil_ct")):
            r[k] = get("global", m)
        rows.append(r)
    t = pd.DataFrame(rows)
    if not len(enc):
        return t
    e = enc[enc.cohort == c]
    if c == "liv_hmouse":
        a = e[(e.metric == "tms_auroc_lr") & (e.depth == "asis")].set_index("model").value.rename("tms_auroc")
        b = e[(e.metric == "tms_auroc_lr") & (e.depth == "depth")].set_index("model").value.rename("tms_auroc_depth")
        s = e[(e.metric == "rho_self") & (e.set == "tms") & (e.depth == "asis") & (e.marker == "all")].set_index("model").value.rename("tms_rho_self")
        t = t.join(a, on="run").join(b, on="run").join(s, on="run")
    else:
        a = e[(e.metric == "stage_auroc_lr") & (e.depth == "asis")].groupby("model").value.mean().rename("mac_auroc")
        s = e[(e.set == "macaque_to_human_in_model") & (e.depth == "asis")].groupby("model").value.mean().rename("mac_rho_cross")
        t = t.join(a, on="run").join(s, on="run")
    return t


def main():
    d = load_measure()
    w = within(d)
    w.to_csv(RU.OUT / "within_label.csv", index=False)
    lf = labelfree(d)
    lf.to_csv(RU.OUT / "labelfree.csv", index=False)
    gm = glob_metrics(d)
    gm.to_csv(RU.OUT / "global.csv", index=False)
    enc = encoded()
    if len(enc):
        enc.to_csv(RU.OUT / "encoded.csv", index=False)
    show = [c for c in ["cohort", "task", "metric", "Chimera", "Chimera_lo", "Chimera_hi", "Chimera_n", "scVI", "scVI_lo", "scVI_hi", "scVI_n", "gap", "CAMEX", "sysVI", "PCA"] if c in w]
    print(w[w.metric.isin(["auroc_lr", "rho_self", "rho_cross", "clisi"])][show].round(3).to_string())
    k = lf[lf.method.isin(["Chimera", "scVI", "CAMEX", "PCA", "sysVI"])].groupby(["cohort", "kind", "metric", "method"]).value.mean().unstack("method").round(3)
    print(k.to_string())
    print(gm[gm.metric.isin(["knn_mixing_entropy", "label_transfer_acc"])].groupby(["cohort", "metric", "method"]).value.mean().unstack("method").round(3).to_string())
    if len(enc):
        print(enc[enc.marker.isin(["all", "author"]) | enc.set.str.contains("macaque")].assign(method=lambda x: x.model.str.startswith("scvi").map({True: "scVI", False: "Chimera"})).pipe(lambda x: x[~x.model.str.startswith("z2-")]).groupby(["cohort", "set", "metric", "depth", "marker", "method"]).value.agg(["mean", "min", "max", "size"]).round(3).to_string())
    try:
        rd = pd.read_csv(RU.OUT / "reading.csv")
    except FileNotFoundError:
        rd = pd.DataFrame()
    for c in ("liv_hmouse", "test_hmouse"):
        t = fac_table(d, enc, rd, c)
        if not len(t):
            continue
        if len(rd):
            keep = ["run", "healthy", "rho_DS", "rho_DS_partial", "rho_D_consensus", "rho_S_consensus", "sat_hi", "tail_up_raw", "tail_dn_raw", "n_active_z", "lr_freeze_epoch", "adv_acc", "frac_S_pos", "mix_entropy"]
            t = t.merge(rd[[x for x in keep if x in rd]], on="run", how="left")
        t.to_csv(RU.OUT / f"factorial_seeds_{c}.csv", index=False)
        print(t.sort_values(["level", "run"]).drop(columns=["run"]).round(3).to_string())


if __name__ == "__main__":
    main()
