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
        elif c == "islet":
            r.update(beta_rho_self=get("beta_stress", "rho_self"), beta_rho_cross=get("beta_stress", "rho_cross"), beta_auroc=get("beta_stress", "auroc_lr"))
        elif c == "kidney":
            r.update(pt_seg_rho_self=get("pt_segment", "rho_self"), pt_seg_auroc=get("pt_segment", "auroc_lr"), pt_seg_cross=get("pt_segment", "rho_cross"), pt_state_auroc=get("pt_state", "auroc_lr"), pt_state_clisi=get("pt_state", "clisi"))
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
    if not len(enc) or c in ("islet", "kidney"):
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
    for c in ("liv_hmouse", "test_hmouse", "islet", "kidney"):
        t = fac_table(d, enc, rd, c)
        if not len(t):
            continue
        if len(rd):
            keep = ["run", "healthy", "rho_DS", "rho_DS_partial", "rho_D_consensus", "rho_S_consensus", "sat_hi", "tail_up_raw", "tail_dn_raw", "n_active_z", "lr_freeze_epoch", "adv_acc", "frac_S_pos", "mix_entropy"]
            t = t.merge(rd[[x for x in keep if x in rd]], on="run", how="left")
        t.to_csv(RU.OUT / f"factorial_seeds_{c}.csv", index=False)
        print(t.sort_values(["level", "run"]).drop(columns=["run"]).round(3).to_string())




def levels(c: str) -> pd.DataFrame:
    """Level means over seeds that pass the Z1 health rule; unhealthy seeds counted and left out (discarded, amendment rule)."""
    f = RU.OUT / f"factorial_seeds_{c}.csv"
    t = pd.read_csv(f)
    t["healthy"] = t.healthy.fillna(False).astype(bool)
    num = [x for x in t.columns if x not in ("run", "level", "healthy", "cohort") and pd.api.types.is_numeric_dtype(t[x])]
    g = t[t.healthy].groupby("level")
    out = g[num].mean()
    out.insert(0, "n_healthy", g.size())
    out.insert(1, "n_runs", t.groupby("level").size())
    out = out.reindex(t.level.unique()).reset_index()
    for k, m in (("rself_min", "hep_rho_self"), ("rself_max", "hep_rho_self"), ("held_min", "tms_auroc"), ("held_max", "tms_auroc")):
        if m in t:
            out[k] = out.level.map(getattr(t[t.healthy].groupby("level")[m], k.split("_")[1])())
    out.insert(0, "cohort", c)
    return out


def decide() -> pd.DataFrame:
    L = levels("liv_hmouse")
    TH, loo = RU.THRESH, pd.read_csv(RU.OUT / "reading_loo.csv")
    b = L[L.level == "base"].iloc[0]
    med = loo[loo.cohort == "liv_hmouse"].rho_loo.median()
    d = L.assign(p_healthy=L.n_healthy >= TH["min_healthy"], p_tms=L.tms_auroc >= b.tms_auroc + TH["gain_liver_tms_auroc"], p_self=L.hep_rho_self >= b.hep_rho_self + TH["gain_liver_rho_self"],
                 p_mix=L.mixing >= TH["mix_ratio"] * b.mixing, p_lt=L.label_transfer >= b.label_transfer - TH["max_loss_label_transfer"], p_ds=L.rho_DS >= b.rho_DS - TH["max_loss_rho_DS"], p_dsp=L.rho_DS_partial >= b.rho_DS_partial - TH["max_loss_rho_DS"],
                 p_cons=L.rho_D_consensus >= med - TH["max_loss_consensus"])
    d["passes_liver"] = d[[x for x in d.columns if x.startswith("p_")]].all(axis=1)
    return d


def tidy() -> pd.DataFrame:
    rows = []

    def add(task, setting, method, metric, v, direction="higher"):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v):
            rows.append(dict(track="E", task=task, setting=setting, method=method, metric=metric, value=v.mean(), ci_lo=v.min() if len(v) > 1 else np.nan, ci_hi=v.max() if len(v) > 1 else np.nan, n=len(v), rank_direction=direction))
    d = load_measure()
    d = d[d.method.isin(["Chimera", "scVI", "scVI-library", "CAMEX", "sysVI", "PCA", "Harmony", "scANVI", "SATURN"])]
    b = d[~d.task.isin(["labelfree", "global"]) & ~d.task.str.contains("__") & d.metric.isin(METRICS)]
    for (c, t, k, m), g in b.groupby(["cohort", "task", "metric", "method"]):
        add(f"within_label:{t}", f"cohort={c};mode=in_model", m, k, g.groupby("latent").value.mean())
    lf = labelfree(d)
    for (c, k, kd, m), g in lf[lf.metric == "within_var_r2"].groupby(["cohort", "metric", "kind", "method"]):
        add("within_label:variance", f"cohort={c};mode=in_model;centring={kd}", m, k, g.value)
    for (c, m), g in lf[lf.metric == "lib_r2"].groupby(["cohort", "method"]):
        add("within_label:depth", f"cohort={c};mode=in_model", m, "lib_r2", g.value, "lower")
    gm = glob_metrics(d)
    for (c, k, m), g in gm[gm.metric.isin(["knn_mixing_entropy", "label_transfer_acc"])].groupby(["cohort", "metric", "method"]):
        add("global", f"cohort={c};mode=in_model", m, k, g.value)
    e = encoded()
    e = e[~e.model.str.startswith("z2-")].assign(method=lambda x: x.model.str.startswith("scvi").map({True: "scVI", False: "Chimera"}))
    for (c, st, k, dp, mk, m), g in e.groupby(["cohort", "set", "metric", "depth", "marker", "method"]):
        if k != "z_scale_sd":
            add(f"held_out:{st}", f"cohort={c};mode=encoder;input={dp};marker={mk}", m, k, g.value)
    cap = pd.read_csv(RU.OUT / "capacity.csv")
    for (c, m), g in cap.groupby(["cohort", "method"]):
        add("latent", f"cohort={c};mode=in_model", m, "capacity_nats", g.capacity)
        add("latent", f"cohort={c};mode=in_model", m, "effective_axes", g.axes)
        add("latent", f"cohort={c};mode=in_model", m, "species_auroc_in_largest_label", g.sp_auc_biglabel, "lower")
    dp = pd.read_csv(RU.OUT / "depth_axis.csv")
    for (c, m), g in dp.groupby(["cohort", "method"]):
        add("latent", f"cohort={c};mode=in_model", m, "z_within_label_variance_by_depth", g.within_z_var_by_lib, "lower")
    for c in ("liv_hmouse", "test_hmouse", "islet", "kidney"):
        f = RU.OUT / f"factorial_seeds_{c}.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        t = t[t.healthy.fillna(False).astype(bool)]
        for lv, g in t.groupby("level"):
            for k in [x for x in g.columns if x not in ("run", "level", "healthy", "cohort") and pd.api.types.is_numeric_dtype(g[x])]:
                add("factorial", f"cohort={c};level={lv}", "Chimera", k, g[k], "lower" if k in ("lib_r2", "sat_hi") else "higher")
    t = pd.DataFrame(rows, columns=["track", "task", "setting", "method", "metric", "value", "ci_lo", "ci_hi", "n", "rank_direction"])
    t.to_csv(RU.OUT / "Z2_tidy.csv", index=False)
    return t


if __name__ == "__main__":
    main()
    L = pd.concat(levels(c) for c in ("liv_hmouse", "test_hmouse", "islet", "kidney") if (RU.OUT / f"factorial_seeds_{c}.csv").exists())
    L.to_csv(RU.OUT / "factorial_levels.csv", index=False)
    D = decide()
    D.to_csv(RU.OUT / "decision.csv", index=False)
    print(D[["level", "n_healthy", "n_runs", "hep_rho_self", "hep_rho_cross", "hep_auroc", "tms_auroc", "tms_auroc_depth", "mixing", "label_transfer", "within_var", "lib_r2", "rho_DS", "rho_DS_partial", "rho_D_consensus", "rho_S_consensus", "passes_liver"]].round(3).to_string())
    print(tidy().shape)
