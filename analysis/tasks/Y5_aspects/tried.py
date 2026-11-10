"""One list of every Chimera-readout test tried, BH across all of them, replication half beside it.

Writes results/analysis/Y5/tried.csv and survivors.csv. Usage: python -m analysis.tasks.Y5_aspects.tried
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from analysis.tasks.T4_3_scoreboard import stats as ST
from analysis.tasks.Y5_aspects import common as K, rules as RU

O = K.OUT
CH = RU.PRIMARY


def rep(est, p_even, est_even):
    return bool(np.sign(est) == np.sign(est_even) and p_even / 2 < 0.05)


def add(rows, fam, test, cohort, y, est, p, est2=np.nan, p2=np.nan, comp=None, note=""):
    rows.append(dict(family=fam, test=test, cohort=cohort, readout=y, est_disc=est, p_disc=p, est_rep=est2, p_rep=p2, comparator=comp, note=note))


def main():
    R = []
    # F1 properties
    t = pd.read_csv(O / "f1_scan.csv")
    t = t[(t["cov"] == "C4")]
    for (c, prop, y), q in t.groupby(["cohort", "property", "y"]):
        if c not in CH or y not in RU.F1_READOUTS:
            continue
        o, e = q[q.half == "odd"].iloc[0], q[q.half == "even"].iloc[0]
        comp = t[(t.cohort == c) & (t.property == prop) & (t.half == "odd") & t.y.isin(["B1b", "Sdir"])].set_index("y").rho_p.round(3).to_dict()
        add(R, "F1", prop, c, y, o.rho_p, o.p, e.rho_p, e.p, str(comp))
    # F1 ridge (fit odd, test even; the hold-out is built in: the even half is the only test)
    t = pd.read_csv(O / "f1_ridge.csv")
    t = t[(t.guard == "no_odd_paralog") & t.train.isin(CH) & t.target_cohort.isin(CH) & t.y.isin(["S", "D_raw"])]
    for r in t.itertuples():
        add(R, "F1b", f"ridge {r.train}>{r.target_cohort}", r.target_cohort, r.y, r.rho_p, r.p, r.rho_p, r.p, "see f1_ridge.csv")
    # F1c
    t = pd.read_csv(O / "f1c_gold.csv")
    t = t[t.train.isin(CH) & t.target.isin(CH) & t.y.isin(["t_c_esm", "t_D_raw", "t_pred_S"]) & (t.gold != "G_H4_primates")]
    for (tr, ta, go, y), q in t.groupby(["train", "target", "gold", "y"]):
        o, e = q[q.half == "odd"].iloc[0], q[q.half == "even"].iloc[0]
        add(R, "F1c", f"{tr}>{ta} {go}", ta, y, o.rho_p, o.p, e.rho_p, e.p)
    # F2
    t = pd.read_csv(O / "F2/depth_rho.csv")
    t = t[t.score.isin(["S", "D_raw", "B1z"]) & t.cohort.isin(CH)]
    for (c, s, g), q in t.groupby(["cohort", "score", "gold"]):
        o, e = q[q.half == "odd"].iloc[0], q[q.half == "even"].iloc[0]
        add(R, "F2", f"depth {g}", c, s, o.rho_p, o.p, e.rho_p, e.p)
    t = pd.read_csv(O / "F2/depth_conditional.csv")
    t = t[t.y.isin(["S", "D_raw", "B1z"]) & t.cohort.isin(CH)]
    for (c, s, d, w), q in t.groupby(["cohort", "y", "deep", "with_identity"]):
        o, e = q[q.half == "odd"].iloc[0], q[q.half == "even"].iloc[0]
        add(R, "F2", f"conditional {d} id={w}", c, s, o.rho_p, o.p, e.rho_p, e.p)
    t = pd.read_csv(O / "F2/class_depth__test_3sp.csv")
    t = t[t.y.isin(["S", "D_raw"])]
    for (cl, y), q in t.groupby(["cls", "y"]):
        o, e = q[q.half == "odd"].iloc[0], q[q.half == "even"].iloc[0]
        add(R, "F2", f"class {cl}", "test_3sp", y, o.rho_p, o.p, e.rho_p, e.p)
    # F3 axes and stages: Chimera healthy seeds, p from a t test over seeds
    from scipy.stats import ttest_1samp
    t = pd.read_csv(O / "F3/zone_rho.csv")
    t = t[t.latent.str.startswith("chimera") & (t.kind == "cross") & t.cohort.isin(CH)]
    for (c, ct, a, b), q in t.groupby(["cohort", "cell_type", "a", "b"]):
        o, e = q[q.half == "odd"].rho, q[q.half == "even"].rho
        if len(o) >= 2:
            add(R, "F3", f"axis {ct} {a}>{b}", c, "rho_cross", o.mean(), ttest_1samp(o, 0).pvalue, e.mean() if len(e) else np.nan, ttest_1samp(e, 0).pvalue if len(e) > 1 else np.nan)
    t = pd.read_csv(O / "F3/stage_rho.csv")
    t = t[t.latent.str.startswith("chimera") & (t.cohort == "test_3sp")]
    for (cl, a), q in t.groupby(["cls", "a"]):
        o, e = q[q.half == "discovery"].rho, q[q.half == "replication"].rho
        add(R, "F3", f"stage {cl} {a}", "test_3sp", "rho_cross", o.mean(), ttest_1samp(o, 0).pvalue, e.mean(), ttest_1samp(e, 0).pvalue)
    t = pd.read_csv(O / "F3/b1z_rho.csv")
    t = t[t.cohort.isin(CH) & ~t.y.str.contains("chimera")]
    for (c, y), q in t.groupby(["cohort", "y"]):
        if not {"odd", "even"} <= set(q.half):
            continue
        o, e = q[q.half == "odd"].iloc[0], q[q.half == "even"].iloc[0]
        z = lambda r: abs(r.absdelta_vs_B1z_chimera_mean) / max((r.absdelta_hi - r.absdelta_lo) / 3.92, 1e-6)
        add(R, "F3", f"B1z Chimera vs {y}", c, "B1z delta", -o.absdelta_vs_B1z_chimera_mean, float(2 * norm.sf(z(o))), -e.absdelta_vs_B1z_chimera_mean, float(2 * norm.sf(z(e))))
    if (O / "F3/within_rho.csv").exists():
        t = pd.read_csv(O / "F3/within_rho.csv")
        t = t[t.y.str.startswith("Swithin") & t.cohort.isin(CH)]
        for (c, l, g), q in t.groupby(["cohort", "label", "gold"]):
            o, e = q[q.half == "odd"].iloc[0], q[q.half == "even"].iloc[0]
            add(R, "F3", f"S_within {l} {g}", c, "S_within", o.rho_p, o.p, e.rho_p, e.p)
    # F4
    if (O / "F4/pairs_auc.csv").exists():
        t = pd.read_csv(O / "F4/pairs_auc.csv")
        t = t[t.cohort.isin(CH) & t.score.isin(["S", "S_given_PB", "S-PB"])]
        for (c, g, s), q in t.groupby(["cohort", "gold", "score"]):
            o, e = q[q.half == "odd"].iloc[0], q[q.half == "even"].iloc[0]
            off = 0 if s == "S-PB" else .5
            add(R, "F4", f"pairs {g}", c, s, o.auc - off, o.p_vs_half, e.auc - off, e.p_vs_half)
    if (O / "F4/modules_enrich.csv").exists():
        t = pd.read_csv(O / "F4/modules_enrich.csv")
        t = t[t.cohort.isin(CH) & (t.profile == "S")]
        for r in t.itertuples():
            add(R, "F4", f"module {r.module} {r.set}", r.cohort, "S module", 1.0, r.p_odd, 1.0 if r.ov_even > 0 else -1.0, r.p_even)
    if (O / "F4/blocks.csv").exists():
        t = pd.read_csv(O / "F4/blocks.csv")
        t = t[t.cohort.isin(CH) & t.y.isin(["S", "D_raw", "c_esm"])]
        w = t.pivot_table(index=["cohort", "y", "set"], columns="half", values=["mean", "p"])
        for (c, y, s), r in w.iterrows():
            add(R, "F4", f"block {s}", c, y, r[("mean", "odd")], r[("p", "odd")], r[("mean", "even")], r[("p", "even")])
    # F5
    if (O / "F5/exception_rho.csv").exists():
        t = pd.read_csv(O / "F5/exception_rho.csv")
        t = t[t.y.isin(["Xneg", "Xabs"]) & t.cohort.isin(CH)]
        for (c, k, y), q in t.groupby(["cohort", "k", "y"]):
            if {"odd", "even"} <= set(q.half):
                o, e = q[q.half == "odd"].iloc[0], q[q.half == "even"].iloc[0]
                add(R, "F5", f"exception {k}", c, y, o.rho_p, o.p, e.rho_p, e.p)
    # F6 (no odd/even possible: replication is the Murat gold, read by hand)
    t = pd.read_csv(O / "F5/xlinked.csv")
    t = t[t.cohort.isin(CH) & (t.y.str.startswith("C_") | t.y.isin(["S", "D_raw"]))]
    for r in t.itertuples():
        add(R, "F6", f"X linked {r.y}", r.cohort, r.y, r.rho_p, r.p)
    # F7
    t = pd.read_csv(O / "f7_paralogs.csv")
    t = t[t.cohort.isin(CH) & t.score.isin(["S", "D_raw", "c_esm"])]
    for r in t.itertuples():
        add(R, "F7", "paralog pairs", r.cohort, r.score, r.rho_p, r.p)
    T = pd.DataFrame(R)
    T["q"] = ST.bh(T.p_disc.fillna(1).clip(lower=1e-300).tolist())
    T["replicated"] = (np.sign(T.est_disc) == np.sign(T.est_rep)) & (T.p_rep / 2 < 0.05)
    T["survivor"] = (T.q < RU.ALPHA_Q) & T.replicated
    T["small"] = (T.est_disc.abs() < RU.SMALL) | (T.est_rep.abs() < RU.SMALL)
    T.to_csv(O / "tried.csv", index=False)
    S = T[T.survivor]
    S.to_csv(O / "survivors.csv", index=False)
    print(len(T), "tried;", (T.q < .05).sum(), "q<0.05;", len(S), "survive")
    print(T.groupby("family").agg(tried=("q", "size"), q05=("q", lambda x: (x < .05).sum()), surv=("survivor", "sum"), notsmall=("small", lambda x: 0)))


if __name__ == "__main__":
    main()
