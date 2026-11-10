"""X4 (EXPLORATORY): where Chimera differs from sequence constraint and direct baselines, and whether
independent data backs it. Definitions in rules.py. Nothing here changes a scoreboard output.

Sections (--sections): repro, map (card 1), right (card 2), disagree (card 3), explain (cards 4, 5),
x3, x1, x2 (item 7), drivers (item 8), adds (item 9). Writes results/analysis/X4/*.csv.
"""

from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from analysis.common import controls as K
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.T4_3_scoreboard import stats as S
from analysis.tasks.X4_where_differs import common as CM
from analysis.tasks.X4_where_differs import rules as RU
from analysis.tasks.X4_where_differs.common import matched, residual_rank, rho_unit, run_units, tertile

OUT = CM.OUT
C, CP = RU.C, RU.C_PAT
EL = ["log_cp10k_spmean", "log_cds_length"]
MAP_COHORTS = ["liv_hmouse", "test_hmouse", "liv_4sp"]
READERS = {"D": "D_raw", "S": "S", "B1b": "B1b_raw"}
X3_CLASSES = ["TF", "TF_nuclear_rec", "TF_homeodomain", "protein_kinase", "receptor", "enzyme", "EC6",
              "mitochondrial"]
X3_DIR = {"TF": 1, "TF_nuclear_rec": 1, "TF_homeodomain": 1, "protein_kinase": 1, "receptor": 1, "enzyme": -1,
          "EC6": -1, "mitochondrial": -1}
PROG_CORE = ["TAT", "HPD", "HAL", "GOT1", "BCKDHB", "PCCA", "ACMSD", "DHTKD1", "PCK1", "ALDOB", "PDK4", "ASL",
             "ACOX1", "CAT", "ALAS1"]
FASTING = ["PCK1", "PDK4", "TAT", "HPD", "ALAS1", "ACOX1"]
DRIVERS = {"Hepatocytes": "hepatocyte", "Endothelial": "endothelial", "fibroblasts": "stellate",
           "Kupffer cells": "macrophage"}
TESTIS_TYPES = {"SG": (["SG"], "Spermatogonia"), "SC": (["SC"], "Spermatocytes"),
                "SD": (["rSD", "eSD"], "spermatids"), "somatic": (["Sertoli", "Other_somatic"], "somatic")}


CLASS_DIR = {"constrained_lowD": -1, "relaxed_highD": 1, "B1bhigh_Dlow": -1, "B1blow_Dhigh": 1}


def write(df: pd.DataFrame, name: str) -> None:
    K.write_csv(df, OUT / name)
    print(f"  wrote {name} ({len(df)} rows)", flush=True)


def golds(c: str):
    return RU.GOLDS[CM.tissue(c)]


def prepare(c: str) -> pd.DataFrame:
    g = CM.load(c)
    for k in ("margin_support", "lacks_margin_support", "tail", "nameable", "sign_stable", "masked",
              "nonzero_all_species"):
        g[k] = g[k].eq(True)
    g["rD"] = residual_rank(g, "D_raw", C)
    g["rS"] = residual_rank(g, "S", C)
    g["tD_raw"] = tertile(g["D_raw"])
    g["tD_res"] = tertile(g["rD"])
    g["t_dNdS"] = tertile(g["dNdS"])
    g["t_pi"] = tertile(g["prior_raw"])
    g["ms"] = g["margin_support"].astype(float)
    return g


def constrained_masks(g, axis):
    a = RU.CONSTRAINT_AXES[axis]
    t = g[f"t_{axis}"]
    return t == a["constrained"], t == a["relaxed"], t == 2


# ---------------------------------------------------------------------------- reproduction
def repro(G):
    rows = []
    ref = pd.read_csv(CM.RES / "T4_3/family3.csv")
    sb = pd.read_csv(CM.RES / "T4_3/scoreboard.csv")
    for h, c, gold, covs in (("H1", "liv_hmouse", "G_H1", C), ("H2", "test_hmouse", "G_H2", C),
                             ("H3", "liv_hmouse", "G_H3", CP), ("H4", "test_hmouse", "G_H4", CP + ["H4_nsp"])):
        g = G[c]
        r0 = sb[(sb.hyp == h) & (sb.analysis == "primary") & (sb.score == "D")].rho_p.iloc[0]
        r1 = ref[(ref.hyp == h) & (ref.analysis == "S_ema_beside_D")].rho_p.iloc[0]
        r2 = ref[(ref.hyp == h) & (ref.analysis == "margin_with_support")].rho_p.iloc[0]
        a = rho_unit(g, gold, covs, ["D"], draws=10)[0]["rho_p"]
        b = rho_unit(g, gold, covs, ["S"], draws=10)[0]["rho_p"]
        d = rho_unit(g, gold, covs, ["D"], rows=g.margin_support, draws=10)[0]["rho_p"]
        rows.append(dict(hyp=h, D=a, D_T43=r0, S=b, S_T43=r1, D_margin=d, D_margin_T43=r2))
    t = pd.DataFrame(rows)
    write(t, "repro_check.csv")
    print(t.round(4).to_string())
    assert (t.D - t.D_T43).abs().max() < 1e-6
    assert (t.S - t.S_T43).abs().max() < 1e-6
    assert (t.D_margin - t.D_margin_T43).abs().max() < 1e-6


# ---------------------------------------------------------------------------- card 1
def conservation_map(G):
    cross, cls, assoc = [], [], []
    for c in MAP_COHORTS:
        g = G[c]
        for axis in ("dNdS", "pi"):
            col = RU.CONSTRAINT_AXES[axis]["col"]
            con, rel, _ = constrained_masks(g, axis)
            ok = g[col].notna()
            for score in ("D_raw", "rD", "S"):
                x = g.loc[ok & g[score].notna()]
                assoc.append(dict(cohort=c, axis=axis, score=score, n=len(x),
                                  spearman=spearmanr(x[col], x[score])[0]))
            for scale, tcol in (("raw", "tD_raw"), ("residual", "tD_res")):
                n = int((ok & g[tcol].notna()).sum())
                for tc, td in itertools.product((1, 2, 3), (1, 2, 3)):
                    m = (g[f"t_{axis}"] == tc) & (g[tcol] == td)
                    e = ((g[f"t_{axis}"] == tc) & g[tcol].notna()).sum() * ((g[tcol] == td) & ok).sum() / n
                    cross.append(dict(cohort=c, axis=axis, scale=scale, t_constraint=tc, t_D=td, n=int(m.sum()),
                                      expected=float(e), ratio=float(m.sum() / e)))
                defs = {"constrained_lowD": (con, 1), "relaxed_highD": (rel, 3),
                        "constrained_highD": (con, 3), "relaxed_lowD": (rel, 1)}
                for name, (mask, td) in defs.items():
                    f = (mask & (g[tcol] == td)).to_numpy()
                    tail = g["tail"].to_numpy()
                    rec = dict(cohort=c, axis=axis, scale=scale, cls=name, n=int(f.sum()),
                               frac_tail=float(tail[f].mean()),
                               frac_margin_support=float(g.margin_support[f].mean()),
                               frac_margin_support_of_tail=float(g.margin_support[f & tail].mean()) if (f & tail).sum() else np.nan,
                               frac_sign_stable=float(g.sign_stable[f].mean()), n_nameable=int(g.nameable[f].sum()),
                               frac_masked=float(g.masked[f].mean()),
                               median_expr=g.log_cp10k_spmean[f].median(), median_logCDS=g.log_cds_length[f].median(),
                               median_tau=g.tau_shared[f].median(), median_pi=g.prior_raw[f].median(),
                               median_dNdS=g.dNdS[f].median(), median_LOEUF=g.LOEUF[f].median(),
                               median_S=g.S[f].median(), median_B1b=g.B1b_raw[f].median())
                    other = {"constrained_lowD": "constrained_highD", "relaxed_highD": "relaxed_lowD"}.get(name)
                    if other:
                        pool = (defs[other][0] & (g[tcol] == defs[other][1])).to_numpy()
                        m = matched(g, f, pool, "ms", EL, key=f"{c}{axis}{scale}{name}")
                        rec.update(n_matched_pool=m["n_pool"], margin_support_set=m["mean_set"],
                                   margin_support_matched=m["mean_null"], margin_support_p=m["p"])
                    cls.append(rec)
    write(pd.DataFrame(cross), "map_crosstab.csv")
    write(pd.DataFrame(cls), "map_classes.csv")
    write(pd.DataFrame(assoc), "map_association.csv")


# ---------------------------------------------------------------------------- card 2
def who_is_right(G):
    units = []
    for c in MAP_COHORTS:
        g = G[c]
        base = [b for b in ["D_raw", "S", "B1b_raw", "B1c_raw", "B1z_raw", "B2b"] if g[b].notna().any()]
        if c.startswith("test"):
            base = ["D"] + base
        for gold, gname, covs in golds(c):
            for axis in ("dNdS", "pi"):
                con, rel, mid = constrained_masks(g, axis)
                for sname, mask in (("constrained", con), ("middle", mid), ("relaxed", rel),
                                    ("all_with_axis", con | mid | rel)):
                    lab = dict(cohort=c, axis=axis, stratum=sname, universe="common")
                    units.append(dict(g=g, gold=gold, covs=covs, scores=base, rows=mask.to_numpy(), label=lab))
                    units.append(dict(g=g, gold=gold, covs=covs, scores=["D_raw", "S"], rows=mask.to_numpy(),
                                      label=dict(lab, universe="own_D_S")))
                    units.append(dict(g=g, gold=gold, covs=covs, scores=["D_raw", "S", "B1b_raw"],
                                      rows=(mask & g.margin_support).to_numpy(),
                                      label=dict(lab, universe="margin_supported")))
    write(run_units(units), "right_strata.csv")

    # discordant vs concordant within a constraint stratum, per reader
    rows = []
    for c in MAP_COHORTS:
        g = G[c].copy()
        uni = g[["D_raw", "S", "B1b_raw"]].notna().all(1)
        rrB = residual_rank(g[uni], "B1b_raw", C)
        rrD = residual_rank(g[uni], "D_raw", C)
        rrS = residual_rank(g[uni], "S", C)
        for nm, r in (("D", rrD), ("S", rrS), ("B1b", rrB)):
            g[f"t_{nm}_u"] = tertile(r.reindex(g.index))
        g["tD_rawu"] = tertile(g["D_raw"].where(uni))
        for gold, gname, covs in golds(c):
            for axis in ("dNdS", "pi"):
                con, rel, _ = constrained_masks(g, axis)
                for reader, tcol in (("D_residual", "t_D_u"), ("D_rawtertile", "tD_rawu"), ("S", "t_S_u"), ("B1b", "t_B1b_u")):
                    for name, stratum, setT, poolT in (("constrained: low vs high", con, 1, 3),
                                                       ("relaxed: high vs low", rel, 3, 1)):
                        for sub, smask in (("all", np.ones(len(g), bool)), ("margin_supported", g.margin_support.to_numpy())):
                            if sub == "margin_supported" and reader == "B1b":
                                continue
                            f = (stratum & (g[tcol] == setT)).to_numpy() & smask
                            p = (stratum & (g[tcol] == poolT)).to_numpy() & smask
                            for mname, mc in (("expr_len", EL), ("covs", covs)):
                                m = matched(g, f, p, gold, mc, key=f"{c}{axis}{reader}{name}{sub}{mname}")
                                rows.append(dict(cohort=c, gold=gold, axis=axis, reader=reader, contrast=name, subset=sub,
                                                 matching=mname, **m))
    write(pd.DataFrame(rows), "right_contrasts.csv")


# ---------------------------------------------------------------------------- card 3
def atlas_columns(g):
    a = pd.read_parquet(CM.CACHE / "atlas_h3.parquet")
    g = g.join(a[[k for k in a.columns if k not in g.columns]], on="hsapiens_gene_id")
    for t in ("hepatocyte", "endothelial", "stellate", "macrophage"):
        d = g[f"TMS_x_{t}"] - g[f"TS_x_{t}"]
        ok = (g[f"TMS_f_{t}"] >= 0.01) & (g[f"TS_f_{t}"] >= 0.01)
        g[f"atlas_lfc_{t}"] = d - d[ok].median()
        g.loc[~ok, f"atlas_lfc_{t}"] = np.nan
    g["atlas_shift"] = g[["atlas_lfc_hepatocyte", "atlas_lfc_endothelial", "atlas_lfc_stellate"]].mean(1, skipna=False)
    g["abs_atlas_shift"] = g["atlas_shift"].abs()
    g["abs_atlas_hep"] = g["atlas_lfc_hepatocyte"].abs()
    return g


def disagreement(G):
    cls_rows, lvl_rows, gold_rows, part_rows, rep_rows = [], [], [], [], []
    units = []
    out = {}
    for c in RU.PRIMARY:
        g = G[c].copy()
        if c == "liv_hmouse":
            g = atlas_columns(g)
        uni = g["B1b_raw"].notna()
        for nm, col in (("B", "B1b_raw"), ("D", "D_raw"), ("Sc", "S")):
            g[f"rp{nm}"] = residual_rank(g[uni], col, CP).reindex(g.index)
            g[f"tp{nm}"] = tertile(g[f"rp{nm}"])
        g["trawB"] = tertile(g["B1b_raw"].where(uni))
        g["trawD"] = tertile(g["D_raw"].where(uni))
        for reader, tc, trawc in (("D", "tpD", "trawD"), ("S", "tpSc", None)):
            for name, tb, tr, ctrl in (("B1bhigh_low", 3, 1, "B1bhigh_high"), ("B1blow_high", 1, 3, "B1blow_low")):
                f = ((g.tpB == tb) & (g[tc] == tr)).to_numpy()
                p = ((g.tpB == tb) & (g[tc] == (4 - tr))).to_numpy()
                tail = g["tail"].to_numpy()
                nm = f"{name}__{reader}"
                g[f"cls_{nm}"] = f
                exp = ((g.tpB == tb).sum() * (g[tc] == tr).sum()) / uni.sum()
                rec = dict(cohort=c, reader=reader, cls=name, n=int(f.sum()), expected=float(exp), n_control=int(p.sum()),
                           frac_margin_support=float(g.margin_support[f].mean()),
                           frac_sign_stable=float(g.sign_stable[f].mean()), n_nameable=int(g.nameable[f].sum()),
                           median_B1b=g.B1b_raw[f].median(), median_expr=g.log_cp10k_spmean[f].median(),
                           median_tau=g.tau_shared[f].median(), median_abs_shift=g.abs_shift[f].median(),
                           median_shape=g["shape"][f].median(), median_frac_level=g.frac_level[f].median(),
                           median_spread=g.spread[f].median(),
                           control_median_abs_shift=g.abs_shift[p].median(), control_median_shape=g["shape"][p].median(),
                           control_median_frac_level=g.frac_level[p].median(), control_median_spread=g.spread[p].median())
                if trawc:
                    fr = ((g.trawB == tb) & (g[trawc] == tr)).to_numpy()
                    rec.update(n_rawtertiles=int(fr.sum()), overlap_with_raw=float((fr & f).sum() / max(f.sum(), 1)))
                cls_rows.append(rec)
                mc = CP + ["rpB"]
                for v in ("abs_shift", "shape", "frac_level", "spread", "msd"):
                    m = matched(g, f, p, v, mc, key=f"{c}{nm}{v}")
                    lvl_rows.append(dict(cohort=c, reader=reader, cls=name, value=v, matching="C_pat+rB1b", **m))
                for gold, _, covs in golds(c):
                    m = matched(g, f, p, gold, list(dict.fromkeys(covs + ["rpB"])), key=f"{c}{nm}{gold}")
                    gold_rows.append(dict(cohort=c, reader=reader, cls=name, gold=gold, subset="all", **m))
                    fm, pm = f & g.margin_support.to_numpy(), p & g.margin_support.to_numpy()
                    m = matched(g, fm, pm, gold, list(dict.fromkeys(covs + ["rpB"])), key=f"{c}{nm}{gold}ms")
                    gold_rows.append(dict(cohort=c, reader=reader, cls=name, gold=gold, subset="margin_supported", **m))
                if c == "liv_hmouse":
                    for v in ("abs_atlas_shift", "abs_atlas_hep"):
                        m = matched(g, f, p, v, mc, key=f"{c}{nm}{v}")
                        gold_rows.append(dict(cohort=c, reader=reader, cls=name, gold=v, subset="all", **m))
                    for v in ("all", "class", "control"):
                        mask = {"all": uni.to_numpy(), "class": f, "control": p}[v]
                        x = g.loc[mask, ["shift", "atlas_shift", "lfc_Hepatocytes", "atlas_lfc_hepatocyte"]]
                        y = x.dropna(subset=["shift", "atlas_shift"])
                        z = x.dropna(subset=["lfc_Hepatocytes", "atlas_lfc_hepatocyte"])
                        rep_rows.append(dict(cohort=c, reader=reader, cls=name, group=v, n_shift=len(y),
                                             rho_shift=spearmanr(y["shift"], y.atlas_shift)[0],
                                             sign_agree_shift=float((np.sign(y["shift"]) == np.sign(y.atlas_shift)).mean()),
                                             n_hep=len(z), rho_hep=spearmanr(z.lfc_Hepatocytes, z.atlas_lfc_hepatocyte)[0],
                                             sign_agree_hep=float((np.sign(z.lfc_Hepatocytes) == np.sign(z.atlas_lfc_hepatocyte)).mean())))
        # continuous: is the D/B1b disagreement level?
        for reader, col in (("D", "D_raw"), ("S", "S")):
            units.append(dict(g=g, gold=col, covs=CP, scores=["abs_shift", "shape", "frac_level", "spread"],
                              extra_spline=["B1b_raw"], label=dict(cohort=c, reader=reader, given="C_pat+B1b")))
            units.append(dict(g=g, gold=col, covs=CP, scores=["abs_shift", "shape", "frac_level", "spread"],
                              label=dict(cohort=c, reader=reader, given="C_pat")))
            units.append(dict(g=g, gold="B1b_raw", covs=CP, scores=["abs_shift", "shape", "frac_level", "spread"],
                              label=dict(cohort=c, reader="B1b", given="C_pat")))
            T = g[[col, "B1b_raw", "abs_shift", "shape", "frac_level", "spread"] + CP + ["block"]].dropna()
            y = S.midrank(T[col].to_numpy())
            Z0 = S.design(T, CP + ["B1b_raw"])
            sse0 = float(((y - Z0 @ np.linalg.lstsq(Z0, y, rcond=None)[0]) ** 2).sum())
            for nm, add in (("abs_shift", ["abs_shift"]), ("shape", ["shape"]), ("shift+shape", ["abs_shift", "shape"]),
                            ("shift+shape+frac+spread", ["abs_shift", "shape", "frac_level", "spread"])):
                Z1 = S.design(T, CP + ["B1b_raw"] + add)
                sse1 = float(((y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]) ** 2).sum())
                part_rows.append(dict(cohort=c, reader=reader, added=nm, n=len(T), partial_R2=1 - sse1 / sse0))
        out[c] = g
    write(pd.DataFrame(cls_rows), "disagree_classes.csv")
    write(pd.DataFrame(lvl_rows), "disagree_level.csv")
    write(pd.DataFrame(gold_rows), "disagree_gold.csv")
    write(pd.DataFrame(part_rows), "disagree_partial_R2.csv")
    if rep_rows:
        write(pd.DataFrame(rep_rows), "disagree_level_replication.csv")
    write(run_units(units), "disagree_continuous.csv")
    for c, g in out.items():
        G[c] = g


def head_to_head(G):
    """Symmetric test of a disagreement, added after the one-sided contrasts of ``disagreement`` were seen:
    those condition on B1b, which favours any second noisy measure of the same quantity. Here the two
    opposite classes meet directly. excess = gold(B1b high, X low) - gold(B1b low, X high), matched on the
    gold's covariates; > 0 sides with B1b, < 0 with X. Baseline pairs calibrate what two measures give."""
    rows = []
    for c in RU.PRIMARY:
        g = G[c].join(pd.read_parquet(CM.CACHE / f"Sdir__{c}.parquet").set_index("var_name")[["Sdir"]], on="var_name")
        for x in ("D_raw", "S", "Sdir", "B1z_raw", "B2b", "B1c_raw"):
            uni = g[["B1b_raw", x]].notna().all(1)
            rb = residual_rank(g[uni], "B1b_raw", CP).reindex(g.index)
            rx = residual_rank(g[uni], x, CP).reindex(g.index)
            tb, tx = tertile(rb), tertile(rx)
            f = ((tb == 3) & (tx == 1)).to_numpy()
            p = ((tb == 1) & (tx == 3)).to_numpy()
            for gold, _, covs in golds(c):
                m = matched(g, f, p, gold, covs, key=f"h2h{c}{x}{gold}")
                rows.append(dict(cohort=c, X=x, gold=gold, n_B1bhigh_Xlow=int(f.sum()), n_B1blow_Xhigh=int(p.sum()), **m))
    write(pd.DataFrame(rows), "disagree_head_to_head.csv")


# ---------------------------------------------------------------------------- cards 4, 5
def t25_long(c, kind):
    return pd.read_parquet(CM.RES / f"T2_5/{c}__{kind}.parquet")


def explain(G):
    """Class-level attribution, named genes, vignettes."""
    if "cls_B1bhigh_low__D" not in G["liv_hmouse"]:
        disagreement(G)
    named, cls_rows, vig = [], [], []
    for c in RU.PRIMARY:
        g = G[c]
        typ = [k for k in g.columns if k.startswith("C_") and not k.startswith("Csp_")]
        sp_ = [k for k in g.columns if k.startswith("Csp_")]
        sets = {"constrained_lowD": ((g.t_dNdS == 1) & (g.tD_res == 1)).to_numpy(),
                "relaxed_highD": ((g.t_dNdS == 3) & (g.tD_res == 3)).to_numpy(),
                "constrained_highD": ((g.t_dNdS == 1) & (g.tD_res == 3)).to_numpy(),
                "relaxed_lowD": ((g.t_dNdS == 3) & (g.tD_res == 1)).to_numpy(),
                "B1bhigh_Dlow": g["cls_B1bhigh_low__D"].to_numpy(), "B1blow_Dhigh": g["cls_B1blow_high__D"].to_numpy(),
                "B1bhigh_Slow": g["cls_B1bhigh_low__S"].to_numpy(), "B1blow_Shigh": g["cls_B1blow_high__S"].to_numpy()}
        Sg = t25_long(c, "S_gsk")
        if c == "test_hmouse":
            Sg["cell_type"] = Sg["cell_type"].map(T42.TESTIS_COARSE).fillna(Sg["cell_type"])
            Sg = Sg[Sg.group_key == "cell_type"]
        Sg = Sg[Sg.group_key == "cell_type"] if "group_key" in Sg else Sg
        for nm, f in sets.items():
            for sub, m in (("all", f), ("nameable", f & g.nameable.to_numpy())):
                if m.sum() == 0:
                    continue
                T = g.loc[m]
                rec = dict(cohort=c, cls=nm, subset=sub, n=int(m.sum()), mean_S=T.S.mean(), mean_D_raw=T.D_raw.mean())
                denom = T[typ].abs().sum(1).sum()
                for k in typ:
                    rec[f"share_{k}"] = float(T[k].sum() / T.S.sum()) if abs(T.S.sum()) > 1e-9 else np.nan
                    rec[f"absshare_{k}"] = float(T[k].abs().sum() / denom)
                for k in sp_:
                    rec[f"share_{k}"] = float(T[k].sum() / T.S.sum()) if abs(T.S.sum()) > 1e-9 else np.nan
                vs = Sg[Sg.var_name.isin(T.var_name)]
                w = vs.groupby(["species", "cell_type"]).agg(u=("S_gsk_meanz", "mean"), wt=("S_gsk_within", "mean"),
                                                            weight=("weight", "first"))
                top = w.reindex(w.u.abs().sort_values(ascending=False).index).head(4)
                rec["top_unweighted"] = "; ".join(f"{a}|{b} {r.u:+.3f} (within {r.wt:+.3f})" for (a, b), r in top.iterrows())
                cls_rows.append(rec)
            names = g.loc[f & g.nameable.to_numpy()].copy()
            names["cls"] = nm
            named.append(names)
        # vignettes (rules.py): top by |rD| then |S|, S must point the way of the class (amendment)
        for nm in ("constrained_lowD", "relaxed_highD", "B1bhigh_Dlow", "B1blow_Dhigh"):
            T = g.loc[sets[nm] & g.nameable.to_numpy() & (np.sign(g.S.to_numpy()) == CLASS_DIR[nm])].copy()
            T["a"], T["b"] = T.rD.abs(), T.S.abs()
            T = T.sort_values(["a", "b"], ascending=False).head(5)
            T["cls"] = nm
            T["rank_in_class"] = np.arange(1, len(T) + 1)
            T["cohort"] = c
            vig.append(T)
    gene_cols = ["cohort", "cls", "rank_in_class", "symbol", "hsapiens_gene_id", "prior_raw", "dNdS", "LOEUF", "D_raw", "rD",
                 "S", "B1b_raw", "B1c_raw", "B1z_raw", "signfrac_D", "signfrac_S", "margin_support", "log_cp10k_spmean",
                 "tau_shared", "G_H1", "G_H2", "G_H3", "G_H4", "abs_shift", "shape", "frac_level", "shift"]
    V = pd.concat(vig, ignore_index=True)
    keep = gene_cols + [k for k in V.columns if k.startswith(("C_", "Csp_", "x_", "f_", "TS_x_", "TMS_x_"))]
    write(V[[k for k in keep if k in V.columns]], "vignettes.csv")
    N = []
    for T in named:
        if len(T):
            T = T.assign(cohort=T.get("cohort", None))
            N.append(T)
    allN = pd.concat(N, ignore_index=True)
    allN["cohort"] = np.where(allN.var_name.isin(G["liv_hmouse"].var_name) & allN.block.notna(), allN.get("cohort"), allN.get("cohort"))
    write(pd.DataFrame(cls_rows), "explain_classes.csv")
    cols = ["cls", "symbol", "hsapiens_gene_id", "prior_raw", "dNdS", "D_raw", "rD", "S", "B1b_raw", "B1c_raw", "signfrac_D",
            "signfrac_S", "abs_shift", "shape", "frac_level", "G_H1", "G_H2", "G_H3", "G_H4"]
    rows = []
    for c in RU.PRIMARY:
        g = G[c]
        for nm, f in (("constrained_lowD", ((g.t_dNdS == 1) & (g.tD_res == 1))), ("relaxed_highD", ((g.t_dNdS == 3) & (g.tD_res == 3))),
                      ("B1bhigh_Dlow", g["cls_B1bhigh_low__D"]), ("B1blow_Dhigh", g["cls_B1blow_high__D"])):
            T = g.loc[f.to_numpy() & g.nameable.to_numpy(), cols[1:]].assign(cohort=c, cls=nm)
            T["class_backed"] = np.sign(T.S) == CLASS_DIR[nm]
            rows.append(T)
    write(pd.concat(rows, ignore_index=True), "named_genes.csv")
    # per species x type for vignette genes
    sg_rows = []
    for c in RU.PRIMARY:
        L = t25_long(c, "S_gsk")
        L = L[L.group_key == "cell_type"] if "group_key" in L else L
        v = V[(V.cohort == c) & (V.rank_in_class == 1)]
        L = L[L.hsapiens_gene_id.isin(v.hsapiens_gene_id)]
        sg_rows.append(L.assign(cohort=c))
    write(pd.concat(sg_rows, ignore_index=True), "vignettes_S_gsk.csv")


# ---------------------------------------------------------------------------- item 7
def re_x3(G):
    rows = []
    for c in RU.COHORTS:
        g = G[c].copy()
        liver = c.startswith("liv")
        extra = ["hep_enrich"] if liver else []
        g["nonesm_D"] = CM.non_esm(g, "D_raw", C)
        g["nonesm_S"] = CM.non_esm(g, "S", C)
        for k in X3_CLASSES:
            f = g[f"cls_{k}"].eq(True).to_numpy()
            pool = ~f
            ms = g.margin_support.to_numpy()
            for v in ("D_raw", "S", "nonesm_D", "nonesm_S", "B1b_raw", "B1c_raw", "B1z_raw"):
                for mname, mc in (("C", C), ("C_pat_dom", CP + extra)):
                    rows.append(dict(cohort=c, cls=k, value=v, kind="readout", subset="all", matching=mname,
                                     **matched(g, f, pool, v, mc, key=f"{c}{k}{v}{mname}")))
                if v in ("D_raw", "S"):
                    rows.append(dict(cohort=c, cls=k, value=v, kind="readout", subset="margin_supported", matching="C",
                                     **matched(g, f & ms, pool & ms, v, C, key=f"{c}{k}{v}ms")))
            for gold, _, covs in golds(c):
                for mname, mc in (("covs", covs), ("covs_dom", covs + extra)):
                    rows.append(dict(cohort=c, cls=k, value=gold, kind="gold", subset="all", matching=mname,
                                     **matched(g, f, pool, gold, mc, key=f"{c}{k}{gold}{mname}")))
                rows.append(dict(cohort=c, cls=k, value=gold, kind="gold", subset="margin_supported", matching="covs",
                                 **matched(g, f & ms, pool & ms, gold, covs, key=f"{c}{k}{gold}ms")))
    t = pd.DataFrame(rows)
    t["predicted_sign"] = t.cls.map(X3_DIR)
    write(t, "reread_x3_classes.csv")


def re_x1(G):
    rows, nam = [], []
    for c in ("test_hmouse", "test_3sp"):
        g = G[c].copy()
        g["nonesm_D"] = CM.non_esm(g, "D_raw", C)
        g["nonesm_S"] = CM.non_esm(g, "S", C)
        ms = g.margin_support.to_numpy()
        groups = {f"{k}": (g.peak == k).to_numpy() for k in ("SG", "SC", "rSD", "eSD", "Sertoli", "Other_somatic")}
        groups["SC_strict"] = (g.peak_strict == "SC").to_numpy()
        for k, f in groups.items():
            pool = ~f
            for v in ("D_raw", "D", "S", "nonesm_D", "nonesm_S", "B1b_raw", "B1c_raw", "B1z_raw"):
                rows.append(dict(cohort=c, group=k, value=v, kind="readout", subset="all",
                                 **matched(g, f, pool, v, CP, key=f"{c}{k}{v}")))
            for v in ("D_raw", "S"):
                rows.append(dict(cohort=c, group=k, value=v, kind="readout", subset="margin_supported",
                                 **matched(g, f & ms, pool & ms, v, CP, key=f"{c}{k}{v}ms")))
            for gold, _, covs in golds(c):
                rows.append(dict(cohort=c, group=k, value=gold, kind="gold", subset="all",
                                 **matched(g, f, pool, gold, covs, key=f"{c}{k}{gold}")))
                rows.append(dict(cohort=c, group=k, value=gold, kind="gold", subset="margin_supported",
                                 **matched(g, f & ms, pool & ms, gold, covs, key=f"{c}{k}{gold}ms")))
        if c == "test_hmouse":
            for gold, _, covs in golds(c):
                g[f"r_{gold}"] = residual_rank(g, gold, covs)
            lst = ["ANKRD62", "LY6K", "LARP1B", "NUP210L", "GLIPR1L2", "ART3", "ADAM2", "EFCAB5", "IZUMO4", "PKD2L2",
                   "SLC9C1", "DNAAF1", "DPY19L2", "STK33", "LDHC", "ZPBP2", "TEX101"]
            T = g[g.symbol.isin(lst)]
            nam.append(T[["symbol", "peak", "peak_strict", "prior_raw", "dNdS", "D_raw", "rD", "S", "B1b_raw", "signfrac_D",
                          "signfrac_S", "margin_support", "nameable", "G_H2", "G_H4", "G_H4_primates",
                          "r_G_H2", "r_G_H4", "r_G_H4_primates"]])
    write(pd.DataFrame(rows), "reread_x1_stages.csv")
    write(pd.concat(nam), "reread_x1_named.csv")


def gdrv(g):
    prof = np.load(CM.RES / "T4_3/golds/h3_profiles.npz", allow_pickle=True)
    types = list(prof["types"])
    n = len(prof["genes"])
    out = {}
    for k, t in enumerate(types):
        rt = rankdata(prof["TS_x"][:, k]) / n
        rm = rankdata(prof["TMS_x"][:, k]) / n
        ok = (prof["TS_f"][:, k] >= 0.01) & (prof["TMS_f"][:, k] >= 0.01)
        out[f"Gdrv_{t}"] = np.where(ok, -np.abs(rt - rm), np.nan)
    d = pd.DataFrame(out, index=pd.Index(prof["genes"], name="hsapiens_gene_id"))
    return g.join(d, on="hsapiens_gene_id")


def re_x2(G):
    lst = pd.read_csv(CM.RES / "X2/divergent_conserved.csv")
    lst = lst[lst.cohort == "liv_4sp"]
    prog = sorted(set(lst.symbol) - {"MT-CO1"})
    sets = {"prog63_noMT": prog, "core15": PROG_CORE, "fasting6": FASTING}
    rows, gene_rows, cal = [], [], []
    rng = np.random.default_rng(CM.useed("x2cal"))
    for c in ("liv_4sp", "liv_hmouse"):
        g = gdrv(atlas_columns(G[c].copy()))
        g["atlas_donorSD_TS"] = g["TS_donorSD_hepatocyte"]
        g["atlas_donorSD_TMS"] = g["TMS_donorSD_hepatocyte"]
        g["atlas_expr_hep"] = (g["TS_x_hepatocyte"] + g["TMS_x_hepatocyte"]) / 2
        g["rGH1"] = residual_rank(g, "G_H1", C)
        ms = g.margin_support.to_numpy()
        chep = CP + ["hep_enrich"]
        vals = [("D_raw", "readout"), ("S", "readout"), ("B1b_raw", "baseline"), ("B1c_raw", "baseline"), ("B1z_raw", "baseline"),
                ("G_H1", "gold"), ("G_H3", "gold"), ("Gdrv_hepatocyte", "gold_type"), ("abs_atlas_hep", "gold_gap")]
        for k, names in sets.items():
            f = g.symbol.isin(names).to_numpy()
            pool = ~g.symbol.isin(prog).to_numpy()
            for v, kind in vals:
                rows.append(dict(cohort=c, set=k, value=v, kind=kind, matching="C_hep",
                                 **matched(g, f, pool, v, chep, key=f"{c}{k}{v}")))
            for v in ("atlas_donorSD_TS", "atlas_donorSD_TMS"):
                rows.append(dict(cohort=c, set=k, value=v, kind="state", matching="C_hep+atlas_expr",
                                 **matched(g, f, pool, v, chep + ["atlas_expr_hep"], key=f"{c}{k}{v}")))
            T = g.loc[f].dropna(subset=["lfc_Hepatocytes", "atlas_lfc_hepatocyte"]) if "lfc_Hepatocytes" in g else None
            if T is not None and len(T) > 4:
                rows.append(dict(cohort=c, set=k, value="sign_agree_hep_lfc_camex_atlas", kind="replication", matching="none",
                                 n_set=len(T), mean_set=float((np.sign(T.lfc_Hepatocytes) == np.sign(T.atlas_lfc_hepatocyte)).mean())))
        if c == "liv_hmouse":
            hepgenes = (g.hep_enrich >= g.hep_enrich.quantile(0.75)).to_numpy() & ~g.symbol.isin(prog).to_numpy()
            T = g.loc[hepgenes].dropna(subset=["lfc_Hepatocytes", "atlas_lfc_hepatocyte"])
            rows.append(dict(cohort=c, set="hepatocyte_enriched_background", value="sign_agree_hep_lfc_camex_atlas",
                             kind="replication", matching="none", n_set=len(T),
                             mean_set=float((np.sign(T.lfc_Hepatocytes) == np.sign(T.atlas_lfc_hepatocyte)).mean())))
        # calibration of the C_hep null on random hepatocyte-enriched sets of size 63
        hp = np.where((g.hep_enrich >= g.hep_enrich.quantile(0.75)).to_numpy() & ~g.symbol.isin(prog).to_numpy())[0]
        for v in ("G_H1", "G_H3", "Gdrv_hepatocyte", "abs_atlas_hep", "S"):
            ps, es = [], []
            for i in range(200):
                f = np.zeros(len(g), bool)
                f[rng.choice(hp, 63, replace=False)] = True
                m = matched(g, f, ~g.symbol.isin(prog).to_numpy() & ~f, v, chep, key=f"cal{c}{v}{i}")
                ps.append(m["p"])
                es.append(m["excess_d"])
            cal.append(dict(cohort=c, value=v, n_sets=200, frac_p05=float(np.mean(np.array(ps) < 0.05)), mean_excess=float(np.nanmean(es))))
        keep = ["symbol", "prior_raw", "D_raw", "S", "signfrac_D", "signfrac_S", "margin_support", "nameable", "B1b_raw", "B1c_raw", "G_H1",
                "G_H3", "Gdrv_hepatocyte", "atlas_lfc_hepatocyte", "TS_donorSD_hepatocyte", "TMS_donorSD_hepatocyte", "hep_enrich"]
        if "lfc_Hepatocytes" in g:
            keep.append("lfc_Hepatocytes")
        for s in PROG_CORE:
            T = g[g.symbol == s]
            if len(T):
                gene_rows.append(T[keep].assign(cohort=c))
    write(pd.DataFrame(rows), "reread_x2_program.csv")
    write(pd.concat(gene_rows, ignore_index=True), "reread_x2_named.csv")
    write(pd.DataFrame(cal), "reread_x2_calibration.csv")


# ---------------------------------------------------------------------------- item 8
def per_type_direct(g, types_h, types_m, name):
    """-|rank_H - rank_M| within the type, mean over the listed stages; detection >= 1% in both."""
    n = len(g)
    H = g[[f"x_Human_{t}" for t in types_h]].mean(1).to_numpy()
    M = g[[f"x_Mouse_{t}" for t in types_m]].mean(1).to_numpy()
    fH = g[[f"f_Human_{t}" for t in types_h]].max(1).to_numpy()
    fM = g[[f"f_Mouse_{t}" for t in types_m]].max(1).to_numpy()
    b = -np.abs(rankdata(H) / n - rankdata(M) / n)
    return pd.Series(np.where((fH >= 0.01) & (fM >= 0.01), b, np.nan), index=g.index, name=name), pd.Series((H + M) / 2, index=g.index)


def split_half_direct(c):
    """Cell-half reliability of the per-type direct baseline (10 random cell halves)."""
    a = T42_adata(c)
    X, spc, lab, lib = a
    rng = np.random.default_rng(CM.useed("bk_rel", c))
    rows = []
    types = sorted(set(lab[spc == "Human"]) & set(lab[spc == "Mouse"]))
    for i in range(10):
        h = rng.random(len(lab)) < 0.5
        res = []
        for side in (h, ~h):
            x, f, _ = T42.pseudobulk(X[side], spc[side], lab[side], lib[side], 15)
            res.append((x, f))
        for t in types:
            if ("Human", t) not in res[0][0] or ("Human", t) not in res[1][0] or ("Mouse", t) not in res[0][0] or ("Mouse", t) not in res[1][0]:
                continue
            n = X.shape[1]
            bs = []
            for x, f in res:
                b = -np.abs(rankdata(x[("Human", t)]) / n - rankdata(x[("Mouse", t)]) / n)
                ok = (f[("Human", t)] >= 0.01) & (f[("Mouse", t)] >= 0.01)
                bs.append(np.where(ok, b, np.nan))
            ok = np.isfinite(bs[0]) & np.isfinite(bs[1])
            rows.append(dict(cohort=c, type=t, split=i, r=np.corrcoef(rankdata(bs[0][ok]), rankdata(bs[1][ok]))[0, 1], n=int(ok.sum())))
    t = pd.DataFrame(rows).groupby(["cohort", "type"]).agg(r=("r", "mean"), n=("n", "mean")).reset_index()
    t["rel"] = 2 * t.r / (1 + t.r)
    return t


def T42_adata(c):
    import scipy.sparse as sp
    from analysis.common import cohorts as CC
    a = CC.read_adata(c)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    lab = a.obs["cell_type"].astype(str)
    if c.startswith("test"):
        lab = lab.map(T42.TESTIS_COARSE)
    tis = "testis" if c.startswith("test") else "liver"
    lib = a.obs[T42.LIBRARY_KEY[tis]].astype(str).to_numpy()
    return X, spc, lab.to_numpy(), lib


def drivers(G):
    units, meta = [], []
    # liver
    for c in ("liv_hmouse", "liv_4sp"):
        g = gdrv(G[c].copy())
        a = pd.read_parquet(CM.CACHE / "atlas_h3.parquet")
        g = g.join(a[[k for k in a.columns if k not in g.columns]], on="hsapiens_gene_id")
        if c == "liv_hmouse":
            sd = pd.read_parquet(CM.CACHE / "Sdir__liv_hmouse.parquet").set_index("var_name")
            g = g.join(sd, on="var_name")
            cgk = [pd.read_parquet(CM.RES / f"T2_5/liv_hmouse{s}__C_gk.parquet") for s in ("", "_s43", "_s44")]
            cgk = [x[x.group_key == "cell_type"] for x in cgk]
            within = cgk[0].pivot(index="var_name", columns="cell_type", values="C_gk_within")
            mean3 = sum(x.pivot(index="var_name", columns="cell_type", values="C_gk_meanz") for x in cgk) / 3
        for k, t in DRIVERS.items():
            col = f"C_{k}"
            if col not in g.columns or f"Gdrv_{t}" not in g:
                continue
            aexpr = (g[f"TS_x_{t}"] + g[f"TMS_x_{t}"]) / 2
            T = g.copy()
            T["aexpr"] = aexpr
            scores = [col, "D_raw", "S", "B1c_raw"]
            lab = dict(cohort=c, type=k)
            if c == "liv_hmouse":
                if f"x_Human_{k}" in g and f"x_Mouse_{k}" in g:
                    T["Bk"], T["texpr"] = per_type_direct(g, [k], [k], "Bk")
                    scores = scores + ["Bk"]
                elif k == "Kupffer cells":  # human label Macrophages, mouse label Kupffer cells
                    T["Bk"], T["texpr"] = per_type_direct(g, ["Macrophages"], ["Kupffer cells"], "Bk")
                    scores = scores + ["Bk"]
                else:
                    T["texpr"] = T["log_cp10k_spmean"]
                T["Cwithin"] = within[k].reindex(T.var_name).to_numpy()
                T["C3seed"] = mean3[k].reindex(T.var_name).to_numpy()
                if f"Cdir_{k}" in T:
                    scores = scores + [f"Cdir_{k}"]
                scores = scores + ["Cwithin", "C3seed"]
                units.append(dict(g=T, gold=f"Gdrv_{t}", covs=C, scores=scores, label=dict(lab, given="C"), ref=col))
                units.append(dict(g=T, gold=f"Gdrv_{t}", covs=C, scores=scores, extra_spline=["texpr"],
                                  label=dict(lab, given="C+type_expr"), ref=col))
                units.append(dict(g=T, gold=f"Gdrv_{t}", covs=C, scores=scores, extra_spline=["texpr", "aexpr"],
                                  label=dict(lab, given="C+type_expr+atlas_expr"), ref=col))
            else:
                units.append(dict(g=T, gold=f"Gdrv_{t}", covs=C, scores=scores, label=dict(lab, given="C"), ref=col))
                units.append(dict(g=T, gold=f"Gdrv_{t}", covs=C, scores=scores, extra_spline=["aexpr"],
                                  label=dict(lab, given="C+atlas_expr"), ref=col))
    # testis
    c = "test_hmouse"
    g = G[c].copy()
    g = g.join(pd.read_parquet(CM.CACHE / "atlas_h4_types.parquet"), on="hsapiens_gene_id")
    sd = pd.read_parquet(CM.CACHE / "Sdir__test_hmouse.parquet").set_index("var_name")
    g = g.join(sd, on="var_name")
    for k, (st, t) in TESTIS_TYPES.items():
        T = g.copy()
        T["Ck"] = g[[f"C_{s}" for s in st]].sum(1)
        T["Cdir_k"] = g[[f"Cdir_{s}" for s in st]].sum(1)
        T["Bk"], T["texpr"] = per_type_direct(g, st, st, "Bk")
        scores = ["Ck", "Bk", "Cdir_k", "D_raw", "D", "S", "B1c_raw"]
        lab = dict(cohort=c, type=k)
        units.append(dict(g=T, gold=f"Gdrv4_{t}", covs=C, scores=scores, label=dict(lab, given="C"), ref="Ck"))
        units.append(dict(g=T, gold=f"Gdrv4_{t}", covs=C, scores=scores, extra_spline=["texpr"],
                          label=dict(lab, given="C+type_expr"), ref="Ck"))
        T["aexpr"] = g[f"H4_x_{t}"]
        units.append(dict(g=T, gold=f"Gdrv4_{t}", covs=C, scores=scores, extra_spline=["texpr", "aexpr"],
                          label=dict(lab, given="C+type_expr+atlas_expr"), ref="Ck"))
    write(run_units(units), "drivers_rho.csv")
    # why it varies: cells, depth, weights, reliability
    rel_g = pd.read_csv(CM.CACHE / "atlas_h3_rel.csv")
    rel_b = pd.concat([split_half_direct(c) for c in RU.PRIMARY])
    rows = []
    for c in RU.PRIMARY:
        X, spc, lab, lib = T42_adata(c)
        L = np.asarray(X.sum(1)).ravel()
        nz = np.asarray((X > 0).sum(1)).ravel()
        for (s, t) in sorted(set(zip(spc, lab))):
            m = (spc == s) & (lab == t)
            rows.append(dict(cohort=c, species=s, type=t, n_cells=int(m.sum()), frac_species=float(m.sum() / (spc == s).sum()),
                             median_counts=float(np.median(L[m])), median_genes=float(np.median(nz[m]))))
        g = G[c]
        vs = g[[k for k in g.columns if k.startswith("C_") and not k.startswith("Csp_")]]
        for k in vs.columns:
            rows.append(dict(cohort=c, species="all", type=k[2:], var_share=float(vs[k].var() / vs.var().sum()),
                             corr_with_S=float(spearmanr(vs[k], g.S)[0])))
    write(pd.DataFrame(rows), "drivers_depth.csv")
    write(rel_b, "drivers_rel_direct.csv")
    write(rel_g, "drivers_rel_gold.csv")


# ---------------------------------------------------------------------------- item 9
def adds(G):
    units = []
    for c in RU.PRIMARY:
        g = G[c].copy()
        sd = pd.read_parquet(CM.CACHE / f"Sdir__{c}.parquet").set_index("var_name")
        g = g.join(sd[["Sdir"]], on="var_name")
        g["nonesm_S"] = CM.non_esm(g, "S", C)
        g["nonesm_D"] = CM.non_esm(g, "D_raw", C)
        if c.startswith("test"):
            dom = pd.read_parquet(CM.RES / f"T2_5/{c}__genes.parquet").set_index("var_name")["dominant_cell_type"]
            v = dom.reindex(g["var_name"])
            g["dom"] = (v == v.mode().iloc[0]).to_numpy().astype(float)
            dom_args = dict(extra_linear=["dom"])
        else:
            g["dom"] = g["hep_enrich"]
            dom_args = dict(extra_spline=["dom"])
        allb = RU.BASELINES
        ok = g[allb + ["D_raw", "S", "Sdir"]].notna().all(1).to_numpy()
        readers = ["D_raw", "S", "Sdir"] + (["D"] if c.startswith("test") else [])
        for gold, _, covs in golds(c):
            base = dict(cohort=c)
            # own and common universe, all readers with baselines, deltas to S
            units.append(dict(g=g, gold=gold, covs=covs, scores=readers + allb, ref="S", label=dict(base, analysis="common_universe")))
            for nm, panel in (("B1b", ["B1b_raw"]), ("B1b+B1c", ["B1b_raw", "B1c_raw"]),
                              ("B1b+B1c+B1z+B2b", allb), ("dNdS+LOEUF", ["dNdS", "LOEUF"]),
                              ("all_direct+dNdS+LOEUF", allb + ["dNdS", "LOEUF"])):
                units.append(dict(g=g, gold=gold, covs=covs, scores=readers, extra_spline=panel, rows=ok,
                                  label=dict(base, analysis="given_panel", panel=nm)))
            for x, given in (("B1b_raw", "S"), ("B1z_raw", "S"), ("B1c_raw", "S"), ("B2b", "S"), ("D_raw", "S"), ("S", "D_raw"),
                             ("S", "Sdir"), ("Sdir", "S"), ("B1b_raw", "D_raw")):
                units.append(dict(g=g, gold=gold, covs=covs, scores=[x], extra_spline=[given], rows=ok,
                                  label=dict(base, analysis="given_one", panel=given)))
            # margin-supported: baselines beside; D sign vs magnitude
            ms = g.margin_support.to_numpy()
            units.append(dict(g=g, gold=gold, covs=covs, scores=["D_raw", "S", "B1b_raw", "B1z_raw", "B1c_raw"], rows=ms & ok,
                              label=dict(base, analysis="margin_supported")))
            units.append(dict(g=g, gold=gold, covs=covs, scores=["D_raw"], extra_spline=["S"], rows=ms & ok,
                              label=dict(base, analysis="margin_supported_given_S")))
            un = ~g.masked.to_numpy()
            units.append(dict(g=g, gold=gold, covs=covs, scores=readers + allb, rows=un & ok,
                              label=dict(base, analysis="unmasked")))
            q = g.S.abs().where(un).rank(pct=True)
            for nm, lo, hi in (("|S| low third", 0, 1 / 3), ("|S| mid third", 1 / 3, 2 / 3), ("|S| top third", 2 / 3, 1.01)):
                units.append(dict(g=g, gold=gold, covs=covs, scores=readers + ["B1b_raw"], rows=((q > lo) & (q <= hi)).to_numpy(),
                                  label=dict(base, analysis="dose_S", panel=nm)))
            # coverage
            nob = g.B1b_raw.isna().to_numpy()
            units.append(dict(g=g, gold=gold, covs=covs, scores=readers, rows=nob, label=dict(base, analysis="no_B1b")))
            units.append(dict(g=g, gold=gold, covs=covs, scores=["D_raw", "S", "B1z_raw"], rows=nob, label=dict(base, analysis="no_B1b_with_B1z")))
            # dominance sensitivity, feature dependence
            units.append(dict(g=g, gold=gold, covs=covs, scores=readers + allb, label=dict(base, analysis="given_dominance"), **dom_args))
            units.append(dict(g=g, gold=gold, covs=covs, scores=["nonesm_S", "nonesm_D", "S", "D_raw"], rows=ok,
                              label=dict(base, analysis="features_removed")))
    # species-resolved margins: does a species' margin carry its own pair? (joint fits)
    g4 = G["liv_4sp"].copy()
    gm = G["liv_hmouse"].set_index("hsapiens_gene_id")[["B1b_raw"]].rename(columns={"B1b_raw": "B1b_pair"})
    g4 = g4.join(gm, on="hsapiens_gene_id")
    sp4 = [k for k in g4.columns if k.startswith("Csp_")]
    for gold, _, covs in golds("liv_4sp"):
        for rows_, nm in ((None, "all"), (~g4.masked.to_numpy(), "unmasked")):
            units.append(dict(g=g4, gold=gold, covs=covs, scores=sp4 + ["S", "B1b_pair"], rows=rows_, ref="Csp_Mouse",
                              label=dict(cohort="liv_4sp", analysis="species_margin", panel=nm)))
    g3 = G["test_3sp"].copy()
    gm = G["test_hmouse"].set_index("hsapiens_gene_id")[["B1b_raw"]].rename(columns={"B1b_raw": "B1b_pair"})
    g3 = g3.join(gm, on="hsapiens_gene_id")
    sp3 = [k for k in g3.columns if k.startswith("Csp_")]
    for gold, _, covs in golds("test_3sp"):
        for rows_, nm in ((None, "all"), (~g3.masked.to_numpy(), "unmasked")):
            units.append(dict(g=g3, gold=gold, covs=covs, scores=sp3 + ["S", "B1b_pair"], rows=rows_, ref="Csp_Mouse",
                              label=dict(cohort="test_3sp", analysis="species_margin", panel=nm)))
    write(run_units(units), "adds_rho.csv")
    # margin-supported: is D's magnitude informative beyond S's sign?
    rows = []
    for c in RU.PRIMARY:
        g = G[c]
        for gold, _, covs in golds(c):
            T = g[g.margin_support & g[[gold] + covs + ["D_raw", "S"]].notna().all(1)].copy()
            obs = rho_unit(T, gold, covs, ["D_raw"], draws=10)[0]["rho_p"]
            rng = np.random.default_rng(CM.useed("signonly", c, gold))
            sim = []
            for _ in range(500):
                T["x"] = np.sign(T.S.to_numpy()) * rng.permutation(T.D_raw.abs().to_numpy())
                sim.append(rho_unit(T, gold, covs, ["x"], draws=2)[0]["rho_p"])
            sgn = T.copy()
            sgn["x"] = np.sign(T.S)
            rows.append(dict(cohort=c, gold=gold, n=len(T), rho_D=obs, rho_sign_only_mean=float(np.mean(sim)),
                             rho_sign_only_lo=float(np.quantile(sim, .025)), rho_sign_only_hi=float(np.quantile(sim, .975)),
                             rho_sign_S=rho_unit(sgn, gold, covs, ["x"], draws=2)[0]["rho_p"]))
    write(pd.DataFrame(rows), "adds_sign_vs_magnitude.csv")
    # S and Sdir relation
    rows = []
    for c in RU.PRIMARY:
        g = G[c].join(pd.read_parquet(CM.CACHE / f"Sdir__{c}.parquet").set_index("var_name")[["Sdir"]], on="var_name")
        un = ~g.masked
        rows.append(dict(cohort=c, n_all=len(g), n_unmasked=int(un.sum()), spearman_all=spearmanr(g.S, g.Sdir)[0],
                         spearman_unmasked=spearmanr(g.S[un], g.Sdir[un])[0],
                         rho_p_given_C=rho_unit(g, "S", C, ["Sdir"], draws=2)[0]["rho_p"],
                         frac_masked=float(g.masked.mean())))
    write(pd.DataFrame(rows), "adds_S_vs_Sdir.csv")


def disease(G):
    """Hepatocyte MASH transfer, read-only from T7.3 (card item 2), EXPLORATORY and separate from T7.3's H-D1.

    Q = sign(l_h) m, shrunken logFC, genes detected in >= 5% of nuclei in both species, rho_p given C + splines
    of rank |l_h| and rank SE(m). Readouts are the 10-seed consensus here (H-D1 specifies 5 seeds)."""
    from analysis.common import gene_ids
    T = CM.RES / "T7_3/de"
    hu = pd.read_csv(T / "human_primary.tsv", sep="\t")
    mo = pd.read_csv(T / "mouse_primary.tsv", sep="\t")
    g = G["liv_hmouse"].copy()
    tab = gene_ids.load_table("liver")
    tab = tab[tab.hsapiens_gene_id.notna() & tab.mmusculus_gene_id.notna()]
    m2h = dict(zip(tab.mmusculus_gene_id, tab.hsapiens_gene_id))
    mo["hsapiens_gene_id"] = mo.gene_id.map(m2h)
    mo = mo.dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    hu["hsapiens_gene_id"] = hu.gene_id.where(hu.gene_id.notna(), hu.gene_symbol.map(dict(zip(g.symbol, g.hsapiens_gene_id))))
    hu = hu.dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    g["lh"] = g.hsapiens_gene_id.map(hu.log2fc_shrunk)
    g["padjh"] = g.hsapiens_gene_id.map(hu.padj)
    g["fh"] = g.hsapiens_gene_id.map(hu.det_frac)
    g["m"] = g.hsapiens_gene_id.map(mo.log2fc_shrunk)
    g["sem"] = g.hsapiens_gene_id.map(mo.lfcse_shrunk)
    g["fm"] = g.hsapiens_gene_id.map(mo.det_frac)
    g["Q"] = np.sign(g.lh) * g.m
    g["abslh"] = g.lh.abs()
    ok = (g.fh >= 0.05) & (g.fm >= 0.05)
    scores = ["D_raw", "S", "B1b_raw", "B1c_raw", "B1z_raw"]
    units = []
    con, rel, mid = constrained_masks(g, "dNdS")
    for nm, mask in (("all", ok), ("constrained", ok & con), ("middle", ok & mid), ("relaxed", ok & rel),
                     ("margin_supported", ok & g.margin_support)):
        units.append(dict(g=g, gold="Q", covs=C, scores=scores, extra_spline=["abslh", "sem"], rows=mask.to_numpy(),
                          label=dict(cohort="liv_hmouse", subset=nm, universe="common")))
        units.append(dict(g=g, gold="Q", covs=C, scores=["D_raw", "S"], extra_spline=["abslh", "sem"], rows=mask.to_numpy(),
                          label=dict(cohort="liv_hmouse", subset=nm, universe="own_D_S")))
    t = run_units(units)
    de = ok & (g.padjh < 0.1)
    t["n_human_DE_genes_in_universe"] = int(de.sum())
    write(t, "disease_hepatocyte.csv")
    rows = []
    for col in ("D_raw", "S", "B1b_raw"):
        tt = tertile(g[col].where(ok))
        for lab, m in (("all", ok), ("human_FDR_0.1", de)):
            agree = (np.sign(g.lh) == np.sign(g.m))[m]
            rows.append(dict(score=col, subset=lab, n=int(m.sum()), agree_top=float(agree[tt[m] == 3].mean()),
                             agree_bottom=float(agree[tt[m] == 1].mean()), n_top=int((tt[m] == 3).sum()), n_bottom=int((tt[m] == 1).sum())))
    write(pd.DataFrame(rows), "disease_hepatocyte_sign.csv")


def species_direct(G):
    """Species-resolved margin of the joint fits against the same sum without a model (added after adds)."""
    units = []
    for c in ("liv_4sp", "test_3sp"):
        g = G[c].join(pd.read_parquet(CM.CACHE / f"Sdir__{c}.parquet").set_index("var_name"), on="var_name")
        spp = [k[4:] for k in g.columns if k.startswith("Csp_")]
        cols = [f"Csp_{s}" for s in spp] + [f"Cspdir_{s}" for s in spp if f"Cspdir_{s}" in g]
        for gold, _, covs in golds(c):
            units.append(dict(g=g, gold=gold, covs=covs, scores=cols, label=dict(cohort=c), ref=f"Csp_Mouse"))
    write(run_units(units), "species_direct.csv")
    # is the zebrafish difference real? genes counted in every species, contrast of contrasts, pairwise baselines
    g = G["liv_4sp"].join(pd.read_parquet(CM.CACHE / "Sdir__liv_4sp.parquet").set_index("var_name"), on="var_name")
    for p, sp_ in (("liv_hmonkey", "Monkey"), ("liv_hmouse", "Mouse"), ("liv_hzeb", "Zebrafish")):
        b = pd.read_csv(CM.RES / f"T4_2/baselines__{p}.csv").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")["B1b_raw"]
        g[f"B1bpair_{sp_}"] = g.hsapiens_gene_id.map(b)
    rows = []
    cols = ["Csp_Mouse", "Csp_Zebrafish", "Cspdir_Mouse", "Cspdir_Zebrafish"]
    for gold, _, covs in golds("liv_4sp"):
        for nm, mask in (("all", np.ones(len(g), bool)), ("counts_in_all_species", g.nonzero_all_species.to_numpy())):
            m = mask & g[[gold] + covs + cols].notna().all(1).to_numpy()
            T = g.loc[m]
            R = np.column_stack([S.midrank(T[gold].to_numpy(float))] + [S.midrank(T[c].to_numpy(float)) for c in cols])
            Z = S.design(T, covs)
            est = S.rho_p_matrix(R, Z)[1:]
            B = S.bootstrap(R, Z, RU.DRAWS, np.random.default_rng(CM.useed("zeb", gold, nm)), T["block"].to_numpy())[:, 1:]
            cc = (B[:, 0] - B[:, 1]) - (B[:, 2] - B[:, 3])
            lo, hi = S.pct_ci(cc)
            rows.append(dict(gold=gold, subset=nm, n=int(m.sum()), Csp_Mouse=est[0], Csp_Zebrafish=est[1], Cspdir_Mouse=est[2],
                             Cspdir_Zebrafish=est[3], chimera_contrast=est[0] - est[1], direct_contrast=est[2] - est[3],
                             difference=(est[0] - est[1]) - (est[2] - est[3]), diff_lo=lo, diff_hi=hi))
    write(pd.DataFrame(rows), "species_zebrafish_check.csv")
    u = []
    for gold, _, covs in golds("liv_4sp"):
        u.append(dict(g=g, gold=gold, covs=covs, scores=["B1bpair_Monkey", "B1bpair_Mouse", "B1bpair_Zebrafish"], label=dict(analysis="pair_baselines_same_genes")))
    write(run_units(u), "species_pair_baselines.csv")


def mechanism(G):
    """Tests of ledgers/CHIMERA_MECHANISM.md, added on the coordinator's request after results were known."""
    lst = pd.read_csv(CM.RES / "X2/divergent_conserved.csv")
    prog = sorted(set(lst[lst.cohort == "liv_4sp"].symbol) - {"MT-CO1"})
    sets = {"prog63_noMT": prog, "core15": PROG_CORE, "fasting6": FASTING}
    units, rows, seeds_rows, masked_rows = [], [], [], []
    frames = {}
    for c in RU.COHORTS:
        g = G[c].join(pd.read_parquet(CM.CACHE / f"Sdir__{c}.parquet").set_index("var_name")[["Sdir", "Sdir_pattern", "Sdir_level"]],
                      on="var_name")
        g["level_cost"] = -g["Sdir_level"]
        frames[c] = g
        for gold, _, covs in golds(c):
            units.append(dict(g=g, gold=gold, covs=covs, scores=["S", "Sdir", "Sdir_pattern", "Sdir_level"], ref="Sdir_pattern",
                              label=dict(cohort=c, analysis="terms_vs_gold")))
            units.append(dict(g=g, gold=gold, covs=covs, scores=["Sdir_pattern"], extra_spline=["Sdir_level"], label=dict(cohort=c, analysis="pattern_given_level")))
            units.append(dict(g=g, gold=gold, covs=covs, scores=["Sdir_level"], extra_spline=["Sdir_pattern"], label=dict(cohort=c, analysis="level_given_pattern")))
        o = dict(cohort=c, S_vs_pattern=spearmanr(g.S, g.Sdir_pattern)[0], S_vs_level=spearmanr(g.S, g.Sdir_level)[0],
                 var_share_level_in_Sdir=float(np.cov(g.Sdir_level, g.Sdir)[0, 1] / g.Sdir.var()),
                 frac_S_negative_unmasked=float((g.S[~g.masked] < 0).mean()), frac_pattern_positive=float((g.Sdir_pattern > 0).mean()),
                 frac_level_negative=float((g.Sdir_level < 0).mean()), expr_vs_absS=spearmanr(g.log_cp10k_spmean, g.S.abs())[0])
        if "abs_shift" in g:
            o.update(level_term_vs_abs_shift=spearmanr(g.Sdir_level, g.abs_shift)[0], pattern_term_vs_B1b=g[["Sdir_pattern", "B1b_raw"]].dropna().corr("spearman").iat[0, 1],
                     level_term_vs_B1b=g[["Sdir_level", "B1b_raw"]].dropna().corr("spearman").iat[0, 1])
        rows.append(o)
        # masked genes: delta-alpha with no count target
        F = [f"esm{i}" for i in range(64)]
        for nm, mask in (("masked", g.masked.to_numpy()), ("unmasked", ~g.masked.to_numpy()),
                         ("unmasked_margin_supported", (~g.masked & g.margin_support).to_numpy())):
            for gold, _, covs in golds(c):
                units.append(dict(g=g, gold=gold, covs=covs, scores=["D_raw"] + (["D"] if c.startswith("test") else []), rows=mask,
                                  label=dict(cohort=c, analysis="masked_split", panel=nm)))
            T = g.loc[mask]
            if len(T) > 300:
                masked_rows.append(dict(cohort=c, subset=nm, n=len(T), frac_all=len(T) / len(g), esm_cv_r2=_cv_r2(T, F, "D_raw"),
                                        median_expr=T.log_cp10k_spmean.median()))
    write(pd.DataFrame(rows), "mechanism_terms.csv")
    write(pd.DataFrame(masked_rows), "mechanism_masked_r2.csv")
    # integrated and poorly mixed seeds of liver H+Mouse
    from analysis.tasks.T4_3_scoreboard import inputs as I
    g = frames["liv_hmouse"].copy()
    runs = {int(k): v for k, v in I.seed_runs_used("liv_hmouse").items()}
    good = [42, 43, 44, 47, 48, 51]
    poor = [s for s in runs if s not in good]
    for nm, ss in (("integrated", good), ("poorly_mixed", poor)):
        A = np.column_stack([K.read_run(runs[s])["alpha"] for s in ss]).mean(1)
        B = np.column_stack([K.read_run(runs[s])["S_g_ema"] for s in ss]).mean(1)
        g[f"S_{nm}"] = B
        g[f"D_{nm}"] = A - g.prior_raw.to_numpy()
    for gold, _, covs in golds("liv_hmouse"):
        units.append(dict(g=g, gold=gold, covs=covs, scores=["S_integrated", "S_poorly_mixed", "D_integrated", "D_poorly_mixed", "Sdir"],
                          ref="S_integrated", label=dict(cohort="liv_hmouse", analysis="seed_state")))
    write(run_units(units), "mechanism_rho.csv")
    # X2 program: predicted signature is pattern gain kept, level cost large
    sig = []
    for c in ("liv_4sp", "liv_hmouse"):
        g = frames[c].copy()
        d = pd.read_parquet(CM.CACHE / f"Sdir__{c}.parquet").set_index("var_name")
        g["Cpat_hep"] = g.var_name.map(d["Cpat_Hepatocytes"])
        g["Clev_hep"] = g.var_name.map(d["Clev_Hepatocytes"])
        L = t25_long(c, "S_gsk")
        L = L[(L.group_key == "cell_type") & (L.cell_type == "Hepatocytes")]
        u = L.pivot_table(index="var_name", values=["S_gsk_meanz", "S_gsk_within"], aggfunc="mean")
        g["hep_unweighted"] = g.var_name.map(u["S_gsk_meanz"])
        g["hep_within"] = g.var_name.map(u["S_gsk_within"])
        vals = ["S", "Sdir", "Sdir_pattern", "Sdir_level", "Cpat_hep", "Clev_hep", "hep_unweighted", "hep_within", "B1b_raw"]
        if "abs_shift" in g:
            vals += ["abs_shift", "shape", "frac_level", "spread"]
        chep = CP + ["hep_enrich"]
        for k, names in sets.items():
            f = g.symbol.isin(names).to_numpy()
            pool = ~g.symbol.isin(prog).to_numpy()
            for v in vals:
                sig.append(dict(cohort=c, set=k, value=v, matching="C_hep", **matched(g, f, pool, v, chep, key=f"sig{c}{k}{v}")))
    write(pd.DataFrame(sig), "mechanism_x2_signature.csv")


def mechanism_adds(G):
    """Do the pattern and level terms beat direct baselines? Same universes as adds; head to head by delta."""
    units = []
    for c in RU.PRIMARY:
        g = G[c].join(pd.read_parquet(CM.CACHE / f"Sdir__{c}.parquet").set_index("var_name")[["Sdir_pattern", "Sdir_level"]], on="var_name")
        ok = g[RU.BASELINES + ["S", "Sdir_pattern"]].notna().all(1).to_numpy()
        for gold, _, covs in golds(c):
            units.append(dict(g=g, gold=gold, covs=covs, scores=["Sdir_pattern", "Sdir_level", "S"] + RU.BASELINES, ref="Sdir_pattern", rows=ok,
                              label=dict(cohort=c, analysis="common_universe")))
            units.append(dict(g=g, gold=gold, covs=covs, scores=["Sdir_pattern", "Sdir_level"], extra_spline=RU.BASELINES, rows=ok,
                              label=dict(cohort=c, analysis="given_all_direct")))
    write(run_units(units), "mechanism_adds.csv")


def _cv_r2(T, F, col):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = T[F].to_numpy(float), T[col].rank().to_numpy(float)
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        pred[te] = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))).fit(X[tr], y[tr]).predict(X[te])
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


SECTIONS = dict(mechanism=mechanism, mech_adds=mechanism_adds, species=species_direct, disease=disease, repro=repro, map=conservation_map, right=who_is_right, disagree=disagreement, h2h=head_to_head, explain=explain,
                x3=re_x3, x1=re_x1, x2=re_x2, drivers=drivers, adds=adds)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", nargs="+", default=list(SECTIONS))
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    G = {c: prepare(c) for c in RU.COHORTS}
    for s in a.sections:
        print(f"== {s}", flush=True)
        SECTIONS[s](G)
    return 0


if __name__ == "__main__":
    sys.exit(main())
