"""Z4 summary: one row per cohort (health, count dependence, agreement of the two gates, masked genes, separate axis, expression) and one
row per independent-reference unit with the comparison rule of the ledger (Z3's): free better if delta >= 0.03 and CI lower > 0, worse if
delta <= -0.03 and CI upper < 0, otherwise no difference shown; cohort verdicts on count-supported genes and on all genes.

    PYTHONPATH=$PWD python -m analysis.tasks.Z4_free_gate_all.summary
Output: results/analysis/Z4/summary_cohorts.csv, summary_references.csv, verdicts.csv, summary.md
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.tasks.H1_harmonise import rules as H1R
from analysis.tasks.Z4_free_gate_all import rules as RU

REF = RU.OUT / "references"
DELTA = 0.03
FLAT_SD = 0.015  # amendment 1


def call(d, lo, hi) -> str:
    if not np.isfinite(d):
        return "na"
    if d >= DELTA and lo > 0:
        return "free better"
    if d <= -DELTA and hi < 0:
        return "free worse"
    return "no difference shown"


def references() -> pd.DataFrame:
    rows = []
    f = REF / "trackc_rho.csv"
    if f.exists():
        R = pd.read_csv(f)
        H = pd.read_csv(REF / "trackc_h2h.csv")
        for (c, kind, gold, stn), g in R.groupby(["cohort", "kind", "gold", "stratum"], sort=False):
            v = g.set_index("method")
            if "D_free" not in v.index:
                continue
            h = H[(H.cohort == c) & (H.kind == kind) & (H.stratum == stn) & (H.focal == "D_free")].set_index("comparator")
            hs = H[(H.cohort == c) & (H.kind == kind) & (H.stratum == stn) & (H.focal == "S_free") & (H.comparator == "S_g")]
            r = dict(cohort=c, family="track C", reference=f"{kind} ({gold})", universe={"all": "all", "count_supported": "count_supported_can", "support_both": "count_supported"}[stn],
                     n=int(v.loc["D", "n"]), D_can=v.loc["D", "value"], D_can_lo=v.loc["D", "ci_lo"], D_can_hi=v.loc["D", "ci_hi"],
                     D_free=v.loc["D_free", "value"], D_free_lo=v.loc["D_free", "ci_lo"], D_free_hi=v.loc["D_free", "ci_hi"],
                     S_can=v.loc["S_g", "value"], S_free=v.loc["S_free", "value"], B1=v["value"].get("B1", np.nan),
                     delta=h.loc["D", "delta"], delta_lo=h.loc["D", "ci_lo"], delta_hi=h.loc["D", "ci_hi"],
                     delta_free_vs_Sfree=h.loc["S_free", "delta"] if "S_free" in h.index else np.nan,
                     delta_Sfree_vs_Scan=float(hs.delta.iloc[0]) if len(hs) else np.nan)
            for comp in ("B1", "B1b", "B1c", "B2b", "B3a", "B1z"):
                if comp in h.index:
                    r[f"vs_{comp}"] = call(h.loc[comp, "delta"], h.loc[comp, "ci_lo"], h.loc[comp, "ci_hi"])
            rows.append(r)
    for fname, fam, keycols in (("cortex_e_b1.tsv", "C1 E-B1 held-out regions", ["cohort", "gold", "universe"]), ("cases_h_d1.tsv", "C1 H-D1 disease transfer", ["case", "cell_type"])):
        f = REF / fname
        if not f.exists():
            continue
        E = pd.read_csv(f, sep="\t")
        for k, g in E.groupby(keycols, sort=False):
            v = g.set_index("score")
            k = k if isinstance(k, tuple) else (k,)
            if fam.startswith("C1 E-B1"):
                c, ref, uni = k[0], k[1], {"all": "all", "support_both": "count_supported"}[k[2]]
            else:
                c, ref, uni = {"ad": "brain_m1_hmouse", "muscle": "muscle_ageing", "lung": "lung_healthy"}[k[0]], f"{k[0]} {k[1]}", "all"
            rows.append(dict(cohort=c, family=fam, reference=ref, universe=uni, n=int(v.loc["D", "n"]), D_can=v.loc["D", "rho_p"], D_can_lo=v.loc["D", "ci_lo"], D_can_hi=v.loc["D", "ci_hi"],
                             D_free=v.loc["D_free", "rho_p"], D_free_lo=v.loc["D_free", "ci_lo"], D_free_hi=v.loc["D_free", "ci_hi"], S_can=v.loc["S_ema", "rho_p"], S_free=v.loc["S_free", "rho_p"],
                             B1=v.loc["B1", "rho_p"], delta=v.loc["D_free", "delta_to_first"], delta_lo=v.loc["D_free", "delta_lo"], delta_hi=v.loc["D_free", "delta_hi"],
                             delta_free_vs_Sfree=v.loc["D_free", "rho_p"] - v.loc["S_free", "rho_p"], delta_Sfree_vs_Scan=v.loc["S_free", "rho_p"] - v.loc["S_ema", "rho_p"]))
    f = REF / "lists.csv"
    if f.exists():
        L = pd.read_csv(f)
        for (c, nm), g in L.groupby(["cohort", "list"]):
            v = g.set_index("method")
            rows.append(dict(cohort=c, family="Shay list (stratified AUROC, oriented)", reference=nm, universe="all", n=int(v.loc["D", "n"]), D_can=v.loc["D", "value"], D_can_lo=v.loc["D", "ci_lo"],
                             D_can_hi=v.loc["D", "ci_hi"], D_free=v.loc["D_free", "value"], D_free_lo=v.loc["D_free", "ci_lo"], D_free_hi=v.loc["D_free", "ci_hi"], S_can=v.loc["S_g", "value"],
                             S_free=v.loc["S_free", "value"], B1=v["value"].get("B1", np.nan), delta=v.loc["D_free", "value"] - v.loc["D", "value"], delta_lo=np.nan, delta_hi=np.nan,
                             note="AUROC CIs unpaired; no paired call"))
    T = pd.DataFrame(rows)
    T["call"] = [call(d, lo, hi) if np.isfinite(lo) else "unpaired" for d, lo, hi in zip(T.delta, T.delta_lo.fillna(np.nan), T.delta_hi.fillna(np.nan))]
    return T


def verdicts(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in T[T.call != "unpaired"].groupby("cohort", sort=False):
        for uni in ("count_supported", "all"):
            x = g[g.universe == uni]
            if not len(x):
                continue
            b, w = int((x.call == "free better").sum()), int((x.call == "free worse").sum())
            v = "mixed" if b and w else "free better" if b else "free worse" if w else "no difference shown"
            rows.append(dict(cohort=c, universe=uni, n_units=len(x), n_better=b, n_worse=w, verdict=v,
                             better=", ".join(x[x.call == "free better"].reference), worse=", ".join(x[x.call == "free worse"].reference)))
    return pd.DataFrame(rows)


def main():
    kept = json.loads((RU.OUT / "kept.json").read_text())
    H = pd.read_csv(RU.OUT / "health.csv")
    C = pd.read_csv(RU.OUT / "cohort_compare.csv")
    A = pd.read_csv(RU.OUT / "axis_compare.csv")
    reg = H1R.cohorts()
    rows = []
    for c in reg:
        h = H[H.cohort == c]
        ks = kept.get(c, {})
        hk = h[h.run.isin(ks.values())]
        r = dict(cohort=c, label=reg[c]["label"], tissue=reg[c]["tissue"], source="Z3" if c in RU.Z3_FREE else "Z4", seeds_tried=len(h), seeds_kept=len(ks),
                 kept=",".join(map(str, ks)), failed="; ".join(f"s{x.seed}: " + ("not finished" if not x.finished else "non-finite" if not x.no_nan else "gate rule" if not x.healthy_gate else "poorly mixed")
                                                        for x in h.itertuples() if x.run not in ks.values() and not (c in RU.Z3_FREE and x.healthy_gate)),
                 entropy_free=hk.int_knn_mixing_entropy.mean(), entropy_can=h.can_entropy_median.iloc[0] if len(h) else np.nan,
                 adv_free=hk.adversary.mean(), label_transfer_free=hk.int_label_transfer_acc.mean(), label_transfer_can=h.can_label_transfer_median.iloc[0] if len(h) else np.nan,
                 sat_max_free=hk[["sat_high", "sat_low"]].max().max() if len(hk) else np.nan, recipe_diff="; ".join(sorted(set(x for x in hk.recipe_diff.fillna("") if x))))
        x = C[C.cohort == c]
        if len(x):
            r.update(x.iloc[0].drop(["cohort", "tissue"]).to_dict())
        for uni in ("all", "count_support"):
            a = A[(A.cohort == c) & (A.universe == uni)]
            if len(a):
                for k in a.columns:
                    if k.startswith("rho") and k != "rho_pi_dNdS_free":
                        r[f"{k}__{uni}"] = float(a[k].iloc[0])
        rows.append(r)
    S = pd.DataFrame(rows)
    S["expr_gap_free"] = (S.get("rho_D_expr_free") - S.get("rho_S_expr_free")).abs()
    S["expr_gap_can"] = (S.get("rho_D_expr_can") - S.get("rho_S_expr_can")).abs()
    # amendment 1: flat free gate (SD of D on count-supported genes < FLAT_SD): ranks are read, amplitude readouts reported only
    S["flat_free"] = S.get("sd_D_support_free") < FLAT_SD
    S.to_csv(RU.OUT / "summary_cohorts.csv", index=False)
    T = references()
    T.to_csv(RU.OUT / "summary_references.csv", index=False)
    V = verdicts(T)
    V.to_csv(RU.OUT / "verdicts.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 400)
    cols = ["cohort", "seeds_kept", "kept", "failed", "share_support_both", "rho_DS_free", "rho_DS_can", "rho_DS_partial_free", "rho_Dfree_Dcan_support", "rho_Dfree_Dcan_all",
            "rho_Sfree_Scan_support", "masked_at_prior_free", "rho_D_pi_free__all", "rho_D_pi_free__count_support", "rho_D_dNdS_free__all", "rho_D_dNdS_free__count_support",
            "rhop_D_dNdS_free__count_support", "rhop_D_pi_free__count_support", "sd_D_free", "sd_D_support_free", "flat_free", "expr_gap_free", "expr_gap_can", "entropy_free", "entropy_can"]
    print(S[[k for k in cols if k in S]].round(3).to_string())
    print(T[["cohort", "family", "reference", "universe", "n", "D_can", "D_free", "S_can", "S_free", "B1", "delta", "delta_lo", "delta_hi", "call"]].round(3).to_string())
    print(V.to_string())


if __name__ == "__main__":
    main()
