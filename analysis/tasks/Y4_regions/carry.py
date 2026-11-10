"""EXPLORATORY: apply the carry rule of rules.py to discovery_units.csv. Writes carried.json and the eligible table."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.tasks.Y4_regions import rules as RU
from analysis.tasks.Y4_regions import screen as SC

OUT = SC.OUT


def tiers(df: pd.DataFrame, global_T4: dict) -> pd.DataFrame:
    df = df.copy()
    g4 = [global_T4.get((c, g, x), np.nan) for c, g, x in zip(df.cohort, df.gold, df.readout)]
    df["T4_global"] = g4
    base = (df.rho > 0) & (df.q_T1 < RU.Q) & (df.lo > 0)
    has = df.J.fillna("").apply(lambda x: {"B1b_raw", "B1c_raw"} <= set(x.split(",")))
    a = (df.q_T3 < RU.Q) & (df.T3 >= RU.T3_MIN) & has
    b = (df.q_T4 < RU.Q) & (df.T4 >= RU.T4_MIN) & (df.T4 >= df.T4_global + RU.T4_OVER_GLOBAL) & has
    c = (df.q_T2 < RU.Q) & (df.T2 >= RU.T2_MIN)
    df["tier"] = np.select([base & a, base & b, base & c], ["A", "B", "C"], default="")
    zs = {"A": df.T3 / (df.T3_hi - df.T3_lo) * 3.92, "B": df.T4 / (df.T4_hi - df.T4_lo) * 3.92,
          "C": df.T2 / (df.T2_hi - df.T2_lo) * 3.92}
    df["tier_z"] = np.select([df.tier == k for k in "ABC"], [zs[k] for k in "ABC"], default=np.nan)
    return df


def masks() -> dict:
    return {c: {r["name"]: r["mask"] for r in SC.regs(c)} for c in RU.COHORTS}


def pick(e: pd.DataFrame) -> list[int]:
    M = masks()
    ids = {c: SC.table(c)["hsapiens_gene_id"].to_numpy() for c in RU.COHORTS}
    sets = {}

    def gs(i):
        if i not in sets:
            r = e.loc[i]
            sets[i] = set(ids[r.cohort][M[r.cohort][r.region] & SC.table(r.cohort)["split"].eq("odd").to_numpy()])
        return sets[i]

    def tissue(i):
        return SC.tissue(e.loc[i, "cohort"])

    order = e.sort_values(["tier", "mech", "tier_z"], ascending=[True, False, False]).index.tolist()
    chosen: list[int] = []
    axes: dict = {}

    def fits(i):
        r = e.loc[i]
        if axes.get(r.axis, 0) >= RU.MAX_PER_AXIS:
            return False
        for j in chosen:
            if tissue(i) == tissue(j):
                u = gs(i) | gs(j)
                if len(gs(i) & gs(j)) / max(len(u), 1) > RU.JACCARD:
                    return False
        return True

    def take(i):
        chosen.append(i)
        axes[e.loc[i, "axis"]] = axes.get(e.loc[i, "axis"], 0) + 1

    # guarantee the tissue minimum first, by the same order, then fill
    for tis in ("liver", "testis"):
        for i in order:
            if sum(tissue(j) == tis for j in chosen) >= RU.MIN_PER_TISSUE:
                break
            if tissue(i) == tis and i not in chosen and fits(i):
                take(i)
    for i in order:
        if len(chosen) >= RU.N_CARRY:
            break
        if i not in chosen and fits(i):
            take(i)
    return chosen


def main():
    df = pd.read_csv(OUT / "discovery_units.csv")
    assert set(df.split) == {"odd"}
    al = df[df.region == "all"]
    g4 = {(r.cohort, r.gold, r.readout): r.T4 for r in al.itertuples()}
    df = tiers(df, g4)
    df.to_csv(OUT / "discovery_units.csv", index=False)
    real = df[df.region != "all"]
    e = real[real.tier != ""]
    e.to_csv(OUT / "discovery_eligible.csv", index=False)
    print("units", len(df), "regions", df.region.nunique(), "eligible", len(e), e.tier.value_counts().to_dict())
    dec = pd.read_csv(OUT / "discovery_decoys.csv")
    dec = dec.assign(T1_nom=(dec.p_T1 < 0.05) & (dec.rho > 0), T2_nom=(dec.p_T2 < 0.05) & (dec.T2 > 0),
                     T3_nom=(dec.p_T3 < 0.05) & (dec.T3 > 0), T4_nom=(dec.p_T4 < 0.05) & (dec.T4 > 0))
    t = tiers(dec.assign(q_T1=dec.p_T1 * 0 + dec.p_T1, q_T2=dec.p_T2, q_T3=dec.p_T3, q_T4=dec.p_T4), g4)
    s = dec.assign(size=dec.region.str.extract(r"decoy(\d+)_")[0].astype(int), tier=t.tier).groupby("size").agg(
        n=("rho", "size"), T2_nom=("T2_nom", "mean"), T3_nom=("T3_nom", "mean"), T4_nom=("T4_nom", "mean"),
        any_tier_unadjusted=("tier", lambda x: (x != "").mean()))
    s.to_csv(OUT / "discovery_decoy_rates.csv")
    print(s)
    idx = pick(e)
    cols = ["cohort", "gold", "readout", "region", "axis", "mech", "seen", "tier", "n", "rho", "lo", "hi", "q_T1", "rho_rest", "T2",
            "T2_lo", "T2_hi", "q_T2", "n_common", "rho_common", "T3", "T3_lo", "T3_hi", "q_T3", "T3_best", "T4", "T4_lo", "T4_hi",
            "q_T4", "J", "J_dropped"]
    car = e.loc[idx, cols]
    car.to_csv(OUT / "carried.csv", index=False)
    (OUT / "carried.json").write_text(json.dumps(car.to_dict("records"), indent=1, default=str))
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 100)
    print(car[["cohort", "gold", "readout", "region", "axis", "tier", "n", "rho", "lo", "hi", "rho_rest", "T3", "T3_best", "T4"]].round(3))


if __name__ == "__main__" and __import__("sys").argv[1:] != ["amended"]:
    main()


# Added after the odd table and the decoy calibration were read and before any even gene was opened.
# The mechanical list is kept as carried_mechanical.csv. Each unit below is eligible under the committed rule and beats every one
# of the 30 same-size decoys of its cohort, gold and readout on its tier's statistic.
CHOSEN = [
    ("liv_hmouse", "G_H3", "S", "expr_T2__spread_T1"),
    ("liv_hmouse", "G_H3", "S", "spread_T1"),
    ("liv_hmouse", "G_H1", "S", "expr_T2__tau_T3"),
    ("test_hmouse", "G_H2", "S", "tau_Q4"),
    ("test_hmouse", "G_H4", "S", "absshift_T3__spread_T3"),
    ("liv_hmouse", "G_H3", "D", "absshift_T3"),
    ("liv_hmouse", "G_H3", "S", "pihi__tau_T2"),
    ("liv_4sp", "G_H3", "D", "pihi__expr_T2"),
    ("liv_hmouse", "G_H3", "D", "pilo__spread_T3"),
    ("test_hmouse", "G_H4", "D", "peakmargin_T3"),
]
TIER_STAT = {"A": "T3", "B": "T4", "C": "T2"}


def amended():
    import shutil

    if not (OUT / "carried_mechanical.csv").exists():
        shutil.copy(OUT / "carried.csv", OUT / "carried_mechanical.csv")
    df = pd.read_csv(OUT / "discovery_units.csv")
    dec = pd.read_csv(OUT / "discovery_decoys.csv")
    dec["size"] = dec.region.str.extract(r"decoy(\d+)")[0].astype(int)
    sizes = np.array(RU.DECOY_SIZES)
    ref = dec.groupby(["cohort", "gold", "readout", "size"])[["T2", "T3", "T4"]].max()
    rows = []
    for c, g, x, reg in CHOSEN:
        r = df[(df.cohort == c) & (df.gold == g) & (df.readout == x) & (df.region == reg)].iloc[0]
        assert r.tier in ("A", "B", "C"), (c, g, x, reg)
        ns = int(sizes[np.argmin(np.abs(np.log(sizes) - np.log(r.n)))])
        st = TIER_STAT[r.tier]
        assert r[st] > ref.loc[(c, g, x, ns), st], (c, g, x, reg)
        rows.append(r)
    car = pd.DataFrame(rows)
    # the committed overlap rule still holds
    masks_ = masks()
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if SC.tissue(a.cohort) == SC.tissue(b.cohort):
                ta, tb = SC.table(a.cohort), SC.table(b.cohort)
                ga = set(ta["hsapiens_gene_id"].to_numpy()[masks_[a.cohort][a.region] & ta["split"].eq("odd").to_numpy()])
                gb = set(tb["hsapiens_gene_id"].to_numpy()[masks_[b.cohort][b.region] & tb["split"].eq("odd").to_numpy()])
                j = len(ga & gb) / max(len(ga | gb), 1)
                print(a.region, b.region, round(j, 3))
                assert j <= RU.JACCARD, (a.region, b.region, j)
    cols = ["cohort", "gold", "readout", "region", "axis", "mech", "seen", "tier", "n", "rho", "lo", "hi", "q_T1", "rho_rest", "T2", "T2_lo", "T2_hi", "q_T2",
            "n_common", "rho_common", "T3", "T3_lo", "T3_hi", "q_T3", "T3_best", "T4", "T4_lo", "T4_hi", "q_T4", "T4_global", "J", "J_dropped",
            "rho_B1b_raw", "rho_B1c_raw", "rho_B1z_raw", "rho_B2b", "rho_B3a", "rho_Sdir"]
    car = car[cols]
    car.to_csv(OUT / "carried.csv", index=False)
    (OUT / "carried.json").write_text(json.dumps(car.to_dict("records"), indent=1, default=str))
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(car[["cohort", "gold", "readout", "region", "tier", "seen", "n", "rho", "lo", "hi", "rho_rest", "q_T2", "n_common", "rho_common", "T3", "T3_best", "T4", "rho_B1b_raw", "rho_B1c_raw", "rho_B1z_raw", "rho_B2b", "rho_B3a", "rho_Sdir"]].round(3).to_string())


if __name__ == "__main__" and __import__("sys").argv[1:] == ["amended"]:
    amended()
