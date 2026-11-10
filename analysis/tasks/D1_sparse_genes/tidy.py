"""Tidy rows in the benchmark schema and the mechanical decision table of rules.py.
    python -m analysis.tasks.D1_sparse_genes.tidy
Writes results/analysis/D1/tidy.csv, signal_cells.csv, target_test.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.tasks.D1_sparse_genes import rules as RU

FEATURE = ["R_S_p", "R_Sdir_p", "R_B1_p", "B3a", "kNN_S", "kNN_Sdir", "Block_S"]
DIRECT = ["Sdir", "Lfc", "B1", "B1c", "B1z", "S_fwd"]


def tidy_a(a: pd.DataFrame) -> pd.DataFrame:
    d = a[a.rho_p.notna()].copy()
    return pd.DataFrame(dict(track="D1a", task="a|" + d.kind + "|" + d.cohort + "|" + d.stratum + "|" + d.universe + "|" + d.split, cohort=d.cohort, gold=d.gold,
                             setting=d.group, method=d.score, metric="rho_p", value=d.rho_p, ci_lo=d.ci_lo, ci_hi=d.ci_hi, n=d.n, rank_direction="higher"))


def tidy_b(b: pd.DataFrame) -> pd.DataFrame:
    d = b[b.rho_p.notna()].copy()
    d["split"] = d.get("split", "all")
    return pd.DataFrame(dict(track="D1b", task="b|" + d.cohort + "|" + d.set + "|" + d.target.astype(str) + "|" + d.split.fillna("all"), cohort=d.cohort, gold=d.target, setting=d.setting,
                             method=d.method, metric=d.metric, value=d.rho_p, ci_lo=d.ci_lo, ci_hi=d.ci_hi, n=d.n, rank_direction="higher"))


def tidy_c(c: pd.DataFrame) -> pd.DataFrame:
    d = c[c.rho_p.notna()].copy()
    return pd.DataFrame(dict(track="D1c", task="c|" + d.cohort + "|" + d.set + "|" + d.target.astype(str), cohort=d.cohort, gold=d.target, setting="fold" + d.fold, method=d.score,
                             metric="rho_p", value=d.rho_p, ci_lo=d.ci_lo, ci_hi=d.ci_hi, n=d.n, rank_direction="higher"))


def tidy_l(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d.rho_p.notna()].copy()
    return pd.DataFrame(dict(track="D1L", task="L|" + d.kind + "|" + d.cohort + "|" + d.set + "|" + d.split, cohort=d.cohort, gold=d.gold, setting=d.set,
                             method=d.score, metric="rho_p", value=d.rho_p, ci_lo=d.ci_lo, ci_hi=d.ci_hi, n=d.n, rank_direction="higher"))


def tidy_completion(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in d.itertuples():
        for m in ("tried", "finished", "finished_gate_tail"):
            rows.append(dict(track="D1b", task=f"completion|{r.cohort}|{r.setting}", cohort=r.cohort, gold="", setting=r.setting, method=r.method, metric=m,
                             value=getattr(r, m), ci_lo=float("nan"), ci_hi=float("nan"), n=r.tried, rank_direction="higher"))
    return pd.DataFrame(rows)


def signal(a: pd.DataFrame) -> pd.DataFrame:
    m = a[(a.stratum == "masked") & (a.universe == "clean") & (a.group == "always")]
    out = []
    for (c, kind, gold), g in m.groupby(["cohort", "kind", "gold"]):
        o = g[(g.split == "odd") & (g.score == "D")]
        e = g[(g.split == "even") & (g.score == "D")]
        al = g[(g.split == "all") & (g.score == "D")]
        if o.empty or e.empty:
            out.append(dict(cohort=c, kind=kind, gold=gold, signal=False, note="too few genes"))
            continue
        o, e = o.iloc[0], e.iloc[0]
        sgn = np.sign(o.rho_p)
        sig = bool(abs(o.rho_p) >= RU.SIG and sgn * o.ci_lo > 0 and sgn * o.ci_hi > 0 and np.sign(e.rho_p) == sgn and e.ci_lo * e.ci_hi > 0)
        pos = sig and sgn > 0
        r = dict(cohort=c, kind=kind, gold=gold, n=int(al.n.iloc[0]) if len(al) else 0, rho_all=al.rho_p.iloc[0] if len(al) else np.nan, rho_odd=o.rho_p, lo_odd=o.ci_lo, hi_odd=o.ci_hi, rho_even=e.rho_p,
                 lo_even=e.ci_lo, hi_even=e.ci_hi, signal=sig, signal_positive=pos)
        matched = []
        beaten_all = True
        for s in FEATURE + ["Sdir", "Lfc", "S_fwd"]:
            do = g[(g.split == "odd") & (g.score == s)]
            de = g[(g.split == "even") & (g.score == s)]
            if do.empty or de.empty:
                continue
            # delta in rows is score - D; D beats s if -delta >= 0.03 and upper of (s-D) < 0
            dd = [(-x.delta.iloc[0], -x.delta_hi.iloc[0]) for x in (do, de)]
            ok = all(v >= RU.SIG and lo > 0 for v, lo in dd)
            r[f"D_minus_{s}_odd"] = dd[0][0]
            r[f"D_minus_{s}_even"] = dd[1][0]
            if not ok:
                beaten_all = False
                if s in FEATURE:
                    matched.append(s)
        r["feature_matched"] = len(matched) > 0
        r["matched_by"] = ",".join(matched)
        r["D_beats_all"] = beaten_all
        out.append(r)
    return pd.DataFrame(out)


def target(a: pd.DataFrame) -> pd.DataFrame:
    m = a[(a.stratum == "masked") & (a.universe == "clean") & (a.group == "target")]
    out = []
    for (c, kind, gold), g in m.groupby(["cohort", "kind", "gold"]):
        r = dict(cohort=c, kind=kind, gold=gold)
        for spl in ("all", "odd", "even"):
            x = g[g.split == spl].set_index("score")
            if "R_S_p" in x.index:
                r[f"R_S_p_{spl}"] = x.rho_p["R_S_p"]
            for s in ("R_Sdir_p", "R_B1_p"):
                if s in x.index:
                    r[f"{s}_{spl}"] = x.rho_p[s]
                    r[f"S_minus_{s}_{spl}"] = -x.delta[s]
                    r[f"S_minus_{s}_lo_{spl}"] = -x.delta_hi[s]
        chim = all(r.get(f"S_minus_{s}_{spl}", -1) >= RU.SIG and r.get(f"S_minus_{s}_lo_{spl}", -1) > 0 for s in ("R_Sdir_p", "R_B1_p") for spl in ("odd", "even"))
        r["target_is_chimera"] = chim
        out.append(r)
    return pd.DataFrame(out)


def main():
    a = pd.read_csv(RU.RES / "a_rho.csv")
    parts = [tidy_a(a)]
    for f, fn in (("b_rho.csv", tidy_b), ("c_rho.csv", tidy_c), ("L_rho.csv", tidy_l), ("completion.csv", tidy_completion)):
        p = RU.RES / f
        if p.exists() and p.stat().st_size > 5:
            parts.append(fn(pd.read_csv(p, low_memory=False)))
    pd.concat(parts, ignore_index=True).to_csv(RU.RES / "tidy.csv", index=False)
    s = signal(a)
    s.to_csv(RU.RES / "signal_cells.csv", index=False)
    t = target(a)
    t.to_csv(RU.RES / "target_test.csv", index=False)
    print(s[["cohort", "kind", "gold", "n", "rho_all", "rho_odd", "lo_odd", "rho_even", "lo_even", "signal", "feature_matched", "matched_by", "D_beats_all"]].round(3).to_string())
    print(t[["cohort", "kind", "gold", "target_is_chimera"]].to_string())


if __name__ == "__main__":
    main()
