"""EXPLORATORY: the verdict rule of rules.py applied mechanically to the re-read tables.

CI degenerate (ESM blocks too few: width < 0.02) is reported as not estimable and counts as null.
Writes results/analysis/X4/verdicts.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.tasks.X4_where_differs import common as CM

OUT = CM.OUT


def state(r, d):
    if r is None or pd.isna(r.excess_d):
        return "na"
    if (r.ci_hi - r.ci_lo) < 0.02:
        return "not_estimable"
    if d * r.ci_lo > 0 or d * r.ci_hi > 0:
        lo, hi = sorted((d * r.ci_lo, d * r.ci_hi))
        if lo > 0:
            return "agrees"
    lo, hi = sorted((d * r.ci_lo, d * r.ci_hi))
    return "contradicts" if hi < 0 else "no_signal"


def rule(s, gs):
    if s != "agrees":
        return "drop"
    a, c = gs.count("agrees"), gs.count("contradicts")
    if c and not a:
        return "drop"
    return "keep" if a and not c else "reword"


def one(sub, d, sval, gold_filter):
    s = sub[sval]
    g = sub[gold_filter]
    return s, g


def main():
    x3 = pd.read_csv(OUT / "reread_x3_classes.csv")
    x1 = pd.read_csv(OUT / "reread_x1_stages.csv")
    rows = []
    for (c, k), t in x3.groupby(["cohort", "cls"]):
        d = int(t.predicted_sign.iloc[0])
        s = t[(t.kind == "readout") & (t.value == "S") & (t.subset == "all") & (t.matching == "C")].iloc[0]
        sm = t[(t.kind == "readout") & (t.value == "S") & (t.subset == "margin_supported")].iloc[0]
        g = t[(t.kind == "gold") & (t.subset == "all") & (t.matching == "covs")]
        st = [state(r, d) for r in g.itertuples()]
        rec = dict(finding="X3", cohort=c, item=k, predicted=d, S=s.excess_d, S_lo=s.ci_lo, S_hi=s.ci_hi, S_state=state(s, d),
                   S_margin_state=state(sm, d), golds="; ".join(f"{r.value} {r.excess_d:+.2f} ({r.ci_lo:+.2f},{r.ci_hi:+.2f})"
                                                                for r in g.itertuples()),
                   gold_states=" ".join(st), rule=rule(state(s, d), st))
        rows.append(rec)
    for (c, k), t in x1[x1.group.isin(["SC", "SC_strict"])].groupby(["cohort", "group"]):
        s = t[(t.kind == "readout") & (t.value == "S") & (t.subset == "all")].iloc[0]
        sm = t[(t.kind == "readout") & (t.value == "S") & (t.subset == "margin_supported")].iloc[0]
        g = t[(t.kind == "gold") & (t.subset == "all")]
        st = [state(r, 1) for r in g.itertuples()]
        rows.append(dict(finding="X1", cohort=c, item=k, predicted=1, S=s.excess_d, S_lo=s.ci_lo, S_hi=s.ci_hi, S_state=state(s, 1),
                         S_margin_state=state(sm, 1), golds="; ".join(f"{r.value} {r.excess_d:+.2f} ({r.ci_lo:+.2f},{r.ci_hi:+.2f})"
                                                                     for r in g.itertuples()),
                         gold_states=" ".join(st), rule=rule(state(s, 1), st)))
    v = pd.DataFrame(rows)
    v.to_csv(OUT / "verdicts.csv", index=False)
    print(v[["finding", "cohort", "item", "S_state", "S_margin_state", "gold_states", "rule"]].to_string())


if __name__ == "__main__":
    main()
