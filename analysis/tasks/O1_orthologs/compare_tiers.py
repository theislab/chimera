"""O1 part B: scoreboard per tier beside the frozen row. Frozen files are read, never written.

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.compare_tiers
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.tasks.O1_orthologs import rules as RU

F = RU.REPO / "results/analysis/T4_3"
T56 = RU.REPO / "results/analysis/T5_6/scoreboard"
TIERS = ["T1", "T2", "T2L", "MAP", "T3", "TZ", "T2T", "R2", "R3"]
HYP = ["H1", "H2", "H3", "H4"]


def main():
    sb0, h0, st0 = pd.read_csv(F / "scoreboard.csv"), pd.read_csv(F / "head_to_head.csv"), json.loads((F / "row.json").read_text())
    drop = pd.read_csv(RU.RES / "random_drop.csv") if (RU.RES / "random_drop.csv").exists() else None
    rows, hh, ro, multi, fam3, sets, tools = [], [], [], [], [], [], []
    p0 = sb0[(sb0.family == "F1") & (sb0.score == "D")].set_index("hyp")
    for t in TIERS:
        d = RU.RES / f"scoreboard_{t}"
        if not (d / "DONE").exists():
            continue
        sb, hd, rw = pd.read_csv(d / "scoreboard.csv"), pd.read_csv(d / "head_to_head.csv"), json.loads((d / "row.json").read_text())
        p = sb[(sb.family == "F1") & (sb.score == "D")].set_index("hyp")
        for h in HYP:
            a, b = p0.loc[h], p.loc[h]
            r = dict(tier=t, hyp=h, n_frozen=int(a.n), n=int(b.n), rho_frozen=a.rho_p, rho=b.rho_p, delta=b.rho_p - a.rho_p, ci_lo=b.ci_lo, ci_hi=b.ci_hi, p_holm=b.get("p_holm"),
                     status_frozen=st0["step1_status"][h], status=rw["step1_status"][h])
            if drop is not None:
                x = drop[(drop.tier == t) & (drop.hyp == h)]
                if len(x):
                    x = x.iloc[0]
                    r.update(k=int(x.k_dropped), drop_lo=x.q025, drop_hi=x.q975, outside_random=bool(b.rho_p < x.q025 or b.rho_p > x.q975))
                    r["material"] = bool(abs(r["delta"]) >= 0.01 and r["outside_random"])
            rows.append(r)
        ro.append(dict(tier=t, row=rw["step3"].get("row") if isinstance(rw["step3"], dict) else rw["step3"], label=rw["step3"].get("label") if isinstance(rw["step3"], dict) else None,
                       h2h=json.dumps(rw["step2_head_to_head"]), modifiers=json.dumps(rw["step4_modifiers"])))
        a = h0[h0.id.str.endswith(":h2h")].set_index(["hyp", "comparator"])
        for r in hd[hd.id.str.endswith(":h2h")].itertuples():
            f = a.loc[(r.hyp, r.comparator)]
            hh.append(dict(tier=t, hyp=r.hyp, comparator=r.comparator, n=r.n_pair, delta_frozen=f.delta, delta=r.delta, change=r.delta - f.delta, pc_lo=r.pc_lo, pc_hi=r.pc_hi, sim95_hi=r.sim95_hi))
        m0 = sb0[sb0.family == "F3_figs4_6"].set_index("id")
        for r in sb[sb.family == "F3_figs4_6"].itertuples():
            multi.append(dict(tier=t, id=r.id, n=r.n, rho_frozen=m0.loc[r.id].rho_p, rho=r.rho_p, ci_lo=r.ci_lo, ci_hi=r.ci_hi))
        f30, f3 = pd.read_csv(F / "family3.csv").set_index("id"), pd.read_csv(d / "family3.csv")
        for r in f3[f3.id.str.contains("S_ema|margin_support|autosome|primates|integrated|beyond_pcs$")].itertuples():
            if r.id in f30.index and pd.notna(r.rho_p):
                fam3.append(dict(tier=t, id=r.id, n=r.n, rho_frozen=f30.loc[r.id].rho_p, rho=r.rho_p, ci_lo=r.ci_lo, ci_hi=r.ci_hi))
        s0, s1 = pd.read_csv(F / "sets.csv").set_index("id"), pd.read_csv(d / "sets.csv")
        for r in s1[s1.family == "F2"].itertuples():
            sets.append(dict(tier=t, id=r.id, n_set=r.n_set, auroc_frozen=s0.loc[r.id].auroc, auroc=r.auroc, q=r.q_bh, passed=r.pass_predicted_direction))
        t0 = T56 / "head_to_head.csv"
        if t0.exists():
            a0 = pd.read_csv(t0)
            a0 = a0[a0.id.str.contains(":tool:")].set_index("id")
            for r in hd[hd.id.str.contains(":tool:")].itertuples():
                if r.id in a0.index:
                    tools.append(dict(tier=t, id=r.id, n=r.n_pair, delta_frozen=a0.loc[r.id].delta, delta=r.delta, change=r.delta - a0.loc[r.id].delta, pc_lo=r.pc_lo, pc_hi=r.pc_hi))
    out = dict(sensitivity=pd.DataFrame(rows), head_to_head=pd.DataFrame(hh), rows=pd.DataFrame(ro), multi=pd.DataFrame(multi), family3=pd.DataFrame(fam3), sets=pd.DataFrame(sets), tools=pd.DataFrame(tools))
    pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_rows", 400, "display.max_colwidth", 60)
    for k, v in out.items():
        v.to_csv(RU.RES / f"compare_{k}.csv", index=False)
        print("==", k)
        print(v.round(4).to_string())


if __name__ == "__main__":
    main()
