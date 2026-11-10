"""O2: the sensitivity arms beside the frozen row. Reads, writes results/analysis/O1/repair/compare_*.csv.

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.repair_tables
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.tasks.O1_orthologs import rules as RU
from analysis.tasks.O1_orthologs.repair_compare import OUT

F = RU.REPO / "results/analysis/T4_3"
ARMS = ["can3", "rep3", "rep3_shared", "rep3_norep", "can_int", "rep3_int"]
HYP = ["H1", "H2", "H3", "H4"]


def main():
    rows, hh, multi, tools, f3, sets, rw = [], [], [], [], [], [], []
    frozen = pd.read_csv(F / "scoreboard.csv")
    p0 = frozen[(frozen.family == "F1") & (frozen.score == "D")].set_index("hyp")
    st0 = json.loads((F / "row.json").read_text())
    for h in HYP:
        rows.append(dict(arm="frozen", hyp=h, n=int(p0.loc[h].n), rho=p0.loc[h].rho_p, ci_lo=p0.loc[h].ci_lo, ci_hi=p0.loc[h].ci_hi, p_holm=p0.loc[h].p_holm, status=st0["step1_status"][h]))
    rw.append(dict(arm="frozen", row=st0["step3"]["row"], h2h=json.dumps(st0["step2_head_to_head"]), modifiers=";".join(k for k, v in st0["step4_modifiers"].items() if v is True)))
    h0 = pd.read_csv(F / "head_to_head.csv")
    h0 = h0[h0.id.str.endswith(":h2h")]
    for r in h0.itertuples():
        hh.append(dict(arm="frozen", hyp=r.hyp, comparator=r.comparator, n=r.n_pair, rho_D=r.rho_p_focal, rho_B=r.rho_p_comparator, delta=r.delta, pc_lo=r.pc_lo, pc_hi=r.pc_hi, sim_hi=r.sim95_hi))
    for a in ARMS:
        d = OUT / f"scoreboard_{a}"
        if not (d / "row.json").exists():
            print("missing", a)
            continue
        sb, hd, st = pd.read_csv(d / "scoreboard.csv"), pd.read_csv(d / "head_to_head.csv"), json.loads((d / "row.json").read_text())
        p = sb[(sb.family == "F1") & (sb.score == "D")].set_index("hyp")
        for h in HYP:
            rows.append(dict(arm=a, hyp=h, n=int(p.loc[h].n), rho=p.loc[h].rho_p, ci_lo=p.loc[h].ci_lo, ci_hi=p.loc[h].ci_hi, p_holm=p.loc[h].p_holm, status=st["step1_status"][h]))
        rw.append(dict(arm=a, row=st["step3"]["row"], h2h=json.dumps(st["step2_head_to_head"]), modifiers=";".join(k for k, v in st["step4_modifiers"].items() if v is True)))
        for r in hd[hd.id.str.endswith(":h2h")].itertuples():
            hh.append(dict(arm=a, hyp=r.hyp, comparator=r.comparator, n=r.n_pair, rho_D=r.rho_p_focal, rho_B=r.rho_p_comparator, delta=r.delta, pc_lo=r.pc_lo, pc_hi=r.pc_hi, sim_hi=r.sim95_hi))
        for r in hd[hd.id.str.contains(":tool:")].itertuples():
            tools.append(dict(arm=a, id=r.id, n=r.n_pair, delta=r.delta, pc_lo=r.pc_lo, pc_hi=r.pc_hi))
        for r in sb[sb.family == "F3_figs4_6"].itertuples():
            multi.append(dict(arm=a, id=r.id, n=r.n, rho=r.rho_p, ci_lo=r.ci_lo, ci_hi=r.ci_hi))
        t = pd.read_csv(d / "family3.csv")
        for r in t[t.id.isin(["H1:S_ema", "H2:S_ema", "H3:S_ema", "H4:S_ema", "H3:beyond_pcs", "H4:primates_only", "H3:multi_nonzero:D"])].itertuples():
            f3.append(dict(arm=a, id=r.id, n=r.n, rho=r.rho_p, ci_lo=r.ci_lo, ci_hi=r.ci_hi))
        s = pd.read_csv(d / "sets.csv")
        for r in s[s.family == "F2"].itertuples():
            sets.append(dict(arm=a, id=r.id, n=r.n_set, auroc=r.auroc, q=r.q_bh, passed=r.pass_predicted_direction))
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 300, "display.max_colwidth", 90)
    for k, v in dict(hyp=rows, h2h=hh, row=rw, multi=multi, tools=tools, family3=f3, sets=sets).items():
        x = pd.DataFrame(v)
        x.to_csv(OUT / f"compare_{k}.csv", index=False)
        print("==", k)
        print(x.round(4).to_string() if k in ("hyp", "row", "multi", "family3") else "")
    x = pd.DataFrame(hh).pivot_table(index=["hyp", "comparator"], columns="arm", values="delta")
    print(x.round(3).to_string())
    y = pd.DataFrame(tools)
    print(y.pivot_table(index="id", columns="arm", values="delta").round(3).to_string())
    print(pd.DataFrame(sets).pivot_table(index="id", columns="arm", values="auroc").round(3).to_string())
    print(pd.read_csv(OUT / "three_seed_range.csv").round(4).to_string())


if __name__ == "__main__":
    main()
