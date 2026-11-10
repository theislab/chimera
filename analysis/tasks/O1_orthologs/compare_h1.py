"""Part B4: recurrence and set positions of analysis/tasks/H1_harmonise without flagged genes, beside its own outputs.

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.compare_h1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.tasks.O1_orthologs import rules as RU

H1 = RU.REPO / "results/analysis/H1"
STRICT = ["D", "S", "D_tau", "D_block", "S_block"]


def recur(d):
    r = pd.read_csv(d / "recurrence.csv")
    return r.pivot(index="key", columns="readout", values="recurrent").fillna(False).astype(bool), r.pivot(index="key", columns="readout", values="mean_z")


def main():
    b0, z0 = recur(H1)
    s0 = b0[STRICT].all(axis=1)
    rows, lost = [], []
    for t in ["T2", "T3"]:
        d = RU.RES / f"h1_{t}"
        if not (d / "recurrence.csv").exists():
            continue
        b, z = recur(d)
        st = b[STRICT].all(axis=1)
        k = b0.index.intersection(b.index)
        for r in ("D", "S", "B1", "pi", "dNdS", "D_tau", "D_block", "S_block"):
            x0, x1 = b0.loc[k, r], b.loc[k, r]
            rows.append(dict(tier=t, readout=r, frozen=int(x0.sum()), now=int(x1.sum()), lost=int((x0 & ~x1).sum()), gained=int((~x0 & x1).sum()), rho_mean_z=float(spearmanr(z0.loc[k, r], z.loc[k, r], nan_policy="omit")[0]), max_abs_dz=float((z0.loc[k, r] - z.loc[k, r]).abs().max())))
        rows.append(dict(tier=t, readout="strict intersection", frozen=int(s0.sum()), now=int(st.sum()), lost=int((s0.reindex(k) & ~st.reindex(k)).sum()), gained=int((~s0.reindex(k) & st.reindex(k)).sum())))
        lost += [dict(tier=t, key=x, change="lost from strict") for x in k[s0.reindex(k) & ~st.reindex(k)]] + [dict(tier=t, key=x, change="lost from D") for x in k[b0.loc[k, "D"] & ~b.loc[k, "D"]]] + [dict(tier=t, key=x, change="gained on D") for x in k[~b0.loc[k, "D"] & b.loc[k, "D"]]]
        a0, a1 = pd.read_csv(H1 / "axis.csv"), pd.read_csv(d / "axis.csv")
        m = a0.merge(a1, on=["cohort", "universe"], suffixes=("_f", "_n"))
        rows.append(dict(tier=t, readout="axis |rho(D,pi)| max", frozen=float(m.rho_D_pi_f.abs().max()), now=float(m.rho_D_pi_n.abs().max())))
        rows.append(dict(tier=t, readout="axis |rho(D,dNdS)| max", frozen=float(m.rho_D_dNdS_f.abs().max()), now=float(m.rho_D_dNdS_n.abs().max())))
        rows.append(dict(tier=t, readout="axis rho(pi,dNdS) range", frozen=f"{m.rho_pi_dNdS_f.min():.3f} to {m.rho_pi_dNdS_f.max():.3f}", now=f"{m.rho_pi_dNdS_n.min():.3f} to {m.rho_pi_dNdS_n.max():.3f}"))
        sa0, sa1 = pd.read_csv(H1 / "set_axis.csv").set_index(["a", "b"]).spearman_over_sets, pd.read_csv(d / "set_axis.csv").set_index(["a", "b"]).spearman_over_sets
        for k2 in [("D", "S"), ("D", "B1"), ("D", "pi"), ("D", "dNdS"), ("pi", "dNdS")]:
            rows.append(dict(tier=t, readout=f"set position {k2}", frozen=float(sa0.loc[k2]), now=float(sa1.loc[k2])))
        for nm, f in (("tissue_specific.csv", "tissue-specific D"), ):
            t0, t1 = pd.read_csv(H1 / nm), pd.read_csv(d / nm)
            kk = "called" if "called" in t0 else ("tissue_specific" if "tissue_specific" in t0 else None)
            if kk:
                c0 = set(t0[(t0.readout == "D") & t0[kk].astype(bool)].key)
                c1 = set(t1[(t1.readout == "D") & t1[kk].astype(bool)].key)
                rows.append(dict(tier=t, readout=f, frozen=";".join(sorted(c0)), now=";".join(sorted(c1))))
        l0, l1 = pd.read_csv(H1 / "lineage_z.csv"), pd.read_csv(d / "lineage_z.csv")
        lm = l0.merge(l1, on=["cohort", "readout", "key"], suffixes=("_f", "_n"))
        lm = lm[lm.readout.isin(["D", "S"])]
        lost += [dict(tier=t, key=f"{r.cohort} {r.readout} {r.key}", change=f"lineage z {r.z_f:.2f} to {r.z_n:.2f}") for r in lm[(lm.z_f - lm.z_n).abs() >= 0.3].itertuples()]
        h0 = pd.read_csv(H1 / "set_z.csv")
        h1 = pd.read_csv(d / "set_z.csv")
        hm = h0.merge(h1, on=["cohort", "readout", "key"], suffixes=("_f", "_n"))
        hx = hm[(hm.readout == "D") & hm.key.isin(["hallmark|Heme Metabolism", "hallmark|E2F Targets"])][["cohort", "key", "z_f", "z_n"]]
        hx = hx[hx.cohort.isin(["bone_marrow", "kidney"])]
        lost += [dict(tier=t, key=f"{r.cohort} {r.key}", change=f"z {r.z_f:.2f} to {r.z_n:.2f}") for r in hx.itertuples()]
    R = pd.DataFrame(rows)
    R.to_csv(RU.RES / "compare_h1.csv", index=False)
    L = pd.DataFrame(lost)
    L.to_csv(RU.RES / "compare_h1_changes.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_rows", 200, "display.max_colwidth", 80)
    print(R.round(3).to_string())
    print(L.to_string())


if __name__ == "__main__":
    main()
