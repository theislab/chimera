"""E3 analysis D: named genes per home group, with replication over the cohorts of the same tissue.

Naming rule (rules.py D): in the cohort the gene is named for, count-supported, nameable and not saturated on the
per-gene gate, sign of D_free stable over its seeds, and the canonical D of the same sign (canonical saturation is
reported, not required). Replication: z of D_free (D_free / SD over count-supported genes) in every cohort of the
tissue where the gene is count-supported; ``rep_same_sign`` = share of those cohorts with the named sign.

    PYTHONPATH=$PWD python revision/tasks/E3_cells/named.py
Output ``results/revision/E3/named_genes.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from revision.tasks.E3_cells import rules as R
from revision.tasks.H1_harmonise import rules as RU

TOP = 25


def main() -> int:
    reg = RU.cohorts()
    G = {c: pd.read_parquet(R.OUT / "genes" / f"{c}.parquet") for c in reg}
    rows = []
    for c, g in G.items():
        tis = reg[c]["tissue"]
        peers = [k for k in reg if reg[k]["tissue"] == tis]
        zs = {}
        for k in peers:
            h = G[k]
            sd = h.D_free[h.count_support_free.fillna(False)].std()
            zs[k] = (h.assign(zf=h.D_free / sd).loc[h.count_support_free.fillna(False)].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id").zf)
        sd = g.D_free[g.count_support_free.fillna(False)].std()
        ok = (g.count_support & g.count_support_free.fillna(False) & g.nameable_free.fillna(False) & ~g.saturated_free.fillna(True)
              & g.sign_D_stable_free.fillna(False) & (np.sign(g.D) == np.sign(g.D_free)))
        for grp, sub in g[ok & (g.subgroup != "broad")].groupby("subgroup"):
            for sign, part in (("above", sub[sub.D_free > 0].nlargest(TOP, "D_free")), ("below", sub[sub.D_free < 0].nsmallest(TOP, "D_free"))):
                for v, r in part.iterrows():
                    rep = {k: zs[k].get(r.hsapiens_gene_id, np.nan) for k in peers}
                    vals = np.array([x for x in rep.values() if np.isfinite(x)])
                    rows.append(dict(cohort=c, tissue=tis, subgroup=grp, home=r.home, sign=sign, symbol=r.symbol, hsapiens_gene_id=r.hsapiens_gene_id,
                                     D_free=r.D_free, z_free=r.D_free / sd, D=r.D, S_ema=r.S_ema, B1=r.B1, lfc_peak_abs=r.lfc_peak_abs,
                                     pi_raw=r.pi_raw, home_spec=r.home_spec, saturated_canonical=bool(r.saturated_flag),
                                     n_peer_cohorts=len(vals), rep_same_sign=float(np.mean(np.sign(vals) == (1 if sign == "above" else -1))) if len(vals) else np.nan,
                                     **{f"z__{k}": x for k, x in rep.items()}))
    out = pd.DataFrame(rows)
    out.to_csv(R.OUT / "named_genes.csv", index=False)
    print(len(out), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
