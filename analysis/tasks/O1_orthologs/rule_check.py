"""Step 4: is_high_confidence as a filter, count-free and gold-free. Compara's flag has an identity arm, so it can fail true syntenic orthologs of fast proteins, which are what
Chimera reads. Classes of one-to-one protein pairs: clean; low_syntenic (not high-confidence, GOC and WGA both at least 50); low_nosynteny (not high-confidence, the rest); plus the wrong-pair signatures.
Readouts: exact zero in the species, B1, dN/dS, identity, on the per-gene tables of H1_harmonise (no gold). Post hoc, after reading PODXL and SPINK2 in the named list.

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.rule_check
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.tasks.O1_orthologs import rules as RU

Y6_WRONG = {"PIN1": "mmusculus", "GJA1": "mmusculus", "RPL36AL": "mmusculus", "FHL1": "mmusculus", "TMSB10": "mmusculus", "CLDN4": "mmusculus", "RPS27": "mmusculus", "TPM3": "mmusculus", "RPL7": "mmurinus",
            "RPS15A": "mmurinus", "HNRNPA2B1": "mmurinus", "JUND": "mmurinus", "GDI2": "mmurinus", "RPL18": "sscrofa", "RPL23": "sscrofa", "SEC11A": "mmulatta", "DNAJC7": "mmulatta", "RPL28": "mmulatta"}


def cls(P):
    syn = (P.goc_score >= 50) & (P.wga_coverage >= 50)
    wrong = P.F_retro | P.F_named
    out = pd.Series("clean", index=P.index)
    out[P.F_lowconf & syn] = "low_syntenic"
    out[P.F_lowconf & ~syn] = "low_nosynteny"
    out[wrong] = "retro_or_named"
    out[P.T1] = "not_protein"
    return out


def main():
    P = pd.read_parquet(RU.RES / "table_pairs_flags.parquet")
    P["cls"] = cls(P)
    P["R2"] = P.T1 | P.F_retro | P.F_named | (P.F_lowconf & ~((P.goc_score >= 50) & (P.wga_coverage >= 50)))
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    t = P.groupby(["table", "species"]).cls.value_counts().unstack(fill_value=0)
    t["T2"] = P.groupby(["table", "species"]).T2.sum()
    t["R2"] = P.groupby(["table", "species"]).R2.sum()
    t.to_csv(RU.RES / "rule_classes_by_table.csv")
    print(t)
    x = P[P.hsapiens_gene_name.isin(Y6_WRONG)]
    x = x[[Y6_WRONG[a] == b for a, b in zip(x.hsapiens_gene_name, x.species)]]
    print("Y6 named pairs:", len(x), "T2", int(x.T2.sum()), "R2", int(x.R2.sum()))
    print(x[~x.R2][["table", "species", "hsapiens_gene_name", "species_gene_name", "identity", "goc_score", "wga_coverage", "is_high_confidence"]].drop_duplicates(["species", "hsapiens_gene_name"]).to_string())
    rows = []
    G = pd.read_parquet(RU.RES / "flags_by_cohort.parquet")
    for c, sp in [("liv_hmouse", "mmusculus"), ("test_hmouse", "mmusculus"), ("kidney", "mmusculus"), ("islet", "mmusculus"), ("vz_hmouse", "mmusculus"), ("bone_marrow", "mmusculus"), ("bone_marrow", "mmurinus"), ("vz_hmmul", "mmulatta"), ("test_hmac", "mmulatta")]:
        tb = RU.COHORT_TABLE[c]
        g = pd.read_parquet(RU.REPO / f"results/analysis/H1/genes/{c}.parquet")
        p = P[(P.table == tb) & (P.species == sp)].drop_duplicates("hsapiens_gene_id")
        z = G[G.cohort == c].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")[f"zero_{sp}"]
        m = g.merge(p[["hsapiens_gene_id", "cls", "identity_table"]], on="hsapiens_gene_id")
        m["zero"] = m.hsapiens_gene_id.map(z)
        e = m[m.expr >= m.expr.median()]
        for k, x in m.groupby("cls"):
            xe = e[e.cls == k]
            rows.append(dict(cohort=c, species=sp, cls=k, n=len(x), identity=float(x.identity_table.median()), dNdS=float(x.dNdS.median()) if "dNdS" in x else np.nan, zero_share=float(x.zero.mean()),
                             detection=float(x.detection.median()), B1=float(x.B1.median()), n_upper_expr=len(xe), B1_upper_expr=float(xe.B1.median()) if len(xe) else np.nan, S_upper_expr=float(xe.S_ema.median()) if len(xe) else np.nan))
    R = pd.DataFrame(rows)
    R.to_csv(RU.RES / "rule_classes_behaviour.csv", index=False)
    print(R.round(3).to_string())


if __name__ == "__main__":
    main()
