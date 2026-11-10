"""Part B6, per claim: each gene named in the text, in the cohort its claim is about: flags, raw counts per species, D, S_ema, strict support.

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.named_claims
"""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.tasks.O1_orthologs import rules as RU
from analysis.tasks.O1_orthologs.flags import sp_key

warnings.filterwarnings("ignore")
LIV = "CPS1 OTC ASS1 ASL ARG1 PCK1 PAH HAL ARG2 CYP7A1 ALAS1 ACOX1 PDK4 GOT1 ALDOB BCKDHB PCCA ACMSD DHTKD1 HNF4A DDX4 COL17A1 F3 SELE VCAM1 CD40LG CR2 KIT PDGFRB C1QC STAB1 STAB2 MERTK RUNX1 ETV1 HPRT1 MT-CO1 MT-ND2 MT-ND3 MT-ND4 MT-CYB MT-ATP6 MT-CO2".split()
CLAIMS = [("liv_4sp", LIV), ("liv_hmouse", ["STAB1", "STAB2", "MERTK", "PNPLA3", "CR2", "PDK4", "ALAS1", "ACOX1", "HNF1A", "NR1H3"]), ("liv_hmonkey", ["CPS1", "ETV1", "HPRT1", "RUNX1"]),
          ("test_hmouse", "ZPBP2 ADAM2 IZUMO4 SLC9C1 LDHC TEX101 MT-CO1 MT-ND4 MT-CYB SPINK2 PRM2 TNP1".split()), ("test_3sp", "ZPBP2 ADAM2 IZUMO4 SLC9C1 LDHC TEX101 SPINK2 DEGS2 RPL26 PRM2 TNP1".split()), ("test_hmac", ["SPINK2", "PRM2", "TNP1", "MT-CO1"]),
          ("bone_marrow", "S100A8 CD3E CD79A CD4 CD3D ELANE CD19".split()), ("kidney", "PODXL CUBN UMOD LRP2 TRPM6 SLC12A3 SLC12A1 AQP2 NPHS1".split()),
          ("islet", "PPY CHGA IAPP GCG SST NKX6-1 MAFA ARX PECAM1 PLVAP".split()), ("vz_hmouse", "MYOC MGP DCN DES MYH11 TYR PMEL ANGPTL7 DNAJB1 HSPH1 BAG3".split())]


def main():
    G = pd.read_parquet(RU.RES / "flags_by_cohort.parquet")
    P = pd.read_parquet(RU.RES / "table_pairs_flags.parquet")
    P["R2"] = P.T1 | P.F_retro | P.F_named | (P.F_lowconf & ~((P.goc_score >= 50) & (P.wga_coverage >= 50)))
    rows = []
    for c, syms in CLAIMS:
        v = RU.COHORTS[c]
        a = ad.read_h5ad(v["h5ad"])
        spc = a.obs["species"].astype(str).to_numpy()
        X = sp.csc_matrix(a.X)
        g = G[G.cohort == c].set_index("symbol")
        H = pd.read_parquet(RU.REPO / f"results/analysis/H1/genes/{c}.parquet").set_index("symbol")
        pp = P[P.table == v["table"]]
        for s in syms:
            if s not in g.index:
                rows.append(dict(cohort=c, symbol=s, note="not in cohort"))
                continue
            r = g.loc[s]
            j = a.var_names.get_loc(r.var_name)
            d = dict(cohort=c, symbol=s, T1=bool(r.T1), T2=bool(r.T2), T2L=bool(r.T2L), M_false_zero=bool(r.M_false_zero), M_wrong_row=bool(r.M_wrong_row), zero_any=bool(r.zero_any))
            cnt = []
            ps = pp[pp.hsapiens_gene_name == s]
            own = [sp_key(x) for x in sorted(set(spc))]
            ps = ps[ps.species.isin(own)]
            d["R2"] = bool(ps.R2.any())
            for x in sorted(set(spc)):
                col = X[:, j][spc == x]
                cnt.append(f"{x}:{int(col.sum())}/{int((col > 0).sum())}of{int((spc == x).sum())}")
            d["umi_cells_by_species"] = "; ".join(cnt)
            d["pairs"] = "; ".join(f"{p.species}:{p.species_gene_name or p.gene_id} id{p.identity:.0f} goc{p.goc_score:.0f} wga{p.wga_coverage:.0f} hc{p.is_high_confidence:.0f}" for p in ps.itertuples())
            if s in H.index:
                h = H.loc[s]
                d.update(D=float(h.D), S_ema=float(h.S_ema), strict=bool(h.strict_support))
            rows.append(d)
    R = pd.DataFrame(rows)
    R.to_csv(RU.RES / "named_claims.csv", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 30, "display.max_rows", 300, "display.max_colwidth", 110)
    f = R[R.T2.fillna(False) | R.T2L.fillna(False) | R.M_false_zero.fillna(False) | R.M_wrong_row.fillna(False) | R.zero_any.fillna(False) | R.note.notna()]
    print(len(R), "claim rows;", len(f), "with any flag or zero")
    print(f[["cohort", "symbol", "T2", "R2", "T2L", "M_false_zero", "zero_any", "strict", "D", "S_ema", "umi_cells_by_species", "pairs", "note"]].round(3).to_string())


if __name__ == "__main__":
    main()
