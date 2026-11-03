"""Derive conserved vs non-conserved liver zonation genes from Aizarani et al. 2019 (Nature 572:199).

Usage: python -I derive_aizarani2019.py <data_dir> <out_tsv>

Inputs (tab-delimited text despite the .xls suffix):
  41586_2019_1373_MOESM8_ESM.xls  Supplementary Table 6, hepatocytes (human vs mouse Halpern 2017, 9 bins)
  41586_2019_1373_MOESM9_ESM.xls  Supplementary Table 7, liver endothelial cells (MaVEC+LSEC; mouse Halpern 2018, 4 bins)

Paper's rule (Supplementary Note 3): among genes with significant zonation, "only 68% (60%) ... exhibited
a positive correlation of zonation profiles for hepatocytes (endothelial cells)". The stated percentages are
reproduced exactly by: significant = human adj. P < 0.05 OR mouse P < 0.05; conserved = Pearson r > 0.
  set = conserved_control      if significant and r > 0
  set = nonconserved_zonation  if significant and r <= 0
  set = zonation_not_significant otherwise (kept for completeness; direction from the sign of r)
"""
import sys
from pathlib import Path

import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]
SIG = 0.05


def rows(path, cell_type, mouse_p_col, mouse_ref):
    df = pd.read_csv(path, sep="\t")
    out = []
    for _, r in df.iterrows():
        hp, mp, rho = float(r["HumanPadj"]), float(r[mouse_p_col]), float(r["ProfileCorrelation"])
        sig = (hp < SIG) or (mp < SIG)
        if sig and rho > 0:
            s, d = "conserved_control", "conserved"
        elif sig:
            s, d = "nonconserved_zonation", "divergent"
        else:
            s, d = "zonation_not_significant", "conserved" if rho > 0 else "divergent"
        mmod = f"; mouse SOM module {r['MouseModule']}" if "MouseModule" in df.columns else ""
        out.append(dict(
            source_id="aizarani2019", set=s, gene_symbol=r["HumanGeneID"], symbol_species="human",
            contrast="human_vs_mouse", cell_type=cell_type, direction=d,
            effect=f"zonation_profile_pearson_r={rho:.4g}",
            evidence=f"human_padj={hp:.3g}; mouse_{'padj' if 'Padj' in mouse_p_col else 'p'}={mp:.3g}; "
                     f"significant if human_padj<0.05 or mouse p<0.05; conserved if r>0 (Suppl. Note 3)",
            circular="no",
            note=f"mouse ortholog {r['MouseGeneID']} ({mouse_ref}); human SOM module {r['HumanModule']}{mmod}; "
                 f"zonation-profile (spatial) divergence, not a cell-type mean-expression difference"))
    return out


def main(data_dir, out_tsv):
    d = Path(data_dir)
    out = rows(d / "41586_2019_1373_MOESM8_ESM.xls", "hepatocyte", "MousePadj", "Halpern 2017, 9 zones")
    out += rows(d / "41586_2019_1373_MOESM9_ESM.xls", "liver endothelial cell (MaVEC+LSEC)",
                "MouseZonation_pvalue_Halpern_et_al", "Halpern 2018, 4 layers")
    df = pd.DataFrame(out, columns=COLS)
    df.to_csv(out_tsv, sep="\t", index=False)
    for ct, g in df.groupby("cell_type"):
        sig = g[g.set != "zonation_not_significant"]
        print(ct, g.set.value_counts().to_dict(), "pct conserved among significant:",
              round(100 * (sig.set == "conserved_control").mean(), 1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
