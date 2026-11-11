"""Literature list: Caglayan et al. 2023 Nature 620:145 (posterior cingulate cortex BA23; human, chimpanzee, rhesus macaque).

Usage:
    python -I derive_caglayan2023.py <converted_dir> <out_tsv>
    converted_dir = chimera/data/analysis/E6/caglayan2023/converted (TSVs made by xlsx_to_tsv.py from Table S3)

Supplementary Table 3 ("Species specific genes per cell type"), pseudobulk edgeR QLRT per cell type,
Methods: DEG = FDR < 0.05 and |logFC| > 0.3; species-specific per the Methods:
"HS-Genes were determined as DEGs that are H > C = M or H < C = M (C = M was determined if FDR > 0.1)".
Sets written (every row of the sheet, no extra filtering):
  human_specific            sheet Human_Specific_Genes; direction from column Regulation (Human_UP / Human_DOWN).
  ape_vs_macaque_divergent  sheet Macaque-Specific_Genes; the paper's "macaque versus human-chimpanzee genes"
                            (column MvsHC = Macaque_UP / Macaque_DOWN). Direction is written human-centric:
                            Macaque_UP -> down_in_human (human and chimpanzee lower than macaque), and vice versa.
Sign check: in Human_Specific rows, Human_UP rows have avg_logfc_hc > 0 (positive = higher in human).
What the table itself shows (checked 2026-11-03): every Human_Specific row has FDR_hc and FDR_hm < 0.05,
|avg_logfc_hc| and |avg_logfc_hm| > 0.25 (not 0.3) and the same sign for hc and hm; adj_p_value_cm is <= 0.1 in
5151 of 7881 rows, so the "C = M if FDR > 0.1" rule is not visible in the cm columns. We keep the authors'
labels as published and do not re-apply any threshold.
"""
import sys
from pathlib import Path

import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type", "direction",
        "effect", "evidence", "circular", "note"]
STEM = "NIHMS1994710-supplement-Table_S3__"


def fmt(x):
    return f"{float(x):.4g}"


def main(conv_dir, out_tsv):
    d = Path(conv_dir)
    hs = pd.read_csv(d / f"{STEM}Human_Specific_Genes.tsv", sep="\t", dtype=str)
    ms = pd.read_csv(d / f"{STEM}Macaque-Specific_Genes.tsv", sep="\t", dtype=str)
    assert (hs["Evolution"] == "Human_Specific").all(), hs["Evolution"].value_counts()
    assert set(hs["Regulation"]) <= {"Human_UP", "Human_DOWN"}
    lfc = hs["avg_logfc_hc"].astype(float)
    assert ((lfc > 0) == (hs["Regulation"] == "Human_UP")).all(), "sign convention check failed"

    a = pd.DataFrame({
        "source_id": "caglayan2023",
        "set": "human_specific",
        "gene_symbol": hs["Gene"],
        "symbol_species": "human",
        "contrast": "human_vs_chimpanzee_macaque",
        "cell_type": hs["CellType"],
        "direction": hs["Regulation"].map({"Human_UP": "up_in_human", "Human_DOWN": "down_in_human"}),
        "effect": "lnFC_hc=" + hs["avg_logfc_hc"].map(fmt) + ";lnFC_hm=" + hs["avg_logfc_hm"].map(fmt),
        "evidence": "TableS3 authors' Human_Specific label (H>C=M or H<C=M); in table: H-C and H-M FDR<0.05, |lnFC|>0.25, same sign",
        "circular": "no",
        "note": ("FDR_hc=" + hs["adj_p_value_hc"].map(lambda x: f"{float(x):.3g}")
                 + ";FDR_hm=" + hs["adj_p_value_hm"].map(lambda x: f"{float(x):.3g}")
                 + ";MvsHC=" + hs["MvsHC"]),
    })

    assert set(ms["MvsHC"]) <= {"Macaque_UP", "Macaque_DOWN"}, ms["MvsHC"].value_counts()
    b = pd.DataFrame({
        "source_id": "caglayan2023",
        "set": "ape_vs_macaque_divergent",
        "gene_symbol": ms["Gene"],
        "symbol_species": "human",
        "contrast": "human_chimpanzee_vs_macaque",
        "cell_type": ms["CellType"],
        "direction": ms["MvsHC"].map({"Macaque_UP": "down_in_human", "Macaque_DOWN": "up_in_human"}),
        "effect": "lnFC_hm=" + ms["avg_logfc_hm"].map(fmt) + ";lnFC_cm=" + ms["avg_logfc_cm"].map(fmt),
        "evidence": "TableS3 edgeR QLRT pseudobulk; macaque vs human-chimpanzee genes (M differs from H and C)",
        "circular": "no",
        "note": ("MvsHC=" + ms["MvsHC"] + ";Evolution=" + ms["Evolution"]
                 + ";FDR_hm=" + ms["adj_p_value_hm"].map(lambda x: f"{float(x):.3g}")
                 + ";FDR_cm=" + ms["adj_p_value_cm"].map(lambda x: f"{float(x):.3g}")),
    })
    out = pd.concat([a, b], ignore_index=True)[COLS]
    out.to_csv(out_tsv, sep="\t", index=False)
    print(out.groupby("set").agg(rows=("gene_symbol", "size"), genes=("gene_symbol", "nunique"),
                                 cell_types=("cell_type", "nunique")).to_string())
    print(out.groupby(["set", "direction"]).size().to_string())
    print(out.pivot_table(index="cell_type", columns="set", values="gene_symbol", aggfunc="size").to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
