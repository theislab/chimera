"""Derive human-vs-mouse Kupffer-cell marker conservation from Guilliams et al. 2022 (Cell 185:379).

Usage: python -I derive_guilliams2022.py <venn_tsv> <out_tsv>

<venn_tsv> = Table S9 (mmc9.xlsx), sheet "Conserved_Unique_KC_DEGs", converted to TSV with xlsx_to_tsv.py.
The sheet lists the regions of the Venn diagram of Figure S7J: "top 50 KC genes per species" across
human, mouse, pig, hamster, chicken and zebrafish (column Names = species set of the region, Genes = one gene per row;
Names/Total are filled only on the first row of each region).

Rule (human vs mouse only; other species ignored):
  human in region and mouse in region     -> conserved_control (direction conserved)
  human in region and mouse not in region -> human_enriched    (direction up_in_human)
  mouse in region and human not in region -> mouse_enriched    (direction up_in_mouse)
  neither                                 -> dropped (not informative for human vs mouse)
Text rows (verbatim, Results section "Potential role for CD45- hepatic cells ..." and Figure 6E legend) are appended.
"""
import sys

import pandas as pd

Y2 = {"FASN", "APOA5", "GPAM", "ACSL5", "DPP4", "NAGS", "OTC", "ASL", "CPS1", "VNN1", "THBD", "INMT", "IGFBP2",
      "HNF4A", "CD5L", "CD163", "C1QB", "PCK2", "SLC2A2", "APOBEC1", "PPARA", "PNPLA3", "SHBG", "FMO1", "FMO3",
      "CLEC4F", "ADGRE1", "CCR1", "FZD5", "WNT2"}  # genes of the earlier Y2 text list (flag only)

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]

Q_TEXT = ("\"This analysis identified CCL23 and CCL14 expression by LSECs and CCL16 by hepatocytes that binds to CCR1 "
          "expressed by human KCs but not murine KCs ( Figure 6 E).\" (Results, section 'Potential role for CD45- hepatic "
          "cells in determining macrophage localization'; Figure 6E legend: 'NicheNet predicted ligand-receptor pairs between "
          "KCs and LSECs, HSCs and hepatocytes uniquely expressed in the human liver')")
TEXT_ROWS = [
    ("CCR1", "Kupffer cell", "up_in_human", "receptor expressed by human KCs but not murine KCs; overlap_Y2"),
    ("CCL23", "LSEC", "up_in_human", "ligand of a human-unique NicheNet pair; no mouse ortholog (not 1:1)"),
    ("CCL14", "LSEC", "up_in_human", "ligand of a human-unique NicheNet pair; no 1:1 mouse ortholog"),
    ("CCL16", "hepatocyte", "up_in_human", "ligand of a human-unique NicheNet pair; no 1:1 mouse ortholog"),
]


def main(venn_tsv, out_tsv):
    v = pd.read_csv(venn_tsv, sep="\t", dtype=str)
    v["Names"] = v["Names"].ffill()
    v = v[v["Genes"].notna() & (v["Genes"].str.strip() != "")]
    out = []
    for _, r in v.iterrows():
        sp = set(r["Names"].split())
        h, m = "Human" in sp, "Mouse" in sp
        if h and m:
            s, d = "conserved_control", "conserved"
        elif h:
            s, d = "human_enriched", "up_in_human"
        elif m:
            s, d = "mouse_enriched", "up_in_mouse"
        else:
            continue
        out.append(dict(
            source_id="guilliams2022", set=s, gene_symbol=r["Genes"].strip(),
            symbol_species="human" if h else "mouse",
            contrast="human_vs_mouse", cell_type="Kupffer cell", direction=d,
            effect="membership in the per-species KC gene list of the Figure S7J Venn (legend: top 50 KC genes per species; the table holds 123 human and 126 mouse genes)",
            evidence=f"Table S9 sheet Conserved_Unique_KC_DEGs; Venn region '{r['Names']}' (Figure S7J)",
            circular="no",
            note="KC-specificity (rank) difference, not proof of absence in the other species' KCs"
                 + ("; overlap_Y2" if r["Genes"].strip().upper() in Y2 else "")))
    for g, ct, d, n in TEXT_ROWS:
        out.append(dict(source_id="guilliams2022", set="text_known_difference", gene_symbol=g, symbol_species="human",
                        contrast="human_vs_mouse", cell_type=ct, direction=d, effect="qualitative",
                        evidence="NicheNet, human-unique ligand-receptor pairs (Figure 6E)", circular="no",
                        note=f"{n}. {Q_TEXT}"))
    df = pd.DataFrame(out, columns=COLS)
    dup = df.duplicated(["set", "gene_symbol", "cell_type"]).sum()
    df.to_csv(out_tsv, sep="\t", index=False)
    print(df.groupby("set").gene_symbol.nunique().to_dict(), "duplicates:", dup)


if __name__ == "__main__":
    main(*sys.argv[1:3])
