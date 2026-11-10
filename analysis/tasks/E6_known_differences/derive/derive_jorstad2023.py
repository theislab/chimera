"""E6 literature list: Jorstad et al. 2023 Science 382:eade9516 (MTG; human, chimpanzee, gorilla, macaque, marmoset).

Usage:
    python -I derive_jorstad2023.py <data_dir> <out_tsv>
    data_dir = chimera/data/analysis/E6/jorstad2023 (raw downloads, see MANIFEST.tsv there)

Sets written (paper's own definitions and thresholds, no extra filtering):
  human_specific            Table S7 (consensus type species-specific expression), rows with species == "human":
                            DESeq2 Wald test, pseudobulk, human vs the other four primates, per consensus type.
                            The table is already restricted by the authors to adjusted p < 0.05.
  human_vs_chimp_divergent  Table S3 (subclass pairwise DE across great apes), rows with group_1 == "human" and
                            group_2 == "chimp": DESeq2 Wald test, pseudobulk, per subclass, adjusted p < 0.05 (pre-filtered).
Direction from the sign of log2FoldChange (positive = higher in human; checked against the paper's in-situ
examples COL11A1 in Chandelier and DACH1 in L5/6 IT Car3, both positive).
"""
import math
import re
import sys
from pathlib import Path

import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type", "direction",
        "effect", "evidence", "circular", "note"]


def main(data_dir, out_tsv):
    d = Path(data_dir)
    s7 = pd.read_csv(d / "extracted/part1/Table_S7_consensus_type_species_degs_human_chimp_gorilla.csv",
                     encoding="utf-8-sig")
    s3 = pd.read_csv(d / "NIHMS1941387-supplement-Table_S3.csv", encoding="utf-8-sig")

    # Subclass names as written in Table S3; S7 cluster ids are the same names with non-alphanumerics
    # removed plus a single trailing digit (consensus type index), e.g. "L56ITCar31" -> "L5/6 IT Car3", 1.
    norm = {re.sub(r"[^A-Za-z0-9]", "", c): c for c in s3["cluster"].unique()}

    h = s7[s7["species"] == "human"].copy()
    m = h["cluster"].str.extract(r"^(?P<stem>.*?)(?P<idx>\d)$")
    h["subclass"] = m["stem"].map(norm)
    if h["subclass"].isna().any():
        raise SystemExit(f"unmapped S7 clusters: {sorted(h.loc[h.subclass.isna(), 'cluster'].unique())}")
    h["consensus_type"] = h["subclass"] + "_" + m["idx"]
    fc14 = h["log2FoldChange"].abs() > math.log2(1.4)  # Fig. 5C counts hDEGs with FC > 1.4 (flag only)
    a = pd.DataFrame({
        "source_id": "jorstad2023",
        "set": "human_specific",
        "gene_symbol": h["gene"],
        "symbol_species": "human",
        "contrast": "human_vs_chimpanzee_gorilla_macaque_marmoset",
        "cell_type": h["consensus_type"],
        "direction": h["log2FoldChange"].map(lambda x: "up_in_human" if x > 0 else "down_in_human"),
        "effect": h["log2FoldChange"].map(lambda x: f"log2FC={x:.4g}"),
        "evidence": "TableS7 DESeq2 Wald pseudobulk, human vs 4 other primates, padj<0.05",
        "circular": "no",
        "note": ("level=consensus_type;subclass=" + h["subclass"] + ";padj=" + h["padj"].map(lambda x: f"{x:.3g}")
                 + ";fig5C_FC>1.4=" + fc14.map({True: "yes", False: "no"})),
    })

    hc = s3[(s3["group_1"] == "human") & (s3["group_2"] == "chimp")]
    b = pd.DataFrame({
        "source_id": "jorstad2023",
        "set": "human_vs_chimp_divergent",
        "gene_symbol": hc["genes"],
        "symbol_species": "human",
        "contrast": "human_vs_chimpanzee",
        "cell_type": hc["cluster"],
        "direction": hc["log2FoldChange"].map(lambda x: "up_in_human" if x > 0 else "down_in_human"),
        "effect": hc["log2FoldChange"].map(lambda x: f"log2FC={x:.4g}"),
        "evidence": "TableS3 DESeq2 Wald pseudobulk, human vs chimpanzee, padj<0.05",
        "circular": "no",
        "note": "level=subclass;padj=" + hc["padj"].map(lambda x: f"{x:.3g}"),
    })
    out = pd.concat([a, b], ignore_index=True)[COLS]
    out.to_csv(out_tsv, sep="\t", index=False)
    print(out.groupby(["set", "cell_type"]).agg(rows=("gene_symbol", "size")).to_string())
    print(out.groupby("set").agg(rows=("gene_symbol", "size"), genes=("gene_symbol", "nunique"),
                                 cell_types=("cell_type", "nunique")).to_string())
    print(out.groupby(["set", "direction"]).size().to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
