"""Derive literature gene lists from Krienen et al. 2020 (Nature 586:262), primate vs mouse cortical interneurons.

Usage: python -I derive_krienen2020.py <converted_dir> <out_tsv>
  <converted_dir>: data/analysis/E6/krienen2020/converted (ST1 = MOESM2 per-species sheets)

The paper publishes NO thresholded cross-species DE table: Supplementary Table 1 holds per-species expression
of the four major interneuron classes and Supplementary Table 5 holds scaled expression of Liger-integrated
marmoset/mouse clusters, without a conserved/divergent label or a cut-off. So the list is text-only: genes the
paper names verbatim as differing (or conserved) between primates and mouse in cortical interneurons
(evidence = text). ST1 is read only to record whether each named gene is present in the human and mouse
sheets (sanity check; no values are used to select or orient genes).
"""
import sys
from pathlib import Path

import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]
SRC = "krienen2020"
FOUR = "major cortical interneuron classes (PVALB, SST, VIP, LAMP5)"

Q_CAT = ("We observed numerous genes with categorical changes in cell-type-specific expression patterns "
         "(including NTNG1, IQGAP2 and TRPC3) and, in rare cases, genes that were a specific marker in one cell class "
         "in some species but not detected at all in the interneurons of the other species (Fig. 1e and Extended Data Fig. 2a).")
Q_NRG1 = ("NRG1 [...] exhibited a pattern of expression levels (across interneuron classes) that was highly correlated "
          "among humans, macaques and marmosets but quite distinct in mice and ferrets (Fig. 1f).")
Q_MARK = ("we found that many markers that are widely used to identify specific subtypes of interneurons in mice "
          "(such as Calb1, Calb2, Htr3a, Reln, Cck and Nos1) had different patterns of expression across the "
          "corresponding subtypes in primates (Extended Data Fig. 6c).")
Q_CALB2 = ("another contributing factor could be the broader expression of CALB2 by additional interneuron types in "
           "primates (Extended Data Fig. 6c).")
Q_ED6 = ("Exemplar genes include those that are widely used as markers for particular interneuron populations but "
         "have divergent gene expression across mice and marmosets, such as CALB1 and CALB2. (Extended Data Fig. 6c legend)")
Q_NKX = ("Notably, marmoset neocortical LAMP5+LHX6+ interneurons selectively expressed NKX2-1 (Extended Data Fig. 8d), "
         "which - in the mouse - is downregulated in MGE-derived interneurons destined for the neocortex but persists "
         "for unknown reasons in some human cortical interneurons.")

# (set, gene, symbol_species, contrast, cell_type, direction, quote, note)
TEXT = [("primate_vs_mouse_divergent", g, "human", "primates(human,macaque,marmoset)_vs_mouse(+ferret)", FOUR,
         "divergent", Q_CAT, "Fig. 1e categorical change in cell-type-specific pattern") for g in ("NTNG1", "IQGAP2", "TRPC3")]
TEXT += [("primate_vs_mouse_divergent", "NRG1", "human", "primates(human,macaque,marmoset)_vs_mouse(+ferret)", FOUR,
          "divergent", Q_NRG1, "pattern across the four classes correlated among primates, distinct in mouse")]
TEXT += [("primate_vs_mouse_divergent", g, "mouse", "marmoset_vs_mouse (Liger-integrated subtypes)",
          "interneuron subtypes (Liger marmoset-mouse clusters)", "divergent", Q_MARK,
          "mouse symbol as published" + ("; also named in ED Fig. 6c legend (CALB1, CALB2)" if g in ("Calb1", "Calb2") else ""))
         for g in ("Calb1", "Calb2", "Htr3a", "Reln", "Cck", "Nos1")]
TEXT += [("primate_vs_mouse_divergent", "CALB2", "human", "primates_vs_rodents (marmoset vs mouse in ED Fig. 6c)",
          "additional interneuron types (broader expression)", "up_in_primate", Q_CALB2,
          "direction = broader expression across types in primates; duplicate of Calb2 row with a direction")]
TEXT += [("primate_vs_mouse_divergent", "NKX2-1", "human", "marmoset_vs_mouse (neocortex)",
          "LAMP5+LHX6+ neocortical interneurons", "up_in_primate", Q_NKX,
          "mouse side is prior knowledge cited by the paper (refs 37, 38), not a measured mouse contrast in this paper")]


def main(conv, out):
    conv = Path(conv)
    present = {}
    for sp in ("human", "mouse", "marmoset"):
        st1 = pd.read_csv(next(conv.glob(f"41586_2020_2781_MOESM2_ESM__{sp}.tsv")), sep="\t")
        present[sp] = set(st1.Genes.str.upper())
    rows = []
    for s, g, symsp, contrast, ct, d, q, note in TEXT:
        pres = ",".join(f"{sp}={'yes' if g.upper() in present[sp] else 'no'}" for sp in present)
        rows.append(dict(set=s, gene_symbol=g, symbol_species=symsp, contrast=contrast, cell_type=ct, direction=d,
                         effect="", evidence="text", note=f"{note}; in_SupplTable1: {pres}; quote: {q}"))
    df = pd.DataFrame(rows)
    df["source_id"], df["circular"] = SRC, "no"
    df = df[COLS]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(df[["set", "gene_symbol", "direction", "cell_type"]].to_string(index=False))
    print("unique genes (case-insensitive):", df.gene_symbol.str.upper().nunique(), "rows:", len(df))
    for r in rows:
        print(r["gene_symbol"], r["note"].split("; quote")[0].split("in_SupplTable1: ")[1])


if __name__ == "__main__":
    main(*sys.argv[1:3])
