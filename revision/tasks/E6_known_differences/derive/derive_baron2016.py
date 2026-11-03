"""Human-vs-mouse islet cell-type statements of Baron et al. 2016 (Cell Systems 3:346), quoted verbatim.

Usage: python -I derive_baron2016.py <data_dir> <out_tsv>
  <data_dir>: data/revision/E6/baron2016 (PMC5228327_fulltext.html)
  <out_tsv>:  revision/tasks/E6_known_differences/lists/baron2016.tsv

Baron 2016 has no species-comparison table (Tables S1-S3 = donors, marker genes, transcription factors; all inside the
supplementary PDF). Rule: a gene enters only if a sentence states a human-vs-mouse difference (text_known_difference)
or conservation (text_conserved) of its expression in named endocrine cell types. Every quote is asserted to occur
verbatim in the downloaded PMC full text.
"""
import html
import re
import sys
from pathlib import Path

import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]
SRC = "baron2016"
D, C = "text_known_difference", "text_conserved"

ITEMS = [
    (["UCN3, initially discovered as a specific marker of beta cell maturation in mice ( Blum et al., 2012 ), is known to "
      "be expressed in both alpha and beta cells in primates ( van der Meulen et al., 2012 ). Our data confirm that Ucn3 "
      "is restricted to beta cells in mice ( Figure 3A , right) and expand the list of human cell types that express "
      "UCN3 to include all human endocrine cells except delta cells ( Figure 3A , left).",
      "The difference in expression pattern between mouse and human suggests an instance of gene expression evolution "
      "across mammalian pancreata."],
     "Results 'Endocrine-Specific Gene Expression', Fig 3A",
     [("UCN3", "alpha", D, "up_in_human", ""), ("UCN3", "gamma", D, "up_in_human", "human: all endocrine except delta"),
      ("UCN3", "epsilon", D, "up_in_human", "human: all endocrine except delta")]),
    (["Our data provide evidence that FFAR4 is expressed in both beta and delta human cells ( Figure 3A , left)",
      "Using our single-cell mouse data, we found an expression pattern that matches neither the existing literature nor "
      "human data: we detected Ffar4 in alpha, beta, gamma, and delta cells ( Figure 3A , right). This highlights a gap "
      "in our knowledge about FFAR4/Ffar4 as well as a key difference between mice and humans"],
     "Results 'Endocrine-Specific Gene Expression', Fig 3A-B",
     [("FFAR4", "alpha", D, "up_in_mouse", ""), ("FFAR4", "gamma", D, "up_in_mouse", "")]),
    (["By contrast, IAPP is believed to be specifically restricted to beta cells ( Johnson et al., 1988 ), which our human "
      "expression data support; however, Iapp is more promiscuously expressed in mouse cells, with detectable "
      "expression in both beta and delta cells."],
     "Results 'Endocrine-Specific Gene Expression', Fig 3A",
     [("IAPP", "delta", D, "up_in_mouse", "")]),
    (["Several studies in mice have suggested a direct effect of leptin on leptin receptor-expressing beta cells ( Morioka "
      "et al., 2007 ). Instead, our human data show that LEPR is a delta cell-specific gene in humans ( Figure 3A , left), "
      "suggesting that leptin action in human pancreatic islets is mediated by delta cells."],
     "Results 'Endocrine-Specific Gene Expression', Fig 3A",
     [("LEPR", "delta", D, "up_in_human", "contrast is to the mouse literature (beta-cell Lepr), not to Baron's mouse data"),
      ("LEPR", "beta", D, "up_in_mouse", "contrast is to the mouse literature (beta-cell Lepr), not to Baron's mouse data")]),
    (["Two have profiles resembling PDX1: ETV1 and MEIS2 are restricted to non-alpha endocrine cells.",
      "However, specific patterns of expression within the endocrine systems, such as ETV1 and MEIS2, which had a "
      "PDX1-like expression profile in the mouse, show “non-beta” expression in the mouse."],
     "Results (transcription factors), Fig 3C",
     [("ETV1", "beta", D, "up_in_human", "AMBIGUOUS wording (sentence says 'in the mouse' twice); read as human "
       "PDX1-like/non-alpha vs mouse non-beta"),
      ("MEIS2", "beta", D, "up_in_human", "AMBIGUOUS wording (sentence says 'in the mouse' twice); read as human "
       "PDX1-like/non-alpha vs mouse non-beta")]),
    (["Pan-endocrine expression was conserved in mice (NEUROD1, INSM1, ISL1, NKX2-2, and PAX6)."],
     "Results (transcription factors), Fig 3C",
     [(g, "pan_endocrine", C, "conserved", "") for g in ["NEUROD1", "INSM1", "ISL1", "NKX2-2", "PAX6"]]),
]


def plain_text(p):
    t = open(p, encoding="utf-8").read()
    t = re.sub(r"<script.*?</script>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return html.unescape(re.sub(r"\s+", " ", t))


def main(data_dir, out):
    full = plain_text(Path(data_dir) / "PMC5228327_fulltext.html")
    rows = []
    for quotes, where, genes in ITEMS:
        for q in quotes:
            assert q in full, q[:90]
        quote = " ... ".join(f'"{q}"' for q in quotes)
        for g, ct, st, dirn, nt in genes:
            rows.append(dict(set=st, gene_symbol=g, symbol_species="human", contrast="human_vs_mouse", cell_type=ct,
                             direction=dirn, effect="text", evidence="text",
                             note=f"{quote} [{where}]" + (f";{nt}" if nt else "")))
    df = pd.DataFrame(rows)
    df.insert(0, "source_id", SRC)
    df["circular"] = "no"
    df = df[COLS]
    assert not df.duplicated(["set", "gene_symbol", "contrast", "cell_type"]).any()
    df.to_csv(out, sep="\t", index=False)
    print("rows", len(df))
    print(df.groupby(["set", "cell_type", "direction"]).gene_symbol.nunique().to_string())
    print("unique genes per set:", df.groupby("set").gene_symbol.nunique().to_dict())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
