"""Textbook human-vs-mouse islet cell-type expression differences, quoted verbatim from paper text.

Usage: python -I derive_islet_text.py <e6_data_root> <out_tsv>
  <e6_data_root>: data/analysis/E6 (reads islet_text/* and benner2014/PMC4124169_fullTextXML.xml)
  <out_tsv>:      analysis/tasks/E6_known_differences/lists/islet_text.tsv

Rule: a gene enters only if a sentence of the cited paper states a human-vs-mouse difference in a named islet cell
type (set text_known_difference), or states high expression in both species' beta cells (set text_conserved; only
Rorsman & Ashcroft 2018 and Dai 2012 make such statements). Every quote is asserted to occur verbatim in the
downloaded full text (tags stripped, whitespace collapsed). Where a paper names a protein/channel and not the gene,
the gene symbol is the standard one for that protein and the published name is kept in `note`.
Baron et al. 2016 statements are in a separate list (derive_baron2016.py).
"""
import html
import re
import sys
from pathlib import Path

import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]
SRC = "islet_text"
FILES = {
    "Benner2014": "benner2014/PMC4124169_fullTextXML.xml",
    "Dai2012": "islet_text/PMC3268985_dai2012_fullTextXML.xml",
    "SkelinKlemen2017": "islet_text/PMC5710702_skelinklemen2017_fullTextXML.xml",
    "RorsmanAshcroft2018": "islet_text/PMC5866358_rorsman2018_fulltext.html",
}
DOI = {"Benner2014": "10.1186/1471-2164-15-620", "Dai2012": "10.1007/s00125-011-2369-0",
       "SkelinKlemen2017": "10.1080/19382014.2017.1342022", "RorsmanAshcroft2018": "10.1152/physrev.00008.2017"}

D, C = "text_known_difference", "text_conserved"
UH, UM, CO = "up_in_human", "up_in_mouse", "conserved"

# (paper, [quotes], where, [(gene, symbol_species, cell_type, set, direction, note)])
ITEMS = [
    ("Benner2014",
     ["Mafa is restricted to beta cells of both species, while Mafb is selectively expressed in mouse alpha cells (A) , "
      "but expressed in human beta cells as well (B) .",
      "In contrast, while Mafa expression is similarly restricted to human beta cells, Mafb expression is retained in "
      "human beta cells and therefore not enriched in either human islet population (Figure 4 B) [ 36 , 37 ]."],
     "Results 'Differences between mouse and human islet cells', Fig 4 legend",
     [("Mafb", "mouse", "beta", D, UH, ""), ("Mafb", "mouse", "alpha", D, UM, "mouse alpha-selective; human not enriched in alpha"),
      ("Mafa", "mouse", "beta", C, CO, "restricted to beta cells of both species")]),
    ("Benner2014",
     ["As a second example of species differences between mouse and human islets, Ucn3 is highly selectively expressed "
      "in mouse beta cells [ 17 , 18 ] (Figure 2 E; Figure 4 A, C; Additional file 7 ) but is a common feature of "
      "human alpha and beta cells [ 17 ]."],
     "Results 'Differences between mouse and human islet cells', Fig 4",
     [("Ucn3", "mouse", "alpha", D, UH, "")]),
    ("Benner2014",
     ["is not detectable in mouse islets (Figure 4 A, D) but is highly enriched in human alpha cells (Figure 4 B, F; "
      "Additional file 7 )"],
     "Results 'Differences between mouse and human islet cells', Fig 4; subject of the sentence: Crh",
     [("Crh", "mouse", "alpha", D, UH, "")]),
    ("Benner2014",
     ["In sharp contrast, IL1R1 expression in human islet populations, while detectable, is significantly lower in "
      "human beta cells in comparison to both mouse beta cells (Figure 5 A, B) and human alpha cells (Figure 7 A, B)."],
     "Results (lncRNA / Il1r1 section), Fig 5, Fig 7",
     [("IL1R1", "human", "beta", D, UM, "also in Benner AddFile9 (mouse enriched)")]),
    ("Dai2012",
     ["Since MAFB is exclusively produced in adult mouse alpha cells [ 26 ]",
      "In contrast, MAFB was produced in human alpha cells (Fig. 6i–l ) and also in a subset of beta cells (Fig. 6e–h ).",
      "Nevertheless, the distinctive features of human islets (such as MAFB production in islet alpha and beta cells) "
      "suggest important biological differences between mouse and human islets."],
     "Results 'MAFB is expressed in human islet beta cells', Fig 6; Discussion",
     [("MAFB", "human", "beta", D, UH, "protein (immunostaining)")]),
    ("Dai2012",
     ["The cellular distribution of MAFA and PDX1 in human and mouse islets was similar, with MAFA only produced in "
      "beta cells (Fig. 6a–d ), and PDX1 in beta cells (Fig. 6m–p ) and in a fraction of delta cells (Fig. 6q–t )."],
     "Results, Fig 6",
     [("MAFA", "human", "beta", C, CO, "protein"), ("PDX1", "human", "beta", C, CO, "protein")]),
    ("SkelinKlemen2017",
     ["In mice, GLUT3 is expressed only at a background level, whereas it is as abundant as GLUT1 in human β cells"],
     "Section 'Glucose transport', Table 1",
     [("SLC2A3", "human", "beta", D, UH, "published as GLUT3")]),
    ("SkelinKlemen2017",
     ["Most of the glucose transporters expressed in mouse β cells are GLUT2",
      "In human β cells on the other hand, GLUT2 are almost completely dispensable, since GLUT1 and GLUT3 are far more "
      "abundantly expressed."],
     "Section 'Glucose transport', Table 1",
     [("SLC2A2", "human", "beta", D, UM, "published as GLUT2"), ("SLC2A1", "human", "beta", D, UH, "published as GLUT1")]),
    ("SkelinKlemen2017",
     ["Human β cells, on the other hand, do not express R-type VDCCs but T-type VDCCs (Ca v 3.2)"],
     "Section 'Voltage-dependent calcium channels', Table 1",
     [("CACNA1E", "human", "beta", D, UM, "published as R-type VDCC (Cav2.3)"),
      ("CACNA1H", "human", "beta", D, UH, "published as T-type VDCC (Cav3.2)")]),
    ("SkelinKlemen2017",
     ["In contrast to mice, human β cells possess both Ca v 1.2 and Ca v 1.3 L-type VDCCs."],
     "Section 'Voltage-dependent calcium channels'",
     [("CACNA1D", "human", "beta", D, UH, "published as Cav1.3; CONFLICT: Rorsman & Ashcroft 2018 give Cacna1d as 40% "
       "of mouse beta-cell Ca channel transcripts")]),
    ("RorsmanAshcroft2018",
     ["Whereas mouse β-cells mainly express Scn9a (Nav1.7), SCN8A (Nav1.6) accounts for 40% of Na + channel "
      "transcripts in human β-cells ( 68 , 475 ) with the remainder being SCN9A (Nav1.7), SCN3A (Nav1.3), and SCN2A "
      "(Nav1.2) (see Supplemental Figure 4 )."],
     "Section V (Na+ channels), Suppl Fig 4",
     [("SCN8A", "human", "beta", D, UH, ""), ("Scn9a", "mouse", "beta", D, UM, "share of Na channel transcripts; human also expresses SCN9A")]),
    ("RorsmanAshcroft2018",
     ["In contrast to mouse β-cells, the delayed rectifying K + current in human β-cells is carried principally by "
      "Kv2.2 ( KCNB2 ) and Kv1.6 ( KCNA6 ) channels (see Supplemental Figure 5 A ), which are expressed at high levels "
      "in the human β-cell"],
     "Section V (K+ channels), Suppl Fig 5A",
     [("KCNB2", "human", "beta", D, UH, ""), ("KCNA6", "human", "beta", D, UH, "")]),
    ("RorsmanAshcroft2018",
     ["KCNMA1 is expressed at fivefold higher levels in human than in mouse β-cells"],
     "Section V (BK channels)", [("KCNMA1", "human", "beta", D, UH, "5-fold")]),
    ("RorsmanAshcroft2018",
     ["In addition, human (but not mouse) β-cells express the ionotropic nicotinic ACh receptor α-subunits CHRNA3 and "
      "CHRNA5 (see Supplemental Figure 12 A ) and high levels of the β-subunit CHRNB2 (see Supplemental Figure 12 C )."],
     "Section VII (acetylcholine), Suppl Fig 12",
     [("CHRNA3", "human", "beta", D, UH, ""), ("CHRNA5", "human", "beta", D, UH, ""), ("CHRNB2", "human", "beta", D, UH, "")]),
    ("RorsmanAshcroft2018",
     ["Mouse islets express Gad1 (GAD65), whereas GAD2 (GAD67) is found in human β-cells ( 348 )."],
     "Section VII (GABA)",
     [("GAD2", "human", "beta", D, UH, "protein names as printed (GAD65/GAD67 swapped relative to HGNC); also Benner AddFile9 human-unique Gad2"),
      ("Gad1", "mouse", "beta", D, UM, "mouse statement is for islets, not beta cells specifically")]),
    ("RorsmanAshcroft2018",
     ["Human β-cells express low but detectable levels of the GABA A receptor α-subunits (see Supplemental Figure 13 B "
      "; almost exclusively GABRA2 ).",
      "Expression of GABA A receptor α-subunits in mouse β-cells is very low (see Supplemental Figure 13 B )"],
     "Section VII (GABA), Suppl Fig 13B", [("GABRA2", "human", "beta", D, UH, "low in human, very low in mouse")]),
    ("RorsmanAshcroft2018",
     ["Human β-cells only express GABA B receptor 1 ( Gabbr1 / GABBR1 ) (see Supplemental Figure 13 B ). It is also "
      "present in mouse β-cells but the GABA B receptor 2 ( Gabbr2/GABBR2 ) is expressed at 10-fold higher levels"],
     "Section VII (GABA), Suppl Fig 13B", [("GABBR2", "human", "beta", D, UM, "10-fold")]),
    ("RorsmanAshcroft2018",
     ["BEST1 is highly expressed in human β-cells, whereas mouse β-cells express Best3 (see Supplemental Figure 2 A )."],
     "Section V (Cl- channels), Suppl Fig 2A",
     [("BEST1", "human", "beta", D, UH, ""),
      ("Best3", "mouse", "beta", D, UM, "CONFLICT within the review: an earlier sentence says Ano8, Ano10 and Best3 are "
       "'all of which are expressed in mouse and human β-cells'")]),
    ("RorsmanAshcroft2018",
     ["Ano6 is also expressed in mouse β-cells, at relatively at high levels, but it is found at a much lower level in "
      "human β-cells.", "Conversely, ANO5 is expressed in human β-cells but not detected in mouse β-cells."],
     "Section V (Cl- channels), Fig 8A", [("Ano6", "mouse", "beta", D, UM, ""), ("ANO5", "human", "beta", D, UH, "")]),
    ("RorsmanAshcroft2018",
     ["In human β-cells, GRIA2 , GRIA3 , and GRIA4 (but not GRIA1 ) are expressed and (in absolute terms) at levels "
      "~10-fold higher than those found in mouse β-cells (see Supplemental Figure 14 A )."],
     "Section VII (glutamate), Suppl Fig 14A",
     [("GRIA2", "human", "beta", D, UH, "~10-fold"), ("GRIA3", "human", "beta", D, UH, "~10-fold"),
      ("GRIA4", "human", "beta", D, UH, "~10-fold")]),
    ("RorsmanAshcroft2018",
     ["Human β-cells also express SLC1A1 (encoding EAAT3) ( 180 ), but a recent proteome analysis indicates that mouse "
      "islets do not contain significant levels of this transporter ( 767 ), in agreement with gene expression data"],
     "Section VI (amino acids), Suppl Fig 11B", [("SLC1A1", "human", "beta", D, UH, "")]),
    ("RorsmanAshcroft2018",
     ["Human β-cells express L-type ( CACNA1C and CACNA1D ), P/Q-type ( CACNA1A ), and T-type ( CACNA1H ) Ca 2+ "
      "channels (see Supplemental Figure 3 A ). N-type ( CACNA1B) Ca 2+ channels are expressed at lower levels than in "
      "mouse β-cells, and CACNA1E is undetectable.",
      "There is no evidence of low-threshold T-type Ca 2+ currents in mouse β-cells, and T-type transcripts are only "
      "expressed at very low levels"],
     "Section V (Ca2+ channels), Suppl Fig 3A",
     [("CACNA1H", "human", "beta", D, UH, ""), ("CACNA1B", "human", "beta", D, UM, "")]),
    ("RorsmanAshcroft2018",
     ["(amylin) is expressed at levels 1 and 10% of that of insulin in human ( 68 , 475 ) and mouse β-cells, "
      "respectively"],
     "Section on IAPP, Suppl Fig 1A",
     [("IAPP", "human", "beta", D, UM, "relative to insulin; also Benner AddFile9 (mouse enriched)")]),
    ("RorsmanAshcroft2018",
     ["Expression of STX1A is also 30-fold higher in human than in mouse β-cells ( FIGURE 16 B )"],
     "Section VIII (exocytosis), Fig 16B", [("STX1A", "human", "beta", D, UH, "30-fold")]),
    ("RorsmanAshcroft2018",
     ["Mouse (but not human) β-cells also express high levels of Vamp4 ."],
     "Section VIII (exocytosis), Suppl Fig 17B", [("Vamp4", "mouse", "beta", D, UM, "")]),
    ("RorsmanAshcroft2018",
     ["Mouse (but not human) β-cells also express low levels of Syt2 , Syt3 , and Syt9 (see Supplemental Figure 18 A )."],
     "Section VIII (synaptotagmins), Suppl Fig 18A",
     [("Syt2", "mouse", "beta", D, UM, ""), ("Syt3", "mouse", "beta", D, UM, ""), ("Syt9", "mouse", "beta", D, UM, "")]),
    ("RorsmanAshcroft2018",
     ["Of the Ca 2+ -independent synaptotagmins, particularly high levels of synaptotagmin 4, 11, and 13 are found in "
      "human β-cells, whereas synaptotagmins 4 and 13 ( Syt4 and 13 ) predominate in mouse β-cells"],
     "Section VIII (synaptotagmins), Suppl Fig 18B", [("SYT11", "human", "beta", D, UH, "published as synaptotagmin 11")]),
    ("RorsmanAshcroft2018",
     ["In human β-cells, only SYN1 is expressed, whereas mouse β-cells express both Syn1 and Syn2"],
     "Section VIII (synapsins)", [("Syn2", "mouse", "beta", D, UM, "")]),
    ("RorsmanAshcroft2018",
     ["Interestingly, mouse β-cells express Epac2 at ~10-fold higher levels than their human counterparts ( 56 )."],
     "Section VII (cAMP)", [("Rapgef4", "mouse", "beta", D, UM, "published as Epac2; ~10-fold; also Benner AddFile9 (mouse enriched)")]),
    ("RorsmanAshcroft2018",
     ["However, although RPH3A is expressed at detectable levels in human β-cells, its expression in mouse β-cells is "
      "very low (see Supplemental Figure 15 F )."],
     "Section VIII (Rab effectors), Suppl Fig 15F", [("RPH3A", "human", "beta", D, UH, "")]),
    # conserved: sentences stating high expression in both species' beta cells
    ("RorsmanAshcroft2018", ["Like mouse β-cells, human islets express very high levels of KCNH2 (Kv11.1 or HERG1)"],
     "Section V (K+ channels)", [("KCNH2", "human", "beta", C, CO, "human statement is for islets")]),
    ("RorsmanAshcroft2018", ["Mouse and human β-cells both express high levels of cationic amino acid transporters CAT-1 "
                             "( Slc7a1/SLC7A1 ) and CAT-2 ( Slc7a2/SLC7A2 )"],
     "Section VI", [("SLC7A1", "human", "beta", C, CO, ""), ("SLC7A2", "human", "beta", C, CO, "")]),
    ("RorsmanAshcroft2018", ["Slc7a5 , which is highly expressed in both mouse and human β-cells"],
     "Section VI", [("SLC7A5", "human", "beta", C, CO, "")]),
    ("RorsmanAshcroft2018", ["(encoded by FFAR1 ), which is expressed at high levels in both mouse and human β-cells"],
     "Section VII", [("FFAR1", "human", "beta", C, CO, "")]),
    ("RorsmanAshcroft2018", ["GLP-1 acts by activation of the GLP-1 receptor ( Glp1r / GLP1R ), which is highly expressed "
                             "in both mouse and human β-cells"],
     "Section VII", [("GLP1R", "human", "beta", C, CO, "")]),
    ("RorsmanAshcroft2018", ["(GIPR, encoded by Gipr/GIPR ), which is expressed at high levels in both mouse and human "
                             "β-cells"],
     "Section VII", [("GIPR", "human", "beta", C, CO, "")]),
    ("RorsmanAshcroft2018", ["Of the GIRK channels, GIRK2 ( Kcnj6/KCNJ6 ) is expressed at particularly high levels in both "
                             "mouse and human β-cells"],
     "Section VII", [("KCNJ6", "human", "beta", C, CO, "")]),
    ("RorsmanAshcroft2018", ["Noc2 ( Rph3al/RPH3AL ; ‟rabphilin3a-likeˮ) is highly expressed in both human and mouse "
                             "β-cells"],
     "Section VIII", [("RPH3AL", "human", "beta", C, CO, "")]),
    ("RorsmanAshcroft2018", ["Both mouse and human β-cells express high levels of the H + /Cl − cotransporter ClC3 ( "
                             "Clcn3/CLCN3 )."],
     "Section V", [("CLCN3", "human", "beta", C, CO, "")]),
    ("RorsmanAshcroft2018", ["Mouse and human β-cells express Rims/RIMS1 , and Rim2 ( Rims/RIMS2 ) at low and high levels, "
                             "respectively"],
     "Section VIII", [("RIMS2", "human", "beta", C, CO, "high in both; RIMS1 low in both")]),
    ("RorsmanAshcroft2018", ["Both mouse and human β-cells express high levels of the RIM-binding proteins Rimbp2/RIMBP2 "
                             "and Erc1/ERC1"],
     "Section VIII", [("RIMBP2", "human", "beta", C, CO, ""), ("ERC1", "human", "beta", C, CO, "")]),
    ("RorsmanAshcroft2018", ["Both mouse and human β-cells express SNAP-25 at high levels ( Snap25/SNAP25 )"],
     "Section VIII", [("SNAP25", "human", "beta", C, CO, "")]),
    ("RorsmanAshcroft2018", ["Both mouse and human β-cells express high levels of VAMP-2 and 3 ( Vamp2–3/VAMP2–3 )"],
     "Section VIII", [("VAMP2", "human", "beta", C, CO, ""), ("VAMP3", "human", "beta", C, CO, "")]),
    ("RorsmanAshcroft2018", ["Complexin-2 ( Cplx2/CPLX2 ) is expressed at high levels in mouse and human β-cells"],
     "Section VIII", [("CPLX2", "human", "beta", C, CO, "")]),
    ("RorsmanAshcroft2018", ["Both mouse and human β-cells also express Syt5 at high/fairly high levels"],
     "Section VIII", [("SYT5", "human", "beta", C, CO, "")]),
]


def plain_text(p):
    t = open(p, encoding="utf-8").read()
    t = re.sub(r"<script.*?</script>", "", t, flags=re.S)
    t = re.sub(r"<style.*?</style>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return html.unescape(re.sub(r"\s+", " ", t))


def main(root, out):
    root = Path(root)
    texts = {k: plain_text(root / v) for k, v in FILES.items()}
    rows = []
    for paper, quotes, where, genes in ITEMS:
        for q in quotes:
            assert q in texts[paper], (paper, q[:90])
        quote = " ... ".join(f'"{q}"' for q in quotes)
        for g, gsp, ct, st, dirn, nt in genes:
            rows.append(dict(set=st, gene_symbol=g, symbol_species=gsp, contrast="human_vs_mouse", cell_type=ct,
                             direction=dirn, effect="text", evidence="text",
                             note=f"paper={paper} (doi:{DOI[paper]});{quote} [{where}]" + (f";{nt}" if nt else "")))
    df = pd.DataFrame(rows)
    df.insert(0, "source_id", SRC)
    df["circular"] = "no"
    df = df[COLS]
    paper = df.note.str.extract(r"paper=(\w+)")[0]
    key = df.assign(paper=paper, g=df.gene_symbol.str.upper())
    assert not key.duplicated(["set", "g", "contrast", "cell_type", "paper"]).any()
    # the same (gene, type, contrast) may be stated by two papers: one row per paper, directions must agree
    multi = key.groupby(["g", "contrast", "cell_type"]).agg(n=("paper", "size"), dirs=("direction", "nunique"))
    assert (multi.dirs == 1).all(), multi[multi.dirs > 1]
    print("stated by >1 paper:", multi[multi.n > 1].index.get_level_values(0).tolist())
    df.to_csv(out, sep="\t", index=False)
    print("rows", len(df))
    print(df.groupby(["set", "cell_type", "direction"]).gene_symbol.nunique().to_string())
    print("unique genes per set:", df.groupby("set").gene_symbol.apply(lambda s: s.str.upper().nunique()).to_dict())
    print("papers:", df.note.str.extract(r"paper=(\w+)")[0].value_counts().to_dict())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
