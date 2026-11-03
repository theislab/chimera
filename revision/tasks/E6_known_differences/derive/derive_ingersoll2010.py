"""E6: text-derived human-vs-mouse monocyte-subset differences from Ingersoll et al. 2010 Blood 115:e10
("Comparison of gene expression profiles between human and mouse monocyte subsets"), doi:10.1182/blood-2009-07-235028.

Usage: python -I derive_ingersoll2010.py <PMC2810986.html> <out.tsv>

Rule: every gene named in the quoted sentences that state a species difference (no selection). Each quote is
checked to occur verbatim in the saved PMC page (after tag stripping and whitespace normalisation).
Supplementary tables were not retrievable (Europe PMC supplementaryFiles empty; ashpublications.org HTTP 403).
"""
import csv
import html
import re
import sys

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type", "direction",
        "effect", "evidence", "circular", "note"]
SRC = "ingersoll2010"
SUBSETS = "monocyte subsets (human CD16- vs CD16+; mouse Ly-6C+ vs Ly-6Clo)"
NONCL = "nonclassical monocyte (mouse Ly-6Clo vs human CD16+)"

Q = [
    ("Abstract",
     "Despite overall conservation, some molecules were conversely expressed between the 2 species' subsets, "
     "including CD36, CD9, and TREM-1.",
     [("CD36", "CD36", SUBSETS, "divergent"), ("CD9", "CD9", SUBSETS, "divergent"),
      ("TREM1", "TREM-1", SUBSETS, "divergent")]),
    ("Abstract",
     "Other differences included a prominent peroxisome proliferator-activated receptor γ (PPARγ) signature in "
     "mouse monocytes, which is absent in humans,",
     [("PPARG", "PPARγ", "monocyte", "up_in_mouse")]),
    ("Results (Figures 5, 6A)",
     "In particular, the most differentially expressed genes between mouse Ly-6C + and Ly-6C lo mouse monocytes "
     "include fatty acid binding protein 4 (Fabp-4) and its transcriptional regulator peroxisome "
     "proliferator-activated receptor γ (PPARγ; Figures 5 and 6 A), along with other genes previously described in "
     "human or mouse cells to be modulated by PPARγ agonists ( Figure 6 A), including CD36 36 ( Figures 5 and 6 A), "
     "Hsd11β1, 37 and G0S2 38 ( Figures 5 and 6 A). By contrast, no PPARγ signature was evident in the human "
     "monocyte array",
     [("FABP4", "Fabp-4", NONCL, "up_in_mouse"), ("PPARG", "PPARγ", NONCL, "up_in_mouse"),
      ("CD36", "CD36", NONCL, "up_in_mouse"), ("HSD11B1", "Hsd11β1", NONCL, "up_in_mouse"),
      ("G0S2", "G0S2", NONCL, "up_in_mouse")]),
    ("Results",
     "Furthermore, macrophage scavenger receptor 1 (Msr-1; scavenger receptor A, CD204) was more highly expressed "
     "on the surface of mouse Ly-6C lo monocytes ( Figure 6 C), though the pattern of mRNA in the mouse subsets was "
     "opposite ( Figure 6 B), and the expression of MSR-1 between human monocyte subsets is also converse to mouse "
     "subset expression ( Figure 3 ).",
     [("MSR1", "Msr-1/MSR-1", SUBSETS, "divergent")]),
    ("Results",
     "In humans, these receptors are either more highly expressed on CD16 − monocytes (SIRPα, MSR-1, CD36, and "
     "thrombospondin 1/Thbs1; Figures 2 – 5 ) or not differentially expressed.",
     [("SIRPA", "SIRPα", SUBSETS, "divergent"), ("MSR1", "MSR-1", SUBSETS, "divergent"),
      ("CD36", "CD36", SUBSETS, "divergent"), ("THBS1", "thrombospondin 1/Thbs1", SUBSETS, "divergent")]),
    ("Discussion",
     "Several receptors, including CD36, MSR-1, SIRPα, and thrombospondin 1, appear enriched on human CD16 − "
     "monocytes by surface protein and/or mRNA analysis ( Figures 2 – 5 ), whereas all of these molecules and others "
     "involved in recognition of dying cells (Treml4 31 ) are enriched on Ly-6C lo mouse monocytes.",
     [("CD36", "CD36", SUBSETS, "divergent"), ("MSR1", "MSR-1", SUBSETS, "divergent"),
      ("SIRPA", "SIRPα", SUBSETS, "divergent"), ("THBS1", "thrombospondin 1", SUBSETS, "divergent"),
      ("TREML4", "Treml4", NONCL, "up_in_mouse")]),
    ("Discussion",
     "However, in humans, CD64 surface protein was absent on CD16 + monocytes, whereas in the mice, Ly-6C lo "
     "monocytes were still positive with only a slightly reduced expression of plasma membrane CD64 relative to "
     "their Ly-6C + counterparts.",
     [("FCGR1A", "CD64", NONCL, "up_in_mouse")]),
]


def main(page, out):
    t = open(page, encoding="utf8", errors="ignore").read()
    s = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))
    rows = []
    for where, quote, genes in Q:
        if quote not in s:
            sys.exit("quote not found verbatim: %s" % quote[:80])
        for gene, pub, ct, direction in genes:
            extra = ""
            if gene == "FCGR1A":
                extra = "; protein-level statement; human FCGR1A has paralogs (FCGR1B), mouse Fcgr1 (not strictly 1:1)"
            if gene == "TREML4":
                extra = "; TREML4 1:1 status uncertain"
            rows.append([SRC, "text_known_difference", gene, "human", "human_vs_mouse", ct, direction,
                         "qualitative (microarray/flow; subset fold changes not tabulated here)",
                         "Ingersoll 2010 text, %s" % where, "no",
                         'published name: %s; quote (%s): "%s"%s' % (pub, where, quote, extra)])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLS)
        w.writerows(rows)
    print("rows", len(rows), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
