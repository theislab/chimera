"""Text-derived human-vs-mouse kidney cell-type expression differences (verbatim sentences).

Usage: python -I derive_kidney_text.py <text_dir> <out_tsv>
  <text_dir> = data/analysis/E6/kidney_text (archived full texts). Every quote below is checked to occur
  verbatim (after tag stripping and whitespace normalisation) in the archived file; the script fails otherwise.
Rule: every gene named in each cited sentence gets a row; no selection within a sentence.
"""
import html
import re
import sys
from pathlib import Path
import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]
HE = ("he2021_PMC8035407_fullText.xml", "He B et al. 2021 Nat Commun 12:2141, doi:10.1038/s41467-021-22331-9 (adult human living-donor biopsies vs C57BL/6J mouse; Smart-seq2)")
KIM = ("kim2023_biorxiv_v1_full.html", "Kim S et al. 2024 Dev Cell 59:2912 (bioRxiv 10.1101/2023.05.16.540880 v1 text), doi:10.1016/j.devcel.2024.07.013 (developing kidney)")
LIN = ("lindstrom2018_PMC5827606.html", "Lindstrom NO et al. 2018 JASN 29:785, doi:10.1681/ASN.2017080887 (human fetal vs mouse developing kidney)")

Q = {
    "he_pdgfrb": (HE, "The majority of human NPHS1 + podocytes expressed PDGFRB , but only a few of mouse podocytes expressed Pdgfrb .", "Results, glomerular cell clustering"),
    "he_nr1h3": (HE, "Interestingly, NR1H3 and Nr2f2 identified in human and mouse MLCs, respectively", "Results (SCENIC regulons, Fig. 1d)"),
    "he_des": (HE, "we chose a mouse-specific gene Des commonly used as a MC marker and a human-specific gene COL6A1 , encoding a poorly characterized glomerular extracellular matrix member for experimental validation.", "Results, Mesangial cells (Fig. 4d,e)"),
    "he_pla2r1": (HE, "identified a number of genes uniquely expressed in the human vs mouse, including for instance PLA2R1, a human podocyte-specific expressed gene that is absent in the mouse.", "Introduction"),
    "he_gec": (HE, "confirmed five of the genes ( IL13RA1 , RXFP1 , GIMAP7 , PLA1A and APLNR ) to be human specific", "Results, GECs (Fig. 6d)"),
    "he_nfasc": (HE, "Evolutionary Nfasc expression in the glomerulus was conserved in primates (human and cynomolgus monkey) but not detected in minipigs or rodents (rats and mouse) as shown by RT-PCR analysis of isolated renal fractions", "Results, podocytes (Fig. 5f,g)"),
    "kim_unc5b": (KIM, "UNC5B was expressed in human NPCs whereas its mouse counterpart Unc5b was absent from NPCs but expressed in multiple differentiating clusters including podocytes", "Results, Figure S4F"),
    "kim_eya4": (KIM, "Conversely, in the nephron lineage, Eya4 and Fgf1 were expressed in mouse-NPCs but expression of their human orthologs was confined to podocytes (Figure S4F).", "Results, Figure S4F"),
    "kim_ish": (KIM, "Secondary validations using in situ hybridization confirmed human-specific mRNA of NRXN1 , ZMAT4 (NPC), NTNG1 , and PLA2R1 (podocytes in mature glomeruli)", "Results, Figure 4D-F, S4D"),
    "kim_tf": (KIM, "Notably, forty four TFs with known-mouse orthologs (e.g., NR3C2 , NPAS2 ; Figure S4E) were identified in this comparative analysis.", "Results, Figure S4E"),
    "lin_loh": (LIN, "Although the locations of CUBN, AQP2, and LRP2 were as expected from mouse studies, UMOD and SLC12A1 showed a reduction in their overlapping domain in the human kidney versus mouse nephron", "Results, Figure 10H, Supplemental Figure 5C"),
}
# (quote key, gene, cell_type, direction, extra note)
ROWS = [
    ("he_pdgfrb", "PDGFRB", "podocyte", "up_in_human", ""),
    ("he_nr1h3", "NR1H3", "mesangial-like cell (MLC)", "up_in_human", "SCENIC regulon activity, not expression per se"),
    ("he_nr1h3", "NR2F2", "mesangial-like cell (MLC)", "up_in_mouse", "SCENIC regulon activity, not expression per se"),
    ("he_des", "DES", "mesangial cell", "up_in_mouse", "protein validated by IF (Fig. 4d)"),
    ("he_des", "COL6A1", "mesangial cell", "up_in_human", "validated by IF (Fig. 4e)"),
    ("he_pla2r1", "PLA2R1", "podocyte", "up_in_human", ""),
    ("he_gec", "IL13RA1", "glomerular endothelial cell", "up_in_human", ""),
    ("he_gec", "RXFP1", "glomerular endothelial cell", "up_in_human", ""),
    ("he_gec", "GIMAP7", "glomerular endothelial cell", "up_in_human", "GIMAP family: check 1:1 orthology"),
    ("he_gec", "PLA1A", "glomerular endothelial cell", "up_in_human", ""),
    ("he_gec", "APLNR", "glomerular endothelial cell", "up_in_human", ""),
    ("he_nfasc", "NFASC", "podocyte", "up_in_human", "RT-PCR on glomerular fractions; Western blot"),
    ("kim_unc5b", "UNC5B", "nephron progenitor cell (NPC)", "up_in_human", "developmental; mouse Unc5b in differentiating clusters incl. podocytes"),
    ("kim_eya4", "EYA4", "nephron progenitor cell (NPC)", "up_in_mouse", "developmental; human ortholog confined to podocytes"),
    ("kim_eya4", "FGF1", "nephron progenitor cell (NPC)", "up_in_mouse", "developmental; human ortholog confined to podocytes"),
    ("kim_ish", "NRXN1", "nephron progenitor cell (NPC)", "up_in_human", "developmental; ISH"),
    ("kim_ish", "ZMAT4", "nephron progenitor cell (NPC)", "up_in_human", "developmental; ISH"),
    ("kim_ish", "NTNG1", "podocyte", "up_in_human", "developmental; cell type by sentence grouping (Fig. 4E arrowhead marks S-shaped body)"),
    ("kim_ish", "PLA2R1", "podocyte", "up_in_human", "developmental; ISH, mature glomeruli"),
    ("kim_tf", "NR3C2", "not stated (human-enriched TF)", "up_in_human", "developmental; example of 44 human-enriched TFs"),
    ("kim_tf", "NPAS2", "not stated (human-enriched TF)", "up_in_human", "developmental; example of 44 human-enriched TFs"),
    ("lin_loh", "CUBN", "proximal tubule", "conserved", "developmental; location as expected from mouse"),
    ("lin_loh", "LRP2", "proximal tubule", "conserved", "developmental; location as expected from mouse"),
    ("lin_loh", "AQP2", "collecting duct", "conserved", "developmental; location as expected from mouse"),
    ("lin_loh", "UMOD", "loop of Henle", "divergent", "developmental; reduced UMOD/SLC12A1 overlapping domain in human"),
    ("lin_loh", "SLC12A1", "loop of Henle", "divergent", "developmental; reduced UMOD/SLC12A1 overlapping domain in human"),
]


def clean(path):
    t = Path(path).read_text()
    t = re.sub(r"<script.*?</script>", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", html.unescape(t))


def main(text_dir, out):
    texts = {}
    for k, ((fn, cit), quote, loc) in Q.items():
        texts.setdefault(fn, clean(Path(text_dir) / fn))
        assert quote in texts[fn], (k, "quote not found verbatim")
    rows = []
    for k, g, ct, di, extra in ROWS:
        (fn, cit), quote, loc = Q[k]
        st = "text_conserved" if di == "conserved" else "text_known_difference"
        rows.append(dict(source_id="kidney_text", set=st, gene_symbol=g, symbol_species="human",
                         contrast="human_vs_mouse", cell_type=ct, direction=di, effect="",
                         evidence="verbatim text statement", circular="no",
                         note=f'"{quote}" [{cit}; {loc}; archived {fn}]' + (f"; {extra}" if extra else "")))
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(out, sep="\t", index=False)
    print(df.groupby(["set", "direction"]).gene_symbol.nunique().to_string())
    print("unique genes:", df.gene_symbol.nunique())


if __name__ == "__main__":
    main(*sys.argv[1:])
