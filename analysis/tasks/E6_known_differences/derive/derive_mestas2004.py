"""E6: derive immune human-vs-mouse differences from Mestas & Hughes 2004 J Immunol 172:2731, Table I
("Summary of some known immunological differences between mouse and human"), doi:10.4049/jimmunol.172.5.2731.

Usage: python -I derive_mestas2004.py <table1_transcribed_webfetch.tsv> <out.tsv>

Rule (mechanical): every Table I row whose feature names a gene product (gene, receptor, chemokine, CD antigen)
and states its presence/expression in mouse vs human gives one output row per (gene, cell type named in the row).
  kind [expression]   = the row states expression of the gene in a cell type that differs between species
  kind [gene_content] = the row states the gene/family is absent from one species ("Absent", "Pseudogene") without
                        a cell type; usually no 1:1 ortholog, kept and flagged
Rows about deficiency phenotypes, function/affinity, regulation/induction, Ig isotype structure, cell
composition, defensin families or sialic acid (no gene named) are excluded (listed in mestas2004.md).
Direction: present/higher in human only -> up_in_human; mouse only -> up_in_mouse; both -> conserved;
pattern change without a side (co-expression vs exclusive) -> divergent.
The verbatim mouse and human Table I cells are copied from the transcription into `note`.
"""
import csv
import sys

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type", "direction",
        "effect", "evidence", "circular", "note"]
SRC = "mestas2004"
E, G = "expression", "gene_content"

# feature (as in Table I) -> list of (gene_symbol, symbol_species, published_name, cell_type, direction, kind, extra)
MAP = {
    "Hemotopoietic stem cells": [
        ("KIT", "human", "c-kit", "hematopoietic stem cell", "up_in_mouse", E, ""),
        ("FLT3", "human", "flt-3", "hematopoietic stem cell", "up_in_human", E, "")],
    "TLR2 expression on PBL": [
        ("TLR2", "human", "TLR2", "peripheral blood leukocytes", "up_in_human", E, "human constitutive vs mouse low"),
        ("TLR2", "human", "TLR2", "T cell", "up_in_mouse", E, "mouse: induced on T cells; human: not on T cells")],
    "TLR3": [
        ("TLR3", "human", "TLR3", "macrophage", "up_in_mouse", E, ""),
        ("TLR3", "human", "TLR3", "dendritic cell", "conserved", E, "")],
    "TLR9": [
        ("TLR9", "human", "TLR9", "myeloid cells (all; incl. monocytes/macrophages)", "up_in_mouse", E, ""),
        ("TLR9", "human", "TLR9", "B cell", "conserved", E, ""),
        ("TLR9", "human", "TLR9", "plasmacytoid dendritic cell", "conserved", E, ""),
        ("TLR9", "human", "TLR9", "N (as printed; abbreviation not expanded in table, probably neutrophils)",
         "conserved", E, "if N = neutrophils, mouse 'all myeloid cells' includes them")],
    "TLR10": [
        ("TLR10", "human", "TLR10", "not stated (widely expressed)", "up_in_human", G, "mouse Tlr10 is a pseudogene; no 1:1 ortholog")],
    "CD33": [
        ("CD33", "human", "CD33", "granulocyte", "up_in_mouse", E, "mouse Cd33/human CD33 (Siglec-3) orthology is imperfect"),
        ("CD33", "human", "CD33", "monocyte", "up_in_human", E, "mouse Cd33/human CD33 (Siglec-3) orthology is imperfect")],
    "CD4 on macrophages": [
        ("CD4", "human", "CD4", "macrophage", "up_in_human", E, "")],
    "CD1 genes": [
        ("CD1A", "human", "CD1a", "not stated", "up_in_human", G, "no mouse ortholog"),
        ("CD1B", "human", "CD1b", "not stated", "up_in_human", G, "no mouse ortholog"),
        ("CD1C", "human", "CD1c", "not stated", "up_in_human", G, "no mouse ortholog"),
        ("CD1D", "human", "CD1d", "not stated", "conserved", G, "mouse Cd1d1/Cd1d2 (not 1:1)")],
    "NK inhibitory Rs for MHC 1": [
        ("Ly49 family", "mouse", "Ly49 family (except Ly49D and H)", "NK cell", "up_in_mouse", G, "Klra family; no 1:1 human ortholog"),
        ("KIR", "human", "KIR", "NK cell", "up_in_human", G, "KIR family; no 1:1 mouse ortholog")],
    "NKG2D ligands": [
        ("H60", "mouse", "H-60", "not stated", "up_in_mouse", G, "no 1:1 human ortholog"),
        ("Raet1b", "mouse", "Rae1β", "not stated", "up_in_mouse", G, "no 1:1 human ortholog"),
        ("MICA", "human", "MIC A", "not stated", "up_in_human", G, "no mouse ortholog"),
        ("MICB", "human", "MIC B", "not stated", "up_in_human", G, "no mouse ortholog"),
        ("ULBP", "human", "ULBP", "not stated", "up_in_human", G, "ULBP/RAET1 family; no 1:1 ortholog")],
    "FcαRI": [
        ("FCAR", "human", "FcαRI", "not stated", "up_in_human", G, "no mouse ortholog")],
    "FcγRIIA, C": [
        ("FCGR2A", "human", "FcγRIIA", "not stated", "up_in_human", G, "mouse has only Fcgr2b; not 1:1"),
        ("FCGR2C", "human", "FcγRIIC", "not stated", "up_in_human", G, "mouse has only Fcgr2b; not 1:1")],
    "CD38 expression on B cells": [
        ("CD38", "human", "CD38", "germinal centre B cell", "up_in_human", E, "mouse low vs human high"),
        ("CD38", "human", "CD38", "plasma cell", "up_in_human", E, "mouse off vs human high")],
    "B cell CD5 and CD23 expression": [
        ("CD5", "human", "CD5", "B cell", "divergent", E, "mouse mutually exclusive with CD23; human co-expressed"),
        ("FCER2", "human", "CD23", "B cell", "divergent", E, "mouse mutually exclusive with CD5; human co-expressed")],
    "Thy 1 expression": [
        ("THY1", "human", "Thy 1", "T cell (peripheral)", "up_in_mouse", E, ""),
        ("THY1", "human", "Thy 1", "thymocyte", "up_in_mouse", E, "")],
    "Caspase 10": [
        ("CASP10", "human", "Caspase 10", "not stated", "up_in_human", G, "no mouse ortholog")],
    "Th expression of IL-10": [
        ("IL10", "human", "IL-10", "Th1 cell", "up_in_human", E, ""),
        ("IL10", "human", "IL-10", "Th2 cell", "conserved", E, "")],
    "IL-4 and IFN-γ expression by cultured Th": [
        ("IL4", "human", "IL-4", "T helper cell (cultured)", "divergent", E, "co-expression with IFN-γ: mouse either/or, human sometimes both"),
        ("IFNG", "human", "IFN-γ", "T helper cell (cultured)", "divergent", E, "co-expression with IL-4: mouse either/or, human sometimes both")],
    "CD28 expression on T cells": [
        ("CD28", "human", "CD28", "CD4+ T cell", "up_in_mouse", E, "100% vs 80% of cells"),
        ("CD28", "human", "CD28", "CD8+ T cell", "up_in_mouse", E, "100% vs 50% of cells")],
    "ICAM3": [
        ("ICAM3", "human", "ICAM3", "not stated", "up_in_human", G, "no mouse ortholog")],
    "GlyCAM": [
        ("GLYCAM1", "human", "GlyCAM", "not stated", "up_in_mouse", G, "human GLYCAM1 is a pseudogene")],
    "MHC II expression on T cells": [
        ("MHC II", "human", "MHC II", "T cell", "up_in_human", E, "class II genes (HLA-DR/DP/DQ vs H2-A/E) are not 1:1 except e.g. CD74/CIITA")],
    "Kv1.3 K+ channel on T cells": [
        ("KCNA3", "human", "Kv1.3", "T cell", "up_in_human", E, "")],
    "MUC1 on T cells": [
        ("MUC1", "human", "MUC1", "T cell", "up_in_human", E, "")],
    "Granulysin": [
        ("GNLY", "human", "Granulysin", "not stated", "up_in_human", G, "no mouse ortholog")],
    "CXCR1": [
        ("CXCR1", "human", "CXCR1", "not stated", "up_in_human", G, "a mouse Cxcr1 ortholog was later annotated; table states absent")],
    "IL-8, NAP-2, ITAC, MCP-4, HCC-1, HCC-2, MPIF-1, PARC, eotaxin-2/3": [
        ("CXCL8", "human", "IL-8", "not stated", "up_in_human", G, "no mouse ortholog"),
        ("PPBP", "human", "NAP-2", "not stated", "up_in_human", G, "mouse Ppbp exists; table states absent"),
        ("CXCL11", "human", "ITAC", "not stated", "up_in_human", G, "mouse Cxcl11 exists (nonfunctional in C57BL/6)"),
        ("CCL13", "human", "MCP-4", "not stated", "up_in_human", G, "no mouse ortholog"),
        ("CCL14", "human", "HCC-1", "not stated", "up_in_human", G, "no mouse ortholog"),
        ("CCL15", "human", "HCC-2", "not stated", "up_in_human", G, "no mouse ortholog"),
        ("CCL23", "human", "MPIF-1", "not stated", "up_in_human", G, "no mouse ortholog"),
        ("CCL18", "human", "PARC", "not stated", "up_in_human", G, "no mouse ortholog"),
        ("CCL24", "human", "eotaxin-2", "not stated", "up_in_human", G, "mouse Ccl24 exists; table states absent"),
        ("CCL26", "human", "eotaxin-3", "not stated", "up_in_human", G, "mouse Ccl26 exists; table states absent")],
    "MRP-1/2, lungkine, MCP-5": [
        ("Ccl6", "mouse", "MRP-1", "not stated", "up_in_mouse", G, "no human ortholog"),
        ("Ccl9", "mouse", "MRP-2", "not stated", "up_in_mouse", G, "no human ortholog"),
        ("Cxcl15", "mouse", "lungkine", "not stated", "up_in_mouse", G, "no human ortholog"),
        ("Ccl12", "mouse", "MCP-5", "not stated", "up_in_mouse", G, "no human ortholog")],
    "Constitutive MHC II on EC": [
        ("MHC II", "human", "MHC II", "endothelial cell", "up_in_human", E, "class II genes are not 1:1 except e.g. CD74/CIITA")],
    "CD58 (LFA-3)": [
        ("CD58", "human", "CD58 (LFA-3)", "not stated (listed with endothelial/costimulation rows)", "up_in_human", G, "no mouse ortholog")],
    "CD40 on EC": [
        ("CD40", "human", "CD40", "endothelial cell", "up_in_human", E, "")],
}


def main(src, out):
    with open(src, newline="") as fh:
        table = list(csv.DictReader(fh, delimiter="\t"))
    feats = {r["feature"]: r for r in table}
    missing = [f for f in MAP if f not in feats]
    if missing:
        sys.exit("features not in transcription: %s" % missing)
    rows = []
    for feat, entries in MAP.items():
        r = feats[feat]
        quote = 'Table I row "%s": Mouse: "%s" | Human: "%s" (refs %s)' % (feat, r["mouse"], r["human"], r["refs"] or "none")
        for gene, sp, pub, ct, direction, kind, extra in entries:
            note = "[%s] published name: %s; %s" % (kind, pub, quote)
            if extra:
                note += "; " + extra
            rows.append([SRC, "text_known_difference", gene, sp, "human_vs_mouse", ct, direction,
                         "qualitative (review table)", "Mestas & Hughes 2004 Table I (literature review)", "no", note])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLS)
        w.writerows(rows)
    print("rows", len(rows), "features used", len(MAP), "of", len(table), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
