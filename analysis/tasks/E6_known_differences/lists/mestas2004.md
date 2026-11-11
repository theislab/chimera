# mestas2004: Of mice and not men (immune differences, Table I)

**Citation.** Mestas J, Hughes CCW. Of mice and not men: differences between mouse and human immunology. *J Immunol* 172(5), 2731–2738 (2004). DOI 10.4049/jimmunol.172.5.2731; PMID 14978070 (no PMC copy).
**DOI verified.** `curl -sIL https://doi.org/10.4049/jimmunol.172.5.2731` gave 302 to academic.oup.com/jimmunol/article/172/5/2731/8060352 (2026-11-03). That page answers 403 to curl because of a bot wall.

**File.** `data/analysis/E6/mestas2004/table1_transcribed_webfetch.tsv` is a transcription of Table I, "Summary of some known immunological differences between mouse and human". It has 67 rows with columns feature / mouse / human / refs, and was transcribed from the OUP article page, fetched on 2026-11-03.
- Direct download was blocked, so this is a transcription and not an original file.
- MANIFEST.tsv records the source URL and the hash of the transcription.
- The italic markup on c-kit and flt-3 was removed.
- The TLR9 human cell reads "Expressed only on B cells, plasmacytoid DC and N" on the page. The "N" is kept as printed: it is probably "neutrophils", but this was not confirmed.
- The only footnote is "* absent in C57BL/6, /10, SJL and NOD mice, which have IgG2c" (IgG2a row).

**Rule** (`derive/derive_mestas2004.py <transcription> <out>`):
- Every Table I row whose feature names a gene product and gives its presence or expression in mouse vs human produces one output row per (gene, cell type named in that row). The mapping from each feature to genes and cell types is written out in the script.
- `note` holds the verbatim mouse and human cells and the reference numbers, prefixed with a tag:
  - `[expression]`: the row gives cell-type expression that differs between the species.
  - `[gene_content]`: one species lacks the gene or family, or has only a pseudogene. These genes are mostly not 1:1 orthologs.
- Direction:
  - `up_in_human` = present or higher in human only.
  - `up_in_mouse` = present or higher in mouse only.
  - `conserved` = named for both species.
  - `divergent` = the co-expression pattern changes without a clear side (CD5/CD23 on B cells; IL-4/IFN-γ in cultured Th).
- `gene_symbol` is the HGNC symbol where the published name maps to one gene (published name in `note`). Otherwise the published name is kept: "MHC II", "KIR", "Ly49 family", "ULBP".
- set = `text_known_difference`; circular = `no` (literature review).

**Excluded rows** (36 of 67), because they name no gene or describe phenotype, function or regulation rather than expression:
- splenic hematopoiesis; BALT
- blood neutrophil and lymphocyte percentages
- sialic acid Neu5Gc (gene CMAH not named)
- leukocyte defensins; Paneth defensins (×2, families, no gene named)
- macrophage NO induction
- predominant skin/mucosal T cells; γδ response to phospho-antigens
- fMLP receptor affinity; serum IgA; Ig classes; CDR-H3
- BLNK, Btk, λ5, γc, Jak3, IL-7R, ZAP70, caspase 8 and ICOS deficiencies
- IL-13 effect on B cells; IFN-α Th1 promotion; B7-H3 effects
- P-selectin promoter inducibility
- IFN-γ in demyelinating disease; DTH lesions
- EC antigen presentation; T-cell dependence on CD2 ligands; CD2-ligand affinity (CD48 vs CD58)
- vascularized grafts; microchimerism; passenger leukocytes

**Counts.** 65 rows from 31 Table I features; 54 unique symbols.

| kind | direction | unique genes | genes (cell type) |
|---|---|---|---|
| expression | up_in_human | 10 | FLT3 (HSC), TLR2 (PBL), CD33 (monocyte), CD4 (macrophage), CD38 (GC B, plasma), IL10 (Th1), MHC II (T cell, endothelial cell), KCNA3 (T cell), MUC1 (T cell), CD40 (endothelial cell) |
| expression | up_in_mouse | 7 | KIT (HSC), TLR2 (T cell, inducible), TLR3 (macrophage), TLR9 (all myeloid cells), CD33 (granulocyte), THY1 (T cell, thymocyte), CD28 (CD4+ and CD8+ T) |
| expression | divergent | 4 | CD5, FCER2 (B cell); IL4, IFNG (cultured Th) |
| expression | conserved | 3 | TLR3 (DC), TLR9 (B, pDC, "N"), IL10 (Th2) |
| gene_content | up_in_human | 26 | TLR10, CD1A, CD1B, CD1C, KIR, MICA, MICB, ULBP, FCAR, FCGR2A, FCGR2C, CASP10, ICAM3, GNLY, CXCR1, CXCL8, PPBP, CXCL11, CCL13, CCL14, CCL15, CCL23, CCL18, CCL24, CCL26, CD58 |
| gene_content | up_in_mouse | 8 | Ly49 family, H60, Raet1b, GLYCAM1, Ccl6, Ccl9, Cxcl15, Ccl12 |
| gene_content | conserved | 1 | CD1D |

**1:1 caveats noted in `note`.**
- Most gene_content rows have no 1:1 ortholog.
- The Table's 2004 "absent in mouse" claims for CXCR1, PPBP, CXCL11, CCL24 and CCL26 are outdated: mouse orthologs exist.
- CD33 orthology is imperfect. MHC II is not 1:1.

**Circularity: no.** This is a review compiled from older independent literature.
