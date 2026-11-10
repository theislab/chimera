# ingersoll2010: human vs mouse monocyte subsets (text-derived)

**Citation.** Ingersoll MA, Spanbroek R, Lottaz C, Gautier EL, Frankenberger M, Hoffmann R, Lang R, Haniffa M, Collin M, Tacke F, Habenicht AJR, Ziegler-Heitbrock L, Randolph GJ. Comparison of gene expression profiles between human and mouse monocyte subsets. *Blood* 115(3), e10–e19 (2010). DOI 10.1182/blood-2009-07-235028; PMID 19965649; PMC2810986.
**DOI verified.** `curl -sIL https://doi.org/10.1182/blood-2009-07-235028` gave 302 to ashpublications.org/blood/article/115/3/e10/27161/... (2026-11-03). That page then answers 403 to curl.

**File.** `data/analysis/E6/ingersoll2010/PMC2810986.html` is the PMC article page (MANIFEST.tsv).

**Table status: not obtained.**
- The Europe PMC supplementaryFiles endpoint returned no zip.
- The publisher site is blocked by a 403.
- The paper's gene lists (132 conserved genes, 33 probe sets in a converse pattern) are in figures and supplemental files that could not be retrieved.
- The list is therefore text-derived.

**Rule** (`derive/derive_ingersoll2010.py <page> <out>`):
- Every gene named in sentences that state a human-vs-mouse difference is kept, with no selection. The script checks that each quote occurs verbatim in the saved page.
- The comparison is between monocyte subsets: human CD14++CD16− / CD16+ vs mouse Ly-6C+ / Ly-6Clo, by microarray on blood monocytes. "Divergent" therefore means a converse subset pattern, not a level difference in one cell type.
- Rows from the PPARγ-signature sentences have direction `up_in_mouse`, with cell type "nonclassical monocyte (mouse Ly-6Clo vs human CD16+)" (or "monocyte" for the abstract sentence).

**Quotes used** (verbatim; Abstract, Results, Discussion; also in `note`):
1. Abstract: "Despite overall conservation, some molecules were conversely expressed between the 2 species' subsets, including CD36, CD9, and TREM-1."
2. Abstract: "Other differences included a prominent peroxisome proliferator-activated receptor γ (PPARγ) signature in mouse monocytes, which is absent in humans,"
3. Results: "...the most differentially expressed genes between mouse Ly-6C + and Ly-6C lo mouse monocytes include fatty acid binding protein 4 (Fabp-4) and its transcriptional regulator ... (PPARγ ...), ... including CD36 ..., Hsd11β1, ... and G0S2 ... By contrast, no PPARγ signature was evident in the human monocyte array"
4. Results: Msr-1 sentence ("...the expression of MSR-1 between human monocyte subsets is also converse to mouse subset expression").
5. Results: "In humans, these receptors are either more highly expressed on CD16 − monocytes (SIRPα, MSR-1, CD36, and thrombospondin 1/Thbs1 ...) or not differentially expressed."
6. Discussion: "Several receptors, including CD36, MSR-1, SIRPα, and thrombospondin 1, appear enriched on human CD16 − monocytes ..., whereas all of these molecules and others involved in recognition of dying cells (Treml4 ...) are enriched on Ly-6C lo mouse monocytes."
7. Discussion: "However, in humans, CD64 surface protein was absent on CD16 + monocytes, whereas in the mice, Ly-6C lo monocytes were still positive ..."

**Counts.** 20 rows; 12 unique genes.

| direction | unique genes | genes |
|---|---|---|
| divergent | 6 | CD36, CD9, TREM1, MSR1, SIRPA, THBS1 |
| up_in_mouse | 7 | PPARG, FABP4, CD36, HSD11B1, G0S2, TREML4, FCGR1A |

CD36 appears in both sets because it is named in both kinds of sentence.

**Caveats.**
- FCGR1A (CD64) is a protein-level statement, and human FCGR1A has paralogs.
- TREML4's 1:1 status is uncertain.

**Circularity: no.** The data are independent microarrays of human volunteers and C57BL/6 mice.
