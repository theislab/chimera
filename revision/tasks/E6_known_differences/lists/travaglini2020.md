# travaglini2020: human vs mouse lung cell-type expression

**Citation.** Travaglini KJ, Nabhan AN, Penland L, Sinha R, Gillich A, Sit RV, et al., Krasnow MA. A molecular cell atlas of the human lung from single-cell RNA sequencing. *Nature* 587, 619–625 (2020). DOI 10.1038/s41586-020-2922-4; PMID 33208946; PMC7704697.
**DOI verified.** `curl -sIL https://doi.org/10.1038/s41586-020-2922-4` gave 302 to nature.com/articles/s41586-020-2922-4 (2026-11-03).

**Files** (`data/revision/E6/travaglini2020/`, see MANIFEST.tsv; TSVs made with xlsx_to_tsv.py into `tsv/`):
- `41586_2020_2922_MOESM9_ESM.xlsx` = Supplementary Table 7, "Genes specific to mouse and human in each cluster and lung wide". There are 34 sheets: 33 homologous cluster pairs (`Cluster_1`..`Cluster_33`) and one lung-wide sheet (`Cluster_34`).
- `41586_2020_2922_MOESM11_ESM.xlsx` = Supplementary Table 9, "Evolutionary and functional classes of genes". The single sheet is named `Table_S8`.
- `41586_2020_2922_MOESM10_ESM.xlsx` = Supplementary Table 8, a summary of the evolution types. Used only as a check.
- `41586_2020_2922_MOESM8_ESM.xlsx` = Supplementary Table 6, the mouse cluster identities. It lists the mouse IDs, which are Tabula Muris Senis animals.
- `PMC7704697_fullText.xml` is the full text. The Methods quotes below come from it.
- The ESM numbering matches the Supplementary Table numbering on the Nature page (Tables 1–9 = MOESM3–MOESM11).

**Paper's rule (Methods, verbatim):** "We defined species-specific gene expression as those enriched 20-fold in either direction (mouse > human or human > mouse) with a p-value less than 10^−5 (calculated by 'MAST' as above) from all cells for the indicated types in the SS2 dataset." The Fig. 4b legend and the main text state "≥20-fold change, p<0.05". The published Table 7 matches the second version: min |avg_logFC| = 3.0 = ln 20, and max raw p = 0.0499. The rows were kept as published, with no extra p filter.

**Derivation** (`derive/derive_travaglini2020.py <tsv_dir> <out>`):
1. Supp Table 7 gives set `species_divergent`, one row per gene × cluster pair.
   - `avg_logFC` is the natural-log fold change of mouse over human: > 0 gives `up_in_mouse`, < 0 gives `up_in_human`.
   - `cell_type` is the sheet title, e.g. "human_Macrophage vs mouse_Alveolar Macrophage". `Cluster_34` is "lung (all SS2 cells, lung-wide)".
   - Symbols are human. The orthologous mouse gene was mapped by the authors.
   - `Hs(...)`/`Mm(...)` are paralog groups collapsed by the authors. They are flagged in `note` as not 1:1.
2. Supp Table 9 gives a gene-level pattern with `cell_type` = "lung (gene-level pattern across homologous SS2 clusters)".
   - Evo_Type 0 → `conserved_control`.
   - Evo_Type 1 → `evo_type1_gain_loss`, with direction `up_in_human` if Human_Spec_Num>0 & Mouse_Spec_Num=0, or `up_in_mouse` if the reverse.
   - Evo_Type 2 → `evo_type2_expansion`, direction `divergent`.
   - Evo_Type 3 → `evo_type3_switch`, direction `divergent`.
   - Evo_Type NA means the gene is not expressed in lung. These 4,291 genes were dropped.
   - Type 2 is labelled `divergent` because its side cannot be recovered. The table has no type-2 gene with Mouse_Spec_Num>0 & Human_Spec_Num=0, yet Supp Table 8 reports about 1,095 "expansion in mouse" genes. The columns may be incomplete or mislabelled.

**Counts** (rows / unique genes):

| set | direction | rows | unique genes |
|---|---|---|---|
| species_divergent | up_in_human | 2,090 | 1,168 |
| species_divergent | up_in_mouse | 4,240 | 1,473 |
| species_divergent | any | 6,330 | 2,516 |
| conserved_control (Type 0) | conserved | 510 | 510 |
| evo_type1_gain_loss | up_in_human | 1,561 | 1,561 |
| evo_type1_gain_loss | up_in_mouse | 1,673 | 1,673 |
| evo_type2_expansion | divergent | 5,619 | 5,619 |
| evo_type3_switch | divergent | 1,286 | 1,286 |

`species_divergent` split by compartment (unique genes):

| compartment | up_in_human | up_in_mouse |
|---|---|---|
| immune clusters (B, plasma, T, NK, neutrophil, basophil, macrophage, DC, monocytes) | 530 | 654 |
| non-immune clusters | 738 | 938 |
| lung-wide | 72 | 391 |

The file has 35 distinct `cell_type` values in total.

**Circularity: yes.** The mouse side is Tabula Muris Senis lung and blood: about 35,000 TMS SS2/10x cells plus 522 extra SS2 cells, re-annotated. The Methods say "Approximately 35,000 mouse lung and blood cell expression profiles by SS2 and 10x from Tabula Muris Senis were combined with 522 cells…", and the mouse IDs in Supp Table 6 are TMS animals. The human side is Travaglini's own Stanford donors, not Tabula Sapiens. Even so, the contrast shares its mouse data with our TMS lung cohort.

**Assumptions.**
- Cluster pairing, as published, is sometimes many-to-one. For example, human Airway and Vascular Smooth Muscle are both compared with mouse Smooth Muscle, and human CD8 Naive and Memory/Effector T with mouse CD8+ T.
- The Table 9 classification uses binarised expression ("expressed if the median of non-zero values ..."; see Methods). Its direction labels are a mechanical reading of the count columns.
