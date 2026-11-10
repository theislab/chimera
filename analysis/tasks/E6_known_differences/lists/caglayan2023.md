# caglayan2023: derivation card

**Citation.** Caglayan E, Ayhan F, Liu Y, et al. Molecular features driving cellular complexity of human brain evolution. *Nature* 620, 145–153 (2023). DOI 10.1038/s41586-023-06338-4. PMID 37468639, PMCID PMC11161302 (author manuscript NIHMS1994710).

**DOI check (2026-11-03).** `curl -sI https://doi.org/10.1038/s41586-023-06338-4` gives HTTP 302 to `https://www.nature.com/articles/s41586-023-06338-4`. The Crossref CSL record gives the title above, *Nature* 620, 145–153, issued 2023-07-19, first author Caglayan.

**Region.** This is **not DLPFC.** The data are snRNA-seq and snATAC-seq of the **posterior cingulate cortex, Brodmann area 23**, from human, chimpanzee and rhesus macaque. The text says: "we examined the evolution of Brodmann area 23 (BA23)… part of the posterior cingulate cortex". The snRNA-seq covers 148,540 nuclei: human 41,397, chimpanzee 53,539, macaque 53,604.

**Circularity.** These are new donors in a different region from our cohorts, so `circular = no`. The paper re-analyses Ma 2022 and Bakken 2021 only for side checks on cell proportion and FOXP2. Those checks do not feed Table S3.

**Files.** All files are in `data/analysis/E6/caglayan2023/`; see `MANIFEST.tsv`.
- The files come from the PMC author-manuscript supplement, fetched with the proof-of-work step described in jorstad2023.md.
- `NIHMS1994710-supplement-Table_S3.xlsx` is "Species specific genes per cell type". It was converted to `converted/` with `xlsx_to_tsv.py`.
- `Supplementary_Info.docx` holds the table legends.
- `PMC11161302.319.txt` is the full text, used for the methods quotes.
- The other tables (S1, S2, S4–S11) were downloaded and converted but not used.

**Script.** `derive/derive_caglayan2023.py <converted_dir> <out_tsv>` writes `lists/caglayan2023.tsv` (39,407 rows).

## Paper's definitions (quoted)

- **Supplementary Info, Table S3.** "This table reports differentially expressed genes per cell type from the snRNA-seq experiments. Species-specific regulation of each gene is annotated along with the direction of regulation (upregulation or downregulation are shown as UP and DOWN respectively). Statistics are obtained using edgeR on aggregated count matrix."
- **Methods, pseudobulk DEG cut.** "DGE analysis was performed using edgeR QLRT approach and differentially expressed genes (DEGs) were determined with FDR (< 0.05) and logFC (logFC >0.3 or logFC < −0.3) cutoffs. DGE analysis was performed with the following covariates: humanized age, sex, and library batch."
- **Methods, species-specific rule.** "HS-Genes were determined as DEGs that are H > C = M or H < C = M (C = M was determined if FDR > 0.1 …). The same criteria were used for CS-Genes. Genes that are consistently different between macaque-human and macaque-chimpanzee were referred to as macaque versus human-chimpanzee genes."
- **Table S3 legend sheet.** `avg_logfc_*` is the "Average log (ln) fold change". `hc` is human–chimpanzee, `hm` is human–macaque, `cm` is chimpanzee–macaque.

## Mechanical rules

1. **`human_specific`.** Every row of sheet `Human_Specific_Genes` is kept. Every row has `Evolution = Human_Specific`. Direction comes from `Regulation`: Human_UP gives `up_in_human`, Human_DOWN gives `down_in_human`. `effect` is `lnFC_hc=…;lnFC_hm=…`. `note` carries `FDR_hc`, `FDR_hm` and `MvsHC`. The contrast is `human_vs_chimpanzee_macaque`.
2. **`ape_vs_macaque_divergent`.** Every row of sheet `Macaque-Specific_Genes` is kept. These are the paper's "macaque versus human-chimpanzee genes", and every row has `MvsHC` set to Macaque_UP or Macaque_DOWN. The contrast is `human_chimpanzee_vs_macaque`.
   - Direction is written human-centric: Macaque_UP gives `down_in_human`, meaning human and chimpanzee are lower than macaque.
   - `note` keeps `MvsHC` and `Evolution`. 3,179 of these rows are also Human_Specific or Chimp_Specific.
3. **Not exported.** The `Chimpanzee_Specific_Genes` sheet (CS-Genes) is not exported.

**Sign check.** In all Human_Specific rows, Human_UP corresponds to `avg_logfc_hc > 0`, and `hc` and `hm` have the same sign in 100% of rows. In the macaque sheet, Macaque_UP corresponds to `avg_logfc_hm < 0` and `avg_logfc_cm < 0` in 100% of rows.

## Counts

| set | rows | unique genes | cell types | up / down in human |
|---|---|---|---|---|
| human_specific | 7,881 | 2,339 | 55 | 5,108 / 2,773 |
| ape_vs_macaque_divergent | 31,526 | 4,594 | 55 | 14,928 / 16,598 |

**`human_specific` rows per cell type:**
- Astrocytes: Ast_Interlaminar 63, Ast_Protoplasmic1 219, Ast_Protoplasmic2 52.
- Microglia 377.
- Oligodendrocytes:
  - OPC1 190, OPC2 86.
  - MOL1 351, MOL2 217, MOL3 220, MOL4 252, MOL5 275, MOL6 261, MOL7 223, MOL8 156.
- Excitatory neurons:
  - L2-3_1 172, L2-3_2 159, L2-3_3 141.
  - L3-5_RORB_1 195, L3-5_RORB_2 159, L3-5_RORB_3 203.
  - L4-5_RORB_1 156, L4-6_FEZF2 193, L4-6_RORB_2 172, L4-6_RORB_3 180.
  - L5-6_FEZF2_1 166, L5-6_FEZF2_2-3 173.
  - L5-6_THEMIS_1 143, L5-6_THEMIS_2 204.
- Inhibitory neurons:
  - ADARB2_L1-2 50, ADARB2_SYT6_L1-3 46.
  - LAMP5_L1-4 337, LAMP5_L2-6 221, LAMP5_NMBR_L1 90.
  - LHX6_FILIP1_L5-6 12, LHX6_MEPE_L4-6 111, LHX6_SULF1_L2-4 60.
  - PAX6_L1-2 28.
  - PVALB_L2-4 150, PVALB_L2-5 107.
  - SST_L1-3 190, SST_L3-5 79, SST_L4-5 123, SST_L4-6 131, SST_L5-6_1 70, SST_L5-6_2 38, SST_NPY_L3-6 8.
  - VIP_L1-2 5, VIP_L1-3 128, VIP_L1-3_1 114, VIP_L1-3_2 67, VIP_L1-4 73, VIP_L2-4 25, VIP_L2-5 78, VIP_L2-6 57, VIP_L3-6 125.

**`ape_vs_macaque_divergent` rows per cell type** range from 121 (VIP_L1-2) to 944 (Ast_Protoplasmic1). The derivation script prints the full crosstab.

## Assumptions and oddities (important)

- **The published table does not match the stated pseudobulk thresholds.** What the table shows, checked 2026-11-03:
  - Every Human_Specific row has FDR_hc < 0.05 and FDR_hm < 0.05.
  - The minimum |avg_logfc| is **0.25**, not the 0.3 stated for pseudobulk. The column names (`avg_logfc`) and the 0.25 cut are those the Methods give for the **single-cell MAST** approach: "FDR < 0.05 and absolute average log (ln) fold change > 0.25".
  - The "C = M (FDR > 0.1)" condition is **not** visible in the `adj_p_value_cm` column: 5,151 of 7,881 Human_Specific rows have FDR_cm ≤ 0.1.
- **Labels are taken as published.** I took the authors' published labels (`Evolution`, `Regulation`, `MvsHC`) and did not re-threshold. I cannot tell from the table alone whether S3 holds pseudobulk or MAST results. The legend says edgeR.
- **Cell types are BA23 subtypes.** They are not the Ma 2022 or Jorstad subclasses, so mapping to our DLPFC labels is a later step. There are no endothelial, VLMC or pericyte types in S3.
- **Gene symbols are human.** `symbol_species = human`, using orthologous genes on human annotation.
- **No conserved counterpart.** The paper provides no conserved list, so there is no `conserved_control` set.
