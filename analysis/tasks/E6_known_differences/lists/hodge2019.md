# hodge2019: derivation card

**Citation.** Hodge RD, Bakken TE, Miller JA, et al. Conserved cell types with divergent features in human versus mouse cortex. *Nature* 573, 61–68 (2019). DOI 10.1038/s41586-019-1506-7.

**DOI check (2026-11-03).** `curl -sIL https://doi.org/10.1038/s41586-019-1506-7` gives 302 to `https://www.nature.com/articles/s41586-019-1506-7`, then the cookie redirects, then HTTP 200. The article page's `citation_title`, volume 573 and pages 61–68 match.

**Data.** Human MTG snRNA-seq (SMART-seq v4) against mouse VISp and ALM scRNA-seq (Tasic et al. 2018). This is not our M1 data, so `circular = no`.

**Files.** All files are in `data/analysis/E6/hodge2019/`; see `MANIFEST.tsv` for URLs and sha256. They were converted to TSV with `xlsx_to_tsv.py` into `converted/`.
- `41586_2019_1506_MOESM1_ESM.pdf`: the guide to Supplementary Tables 1–6.
- `MOESM6_ESM.xlsx` is **Supplementary Table 4** (homologous types). Its sheet is named `SI_Table_1_human_mouse_homology`.
- `MOESM7_ESM.xlsx` is **Supplementary Table 5** (gene expression summary). It has `homol_expr_score` and `homol_cor` per gene.
- MOESM3/4/5/8 (ST1, 2, 3, 6) were downloaded and converted but not used for the lists.

**Script.** `derive/derive_hodge2019.py <converted_dir> <out_tsv>` writes `lists/hodge2019.tsv`. The contrast for every row is `human_MTG_vs_mouse_VISp_ALM`.

## Sets and rules

The rules below are the paper's own; nothing was tuned.

1. **`human_mouse_divergent`** (3,382 rows, 3,382 unique genes)
   - **Source.** ST5 `homol_expr_score`, the paper's β-score of log2 human–mouse expression differences across the 37 homologous types (Methods, "Quantification of expression divergence").
   - **Coverage.** 14,553 genes have a score. That matches the paper's 14,553 orthologous genes.
   - **Rule.** Genes are ranked by score, descending, and the top **3,382** are kept. This is the paper's own count: "Twenty-three per cent of genes (3,382) were more highly divergent than 95% of 252 housekeeping genes".
   - **Why rank and not the housekeeping cut.** The 252-gene housekeeping list (ref. 49) is not in the supplement, so the cut is set by the published count. There is no tie at the cut-off (2.6653 vs 2.6646).
   - **Tiers.** `note` records the paper's other two cut-offs as tiers. top3pct means rank ≤ 437 (437 genes). top10pct means rank ≤ 1,455 (1,018 more genes). The other 1,927 genes are tagged top23pct.
   - **Not per cell type.** The score summarises all 37 types, so the gene list has no cell-type resolution. `cell_type = all_37_homologous_types`. `note` also gives the human cluster with the highest FPKM.
   - **Direction.** `divergent`, unsigned. A high score means the cell-type pattern shifted between species.
   - **Effect column.** `effect` = `homol_expr_score`, rank and `homol_cor`.
2. **`conserved_control`** (175 rows, 152 unique genes, 31 cell types)
   - **Source.** ST4 `homol_shared_markers`, up to 10 per homologous type.
   - **Paper's definition.** "cluster-enriched expression (expressed in >50% of cells or nuclei in the cluster of interest and five or fewer additional clusters) in both species".
   - **Rows.** One row per (gene, homologous type). `note` lists the member human and mouse clusters.
   - **Types with no shared markers ("none").** Of the 37 homologous types, 31 have markers. These six have none, so they have no rows: Exc L5/6 IT 3, Pvalb 1, Sst 4, Vip 1, Vip 5 and Vip Sncg. The paper notes that homologous types may lack shared markers.
   - **Mixed and unmatched rows.** The five extra ST4 rows ("Mixed (…)" and "Not identified") also have "none" and are skipped.
3. **`text`** (10 rows, 10 genes)
   - All statements are quoted verbatim in `note`, with `evidence = text`.
   - **HTR3A** in human CGE types: `up_in_mouse` ("the canonical mouse CGE interneuron marker HTR3A was not expressed in human CGE types").
   - **NPY** in deep SST / Sst Chodl: `up_in_mouse` ("NPY was much more sparsely expressed in human"). This is ISH cell density, not a level.
   - **TH** in Inh L5-6 SST TH: `conserved`.
   - **Chandelier cells.** VIPR2 is `up_in_mouse`; COL15A1 and NOG are `up_in_human` ("chandelier cells express Vipr2 in mouse but COL15A1 and NOG in human"). Vipr2 is written as a mouse symbol in the paper; its human ortholog VIPR2 is used here.
   - **COL24A1, COL12A1, GRIK1 and GRIN3A:** `divergent` ("expressed in different cell types between species").
   - **Effect column.** `effect` gives the gene's ST5 rank as a cross-check.

Breakdown of the 175 conserved rows by type: Astrocyte 10, Chandelier 8, Endothelial 10, Exc L2/3 IT 3, Exc L3/5 IT 10, Exc L4/5 IT 2, Exc L5 PT 10, Exc L5/6 IT 1 8, Exc L5/6 IT 2 5, Exc L5/6 NP 1, Exc L6 CT 1, Exc L6 IT 1 3, Exc L6 IT 2 10, Exc L6b 5, Lamp5 1 3, Lamp5 2 3, Lamp5 Lhx6 4, Lamp5 Rosehip 2, Microglia/PVM 10, OPC 10, Oligo 10, Pax6 6, Pvalb 2 11, Sst 1 3, Sst 2 7, Sst 3 2, Sst 5 1, Sst Chodl 10, Vip 2 3, Vip 3 2, Vip 4 2.

## Assumptions and oddities

- **The per-type list was not published.** The paper's per-type rule was a fold difference of more than 10 between human and mouse in each of the 37 homologous types (9,748 genes, Fig. 6b). Its table is not in the supplement, and neither is it in the code repository (github.com/AllenInstitute/MTG_celltypes, checked: no per-type expression tables). The list was not recomputed from raw data. So the only divergent set from this paper is the gene-level β-score set above.
- **The scores do not lie in 0–1.** The Methods say the scores range from 0 to 1, but ST5 `homol_expr_score` runs from 0.21 to 10.0. Only ranks are used, so this does not matter here.
- **The serotonin-receptor check is off by one.** The paper says "four of seven G-protein-coupled receptors" are in the top 10%. With the ST5 ranking, five are: HTR1F (rank 68), HTR4 (279), HTR2C (402), HTR7 (456) and HTR2A (1,435, just inside rank 1,455). HTR3A (1,232) and HTR3B (1,450) are also in the top 10%, as the paper says. ST5 scores eight serotonin GPCRs, while the paper's Fig. 6e shows seven, so the rounding or the gene universe probably differs slightly.
- **GRIK1 does not fall in the top 3%.** The paper names it among the top 3%, but ST5 ranks it 896 (top 6.2%). COL24A1 (353), COL12A1 (178) and GRIN3A (65) do fall in the top 3%. GRIK1 is kept in `text` as quoted, and in `human_mouse_divergent` with tier top10pct.
- **NOG has no score in ST5.** Its text row has `effect` = NA.
- **Pvalb 2 has 11 shared markers** although the paper says "up to 10". All 11 are kept as published.
- **The areas differ.** The comparison is human MTG against mouse VISp/ALM, not M1 against M1.
