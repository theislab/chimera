# khrameeva2020: derivation card

**Citation.** Khrameeva E, Kurochkin I, Han D, et al. Single-cell-resolution transcriptome map of human, chimpanzee, bonobo, and macaque brains. *Genome Research* 30, 776–789 (2020). DOI 10.1101/gr.256958.119. PMID 32424074, PMCID PMC7263190 (open access).

**DOI check (2026-11-03).** `curl -sI https://doi.org/10.1101/gr.256958.119` gives HTTP 302 to `http://genome.cshlp.org/lookup/doi/10.1101/gr.256958.119`. The Crossref CSL record gives the title above, *Genome Research* 30, 776–789, issued 2020-05, first author Khrameeva, 19 authors.

**Region.** snRNA-seq covers three regions, each from human, chimpanzee, bonobo and rhesus macaque (3 individuals per species, pooled per region):
- **anterior cingulate cortex (AC, BA24)**, the only cortical region;
- caudate nucleus (CN);
- cerebellar gray matter (CB).

**Circularity.** These are not our donors or regions, so `circular = no`.

**Files.** All files are in `data/analysis/E6/khrameeva2020/`; see `MANIFEST.tsv`.
- `supplementaryFiles` is the Europe PMC bundle of the whole supplement, a zip. Its file name is the URL basename. It was unzipped to `extracted/`.
- `extracted/supp_gr.256958.119_Supplemental_Table_S6.xlsx` is Table S6, "Human accelerated regions (HARs) and human-specific genes". It was converted to `converted/`. The sheet used is `human-specific genes`.
- `PMC7263190.1.txt` is the full text (PMC AWS open-data).
- `hgnc_complete_set.txt` is the HGNC complete set, downloaded 2026-11-03. It is used only to add a symbol to `note`.

**Script.** `derive/derive_khrameeva2020.py <data_dir> <out_tsv>` writes `lists/khrameeva2020.tsv` (2,960 rows).

## Paper's definition (quoted, Methods "HARs analysis")

"Analysis of snRNA-seq data resulted in a list of 1271 unique genes showing human-specific expression differences in at least one cell type (FindMarkers function in Seurat 3.0; human nuclei compared to chimpanzee, bonobo, and macaque nuclei combined in each t-SNE cluster separately; fold-change threshold = 2; Wilcoxon rank-sum test; Bonferroni adjusted P-value < 0.05)."

## Mechanical rule

- **Paper's own cut.** Keep the rows of the `human-specific genes` sheet with `p_val_adj < 0.05`.
  - The sheet holds 3,348 rows; 388 rows have `p_val_adj ≥ 0.05` and were dropped.
  - All kept rows already have |avg_logFC| ≥ ln 2, which is the fold-change-2 threshold on Seurat 3's natural-log scale.
- **Direction.** Taken from the sign of `avg_logFC`. Positive means higher in human: it agrees with `pct.H > pct.O` in 99.8% of rows.
- **Columns.**
  - `effect` is `lnFC=…`.
  - `note` carries `hgnc_symbol=`, `region=`, `p_val_adj`, `pct.H` and `pct.O`.
  - The contrast is `human_vs_chimpanzee_bonobo_macaque`.
- **`gene_symbol` holds the published Ensembl gene ID** (ENSG…). The HGNC symbol is in `note` as `hgnc_symbol=…`. 2 IDs (4 rows) have no HGNC entry and an empty symbol.

**Check against the text.** After the cut, the list has 1,268 unique Ensembl IDs. The paper says 1,271. The small gap is probably rounding or ID versions.

## Counts

| set | rows | unique genes | cell types | up / down in human |
|---|---|---|---|---|
| human_specific | 2,960 | 1,268 | 16 | 2,256 / 704 |

**By region:**

| region | rows | unique genes |
|---|---|---|
| AC (cortex) | 1,169 | 818 |
| CN | 1,226 | 813 |
| CB | 565 | 408 |

**Rows per cell type (up / down in human):**
- AC: Ast 245 (170/75), Ex 523 (360/163), In 94 (77/17), MG 79 (50/29), OD 140 (103/37), OPC 88 (80/8).
- CN: Ast 442, In 138, MG 59, Neu 141, OD 323, OPC 123.
- CB: Ast 105, In 234, Neu 105, OD 121.

## Assumptions and oddities

- **All three regions are kept.** The rule is to keep all cell types. Only the `AC *` rows are cortex; filter on `cell_type` prefix or on `region=` in `note`.
- **Cell types are broad.** For example, `AC Ex` is all excitatory neurons and `AC In` is all inhibitory neurons. There are no subclasses.
- **What "human-specific" means here.** Human is compared with the three non-human primates pooled, using a single-cell Wilcoxon test. This is not pseudobulk. The donor count is small (3 per species).
- **The fold-change rule is not applied.** The text also defines bulk "human-specific" genes by a twofold-greater human/macaque vs chimp/macaque rule. That definition belongs to the bulk analysis (Table S2) and is **not** applied here.
- **No conserved counterpart.** The paper provides no conserved list.
