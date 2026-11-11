# ezran2025a_tmc: primate (human + mouse lemur) vs mouse cell-type expression, Tabula Microcebus

**Citation.** Ezran C, Liu S, Chang S, et al. (Tabula Microcebus Consortium). A molecular cell atlas of mouse lemur, an emerging model primate. *Nature* 644, 173–184 (2025). DOI 10.1038/s41586-025-09113-9.
**DOI verified.** `curl -sIL https://doi.org/10.1038/s41586-025-09113-9` gave 302 to nature.com/articles/s41586-025-09113-9 (2026-11-03).

**Files** (`data/analysis/E6/ezran2025a_tmc/`, MANIFEST.tsv):
- `41586_2025_9113_MOESM3_ESM.zip` (Supplementary Tables 1–8). These tables were extracted to `unz/` and converted to `tsv/`:
  - Table 4 "Datasets and cell types analyzed across species"
  - Table 6 "DEGs across species", sheets `primate_high`, `primate_low` and `conserved`
  - Table 7 "Gene expression conservation", which is gene-level and was not used for rows
- `41586_2025_9113_MOESM1_ESM.pdf` (Supplementary Information)
- `s41586-025-09113-9.html` (article page; source of the Methods quote)

**Paper's rule (Methods, verbatim):**
> "First, genes with significantly primate-enriched expression, which requires that (1) cell-type mean expression of the gene is above 0.5 in both humans and lemurs and (2) 5-fold greater expression and P < 1 × 10–5 in both species compared with the orthologous mouse cell type. Second, genes with significantly primate-depleted expression, which requires that (1) cell-type mean expression of the gene is above 0.5 in mouse and (2) 5-fold lower expression and P < 1 × 10–5 in human and lemur compared with the orthologous mouse cell type. Third, genes that are significantly enriched in a cell type, which requires that (1) cell-type mean expression of the gene is above 0.5 in all three species and (2) 5-fold greater expression and P < 1 × 10–5 when comparing this cell type versus other cell types. The full list of the identified genes is provided in Supplementary Table 6."

The tests are Wilcoxon rank-sum tests.

**Derivation** (`derive/derive_ezran2025a_tmc.py <tsv_dir> <out>`):
- `primate_high` → set `primate_enriched`, direction `up_in_human` (human and lemur both higher than mouse).
- `primate_low` → set `primate_depleted`, direction `up_in_mouse`.
- `conserved` → set `conserved_control`, direction `conserved` (cell-type-enriched in all three species).
- All rows are kept, for all tissue groups. `cell_type` is "tissue group | compartment | unified annotation".
- `contrast` is `primate(human+lemur)_vs_mouse`, or `human_lemur_mouse_celltype_vs_rest` for the conserved rows.
- Symbols are human NCBI names.
- A fold change of 65535 appears as published; it looks like a sentinel for division by zero.

**Counts.** Rows 17,232; 5,887 unique genes overall.

| set | rows | unique genes |
|---|---|---|
| primate_enriched (up_in_human) | 4,874 | 1,627 |
| primate_depleted (up_in_mouse) | 6,086 | 2,482 |
| conserved_control | 6,272 | 3,307 |

Unique genes by tissue group (non-conserved sets only):

| tissue group | up_in_human | up_in_mouse |
|---|---|---|
| Lung | 599 | 1,075 |
| Marrow & Spleen | 709 | 779 |
| Liver | 103 | 200 |
| Muscle | 696 | 636 |
| Testes | 254 | 832 |

**Circularity: yes.** Supp Table 4 lists the data: human = Tabula Sapiens (lung, bone marrow, spleen, liver, muscle), lemur = Tabula Microcebus, mouse = Tabula Muris Senis. This is the same data family as our bone marrow, lung and liver cohorts. The liver rows are TS-vs-TMS based, a comparison that is already used as an independent reference (`analysis/tasks/T4_1_gold`).

Testes rows use Shami et al. (human/macaque) and Ernst et al. (mouse), so they are marked `circular=no`. Testes is not one of the cohorts of this task.

**Assumptions.**
- "Marrow & Spleen" pools the bone marrow and spleen datasets, as the paper does.
- "(PF)" annotations are the paper's own labels.
