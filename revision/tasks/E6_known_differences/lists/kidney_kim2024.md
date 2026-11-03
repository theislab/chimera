# kidney_kim2024: Kim et al. 2024, human vs mouse developing nephron and ureteric epithelium

**Citation.** Kim S, Koppitch K, Parvez RK, ..., Lindström NO, McMahon AP. Comparative single-cell analyses identify shared and divergent features of human and mouse kidney development. *Dev Cell* 59:2912–2930.e7 (2024). DOI 10.1016/j.devcel.2024.07.013. Preprint: bioRxiv 10.1101/2023.05.16.540880.
**DOI check.** `curl -sIL https://doi.org/10.1016/j.devcel.2024.07.013` returns 302 to linkinghub.elsevier.com/retrieve/pii/S1534580724004507, then 200. Crossref metadata matches.

**Data.**
- Human: fetal kidney snRNA-seq.
- Mouse: embryonic / P0 scRNA-seq.
- The two were integrated with Seurat CCA into 25 integrated clusters (nephron + ureteric epithelium).

**Developmental, not adult.** Only the mature-like clusters plausibly map onto the adult cohort: podocyte (1), PT (7), TAL (6/22), DCT/CNT (4), PC (3), IC (23).
**Circularity: no.** The authors' own data; not KPMP / Lake 2023, not CELLxGENE adult mouse.

**Files** (`chimera/data/revision/E6/kidney_kim2024/`, with MANIFEST.tsv). Cell Press mmc are legacy .xls (BIFF8), converted to `tsv/` with `derive/xls_to_tsv.py` (stdlib OLE2/BIFF8 reader written for this task).
- mmc6.xls = **Table S5** (README: "Conserved and human-specific DEGs and DARs for each corresponding human/mouse cluster"):
  - sheet 1 = Conserved RNA, 352 unique
  - sheet 2 = Divergent human RNA, 885 unique
  - sheet 3 = Divergent mouse RNA, 579 unique
- mmc5.xls = **Table S4**, per-cluster DEGs. Used only to label clusters: rank-1 marker of top50 sheets 1 (mouse) and 6 (human).
- mmc7 and mmc8: downloaded while locating the table, **not used**. mmc7 is Table S6 (mouse stage DEGs); mmc8 is Table S7 (CAKUT / GWAS).

**Rule (paper's own).** Text: "avg.FC >0.5, pct.diff > 0.3, p.adj < 0.05". The human-enriched set excludes genes that are also mouse DEGs in the same cluster.
- Sets:
  - human_enriched (direction up_in_human) = Table S5 sheet 2
  - mouse_enriched (up_in_mouse) = sheet 3
  - conserved_control (conserved) = sheet 1
- All published rows are kept.
- 3 rows in sheet 3 have p_val_adj ≥ 0.05 despite the stated rule. They are flagged in `note`.

**Counts.**
| Set | Rows | Unique genes |
|---|---|---|
| human_enriched | 1577 | 885 |
| mouse_enriched | 977 | 579 |
| conserved_control | 570 | 352 |

**Assumptions / caveats.**
- `cell_type` = "integrated cluster N (top marker mouse X, human Y)". The tables give no names, and I did not hand-annotate clusters.
- Sheet 3 symbols are mostly uppercased mouse genes; mixed-case ones are kept as published. `symbol_species` is mouse_uppercased or mouse.
- The paper says 229 of the 885 human genes lack mouse orthologs. However, the Homologene_ID column is filled for all rows, so non-1:1 cases cannot be flagged from this table; the ortholog map downstream will decide.

**Script.** `derive/derive_kidney_kim2024.py <tsv_dir> <out_tsv>`
