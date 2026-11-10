# bakken2021: derivation card (CIRCULAR)

**Citation.** Bakken TE, Jorstad NL, Hu Q, et al. Comparative cellular analysis of motor cortex in human, marmoset and mouse. *Nature* 598, 111–119 (2021). DOI 10.1038/s41586-021-03465-8.

**DOI check (2026-11-03).** `curl -sIL https://doi.org/10.1038/s41586-021-03465-8` gives 302 to `https://www.nature.com/articles/s41586-021-03465-8`, then the cookie redirects, then HTTP 200. The page's `citation_title`, volume 598 and pages 111–119 match.

**Circularity.** These are the same M1 10x v3 data as our M1 cohorts, so every row has `circular = yes`. Use this list only as a contrast arm.

**Files.** All files are in `data/analysis/E6/bakken2021/`; see `MANIFEST.tsv`.
- `41586_2021_3465_MOESM1_ESM.pdf` holds the legends for ST1–23.
- `MOESM4_ESM.zip` holds ST1–23.
- The tables used were unzipped to `extracted/`; their sha256 is appended to the MANIFEST.
- ST20–22 were converted to `converted/` with `xlsx_to_tsv.py`.

**Script.** `derive/derive_bakken2021.py <extracted_dir> <converted_dir> <out_tsv>` writes `lists/bakken2021.tsv`.

## Paper's rules (Methods, quoted)

- **ST7 (GABAergic) and ST10 (glutamatergic).** "Positive DEGs were then found using Seurat's FindAllMarkers function, using the ROC test with default parameters (min.pct = 0.1, AUROC threshold = 0.7). We compared each subclass within species to all remaining nuclei in that class". The column `conservation` names the species in which the gene is a subclass marker.
- **ST20.** Chandelier cells (ChC) against basket cells, ROC test per species. The flag is 1 if the gene is ChC-enriched.
- **ST21.** L5 ET against L5 IT, ROC test per species in human, macaque, marmoset and mouse. The flag is 1 if the gene is ET-enriched.
- **ST22.** 131 genes whose ET-vs-IT logFC decreases from human to macaque to marmoset to mouse, with human logFC ≥ 0.5.

## Sets

A set is named by the gene's marker status in human, marmoset and mouse (H, Mar, Mus). The direction is about subclass specificity: is the gene a marker within its class in that species? It does not compare absolute expression between species.

| set | (H,Mar,Mus) | direction | contrast |
|---|---|---|---|
| conserved_control | 1,1,1 | conserved | human_vs_mouse_M1(+marmoset) |
| primate_not_mouse_marker | 1,1,0 | up_in_human | human_vs_mouse_M1(+marmoset) |
| human_only_marker | 1,0,0 | up_in_human | human_vs_mouse_M1(+marmoset) |
| human_mouse_not_marmoset_marker | 1,0,1 | conserved | human_vs_mouse_M1(+marmoset) |
| marmoset_mouse_not_human_marker | 0,1,1 | up_in_mouse | human_vs_mouse_M1(+marmoset) |
| mouse_only_marker | 0,0,1 | up_in_mouse | human_vs_mouse_M1(+marmoset) |
| marmoset_only_marker | 0,1,0 | divergent | marmoset_vs_human_mouse_M1 |
| macaque_only_marker (ST21 only) | 0,0,0, macaque 1 | divergent | macaque_vs_human_marmoset_mouse_M1 |
| primate_gradient_L5ET_marker (ST22) | — | up_in_human | human>macaque>marmoset>mouse_M1 |

- **ST21 rows.** The set is decided by human, marmoset and mouse status only. Macaque status is kept in `note`/`effect` as `marker_in=…`.
- **ST7 and ST10 effect column.** `effect` holds the per-species AUC and avg.1/avg.2 (log SCT-normalised expression in the subclass and in the rest of its class).

## Counts

Each cell shows rows / unique genes.

| set | ST7 GABA | ST10 Glut | ST20 ChC | ST21 L5 ET | ST22 | all: unique genes |
|---|---|---|---|---|---|---|
| conserved_control | 186 / 167 | 228 / 201 | 25 | 67 | – | 349 |
| primate_not_mouse_marker | 191 / 184 | 292 / 267 | 36 | 76 | – | 452 |
| human_only_marker | 494 / 472 | 925 / 803 | 129 | 229 | – | 1,244 |
| human_mouse_not_marmoset_marker | 102 / 101 | 186 / 167 | 32 | 38 | – | 292 |
| marmoset_mouse_not_human_marker | 69 / 66 | 94 / 92 | 4 | 26 | – | 172 |
| mouse_only_marker | 543 / 502 | 1,045 / 884 | 96 | 156 | – | 1,264 |
| marmoset_only_marker | 300 / 291 | 462 / 419 | 35 | 141 | – | 698 |
| macaque_only_marker | – | – | – | 120 | – | 120 |
| primate_gradient_L5ET_marker | – | – | – | – | 131 | 131 |

There are 6,458 rows in total.

Each subclass is in the TSV, and the derivation script prints the full subclass × set crosstab. Two examples:
- ST7 conserved markers per subclass: Lamp5 29, Pvalb 41, Sncg 17, Sst 24, Sst Chodl 54, Vip 21.
- ST10 conserved markers per subclass: L2/3 IT 24, L5 ET 66, L5 IT 6, L5/6 NP 32, L6 CT 13, L6 IT 9, L6 IT Car3 47, L6b 31.

## Checks against the paper text (all pass)

- **GABAergic conserved markers.** "17 to 54 subclass markers were conserved" matches ST7: the per-subclass minimum is 17 and the maximum is 54.
- **Glutamatergic conserved markers.** "6–66 conserved markers" matches ST10: minimum 6, maximum 66.
- **Chandelier DEGs.** "357 DEGs in ChCs versus basket cells … only 25 DEGs were conserved across all three species" matches ST20: 357 rows, 25 rows with 1,1,1.
- **L5 ET gradient genes.** "131 genes": all 131 ST22 rows are monotone (human ≥ macaque ≥ marmoset ≥ mouse), and the minimum human logFC is 0.501.

## Assumptions and oddities

- **Markers, not cross-species DE.** These are marker-status lists: within-species subclass-vs-rest. They are not direct human-vs-mouse DE of the same subclass. The paper publishes no direct between-species DE table per subclass.
- **The astrocyte DE table is not published.** The ILA astrocyte human-vs-mouse DEGs (560; Wilcox, FDR < 0.01, logFC > 2) appear only in ED Fig. 5e, with no supplementary table. They are not included.
- **ST21 has macaque-only rows.** 120 ST21 genes are ET markers only in macaque; their set is `macaque_only_marker`.
- **Symbols.** All gene symbols are human and orthologs as published, so `symbol_species = human`.
