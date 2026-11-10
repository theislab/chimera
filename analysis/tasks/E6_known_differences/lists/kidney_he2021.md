# kidney_he2021: He et al. 2021, human vs mouse glomerular cells

**Citation.** He B, Chen P, Zambrano S, et al. Single-cell RNA sequencing reveals the mesangial identity and species diversity of glomerular cell transcriptomes. *Nat Commun* 12:2141 (2021). DOI 10.1038/s41467-021-22331-9. PMC8035407.
**DOI check.** `curl -sIL https://doi.org/10.1038/s41467-021-22331-9` redirects (302) to nature.com/articles/s41467-021-22331-9. Crossref metadata matches title, journal, volume and article number.

**Data.** Smart-seq2. Human cells come from glomeruli in biopsies of 8 healthy living kidney donors (the authors' own data, EGA EGAD00001006861). Mouse cells come from C57BL/6J glomeruli. Human symbols; 1:1 homologues from Ensembl v92 only. The paper says "Genes with multiple orthologues ... were excluded."
**Circularity: no.** The data are independent of KPMP / Lake 2023 and of the CELLxGENE mouse atlas. The validation sets the paper used (Young 2018 human, Park 2018 mouse) are not inputs here.

**Files** (`chimera/data/analysis/E6/kidney_he2021/`, with MANIFEST.tsv):
- MOESM11/12/13 = Supplementary Data 8/9/10 (conserved genes for MC, podocyte, GEC). Converted to `tsv/` with xlsx_to_tsv.py.
- MOESM17 = Source Data zip. `sourcedata/source_datafile{1,2,6,9,10,11}.txt` were extracted from it.
- MOESM3 = description of supplementary files.

**Rule (paper's own, Methods "Identification of conserved and species-specific genes").**
- **species_divergent:** the genes in the species heatmaps:
  - Fig. 4c = Source Data 9 (mesangial cells, 75 genes)
  - Fig. 5c = Source Data 10 (podocytes, 80 genes)
  - Fig. 6c = Source Data 11 (GEC, 6 genes)

  Criteria: mean log2-RPKM < 0.1 in one species and ≥ 3 in the other; Kruskal–Wallis BH-adjusted P < 0.01; not species-specific in more than one glomerular cell type.
- **Direction:** recomputed from per-cell values in the same file. Cells are mapped to species through Source Data 1 and 6 (mouse) and Source Data 2 (human). All 161 rows satisfy the 3 / 0.1 rule, with no exceptions. Three GEC cells (plate 308s) are absent from Source Data 1, 2 and 6, so they were excluded from the means.
- **conserved_control:** Supplementary Data 8–10, median log2-RPKM ≥ 4 in both species.

**Counts (unique genes).**
| Set | Cell type | Direction | n |
|---|---|---|---|
| species_divergent | podocyte | up_in_human | 47 |
| species_divergent | podocyte | up_in_mouse | 33 |
| species_divergent | mesangial cell | up_in_human | 22 |
| species_divergent | mesangial cell | up_in_mouse | 53 |
| species_divergent | GEC | up_in_human | 6 |
| species_divergent | GEC | up_in_mouse | 0 |
| species_divergent | all | all | 161 unique |
| conserved_control | MC / podocyte / GEC | conserved | 243 / 318 / 170 (469 unique) |

The podocyte (47/33) and GEC (6/0) counts match the paper text exactly. For MC the text gives only the total of 75.

**Assumptions.**
- Cell type is the paper's own label.
- The Fig. 4a/5a/6a scatter counts (MC 40/70, podo 60/46, GEC 19/7) use criterion 1 only. They are not reproduced here: the heatmap sets are the paper's final species-specific lists.
- Mesangial cells correspond to glomerular pericyte-like / "mesangial cell" annotations in kidney atlases.

**Script.** `derive/derive_kidney_he2021.py <data_dir> <out_tsv>`
