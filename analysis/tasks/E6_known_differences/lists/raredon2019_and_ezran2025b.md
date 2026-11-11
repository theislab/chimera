# Checked sources that produced no list

## raredon2019: no list

Raredon MSB, Adams TS, Suhail Y, et al., Niklason LE. Single-cell connectomic analysis of adult mammalian lungs. *Sci Adv* 5(12), eaaw3851 (2019). DOI 10.1126/sciadv.aaw3851; PMC6892628.

**DOI verified.** `curl -sIL https://doi.org/10.1126/sciadv.aaw3851` gave 302 to science.org/doi/10.1126/sciadv.aaw3851 (2026-11-03).

**Why no list.** The Europe PMC supplementary zip contains only:
- Data file S1: ligand-receptor families
- Data file S2: antibodies
- the SM PDF, whose fig. S3 is a cross-species marker atlas shown as a figure only

None of these is a human-vs-mouse divergent gene table. The paper studies conserved connectivity. A full-text scan found one species-difference sentence, and it concerns which cell types were captured, not gene expression: "Certain cell types were only profiled in certain species, such as basal cells marked by KRT5 (pig and human) and Sox9 + epithelium (rat)."

Nothing was downloaded into `data/`.

## ezran2025b_tmc: downloaded, not derived

Ezran C, Liu S, et al. (Tabula Microcebus Consortium). Mouse lemur cell atlas informs primate genes, physiology and disease. *Nature* 644, 185–196 (2025). DOI 10.1038/s41586-025-09114-8.

**DOI verified.** `curl -sIL` gave 302 to nature.com/articles/s41586-025-09114-8 (2026-11-03).

**Files** (`data/analysis/E6/ezran2025b_tmc/`): `41586_2025_9114_MOESM1_ESM.pdf` and `41586_2025_9114_MOESM3_ESM.zip`. Tables 5, 6, 7 and 9 were extracted to `unz/`, and Table 5 was converted to `tsv/`.

**Why not derived.** This paper's gene-level content does not fit the test of this task:
- Its primate genes are genes missing in mice, so they have no 1:1 ortholog by definition.
- Table 5, "Expression homolog triads of human, lemur, and mouse detected by SAMap", lists conserved expression homologs only. It has 1,279 Lung rows and 1,686 Limb_Muscle rows, and contains no divergent set.

Table 5 could later serve as a lung `conserved_control`.

**Circularity.** Yes: TS, TMC and TMS data.
