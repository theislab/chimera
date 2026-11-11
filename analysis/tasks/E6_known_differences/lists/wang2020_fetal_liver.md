# wang2020_fetal_liver: species-specific cell-type genes, human vs mouse fetal liver

**Citation.** Wang X, Yang L, Wang YC, Xu ZR, Feng Y, Zhang J, Wang Y, Xu CR. Comparative analysis of cell lineage differentiation
during hepatogenesis in humans and mice at the single-cell transcriptome level. *Cell Research* 30, 1109–1126 (2020).
DOI 10.1038/s41422-020-0378-6; PMID 32690901; PMC7784864. DOI checked 2026-11-03 with `curl -sIL https://doi.org/10.1038/s41422-020-0378-6`
(→ nature.com/articles/s41422-020-0378-6, HTTP 200 after the cookie redirects).

**File** (`chimera/data/analysis/E6/wang2020_fetal_liver/`): `41422_2020_378_MOESM12_ESM.xlsx` = Supplementary Table S3,
"Genes-to-eigengenes correlations for human and mouse. Related to Supplementary information, Fig. S2d". It has one sheet per cell
type (13 sheets, 1,739–1,740 genes each). I converted it to `tsv/` with `xlsx_to_tsv.py`.

**Paper's rule** (Methods, verbatim): "Genes specifically correlated with the eigengene of one species (Pearson correlations > 0.5) but
not correlated with the other (the differences between Pearson correlations of two species > 0.5) were regarded as having
species-specific expression patterns." The comparison uses 1:1 Ensembl orthologues only ("Our cross-species comparison was based on
one-to-one orthologues annotated by the Ensembl genome annotation system"). The eigengenes are WGCNA eigengenes of cell-type-specific genes.

**Mechanical rule** (`derive/derive_wang2020_fetal_liver.py`). I took the published column `Species_specific` as is. I re-checked it against the
rule above, and on the Kupffer sheet it matches with 0 mismatches. `human` → `human_enriched` (`up_in_human`), `mouse` → `mouse_enriched` (`up_in_mouse`).
The `none` rows (the vast majority) are not exported. The paper does not define a conserved set, so the list has no conserved_control.
Text check: the paper's examples are "ORM1 / Orm1 , NRN1 / Nrn1 and LDHB / Ldhb". Orm1 is human-specific in hepatoblast/hepatocyte,
Nrn1 is mouse-specific in hepatoblast/hepatocyte, and Ldhb is human-specific in erythroid progenitor and HSC; all three are recovered.

**Counts.** 161 unique genes in total (human_enriched 105, mouse_enriched 56); 168 rows. By cell type (human/mouse): hepatoblast/hepatocyte 14/4,
cholangiocyte 7/3, Kupffer cell 12/12, liver endothelial cell 3/6, hepatic stellate cell 6/3, mesothelial 24/5,
septum transversum 5/3, myeloid/lymphoid/megakaryoid 27/14, HSC 5/0, erythroid progenitor 5/0, erythroblast 3/4,
early erythrocyte 0/1, primitive erythrocyte 0/3.

**Symbols.** The table uses mouse-style capitalisation for the orthologue pair (e.g. `Orm1` for ORM1/Orm1). Upper-case them to get the human symbol.

**Circularity.** `no`. This is the authors' own fetal data (human W5–W19, mouse E11.0–E17.5).

**Caveats.** This is FETAL liver (hepatoblasts and the hematopoietic fetal liver), and our cohorts are adult. The statistic is
species-specific co-expression with a cell-type eigengene, not a mean-expression difference. Treat it as weak, secondary
evidence for adult liver. The hematopoietic rows (HSC, erythroid, myeloid/lymphoid) could also serve the bone-marrow readouts, with the same fetal caveat.
