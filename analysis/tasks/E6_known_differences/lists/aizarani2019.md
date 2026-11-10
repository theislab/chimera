# aizarani2019: conserved vs non-conserved liver zonation, human vs mouse

**Citation.** Aizarani N, Saviano A, Sagar, Mailly L, Durand S, Herman JS, Pessaux P, Baumert TF, Grün D.
A human liver cell atlas reveals heterogeneity and epithelial progenitors. *Nature* 572, 199–204 (2019).
DOI 10.1038/s41586-019-1373-2; PMID 31292543; PMC6687507.
DOI checked on 2026-11-03 with `curl -sIL https://doi.org/10.1038/s41586-019-1373-2`, which redirects to
nature.com/articles/s41586-019-1373-2 (HTTP 200).

**Files** (`chimera/data/analysis/E6/aizarani2019/`, see MANIFEST.tsv). The `.xls` files are plain tab-delimited text.
- `41586_2019_1373_MOESM8_ESM.xls`: Supplementary Table 6, "Conservation of hepatocyte zonation between mouse and human"
  (967 ortholog pairs; columns HumanGeneID, HumanModule, HumanPadj, MouseGeneID, MouseModule, MousePadj, ProfileCorrelation).
  Mouse data: Halpern et al. 2017 (Nature 542:352, GSE84498), re-analysed by the authors and binned into 9 zones.
- `41586_2019_1373_MOESM9_ESM.xls`: Supplementary Table 7, "Conservation of liver endothelial cell zonation between mouse and human"
  (494 pairs; mouse p-values taken from Halpern et al. 2018 Nat Biotechnol, 4 layers). Human endothelial cells = MaVECs + LSECs.
- `41586_2019_1373_MOESM1_ESM.pdf`: Supplementary Notes (Note 3 gives the rule). `MOESM4`/`MOESM6` (human-only zonation tables 2 and 4) were downloaded for reference and are not used.

**Paper's definition (Supplementary Note 3, verbatim).** "A correlation analysis of zonation patterns revealed only a limited
degree of evolutionary conservation, comparable between endothelial cells and hepatocytes: only 68% (60%) of genes with
significant zonation patterns exhibited a positive correlation of zonation profiles for hepatocytes (endothelial cells)
(Extended Data Fig. 3k,l and Supplementary Data Tables 6,7), suggesting widespread evolutionary changes in zonation patterns."
Methods: "Conservation of zonation was assessed based on the Pearson's correlation of zonated expression profiles after
binning the human data into nine equally-sized bins akin to the nine zones derived in Halpern et al."

**Mechanical rule** (`derive/derive_aizarani2019.py`). The paper does not print the significance cut-off in Note 3. I tested
the obvious readings against the stated percentages; "significant = human adj. P < 0.05 OR mouse P < 0.05" is the one that reproduces
both of them (hepatocytes 543/801 = 67.8% → "68%"; endothelium 256/423 = 60.5% → "60%"). Other readings (human only, mouse only,
both, cut-off 0.01 or 0.1) give 69–78% for hepatocytes. Then:
- `conserved_control`: significant and r > 0 (`direction=conserved`)
- `nonconserved_zonation`: significant and r ≤ 0 (`direction=divergent`)
- `zonation_not_significant`: all other rows, kept for completeness (direction from the sign of r); exclude them downstream.

**Counts** (rows = unique human genes within each cell type).

| cell_type | nonconserved_zonation | conserved_control | zonation_not_significant |
|---|---|---|---|
| hepatocyte | 258 | 543 | 166 |
| liver endothelial cell (MaVEC+LSEC) | 167 | 256 | 71 |

**Circularity.** `no`. The human data are Aizarani's own (9 donors, CEL-seq2; not Tabula Sapiens) and the mouse data are
Halpern 2017/2018 (not Tabula Muris Senis). Check whether any of our CAMEX liver cohorts includes Aizarani (GSE124395) or
Halpern (GSE84498/GSE108561) samples. If one does, these rows become `possible`.

**Caveats.**
- The contrast is the shape of zonation along the porto-central axis (the correlation of spatial profiles), not a difference in mean expression in
  a cell type. A gene can be "nonconserved_zonation" and still have the same mean level in both species. Use this list only for
  readouts about zonation or within-hepatocyte state. For mean-level readouts, treat it as weak evidence.
- Mouse orthologs come from the authors (identical symbol after capitalisation, plus mutual best blastn hit). The table lists 1:1 pairs only.
- Low-expressed genes have noisier correlations (Extended Data Fig. 3d), so some r ≤ 0 calls are noise.
