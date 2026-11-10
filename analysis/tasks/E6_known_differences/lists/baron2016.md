# baron2016: derivation card

**Citation.** Baron M, Veres A, Wolock SL, Faust AL, Gaujoux R, Vetere A, Ryu JH, Wagner BK, Shen-Orr SS, Klein AM, Melton DA, Yanai I. A single-cell transcriptomic map of the human and mouse pancreas reveals inter- and intra-cell population structure. *Cell Systems* 3(4), 346–360.e4 (2016). DOI 10.1016/j.cels.2016.08.011. PMID 27667365, PMCID PMC5228327.

**DOI check (2026-11-03).** `curl -sIL https://doi.org/10.1016/j.cels.2016.08.011` gives 302 to `https://linkinghub.elsevier.com/retrieve/pii/S2405471216302666`, then HTTP 200. Europe PMC lists the same DOI with this title.

**Is there a table?** No species-comparison table exists.
- The supplement is one PDF (`1-s2.0-S2405471216302666-mmc1.pdf`, downloaded). Per the PMC captions, its tables are S1 (donor age, BMI and sex), S2 (marker genes per cell type) and S3 (transcription factors of Fig 3C, with literature).
- The human-vs-mouse comparison is only in Fig 2E–F (correlation plots) and Fig 3A/3C, so the list is **text only**.
- Europe PMC has no supplementary bundle for this article (not open access).

**Circularity.**
- Baron's data are inDrop (GSE84133).
- Our human islet cohort is the CELLxGENE dataset 464a2c70 ("Integrated transcriptomes of the 14 pancreatic islet cell types", collection 58e85c2f, DOI 10.1038/s44318-026-00744-w). It holds 48 donors (HPAP `HP…` and `SAMN…` IDs), assays 10x 3' v2/v3 only.
- Our mouse cohort is Hrovatin 2023 (10x).
- Neither contains Baron's inDrop cells, so `circular = no`. This was checked from the CELLxGENE curation API metadata only; no project data were opened.

**Files.** `data/analysis/E6/baron2016/`: `PMC5228327_fulltext.html` (author manuscript text) and the supplement PDF; see `MANIFEST.tsv`.

**Script.** `derive/derive_baron2016.py <data_dir> <out_tsv>` writes `lists/baron2016.tsv` (15 rows). Every quote is asserted verbatim against the full text and is reproduced in `note`. `symbol_species = human`, `contrast = human_vs_mouse`.

## Rows

**`text_known_difference`** (10 rows, 6 genes):

| gene | cell type | direction | basis |
|---|---|---|---|
| UCN3 | alpha, gamma, epsilon | `up_in_human` | "Ucn3 is restricted to beta cells in mice … human cell types that express UCN3 … include all human endocrine cells except delta cells" |
| FFAR4 | alpha, gamma | `up_in_mouse` | human beta and delta only; mouse alpha, beta, gamma and delta; "a key difference between mice and humans" |
| IAPP | delta | `up_in_mouse` | "Iapp is more promiscuously expressed in mouse cells, with detectable expression in both beta and delta cells" |
| LEPR | delta | `up_in_human` | see LEPR note below |
| LEPR | beta | `up_in_mouse` | see LEPR note below |
| ETV1, MEIS2 | beta | `up_in_human` | see ETV1/MEIS2 note below |

- **LEPR.** "LEPR is a delta cell-specific gene in humans", stated against "studies in mice … leptin receptor-expressing beta cells". The contrast is to the mouse *literature*, not to Baron's mouse data. Both rows are flagged.
- **ETV1, MEIS2 (flagged AMBIGUOUS).** The printed sentence reads "which had a PDX1-like expression profile in the mouse, show 'non-beta' expression in the mouse". It is read together with the human statement ("ETV1 and MEIS2 are restricted to non-alpha endocrine cells") as human beta+ vs mouse non-beta.

**`text_conserved`** (5 genes, `pan_endocrine`): NEUROD1, INSM1, ISL1, NKX2-2, PAX6. Basis: "Pan-endocrine expression was conserved in mice".

**Not used.**
- DLK1: human beta-specific, but no mouse comparison is stated.
- MAFB "(alpha and beta)" in human: no mouse statement in Baron. MAFB is covered in `islet_text`.
