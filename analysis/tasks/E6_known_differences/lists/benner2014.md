# benner2014: derivation card

**Citation.** Benner C, van der Meulen T, Cacéres E, Tigyi K, Donaldson CJ, Huising MO. The transcriptional landscape of mouse beta cells compared to human beta cells reveals notable species differences in long non-coding RNA and protein-coding gene expression. *BMC Genomics* 15, 620 (2014). DOI 10.1186/1471-2164-15-620. PMID 25051960, PMCID PMC4124169.

**DOI check (2026-11-03).** `curl -sIL https://doi.org/10.1186/1471-2164-15-620` gives 302 to `bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-15-620`, then 301 to `link.springer.com/article/10.1186/1471-2164-15-620`. Europe PMC lists the same DOI with this title.

**Data and circularity.**
- **Mouse.** FACS-purified beta cells from the authors' reporter mice, bulk RNA-seq (GSE54973, GSE58384).
- **Human.** Published sorted human beta-cell RNA-seq (their ref. 16, Nica et al. 2013).
- Neither is Hrovatin 2023 or the 48-donor CELLxGENE human islet atlas (10x, HPAP donors), so `circular = no`.

**Files.** All files are in `data/analysis/E6/benner2014/`; see `MANIFEST.tsv`.
- Additional file 8 (`…MOESM8_ESM.xlsx`): the common beta-cell transcriptome.
- Additional file 9 (`…MOESM9_ESM.xlsx`): species-enriched or species-unique genes.
- Both were converted with `xlsx_to_tsv.py` into `converted/`.
- `PMC4124169_fullTextXML.xml` (Europe PMC) is the article text.

**Script.** `derive/derive_benner2014.py <data_dir> <out_tsv>` writes `lists/benner2014.tsv`. Every row has `cell_type = beta` and `contrast = human_vs_mouse`.

## Sets and rules (the paper's own)

### 1. `mouse_enriched` (725 genes) and `human_enriched` (815 genes), from Additional file 9

**Rule (Results).** "we focused only on the genes that demonstrated both significant enrichment (p-value < 1 × 10⁻⁷, FDR < 0.1%) and over 10-fold difference in beta cells of one species over the other".
- `status = unique` means the other species has RPKM < 1.
- `status = enriched` means the other species has RPKM > 1.

**Counts match the paper exactly.**

| set | unique | enriched | total |
|---|---|---|---|
| mouse_enriched | 569 | 156 | 725 |
| human_enriched | 666 | 149 | 815 |

**Check against the table.** Every row has adj. p < 1e-7, and `status` agrees with RPKM < 1 or > 1. Four rows sit at about 9.95–10.0-fold on raw RPKM: Rapgef4, Mdm1, Rarg and Pacsin3. The paper's fold presumably uses a pseudocount. These four are kept as published.

**Column conventions.**
- `direction`: `up_in_mouse` or `up_in_human`.
- `effect`: both RPKMs, log fold and adj. p.
- `note`: status and accession.

### 2. `conserved_control` (9,179 rows, 9,178 unique symbols), from Additional file 8

- **Rule.** The paper's "common beta cell transcriptome", genes with RPKM > 1 in beta cells of both species.
- **Removed.** 295 genes that are also in Additional file 9. They are species-enriched by the paper's own rule even though they pass RPKM > 1 in both. Iapp is one example.
- **Discrepancy.** The file has 9,474 genes, while the text says 9,905. The file is used as published.

## Symbols and assumptions
- **Species of the symbols.** Symbols are mouse (the human data were mapped to mouse orthologs by the authors), so `symbol_species = mouse`. No human symbol is given, except "Ins2/INS" in file 8, which is kept verbatim.
- **Date-converted symbol.** The symbol `40238` appears twice in file 8. It is an Excel date-converted name (a Sept/March gene). It is kept and flagged in `note`, and it will not match any gene.
- **Cell-type coverage.** Only beta cells. The alpha-cell mouse data have no human counterpart table. The alpha-cell species statements from the text (Ucn3, Crh, Mafb) are in `islet_text.tsv`.
