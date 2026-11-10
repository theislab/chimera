# kidney_text: verbatim text statements of human vs mouse kidney cell-type expression differences

**Sources.** Full texts are archived in `chimera/data/analysis/E6/kidney_text/` with MANIFEST.tsv. The script asserts that each quote occurs verbatim in the archived file, after tag stripping and whitespace normalisation.
1. **He B et al. 2021**, *Nat Commun* 12:2141. DOI 10.1038/s41467-021-22331-9, verified (302 to nature.com). File: Europe PMC fullTextXML of PMC8035407. Adult glomerulus.
2. **Kim S et al. 2024**, *Dev Cell* 59:2912. DOI 10.1016/j.devcel.2024.07.013, verified. Quotes are taken from the bioRxiv v1 full text (10.1101/2023.05.16.540880; doi.org resolves to biorxiv, which returned 429 to the HEAD request). Developmental.
3. **Lindström NO et al. 2018**, *JASN* 29:785. DOI 10.1681/ASN.2017080887, verified. File: PMC5827606 html. Developmental.

`mcevoy2022_PMC9741629_fullText.xml` (McEvoy et al. 2022 *Nat Commun*, DOI 10.1038/s41467-022-35297-z) was archived but **not used**. Its only human-mouse statements concern sex-biased PT genes ("the overlap in sexually dimorphic PT genes between human and mouse is small"), not species differences in expression.

**Rule.** Every gene named in each cited sentence gets one row. No selection within a sentence. The quote, paper location and archived file are in `note`.
- Set = `text_known_difference`, or `text_conserved` for genes the sentence calls conserved (Lindström CUBN / LRP2 / AQP2).

**Counts (unique genes = 25).**
| Set | Direction | n | Genes |
|---|---|---|---|
| text_known_difference | up_in_human | 16 | PDGFRB, NR1H3, COL6A1, PLA2R1, IL13RA1, RXFP1, GIMAP7, PLA1A, APLNR, NFASC, UNC5B, NRXN1, ZMAT4, NTNG1, NR3C2, NPAS2 |
| text_known_difference | up_in_mouse | 4 | NR2F2, DES, EYA4, FGF1 |
| text_known_difference | divergent | 2 | UMOD, SLC12A1 (LoH overlap domain) |
| text_conserved | conserved | 3 | CUBN, LRP2, AQP2 |

**Notes.**
- NR1H3 / NR2F2 refer to SCENIC regulon activity, not expression.
- NTNG1's cell type follows the sentence grouping (podocytes); the figure arrow marks an S-shaped body.
- GIMAP7 belongs to the GIMAP family, so 1:1 orthology should be checked.
- Many He genes also appear in kidney_he2021 (table) and Kim genes in kidney_kim2024. Deduplicate downstream if pooling.
- **Circularity: no** for all three sources.

**Script.** `derive/derive_kidney_text.py <text_dir> <out_tsv>`
