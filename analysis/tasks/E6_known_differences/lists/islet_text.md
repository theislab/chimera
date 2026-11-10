# islet_text: derivation card

Textbook human-vs-mouse islet cell-type expression differences, quoted verbatim from paper text (`evidence = text`). The paper and its DOI are given in each row's `note`.

## Papers used

| key | citation | DOI | full text used |
|---|---|---|---|
| Benner2014 | Benner C et al. *BMC Genomics* 15:620 (2014) | 10.1186/1471-2164-15-620 | Europe PMC XML (in `benner2014/`) |
| Dai2012 | Dai C, Brissova M, …, Powers AC. Islet-enriched gene expression and glucose-induced insulin secretion in human and mouse islets. *Diabetologia* 55:707–718 (2012) | 10.1007/s00125-011-2369-0 | Europe PMC XML (PMC3268985) |
| SkelinKlemen2017 | Skelin Klemen M, Dolenšek J, Slak Rupnik M, Stožer A. The triggering pathway to insulin secretion: functional similarities and differences between the human and the mouse β cells and their translational relevance. *Islets* 9:109–139 (2017) | 10.1080/19382014.2017.1342022 | Europe PMC XML (PMC5710702) |
| RorsmanAshcroft2018 | Rorsman P, Ashcroft FM. Pancreatic β-cell electrical activity and insulin secretion: of mice and men. *Physiol Rev* 98:117–214 (2018) | 10.1152/physrev.00008.2017 | PMC HTML (PMC5866358) |

**DOI check (2026-11-03, `curl -sIL https://doi.org/<doi>`).** Every DOI gives 302 from doi.org to the publisher.
- 10.1186/1471-2164-15-620 → link.springer.com
- 10.1007/s00125-011-2369-0 → link.springer.com
- 10.1080/19382014.2017.1342022 → tandfonline.com, which then answers 403 to curl (bot block)
- 10.1152/physrev.00008.2017 → physiology.org (202)

Europe PMC records give the same DOI and title for each.

**Also read, not used.** Segerstolpe et al. 2016 *Cell Metab* 24:593 (10.1016/j.cmet.2016.08.020, PMC5069352). It has no sentence stating a human-vs-mouse difference in a named islet cell type; it only cites earlier rodent and human receptor-localization reports.

**Circularity.** `no` for all rows. None of these papers uses Hrovatin 2023 or the 48-donor CELLxGENE human atlas.

**Files.** `data/analysis/E6/islet_text/` (see `MANIFEST.tsv`) plus `data/analysis/E6/benner2014/PMC4124169_fullTextXML.xml`.

**Script.** `derive/derive_islet_text.py <e6_data_root> <out_tsv>` writes `lists/islet_text.tsv` (65 rows). Every quote is asserted to occur verbatim in the downloaded full text (tags stripped, whitespace collapsed).

## Rule
- **`text_known_difference`.** A gene enters only if a sentence states a human-vs-mouse difference in a named islet cell type.
- **`text_conserved`.** A gene enters if a sentence states high expression in both species' beta cells (Rorsman & Ashcroft), or the same restricted pattern in both species (Benner, Dai).
- **Gene symbols.** Where the paper names a protein or channel, the gene symbol is the standard one, and the published name is in `note` (GLUT1/2/3 → SLC2A1/2/3, R-type VDCC → CACNA1E, T-type Cav3.2 → CACNA1H, Cav1.3 → CACNA1D, Epac2 → Rapgef4, synaptotagmin 11 → SYT11). Otherwise the symbol is as printed (mouse case where the paper wrote the mouse gene).
- **One row per paper.** When two papers state the same (gene, cell type), each gets a row: CACNA1H (Skelin Klemen, Rorsman), MAFB (Benner, Dai), MAFA (Benner, Dai). The directions agree; this is asserted.

## Counts

`text_known_difference`: 44 rows, 41 genes (case-insensitive).

| cell type | direction | rows | genes |
|---|---|---|---|
| beta | `up_in_human` | 25 | 23 genes: MAFB, SLC2A1, SLC2A3, CACNA1D, CACNA1H, SCN8A, KCNB2, KCNA6, KCNMA1, CHRNA3, CHRNA5, CHRNB2, GAD2, GABRA2, BEST1, ANO5, GRIA2, GRIA3, GRIA4, SLC1A1, STX1A, SYT11, RPH3A. MAFB and CACNA1H each have two rows (two papers). |
| beta | `up_in_mouse` | 16 | IL1R1, SLC2A2, CACNA1E, Scn9a, Gad1, GABBR2, Best3, Ano6, CACNA1B, IAPP, Vamp4, Syt2, Syt3, Syt9, Syn2, Rapgef4 |
| alpha | `up_in_human` | 2 | Ucn3, Crh |
| alpha | `up_in_mouse` | 1 | Mafb |

`text_conserved` (beta): 21 rows, 20 genes. Mafa/MAFA, PDX1, KCNH2, SLC7A1, SLC7A2, SLC7A5, FFAR1, GLP1R, GIPR, KCNJ6, RPH3AL, CLCN3, RIMS2, RIMBP2, ERC1, SNAP25, VAMP2, VAMP3, CPLX2, SYT5.

## Flags kept in `note`
- **CACNA1D.** Skelin Klemen say Cav1.3 is human-only. This conflicts with Rorsman & Ashcroft, who give Cacna1d as 40% of mouse beta-cell Ca-channel transcripts.
- **Best3.** Rorsman & Ashcroft say elsewhere that Best3 is expressed in both species.
- **Gad1 / GAD2.** The printed protein names are swapped relative to HGNC. The mouse statement is for islets.
- **Scn9a and IAPP.** These are relative statements (share of Na-channel transcripts; ratio to insulin).
- **Overlap with Benner.** IL1R1, IAPP, Gad2 and Rapgef4 are also in Benner Additional file 9, with the same direction.

## Not used
- Whole-islet differences that are not cell-type statements (e.g. Dai: MAFA, MAFB, PDX1 mRNA "greater in human than mouse islets").
- Functional-only statements (e.g. BK channels "more prominent" in human).
- Table-only contrasts in Skelin Klemen Table 1 that are not stated in a sentence (TRP channels, SK channels, Kir2.1).
