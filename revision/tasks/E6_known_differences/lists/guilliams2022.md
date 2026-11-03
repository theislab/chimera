# guilliams2022: Kupffer-cell markers conserved vs specific to human or mouse (+ text rows)

**Citation.** Guilliams M, Bonnardel J, Haest B, Vanderborght B, Wagner C, et al. Spatial proteogenomics reveals distinct and
evolutionarily conserved hepatic macrophage niches. *Cell* 185(2):379–396.e38 (2022). DOI 10.1016/j.cell.2021.12.018;
PMID 35021063; PMC8809252. DOI checked 2026-11-03 with `curl -sIL https://doi.org/10.1016/j.cell.2021.12.018`
(→ linkinghub.elsevier.com/retrieve/pii/S0092867421014811, HTTP 200).

**Files** (`chimera/data/revision/E6/guilliams2022/`, see MANIFEST.tsv): `PMC8809252_supplementaryFiles.zip` from the Europe PMC
supplementary endpoint. From it I unzipped `extracted/mmc9.xlsx` (Table S9), `extracted/mmc2.xlsx` (Table S2; kept for reference,
not used) and figures S6, S7 and 4, then converted the xlsx files with `xlsx_to_tsv.py` into `extracted/tsv/`.
Input to the list: `extracted/tsv/mmc9__Conserved_Unique_KC_DEGs.tsv`.

**What the sheet is.** The Figure S7J legend reads: "Venn diagram showing convergence and divergence of expression of top 50 KC genes per species
across species, see Table S9 for genes lists per species." The sheet lists each Venn region (species set) with its genes, for
human, mouse, pig, hamster, chicken and zebrafish (no macaque). Main text: "A strong overlap in transcriptomes across species was
observed likely due to the conserved expression of core KC transcription factors ( Figure S7 I). However, each species also
harbored a number of unique KC genes ( Figure S7 J; Table S9 )."

**Mechanical rule** (`derive/derive_guilliams2022.py`), applied to human vs mouse only:
- region contains human and mouse → `conserved_control`
- human but not mouse → `human_enriched` (`up_in_human`)
- mouse but not human → `mouse_enriched` (`up_in_mouse`)
- regions with neither species are dropped.
"Up" means the gene is in that species' KC marker list (KC vs other liver cells) and not in the other species' list. It does
not show that the gene is absent from the other species' KCs. Symbols are kept as published: human-containing regions use human
symbols, mouse-only regions use mouse symbols (e.g. `Clec4f`).

**Text rows** (`text_known_difference`). Verbatim from Results ("Potential role for CD45− hepatic cells in determining macrophage localization"):
"This analysis identified CCL23 and CCL14 expression by LSECs and CCL16 by hepatocytes that binds to CCR1 expressed by human KCs
but not murine KCs ( Figure 6 E)." The Figure 6E legend calls these "ligand-receptor pairs ... uniquely expressed in the human liver".
I took all four genes the sentence names: CCR1 (KC, up_in_human), CCL23 and CCL14 (LSEC), and CCL16 (hepatocyte). CCL14, CCL16 and CCL23
have no 1:1 mouse ortholog (noted in each row), so they will drop out downstream.

**Counts** (unique genes): conserved_control 95, human_enriched 28, mouse_enriched 31, text_known_difference 4.
Genes also in the earlier Y2 text list (flagged `overlap_Y2` in note): C1QB, CD163, CD5L (conserved_control), Clec4f (mouse_enriched), CCR1 (text).

**Discrepancy.** The legend says "top 50 KC genes per species", but the sheet holds 123 human and 126 mouse genes (95 shared). I used the
table as published. The ranking or threshold behind the lists is not stated beyond "top KC-specific genes compared with other cells per species".

**Circularity.** `no`. Guilliams' human data (19 biopsies, Ghent) and mouse data (their own C57BL/6 CITE-seq/sc/snRNA-seq) are
neither Tabula Sapiens nor Tabula Muris Senis. Check that the CAMEX liver cohorts do not include Guilliams data (livercellatlas.org,
GSE192742). If they do, change these rows to `possible`.

**Not extracted.** The 15-gene "human-mouse conserved KC signature" (Figure 4K/S7A–B) is printed only in figure panels whose text cannot be read at the
available resolution. Per-cell-type conserved markers (Figure S7N/O) are not extracted for the same reason.

## Circularity (coordinator, at commit)
The canonical liver cohorts are built from Liver Cell Atlas samples of this paper (human CS170, mouse ABU20; see the X2 ledger). This list is therefore treated as **circular** in `assign.tsv` (`guil_circ`), whatever the `circular` column says.
