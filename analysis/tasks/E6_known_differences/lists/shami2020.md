# shami2020: derivation card

**Citation.** Shami AN, Zheng X, Munyoki SK, Ma Q, Manske GL, Green CD, Sukhwani M, Orwig KE, Li JZ, Hammoud SS. Single-cell RNA sequencing of human, macaque, and mouse testes uncovers conserved and divergent features of mammalian spermatogenesis. *Developmental Cell* 54(4), 529–547.e12 (2020). DOI 10.1016/j.devcel.2020.05.010. PMID 32504559, PMCID PMC7879256.

**DOI check (2026-11-03).** `curl -sIL https://doi.org/10.1016/j.devcel.2020.05.010` gives 302 to `https://linkinghub.elsevier.com/retrieve/pii/S1534580720303993`, then HTTP 200. Europe PMC lists the same DOI with this title, volume 54, issue 4, pages 529–547.e12.

**Data and circularity.** The data are Drop-seq (human, macaque) and the authors' own earlier mouse Drop-seq (Green et al. 2018). Our testis cohorts use Murat et al. 2023, which generated its own snRNA-seq. So `circular = no` for every row.

**Files.** All files are in `data/analysis/E6/shami2020/`; see `MANIFEST.tsv` for URLs and sha256.
- The PMC `bin/` links return a proof-of-work page and Europe PMC answers "not open access". The tables were therefore taken from the Elsevier CDN (`ars.els-cdn.com`, `…-mmcN.xlsx`). File sizes match PMC's listing: mmc2 = Table S1, mmc4 = S3, mmc5 = S4, mmc6 = S5, mmc7 = S6.
- Converted to TSV with `xlsx_to_tsv.py` into `converted/`.
- `PMC7879256_fulltext.html` is the author manuscript text. It is used to check every quote verbatim.

**Script.** `derive/derive_shami2020.py <data_dir> <out_tsv>` writes `lists/shami2020.tsv` (9,180 rows, 3,815 unique symbols when case is ignored).

## Sets and rules (the paper's own; nothing tuned)

### 1. Table S5 A–C: pairwise germ-cell k-means (Figure 5A, S5A)

**What the table holds.**
- Per species, k-means (k = 6) clusters the highly variable 1-1-1 orthologues by their profile over 21 ordered germ states (SPG + 20 aligned GC stages).
- A gene that is not highly variable in a species is assigned to a cluster if r > 0.8 with a cluster centroid. Otherwise it is `NULL`.
- Cluster numbers K1–K6 are ordered early to late and are comparable across species. Checked: a human gene's cluster is identical in sheets A and C for all 2,161 genes that are non-NULL in both.

**Classification.** The three classes below are the paper's own ("in-phase", "phase-shifted", "unique"; Results and Fig S5A).

| set | rule | paper term |
|---|---|---|
| `conserved_control` (direction `conserved`) | same cluster in both species | in-phase |
| `species_divergent` (direction `divergent`) | different clusters | phase-shifted |
| `species_specific` (direction `divergent`) | cluster in one species, `NULL` in the other | unique |

**Column conventions.**
- `cell_type` = `germ_kmeans_K<a>|K<b>`: the first species' cluster, then the second's (`NA` = NULL).
- `effect` holds both cluster numbers.
- `gene_symbol` is the first species' symbol. The second species' symbol is in `note`.

**Counts (rows = unique genes within a contrast).**

| contrast | conserved_control | species_divergent | species_specific |
|---|---|---|---|
| human_vs_macaque | 1,133 | 737 | 774 |
| human_vs_mouse | 847 | 1,151 | 1,239 |
| macaque_vs_mouse | 945 | 1,065 | 758 |
| unique genes, all contrasts | 1,916 | 1,902 | 1,863 |

**By first-species cluster (K1…K6, NA):**
- human_vs_mouse
  - conserved: 205 / 180 / 142 / 46 / 132 / 142
  - divergent: 164 / 321 / 261 / 112 / 160 / 133
  - specific: 44 / 79 / 77 / 68 / 86 / 138, plus 747 NA
- human_vs_macaque
  - conserved: 230 / 226 / 204 / 78 / 193 / 202
  - divergent: 86 / 208 / 152 / 70 / 108 / 113
  - specific: 22 / 104 / 72 / 62 / 72 / 84, plus 358 NA
- macaque_vs_mouse
  - conserved: 261 / 178 / 166 / 55 / 142 / 143
  - divergent: 256 / 239 / 204 / 99 / 150 / 117
  - specific: 24 / 47 / 33 / 41 / 59 / 74, plus 480 NA

**Caveats.**
- This is a divergence in the *timing* of a gene along the germ-cell trajectory, not an up/down level difference in one cell type, so the direction is unsigned.
- `species_specific` also catches genes that are merely low or flat in the other species ("not highly variable, or not highly expressed", Methods). It is the weakest of the three classes.

### 2. `conserved_marker_spg` (Table S3A; 432 rows, 415 unique genes)

- **Source.** Seurat `FindConservedMarkers` (minimump meta-analysis) markers of the six consensus spermatogonia states, three species, 1-1-1 orthologues (human symbols).
- **Thresholds (caption).** ≥ 10% detection difference and ≥ 1.3-fold.
- **Genes per state.** SPG1 34, SPG2 33, SPG3 23, SPG4 54, SPG5 62, SPG6 226.
- **Effect column.** Per-species logFC and adjusted p.
- **Caveat.** Some genes have padj = 1 in one species, because the meta-analysis does not require significance everywhere. The paper calls these markers "detected using meta-analysis for three species". Filter on `effect` if a stricter "conserved in all three" is wanted.
- The paper cites S3A for its SPG6 species-specific examples too. None of those text genes is in S3A.

### 3. `primate_msci_escapee` (Table S1E; human 16, macaque 14 rows; 23 unique symbols)

- **Rule (paper).** X-linked genes with mean normalized expression > 0.5 in spermatocytes and > 2-fold over spermatogonia.
- **Why it is a species difference.** "Using these criteria we identified 16 human and 14 macaque genes that escape MSCI, whereas the same threshold in mice identified a single gene, Tsga8."
- **Rows.** `cell_type = spermatocyte`, contrast `human_vs_mouse` or `macaque_vs_mouse`, direction `up_in_human` or `up_in_macaque`.
- **Symbols.** Five macaque entries are Ensembl IDs (ENSMMUG…), as published.
- Tsga8 (the mouse escapee) is not listed, because no human or macaque contrast is stated for it.

### 4. Text statements (`text_known_difference`: 62 rows, 31 genes; `text_conserved`: 7)

Every quote is in `note` and is asserted verbatim against the full text.

**SPG6** (Results, "Consensus SPG6"; the quote cites Table S3A):
- Expressed in all three species, so `text_conserved`: FMR1NB, ZCWPW1, DPEP3, IQBP1, CALR.
- Primates only: BEND2, ZRANB2, PAGE4, ZNFX1, HLTF, ZBED5. Rows are `up_in_human` vs mouse and `up_in_macaque` vs mouse.
- Mouse only: Swt1, Tgfbr1, Rhox13, PhTF1, Fmr1. Rows are `up_in_mouse` vs human and vs macaque.
- Macaque only: ZNF850, MAGEB17, ZMYM1, TCEA1.
- Human only: PRDM7, "full VCX family", ERBB3, ERVK13-1, SSX3, TRIML2.
  - **Assumption.** "full VCX family" was expanded to VCX, VCX2, VCX3A and VCX3B (flagged in `note`).
- PhTF1 is kept as printed. The MGI symbol is Phtf1.

**Other spermatogonia statements.**
- **SPG2.** L1TD1, "macaque and human only", is up in primates vs mouse. GFRA1, "all three species", is `text_conserved`.
- **SPG1.** PIWIL4, "a SPG1-specific marker for all three species", is `text_conserved`.
- **SPG3.** MKI67, "upregulated in SPG3 of mice and macaques, but not until SPG4 in humans", gives `up_in_mouse` (vs human) and `up_in_macaque` (vs human). This is a timing statement.

**Peritubular myoid cells.**
- "Unlike in mouse, primate myoid cells don't express … GDNF, but rather express … PDGFRA, EGFR, PTCH1, and PPARA."
- GDNF is `up_in_mouse`. The other four are up in human and up in macaque, each vs mouse.

**Not used.** "ZBTB16 (mouse and human)" in SPG2 is not a stated absence in macaque (it may be unannotated there), so it was not used.

## Not used
- Table S4 (stage markers per species; no cross-species rule).
- Table S6 (somatic markers, human and macaque only; no mouse).
- Table S7 (ligand–receptor scores).
- Table S5 regulon sheets: these are regulons, not genes. The 5 "divergent" regulons are only in Fig S5B.
