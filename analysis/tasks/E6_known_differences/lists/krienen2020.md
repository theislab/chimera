# krienen2020: derivation card

**Citation.** Krienen FM, Goldman M, Zhang Q, et al. Innovations present in the primate interneuron repertoire. *Nature* 586, 262–269 (2020). DOI 10.1038/s41586-020-2781-z.

**DOI check (2026-11-03).** `curl -sIL https://doi.org/10.1038/s41586-020-2781-z` gives 302 to `https://www.nature.com/articles/s41586-020-2781-z`, then the cookie redirects, then HTTP 200. The page's `citation_title`, volume 586 and pages 262–269 match.

**Data.** snRNA-seq / Drop-seq of cortical interneurons in ferret, mouse, marmoset, macaque and human. This is not our M1 data, so `circular = no`.

**Files.** All files are in `data/analysis/E6/krienen2020/`; see `MANIFEST.tsv`. This folder was new and was not shared with another agent at the time.
- `41586_2020_2781_MOESM1_ESM.pdf` is the reporting summary.
- `MOESM2_ESM.xlsx` is **ST1**: per-species expression of the four major classes (PVALB, SST, ID2/LAMP5, VIP and OTHER.CGE). It is converted in `converted/`.
- `MOESM6_ESM.xlsx` is **ST5**: scaled expression of the Liger-integrated marmoset and mouse clusters. It is converted in `converted/`.

**Script.** `derive/derive_krienen2020.py <converted_dir> <out_tsv>` writes `lists/krienen2020.tsv`.

## Status: no mechanical DE table exists, so the list is text-only

- **ST1 and ST5 have no labels or cut-offs.** Neither table carries a conserved/divergent label or a threshold.
- **The Methods give no divergence cut.** The cortical comparison ("Identification of conserved and divergent genetic programs") filters candidate markers: at least 10 per 100k in each species, and more than 3-fold max/min across the four types in at least one species. It then computes pairwise-species Pearson r across the four classes. Fig. 1g shows the distribution of r, but no r threshold for "divergent" is given.
- **The ED Fig. 6 genes were picked by eye.** The divergent genes in ED Fig. 6 are exemplars "inspected to identify" by the authors. They appear only in the figure, and ST5 does not flag them.
- **So no rule was invented.** Inventing a cut-off would break the "paper's own definitions" rule. The list therefore holds only the genes the text names verbatim. Each quote is in `note` (citation superscripts dropped, em dashes written as " - ").
- **ST1 is a presence check only.** The script reads ST1 only to record whether each gene appears in the human, mouse and marmoset sheets (`in_SupplTable1` in `note`). It does not use ST1 to pick or orient genes.

## Set: `primate_vs_mouse_divergent` (12 rows, 11 unique genes case-insensitive)

| genes | cell_type | direction | quote (abridged; full in `note`) |
|---|---|---|---|
| NTNG1, IQGAP2, TRPC3 | four major classes (PVALB, SST, VIP, LAMP5) | divergent | "numerous genes with categorical changes in cell-type-specific expression patterns (including NTNG1, IQGAP2 and TRPC3)" (Fig. 1e) |
| NRG1 | four major classes | divergent | "highly correlated among humans, macaques and marmosets but quite distinct in mice and ferrets" (Fig. 1f) |
| Calb1, Calb2, Htr3a, Reln, Cck, Nos1 (mouse symbols as published) | Liger marmoset–mouse subtypes | divergent | "markers that are widely used to identify specific subtypes of interneurons in mice (such as Calb1, Calb2, Htr3a, Reln, Cck and Nos1) had different patterns of expression across the corresponding subtypes in primates" (ED Fig. 6c) |
| CALB2 | additional interneuron types | up_in_primate | "broader expression of CALB2 by additional interneuron types in primates" (ED Fig. 6c) |
| NKX2-1 | LAMP5+LHX6+ neocortical interneurons | up_in_primate | "marmoset neocortical LAMP5+LHX6+ interneurons selectively expressed NKX2-1 … in the mouse … is downregulated in MGE-derived interneurons destined for the neocortex" |

## Assumptions and oddities

- **Direction value `up_in_primate`.** It is used because the contrast is primates against mouse, not human alone. `contrast` names the species involved.
- **CALB2 appears twice.** It is once as unsigned `Calb2` (mouse symbol) and once as signed `CALB2`. These are two separate statements, so both are kept.
- **The NKX2-1 mouse side is not measured here.** It is prior knowledge the paper cites (refs 37 and 38), not a mouse contrast measured in this paper. NKX2-1 is absent from the mouse ST1 sheet.
- **Some statements were left out.**
  - **Out of scope:** the striatal statements (TAC3+ type, CALB1 human-specific in striatal interneurons, ST6), because they are not cortex.
  - **Not gene-level:** the LAMP5+LHX6+ abundance difference, because it is a proportion, not gene expression.
- **The list is small.** It is a curated-by-authors text set, not a genome-wide list. Its statistical power is low.
- **A fuller list would need a new threshold.** ST1 could support a genome-wide primate-vs-mouse divergence list, for example by r below some cut. That would need a threshold the paper does not give, so it was not done.
