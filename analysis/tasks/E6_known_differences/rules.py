"""E6: does Chimera recover literature-documented species differences in cell-type gene expression? (EXPLORATORY)

Card fixed and committed before any Chimera value of any list gene was read. Vocabulary: CHIMERA_MECHANISM.md
(Δα = deployment relative to the protein prior; S_g = count evidence for a shared cell-state pattern; never
"conservation" or "selection").

Truth. Published lists of genes whose cell-type expression differs between species (lists/<source>.tsv, one card per
source in lists/<source>.md, raw supplements in data/analysis/E6/<source>/ with MANIFEST.tsv). Taken mechanically at the
authors' own thresholds, or verbatim from paper text where no table exists (set name `text_*`). Lists derived from the
same data as a cohort are marked `circular` and reported as a contrast arm, never as recovery.

Readouts (per gene, cohort tables; nothing retrained):
  D_can   canonical gate Δα, H1 table  results/analysis/H1/genes/<cohort>.parquet, column D
  S_can   canonical S_g (S_ema), same table
  D_free  per-gene gate Δα, Z4 table  results/analysis/Z4/h1/genes/<cohort>.parquet, column D
  S_free  per-gene gate S_g (reported in the supplement table only)
  B1      pseudobulk profile correlation between species across cell types (H1 table column B1), the baseline
  LVL     direct detection gap between the two species of the list's contrast, -|logit det_h - logit det_o|
          (a no-model level baseline; oriented so that "lower = more different", like the others)
  pi_raw  protein identity (sequence baseline)

Universe (primary): genes of the cohort with count support in both fits (|S_ema| >= 1e-6 in H1 and in Z4) and B1
defined. Secondary: all genes of the cohort with D_can (masked genes carry feature-imputed D_can; coverage only).

Statistic. Stratified AUROC = P(score of a list gene > score of a matched non-list gene), matched within strata of
species-balanced expression (expr, quintiles) x cell-type specificity (tau, quintiles) x protein identity (pi_raw,
tertiles): 75 strata, computed in each cohort's universe. Within each stratum with >= 1 list gene and >= 5 non-list
genes, Mann-Whitney AUROC; combined as the list-gene-weighted mean. 95% CI by bootstrap (list and non-list genes
resampled within strata, 2,000 draws); one-sided p (lower) by permuting membership within strata (2,000 draws).
Stratifying on tau is primary because DE lists are enriched for cell-type-variable genes and S_g rewards having a
pattern (T12.3, Shay); the unstratified-on-tau AUROC (expr x pi_raw only) is reported beside it to show the confound.
Where a source provides its own conserved counterpart, divergent vs conserved is computed the same way (strata on the
union).
A list enters a cohort with >= 10 genes in the universe; it counts toward the predictions with >= 30.

Large lists. A set with more than CORE_N (500) unique mapped genes also enters as <set>__core: the 500 genes called in the
most source cell types, ties by largest |log fold change| (|ln FC| for plain fold changes), then symbol (Hodge 2019: the
paper's own top-3% divergence tier, 437 genes). Shami 2020 has cluster labels only (no strength): no core, full sets read.
The core is the primary reading for large lists (thousands of DE genes cover most expressed genes and leave little
background); the full set is reported beside it. Sets with "conserved" in the name are controls, not truth; "text_*" sets
are verbatim statements. Controls in a vs-conserved comparison are the union of the source's conserved sets.
Assignment of each list to cohorts, the primary cohort, cell-type filters and the contrast species: assign.tsv
(committed with this card). Primary cohort = the cohort whose species match the paper's contrast most closely.

Predictions (before reading):
  P1  Non-circular lists sit below matched genes on S_can: AUROC < 0.5 with CI upper < 0.5 in at least half of the
      (list, primary cohort) pairs with n >= 30.
  P2  D_free follows S_can (Z4: free Δα reads S_g): same direction, |AUROC - 0.5| within 0.03 of S_can's.
      D_can is closer to 0.5 than S_can (it projects S_g onto protein features).
  P3  B1 also sits below 0.5 on these lists, and at least as far from 0.5 as S_can on most lists (labelled direct
      profile correlation leads outside liver, Z4/X6).
  P4  LVL separates level-type lists (up/down in one species within a type) more than S_can does.
  P5  Circular lists separate more strongly than independent lists in the same cohort, on every score.
  P6  Divergent below the paper's own conserved counterpart on S_can, B1 (AUROC < 0.5).
  P7  Without tau in the strata the AUROC of S_can moves up (toward or above 0.5), the T12.3 confound.

Named genes (O1 rules): a list gene is named only if nameable in the H1 table (strict support, no exact zero in a
species, no O1 flag, not quiet in one species, not a DLPFC reference artefact), sign of D and S stable over seeds in
both gates, not saturated, and in the lower or upper decile of S_can within its cohort. Text-derived textbook genes
are reported one by one with their within-stratum percentile on every score and these flags, named or not.

Cell-type check (secondary, only where the list names a cell type that maps to a cohort label without ambiguity):
the cohort type where the gene's per-type count-evidence decomposition is most negative (H1 drivers) equals the
paper's type. Chance = 1 / number of cohort types. Reported, not predicted strongly (C1 E-B4 was at chance).

Everything here is exploratory and post hoc relative to the frozen plan. No status, row, modifier or H1/Z4 table changes.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
H1_GENES = REPO / "results/analysis/H1/genes"
Z4_GENES = REPO / "results/analysis/Z4/h1/genes"
H1_DRIVERS = REPO / "results/analysis/H1/drivers"
LISTS = Path(__file__).resolve().parent / "lists"
OUT = REPO / "results/analysis/E6"
HGNC = REPO / "data/analysis/gold/G10/hgnc_complete_set.txt"

EPS = 1e-6
N_BOOT = 2000
N_PERM = 2000
SEED = 20261103
Q_EXPR, Q_TAU, Q_PI = 5, 5, 3
MIN_STRATUM_BG = 5
MIN_LIST = 10
MIN_LIST_PRED = 30
CORE_N = 500
SCORES = ["S_can", "D_free", "D_can", "B1", "LVL", "pi_raw"]

# Cohort species columns in the tables (detection_<Name>) and the species each cohort holds.
SPECIES = {
    "liv_hmouse": ["Human", "Mouse"], "liv_hmonkey": ["Human", "Monkey"], "liv_4sp": ["Human", "Monkey", "Mouse", "Zebrafish"],
    "test_hmouse": ["Human", "Mouse"], "test_hmac": ["Human", "Macaque"], "test_3sp": ["Human", "Macaque", "Mouse"],
    "kidney": ["Human", "Mouse"], "islet": ["Human", "Mouse"], "bone_marrow": ["Human", "Lemur", "Mouse"],
    "lung_healthy": ["Human", "Lemur", "Mouse"], "brain_m1_hmouse": ["Human", "Mouse"], "brain_m1": ["Human", "Marmoset", "Mouse"],
    "brain_dlpfc": ["Human", "Chimpanzee", "Macaque", "Marmoset"], "brain_dlpfc_hchimp": ["Human", "Chimpanzee"],
    "brain_dlpfc_hmac": ["Human", "Macaque"], "brain_dlpfc_hmac_own": ["Human", "Macaque"], "brain_dlpfc_hmarm": ["Human", "Marmoset"],
    "vz_hmouse": ["Human", "Mouse"], "muscle_ageing": ["Human", "Mouse"], "tumour_zilionis": ["Human", "Mouse"],
}
