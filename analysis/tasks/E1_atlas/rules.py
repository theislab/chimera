"""E1: a cross-tissue atlas of deployment programmes. Card: definitions and predictions, fixed before any E1 association was computed.

EXPLORATORY throughout. Nothing here changes a status, row or modifier of the plan. Vocabulary: CHIMERA_MECHANISM.md (the canonical
gate is the main model; the per-gene gate is a companion read-out). The gap between the two read-outs is task X8 and is not analysed here.

What I knew when writing this
-----------------------------
H1's results on 72 sets (50 Hallmark, 12 classes, 10 first-submission categories) on the canonical gate: 57 recurrent on D over eight
tissues, 35 strict (receptors, EMT, ECM, inflammatory, coagulation ... above; mitochondrial proteins, chromatin regulators, OXPHOS,
translation ... below); three tissue-specific sets (heme in marrow, bile acid in kidney, protein secretion in testis); set position on
D agrees with S_g 0.98, with B1 0.92. X3 (class axis, liver/testis), X1, X2, Y1, Y2, Z4 ledgers. Z4: free D follows its own S_g
(0.93-0.99). Cohort sizes and count-supported/nameable counts of both tables. No set z on the per-gene gate, no Reactome, GO or
UniProt-keyword z on any cohort, no recurrent gene list across tissues, no distance comparison of set positions.

Read-outs (per cohort, H1 schema; centred within cohort)
-------------------------------------------------------
D_can   canonical gate, results/analysis/H1/genes/<cohort>.parquet, column D.
D_free  per-gene (free) gate, results/analysis/Z4/h1/genes/<cohort>.parquet, column D (Z4 kept seeds).
S       count-evidence margin of the canonical fits (S_ema), beside, no features.
B1      pseudobulk cross-species profile correlation (baseline). pi (identity), dNdS (log10 human-mouse dN/dS): sequence baselines.
Universe: genes count-supported in BOTH fits (|S_ema| >= 1e-6 in canonical and free consensus), one per human Ensembl ID. Flat free
gates (Z4: liver 4-sp, testis H+Mouse, H+Macaque, 3-sp) enter as ranks, which is all a set test uses.

Gene sets (sets.py), by human Ensembl ID through the cohort's human symbols
-----------------------------------------------------------------------
hallmark  MSigDB Hallmark 2020 (Enrichr; T6.1 copy), 50.
class     X3 gene classes (H1's 12) and the first-submission broad categories (10).
reactome  ReactomePathways.gmt (reactome.org, gmt of 2026-10-19, T6.1 copy), human pathways with 15-300 genes in the union vocabulary;
          plus every Reactome top-level pathway (root of ReactomePathwaysRelation.txt, fetched 2026-11-03) as its own set, any size.
goslim    GO generic slim (goslim_generic.obo, release 2026-07-26), annotations from goa_human.gaf (fetched 2026-11-03), NOT qualifiers
          dropped, propagated to slim ancestors over is_a and part_of in go-basic.obo; slim terms with >= 15 genes.
keyword   UniProt keywords (X3 download), categories Biological process, Molecular function, Cellular component, Domain, PTM, Ligand,
          >= 40 genes in the union vocabulary. These are the protein-level sets (domains, PTMs, topology).
tissue    A priori tissue programmes (TISSUE_PROGRAMMES below), fixed here before reading, by ID.
Min 15 genes in the cohort universe for a z (lineage-sized sets are H1's and are not redone).

Set statistic (atlas.py; H1's recurrence.perm_scores and strata reused unchanged)
---------------------------------------------------------------------------------
Value: inverse-normal transform of the within-cohort rank of the read-out on the universe. Set score: member mean. Null: values
permuted within strata of pi_raw (5 quantile bins) x expression (10 bins) in the cohort (identity- and expression-matched
background), N_PERM draws (10,000 on the eight-tissue thread, 2,000 elsewhere). effect = score - null mean (SD units of the
transformed read-out, comparable across cohorts); z = effect / null SD; two-sided empirical p; q = BH within library and cohort.
Sensitivities on the thread: tau (strata also on cell-type specificity, 5 bins) and block (ESM-module block permutation, 2,000 draws),
for D_can and D_free. pi and dNdS read-outs use expression strata only (as H1).

Recurrence (eight-tissue thread = H1 PRIMARY: one human+mouse model per tissue; marrow and lung human, lemur, mouse)
--------------------------------------------------------------------------------------------------------------
H1's rule per read-out: R = max(#tissues z >= 1.96, #tissues z <= -1.96), null from the same draws, q_rec BH within library;
recurrent = R >= 3 and q_rec < 0.05. AGREE (the robust layer) = recurrent on D_can and on D_free with the same direction.
STRICT = AGREE and recurrent in that direction under tau and block for both gates. Replication over all 26 cohorts: number of
cohorts with |z| >= 1.96 in each direction, per gate (descriptive).
Redundancy (reporting only): Reactome/GO/keyword sets ranked by |Stouffer z on D_can + D_free|; a set is dropped from summary
tables when a higher-ranked set of any library shares Jaccard >= 0.5 of its genes. Full tables keep every set.

Tissue-matched programmes (tissue.py)
-------------------------------------
For each tissue's a priori programmes: z in its own tissue, and contrast = z_own - mean z over the other seven tissues, with the
contrast's permutation null (H1's tissue_specific contrast), BH over all tissue programmes per read-out. Read on D_can, D_free, S,
B1, and on D_can, D_free with tau strata (the margin rewards having a pattern, H1 P6).

Evolutionary distance (distance.py)
-----------------------------------
Divergence from human (TimeTree, median estimates, My): DIVERGENCE below. Ladders inside one data source: eye (Van Zyl: M. mulatta,
M. fascicularis, pig, mouse), liver (CAMEX: macaque, mouse, zebrafish), testis (macaque, mouse), cortex DLPFC (chimpanzee, macaque,
marmoset; plus M1 H+Mouse, another study, labelled). Per pair cohort: effect of every set; agreement of the set-effect vector with
the eight-tissue consensus (Stouffer over the thread); pole scores = mean effect of the AGREE-above and AGREE-below sets; per set,
within-ladder Spearman of effect with log10 divergence, sign-consistency over ladders.

Recurrent named genes (genes.py)
--------------------------------
Call per cohort and gate: nameable (H1/O1/C1 rule: strict support = count support, |S| >= median, sign(S) = sign(D), seed-stable
signs; no zero or quiet species, no mapping flag, not a DLPFC reference artefact), not saturated (alpha > 0.95 in no kept seed of
either gate), |INT(D)| >= 1 within the cohort universe. AGREE call = called in both gates with the same sign. Recurrent gene =
AGREE call in >= 3 of 8 thread tissues in one direction and no AGREE call in the other direction. Expected count from 200 draws
permuting gene labels within pi x expression strata in each tissue independently (fold over expectation). Replication: AGREE calls
over all 26 cohorts. Mitochondrially encoded genes reported, not named as biology (X2: assay capture).

Predictions (before any association)
-----------------------------------
E-P1 Gates agree on programmes: over all sets with a z in both, Spearman(set z D_can, set z D_free) >= 0.7 in every thread tissue
     (free D is S_g; H1 set position D vs S 0.98). More sets recur on D_free than on S (D_free = S, plus nothing), fewer than D_can.
E-P2 Reactome top level, AGREE: above for Signal Transduction, Immune System, Extracellular matrix organization, Hemostasis,
     Developmental Biology, Muscle contraction; below for Metabolism of RNA, Metabolism of proteins (translation), Cell Cycle,
     DNA Repair, Chromatin organization, Gene expression (Transcription).
E-P3 GO slim, AGREE: above for extracellular region/matrix, plasma membrane, cell adhesion, signalling receptor activity, immune
     system process, cell motility; below for nucleus/nucleolus, RNA binding, ribosome/translation, mitochondrion, DNA repair,
     mRNA processing.
E-P4 Protein-level keywords: above for Signal, Glycoprotein, Disulfide bond, Transmembrane, EGF-like and Immunoglobulin domains
     (secreted and surface proteins with cell-state patterns); below for Acetylation, Ribonucleoprotein, Nucleus, Mitochondrion.
     Expect these to follow the pi/dNdS opposite sign (fast extracellular proteins above), so they are the "fast protein, shared
     use" archetype at programme level.
E-P5 Tissue programmes: own-tissue z >= 1.96 on both gates in >= 5 of 8 tissues; contrast q < 0.05 in >= 3; with tau strata
     fewer (<= 3 own-tissue). Phototransduction in eye: no expectation (Van Zyl is anterior segment, no photoreceptors).
E-P6 Distance: the above pole is present in every pair cohort including human-chimpanzee; the below pole grows with distance (more
     level cost); agreement with the consensus is highest for human-mouse cohorts (the consensus is built from them) and lowest
     for human-zebrafish (unmixed). Within eye, H+Mouse and H+Pig closer to each other than to the macaque pairs.
E-P7 Recurrent named genes: at least 1.5-fold over expectation in both directions; fewer than 150 genes each; above-recurrent
     enriched for receptors, ECM and signalling; below-recurrent includes genes with a recurrent human-mouse level gap (Y1: PDK4,
     RUNX1-like) and machinery.
"""

from __future__ import annotations

import os
from pathlib import Path

from analysis.tasks.H1_harmonise import rules as H1

REPO = H1.REPO
RES = H1.RES
OUT = Path(os.environ.get("E1_OUT", RES / "E1"))
DATA = REPO / "data" / "analysis" / "E1"
FUNC = REPO / "data" / "analysis" / "functional"
X3ANN = REPO / "data" / "analysis" / "x3_annotations"
CAN = RES / "H1" / "genes"
FREE = RES / "Z4" / "h1" / "genes"
SEED = 20261103

MIN_SET = 15
REACTOME_MIN, REACTOME_MAX = 15, 300
KEYWORD_MIN = 40
KEYWORD_CATEGORIES = {"Biological process", "Molecular function", "Cellular component", "Domain", "PTM", "Ligand"}
N_PERM_THREAD = 10_000
N_PERM_OTHER = 2_000
N_PERM_BLOCK = 2_000
Z_CALL = 1.96
R_MIN = 3
JACCARD_PRUNE = 0.5
INT_CALL = 1.0
GENE_R_MIN = 3
N_GENE_NULL = 200

TISSUES = H1.TISSUE_ORDER
PRIMARY = H1.PRIMARY

# Divergence time from human, My (TimeTree 5 median estimates).
DIVERGENCE = {"chimpanzee": 6.4, "macaque": 28.8, "M. mulatta": 28.8, "M. fascicularis": 28.8, "marmoset": 43.2, "mouse lemur": 74.0,
              "mouse": 87.0, "pig": 94.0, "zebrafish": 429.0}
PAIRS = {  # pair cohorts: partner species and ladder
    "liv_hmonkey": ("macaque", "liver"), "liv_hmouse": ("mouse", "liver"), "liv_hzeb": ("zebrafish", "liver"),
    "test_hmac": ("macaque", "testis"), "test_hmouse": ("mouse", "testis"),
    "vz_hmmul": ("M. mulatta", "eye"), "vz_hmfas": ("M. fascicularis", "eye"), "vz_hpig": ("pig", "eye"), "vz_hmouse": ("mouse", "eye"),
    "brain_dlpfc_hchimp": ("chimpanzee", "cortex"), "brain_dlpfc_hmac": ("macaque", "cortex"), "brain_dlpfc_hmac_own": ("macaque", "cortex"),
    "brain_dlpfc_hmarm": ("marmoset", "cortex"), "brain_m1_hmouse": ("mouse", "cortex"),
    "kidney": ("mouse", "kidney"), "islet": ("mouse", "islet"), "muscle_ageing": ("mouse", "skeletal muscle"), "tumour_zilionis": ("mouse", "tumour myeloid"),
}
MULTI = {"liv_4sp": "zebrafish", "test_3sp": "mouse", "vz_primates": "M. mulatta", "vz_full": "pig", "brain_dlpfc": "marmoset", "brain_m1": "mouse",
         "bone_marrow": "mouse", "lung_healthy": "mouse"}  # most distant partner, labelled

# A priori tissue programmes, by ID (R-HSA Reactome, GO full ontology with propagation, hallmark name, class name).
TISSUE_PROGRAMMES = {
    "liver": ["R-HSA-211859", "R-HSA-174824", "GO:0006629", "hallmark:Bile Acid Metabolism", "hallmark:Xenobiotic Metabolism"],
    "testis": ["R-HSA-1500620", "GO:0022414", "GO:0044782", "hallmark:Spermatogenesis"],
    "bone marrow": ["R-HSA-168256", "GO:0002376", "GO:0030097", "hallmark:heme Metabolism"],
    "kidney": ["R-HSA-382551", "R-HSA-425407", "GO:0055085", "GO:0005215"],
    "islet": ["R-HSA-422356", "R-HSA-2980736", "GO:0005179", "GO:0030141", "hallmark:Pancreas Beta Cells"],
    "eye": ["R-HSA-1474244", "R-HSA-397014", "GO:0003012", "GO:0031012", "R-HSA-2187338"],
    "cortex": ["R-HSA-112316", "R-HSA-112315", "GO:0050877", "GO:0045202"],
    "lung": ["R-HSA-5683826", "R-HSA-168256", "GO:0002376", "GO:0001944", "GO:0003013"],
}
