"""E4: translational reading of deployment. Card: definitions and predictions, fixed before any E4 association was computed.

EXPLORATORY throughout. Nothing here changes a status, row or modifier of the plan or an H1, Z4 or E1 table. Vocabulary:
CHIMERA_MECHANISM.md (canonical gate = main model; per-gene gate = companion read-out). Delta-alpha is "deployment relative to the
protein prior", never conservation or selection. The gap between the two read-outs is task X8 and is not analysed here; agreement of
the two is used only to define robust named calls, as E1 did.

Question
--------
Does the placement of a human gene's cell-state deployment against its protein prior, read in a human+model-species fit of healthy
tissue, say anything about how faithfully that model species represents the human gene? Four angles: (Q1) mouse model concordance
with the human disease of the same gene; (Q2) where drug targets sit per tissue; (Q3) monogenic disease genes of a tissue's organ
system in that tissue, mouse and primate pairs; (Q4) gene-level link to the disease-programme transfer results (P13 kidney, T7.9
islet, T7.3 liver), descriptive.

What I knew when writing this
-----------------------------
E1 (tissue programmes on both gates, recurrent genes: cytokine receptors and junction scaffolds above; nuclear RNA machinery below;
liver metabolism at or below the prior in its own tissue while pseudobulk ranks it first), H1, Z4 (per-gene gate validates on every
gene-level gold), Y2 (MGI/IMPC phenotype-system concordance vs canonical D and S on liver, testis, marrow, kidney: not usable; only
marrow S_g +0.117 own-system; built on pre-Z4 tables), Y6 use 3 (approved vs failed targets, Minikel: a lean for S, not a result;
target vs not falls to ~0 with tau and PubMed), P13/T7.3/T7.9 H-D1 (healthy D does not predict gene-level disease transfer), the
van Zyl glaucoma result (glaucoma genes AUROC 0.55 on D). No E4 resource has been joined to any read-out. I have not looked at where
any named drug target or disease gene sits in any cohort.

Read-outs (per cohort, H1 schema, centred within cohort; table from E1 genes.calls, i.e. E1 cohort_table)
------------------------------------------------------------------------------------------------------
D_can   canonical gate (H1 tables), the main model.            D_free  per-gene gate (Z4 tables, Z4 kept seeds).
S       count-evidence margin S_ema (canonical fits).            B1      pseudobulk cross-species profile correlation (the simple
                                                                          pseudobulk comparison asked for in one clause per result).
Universe = E1's: count-supported in both fits, one row per human Ensembl ID. Every value enters as its inverse-normal rank (INT)
within the cohort universe. Covariates: pi_raw (identity), expr, det_human, det_partner_min (detection), tau (cell-type specificity),
log PubMed counts human and mouse (Y2 annot/pheno.parquet, NCBI gene2pubmed of 2026-11-02). Mouse ortholog for disease joins:
MGI HOM_MouseHumanSequence one-to-one class; it must agree with the Ensembl/MGI mapping Y2 used (else dropped).

Cohorts
-------
HM thread (human+mouse, one per tissue; marrow and lung are human, lemur, mouse fits): H1 PRIMARY.
HM extra: muscle_ageing (sensitivity tissue), tumour_zilionis (not healthy: Q2 descriptive only).
Primate pairs for Q3 replication: liv_hmonkey, test_hmac, vz_hmmul, vz_hmfas, brain_dlpfc_hchimp, brain_dlpfc_hmac, brain_dlpfc_hmarm
(cortex primates are another study, DLPFC, than cortex mouse, M1; labelled). Other cohorts (pig, zebrafish, multi-species) not used.

Named calls (robust layer) = E1 rule unchanged: nameable on both gates (strict support, seed-stable signs, O1/C1 flags, no zero or
quiet species, not a reference artefact), saturated in neither, |INT| >= 1 on both gates with the same sign. "below" = call -1,
"above" = +1, "at" = tested, no call. Mitochondrially encoded genes never named. Every named gene is printed with pi, S, B1,
detection human/partner and both INT values.

Q1 Mouse model concordance (MGI human-mouse disease connection; IMPC Phenodigm via Open Targets as replication)
----------------------------------------------------------------------------------------------------------
Human disease gene: human row of MGI_DO.rpt (DO disease with OMIM support) -> set of DOIDs per gene.
Tested in mouse: the one-to-one mouse ortholog has >= 1 genotype with MP annotation in MGI_GenePheno.rpt.
Y_mgi = 1 if a genotype whose marker list includes the ortholog is annotated in MGI_Geno_DiseaseDO.rpt to a DOID of the human gene
(exact DOID, or the same OMIM ID); 0 otherwise. Sensitivity: only genotypes with one marker and no transgene ("Tg(") allele.
Y_not (secondary): no positive annotation and a genotype of the ortholog annotated in MGI_Geno_NotDiseaseDO.rpt to the human
gene's disease ("this mouse does not model this disease").
Y_impc (replication): among targets with a Mendelian disease in Open Targets (orphanet, gene2phenotype, clingen, genomics_england
evidence) and >= 1 animal-model evidence row (evidence_impc, Phenodigm), 1 if an animal-model row names one of the target's own
Mendelian diseases (same diseaseId).
Organ-matched subset per tissue: the gene's human disease has the tissue's ORGAN_DO ancestor (DO is_a closure).
Statistic: logistic Y ~ INT(read-out) + INT covariates (pi, expr, det_human, det_partner_min, tau, log PubMed human and mouse,
number of human diseases); effect = log OR per SD of INT(read-out). Per cohort (HM thread), and pooled over the HM thread with
cohort fixed effects and a gene-cluster bootstrap (1,000 draws) for the CI. Raw (no covariates) beside. Read-outs D_can, D_free,
S, B1, and pi itself as a reference.

Q2 Drug targets (Open Targets clinical_target, latest)
-------------------------------------------------------
approved = target of >= 1 drug with maxClinicalStage APPROVAL or PHASE_4; clinical = max stage PHASE_1 .. PREAPPROVAL, never
approved; withdrawn-only targets are neither. Tissue-indicated approved = an approved drug-target pair whose indications include a
non-oncology disease below the tissue's TA_OT ancestor (Open Targets disease index ancestors; oncology = ancestor MONDO_0004992
cancer or EFO_0000616 neoplasm).
Set statistic = H1/E1's: mean INT(read-out) of members minus the mean under permutation within pi x expression strata (10,000
draws thread, 2,000 others); effect in SD units, z, two-sided p, BH within question. tau-strata sensitivity. Partial effect: OLS
INT(read-out) ~ member + spline covariates (pi, expr, det_human, det_partner_min, tau, log PubMed human), member coefficient with
gene bootstrap CI (1,000). Call shares: fraction of members called below / above against the stratum-matched expectation.
Naming: per tissue, tissue-indicated approved targets (then any approved) with a robust call; "mouse deployment differs" = below,
with pi in the top half of the cohort (protein conserved); "well modelled" = above or at prior with B1 in the top half.

Q3 Disease-gene classes (Genomics England PanelApp green genes; DO organ classes and HPO male infertility as second source)
-------------------------------------------------------------------------------------------------------------------------
Own-organ disease set per tissue = union of green (confidence 3) genes of PANELS[tissue]; each panel also alone. Testis: PanelApp
DSD plus HPO genes annotated to Male infertility (HP:0003251) or descendants. Second source: MGI_DO human genes with the tissue's
ORGAN_DO ancestor. Statistic as Q2 (set effect within pi x expr strata; tau sensitivity; partial with PubMed). Own versus other:
z in the own tissue minus mean z of the same set in the other seven thread tissues, null from the same draws (E1's contrast).
Retinal disorders panel is scored in eye as an off-tissue check (anterior segment, no photoreceptors; no expectation).
Primate replication: own set in the primate pairs of the same tissue.

Q4 Disease cohorts (descriptive)
-------------------------------
Kidney P13 primary DE (human AKI vs control, mouse IRI vs sham; PT, TAL, DCT, PC, EC, FIB), islet T7.9 (human T2D vs ND, mouse
joint db/db + STZ; beta, alpha, delta), liver T7.3 hepatocytes. For own-organ disease genes and tissue-indicated targets in the
universe: human DE (padj < 0.1) in a type, mouse log2FC same sign (transfer) or not; cross-tabulated with the healthy call
(below / at / above). One Fisher test per tissue on (below vs rest) x (transfer vs not) among human-DE genes; named examples only.

Predictions (before any association)
-----------------------------------
E4-P1 Q1: pooled log OR per SD for D_free and S small positive (OR 1.02-1.15), CI likely including 1 on all disease genes;
      organ-matched larger but wider; D_can nearer 1; B1 about the same as S; PubMed the largest covariate. Y_impc same sign as
      Y_mgi. Reason: Y2 found nothing at system level; a shared cell-state deployment of the gene in the affected organ should help
      a knockout reproduce the disease, but most Mendelian failures of mouse models are compensation, lifespan and dosage, which
      deployment does not see.
E4-P2 Q2: approved targets above the prior (effect +0.1 to +0.3 SD) in most tissues on all four read-outs, smaller with tau;
      tissue-indicated approved targets at or above in their own tissue in kidney, islet, marrow, cortex, eye; liver-indicated at
      or below in liver (E1's liver exception). Below-prior targets will be mostly enzymes and nuclear proteins with conserved
      sequence; above-prior targets receptors, channels, transporters.
E4-P3 Q3: own-organ disease genes above matched genes in their own tissue in >= 5 of 8 tissues on D_free and S (disease genes
      have cell-type patterns), most of it gone with tau strata; liver at or below; own-versus-other contrast positive in >= 3
      tissues; B1 the same direction (pseudobulk shows it too); primate pairs the same direction. Retinal panel in eye: no
      expectation.
E4-P4 Q4: no association between healthy call and disease-response transfer among disease genes (H-D1 was null); named
      examples only.

Limits fixed in advance
-----------------------
Disease-model curation (MGI) follows publication: a gene without a curated model may simply be unstudied, so PubMed enters every
Q1 model; NOT annotations are few. Panels and targets are tissue-specific genes, so tau is the main alternative explanation and is
in every partial model. One donor or study per species in several tissues; marrow and lung are three-species fits. Mouse is the
only model species with disease annotation; primate pairs enter Q3 only.
"""

from __future__ import annotations

import os
from pathlib import Path

from revision.tasks.H1_harmonise import rules as H1

REPO = H1.REPO
RES = H1.RES
OUT = Path(os.environ.get("E4_OUT", RES / "E4"))
DATA = REPO / "data" / "revision" / "E4"
SEED = 20261104

# reused downloads of earlier tasks (each pinned in its own MANIFEST.tsv)
REUSED = {
    "pheno": RES / "Y2" / "annot" / "pheno.parquet",            # PubMed counts (gene2pubmed 2026-11-02) and Y2's mouse mapping
    "hpo_g2p": REPO / "data" / "revision" / "Y2" / "hpo" / "genes_to_phenotype.txt",
    "hp_obo": REPO / "data" / "revision" / "Y2" / "onto" / "hp.obo",
}

HM_THREAD = dict(H1.PRIMARY)  # tissue -> cohort
HM_EXTRA = {"skeletal muscle": "muscle_ageing", "tumour myeloid": "tumour_zilionis"}
PRIMATE = {"liver": ["liv_hmonkey"], "testis": ["test_hmac"], "eye": ["vz_hmmul", "vz_hmfas"],
           "cortex": ["brain_dlpfc_hchimp", "brain_dlpfc_hmac", "brain_dlpfc_hmarm"]}
TISSUES = H1.TISSUE_ORDER

N_PERM_THREAD = 10_000
N_PERM_OTHER = 2_000
N_BOOT = 1_000
INT_CALL = 1.0

# Q3: PanelApp panels (id: name), green genes, fixed before reading.
PANELS = {
    "liver": {544: "Cholestasis", 385: "Neonatal cholestasis", 528: "Glycogen storage disease", 267: "Hyperammonaemia",
              653: "Polycystic liver disease", 6: "Familial hypercholesterolaemia"},
    "kidney": {106: "Proteinuric renal disease", 292: "Renal tubulopathies", 487: "Cystic renal disease",
               149: "Nephrocalcinosis or nephrolithiasis", 234: "CAKUT", 548: "Tubulointerstitial kidney disease",
               139: "Atypical haemolytic uraemic syndrome"},
    "islet": {472: "Monogenic diabetes", 293: "Neonatal diabetes", 308: "Congenital hyperinsulinism", 152: "Familial diabetes",
              26: "Diabetes with additional phenotypes suggestive of a monogenic aetiology"},
    "eye": {249: "Glaucoma (developmental)", 250: "Corneal abnormalities", 658: "Corneal dystrophy"},
    "bone marrow": {398: "Primary immunodeficiency or monogenic inflammatory bowel disease", 159: "Cytopenias and congenital anaemias",
                    518: "Rare anaemia", 545: "Bleeding and platelet disorders", 1075: "Autoinflammatory disorders"},
    "lung": {200: "Familial pulmonary fibrosis", 1638: "Childhood interstitial lung disease",
             550: "Respiratory ciliopathies including non-CF bronchiectasis", 193: "Pulmonary arterial hypertension"},
    "cortex": {285: "Intellectual disability", 402: "Early onset or syndromic epilepsy", 96: "Malformations of cortical development",
               162: "Severe microcephaly"},
    "testis": {9: "Differences in sex development"},
    "skeletal muscle": {225: "Congenital myopathy", 185: "Limb girdle muscular dystrophies, myofibrillar myopathies and distal myopathies",
                        207: "Congenital muscular dystrophy", 66: "Rhabdomyolysis and metabolic muscle disorders"},
}
EXTRA_PANELS = {"eye": {307: "Retinal disorders"}, "lung": {398: "Primary immunodeficiency or monogenic inflammatory bowel disease"}}
HPO_MALE_INFERTILITY = "HP:0003251"

# DO organ-system ancestors (names fixed here; IDs checked against doid.obo names before use).
ORGAN_DO = {
    "liver": {"DOID:409": "liver disease", "DOID:655": "inherited metabolic disorder"},
    "kidney": {"DOID:557": "kidney disease", "DOID:18": "urinary system disease"},
    "islet": {"DOID:9351": "diabetes mellitus", "DOID:4194": "glucose metabolism disease"},
    "eye": {"DOID:5614": "eye disease"},
    "bone marrow": {"DOID:2914": "immune system disease", "DOID:74": "hematopoietic system disease"},
    "lung": {"DOID:1579": "respiratory system disease", "DOID:2914": "immune system disease"},
    "cortex": {"DOID:863": "nervous system disease", "DOID:0060037": "developmental disorder of mental health"},
    "testis": {"DOID:15": "reproductive system disease", "DOID:12336": "male infertility"},
    "skeletal muscle": {"DOID:0080000": "muscular disease"},
}

# Open Targets therapeutic ancestors for tissue-indicated targets. Amendment 1 (before any association): the 2026-09 disease index is
# MONDO-based and has none of the EFO names first written here, so each is replaced by the MONDO term of the same name ("disorder").
TA_OT = {
    "liver": {"MONDO_0005154": "liver disorder"},
    "kidney": {"MONDO_0005240": "kidney disorder", "MONDO_0002118": "urinary system disorder"},
    "islet": {"MONDO_0005015": "diabetes mellitus"},
    "eye": {"MONDO_0005328": "eye disorder"},
    "bone marrow": {"MONDO_0005570": "hematologic disorder", "MONDO_0005046": "immune system disorder"},
    "lung": {"MONDO_0005087": "respiratory system disorder"},
    "cortex": {"MONDO_0005071": "nervous system disorder", "MONDO_0002025": "psychiatric disorder"},
    "testis": {"MONDO_0003150": "male reproductive system disorder"},
    "skeletal muscle": {"EFO_0002970": "muscular disease"},
}
ONCOLOGY_OT = {"MONDO_0045024", "MONDO_0004992", "MONDO_0005070"}  # cancer or benign tumor, cancer, neoplasm
APPROVED_STAGES = {"APPROVAL", "PHASE_4"}
CLINICAL_STAGES = {"PHASE_1", "PHASE_1_2", "PHASE_2", "PHASE_2_3", "PHASE_3", "EARLY_PHASE_1", "PREAPPROVAL"}

# Q4 disease DE files (relative to RES) and cell types
DISEASE_DE = {
    "kidney": dict(human="P13/de/human_primary_{t}.tsv", mouse="P13/de/mouse_primary_{t}.tsv", types=["pt", "tal", "dct", "pc", "ec", "fib"]),
    "islet": dict(human="T7_9/de/human_t2d_vs_nd_{t}.tsv", mouse="T7_9/de/mouse_joint_{t}.tsv", types=["beta", "alpha", "delta"]),
    # liver: primary contrast and the biopsy-type control sensitivity (T7.3: confounded by control tissue source; descriptive only)
    "liver": dict(human="T7_3/de/human_primary.tsv", mouse="T7_3/de/mouse_primary.tsv", types=["hepatocyte"],
                  human_sens="T7_3/de/human_biopsy_controls.tsv"),
}
PADJ = 0.1
