"""Y6 (EXPLORATORY): uses of existing fits nobody planned. Definitions, fixed before any association between a readout and a label.

Each use is a benchmark-style test with a ground truth that does not come from the counts. Nothing here changes a status, row or
modifier. Vocabulary: ledgers/CHIMERA_MECHANISM.md. S_gs below is a count margin of one species in one fit; it may be read as count
evidence of shared cell-state pattern with a level cost in it, not as conservation. Delta-alpha of masked genes is not a count call.

Rows. A row is (fit, species), species != human. S_gs = mean over the cells of species s of l_cg (T2.5 margin, posterior mean z, no
batch mask), mean over the seeds with a stored forward pass. Fits and sources: FITS below.

USE 1. Genome annotation check.
  Universe U_expr of a row: genes of the fit's table whose largest detection among the other species of the fit is >= 0.05, so that
  near-absence in s is informative. U_hi: U_expr with identity_s >= 80. U_all: every gene of the table, sensitivity.
  Scores, higher means more suspect:
    S_row   = -S_gs                       Chimera, this species in this fit
    S_gene  = -S_ema                      Chimera, gene level, as stored (masked genes 0)
    D_neg   = -D                          Chimera, Delta-alpha, centred
    zero_s  = 1 - detection_s             baseline named in the task
    zero_gap= max other detection - detection_s    baseline, the fair form of zero_s (knows the gene is on elsewhere in the tissue)
    B1_neg  = -E(human, s) of the fit's counts; an undefined E counts as the most suspect value, because it is undefined when s lacks the gene
    id_neg  = -identity_s                 baseline
  Labels, from Ensembl only (results/analysis/Y6/annot/genome_labels.parquet), no count enters:
    Primary
      T_lowconf : Compara 110 pair not high confidence (is_high_confidence == 0), or the pair absent from Compara 110 as any type.
      T_later   : by release 116 the species gene is retired, or is no longer protein_coding, or the same human-species pair is no longer ortholog_one2one.
      T_newasm  : species gene's stable ID absent from an older assembly's annotation: macaque Mmul_8.0.1 (E97), pig Sscrofa10.2 (E89), zebrafish GRCz10 (E91), lemur Mmur_2.0 (E90),
                  mouse GRCm38 E84 (an annotation many mouse atlases used), M. fascicularis 5.0 (E98). The model was added or replaced after a reference an atlas may have used.
                  Which release each atlas used is not fully known per dataset, so this is a proxy for "the gene was not in the counting reference".
                  Amended before any join with a readout: Ensembl 98 to 116 proved frozen for macaque, zebrafish and lemur (prevalence near 0, counts only), so the planned label T_new98 had nothing to test there; kept as a column.
      Only genes with biotype protein_coding in Ensembl 110 enter use 1: liver and testis tables also hold 80 and about 640 non-protein pairs, without a protein-Compara row.
    Secondary
      T_goc     : GOC < 50, on pairs where GOC exists.   T_wga: WGA < 50, on pairs where WGA exists.
      T_no3utr  : no three_prime_utr feature on the gene in the Ensembl 110 GTF. 3' chemistry reads the 3' end, so this is the mechanism most directly tied to losing counts.
      T_moved   : same assembly only (macaque, pig, zebrafish, lemur 98 -> 110; all but none across 110 -> 116): contig or strand changed, or the 3' end moved by > 2 kb, or CDS length changed by > 10%.
      T_tsl     : mouse only: best transcript support level of the gene > 1 or none.
  Split: human chromosome, Ensembl 110. Tuning on odd autosomes, scoring on even autosomes. X, Y, MT, scaffolds in neither.
  Thresholds: on odd genes, for each (row, score, label), the score value at the top-q cut with the best F1, q in GRID_Q; applied unchanged on even genes.
  Readouts on even genes: precision, recall, F1 at that threshold; lift = precision / base rate; precision at a fixed budget (top 5%); average precision (AP); AUROC.
  CI: 500 block draws on ESM modules (Y1 blocks), same draws for every score of a row. A row and label pair is scored only with >= MIN_POS positives in even U_expr.
  Beyond baselines: on even genes, logit(label) ~ rank S_row + rank zero_gap + rank id_neg + rank B1_neg + rank other-species expression; coefficient of S_row, block CI.
  Rule flag, as the task words it: F_abs = detection_s < 0.005, other-species detection >= 0.05, identity_s >= 80, on even genes; precision against the base rate of U_hi.
  Verdict rule, per primary label over the joint-fit rows (JOINT_ROWS), BH over rows x labels:
    differentiates: AP(S_row) - best baseline AP has CI above 0 (q < 0.05) on at least half of the scored rows of a label.
    matches baseline: AP lift over base rate has CI above 1 on at least half of the scored rows, and the above fails.
    not usable: otherwise. A pass on a label against baselines that share its definition (identity on T_lowconf) is read with that in mind.

USE 2. Choosing a model species.
  Contrast in one fit: Score = S_g,a - S_g,b, against B1c = E(human, a) - E(human, b) and idc = identity_a - identity_b, all from the same genes.
  Independent evidence:
    testis  test_3sp  a Macaque, b Mouse. Gold Gt = zPP - zPO (X7 murat_tiers): held-out primates among themselves minus held-out primate-outgroup. Covariates C_pat + H4_nsp.
                      The same contrast from two separate pair fits (test_hmac, test_hmouse) beside it.
    liver   liv_4sp   a Monkey, b Mouse. Gold G_prim = G_lemur - G_H3: TS vs Tabula Microcebus liver (lemur, a held-out primate, H3 recipe) minus TS vs TMS. C_pat.
                      Beside it, the mouse row alone against G_H3 (X7's result, to be reproduced on these genes).
    marrow  bone_marrow a mouse, b mouse_lemur. Gold COE (Shay 2013, human vs mouse microarray). One-sided: no independent lemur marrow. Also each of S_mouse, E(human,mouse), identity_mouse alone, and at matched variability (T12.3, >= 4-fold both species).
    eye     vz_hmmul pair fit S_g against E(human, M. fascicularis), a held-out sister species of the same study and lab; the human cells are shared, so independent in species, not in sample.
            No independent mouse or pig eye atlas here; recorded.
  Statistics: rho_p(score, gold | covariates), 2,000 block draws (X4 rho_unit); deltas to identity and B1 on the same genes; seed range for Chimera. Odd and even autosomes reported apart; nothing is tuned.
    Pick accuracy: scores and gold median-centred within the fit; share of genes where sign(score) = sign(gold), on genes with |gold| above its median (clear cases), block CI.
    Pathway level: Hallmark sets (H1 rules) with >= 15 genes: set mean of the covariate-residual rank of each score and of the gold; Spearman across sets, set bootstrap.
    Hagai 2018: Supplementary Table 4 has response divergence per gene, not per species pair, in fibroblasts and phagocytes; it cannot rank species and is not used.
  Verdict rule: usable if, in at least 2 of the tissues with a gold of two species or a held-out species, rho_p > 0 with CI above 0 and delta to identity CI above 0.
    Differentiates only if also delta to B1 CI above 0 in those tissues. Otherwise not usable as a species picker.

USE 3. Drug targets, human-mouse.
  Labels, gene level, per target gene:
    Minikel 2024 (pp.tsv, Pharmaprojects; merge2.tsv.gz): y_appr = 1 launched for any indication; 0 "failed" = no launched pair, no active program at any pair (acat empty), highest historical phase I-III.
    y_gen = 1 if any clinical-stage T-I pair has genetic support (comb_norm >= 0.8), 0 otherwise, on clinical-stage targets.
    Open Targets replication: y_appr_ot = 1 APPROVAL or PHASE_4, 0 otherwise among stage PHASE_1 to PHASE_3 (WITHDRAWAL and UNKNOWN left out). y_gen_ot = max direct genetic_association score >= 0.5 among clinical targets.
    y_tgt = in the Minikel clinical set vs not, on the whole vocabulary: only to show the study-depth bias.
  Readouts, per tissue (H+Mouse rows, Y1 matrices): S_fw, D, B1; identity_h-m as a baseline (not a covariate in its own row). Summary across tissues: mean of the within-tissue z (Y1 z__ED) over supported tissues, >= 3.
  Universe: count-supported genes per tissue; summary on genes supported in >= 3.
  Ladder: C0 none; C1 expression, detection, CDS length; C2 + tau; C3 + identity (pi_raw); C4 + dN/dS, LOEUF; C5 + log PubMed human, mouse = fixed set; all as ranks, linear.
  Primary: rho_p(X, y | C5) with 2,000 block draws, BH over (X in S, D) x (y_appr, y_gen) x (summary, 6 tissues). Sensitivities: non-oncology targets only; gene class (H1 classes) added; first-target rule not available.
  Tissue match: therapy area -> tissue (hematology, immune -> bone marrow; metabolic -> liver; endocrine -> islet; ophthalmology -> eye). Same y within the area's indications; Delta_match = rho_p(X_T, y_area(T) | C5) - mean over other tissues.
  Verdict rule: usable if rho_p(S or D | C5) on the summary has q < 0.05 for a label, holds out of oncology and with class, and has the same sign on the Open Targets label, and is not matched by B1 (delta CI above 0). "Matches baseline" if B1 shows the same with CI spanning the difference. Otherwise not usable.

USE 4. Cis-regulatory evidence.
  Hybrid references: Wang, Starr & Fraser 2024 eLife (GSE232948: hybrid lines Hyb1, Hyb2; CM, HP, MN, PP, SKM, RPE; allele counts on GRCh38 and PanTro6). Agoglia 2021 (GSE144825: iPS and cortical spheroids, bulk allele counts).
  a_gk = log2((h + 1) / (c + 1)) on sums over the replicates of type k, computed on each reference, averaged, centred on the type median. Scorable with >= MIN_ALLELIC allelic counts in the type on both references.
  Golds: L_k = -|a_gk| (cis level agreement in type k); P = -SD over types of a_gk (cis effect that does not change across types, i.e. no cis change of pattern) on genes scorable in >= 4 types.
  There is no chimpanzee in any finished fit and no brain or pluripotent fit, so the closest available readouts are human-macaque: liver H+Monkey (hepatocytes against HP), eye H+M. mulatta melanocytes (against RPE), testis H+Macaque (no matched type, negative context), and islet H+Mouse (against PP, a mouse row).
  Tests: rho_p(X, gold | C_pat) for X in S, D, B1, level gap (|x_human - x_macaque| in the matched type, median-centred, sign flipped), identity; 2,000 block draws. Tissue contrast: rho_p(X_T, L_matched) - rho_p(X_T, mean L of the other types).
  Verdict rule: usable if S on P or matched L has CI above 0 in at least 2 contexts and its tissue contrast has CI above 0 in at least 1, beyond the level gap and B1. Not usable otherwise; the reference table is kept for a chimpanzee fit (P14).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "analysis"
OUT = RES / "Y6"
DATA = ROOT / "data" / "analysis" / "Y6"
SEED = 20261102

GRID_Q = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
BUDGET_Q = 0.05
MIN_POS = 15
DET_EXPR = 0.05
DET_ABSENT = 0.005
ID_HIGH = 80.0
DRAWS_PR = 500
DRAWS_RHO = 2000
MIN_ALLELIC = 30
MOVE_BP = 2000
MOVE_CDS = 0.10

# fit -> dict(table, h1 genes file or T11 readout, forward files per seed, species label -> ortholog-table prefix, human label)
SEEDS_X7 = {"liv_4sp": [None, 43, 44, 45, 46], "test_3sp": [None, 43, 44, 45, 46], "liv_hmouse": [None, 43, 44, 45, 46],
            "liv_hmonkey": [None, 43, 44], "liv_hzeb": [None], "test_hmac": [None], "test_hmouse": [None, 43, 44, 45, 46]}
FITS = {
    "liv_4sp": dict(table="liver", kind="x7", species={"Monkey": "mmulatta", "Mouse": "mmusculus", "Zebrafish": "drerio"}, human="Human"),
    "test_3sp": dict(table="testis", kind="x7", species={"Macaque": "mmulatta", "Mouse": "mmusculus"}, human="Human"),
    "bone_marrow": dict(table="bone_marrow", kind="p12", species={"mouse": "mmusculus", "mouse_lemur": "mmurinus"}, human="human"),
    "vz_full": dict(table="van_zyl", kind="t11", t11="full", species={"macaca_fascicularis": "mfascicularis", "macaca_mulatta": "mmulatta", "mouse": "mmusculus", "pig": "sscrofa"}, human="human"),
    "liv_hmouse": dict(table="liver", kind="x7", species={"Mouse": "mmusculus"}, human="Human"),
    "liv_hmonkey": dict(table="liver", kind="x7", species={"Monkey": "mmulatta"}, human="Human"),
    "liv_hzeb": dict(table="liver", kind="x7", species={"Zebrafish": "drerio"}, human="Human"),
    "test_hmac": dict(table="testis", kind="x7", species={"Macaque": "mmulatta"}, human="Human"),
    "test_hmouse": dict(table="testis", kind="x7", species={"Mouse": "mmusculus"}, human="Human"),
    "kidney": dict(table="kidney", kind="y1", y1="kidney", species={"Mouse": "mmusculus"}, human="Human"),
    "islet": dict(table="testis", kind="y1", y1="islet", species={"Mouse": "mmusculus"}, human="Human"),
    "vz_hmouse": dict(table="van_zyl", kind="t11", t11="hmouse", species={"mouse": "mmusculus"}, human="human"),
    "vz_hmmul": dict(table="van_zyl", kind="t11", t11="hmmul", species={"macaca_mulatta": "mmulatta"}, human="human"),
    "vz_hmfas": dict(table="van_zyl", kind="t11", t11="hmfas", species={"macaca_fascicularis": "mfascicularis"}, human="human"),
    "vz_hpig": dict(table="van_zyl", kind="t11", t11="hpig", species={"pig": "sscrofa"}, human="human"),
}
JOINT = ["liv_4sp", "test_3sp", "bone_marrow", "vz_full"]
SCORES_U1 = ["S_row", "S_gene", "D_neg", "zero_s", "zero_gap", "B1_neg", "id_neg"]
BASELINES_U1 = ["zero_s", "zero_gap", "B1_neg", "id_neg"]
LABELS_PRIMARY = ["T_lowconf", "T_later", "T_newasm"]
LABELS_SECONDARY = ["T_goc", "T_wga", "T_no3utr", "T_moved", "T_tsl"]
AREA_TISSUE = {"hematology": "bone_marrow", "immune": "bone_marrow", "metabolic": "liver", "endocrine": "islet", "ophthalmology": "eye"}
TISSUES = ["liver", "testis", "kidney", "islet", "bone_marrow", "eye"]
ASE_TYPES = ["CM", "HP", "MN", "PP", "SKM", "RPE"]
BENCH = ["track", "task", "setting", "method", "metric", "value", "ci_lo", "ci_hi", "n", "rank_direction"]
