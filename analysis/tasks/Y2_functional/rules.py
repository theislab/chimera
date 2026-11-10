"""EXPLORATORY, Family 3: does the deployment axis have consequences outside expression?

Definitions fixed before any association between a Chimera readout or baseline and a
functional annotation is computed. Nothing here changes a status, row or modifier of the plan.

Readouts. D (T3_3_consensus consensus; clamped prior on testis), D_raw, S (S_g EMA, mean over seeds). Beside them,
no-model terms from X4_where_differs: Sdir, Sdir_pattern, Sdir_level. Baselines: B1b_raw, B1c_raw, B1z_raw, B2b, B3a.
Replication tables (bone-marrow cohort, kidney P13_kidney) carry D, S_ema and B1b (kidney also B2b), nothing else.

Universes. U_ms (primary): T2_5_celltype_margin margin_support. That flag selects on sign(S) = sign(D) and
|S| >= median |S|, so inside it S is selected and S-vs-baseline differences favor S (X4_where_differs disclosure). U_un
(unmasked, |S| >= 1e-6) is not selected on sign or size, and is the universe for any S-versus-baseline
head-to-head claim. Every table has both. Masked genes are never used.

Covariates. C_Y2 = pi_raw, expression (species-balanced), detection, log CDS length, tau_shared, dN/dS, LOEUF,
log PubMed human, log PubMed mouse. Natural cubic spline df 4 of each rank at n >= SPLINE_MIN, otherwise linear
ranks (a 100-gene system universe cannot carry 37 spline columns). For regulatory outcomes, C_REG = C_Y2 without
the PubMed terms, plus the GC of the region and log region length. For the regulatory-vs-coding comparison
C_CMP = pi_raw, expression, detection, log CDS length, tau_shared, plus GC and log length of the outcome region.

(a) Phenotype translation. A = gene has a mouse null phenotype (MGI null single-marker or IMPC) and a human
Mendelian annotation (HPO on OMIM disease, Mendelian gene-disease pair). For system k, U_k = A and (M_k or H_k);
y_k = M_k and H_k. Statistic: rho_p(X, y_k | C_Y2) on U_k, block bootstrap over a global gene partition.
Decisive contrast, for each score X:
  Delta_match(X) = 1/2 { [rho(X_liver, y_hepmet) - rho(X_liver, y_repro)] + [rho(X_testis, y_repro) - rho(X_testis, y_hepmet)] }
Positive = tissue-matched concordance tracks the score more than mismatched. Bootstrap draws share one gene
partition across cohorts. Second reading: head-to-head Delta_match(S) - Delta_match(B1b), Delta_match(D) - Delta_match(B1b),
on genes where both are defined (U_un is the universe that counts).
Sensitivities: hepatic only and male-reproductive only systems; IMPC-only mouse side on IMPC-tested genes (no
investigator selection); ClinGen moderate-or-better genes; adding pleiotropy (n_mp_sys, n_hp_sys); adding ESM-free
hepatocyte enrichment on liver; logistic interaction logit(H_k) ~ M_k + X + M_k:X + C_Y2 on A, which does not condition
on M_k or H_k (the U_k design conditions on a collider); without PubMed terms, to show what study depth moves.
Replication: bone marrow (hematopoietic-immune) and kidney (renal) against the same four systems; diagonal minus
mean off-diagonal per tissue.

(a2) Viability. Mouse lethal (MGI homozygous null under lethal terms, IMPC lethal) or viable (IMPC viable and never
annotated lethal). y_L = lethal and LOEUF < 0.35 on genes lethal or LOEUF < 0.35 (LOEUF out of the covariates);
y_E = lethal and DepMap common essential on genes lethal or common essential. rho_p(X, y | C_Y2 minus LOEUF for y_L).

(b) Regulatory. Canonical transcript (Ensembl 110). Promoter [-1000, +200] of the TSS, 5'UTR, 3'UTR, CDS. Mean Zoonomia
241-way phyloP per base; fraction of bases >= 2.27. Outcomes: phyloP in promoter, 5'UTR, 3'UTR (C_REG). CpG island over
the promoter window; TATA class (TATAWAW in -35..-20 of the EPDnew TSS, genes with EPDnew promoter); ABC enhancer links
per biosample (Nasser 2021, all biosamples; liver-ENCODE and hepatocyte-ENCODE for liver). rho_p(X, outcome | C_REG), same
blocks, for every X. Regulatory vs coding: on C_CMP, for each X, rho_p(X, promoter) and rho_p(X, 3'UTR) against rho_p(X, CDS
phyloP), rho_p(X, -log dN/dS) and rho_p(X, -LOEUF), with differences on the same draws.

(c) Discordant genes. Within a cohort, rS = residual rank of the score on C = pi, expression, detection, length (X4_where_differs scale).
Class PC-low: pi top tertile, residual tertile 1 of score. Class PD-high: pi bottom tertile, residual tertile 3.
Chimera classes: on D, in margin_support with S of the class sign (primary); on S, unmasked. Baseline classes: on B1b_raw, B1c_raw,
in unmasked genes. Also the fixed X2_liver_biology list (64 genes, 4-sp rule, divergent_conserved.csv). Excess of each outcome against the 40 nearest
class-free genes in rank space of pi, expression, detection, length, tau, hepatocyte enrichment on liver, log PubMed
(X4_where_differs ``matched``); margin-supported pool for Chimera classes, unmasked pool for baselines. Outcomes: y_system (matched
system, restricted to U_k), M_any, H_any, mouse lethal, phyloP promoter, 5'UTR, 3'UTR, CpG, n_ABC. BH over each cohort x class set.

(d) Literature cases. The list (``literature_cases.tsv``) was curated from the literature blind to any readout, in a separate session with no access to
the project, and committed before any join. For each case and cohort that holds the gene: S, D, B1b, B1c, the raw pattern per species and cell type. "Recovered" by score X:
defined, and X at or below the 25th percentile of its 200 nearest genes in rank space of expression, detection, tau, pi. Masked genes are
reported as not callable for S and D. A gene is "supported by these data" if its documented difference is visible in the counts by the
rule in ``literature.py``, written before the join.

Multiplicity. p = 2 Phi(-|est| / SD(draws)). BH within: (a) primary cells and contrasts; (a2); (b); (c) as above. All exploratory, Family 3.
"""

from __future__ import annotations

SEED = 20261103
DRAWS = 2000
SPLINE_MIN = 300
COHORTS = {"liver": "liv_hmouse", "testis": "test_hmouse"}
SCORES = ["D", "D_raw", "S", "Sdir", "Sdir_pattern", "Sdir_level", "B1b_raw", "B1c_raw", "B1z_raw", "B2b", "B3a"]
C_Y2 = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length", "tau_shared", "dNdS", "LOEUF",
        "log_pubmed_h", "log_pubmed_m"]
C_BASE = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length", "tau_shared"]

# (mouse MP terms, human HPO terms); each annotated gene counts if a term is below one of these.
MP_SYSTEMS = {
    "hepmet": ["MP:0005370", "MP:0005376"],  # liver/biliary, homeostasis/metabolism
    "hepatic": ["MP:0005370"],
    "repro": ["MP:0005389"],  # reproductive system phenotype
    "male": ["MP:0001925", "MP:0001146"],  # male infertility, abnormal testis morphology
    "heme": ["MP:0005397", "MP:0005387"],  # hematopoietic, immune
    "renal": ["MP:0005367"],  # renal/urinary
    "eye": ["MP:0005391"],  # vision/eye: negative-control system, matched to no cohort here
}
HP_SYSTEMS = {
    "hepmet": ["HP:0001392", "HP:0001080", "HP:0001939"],  # liver, biliary tract, metabolism/homeostasis
    "hepatic": ["HP:0001392", "HP:0001080"],
    "repro": ["HP:0000078"],  # abnormality of the genital system (includes reproductive physiology)
    "male": ["HP:0003251", "HP:0000035", "HP:0000027"],  # male infertility, abnormal testis morphology, azoospermia
    "heme": ["HP:0001871", "HP:0002715"],  # blood, immune
    "renal": ["HP:0000077"],  # kidney
    "eye": ["HP:0000478"],  # abnormality of the eye
}
LETHAL_MP = ["MP:0011100", "MP:0008762", "MP:0010770"]  # preweaning complete penetrance, embryonic, preweaning
LOEUF_CONSTRAINED = 0.35
PHYLOP_CONSTRAINED = 2.27  # Zoonomia 5% FDR threshold
PROMOTER = (-1000, 200)
TATA_WINDOW = (-35, -20)
