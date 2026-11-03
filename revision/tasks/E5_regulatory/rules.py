"""E5: regulatory and gene-architecture properties of deployment (EXPLORATORY). Definitions fixed before any association.

Exploratory throughout; nothing here changes a status, row, modifier or H1/Z4 table. Vocabulary: CHIMERA_MECHANISM.md.
Question: which regulatory and gene-architecture features go with deployment above vs below the protein prior, at equal
protein identity, expression and detection? Scope is regulation; protein properties are E2, cell types E3, disease E4.
The gap between the two read-outs is X8 and is not analysed here.

Read-outs and universe: exactly E2 (rules and loader reused from revision.tasks.E2_protein):
  D_can   canonical gate (main model), H1 per-gene tables, column D_raw.
  D_free  per-gene gate, Z4/h1/genes (kept seeds of Z4/kept.json), column D_raw.
  S       S_ema of the canonical consensus, beside.
  B1      pseudobulk profile correlation (H1 column), the simple comparator beside every row.
Universe: count-supported in both tables; 26 cohorts, 10 tissues.

Statistic: partial Spearman rho_p (T4_3 stats.design natural-spline design of covariate ranks). Covariates:
  C     pi_raw, expr, detection, log protein length (as E2; primary).
  Ctau  C + cell-type tau of the cohort: does the property act beyond having a cell-state pattern.
  Cbrd  C + GTEx tau (E2 props): beyond organism-wide breadth of expression (the housekeeping axis; key for CpG, HK, TATA).
  Cgc   C + promoter GC and promoter CpG o/e: for phyloP and TATA rows, sequence composition held.
  Cpub  C + log PubMed count (Y2): for annotation-depth-sensitive rows (isoforms, TF in-degree, TF targets, PRC2 sets).
  Ccls  C + TF, homeodomain TF, mitochondrial, enzyme indicators (E2 props): beyond protein class (developmental TFs are
        Polycomb targets with large regulatory domains).
Uncertainty: paralog-family block bootstrap, 300 draws (E2). BH over every (cohort, property, read-out) cell on C.

Generality (E2 rule, unchanged): tissue value = median over the tissue's cohorts. GENERAL on a read-out if one sign in
>= 8 of 10 tissues and BH q < 0.05 in >= 1 cohort of >= 5 tissues; ROBUST = GENERAL on both gates, same sign; ONE-GATE;
NEGATIVE. |median| < 0.03 is "general but small". Species-pair robustness: tissue-free medians over the primate-pair
cohorts (human with chimpanzee, macaque, marmoset, M. mulatta, M. fascicularis; liver H+Monkey; testis H+Macaque; eye
primates), the human-mouse cohorts, and the rest (pig, zebrafish, lemur, multi-species), reported per row.

Properties (one row per human gene; Ensembl 110 canonical transcript unless stated):
 Promoter class and sequence (Y2 annotation reused: promoter [-1000, +200]):
  promoter_cgi      UCSC CpG island overlaps the promoter (binary)
  cpg_oe_promoter   CpG observed/expected in the promoter
  tata              EPDnew TATA motif in [-35, -20] (binary; sparse, low power)
  housekeeping      Eisenberg and Levanon 2013 housekeeping list (binary)
  bidirectional     another gene's canonical TSS on the opposite strand within 1 kb, head to head (binary)
  phylop_promoter, phylop_utr5, phylop_utr3   mean Zoonomia 241-way phyloP of the region; phylop_cds as coding comparator
 Regulatory complexity:
  log_great_domain  GREAT basal-plus-extension domain (basal -5 kb/+1 kb around the canonical TSS, extended to the
                    nearest basal domains up to 1 Mb; protein-coding genes), log bp
  n_dels_domain     ENCODE cCRE V4 distal enhancer-like (dELS) elements in the domain, log1p
  n_ccre_50kb       all ENCODE V4 cCREs within +-50 kb of the TSS, log1p (local, domain-free)
  n_pels_2kb        proximal enhancer-like (pELS) within +-2 kb, log1p
  abc_links_mean    Y2: ABC enhancer links averaged over biosamples
  tf_indegree       distinct TFs regulating the gene in DoRothEA A-C (log1p); CollecTRI as sensitivity (tf_indegree_ct)
 Transcript architecture (Ensembl 110 GTF, all annotated transcripts of the gene):
  n_transcripts     log1p transcripts; n_pc_transcripts log1p protein-coding transcripts
  n_tss             distinct transcript starts more than 500 bp apart (log1p): alternative promoters
  n_tes             distinct transcript ends more than 500 bp apart (log1p): alternative 3' ends
  n_exons           exons of the canonical transcript (log1p)
  log_gene_span     genomic span of the gene
  log_length_utr5, log_length_utr3   Y2 canonical UTR lengths
 Post-transcriptional:
  mir_cons_sites    TargetScan 8 conserved sites of broadly conserved miRNA families (Family Conservation >= 2) on the
                    representative transcript, log1p; mir_cons_families: number of such families
  mir_density       conserved sites per kb of 3'UTR (sensitivity given 3'UTR length)
  are_density       AUUUA pentamers per kb of the canonical 3'UTR (AU-rich elements: TTP/HuR targets)
  are_class2        >= 1 WWAUUUAWW nonamer in the canonical 3'UTR (binary)
 Chromatin (MSigDB 2025.1 C2 CGP, gene symbols via HGNC):
  prc2_targets      BENPORATH_PRC2_TARGETS (Suz12, Eed and H3K27me3 bound in hESC)
  es_h3k27me3       BENPORATH_ES_WITH_H3K27ME3
  bivalent_es       MIKKELSEN_ES_HCP_WITH_H3K4ME3_AND_H3K27ME3 when present, else the closest ES HCP bivalent set (named
                    in the output)
 Genomic context:
  gene_density      protein-coding canonical TSSs within +-500 kb (log1p)

Target sets (family scan as E2): DoRothEA A-C regulons per TF with >= 15 universe targets: mean C-residual standardised
rank of the read-out in the regulon vs the rest; Welch p; generality rule as above. Targets that are themselves the TF
are dropped.

Neighbourhood (co-deployment): pairs of universe genes on the same chromosome with canonical TSSs within 1 Mb, not
paralogs (any Compara 110 human within-species paralog pair removed) and not overlapping genes. Statistic: Spearman of
C-residual read-outs over pairs, by TSS-distance bin (< 10 kb, 10-50, 50-200, 200-1000 kb) and for adjacent pairs
(consecutive universe genes); null: each gene paired with a random universe gene on another chromosome (200 draws).
TAD contrast: ENCODE contact domains in 7 biosamples; a pair is same-TAD if one domain contains both TSSs in >= 4
biosamples, cross-TAD if in 0 (both TSSs inside some domain in >= 4). Same minus cross within distance bins 20-100 and
100-500 kb. Bidirectional pairs reported separately.

Predictions (mechanism: S_g rewards a shared cell-state pattern, penalises level gaps; the per-gene gate reads S_g against
pi; the canonical gate projects it onto features and amplifies expression):
 P1 CpG-island promoters and housekeeping genes BELOW the prior on both gates (broad, flat, little pattern gain), most of it
    removed by Ctau and Cbrd. The intuitive alternative (CpG/housekeeping keep shared deployment above the prior) is the one
    tested against. Y2 saw a CpG-poor lean in four cohorts.
 P2 TATA promoters above (tissue-specific), mostly via tau; low power.
 P3 Promoter and 5'UTR phyloP small positive (<= 0.05) on both; 3'UTR near 0; CDS phyloP near 0 (X4: sequence constraint is a
    separate axis). Smaller with Cgc.
 P4 Regulatory complexity (GREAT domain, dELS count, ABC links, TF in-degree) ABOVE on both gates; partly tau and TF class.
 P5 Long 3'UTRs and conserved miRNA target load ABOVE; miRNA density given 3'UTR length near 0.
 P6 ARE density: no direction claimed.
 P7 Alternative promoters (n_tss) small positive; transcript count near 0 at equal PubMed.
 P8 PRC2/H3K27me3/bivalent targets ABOVE on both, shrinking with Ccls (developmental TFs).
 P9 Non-paralog neighbours co-deploy: residual correlation > 0 within 50 kb, decaying with distance; same-TAD > cross-TAD at
    matched distance; bidirectional pairs highest. B1 shows the same direction.
 P10 Regulons: targets of signalling and inflammatory TFs (STAT, NFKB/RELA, AP-1, SMAD, IRF) above; of E2F, MYC, NRF1, YY1
    (proliferation, housekeeping) below.

Named genes: E2 rule (count-supported, nameable, not saturated, seed-stable in both tables, same sign, >= 3 cohorts of >= 2
tissues, sign identical wherever named).
"""
from __future__ import annotations

DRAWS = 300
MIN_FAMILY = 15
NULL_DRAWS = 200
GREAT_BASAL = (5000, 1000)
GREAT_MAX = 1_000_000
TSS_SEP = 500
BIDIR_MAX = 1000
DIST_BINS = [0, 10_000, 50_000, 200_000, 1_000_000]
TAD_BINS = [(20_000, 100_000), (100_000, 500_000)]
TAD_MIN_BIOSAMPLES = 4

PRIMATE = {"brain_dlpfc_hchimp", "brain_dlpfc_hmac", "brain_dlpfc_hmac_own", "brain_dlpfc_hmarm", "liv_hmonkey", "test_hmac",
           "vz_hmfas", "vz_hmmul", "vz_primates"}
HMOUSE = {"liv_hmouse", "test_hmouse", "kidney", "islet", "muscle_ageing", "tumour_zilionis", "brain_m1_hmouse", "vz_hmouse"}

CONTINUOUS = {
    "cpg_oe_promoter": "CpG o/e of the promoter",
    "phylop_promoter": "mean phyloP, promoter", "phylop_utr5": "mean phyloP, 5'UTR", "phylop_utr3": "mean phyloP, 3'UTR",
    "phylop_cds": "mean phyloP, CDS (coding comparator)",
    "log_great_domain": "GREAT regulatory domain, log bp", "n_dels_domain": "dELS in the GREAT domain, log1p",
    "n_ccre_50kb": "cCREs within 50 kb, log1p", "n_pels_2kb": "pELS within 2 kb, log1p", "abc_links_mean": "ABC links",
    "tf_indegree": "DoRothEA A-C regulators, log1p", "tf_indegree_ct": "CollecTRI regulators, log1p",
    "n_transcripts": "transcripts, log1p", "n_pc_transcripts": "protein-coding transcripts, log1p",
    "n_tss": "alternative starts, log1p", "n_tes": "alternative ends, log1p", "n_exons": "exons, log1p",
    "log_gene_span": "gene span, log bp", "log_length_utr5": "5'UTR length, log", "log_length_utr3": "3'UTR length, log",
    "mir_cons_sites": "conserved miRNA sites, log1p", "mir_cons_families": "conserved miRNA families, log1p",
    "mir_density": "conserved miRNA sites per kb 3'UTR", "are_density": "AUUUA per kb 3'UTR",
    "gene_density": "protein-coding genes within 500 kb, log1p",
}
BINARY = {
    "promoter_cgi": "CpG-island promoter", "tata": "TATA box (EPDnew)", "housekeeping": "housekeeping (Eisenberg-Levanon)",
    "bidirectional": "bidirectional promoter", "are_class2": "class II ARE nonamer",
    "prc2_targets": "PRC2 targets in hESC", "es_h3k27me3": "H3K27me3 in hESC", "bivalent_es": "bivalent HCP in ES",
}
# sensitivity covariate sets beyond C and Ctau, per property
EXTRA_COVS = {
    "Cbrd": ["gtex_tau"],
    "Cgc": ["gc_promoter", "cpg_oe_promoter"],
    "Cpub": ["log_pubmed_h"],
}
EXTRA_FOR = {
    "Cbrd": list(CONTINUOUS) + list(BINARY),
    "Cgc": ["phylop_promoter", "phylop_utr5", "tata", "promoter_cgi", "housekeeping"],
    "Cpub": ["n_transcripts", "n_pc_transcripts", "n_tss", "n_tes", "tf_indegree", "tf_indegree_ct", "prc2_targets",
             "es_h3k27me3", "bivalent_es", "housekeeping", "abc_links_mean", "mir_cons_sites"],
}
CLASS_COVS = ["tf", "tf_homeodomain", "hpa_loc_mitochondria", "enzyme"]
JOINT = ["promoter_cgi", "housekeeping", "tata", "phylop_promoter", "log_length_utr3", "mir_cons_sites", "are_density",
         "n_tss", "n_transcripts", "log_great_domain", "n_dels_domain", "tf_indegree", "prc2_targets", "gene_density"]
