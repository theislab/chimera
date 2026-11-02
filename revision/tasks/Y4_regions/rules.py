"""Y4 (EXPLORATORY, Family 3): where in gene space do D and S_g carry signal on independent golds.

Everything in this module is fixed and committed before any association between a readout or baseline and
a gold is computed for this task. All-gene and some stratified values were known from T4.3, X3, X4 and X6;
regions touched by those analyses carry ``seen_before`` in ``regions.py``. Nothing here changes a status, row
or modifier of the plan.

Split. Human chromosome (Ensembl 110 GTF). Discovery: odd autosomes. Replication: even autosomes. Genes on
X, Y, MT and unplaced scaffolds are in neither; the X is read once, for regions that replicated, as a third
holdout (meiotic sex-chromosome inactivation makes it a poor member of either half in testis).

Units. A unit is (cohort, gold, readout, region). Cohorts: the seven canonical ones. Golds: liver G_H1 (EVEE
level, C) and G_H3 (TS vs TMS pattern, C_pat); testis G_H2 (EVEE level, C) and G_H4 (Murat held-out species
pattern, C_pat + H4_nsp). Readouts: D (T3.3 consensus) and S (S_g EMA, mean over seeds). Murat primates and
D_raw are sensitivities on survivors only.

Statistic. rho_p as in section 3 of the plan, taken inside the region: mid-ranks within the region's genes of
the split, residualised on the covariate design refit inside the region. Design: spline df 4 of each covariate
rank when the region has at least SPLINE_MIN genes, otherwise intercept + linear rank of each covariate.
Block bootstrap on ESM modules (BOOT_SCREEN draws on discovery, BOOT_FINAL on carried units).

T1  within-region rho_p(X, G | C*) on genes of the region with X, G and covariates.
T2  heterogeneity: rho_p in the region minus rho_p in the rest of the split, same draws.
T3  head-to-head: rho_p(X) - max_j rho_p(B_j) on the same genes, the region's genes with every direct baseline
    of the cohort defined (U_common). J = B1b, B1c, B1z, B2b, B3a, each where defined for at least half the
    cohort's genes. A baseline defined for under half of the region's own genes is dropped from that region's J and
    recorded (J_dropped); such a unit can only be said to go beyond that baseline's coverage, not to beat it.
    Bootstrap of the minimum over j, which is conservative. Sdir (labelled no-model margin) is reported beside but
    is outside J, as X6 treated it. U_common also requires Sdir so its terms stand on the same genes.
T4  incremental: rho_p(X, G | C* + rank splines of all J) on U_common. Sdir given the same is reported beside.

Multiplicity. p = 2 Phi(-|est| / SD(draws)). BH over every unit tried on discovery, one family for each of T1,
T2, T3, T4, regions of the same gene sets in other cohorts included.

Carry rule (at most 10 units, committed to the ledger before even genes are opened). Eligible: rho_p > 0,
q_T1 < 0.05, CI lower > 0, and one of
    tier A  q_T3 < 0.05 and T3 >= 0.03          (ahead of every direct baseline)
    tier B  q_T4 < 0.05 and T4 >= 0.05 and T4 >= global T4 of the cohort-gold-readout + 0.03
    tier C  q_T2 < 0.05 and T2 >= 0.05          (clearly better than the rest of the genes)
Tiers A and B also need B1b and B1c inside the unit's J; otherwise the unit cannot be ahead of the direct baselines and is
only tier C eligible (this removes H+Zebrafish, where only B3a is defined for half the genes, and low-detection regions).
Order: tier, then mechanism-guided before wider search, then |z| of the tier's statistic. Pick greedily with: at
most 2 units per axis; no unit whose genes overlap a picked unit of the same gold's tissue with Jaccard above
0.5, whatever the cohort; at least 3 units per tissue if eligible units exist; D units are not preferred over S
units or the reverse.

Replication (even autosomes, same definitions). A carried unit replicates if rho_p has the same sign, the
one-sided BH q over carried units is below 0.05 and the CI excludes 0. It is called
    differentiates  every Delta_j on even genes has CI lower > 0 and min Delta_j >= 0.03
    baseline ahead  some Delta_j has CI upper < 0
    matches         otherwise (no difference shown; "tie" in the plan's words)
The ranking baseline is reported per j, never a baseline chosen on even genes. Independent tissue: the same
region in the other tissue's H+Mouse cohort on even genes, T1 only.

Matched controls (survivors, even genes). (1) rho_p again with spline covariates extended by total variance,
within-type variance, spread and hepatocyte enrichment or peak margin. (2) Within-region rho_p against
rho_p on 1:1 nearest-neighbour genes from outside the region, matched in rank space on expression, detection,
residual variance, spread, tau and pi, 200 re-draws of the match, block CI.

Decoys. BLOCK-DRAWN random regions at three sizes, per cohort x gold, analysed like real regions on discovery,
outside every BH family; their pass rate for T1-T4 is reported as the empirical false-positive rate of this screen.

Gap map (descriptive, no gold enters the region or the rule). For every region, on each split: mean percentile
of pi minus mean percentile of S, D and the pattern gold, given expression and detection, and the share of genes
with pi in the top cohort tertile and S in the bottom tertile of its residual.
"""

from __future__ import annotations

COHORTS = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp", "test_hmac", "test_hmouse", "test_3sp"]
PRIMARY = {"liver": "liv_hmouse", "testis": "test_hmouse"}
PARTNER = {"liv_hmonkey": "Monkey", "liv_hmouse": "Mouse", "liv_hzeb": "Zebrafish", "liv_4sp": "Mouse",
           "test_hmac": "Macaque", "test_hmouse": "Mouse", "test_3sp": "Mouse"}
C = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]
C_PAT = C + ["tau_shared"]
GOLDS = {
    "liver": {"G_H1": ("EVEE level, liver", C), "G_H3": ("TS vs TMS pattern", C_PAT)},
    "testis": {"G_H2": ("EVEE level, testis", C), "G_H4": ("Murat held-out species pattern", C_PAT + ["H4_nsp"])},
}
SECONDARY_GOLD = {"G_H4": ("G_H4_primates", C_PAT + ["H4_npri"])}
READOUTS = ["D", "S"]
BASELINES = ["B1b_raw", "B1c_raw", "B1z_raw", "B2b", "B3a"]
BESIDE = ["Sdir", "Sdir_pattern", "Sdir_level"]
SPLINE_MIN = 300
MIN_GENES = 60        # with gold and covariates, on odd and on even
MIN_BLOCKS = 8
BOOT_SCREEN = 1000
BOOT_FINAL = 5000
N_CARRY = 10
MAX_PER_AXIS = 2
MIN_PER_TISSUE = 3
JACCARD = 0.5
T3_MIN = 0.03
T4_MIN = 0.05
T4_OVER_GLOBAL = 0.03
T2_MIN = 0.05
Q = 0.05
DECOY_SIZES = [100, 300, 1000]
DECOY_N = 30
SEED = 20261102

ODD = {str(i) for i in range(1, 23, 2)}
EVEN = {str(i) for i in range(2, 23, 2)}

# Landmark species for the age proxy: deepest clade with an Ensembl 110 Compara ortholog of the human gene.
# The species panel is vertebrate-heavy and the 1:1 vocabulary removes the youngest genes; it is a proxy.
AGE_CLADES = [
    ("opisthokont", ["saccharomyces_cerevisiae"]),
    ("bilaterian", ["drosophila_melanogaster", "caenorhabditis_elegans"]),
    ("vertebrate", ["ciona_intestinalis", "ciona_savignyi", "petromyzon_marinus", "eptatretus_burgeri",
                    "callorhinchus_milii", "danio_rerio", "lepisosteus_oculatus", "oryzias_latipes"]),
    ("tetrapod", ["xenopus_tropicalis", "gallus_gallus", "anolis_carolinensis", "chelonia_mydas"]),
    ("mammal", ["ornithorhynchus_anatinus", "monodelphis_domestica", "mus_musculus"]),
]
