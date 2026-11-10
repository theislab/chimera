"""X4 (EXPLORATORY): definitions fixed before any X4 test was run. All tests here were chosen after
G3 (row, modifiers and Family 3 values were known), so none of them is pre-specified and none
changes a scoreboard output, status, row or modifier.

Readouts. D_raw = mean over seeds of (alpha - pi_raw) minus the cohort median (the card's D);
D = the T3.3 consensus (clamped prior on testis; identical to D_raw on liver H+Mouse); S = S_ema.
Direct baselines B1b, B1c, B1z, B2b as scored in T4.3 (unresidualised columns).

Golds. Liver: G_H1 (EVEE level, covariates C), G_H3 (TS vs TMS pattern, C_pat). Testis: G_H2 (EVEE
level, C), G_H4 (Murat 8 held-out species pattern, C_pat + H4_nsp), G_H4_primates (C_pat + H4_npri).
Not available at the time of X4: bone-marrow D (T12.2 still training; the Shay COE gold is left
unjoined so that P12 stays blind) and disease Q_tg (T7.3, T7.9 still running).

Tertiles (global, per cohort, over genes with the value defined; 1 = lowest, 3 = highest).
Constraint axes: dN/dS (constrained = tertile 1), pi_raw (constrained = tertile 3). Item 1 classes:
    constrained_lowD  : constrained, D_raw tertile 1
    relaxed_highD     : relaxed, D_raw tertile 3
    constrained_highD : constrained, D_raw tertile 3 (concordant control)
    relaxed_lowD      : relaxed, D_raw tertile 1 (concordant control)
The same on the residual scale rD = rank(D_raw) residualised on spline(C), because raw D is largely
its covariates (testis rho(D_raw, pi) -0.86).

Item 3 classes (residual scale on C_pat, over genes with B1b defined):
    B1bhigh_Dlow  : B1b tertile 3, D tertile 1  (B1b conserved, D divergent)
    B1blow_Dhigh  : B1b tertile 1, D tertile 3  (B1b divergent, D conserved)
    B1bhigh_Dhigh, B1blow_Dlow : concordant controls.  Same with S in place of D.
Level vs pattern over the K shared types, d_k = x_Mouse,k - x_Human,k, each d_k centred on its median
over genes detected in both species (composition offset):
    mean_k d_k^2 = shift^2 + shape^2, shift = mean_k d_k, shape^2 = var_k d_k; frac_level = shift^2 / msd.
spread = mean over the two species of the SD of x across shared types (how much profile a correlation has).

Matched contrast. 40 nearest pool genes in rank space of the covariates; excess = mean over set genes of
(value - mean of its 40 neighbours), in SD of the value over the universe; CI from resampling ESM blocks
of the set genes (2,000); p from 1,000 draws of one neighbour per set gene, two-sided, as X1-X3.

Naming rule (item 4). A gene may be named only if all hold: T2.5 margin_support; D_raw has the sign of its
consensus in every seed; S has the sign of its consensus in every seed; sign(D_raw) = sign(S); S unmasked;
counts in every species of the cohort. mtDNA-encoded genes are never named on liver (X2 finding 3, assay).

Amendment, 2026-11-02, after the first vignette table was read and before any gene was reported: on the
residual scale a gene can sit in a low class with positive raw D and S (testis, where D_raw is about -pi), so S
does not back its class. For a named gene in a class, sign(S) must also equal the class direction (low class
S < 0, high class S > 0); column ``class_backed``. The first table is not used.

Vignettes (item 5). Within each class of item 1 (constrained_lowD, relaxed_highD on rD) and item 3
(B1bhigh_Dlow, B1blow_Dhigh), per cohort (liver H+Mouse, testis H+Mouse), genes passing the naming rule are
ranked by |rD| then |S|; the top gene per class and cohort is the vignette, the top 5 are tabled.

Verdicts on X1-X3 (item 7), for one finding in one tissue; a "gold" is an independent gold of the tissue.
    keep   : S agrees in direction (block CI excludes 0) and at least one gold agrees in direction
             (block CI excludes 0), no gold contradicts it with a CI excluding 0.
    reword : S agrees but golds are null or mixed, or the wording must change under REVERSED-TESTIS or
             to say "level"/"state"/"class-expected", or the finding holds only on features.
    drop   : S disagrees or is null, or golds contradict it.

Direct margin (item 9). Sdir is the T2.5 margin computed with no neural model: mu1 = pooled per-type rate
(mean over species of each species' rate in the type; a species' own rate for a type it alone has) x library,
mu0 = per-species rate x library, NB with gene-wise moment dispersion; same weights w_sk, so
Sdir = sum_k Cdir_k exactly. It needs cell labels; S does not.

Later additions, all after results were seen: disagree_head_to_head (symmetric test), species_direct and the zebrafish check,
disease (read-only from T7.3), relations (owner), mechanism and mech_adds (tests of the mechanism notes), verify (independent check).
"""

from __future__ import annotations

CONSTRAINT_AXES = {"dNdS": dict(col="dNdS", constrained=1, relaxed=3),
                   "pi": dict(col="prior_raw", constrained=3, relaxed=1)}
C = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]
C_PAT = C + ["tau_shared"]
GOLDS = {
    "liver": [("G_H1", "EVEE level", C), ("G_H3", "TS vs TMS pattern", C_PAT)],
    "testis": [("G_H2", "EVEE level", C), ("G_H4", "Murat held-out pattern", C_PAT + ["H4_nsp"]),
               ("G_H4_primates", "Murat primates pattern", C_PAT + ["H4_npri"])],
}
COHORTS = ["liv_hmouse", "test_hmouse", "liv_4sp", "test_3sp"]
PRIMARY = ["liv_hmouse", "test_hmouse"]
BASELINES = ["B1b_raw", "B1c_raw", "B1z_raw", "B2b"]
NN = 40
DRAWS = 2000
NULL_DRAWS = 1000
SEED = 20261102
