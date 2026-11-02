# Z1: count-dependence diagnostics and recipe rounds on new cohorts

Chimera's product is the gate shift D = alpha - prior, not the embedding. The target needs no gold: D should
follow the run's own count evidence. Blind: no gold, test or disease result is opened here.

`diagnostics.py` per run: rho(D, S_g_ema) on genes with count support (|S_g_ema| >= 1e-6), raw and partial on
expression; fraction with count support; tails on the raw prior; saturation; NaN; active z dims; adversary;
mixing; learning-rate freeze epoch. `curves.py` prints the same over training.

## Rule (amendment log, row "2 or more seeds", binding), fixed before each round
A seed is healthy if: finite alpha, z and S_g_ema; 10/10 z dims with Var(z_mean) >= 0.01; alpha - raw prior
beyond +0.05 and beyond -0.05 in at least 2% of genes each; alpha > 0.95 and alpha < 0.05 each in at most 10%
of genes (genes above 0.95 are flagged). Unhealthy seeds are discarded. A row qualifies with at least 2 healthy
seeds. Among qualifying rows of a cohort, the highest mean rho(D, S_g_ema) over healthy seeds wins, ties
(difference < 0.005) to the lower row number. Mixing, label transfer and saturation are reported, not ranked.
At most one round of seeds per row: no seed is added to change a ranking.

## Rounds
Round 1, Van Zyl H+Pig and H+M. fascicularis (`round1_vz_pairs.tsv`): reasons in the private ledger Z1_retune.md.
