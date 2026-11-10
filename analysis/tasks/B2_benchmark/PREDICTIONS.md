# B2 predictions, written before any setting met a gold

What I knew when writing these: X4 (S = pattern gain minus level cost), X6 (label noise on liver and testis with its numbers), Y4 (two survivors), Y1's and H1's cards and tables, T5.6 liver tool values, the mechanism file. I have seen no Track B or Track C value, and no Track A result.

## Track B
- **PB1, label noise, liver.** S_g and B1z do not read labels, so their gold rho_p stays at `full`; B1b, B2b fall in steps with the noise, and at 25% and above S_g is ahead of B1b with CI above 0. Sdir falls much less than B1b, so S_g - Sdir stays under 0.05.
- **PB2, label noise, testis and kidney.** Fits read labels through conditional MMD. S_g falls little. B1 and B1z stay ahead of S_g on testis pattern at 10-25% (as X6); only at 50% may S_g catch B1b. Kidney: S_g and B1b close, Sdir at least level.
- **PB3, near confusion.** Confusing similar types hurts direct baselines less than random noise at the same 25% (the profile keeps its shape), so the X6 lead of S_g on liver shrinks: S_g - B1b under 0.05.
- **PB4, coarse labels.** Baselines do not fall and may rise on liver, as X6 found. On testis and kidney, where there are more real types, B1b falls with coarsening. Chimera moves little.
- **PB5, over-fine labels.** Baselines lose little or nothing on rho_p but their agreement falls, because the profile is on other groups. Conditional MMD on fine labels leaves S_g about stable.
- **PB6, species_diff.** Direct baselines lose half their shared types and fall in rho_p and agreement; kidney with 16 types least, liver (K 4 to 2) most, where B1 becomes uninformative. Chimera nearly unchanged. Chimera leads here if anywhere among label settings.
- **PB7, no labels.** On liver S_g ties the annotation-free direct workflow (X6). On testis and kidney the no-label form falls below labelled S_g in rho_p and direct Leiden-matching stays level or ahead, since B1z on a no-label latent inherits its mixing.
- **PB8, type missing.** Liver's global MMD has to place the missing type somewhere, so S_g and D move more there than on testis and kidney, where conditional MMD skips it. Direct baselines lose one group and stay close to full.
- **PB9, rare types.** Direct baselines keep types through the fallback but with noisy profiles, and fall; S_g, an average over cells, shifts toward the two large types and falls less on pattern where those types carry it (liver), more where the pattern is across many types (kidney).
- **PB10, species sizes.** 1:1 and 1:5 change little. At 1:20 the small species has about 12 cells per batch; integration and S_g suffer, D loses the S_g link, and direct baselines lead wherever the small species still has types with at least 30 cells.
- **PB11, cells.** At 50% nothing moves much. At 10% direct per-type scores on small types and Chimera's fit (few steps per epoch at fixed epochs, liver about 3 per epoch) both drop; I expect Chimera to drop more on liver, where one tenth is 809 cells.
- **PB12, depth.** Thinning raises the 20-cell mask, so S_g loses low genes and falls more than B1 and B1z. Depth is S_g's weakest setting. Sdir, with no mask, holds up better. D falls with S_g but the feature-predictable part keeps its agreement.
- **PB13, few donors.** Direct baselines and S_g fall by similar small amounts; kidney less than testis.
- **PB14, extra batch.** Model has a batch term and pseudobulk averages libraries: all methods near full.
- **PB15, chemistry.** A species-wide gene factor is a level gap. B1 is insulated (profile centred within species); S_g pays through the level cost, B1c and Lfc fall most on the level gold. B1 ahead of S_g here.
- **PB16, stability against validity.** D has the highest agreement among Chimera readouts on testis and kidney (features) with little gold signal; on liver its agreement is low (two training states). The two metrics disagree often, and no method wins both across tracks.

## Track C
- **PC1, pattern.** B1 or B1z lead on most golds; S_g ties on liver, behind on testis, near B1 on kidney and pancreas; D behind S_g everywhere and near 0 on testis.
- **PC2, level.** B1c, B2b and Lfc lead; S_g and D near 0, except liver and kidney EVEE where the level term may leave S_g small and positive.
- **PC3, count support.** On count-supported genes S_g improves and approaches B1. Sdir close to S_g everywhere (PC9 of Y1's kind): what carries is the margin.
- **PC4, identity.** At near-1 identity, pi gives no order and every count-based readout still does. On low identity S_g and B1 hold as well.
- **PC5, expression.** S_g best in the top quartile; bottom quartile mostly masked, so D there is feature extrapolation, with B3a about as good.
- **PC6, multi-species models.** The Figs 4-6 rule: on 4-sp liver S_g keeps sign on TS/TMS; D's sign holds too.
- **PC7, Y4 survivors on new tissues.** `pihi__tau_T2` positive on the kidney and pancreas pattern golds and not beyond baselines; `expr_T2__tau_T3` on EVEE kidney smaller than liver and with CI including 0, as it failed on testis.
- **PC8, lists.** No readout separates C1 or C2 strongly at equal expression; V3 and V2 run opposite to the plan for D (X3); B1 the same direction. Bone marrow: as T12.3, against Shay the pattern gain wins for D and S_g until variability is matched.

## Tools
- **PT1.** S_deploy and C_cos below S_g on pattern golds on the shared genes, and well below B1; CAMEX on few genes and unstable.
- **PT2.** SATURN falls under label noise (metric learning reads labels). CAMEX, trained without labels, stays stable on noise but its coverage shifts.
- **PT3.** GeneSpectra's specificity score behaves like B2b. SAMap's refined pair score behaves like a coarse B1z. CACIMAR, marker-based, covers few genes.

## Overview
- **PO.** On the fixed aggregate, a direct baseline (B1, B1c or B2b) ranks first overall. S_g is in the top half and first on label-damage tasks on liver. D ranks in the lower half on gold tasks; static scores last on agreement by rule (not ranked) and low on gold.

## Track I (integration on the new cohorts, added by coordinator's instruction; integration is the substrate, not Chimera's aim)
- **PI1.** scANVI, which uses labels, first or near first on composite in the new cohorts, as on the original seven and T11.4; Chimera in the upper half of the panel, highest or near highest on species mixing and below scANVI and scVI on cell-type conservation.
- **PI2.** No old value moves: scaling is per cohort.
- **PI3.** SATURN and CAMEX through 1:1 orthologs score lower than their native rows on comparable cohorts would suggest (fewer genes).
