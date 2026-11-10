# T12.2b Bone-marrow recipe: second grid on build 2, selection rule unchanged

Cohort: `data/analysis/bone_marrow/v2/bone_marrow_unified.h5ad` (build 2 of
`analysis/tasks/T12_1_bone_marrow/build_bm.py`: no mitochondrial filter in any species, every
species x type capped at 1,150 cells so the cohort stays within the testis 3-species size;
28,239 cells x 12,417 genes, K = 3; `check_cohort` OK with `--batch-key species
--celltype-key cell_ontology_class`). This grid and this file are committed before any build-2
run is trained. The cohort was added during the revision; its recipe is fixed by this rule.

## Why a second grid (decided on training health only)

The T12.2 grid (`../T12_2_recipe/`) was trained on build 1 (128,328 cells). Diagnostics
(`../T12_2b_debug/`, runs `rev-bm-dbg-*`) read only health quantities: NaN, active z dims,
adversary accuracy, and the G-V saturation and tail fractions. No Delta-alpha statistic beyond
those fractions, no S_g, COE, baseline or biology readout was read.

1. Latent collapse and NaN were a scale effect: at 452 steps per epoch every epoch-based
   schedule (KL and FQR warm-ups, adversary start, plateau LR decay) ran 4-6 times more updates
   than in the canonical cohorts (29-102 steps). A proportional 25k draw of the same cells kept
   10/10 dims with no NaN in all six diagnostic rows. Build 2 fixes this in the data.
2. Saturation was not a scale effect: the same 25k draw with the base row still had 22% of genes
   at alpha > 0.95. It is a depth effect: thinning every count to 1/3 (mean species-median
   library 1,107, the testis 3-species value 1,097) brought it to 2.4% with the same recipe.
   The fit reward scales with counts per cell, so the anchor that sets Delta-alpha's scale (Law 2)
   was set at canonical depth. Build-2 depth ratio: mean of species-median libraries 3,672 /
   1,097 = 3.35.
3. A symmetric anchor did not meet both G-V alpha criteria on the 25k draw: at anchor 0.003
   saturation was 5.8% with 0.3% of genes below -0.05 (epoch 50, still rising); at 0.01, 0.35%
   and 0.43% (epoch 60). The negative tail falls faster
   than saturation (the tail test is against the clamped prior: `audits/R5` M2). The
   negative-side anchor factor (`--lambda-anchor-neg-factor`, iter118, default 1.0) relaxes the
   penalty on negative corrections only. In the free-correction equilibrium it rescales negative
   corrections, so gene ranks and signs of Delta-alpha are kept.

## Grid (`grid.tsv`, 8 configurations, seed 42, names `rev-bm2-grid<k>-s42`)

Common: `--config A --max-epochs 200 --accelerator cpu --lambda-mmd 100 --free-bits 1
--celltype-conditional-mmd --celltype-key cell_ontology_class --prior-clamp-max 0.85
--batch-key species` (testis 3-species row with the K = 3 laws of `../T12_2_recipe/README.md`).

| k | adv | anchor | negative-side factor | role |
|---|---|---|---|---|
| 1 | 150 | 0.001 | 1.0 | T12.2 base row unchanged: build-2 data fix alone |
| 2 | 150 | 0.0033 | 1.0 | Law 2 band 1/K edge x depth ratio |
| 3 | 150 | 0.0057 | 1.0 | Law 2 band 1/sqrt K edge x depth ratio |
| 4 | 150 | 0.0033 | 0.5 | row 2, negative side relaxed |
| 5 | 150 | 0.0057 | 0.5 | row 3, negative side relaxed |
| 6 | 150 | 0.0057 | 0.3 | row 3, negative side at the iter118 value |
| 7 | 150 | 0.01 | 0.3 | 25k diagnostic anchor, negative side 0.3 |
| 8 | 200 | 0.0057 | 0.3 | row 6 with Law 5 at the K = 4 value |

`cpu_p` / `cpu_normal`, 12 CPUs, 32G, 12 h.

## Selection (`select_recipe.py`): the T12.2 rule, unchanged

1. Gate G-V per finished run, same thresholds as T12.2: no NaN (and no E2.1 failure);
   saturation <= 5% in each alpha tail (alpha > 0.95, alpha < 0.05); 10/10 active z dims
   (Var(z_mean) >= 0.01); both tails of alpha - prior present (each of > 0.05 and < -0.05 in at
   least 2% of genes). A run that ends without `summary.json` fails the gate.
2. Among passing runs: `compute_integration_metrics` from `training/compare_vs_scvi.py`
   (k = 15, seed 42) on `z_mean` with labels `cell_ontology_class`; the six metrics are
   direction-corrected, min-max normalised across passing runs, and combined as
   0.4 mixing + 0.6 biology. Highest composite wins; ties go to the lower grid id.
3. No Delta-alpha statistic beyond the gate's fractions, no COE, no baseline and no biology
   readout enters the selection.
4. The selected configuration is trained at seeds 43-46 (`rev-bm2-s<seed>`, from
   `seeds_manifest.tsv`, written by `select_recipe.py --write-seed-manifest`); its grid run is the
   s42 member (`seed_map.tsv`). If no configuration passes G-V, P12 training stops and the
   result is reported as such.

Selection runs once, after every grid array task has reached a terminal Slurm state.
