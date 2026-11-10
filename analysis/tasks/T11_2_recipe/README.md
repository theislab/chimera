# T11.2 Van Zyl recipe: fixed grid and selection rule

Written and committed before the training run whose results are used. The cohort was added
during the revision; its recipe is fixed by the rule below before training. Earlier
submissions of this grid were cancelled before any run finished and no result of theirs is
used (default sample-level batch key; then objects replaced after the build audits R2 and R4).
The object is build 3 (ledger T11_1): the same 10,081 genes as build 2, with each gene mapped to
its Ensembl ID row where the matrix has one, and types with < 20 cells in a species removed;
frac(prior >= 0.95) is unchanged. The full-cohort grid is submitted in one array with the
T11.2b subset grids (`analysis/tasks/T11_2b_vz_subsets/array_all_grids.tsv`).

## Base recipe (pre-specified rule, section 11)

On the rebuilt object (`data/analysis/van_zyl/van_zyl_unified.h5ad`, T11.1),
frac(prior >= 0.95) = 0.3356 (3,383 of 10,081 ortholog groups measured in all five species).
This is at least 20%, so the
coupled pair applies: the testis 3-species row is the base, with conditional MMD on the
harmonised `cell_ontology_class` and the prior clamped at 0.85. K-dependent weights for
K = 5 species follow the K-scaling table (`tables/k_scaling_laws.tex`), using the K = 4
anchor where a law saturates or K = 5 lies outside the calibrated K = 2-4 range:

| Law | Quantity | K = 5 value | Reason |
|---|---|---|---|
| 1 | max epochs | 200 | K = 4 budget (150 * sqrt(K/2) is 237 at K = 5, beyond the calibrated range) |
| 2 | lambda-anchor | 0.001 | K >= 3 step |
| 3 | lambda-mmd | 100 | 50 * (K - 1), saturating at 100 for K >= 4 |
| 4 | free-bits | 1.5 | 0.5 * max(1, ceil(log2 K)) = 0.5 * 3 |
| 5 | adv-weight (kappa * G) | 200 | K = 4 value; the (1 - 1/K) headroom rises from 0.75 to 0.80 |

## Grid (8 configurations, s42; `grid.tsv`, array manifest `manifest.tsv`)

All rows: `--config A --max-epochs 200 --adata data/analysis/van_zyl/van_zyl_unified.h5ad --seed 42 --batch-key source_batch --accelerator cpu`.
Rows 1-6 and 8 add `--celltype-conditional-mmd --prior-clamp-max 0.85`. Every row passes `--celltype-key cell_ontology_class` explicitly (row 7 uses no conditional MMD, so there it has no effect).

batch_key = `source_batch` = species|chemistry (10x v2 or v3), 7 levels, at most 2 per species.
Human and M. fascicularis mix v2 and v3 and the other species are v3 only, so a species-level
batch would leave chemistry unmodelled. Sample-level batches fail (iter94), so they stay in
`obs['batch']`, unused in training; two batches in one species has the liver H+Monkey precedent.

| grid_id | run | adv-weight | lambda-anchor | lambda-mmd | free-bits | clamp | role |
|---|---|---|---|---|---|---|---|
| 1 | rev-vz-grid1-s42 | 200 | 0.001 | 100 | 1.5 | 0.85 | base (rule above) |
| 2 | rev-vz-grid2-s42 | 100 | 0.001 | 100 | 1.5 | 0.85 | Law 5 at the testis 3-sp value |
| 3 | rev-vz-grid3-s42 | 200 | 0.001 | 50 | 1.5 | 0.85 | Law 3 at the K = 2-3 value |
| 4 | rev-vz-grid4-s42 | 200 | 0.001 | 100 | 1.0 | 0.85 | Law 4 at the K = 4 step |
| 5 | rev-vz-grid5-s42 | 200 | 0.0006 | 100 | 1.5 | 0.85 | Law 2 at the 1/K edge of its band (0.003/5) |
| 6 | rev-vz-grid6-s42 | 100 | 0.001 | 50 | 1.0 | 0.85 | testis 3-sp row weights unchanged |
| 7 | rev-vz-grid7-s42 | 200 | 0.001 | 100 | 1.0 | none | liver 4-sp row (iter101), unclamped, no conditional MMD |
| 8 | rev-vz-grid8-s42 | 200 | 0.001 | 50 | 1.5 | 0.85 | base with dual MMD: `--lambda-mmd 50 --lambda-mmd-aux 100` |

Rows 2-5 change one K-dependent weight of the base at a time. Row 7 is the former fallback.
Row 8 adds a global MMD term beside the conditional one, because 62.8% of mouse cells are in
classes with no partner species, where conditional MMD does not act. Its two values are the
only documented dual-MMD pair (iter121); that run collapsed z dims on liver 4-sp, which has 2
shared classes against 8 here, so Gate G-V decides whether it holds on this cohort.

## Selection rule (`select_recipe.py`)

Only s42 grid runs are used. No Delta-alpha statistic beyond the gate's tail check, no
glaucoma gene and no biology readout enters selection.

1. **Gate G-V** on each run's `final_state.npz` and `summary.json`; a run passes only if all hold:
   - no NaN: alpha, prior and `z_mean` all finite, and E2.1 (loss finite) not failed;
   - saturation <= 5%: frac(alpha > 0.95) <= 0.05 and frac(alpha < 0.05) <= 0.05;
   - 10/10 active z dims: Var over cells of `z_mean` >= 0.01 in each of the 10 dims;
   - both tails present: frac(alpha - prior > +0.05) >= 2% and frac(alpha - prior < -0.05) >= 2%
     (the E9.2 criterion; prior is the training prior, clamped where the row clamps).
   A run that did not finish (no `summary.json` or `final_state.npz`) fails.
2. **Integration composite** among runs that pass: `compute_integration_metrics` from
   `training/compare_vs_scvi.py`, unmodified, k = 15, seed = 42, on `z_mean`, species
   `obs['species']`, labels `obs['cell_type_harmonised']`. The six metrics are
   direction-corrected and min-max normalised across the passing runs as in
   `notebooks/07_baselines_master.ipynb`; composite = 0.4 * mean(kNN species purity,
   mixing entropy, species silhouette) + 0.6 * mean(kNN cell-type purity, cell-type
   silhouette, label-transfer accuracy).
3. The highest composite is selected; ties (|difference| < 1e-9) go to the lower grid_id.
   If only one run passes, it is selected. If none passes, P11 training stops and this is
   reported; T11.1 baselines still run.

The selected run's s42 counts as `rev-vz-s42`. T11.3 trains seeds 43-46 with the same
arguments as `rev-vz-s<seed>` (`analysis/tasks/T11_3_vz_seeds/`, written by
`select_recipe.py --write-t113-manifest`).

Disclosure: rows 1-6 are label-informed alignment (conditional MMD on harmonised cell
classes), as on testis. The grid is reported in Methods.

## Per-run report (`run_profile.py`, not part of selection)

Runs only after selection is complete and is never an input to it. For each grid and seed
run it reports the Delta-alpha sign profile against the training prior and the raw prior
(fractions positive and negative, beyond +0.05 and -0.05, SD, median), knn_mixing_entropy,
final adversary accuracy and Gate G-V. A seed with knn_mixing_entropy below 0.5 times the
selected run's is flagged poorly mixed and reported separately; the primary D stays the
mean over seeds (section 2 of the pre-specified plan).
