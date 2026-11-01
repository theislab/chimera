# T12.2 Bone-marrow recipe: fixed grid and selection rule

Cohort: `data/revision/bone_marrow/bone_marrow_unified.h5ad` (T12.1; human, mouse lemur,
mouse; K = 3). The grid and this rule are committed before any bone-marrow run is trained.
The cohort was added during the revision; its recipe is fixed by this rule.

## Base recipe (section 11 rule logic)

frac(π ≥ 0.95) on the built object = 26.8% (12,417 ortholog groups; `build_report.json`),
which is at least 20%, so the base is the testis 3-species row with conditional MMD on
`cell_ontology_class` (the nine harmonised types) and prior clamp 0.85. The other base row,
the liver 4-species row (iter101, unclamped, no conditional MMD), is grid configuration 8.

K-dependent weights at K = 3, from `chimera-manuscript/tables/k_scaling_laws.tex`:

| Law | Quantity | K = 3 value used in the base |
|---|---|---|
| 1 | epochs | 200 (150·sqrt(3/2) = 184; the K = 4 budget; runs stop on convergence) |
| 2 | λ_anc | 0.001 (band 0.003·[1/K, 1/sqrt K] = [0.001, 0.0017]; 1/K edge, testis 3-sp value) |
| 3 | λ_mmd | 100 (50·(K−1)) |
| 4 | β_fb | 1.0 (0.5·max(1, ceil(log2 K))) |
| 5 | κ·G | 150 (between the anchored K = 2 value 100 and K = 4 value 200; (1−1/K) scaling gives 133 from K = 2 and 178 from K = 4) |

## Grid (`grid.tsv`, 8 configurations, seed 42, names `rev-bm-grid<k>-s42`)

| k | adv | anc | mmd | fb | clamp / cond. MMD | role |
|---|---|---|---|---|---|---|
| 1 | 150 | 0.001 | 100 | 1.0 | 0.85 / cell_ontology_class | base |
| 2 | 100 | 0.001 | 100 | 1.0 | same | Law 5 at K = 2 value |
| 3 | 200 | 0.001 | 100 | 1.0 | same | Law 5 at K = 4 value |
| 4 | 150 | 0.001 | 50 | 1.0 | same | Law 3 at K = 2 value |
| 5 | 150 | 0.0017 | 100 | 1.0 | same | Law 2 at 1/sqrt K edge |
| 6 | 150 | 0.001 | 100 | 1.5 | same | Law 4 one step up |
| 7 | 100 | 0.001 | 50 | 1.0 | same | testis 3-sp row unchanged |
| 8 | 200 | 0.001 | 100 | 1.0 | none / none | liver 4-sp row (other base) |

All: `--config A --max-epochs 200 --accelerator cpu --batch-key species`, `cpu_p` / `cpu_normal`,
16 CPUs, 64G.

batch_key = species, as in all canonical recipes (multi-batch-per-species fails, iter94:
`training/workflow.md`, z contracted 4x with donor batches). `obs['batch']` in the h5ad stays at
species|donor|tissue level (47 values) and is not used by training; it remains available for
donor-level analyses (DE, pseudobulk). The first submission (array 42408922) used the default
`--batch-key batch` and was cancelled after 56 min, before any output was read; its run
directories are in `training/runs/_failed/rev-bm-grid<k>-s42_42408922`.

## Selection (`select_recipe.py`)

1. Gate G-V per finished run: no NaN (and no E2.1 failure); saturation ≤ 5% in each α tail
   (α > 0.95, α < 0.05); 10/10 active z dims (Var(z_mean) ≥ 0.01); both tails of α − π present
   (each of α − π > 0.05 and < −0.05 in at least 2% of genes). A run that ends without
   `summary.json` fails the gate.
2. Among passing runs: `compute_integration_metrics` from `training/compare_vs_scvi.py`
   (k = 15, seed 42) on `z_mean` with labels `cell_ontology_class`; the six metrics are
   direction-corrected, min-max normalised across passing runs, and combined as
   0.4·mixing + 0.6·biology. Highest composite wins; ties go to the lower grid id.
3. No Δα statistic beyond the gate's tail check, no COE, no baseline and no biology readout
   enters the selection.
4. The selected configuration is trained at seeds 43-46 (`rev-bm-s<seed>`, from
   `seeds_manifest.tsv`, written by `select_recipe.py --write-seed-manifest`); its grid run is the
   s42 member (`seed_map.tsv`). If no configuration passes G-V, P12 training stops and the
   result is reported as such.

Selection runs once, after every grid array task has reached a terminal Slurm state.
