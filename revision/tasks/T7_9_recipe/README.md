# T7.9 Islet healthy-model recipe: fixed grid and selection rule

Cohort: `data/revision/disease_islet/islet_healthy_train.h5ad` (T7.9 build, `revision/tasks/T7_9_islet/build.py`):
human non-diabetic training-half donors and healthy adult training mice only (`donor_roles.tsv`), K = 2.
The model supplies the gene readout D for H-D1 (plan section 8b); it shares no donor or library with
any disease contrast. The grid and this rule are committed before any islet run is trained.

## Base recipe (section 8 rule, T7.4)

frac(π ≥ 0.95) on the built object = 33.1% (14,017 genes, 17,238 cells; `results/revision/T7_9/healthy_summary.json`; 31.1% on the table), which is at
least 20%, so the base is the testis H+Mouse row (iter111): conditional MMD on `cell_ontology_class`,
prior clamp 0.85, adv 200, λ_anc 0.003, λ_mmd 50 (module default), free bits 1.0, 200 epochs. The other
base row, liver H+Mouse (iter90, unclamped, no conditional MMD, 150 epochs), is configuration 8, the T7.4
fallback. Every row passes `--batch-key batch`, with `obs["batch"]` the technical source dataset
within a species, as in the canonical cohorts: `Human_human48_v2`, `Human_human48_v3`, `Mouse_4m`,
`Mouse_aged` (species + study, plus chemistry where it differs within a study). Donors and libraries
stay in `obs["sample_batch"]`; a donor-level batch is the iter94 failure.

## Grid (`grid.tsv`, 8 configurations, seed 42, names `rev-isl-grid<k>-s42`)

| k | adv | anc | mmd | fb | clamp / cond. MMD | epochs | role |
|---|---|---|---|---|---|---|---|
| 1 | 200 | 0.003 | 50 | 1.0 | 0.85 / cell_ontology_class | 200 | base: testis H+Mouse row |
| 2 | 100 | 0.003 | 50 | 1.0 | same | 200 | Law 5 one step down |
| 3 | 300 | 0.003 | 50 | 1.0 | same | 200 | Law 5 one step up |
| 4 | 200 | 0.003 | 100 | 1.0 | same | 200 | Law 3 at the K = 3 value |
| 5 | 200 | 0.0015 | 50 | 1.0 | same | 200 | Law 2 at the 1/K edge of its K = 2 band |
| 6 | 200 | 0.003 | 50 | 1.5 | same | 200 | Law 4 one step up |
| 7 | 200 | 0.003 | 50 | 1.0 | none / cell_ontology_class | 200 | base unclamped |
| 8 | 200 | 0.003 | 50 | 1.0 | none / none | 150 | liver H+Mouse row (other base) |

All: `--config A --accelerator cpu --batch-key batch`, `cpu_p` / `cpu_normal`, 16 CPUs, 64G, 23:50,
array of 8 over `manifest.tsv` with `revision/slurm/array.sbatch`.

## Selection (`select_recipe.py`)

1. Gate G-D2 (plan section 8b) per finished run: no NaN (and no E2.1 failure); each α tail
   (α > 0.95, α < 0.05) ≤ 5%; 10/10 active z dims (Var(z_mean) ≥ 0.01); both tails of α − π present
   (α − π > 0.05 and < −0.05 each in at least 2% of genes). A run without `summary.json` fails.
2. Among passing runs: `compute_integration_metrics` from `training/compare_vs_scvi.py` (k = 15,
   seed 42) on `z_mean` with labels `cell_ontology_class`; the six metrics direction-corrected,
   min-max normalised across passing runs, 0.4·mixing + 0.6·biology. Highest composite wins; ties
   go to the lower grid id.
3. No Δα statistic beyond the gate's tail check, no DE, no disease and no biology readout enters
   the selection.
4. The selected configuration is trained at seeds 43-46 (`rev-isl-s<seed>`, `seeds_manifest.tsv`,
   written by `select_recipe.py --write-seed-manifest`); its grid run is the s42 member
   (`seed_map.tsv`). If no configuration passes G-D2, islet training stops and H-D1 is not testable;
   H-D4 stands.

Selection runs once, after every grid array task has reached a terminal Slurm state.

## Run report (`run_report.py`)

Descriptive; it selects nothing and runs only after selection. For every grid and seed run: α − π sign
profile against the anchored (clamped) and the raw prior, `knn_mixing_entropy`, `knn_species_purity`,
final adversarial accuracy. Flag, fixed before any seed run: a seed with `knn_mixing_entropy` below
0.75 × the selected grid run's is listed as poorly mixed and reported separately; the primary D stays
the mean over seeds.
