# P13 Kidney healthy-model recipe: fixed grid and selection rule

Cohort: `data/revision/disease_kidney/kidney_healthy_train.h5ad` (P13 build, `revision/tasks/P13_kidney/build.py`):
human training-half multiome donors and untreated Ref mice only (`unit_roles.tsv`), K = 2, kidney human-mouse
1:1 table. The model supplies the gene readout D for H-D1 (plan section 8c); it shares no unit with any
disease contrast. The grid and this rule are committed before any kidney run is trained.

## Base recipe (section 8 rule, T7.4)

frac(π ≥ 0.95) on the built object = 0.2177 (`results/revision/P13/healthy_summary.json`). It is at least 20%,
so the base is the testis H+Mouse row (iter111): conditional MMD on `cell_ontology_class`, prior clamp 0.85,
adv 200, λ_anc 0.003, λ_mmd 50 (module default), free bits 1.0, 200 epochs. The other base row, liver H+Mouse
(iter90, unclamped, no conditional MMD, 150 epochs), is configuration 8, the T7.4 fallback. The grid is the
T7.9 islet grid (same K = 2 and the same base) with the kidney object. Every row passes `--batch-key batch`,
where `obs["batch"]` is the technical source within a species (canonical builder convention). The training
object holds one source per species (`Human|KPMP|10x multiome`, `Mouse|current study|10x multiome`), so
`batch` is exactly the species partition, as section 8c's `--batch-key species` requires; donors and animals
are in `obs["sample_batch"]` (T7.4: `batch = sample` is the iter94 failure). Rows 1-7 pass
`--celltype-key cell_ontology_class` explicitly for the conditional MMD. `python -m revision.common.check_cohort`
passes with both `--batch-key batch` and `--batch-key species` (`results/revision/P13/object_checks.json`).

## Grid (`grid.tsv`, 8 configurations, seed 42, names `rev-kid-grid<k>-s42`)

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

All rows use `--config A --accelerator cpu --batch-key batch` on `cpu_p` / `cpu_normal`, with 16 CPUs, 64G and 23:50.
They run as an array of 8 over `manifest.tsv` with `revision/slurm/array.sbatch`.

## Selection (`select_recipe.py`)

1. **Gate G-D2** (plan sections 8b and 8c) on each finished run:
   - no NaN and no E2.1 failure;
   - each α tail (α > 0.95, α < 0.05) at most 5%;
   - 10/10 active z dims (Var(z_mean) ≥ 0.01);
   - both tails of α − π present: α − π > 0.05 and α − π < −0.05 each in at least 2% of genes.

   A run without `summary.json` fails.
2. **Composite** among passing runs:
   - `compute_integration_metrics` from `training/compare_vs_scvi.py` (k = 15, seed 42) on `z_mean`, with labels `cell_ontology_class`;
   - the six metrics are direction-corrected, then min-max normalised across passing runs;
   - composite = 0.4·mixing + 0.6·biology.

   The highest composite wins; ties go to the lower grid id.
3. **Inputs excluded:** no Δα statistic beyond the gate's tail check, no DE, and no disease or biology readout.
4. **Seeds:** the selected configuration is trained at seeds 43-46 (`rev-kid-s<seed>`). `select_recipe.py --write-seed-manifest` writes `seeds_manifest.tsv`; the grid run is the s42 member (`seed_map.tsv`).
5. **No configuration passes G-D2:** kidney training stops, and H-D1 is not testable. H-D4 stands.

Selection runs once, after every grid array task has reached a terminal Slurm state.
