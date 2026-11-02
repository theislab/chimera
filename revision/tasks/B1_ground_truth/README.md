# B1: gene-level benchmark, Track A (ground truth by spike-in) and Track D (practical)

Written before any altered fit was scored. Definitions in `rules.py`, alteration in `alter.py`, metrics in `metrics.py`, expected signs in `PREDICTIONS.md`.

## Design
Cohorts: liver H+Mouse, testis H+Mouse, kidney H+Mouse (healthy training object of P13: all five seeds healthy, 16 shared types; bone marrow was not used because 2 of 5 seeds went non-finite).

Full factorial on every cohort: alteration x effect (`pattern` q 0.25, 0.5, 1; `level` f 2, 4, 8; `loss` e 0.5, 0.9, 1) x expression band (quartiles of species-balanced log CP10K) x share of altered genes (2, 5, 10%) x 2 seeds = 216 fits per cohort. Identity is a within-fit stratum: altered genes are balanced over tertiles of π_raw and scored per tertile. Nulls: `sham_swap` and `thin_both` x band x share x 2 seeds = 48 fits per cohort; `none`: the existing unaltered CPU fits of the same recipe (5-6 per cohort), which also supply the reference for signed direction, with placebo sets drawn on them.

A row whose gene set would equal that of a smaller share is listed with `duplicate_of` and not trained (`grid_all.tsv`): the band has too few alterable genes for the larger share, and the achieved share is reported (a gene needs 20 counts in each species to be alterable at all).

Chimera is retrained with the canonical recipe unchanged, no tuning on results. p_null and the three expression feature columns follow the data as in the pipeline; sequence features and π stay. An unaltered object gets exactly the canonical features, because the canonical liver and testis expression columns come from the larger multi-species object: only the change in the statistic is added.

Discard rule, fixed now: non-finite α, z or S_g, or fewer than 10 active z dims. Mixing, tails and saturation are recorded for all fits and read as a sensitivity.

## Files
| File | What |
|---|---|
| `rules.py` | cohorts, recipes, grid, thresholds |
| `alter.py` | selection, matching, alterations, features |
| `make_grid.py`, `grid_*.tsv` | manifests |
| `step.py`, `array.sbatch`, `none.sbatch` | one row: build, train, score, clean |
| `score_run.py` | all per-gene readouts of one fit |
| `metrics.py`, `score_metrics.py` | metrics per fit, pair bootstrap |
| `placebo.py` | type I error on unaltered fits |
| `qc.py` | realism of the altered objects, no method read |
| `aggregate.py` | tidy tables for the overview |
| `tools.py`, `tool_array.sbatch` | SATURN and CAMEX on the main settings |
| `track_d/` | runtime, memory, scaling |
| `figures.py` | one figure per task |
