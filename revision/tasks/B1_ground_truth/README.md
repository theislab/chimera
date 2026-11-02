# B1: gene-level benchmark, Track A (ground truth by spike-in) and Track D (practical)

Written before any altered fit was scored. Definitions in `rules.py`, alteration in `alter.py`, metrics in `metrics.py`, expected signs in `PREDICTIONS.md`.

## Design
Cohorts: liver H+Mouse, testis H+Mouse, kidney H+Mouse (healthy training object of P13: all five seeds healthy, 16 shared types; bone marrow was not used because 2 of 5 seeds went non-finite).

Grid: the full factorial on every cohort (`BENCHMARK_PLAN.md`, second amendment): alteration x effect (`pattern` q 0.25, 0.5, 1; `level` f 2, 4, 8; `loss` e 0.5, 0.9, 1) x expression band (quartiles of species-balanced log CP10K) x share of altered genes (2, 5, 10%) x 2 seeds = 216 fits per cohort. Identity is a within-fit stratum: altered genes are balanced over tertiles of pi_raw and scored per tertile. Nulls: `sham_swap` and `thin_both` x band x share x 2 seeds = 48 fits per cohort; `none`: the existing unaltered CPU fits of the same recipe (5-6 per cohort), which also supply the reference for signed direction, with placebo sets drawn on them. Tools: SATURN 1:1, CAMEX, GeneSpectra and CACIMAR at share 5% (the last two through the adapters of the overview benchmark, revision/tasks/B2_benchmark/tools, whose primary scalars were fixed before any comparison; SAMap when its graph exists).

The only cells left out add no information: a row whose gene set would equal that of a smaller share (the band has too few alterable genes for the larger share; a gene needs 20 counts in each species to be alterable) is listed with `duplicate_of` in `grid_all.tsv` and not trained; the achieved share is reported.

Submission order (`tier` in the grids, `make_queue.py`, `feeder.sh`): central setting first (strongest effect of each alteration, band midhigh, share 5%, seed 42), then main effects one factor at a time, effect x band, the tools' main settings, the remaining shares, seed 43, the tools' other rows. Results are reported for what has finished at each stage. A run of the grid was briefly reduced under a compute budget and restored; no reduction was driven by a result.

Chimera is retrained with the canonical recipe unchanged, no tuning on results. p_null and the three expression feature columns follow the data as in the pipeline; sequence features and π stay. An unaltered object gets exactly the canonical features, because the canonical liver and testis expression columns come from the larger multi-species object: only the change in the statistic is added.

Discard rule, fixed now: non-finite α, z or S_g, or fewer than 10 active z dims. A discarded fit removes only Chimera's row: the direct, sequence and tool readouts of that altered object are still scored. Mixing, tails and saturation are recorded for all fits and read as a sensitivity.

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
| `completion.py` | runs tried and finished per method and setting, `completion.csv` |
| `tools.py`, `tool_array.sbatch` | SATURN and CAMEX on the main settings |
| `track_d/` | runtime, memory, scaling |
| `figures.py` | one figure per task |
