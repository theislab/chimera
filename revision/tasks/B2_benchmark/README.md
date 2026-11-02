# B2: Tracks B and C of the gene-level benchmark, extra tools, overview figure

Binding plan: `chimera-private/reviews/cell_system_review_1/response/drafts/BENCHMARK_PLAN.md`. Definitions: `rules.py`. Predictions: `PREDICTIONS.md`. Everything here was fixed and committed before any setting was scored. The tasks were chosen after X4, X6 and Y4 were seen, and the text says so. The earlier label-noise run (X6) is rerun under this design; none of its results is reused. Nothing changes in the chimera package.

## Track B, robustness
Cohorts: liver H+Mouse, testis H+Mouse, kidney H+Mouse. Every setting is generated three times (replicates r0, r1, r2), each with its own perturbation draw and its own training seed (142, 143, 144). Every method is recomputed on the setting's object.

| Family | Setting | What is done |
|---|---|---|
| control | full | no change; same pipeline and new seeds. For Chimera this gives the seed-noise ceiling of agreement |
| label | relabel10, relabel25, relabel50 | in each species that fraction of cells gets another type, in proportion to abundance (X6's noise model) |
| label | relabel25_near | 25% of cells get the type most similar to theirs (the confusion X6 did not test) |
| label | coarse | merged types (X6's coarse map on liver, testis; kidney by nephron group, from names only) |
| label | fine | each type split in 3 by k-means within each species, sub-labels matched inside the parent by Hungarian assignment |
| label | species_diff | in mouse every second shared type by size carries a mouse-only name, as macrophages and Kupffer cells do in the liver data |
| label | nolabel, nolabel_r02 | no annotation. Direct: X6's per-species Leiden with marker matching at r >= 0.3 (0.2). Chimera: the no-label form |
| composition | type_missing | one type removed from one species: second largest shared type from mouse (r0), from human (r1), third largest from mouse (r2) |
| composition | rare_types | every type but the two largest shared ones cut to 40 cells per species |
| composition | size_1to1, 1to5, 1to20 | species sizes 1:1, 1:5, 1:20. Mouse is the small species in r0, r2, human in r1 |
| sampling | cells50, cells25, cells10 | random cells within each species |
| sampling | depth50, depth25 | binomial thinning of all counts |
| sampling | few_donors | largest library (r0), or random ones (r1, r2): testis 1 sample per species, kidney 2 donors per species. Liver: one library per species, not applicable |
| technical | extra_batch | half of each species' cells moved to a new batch with its own gene-wise capture factor exp(-abs N(0, 0.5)) x 0.7, by binomial thinning. Batch key and library key both see it |
| technical | chemistry | no cohort has a chemistry difference across species, so a real one is transplanted: gene-wise capture of Smart-seq2 against 10x 3' v3 in Tabula Sapiens liver, by binomial thinning of one species (human in r0, r2; mouse in r1) |

**Chimera.** Recipe unchanged. Where the setting changes only labels and the recipe reads none (liver), there is nothing to retrain: the setting's Chimera reading is the `full` fits, marked `label-independent`. Where the recipe reads labels (testis, kidney) the fit sees the perturbed labels, except `nolabel`: conditional MMD removed, the form T5.4b uses for controls (kidney keeps its `--lambda-mmd 50`). `nolabel_r02` reuses the `nolabel` fits. Health rule as Track A; mixing is recorded, never used to discard.

**Two metrics in each setting**, so stability alone cannot win:
1. *Agreement with the same method's full-data reading.* Partial Spearman given covariates C (primary, so that expression cannot supply agreement), and raw Spearman beside it, on the genes both define, with the share of genes kept. The reference of a Chimera readout is the mean over the cohort's existing unaltered fits (6 seeds, 5 on kidney), all with seeds other than 142-144, so the agreement of Chimera carries its seed noise and `full` shows the ceiling. Direct methods equal 1 on `full` by construction. Tools have reference fits with seeds 45-47. Static scores (pi, dN/dS, LOEUF) do not change and are not ranked on agreement.
2. *rho_p with the independent gold* (05 section 3), pattern and level. Liver G_H3, G_H1; testis G_H4, G_H2; kidney TS vs TMS, living donor vs TMS, EVEE kidney. Covariates are the unperturbed cohort's in every setting (with tau from the true labels), so the estimand does not move. pi has no rho_p given itself: its value is given C without pi (flagged).

**Aggregation of replicates.** Every method, the same way: per-gene mean of its three replicate scores (at least two healthy, for Chimera), then the metric. The mean of the metric over single replicates is reported beside it. Bootstrap: 2,000 ESM-module block draws per task, one set of draws per task, so differences have CIs.

**Universe.** Per task, genes with gold, covariates and every core score (S_g, D, B1z, B1, B1b, B1c, B2b, B3a, Sdir, Lfc). Scores outside it (dN/dS, LOEUF, tools, or a score with under half of the universe defined) are computed on the intersection, with their n; D and S_g are recomputed on the same genes for head-to-head.

**Pooled fallback** for direct baselines, in the settings that remove or split cells: see `score_run.py`. Without it the per-library rule of T4.2 would fail the baselines for a reason unrelated to the method.

## Track C, independent data across tissues
For every cohort with an independent reference: pattern (TS vs TMS liver, Murat held-out species testis and its primates, TS vs TMS kidney and pancreas, living donor vs TMS kidney, Muraro vs TMS pancreas), level (EVEE liver, testis, kidney; TS vs TMS per-type rank-matched level on liver, kidney and pancreas), named reference lists (below). Statistic: rho_p, 10,000 ESM-module block draws, head-to-head on U_pair with simultaneous CIs, as section 3 and 5 of the plan. Strata: count-supported genes, identity bands with near-1 (>= 0.98), expression quartiles.

Named lists, stated in advance, stratified AUROC (section 4): C1 housekeeping, C2 ribosome, proteasome, spliceosome core (expected above), V1 xenobiotic and bile (liver only), V2 interferon and inflammatory, V3 extracellular (expected below, read as 1 - AUROC so that higher is better), Shay divergent against COE conserved on bone marrow and, with Y1's caution, the variability-matched form.

Replication of Y4's two survivors on tissues not searched: `expr_T2__tau_T3` on EVEE kidney, and `pihi__tau_T2` on the kidney and pancreas pattern golds; also odd and even autosomes. New cohorts join by adding a row to `rules_c.py` and re-running `track_c.py`.

## Tools
SATURN and CAMEX through the T5 pipelines, on `TOOL_SETTINGS` of Track B and on the cohorts of Track C. Primary scalars as frozen: S_deploy, C_cos. To run on perturbed objects both go through the 1:1 adapter of T5.6: every ortholog is the gene set, and the perturbed counts are the input. GeneSpectra, CACIMAR, SAMap gene-pair scores in their own envs; a tool that will not run is reported with the reason. All are oriented so that higher means more conserved.

## Overview
`overview.py`. Fixed now: for each task, rank of every applicable method (best = 1, ties share the mean rank), scaled to [0, 1] by (rank - 1) / (n - 1), so tasks with fewer methods count the same. Aggregate = mean of within-task scaled ranks within each track, then the mean over tracks with equal weight per track. Unscaled mean ranks are written beside it. A method applicable to a task but with no value (failed fit, tool that failed there) ranks last in that task; a method not applicable (static score on agreement; a tool not run on that setting) is left out and drawn hatched. Every task appears as a column. Winners stated as computed.

## Order of work (see ledger)
Settings, objects, scoring of per-gene readouts: committed first, fits launched. Scoring code and predictions committed before any setting met a gold. See the ledger `R/ledgers/B2_benchmark.md`.

## Run order
From `chimera/`, in `chimera_env` through SLURM: `make_grid`, `array.sbatch` per grid, `ref.sbatch`; `tools/make_tool_grid`, `tools/t5.sbatch`, tool sbatch files; `golds_c`; `track_c.sbatch baselines`, `track_c units`, `track_c.sbatch unit|lists|y4`; `score_b.sbatch`; `overview`.

## Compute (the plan's second amendment)
The full grid stands, with three replicates per setting. Runs are ordered by `make_queue.py`: central settings and main effects first, replicates and other severities after, and `feeder.sh` holds about a third of the per-user CPU cap. Only cells that add no information are dropped: CAMEX and SAMap on label-only settings, because neither reads labels, so their reading there is that of `full` and is marked label-independent. SATURN runs one replicate on every setting and three on the main settings; GeneSpectra and CACIMAR one replicate on every setting (they are deterministic given the data, so their reference reading is `full`); SAMap one replicate on every setting that changes counts. Scoring uses `pool.sbatch`.
