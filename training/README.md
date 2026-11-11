# Training

How a Chimera model is trained and checked. Run everything from the repository root, inside
`chimera_env`.

## The entry point

```bash
python training/run_experiment.py --run-name <name> --adata <cohort.h5ad> \
    --config A --max-epochs 150 --seed 42 --accelerator cpu [recipe options]
python training/run_experiment.py --help
```

One command trains (`train_and_diagnose.py`), evaluates the training expectations
(`check_expectations.py`) and writes `summary.json`. The exit code is 1 when an expectation
failed; the run itself is complete when `summary.json` and `final_state.npz` exist. The recipes
of the canonical models are in [`../configs/canonical_recipes.tsv`](../configs/canonical_recipes.tsv);
`bash analysis/retrain_from_recipe.sh <cohort> <seed>` builds the command from that table.
`--config A` is the protein-feature gate, `--config M` the per-gene gate; the other letters are
the ablation configurations of `chimera/_constants.py`. `--plan <json>` reads all arguments from
a file (no plan file is shipped).

A run folder `training/runs/<name>/` holds:

| File | Content |
|---|---|
| `final_state.npz` | per gene `alpha`, `prior`, `correction`, `S_g_ema` and decoder summaries; per cell `z_mean`, `species_labels` |
| `summary.json`, `run_summary.json`, `report.md` | expectations passed and failed, key metrics |
| `config.json` | every setting, including `adata_path` and `clamp_z_logvar` |
| `model/` | the saved model (`ChimeraModel.load`; per-gene-gate runs: `analysis/common/load_run.py`) |
| `alpha_snapshots/`, `lightning/` | α every few epochs; per-epoch metrics |
| `pretrain_checks.json` | checks of the cohort object before training |

`training/runs/` is not in git. The archived runs come from the deposit
([`../REPRODUCE.md`](../REPRODUCE.md), sections 3 and 5).

## Files

Two groups: the current code, and the record of the period in which the recipes were tuned.

**Current: training, checks and diagnostics of a run, baselines** (used by training runs and by the analyses)

| File | Role |
|---|---|
| `run_experiment.py` | the entry point above |
| `train_and_diagnose.py` | trains one model and writes the run folder |
| `check_expectations.py` | evaluates the training expectations (E1 to E9, defined in `workflow.md`, "Pre-registered expectations") on a run |
| `compare_vs_scvi.py` | integration scorecard of a run against scVI on the same cells; its metric functions are used by the analyses |
| `baseline_runner.py`, `run_all_baselines.py` | train the integration baselines PCA, Harmony, scVI, scANVI and sysVI for one data set, and for the whole matrix of the seven liver and testis cohorts (`notebooks/02_baselines.ipynb`); each latent goes to `training/runs/baselines_<method>/`. The runner names SATURN and CAMEX but does not run them: their runs are `analysis/tasks/T5_3_runs`, which writes latents in the same format |
| `build_subset.py` | builds a species subset of a multi-species object |
| `cross_seed_check.py` | agreement of α across seeds of one recipe |
| `summarize_runs.py` | one comparison table over the runs in `training/runs/` |
| `plot_diagnostics.py`, `umap_check.py`, `z_leakage_probe.py`, `gin_signed_diagnostics.py`, `quick_adv_trajectory.py` | diagnostics of one run: training curves, UMAP, species information left in the latent, the gate network's correction, the adversary's trajectory |

**Record of the model's development** (kept as they were run; not needed for any result; nothing current calls them)

| File | What it was for | State |
|---|---|---|
| `workflow.md` | the tuning protocol: expectations, operating rules, lessons per iteration | text |
| `iteration_log.md` | one entry per tuning iteration: change, outcome, decision | text |
| `workflow-best.md` | registry of the best-known recipe per training scenario during tuning, with its metrics; the recipes of the paper are `configs/canonical_recipes.tsv` | text |
| `run_xds_sweep.sh` | one early recipe on every Human+other subset | runs as written only on Apple MPS (`--accelerator mps`); `PYTHON` selects the interpreter |
| `run_gin_input_sweep.sh` | four runs with blocks of gate-network inputs zeroed | as above |
| `compare_gin_ablation.py` | table over the four runs of the sweep above | needs those runs |
| `iter119_bias_extract.py`, `_iter120_eval.sh` | post-processing of two tuning iterations | need runs that are not in the deposit |
| `recompute_esm2_variance.py` | one-time recomputation of the ESM-2 variance feature in the processed objects | already applied to the objects in the deposit |

Run names of the form `iter<N>-*`, `xds-*`, `testis-*` come from this development; the 11
canonical models are among them (`configs/canonical_recipes.tsv` names each).
