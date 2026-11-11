# Stand-alone scripts

Stand-alone scripts of the first version of the model, kept as starting points: no analysis of
the paper uses them. They take a cohort object or a trained model and need nothing else from this
repository.
The analyses of the paper are in [`../analysis/`](../analysis/README.md) and
[`../notebooks/`](../notebooks/README.md). Every script here parses and answers `--help`; none was run
again for the release, so treat them as starting points. The last column of the table names the
code that the paper uses for the same purpose, where there is one.

Run from the repository root, inside `chimera_env`:

```bash
python -m scripts.<name> --help
```

| Script | Does | Needs | What the paper uses for this |
|---|---|---|---|
| `pretrain_checks.py` | checks a prepared cohort object before training: is the protein prior informative, library-size and cell-type composition by species | a cohort object | `python -m analysis.common.check_cohort`; the checks that `training/run_experiment.py` writes into `pretrain_checks.json` |
| `run_ablations.py` | trains the model under named ablation configurations (`chimera/_constants.py`) and tabulates α and Δα | a cohort object | `training/run_experiment.py --config <letter>` |
| `run_optuna.py` | hyperparameter search over [`../configs/optuna_search_space.json`](../configs/optuna_search_space.json), validation loss as objective | a cohort object | - (no recipe of the paper was chosen with it) |
| `train_baselines.py` | trains a small baseline ladder (scVI, Harmony, scANVI, two ablations) | a cohort object | `training/baseline_runner.py` |
| `compute_metrics.py` | scIB-style integration metrics for given latent keys | an object with latents | `training/compare_vs_scvi.py` (`compute_integration_metrics`) |
| `alpha_analysis.py` | α against the prior, integration by conservation quartile, residual decomposition | a trained model and its object | `notebooks/03_alpha_master.ipynb` |
| `biological_validation.py` | α of known conserved and divergent gene families against the gene-wide median | a trained model and its object | `notebooks/04_delta_alpha_biology.ipynb`; `analysis/tasks/T2_6_matched` |
| `application_demos.py` | four demonstrations: integration magnitude per species pair, convergent-evolution detection, α by phylogenetic distance, species-aware imputation | a trained model and its object | - |
| `alpha_reproducibility.py` | α across data sets with shared species, and across seeds | cohort objects; trains | `analysis/tasks/T3_3_consensus` (seeds); `analysis/tasks/T2_3_tissue` (tissues) |
| `alpha_sensitivity.py` | rank stability of α under doubled or halved loss weights and across seeds | a cohort object; trains | - |
| `robustness_tests.py` | ortholog dropout, shuffled priors, false orthologs | a cohort object; trains | the retraining controls `analysis/tasks/T3_4_destroy_features`, `T3_5_count_perm`, `T3_6_matched`, `T6_8_priorswap` |
| `fewshot_transfer.py` | holds out one species and tests label transfer | a cohort object; trains | - |
| `nonlinear_leakage.py` | deep classifier for species information left in the latent | a trained model and its object | `training/z_leakage_probe.py` |

For the current way to train, see [`../training/README.md`](../training/README.md); for the
current checks of a fit, `training/check_expectations.py`.
