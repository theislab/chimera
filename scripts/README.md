# Stand-alone scripts

Evaluation scripts that take a cohort object or a trained model and need nothing else from this
repository. They were written with the first version of the model. The analyses of the paper do
not use them: those are in [`../analysis/`](../analysis/README.md) and
[`../notebooks/`](../notebooks/). Every script here parses and answers `--help`; none was run
again for the release, so treat them as starting points.

Run from the repository root, inside `chimera_env`:

```bash
python -m scripts.<name> --help
```

| Script | Does | Needs |
|---|---|---|
| `pretrain_checks.py` | checks a prepared cohort object before training: is the protein prior informative, library-size and cell-type composition by species | a cohort object |
| `run_ablations.py` | trains the model under named ablation configurations (`chimera/_constants.py`) and tabulates α and Δα | a cohort object |
| `run_optuna.py` | hyperparameter search over [`../configs/optuna_search_space.json`](../configs/optuna_search_space.json), validation loss as objective | a cohort object |
| `train_baselines.py` | trains a small baseline ladder (scVI, Harmony, scANVI, two ablations) | a cohort object |
| `compute_metrics.py` | scIB-style integration metrics for given latent keys | an object with latents |
| `alpha_analysis.py` | α against the prior, integration by conservation quartile, residual decomposition | a trained model and its object |
| `biological_validation.py` | α of known conserved and divergent gene families against the gene-wide median | a trained model and its object |
| `application_demos.py` | four demonstrations: integration magnitude per species pair, convergent-evolution detection, α by phylogenetic distance, species-aware imputation | a trained model and its object |
| `alpha_reproducibility.py` | α across data sets with shared species, and across seeds | cohort objects; trains |
| `alpha_sensitivity.py` | rank stability of α under doubled or halved loss weights and across seeds | a cohort object; trains |
| `robustness_tests.py` | ortholog dropout, shuffled priors, false orthologs | a cohort object; trains |
| `fewshot_transfer.py` | holds out one species and tests label transfer | a cohort object; trains |
| `nonlinear_leakage.py` | deep classifier for species information left in the latent | a trained model and its object |
| `rclone_sync_chimera.sh` | maintainer utility: mirrors a working folder to a remote with `rclone` | not needed for any result |

For the current way to train, see [`../training/README.md`](../training/README.md); for the
current checks of a fit, `training/check_expectations.py`.
