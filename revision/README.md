# Revision analyses

Analyses added for the Cell Systems revision. Nothing here changes the `chimera` package.

| Path | Holds |
|---|---|
| `common/gene_ids.py` | The only gene-identity helper. `ortho_N` IDs are table-local, so lookups need a cohort |
| `envs/` | Extra environment definitions and the `chimera_env` lock file |
| `slurm/` | CPU job templates, single and array |
| `tasks/T<major>_<minor>_<slug>/` | One folder per task; outputs go to `results/revision/T<major>_<minor>/` |

## Task map

| Task | Script | Output |
|---|---|---|
