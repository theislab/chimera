<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T2_2_regen

Recomputes, with each cohort's own ortholog table and joins on the human Ensembl gene ID, the statistics that depend on testis gene identity and on cross-tissue gene joins; one row per number.

| | |
|---|---|
| Question | Data: Which objects, ortholog tables and independent references do the analyses read? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.3 (`sssec:rev-identity`); Supplementary Section D.2 (`ssec:er-axis`); Supplementary Figure S3 (`fig:supp-alpha-robustness`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T2_2_regen.before_after

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T2_2/`: not deposited

**Reads** (as named in the code)

- `data/processed/`
- `notebooks/cache/`
- `training/runs/`

**Notes.** The column new_value of results/analysis/T2_2/before_after.csv holds the values printed in supplementary captions. The folder is not part of the data deposit. The script also reads the manuscript sources from a checkout beside this repository.
