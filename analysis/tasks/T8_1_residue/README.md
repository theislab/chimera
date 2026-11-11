<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T8_1_residue

Specificity control of the residue-level attribution: the same contrasts recomputed with genes swapped and with random directions, on the cached attribution inputs.

| | |
|---|---|
| Question | Protein space: Is the residue-level attribution specific? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.6 (`sec:protein`); Supplementary Table S26 (`tab:residue-specificity`); Supplementary Section D.16 (`ssec:er-residue`); Supplementary Section C.15.8 (`sssec:rev-residue`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T8_1_residue.run

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T8_1/`: bundle `analysis_results`

**Reads** (as named in the code)

- `notebooks/cache/`

**Notes.** Reads notebooks/cache/gin_attribution/ (deposit bundle notebook_caches). Writes the LaTeX table residue-specificity.tex. The other analyses of this part of the paper are the notebooks 05, 09, 10, 10b and 11.
