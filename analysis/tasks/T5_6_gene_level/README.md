<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T5_6_gene_level

Derives per-ortholog scores from SATURN (macrogene weights) and CAMEX (gene embeddings), measures their seed reliability and relation to Δα, and scores them on the pre-specified tests beside Δα.

| | |
|---|---|
| Question | What the count evidence adds: What do the readouts add to direct baselines and to other tools, on real and on ground-truth data? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.4 (`sec:adds`); Supplementary Figure S26c (`fig:supp-validation-tests`); Supplementary Section D.3 (`ssec:er-validation`); Supplementary Section C.21.9 (`sssec:rev-gene-level`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T5_6_gene_level.scalars
    python -m analysis.tasks.T5_6_gene_level.scalar_analysis
    python -m analysis.tasks.T4_3_scoreboard.run --mode real --tools results/analysis/T5_6/tools_primary.csv --outdir results/analysis/T5_6/scoreboard (all three: sbatch analysis/tasks/T5_6_gene_level/analysis.sbatch)

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T5_6/`: bundle `tool_runs_saturn_camex`
- `training/runs/baselines_saturn_1to1/`: bundle `notebook_caches`

**Reads** (as named in the code)

- code of `T4_3_scoreboard`

**Notes.** saturn.tsv lists the three-seed gene-level SATURN runs.
