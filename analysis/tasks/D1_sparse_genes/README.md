<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# D1_sparse_genes

Asks whether the protein-feature gate gives a usable reading for genes with too few counts: masked genes against independent data, thinned-depth fits, held-out chromosomes, and gates fed selected feature blocks.

| | |
|---|---|
| Question | Reliability and controls: How stable are the readouts across seeds, and what do they read from counts and from features? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.4 (`sec:reliability`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.D1_sparse_genes.fwd --cohort <cohort>
    python -m analysis.tasks.D1_sparse_genes.build <cohort>
    python -m analysis.tasks.D1_sparse_genes.a_masked run
    python -m analysis.tasks.D1_sparse_genes.make_grid
    sbatch --array=<rows> analysis/tasks/D1_sparse_genes/array.sbatch
    python -m analysis.tasks.D1_sparse_genes.b_depth
    python -m analysis.tasks.D1_sparse_genes.c_heldout
    python -m analysis.tasks.D1_sparse_genes.tidy
    python -m analysis.tasks.D1_sparse_genes.figure

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/D1/`: bundle `analysis_results`
- `data/analysis/D1/`: not deposited (rebuilt by the task code)
- `training/runs/d1-*`: not deposited (the tables computed from these runs are)

**Reads** (as named in the code)

- results of `H1_harmonise`, `T4_3_scoreboard`, `X1_testis_discovery`
- `data/analysis/D1/`
- `data/analysis/ensembl110/`
- `data/processed/`
- `training/runs/`
- code of `F1_figures`, `X6_adds_beyond`

**Notes.** Labeled exploratory in the paper; it uses the held-out-chromosome arm (c_rho.csv). The d1-* runs are not deposited; their tables are. Definitions in rules.py.
