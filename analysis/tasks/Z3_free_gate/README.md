<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# Z3_free_gate

Trains the per-gene gate (--config M: one free offset per gene in place of the protein-feature network) on liver Human+Mouse, four-species liver and testis Human+Mouse, and scores its Δα on the pre-specified tests beside the protein-feature gate on the same seeds.

| | |
|---|---|
| Question | The per-gene gate: What does Δα show when it is read gene by gene from counts, without the protein-feature network? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.5 (`sec:per-gene`); Figure 5e, f, h, i (`fig:validation`); Supplementary Section C.8.4 (`sssec:per-gene-gate`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    sbatch --array=<rows> analysis/tasks/Z3_free_gate/train.sbatch
    python -m analysis.tasks.Z3_free_gate.health
    python -m analysis.tasks.Z3_free_gate.scoreboard --line free5 --workers <n>
    python -m analysis.tasks.Z3_free_gate.compare check tables reversal x5 y3

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `training/runs/z3-<cohort>-m-s<seed>/`: bundle `models_per_gene_gate`
- `results/analysis/Z3/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `T3_1b_integration_gate`, `T4_3_scoreboard`, `T5_6_gene_level`
- `training/runs/`
- code of `O1_orthologs`, `X4_where_differs`

**Notes.** The code calls the per-gene gate the free gate. Evaluated after the pre-specified test. Load these models with analysis/common/load_run.py. manifest.tsv holds the argv of every fit.
