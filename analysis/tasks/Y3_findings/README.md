<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# Y3_findings

Worked examples in the counts (spermatocyte-peak and hepatocyte catabolic genes), the same profile-correlation recipe applied to the latents of other tools, and controls of the kidney count evidence.

| | |
|---|---|
| Question | What the readouts find in liver and testis: Which genes, cell types, pathways and gene classes sit above or below the protein prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.3 (`sec:findings`); Figure 6a, b (`fig:validation`); Supplementary Figure S25 (`fig:supp-worked-genes`); Supplementary Section C.21.11 (`sssec:rev-exploratory`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.Y3_findings.worked_genes
    python -m analysis.tasks.Y3_findings.latent_b1z
    python -m analysis.tasks.Y3_findings.kidney_s_controls
    python -m analysis.tasks.Y3_findings.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/Y3/`: bundle `analysis_results`

**Reads** (as named in the code)

- `training/runs/`

**Notes.** Labeled exploratory in the paper. figures.py draws the two worked genes (main_validation/worked_genes) from worked_pseudobulk.csv and worked_readouts.csv.
