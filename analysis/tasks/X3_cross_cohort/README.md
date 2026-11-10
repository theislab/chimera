<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# X3_cross_cohort

Gene classes (transcription factors, receptors, enzymes and others) and protein-space neighborhood coherence of the readouts over the seven canonical cohorts.

| | |
|---|---|
| Question | What the readouts find in liver and testis: Which genes, cell types, pathways and gene classes sit above or below the protein prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.9 (`sec:protein`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.X3_cross_cohort.annotations
    python -m analysis.tasks.X3_cross_cohort.build
    python -m analysis.tasks.X3_cross_cohort.analyses
    python -m analysis.tasks.X3_cross_cohort.summarize
    python -m analysis.tasks.X3_cross_cohort.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/X3/`: bundle `analysis_results`
- `data/analysis/x3_annotations/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`

**Reads** (as named in the code)

- results of `T2_4_covariates`, `T2_5_celltype_margin`, `T3_3_consensus`, `T4_2_gene_baselines`
- `data/analysis/x3_annotations/`
- `data/processed/`

**Notes.** The paper uses Moran's I on the protein-space graph (esm_morans_i.csv). The gene classes are reused by X8_gate_divergence and H1_harmonise.
