<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# Y5_aspects

Further aspects of the readouts on the canonical cohorts: protein properties, phylogenetic depth, cell states inside a label, gene modules, species by cell-type margins and X linkage in testis.

| | |
|---|---|
| Question | What the readouts find in liver and testis: Which genes, cell types, pathways and gene classes sit above or below the protein prior? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.3 (`sec:findings`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.Y5_aspects.build annot
    python -m analysis.tasks.Y5_aspects.build forward --threads 10
    python -m analysis.tasks.Y5_aspects.build genes
    python -m analysis.tasks.Y5_aspects.health
    python -m analysis.tasks.Y5_aspects.f1
    python -m analysis.tasks.Y5_aspects.f2 <section>
    python -m analysis.tasks.Y5_aspects.f3 <section>
    python -m analysis.tasks.Y5_aspects.f4 <section>
    python -m analysis.tasks.Y5_aspects.f56 f5|f6|cases
    python -m analysis.tasks.Y5_aspects.tried
    python -m analysis.tasks.Y5_aspects.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/Y5/`: bundle `analysis_results`
- `data/analysis/y5_annotations/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`

**Reads** (as named in the code)

- results of `T11_2c_count_rule`, `T11_4_vz_integration`, `T11_5_vz_genes`, `T11_6_vz_glaucoma`, `T11_7_vz_figure`, `T11_8_vz_class_filter`, `T12_3_tests`, `T4_2_gene_baselines`, `X2_liver_biology`, `X3_cross_cohort`, `X4_where_differs`, `X6_adds_beyond`
- `data/analysis/ensembl110/`
- `data/analysis/gold/`
- `data/analysis/x3_annotations/`
- `training/runs/`
- code of `X4_where_differs`, `X7_architecture_outputs`

**Notes.** The paper uses the X-linkage result (f56). tried.py lists every test tried, with multiple-testing correction across all of them. Definitions in rules.py.
