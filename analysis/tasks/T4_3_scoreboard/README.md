<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T4_3_scoreboard

The pre-specified validation: partial rank correlation of Δα with four independent references given covariates, with block-bootstrap intervals, gene-set tests, head-to-head against the direct baselines, and the mechanical decision procedure.

| | |
|---|---|
| Question | Pre-specified validation: Do independent data back Δα and the count evidence S_g, and how do direct baselines compare? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Section 3.4 (`sec:validation`); Table 1 (`tab:validation`); Figure 5a, b, c (`fig:validation`); Supplementary Section C.21.8 (`sssec:rev-validation`); Supplementary Section D.3 (`ssec:er-validation`) |
| How we know | numbers printed in the paper are read from this task's result files; a script of this task draws panels of the figure |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T4_3_scoreboard.pipeline (runs golds, baseline_splits, run --mode real, decide, figures from MANIFEST.tsv)
    python -m analysis.tasks.T4_3_scoreboard.null_check --outdir <dir>
    python -m analysis.tasks.T4_3_scoreboard.verify_primary
    python -m analysis.tasks.T4_3_scoreboard.calibration_dominant

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T4_3/`: bundle `analysis_results`

**Reads** (as named in the code)

- results of `T2_5_celltype_margin`, `T3_3_consensus`, `T4_1_gold`, `T4_2_gene_baselines`
- `data/analysis/ensembl110/`
- `data/raw/`
- `training/runs/`

**Notes.** stats.py holds the statistics (pure functions) and is reused by the benchmark and the cohort analyses. dominance_exploratory.py is labeled exploratory. golds.py needs two files that are not in the repository: the cell-type label map h3_label_map.tsv (seven lines; its location is named in MANIFEST.tsv) and the Hallmark library in the gseapy cache (~/.cache/gseapy/, filled by T6_1_functional/fetch_enrichr_cache.py). Without the label map the gold step cannot be run again; the other steps run on the archived golds.
