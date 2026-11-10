<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T7_1_holdouts

Downloads the liver disease data sets, checks them against the file lists of GEO, and fixes the mouse hold-out samples before any expression is read.

| | |
|---|---|
| Question | Data: Which objects, ortholog tables and independent references do the analyses read? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `chimera_env`, R for extract_seurat_meta.R |
| Used in the paper | Supplementary Section C.21.14 (`sssec:rev-disease`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/T7_1_holdouts/fetch_disease.sh
    bash analysis/tasks/T7_1_holdouts/fetch_ranged.sh
    python -m analysis.tasks.T7_1_holdouts.verify_downloads
    python -m analysis.tasks.T7_1_holdouts.fix_holdouts
    python -m analysis.tasks.T7_1_holdouts.sample_counts
    python -m analysis.tasks.T7_1_holdouts.gd1_human_counts

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `data/analysis/disease/raw/`: bundle `data_other_cohorts`
- `results/analysis/T7_1/`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/analysis/disease/`

**Notes.** holdouts.tsv is the fixed hold-out list.
