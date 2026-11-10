<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T4_1_gold

Downloads the independent references of the pre-specified tests and the Ensembl annotation, homology and dN/dS files, and checks identifiers, scales and coverage. No Chimera readout is read.

| | |
|---|---|
| Question | Data: Which objects, ortholog tables and independent references do the analyses read? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.8 (`sssec:rev-validation`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/T4_1_gold/fetch_gold.sh
    bash analysis/tasks/T4_1_gold/fetch_ensembl.sh
    python -m analysis.tasks.T4_1_gold.h4_map_union
    python -m analysis.tasks.T4_1_gold.dnds_release99_coverage
    python -m analysis.tasks.T4_1_gold.gold_checks --label-map <tsv> <repo> <out>

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `data/analysis/gold/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`
- `data/analysis/ensembl110/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`
- `results/analysis/T4_1/`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/analysis/ensembl110/`
- `data/analysis/gold/`
- `data/processed/`
- `data/raw/`

**Notes.** fetch_item.sh and fetch_ensembl_species.sh are called by the fetch scripts. Downloads are recorded in MANIFEST.tsv files (URL, date, sha256).
