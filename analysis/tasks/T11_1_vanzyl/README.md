<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T11_1_vanzyl

Rebuilds the eye cohort (aqueous-humor outflow tissue of five species, Van Zyl et al. 2020) from the GEO count matrices, verifies it gene by gene against the raw matrices, and fixes the glaucoma gene list.

| | |
|---|---|
| Question | Data: Which objects, ortholog tables and independent references do the analyses read? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `chimera_env` |
| Used in the paper | Section 3.7 (`sec:tissues`); Supplementary Section C.21.14 (`sssec:rev-disease`); Supplementary Table S18 (`tab:data-preparation`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T11_1_vanzyl.build_vanzyl [--out DIR] [--no-compare]
    python -m analysis.tasks.T11_1_vanzyl.verify_build [--out DIR]
    bash analysis/tasks/T11_1_vanzyl/fetch_glaucoma_sources.sh
    python -m analysis.tasks.T11_1_vanzyl.build_glaucoma_list

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `data/analysis/van_zyl/`: bundle `data_other_cohorts`
- `data/analysis/glaucoma/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`
- `results/analysis/T11_1/`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/analysis/ensembl110/`
- `data/analysis/gold/`
- `data/analysis/van_zyl/`
- `data/processed/`
- `data/raw/`

**Notes.** Folders T11_1 to T11_8 are the eye cohort; T11_4 to T11_7 write to results/analysis/T11/.
