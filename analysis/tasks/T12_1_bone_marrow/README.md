<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T12_1_bone_marrow

Builds the bone-marrow cohort (human, mouse lemur, mouse) from the CELLxGENE files with its one-to-one ortholog table, and recomputes the conservation of expression of Shay et al. 2013 used as its reference.

| | |
|---|---|
| Question | Data: Which objects, ortholog tables and independent references do the analyses read? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `chimera_env` |
| Used in the paper | Section 3.7 (`sec:tissues`); Supplementary Section C.21.14 (`sssec:rev-disease`); Supplementary Table S18 (`tab:data-preparation`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/T12_1_bone_marrow/fetch_bm.sh
    python -m analysis.tasks.T12_1_bone_marrow.build_orthologs [--out CSV]
    python -m analysis.tasks.T12_1_bone_marrow.build_bm [--out DIR]
    bash analysis/tasks/T12_1_bone_marrow/fetch_probe_maps.sh
    python -I analysis/tasks/T12_1_bone_marrow/compute_coe.py [--out DIR]
    python -I analysis/tasks/T12_1_bone_marrow/fetch_portal_coe.py

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `data/analysis/bone_marrow/`: bundle `data_other_cohorts`
- `results/analysis/T12_1/`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/analysis/bone_marrow/`
- `data/analysis/ensembl110/`
- `data/processed/`
- `data/raw/`

**Notes.** The tests on this cohort are in T12_3_tests, which writes to results/analysis/P12/.
