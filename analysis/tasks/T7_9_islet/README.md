<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T7_9_islet

Pancreatic islet, type 2 diabetes: objects of healthy and diabetic islets, pseudobulk differential expression, the gene-level test of healthy Δα, and program transfer between human and mouse with label-permutation nulls.

| | |
|---|---|
| Question | Disease: Which human disease responses does a mouse model reproduce, and where do drug targets and disease genes sit? |
| Status | in the paper |
| Environment | `chimera_env`, revision_de_env for de.py |
| Used in the paper | Section 3.8 (`sec:disease`); Figure 7 (`fig:disease`); Supplementary Table S18 (`tab:data-preparation`); Supplementary Section D.11 (`ssec:er-disease`); Supplementary Section C.21.14 (`sssec:rev-disease`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T7_9_islet.build
    python -I analysis/tasks/T7_9_islet/split_donors.py <human.h5ad> <mouse.h5ad> <out.tsv>
    python analysis/tasks/T7_9_islet/de.py
    sbatch --array=0-5 analysis/tasks/T7_9_islet/loo_array.sbatch
    python -m analysis.tasks.T7_9_islet.gd3h
    python -I analysis/tasks/T7_9_islet/readout.py
    python -m analysis.tasks.T7_9_islet.hd1
    python -m analysis.tasks.T7_9_islet.hd4
    python -m analysis.tasks.T7_9_islet.hd4_null

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `data/analysis/disease_islet/`: bundle `data_other_cohorts`
- `results/analysis/T7_9/`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/analysis/bone_marrow/`
- `data/analysis/disease_islet/`
- `data/analysis/ensembl110/`
- `data/processed/`
- `training/runs/`

**Notes.** The healthy islet model comes from T7_9_recipe and T7_9_recipe_b, which write to the same result folder.
