<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# P13_kidney

Kidney, human acute kidney injury against mouse ischemia-reperfusion injury: objects and ortholog table, pseudobulk differential expression with permutation and leave-one-out fits, the gene-level test of healthy Δα, program transfer by injury stage, and a published list of injury genes.

| | |
|---|---|
| Question | Disease: Which human disease responses does a mouse model reproduce, and where do drug targets and disease genes sit? |
| Status | in the paper |
| Environment | `chimera_env`, revision_de_env for de.py |
| Used in the paper | Supplementary Section D.12 (`sec:disease`); Supplementary Figure S59a, b, c (`fig:disease`); Supplementary Section D.12 (`ssec:er-disease`); Supplementary Section C.21.13 (`sssec:rev-disease`); Supplementary Table S18 (`tab:data-preparation`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/P13_kidney/fetch_kidney.sh $PWD
    python -I analysis/tasks/P13_kidney/build_orthologs.py <repo>
    python -m analysis.tasks.P13_kidney.build
    python -m analysis.tasks.P13_kidney.split_units
    python analysis/tasks/P13_kidney/de.py
    sbatch --array=0-8 analysis/tasks/P13_kidney/loo_array.sbatch
    sbatch --array=<tasks> analysis/tasks/P13_kidney/perm_array.sbatch <n chunks>
    python -m analysis.tasks.P13_kidney.readout
    python -m analysis.tasks.P13_kidney.hd1
    python -m analysis.tasks.P13_kidney.hd4
    python -I analysis/tasks/P13_kidney/cippa.py
    python -m analysis.tasks.P13_kidney.drivers --forward 42
    python -m analysis.tasks.P13_kidney.drivers --combine

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `data/analysis/disease_kidney/`: bundle `data_other_cohorts`
- `results/analysis/P13/`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/analysis/bone_marrow/`
- `data/analysis/disease_kidney/`
- `data/analysis/ensembl110/`
- `data/processed/`
- `data/raw/`
- `training/runs/`

**Notes.** Shares results/analysis/P13/ with P13_recipe (the healthy kidney model). Its fit function is reused by C1_cohorts for the disease cases of lung and cortex.
