<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# O1_orthologs

Audits every ortholog pair of the cohorts against Ensembl Compara, flags pairs and count columns, repeats the validation without flagged genes, and builds the liver and testis objects with counts joined on the Ensembl gene ID together with the models trained on them.

| | |
|---|---|
| Question | Data: Which objects, ortholog tables and independent references do the analyses read? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C (`sec:extended-methods`); Supplementary Table S18 (`tab:data-preparation`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.O1_orthologs.audit_pairs
    python -m analysis.tasks.O1_orthologs.mapping
    python -m analysis.tasks.O1_orthologs.flags
    python -m analysis.tasks.O1_orthologs.blind
    python -m analysis.tasks.O1_orthologs.sensitivity
    python -m analysis.tasks.O1_orthologs.compare_tiers
    python -m analysis.tasks.O1_orthologs.compare_h1
    python -m analysis.tasks.O1_orthologs.repair --tissue liver --mode repair
    sbatch --array=0-<rows-1> analysis/slurm/array.sbatch analysis/tasks/O1_orthologs/manifest_o2_liv_hmouse.tsv
    python -m analysis.tasks.O1_orthologs.repair_compare

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/O1/`: bundle `analysis_results`
- `data/analysis/O1/repair_liver/`: bundle `data_identifier_joined`
- `data/analysis/O1/repair_testis/`: bundle `data_identifier_joined`
- `training/runs/o1-*`: not deposited (the tables computed from these runs are)
- `training/runs/o2-*`: bundle `data_identifier_joined`

**Reads** (as named in the code)

- results of `H1_harmonise`, `T2_6_matched`, `T3_1b_integration_gate`, `T4_2_gene_baselines`, `T4_3_scoreboard`, `T5_6_gene_level`, `X2_liver_biology`, `Y6_uses`
- `data/analysis/O1/`
- `data/analysis/bone_marrow/`
- `data/analysis/disease_kidney/`
- `data/analysis/new_usecases/`
- `data/analysis/van_zyl/`
- `data/processed/`
- `data/raw/`
- `training/runs/`
- code of `H1_harmonise`

**Notes.** The identifier-joined objects are the recommended data for new work (deposit bundle data_identifier_joined). The models of the paper were trained on the objects in data/processed/. Definitions in rules.py; further scripts (named_genes, named_claims, set_exclusion, rule_check, repair_*, retrain_compare, rebuild, findings) are single checks, each with its command in its docstring. named_genes.py reads the manuscript sources from a checkout beside this repository.
