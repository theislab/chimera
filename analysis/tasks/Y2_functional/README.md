<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# Y2_functional

Asks whether Δα or the count evidence relate to data outside expression: mouse phenotypes, viability, regulatory sequence conservation and literature cases.

| | |
|---|---|
| Question | What the readouts find in liver and testis: Which genes, cell types, pathways and gene classes sit above or below the protein prior? |
| Status | exploratory, not in the paper |
| Environment | `chimera_env` |
| Used in the paper | no |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/Y2_functional/fetch.sh
    python -I analysis/tasks/Y2_functional/annot_pheno.py
    python -I analysis/tasks/Y2_functional/annot_reg.py
    python -m analysis.tasks.Y2_functional.build
    python -m analysis.tasks.Y2_functional.analyses
    python -m analysis.tasks.Y2_functional.literature
    python -m analysis.tasks.Y2_functional.followup
    python -m analysis.tasks.Y2_functional.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/Y2/`: bundle `analysis_results`
- `data/analysis/Y2/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`

**Reads** (as named in the code)

- `data/analysis/Y2/`
- `data/analysis/bone_marrow/`
- `data/analysis/disease_kidney/`
- `data/analysis/ensembl110/`
- `data/analysis/x3_annotations/`
- `data/processed/`

**Notes.** Not in the paper: no association passed the rule fixed in rules.py. Its regulatory annotations are read by E5_regulatory.
