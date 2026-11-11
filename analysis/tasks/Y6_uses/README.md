<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# Y6_uses

Tries four further uses of existing fits: checking genome annotation, choosing a model species, drug targets, and cis-regulatory evidence by proxy.

| | |
|---|---|
| Question | What the count evidence adds: What do the readouts add to direct baselines and to other tools, on real and on ground-truth data? |
| Status | exploratory, not in the paper |
| Environment | `chimera_env` |
| Used in the paper | no |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/Y6_uses/fetch.sh ensembl (then: small, big)
    python -I analysis/tasks/Y6_uses/annot_genome.py
    python -I analysis/tasks/Y6_uses/labels.py
    python -I analysis/tasks/Y6_uses/golds.py
    python -m analysis.tasks.Y6_uses.build_rows
    python -m analysis.tasks.Y6_uses.use1
    python -m analysis.tasks.Y6_uses.use2
    python -m analysis.tasks.Y6_uses.use3
    python -m analysis.tasks.Y6_uses.use4
    python -m analysis.tasks.Y6_uses.checks
    python -m analysis.tasks.Y6_uses.figures

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/Y6/`: bundle `analysis_results`
- `data/analysis/Y6/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`

**Reads** (as named in the code)

- results of `T4_3_scoreboard`
- `data/analysis/Y6/`
- `data/analysis/bone_marrow/`
- `data/analysis/disease_islet/`
- `data/analysis/disease_kidney/`
- `data/analysis/ensembl110/`
- `data/analysis/functional/`
- `data/analysis/new_usecases/`
- `data/analysis/van_zyl/`
- `data/processed/`
- code of `F1_figures`

**Notes.** Not in the paper. Its annotation flags (zero-count genes, candidate wrong orthologs) are read by O1_orthologs and C1_cohorts. Definitions in rules.py.
