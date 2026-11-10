<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T7_3_mash

Liver MASH: hepatocyte pseudobulk differential expression in human and mouse, the gene-level test of healthy Δα against human-mouse agreement of the disease response, program transfer, and a second contrast with an innate-immune response list.

| | |
|---|---|
| Question | Disease: Which human disease responses does a mouse model reproduce, and where do drug targets and disease genes sit? |
| Status | in the paper |
| Environment | `revision_de_env` |
| Used in the paper | Section 3.8 (`sec:disease`); Figure 7 (`fig:disease`); Supplementary Section D.11 (`ssec:er-disease`); Supplementary Section C.21.14 (`sssec:rev-disease`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    sbatch analysis/tasks/T7_3_mash/run.sbatch fits
    sbatch analysis/tasks/T7_3_mash/run.sbatch analysis
    sbatch analysis/tasks/T7_3_mash/run.sbatch t78 (guilliams.py first; post.py last)

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T7_3/`: bundle `analysis_results`
- `results/analysis/T7_8/`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/analysis/disease/`
- `data/raw/`
- code of `P13_kidney`

**Notes.** t78_second_contrast.py writes results/analysis/T7_8/. de.py needs pydeseq2, which is in revision_de_env only (analysis/envs/revision_de_env.yml).
