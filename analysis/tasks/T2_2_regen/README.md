<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T2_2_regen

Computes, with each cohort's own ortholog table and genes matched between tissues on the human Ensembl gene ID, the statistics that depend on gene identity in testis and on the match of genes between tissues: program-level agreement of Δα within testis and between testis and liver, sign quadrants of the tissue means, category effects on testis, and the gene-set enrichment tables; one row per quantity.

| | |
|---|---|
| Question | Data: Which objects, ortholog tables and independent references do the analyses read? |
| Status | in the paper |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.3 (`sssec:rev-identity`); Supplementary Section D.1 (`ssec:er-model`); Supplementary Section D.2 (`ssec:er-axis`); Supplementary Section D.15 (`ssec:er-residue`); Supplementary Figure S3g, h, i, j, k (`fig:supp-alpha-robustness`); Supplementary Figure S18d (`fig:supp-dalpha-broad-categories`); Supplementary Figure S20 (`fig:supp-dalpha-testis-detailed`); Supplementary Figure S21c (`fig:supp-dalpha-hallmark`); Supplementary Figure S65e (`fig:supp-main-gin-protein`) |
| How we know | numbers printed in the paper are read from this task's result files |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    python -m analysis.tasks.T2_2_regen.values

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T2_2/values.csv`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/processed/`
- `notebooks/cache/`
- `training/runs/`

**Notes.** values.csv has one row per quantity: cohort, quantity, value, source (the runs or the cached notebook table it comes from), paper (the LaTeX labels of the figures and sections that print it) and note. It is the source of the values printed in the supplementary captions and Extended Results named in that column. The formulas are those of the notebooks 03, 04, 06 and 08; the script reads notebooks/_paper_utils.py and the two gene-category modules beside it. The folder results/analysis/T2_2/ also holds job logs and file copies of a notebook run; they are not deposited.
