# Per-gene tables

## Canonical cohorts: one seed-consensus table per cohort

`<cohort>__consensus.csv`: one row per one-to-one ortholog group of the cohort, pooled over the
seeds that pass the seed-pooling rule (`seed_pooling.json`). Cohort keys: `liv_hmonkey`,
`liv_hmouse`, `liv_hzeb`, `liv_4sp` (liver), `test_hmac`, `test_hmouse`, `test_3sp` (testis).

| Column | Meaning |
|---|---|
| hsapiens_gene_id | Human Ensembl gene ID (release 110): the only valid join key across cohorts and tissues |
| symbol | Human gene symbol |
| var_name | `ortho_N` in the cohort AnnData; N is local to the tissue's ortholog table |
| prior | Protein-identity prior pi used in training (testis: clamped at 0.85) |
| prior_raw | Unclamped prior (mean ortholog identity) |
| alpha | Learned alpha, mean over pooled seeds |
| correction | Learned correction c (logit scale), mean over pooled seeds |
| delta_alpha | alpha - prior, mean over pooled seeds (MPS seeds kept by the pooling rule plus the CPU seeds) |
| sd_delta_alpha | Standard deviation of delta_alpha over pooled seeds |
| D | Consensus readout D (analysis plan, section 2): delta_alpha minus its median over the cohort's genes |
| S_ema | Species-specific residual share S_g (EMA), mean over pooled seeds |
| sign_consistency | Fraction of pooled seeds whose delta_alpha has the sign of the mean |
| n_seeds | Number of pooled seeds |

`reliability.csv`: seed reliability per cohort; `seed_pairs.csv`: per-pair Spearman of each readout.
Produced by `analysis/tasks/T3_3_consensus/run.py` (see `REPRODUCE.md`).

## Harmonized cohorts, both gates (`harmonised/`)

`harmonised/canonical/<cohort>.tsv.gz`: the 26 harmonized cohorts (liver, testis, bone marrow, kidney,
islet, eye, cortex, lung, skeletal muscle, tumor myeloid cells) read with the protein-feature gate,
pooled over the seeds kept in `runs_canonical.tsv` (`kept` true). `harmonised/free_gate/<cohort>.tsv.gz`:
the same cohorts and genes with the per-gene gate (`--config M`), pooled over the seeds in
`seeds_free_gate.json`. One row per ortholog group, one schema for both gates (definitions in
`analysis/tasks/H1_harmonise/rules.py`); `cohort_table.csv` describes each cohort (cells, genes, species,
recipe, seeds, health). The same tables, as parquet, are `results/analysis/H1/genes/` and
`results/analysis/Z4/h1/genes/` in the `analysis_results` bundle.

| Column | Meaning |
|---|---|
| cohort, tissue | Cohort key and tissue |
| hsapiens_gene_id, symbol | Human Ensembl gene ID (join key) and symbol |
| identity_human_mouse | Human-mouse protein identity of the ortholog pair, where the table has it |
| pi_raw, pi_clamped | Unclamped prior; prior the fit anchors on (clamped at 0.85 on clamped recipes) |
| alpha_s<seed>, alpha | alpha per kept seed; their mean |
| delta_alpha | alpha - pi_clamped |
| D, D_raw | delta_alpha minus its cohort median; the same on pi_raw |
| S_ema | Species-specific residual share S_g (EMA), mean over kept seeds |
| count_support | abs(S_ema) >= 1e-6 (the gene has count evidence) |
| strict_support | count support, abs(S_ema) at least the cohort median, signs of S_ema and D agree, both stable over seeds |
| sign_D_stable, sign_S_stable, saturated_flag | Sign stability over seeds; alpha above 0.95 in any seed |
| expr, detection, log_mean, tau, ppass, detection_<species> | Species-balanced expression and detection, cell-type specificity |
| B1 | Pseudobulk profile correlation between species (direct baseline) |
| o1_*, quiet_in_species, reference_artefact, nameable | Ortholog-audit and naming flags; only `nameable` genes are named in the paper |
| dNdS, LOEUF | Human-mouse dN/dS (Ensembl Compara) and gnomAD LOEUF, for reference |

`analysis/check_gene_tables.py` recomputes every value column of both sets from the archived runs.
