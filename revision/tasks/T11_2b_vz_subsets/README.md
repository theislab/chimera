# T11.2b Van Zyl subsets: build, grid rule, selection

Owner instruction (amendment log 2026-11-02): beside the full five-species cohort (T11.2), five
subsets, as the CAMEX cohorts were split (liver 4-sp into H+Monkey, H+Mouse, H+Zebrafish; testis
3-sp into H+Mac, H+Mouse). This README, `build_subsets.py` and `select_subsets.py` are committed
before any subset is built and before any subset run.

## Subsets (`build_subsets.py`)

| tag | species | K |
|---|---|---|
| hmfas | human, macaca_fascicularis | 2 |
| hmmul | human, macaca_mulatta | 2 |
| hpig | human, pig | 2 |
| hmouse | human, mouse | 2 |
| primates | human, macaca_fascicularis, macaca_mulatta | 3 |

- Source: the T11.1 object `data/revision/van_zyl/van_zyl_unified.h5ad`, build 3 (10,081 genes
  measured in all five species, each mapped to its Ensembl ID row where the matrix has one; no
  mitochondrial filter; harmonised types with < 20 cells in a species removed; `source_batch`).
  The R4 re-audit gate is satisfied (coordinator, 2026-11-02).
- Output `data/revision/van_zyl/van_zyl_<tag>.h5ad` and `_config.json`;
  `subsets_build_report.json`.
- `training/build_subset.py::build_subset` unchanged, with `drop_empty_genes=True,
  min_cells_per_gene=3`, as notebook 01 cells 34 and 46. As for the canonical subsets, the
  conservation prior and the 72 gene features are carried over from the five-species object (the
  prior is the mean identity over the four non-human species) and p_null is recomputed.
- Category levels of removed species are dropped from the categorical obs columns, values
  unchanged, so `source_batch` holds only the subset's species|chemistry levels (at most 2 per
  species: human and M. fascicularis v2 and v3, the others v3).
- `revision.common.check_cohort <h5ad> --batch-key source_batch` must print OK for each subset.

## Grid per subset (at most 4 rows, s42; `grid.tsv`; `manifest.tsv` = all subsets)

`array_all_grids.tsv` is the one array submitted: the 8 full-cohort rows of
`revision/tasks/T11_2_recipe/manifest.tsv` followed by the subset rows of `manifest.tsv`, unchanged.

Every row passes `--config A --seed 42 --batch-key source_batch --celltype-key cell_ontology_class --accelerator cpu`
and all its weights explicitly. Rows from canonical recipes (`07_EXPERIMENT_MATRIX.md`) are used unchanged.

Pairs (K = 2):

| row | source | epochs | adv | anchor | mmd | free-bits | conditional MMD | clamp |
|---|---|---|---|---|---|---|---|---|
| rule_base_k2 | section 11 rule, K = 2 | 150 | 100 | 0.003 | 50 | 0.5 | yes | 0.85 |
| testis_hmouse | canonical testis H+Mouse (iter111) | 200 | 200 | 0.003 | 50 | 1.0 | yes | 0.85 |
| liver_hmouse | canonical liver H+Mouse (iter90 xds) | 150 | 200 | 0.003 | 50 | 1.0 | no | none |
| liver_hmonkey | canonical liver H+Monkey (iter90) | 150 | 100 | 0.003 | 50 | 0.5 | no | none |

Primates (K = 3):

| row | source | epochs | adv | anchor | mmd | free-bits | conditional MMD | clamp |
|---|---|---|---|---|---|---|---|---|
| rule_base_k3 | section 11 rule, K = 3 | 200 | 133 | 0.001 | 100 | 1.0 | yes | 0.85 |
| testis_3sp | canonical testis 3-sp (iter115) | 100 | 100 | 0.001 | 50 | 1.0 | yes | 0.85 |
| liver_4sp | canonical liver 4-sp (iter101), unclamped | 200 | 200 | 0.001 | 100 | 1.0 | no | none |
| liver_hmonkey | canonical liver H+Monkey (iter90), close primates | 150 | 100 | 0.003 | 50 | 0.5 | no | none |

- **Rule base**: the section 11 rule on the subset. If frac(prior >= 0.95) >= 0.20 the coupled pair
  applies: the testis 3-sp row form (conditional MMD on `cell_ontology_class`, clamp 0.85) with K-dependent
  weights from `k_scaling_laws.tex`. K = 2: Law 1 150 epochs, Law 2 anchor 0.003, Law 3 mmd 50, Law 4
  free-bits 0.5, Law 5 adv 100. K = 3: Law 1 150·sqrt(3/2) = 184, run as 200; Law 2 0.001; Law 3
  50·(K−1) = 100; Law 4 0.5·ceil(log2 3) = 1.0; Law 5 100·(1−1/3)/(1−1/2) = 133. Below 0.20 the rule's
  start is an unclamped row already in the grid, so the subset gets the three canonical rows only.
- Rows are numbered in table order (grid1 = rule base when present). Names `rev-vz-<tag>-grid<k>-s42`.

## Selection (`select_subsets.py select`), per subset

The T11.2 rule unchanged, using its functions: Gate G-V (finite alpha, prior, z and E2.1;
frac(alpha > 0.95) and frac(alpha < 0.05) each <= 5%; 10/10 z dims with Var >= 0.01;
frac(alpha − prior > +0.05) and frac(< −0.05) each >= 2%), then the highest integration composite among
the subset's passing runs (`compute_integration_metrics`, k = 15, seed 42, labels `cell_type_harmonised`;
six metrics direction-corrected, min-max over passing runs, 0.4 mixing + 0.6 biology). Ties go to the
lower grid id. If no row passes, that subset stops. No Delta-alpha statistic beyond the gate, no glaucoma
gene and no biology readout enters selection.

## Seeds

Seeds 42-44 per subset: `rev-vz-<tag>-s42` is the selected grid run; `rev-vz-<tag>-s43` and `-s44` are
trained with its arguments. `select_subsets.py select --subsets <tags> --write-seed-manifest` writes
`seeds_<tag>.tsv` and `seed_map_<tag>.tsv` per subset; it is run for a subset only once all of that
subset's grid rows have ended, since the grids end in waves.

## Per-run report (`select_subsets.py profile`, not part of selection)

Runs only after every subset's selection is written: Delta-alpha sign profile against the training and raw
prior (fractions positive and negative, beyond ±0.05, SD, median), knn_mixing_entropy, mixing entropy over
each non-human species' cells, adversary accuracy, first NaN snapshot, G-V. A seed with knn_mixing_entropy
below 0.5 times the selected run's is flagged poorly mixed and reported separately; the primary D stays the
mean over seeds.
