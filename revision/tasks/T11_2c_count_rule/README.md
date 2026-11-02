# T11.2c Van Zyl full cohort: recipe by count dependence (amendment log, Z1 close-out row)

The full-cohort recipe was first chosen by the integration composite. Under the amended rule, recipes rank on count dependence. Grids 2 and 6 were not selectable before (saturation 7.7%, 6.7%, now allowed up to 10%).

**Runs.** `rev-vz-grid2-s43`, `-s44`, `rev-vz-grid6-s43`, `-s44` (`manifest.tsv`; the grid rows of `T11_2_recipe/grid.tsv`, seed replaced, on T11.1 build 3). With the s42 grid runs, and grid 1's seeds (`rev-vz-grid1-s42`, `rev-vz-s43` to `s46`).

**Rule (`select_count.py`).** Among grids 1, 2, 6 with at least 2 healthy seeds (no NaN, 10/10 active z dims, saturation at most 10%, both tails against the raw prior), per grid the mean over healthy seeds of rho = Spearman(alpha - prior as anchored, S_g_ema) on genes with count support (|S_g_ema| >= 1e-6). Highest mean wins; a difference under 0.02 goes to the lower grid id. No gold, gene set or glaucoma result enters.

**Stated.** V1-V3 on grid 1 had already been computed and read by the analyst before this rule was applied. The selection is mechanical and reads rho only. Grid 1's readouts are reported beside the winner's.
