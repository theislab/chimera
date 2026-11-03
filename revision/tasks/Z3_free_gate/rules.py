"""Z3 free gate on independent data: seed sets, lines and fixed definitions. Rule text in the private ledger Z3_free_gate.md,
committed before any gold was read on a free-gate D.

Free gate = Config M (no GIN, free per-gene c_g, same model form, package unchanged); recipe otherwise the cohort's
canonical recipe. Outputs are read from final_state.npz (alpha, prior, S_g_ema), never through ChimeraModel.load,
which does not restore per_gene_correction.

Lines, each a whole T4.3 scoreboard run with the frozen draws, golds, blocks, covariates and baselines:
  frozen  the T3.3 seeds through this harness: must return results/revision/T4_3 and T5_6/scoreboard values (check only);
  free5   Config M, 5 healthy seeds per cohort (liver H+Mouse, testis H+Mouse, liver 4-sp);
  can5    canonical recipe on the same seed numbers and platform (CPU) where they exist: the like-for-like comparator;
  can_state (liver H+Mouse) canonical seeds matched on training state to free5, other cohorts as can5.
Testis 3-sp (multi-species estimate for H2, H4) is not retrained: its rows stay canonical in every line.
D_3 for the tool comparison: the three lowest seed numbers of each line.
"""

from __future__ import annotations

from revision.common import gene_ids

REPO = gene_ids.REPO_ROOT
RES = REPO / "results/revision"
OUT = RES / "Z3"
TOOLS = RES / "T5_6/tools_primary.csv"

COHORTS = ("liv_hmouse", "test_hmouse", "liv_4sp")
# Free-gate candidates in seed order; a line takes the first 5 that pass the health rule (fixed before any was read).
FREE = {
    "liv_hmouse": {42: "rev-ctl-liv_hmouse-m-s42", 45: "rev-ctl-liv_hmouse-m-s45", 46: "rev-ctl-liv_hmouse-m-s46",
                   **{s: f"z3-liv_hmouse-m-s{s}" for s in (47, 48, 49, 50, 51, 52, 53)}},
    "test_hmouse": {s: f"z3-test_hmouse-m-s{s}" for s in (42, 43, 44, 45, 46)},
    "liv_4sp": {s: f"z3-liv_4sp-m-s{s}" for s in (42, 43, 44, 45, 46)},
}
# Canonical recipe on CPU by seed number (T3.1 s42 CPU runs, T3.2 seeds); 4-sp has no CPU s42, so its s42 is the MPS run.
CAN_CPU = {
    "liv_hmouse": {42: "rev-cpu-liv_hmouse-s42", **{s: f"rev-liv_hmouse-s{s}" for s in range(45, 52)}},
    "test_hmouse": {42: "rev-cpu-test_hmouse-s42", **{s: f"rev-test_hmouse-s{s}" for s in range(43, 52)}},
    "liv_4sp": {42: "xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42", **{s: f"rev-liv_4sp-s{s}" for s in range(43, 47)}},
}
# can5 = the canonical runs with the free5 seed numbers; a number with no canonical CPU run takes the lowest unused
# canonical CPU seed. can_state (liver H+Mouse only, where seeds fall into two training states) = canonical seeds of
# the T3.3 pool matched one for one on state (integrated / poorly mixed, O2's four markers), CPU first, by seed number.
# D_3 for the tool comparison: the three lowest seed numbers of each line (T5.6 used 42-44, the lowest of the frozen set).

# Count support: |S_ema| >= EPS in BOTH the free5 and the can5 consensus (same genes in both lines).
EPS = 1e-6

# Health rule: the Z1 rule (05 section 13.0), the one under which the frozen canonical seeds pass: finite alpha, prior,
# z and no E2.1; 10 of 10 z dimensions with Var >= 0.01; alpha - raw prior beyond +0.05 and beyond -0.05 each in at
# least 2% of genes; alpha > 0.95 and alpha < 0.05 each in at most 10%. (The literal G-V gate, 5% saturation and tails on
# the anchored prior, would drop frozen canonical seeds: liver s46, s48 on saturation, every testis seed on the clamped
# lower tail; it is reported beside.) Poorly mixed but healthy seeds stay in D, with mixing reported.
SAT_MAX, ZVAR_MIN, N_ACTIVE, TAIL_DELTA, TAIL_MIN = 0.10, 0.01, 10, 0.05, 0.02
SAT_MAX_GV = 0.05
