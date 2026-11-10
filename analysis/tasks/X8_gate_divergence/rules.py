"""X8: where the two gate read-outs place a gene differently. Definitions, fixed before any gap statistic was computed.

Two read-outs of one fit design. Canonical gate (the main model): Delta-alpha places a gene's deployment against the protein prior as
predicted from its protein-feature neighbourhood. Per-gene (free) gate: one offset per gene, Delta-alpha is the gene's own count
evidence placed against the prior. The gap marks genes whose counts place them differently from their protein neighbourhood.

Universe U_c: genes count-supported in both consensus tables (|S_ema| >= 1e-6 in both), human Ensembl ID present and unique.
Values: per seed d_s = alpha_s - pi_clamped, z_s = inverse-normal rank within U_c; consensus z_f, z_c = INT(mean d_s) (= INT of H1's D).
Gap delta = z_f - z_c. Sensitivities r_f|c and r_c|f: residual of one on a df-4 spline of the other.
Amendment 1 (after the gold-free calibration null failed, before any gold, set, constraint or gene name was read): consensus is the
mean of the seed z (equal weight per seed; INT(mean d) let one high-amplitude canonical seed dominate, so v/n understated its noise),
and seed variances are moderated by an empirical-Bayes prior (limma fitFDist, prior mean by expression decile) instead of a fixed
4-df shrinkage. SE^2 = v_f/n_f + v_c/n_c, Welch df, two-sided t p, BH q over the universe. A gate with one seed takes the prior
(variance and df) of its tissue's H+Mouse canonical table.
Call: q < Q_CALL, |delta| >= MIN_GAP, every cross-gate seed pair has the sign of delta; strict: q < Q_STRICT.
Calibration null: one gate with >= 4 seeds split into two disjoint groups (2 vs 2-3), N_SPLITS splits, same rule on z(A) - z(B).

Exploratory; nothing here changes a status, row, modifier, H1 or Z4 table. Ledger: chimera-private .../ledgers/X8_gate_divergence.md.
"""

from __future__ import annotations

from analysis.common import gene_ids

REPO = gene_ids.REPO_ROOT
RES = REPO / "results" / "analysis"
OUT = RES / "X8"
FREE = RES / "Z4" / "h1" / "genes"
CAN = RES / "H1" / "genes"

SEED = 20261103
MASK_EPS = 1e-6
MIN_GAP = 1.0
Q_CALL = 0.05
Q_STRICT = 0.01
D0_MAX = 1000.0
EXPR_DECILES = 10
N_SPLITS = 20
KNN = 15
N_PCS = 64
N_NULL_NBR = 200
N_PERM_REC = 10_000
N_BOOT_SEQ = 1000
R_MIN = 3
SPECIFIC_MAX_OTHER = 0.5
SAT_HI = 0.95
MIN_PAIR_IDENTITY = 20.0

# Where a gate has one seed, its decile variance is borrowed from the same tissue's H+Mouse canonical table.
BORROW = {"liv_hzeb": "liv_hmouse", "test_hmac": "test_hmouse"}
# Calibration: (cohort, gate) with >= 4 seeds.
CALIB = [("liv_hmouse", "can"), ("test_hmouse", "can"), ("kidney", "can"), ("islet", "can"), ("test_3sp", "can"), ("liv_4sp", "can"),
         ("liv_hmouse", "free"), ("liv_hmonkey", "free"), ("liv_4sp", "free"), ("test_hmouse", "free")]
SIBLINGS = {"liver": ["liv_hmouse", "liv_hmonkey", "liv_hzeb", "liv_4sp"], "testis": ["test_hmouse", "test_hmac", "test_3sp"],
            "eye": ["vz_hmouse", "vz_hmmul", "vz_primates", "vz_full", "vz_hmfas", "vz_hpig"],
            "cortex": ["brain_m1_hmouse", "brain_m1", "brain_dlpfc", "brain_dlpfc_hchimp", "brain_dlpfc_hmac", "brain_dlpfc_hmarm", "brain_dlpfc_hmac_own"]}
COVARIATES = ["expr", "detection", "abs_S_free", "abs_S_can", "pi_raw", "tau", "det_gap"]
COV_LABEL = {"expr": "expression", "detection": "detection", "abs_S_free": "|S_g|, per-gene fit", "abs_S_can": "|S_g|, canonical fit", "pi_raw": "protein identity",
             "tau": "cell-type specificity", "det_gap": "detection gap between species"}


def cohorts() -> dict:
    """The 26 cohorts of H1/Z4, registry order."""
    import os

    os.environ.setdefault("H1_OUT", str(RES / "H1"))
    from analysis.tasks.H1_harmonise import rules as H1R

    return {c: v for c, v in H1R.cohorts().items() if (FREE / f"{c}.parquet").exists()}


def thread() -> dict:
    from analysis.tasks.H1_harmonise import rules as H1R

    return {t: H1R.PRIMARY[t] for t in H1R.TISSUE_ORDER}
