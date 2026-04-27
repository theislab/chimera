# Chimera — Best-Known Model Registry

**Purpose.** Authoritative answer to: *"What should I run on dataset X right
now?"* For each training scenario, records: config, checkpoint path, headline
metrics (integration + alpha health + bidirectional Δα), and iteration number.

Updated atomically with every promotion — if this file, the model defaults in
`_constants.py`, and the notebook pointers ever drift apart, that is a bug.

**Last updated:** 2026-04-26, **iter124 ROLLBACK** — 4-sp pancreas adversary classifier capacity bump co-tuned with HALVED adv_weight (`--adv-classifier-hidden "64,32" --adv-weight 100` on iter101 recipe, 200ep s42, 66 min). **HARD FAIL — sil-ct collapsed FURTHER to 0.079** (vs iter123 0.096, iter101 0.408 — HARD falsifying probe <0.30 trigger); LT-acc REGRESSED FURTHER to 0.753 (vs iter123 0.893, iter101 0.985, scVI 0.929 — fails E8.6 ≥ scVI−0.05); E8 still 2/6 (same fail pattern as iter123). kNN-sp improved to 0.946 (hit predicted band 0.91-0.95) but FAIL E8.1 (Δ vs scVI=+0.016 > 0.01). 33/33 E1-E9 PASS but **α-biology DEGENERATE**: gin_signed_diagnostics revealed `mean_corr_on_pos_S_g=−1.58` (WRONG SIGN — pushes positive class toward divergence), `frac_corr_pos_on_pos_S_g=0.00` (every positive-class gene gets a negative correction). E9.2 PASS metrics MASKED this in iter123; iter124 surfaces it. z PC1/PC2 SVD aspect=5.27 (HARD thin) while UMAP aspect=1.19 (round) — linear-projection diagnostic catches encoder collapse onto thin manifold. Post-hoc MLP-64-32 z-leakage 0.738→0.694 (predicted ≤0.50 — encoder did somewhat more legitimate species-scrubbing than iter123's 0.709, but inadequate). Endothelial close-3 mixing IMPROVED (Δ vs scVI iter101 +0.128 → iter124 +0.024) but global CT structure collapsed (sil-ct 0.408→0.079). **Mechanism (definitive)**: the (64,32) × 4-sp encoder interaction is binding regardless of adv_weight — halving adv_weight does NOT halve CT damage; it makes it slightly worse. The encoder still flattens CT structure to evade the bigger classifier. **Adv-side family on 4-sp is now EXHAUSTED** (2 consecutive sil-ct collapses iter123+iter124). **Lessons new this invocation**: (1) `mean_corr_on_pos_S_g` and `frac_corr_pos_on_pos_S_g` from `gin_signed_diagnostics.py` are co-primary diagnostics for 4-sp z-side perturbation — α-biology degeneration is masked by E9.2 metrics; (2) z PC1/PC2 SVD aspect catches what UMAP aspect misses for encoder linear-subspace collapse; (3) bigger classifier × 4-sp pancreas is a fundamental mismatch — future capacity bumps to in-training adversary on 4-sp are deprecated. **iter101 retained as 4-sp pancreas best-known** (no registry change). Code preserved: both flags (`--adv-classifier-hidden`, `--adv-weight`) defaults are byte-identical to all best-knowns when not passed. **FIVE 4-sp attacks now ruled out** (iter107, 121, 122, 123, 124). Recommended next: **iter125 (η-r) — `--n-hidden 256` encoder capacity bump on iter101 recipe** (structurally different family — encoder, not adversary). PIVOT REQUIRED — see workflow.md "Active hypothesis queue" for full backup queue.

**Earlier (iter123 ROLLBACK)** — 4-sp pancreas adversary classifier capacity bump (`--adv-classifier-hidden "64,32"` on iter101 recipe, 200ep s42, ~52 min). HARD FAIL — sil-ct 0.408→0.096; E8 3/6→2/6; kNN-sp 0.979→0.962; LT-acc 0.985→0.893. 33/33 E1-E9 PASS (improved); post-hoc MLP-64-32 z-leakage 0.738→0.709 (deeper z-leakage barely budged) even though in-training adv_acc 0.765→0.389. Mechanism: bigger (64,32) classifier at FIXED `adv_weight=200` over-tips GRL game; encoder fools specific classifier instance by collapsing celltype structure (cheapest axis), not learning species-blind z. New diagnostic `training/z_leakage_probe.py` retained. iter124 (this invocation) tested the "co-tune lower" lesson directly and FAILED — see Last updated row above.

**Earlier (iter122 ROLLBACK)** — 4-sp pancreas species-pair-conditional MMD pivot (`--mmd-pair-weights "1.0,1.0,0.3,1.0,0.3,0.3"` on iter101 recipe at `--lambda-mmd 100`, 200ep s42). **TRAINING-LEVEL FAILURE — model NaN-diverged at epoch 17→18**, before FQR activation (ep20) and before KL warmup completion (ep20). All training-tracked tensors (alpha, correction, S_g_ema, mu_full, z_mean, residual) are NaN at final state. Pre-NaN trajectory was healthy through ep17 (grad_enc 1.7-2.6, L_recon 2500-2900). Mechanistic: heterogeneous per-pair weighting concentrates MMD gradient on close pairs at ~1.5× uniform per-pair (close share 1.0/3.9 = 0.256 vs uniform 1/6 = 0.167). Combined with iter101's already-aggressive regime (adv=200, lambda_mmd=100, fb=1.0, anch=0.001), the close-pair concentrated gradient pushed the encoder past stability around ep17. Per-pair heterogeneous concentration is *more numerically dangerous* than uniform high pressure — iter121 same total pressure produced graceful 4/10 z-dim collapse (no NaN); iter122 produced instant NaN. iter101 retained as 4-sp pancreas best-known (no registry change). Code preserved opt-in: `--mmd-pair-weights` flag default None = byte-identical legacy on every existing best-known (smoke test verified uniform-weight equivalence). **Three 4-sp MMD-family attacks now ruled out** (iter107 CT-cond ALONE → over-integrates; iter121 dual MMD → 4/10 z dim collapse; iter122 per-pair heterogeneous → NaN). Per refinement.md anti-pattern rule, the MMD family is exhausted on 4-sp; next attack must be a structurally-different mechanism family (encoder/residual capacity, cycle-consistency, or principled E8 metric refinement). **Lessons for any future per-pair attempt**: lower lambda_mmd to compensate (~65 if max_weight=1.0 and sum=3.9), and/or use softer weight ratios (e.g. far pairs at 0.5 instead of 0.3), and/or add explicit per-encoder gradient clipping. PIVOT REQUIRED — see workflow.md "Strategic question for the NEXT invocation".

**Earlier (iter120 ROLLBACK)** — per-gene bias hybrid WITH separate LR=1e-3 param group, on testis H+Mac, 200ep, s42: E9.2 push<-0.05 = **0.020%** (HARD FAIL primary metric, predicted 5-15%; essentially unchanged from iter109's 0.007%). **The LR fix WORKED MECHANICALLY** — bias_share=22.6% (vs corrected iter119 2.7%, 8× engagement), mean_per_gene_bias_on_neg_S_g=-0.296 (vs iter119 -0.027, 11× larger and direction correct), mean correction on neg-S_g class shifted from +0.353 (iter109) to **+0.011** (huge — class mean almost flipped to zero). E8 6/6 vs scVI(donor) MAINTAINED with sil-ct=0.346 (BEST of iter109 family, vs iter109's 0.321), LT-acc 0.881 (within E8.6 tolerance), UMAP aspect 1.07 (very round), z PC1/PC2 2.55 (better than iter119's 4.43). E3.2 frac>0.95=0.18% PASS, α.std=0.0506 PASSES E1.5 0.05 floor. **BUT primary metric stuck at 0.020% because the testis dataset's S_g_ema distribution (only 1.06% of genes with S_g<-0.05, mean=+0.007) mathematically caps E9.2 down attainable mass below the 2% threshold regardless of mechanism quality** — even with PERFECT mechanism reaching FQR equilibrium, only ~155 genes (1.06%) have target correction below -0.349. **EIGHT consecutive testis E9.2 down attacks (iter113/114/116/117/118/119/120) have failed.** Per-gene mechanism family is now EXHAUSTED — the architectural ceiling is in the FQR baseline definition (`p_null × lib`), not in any per-gene mechanism. Code change preserved opt-in (default OFF = byte-identical legacy; verified via smoke test). **iter109 retained as testis H+Mac best-known** (no registry change). PIVOT REQUIRED for next iteration — see workflow.md "Active hypothesis queue" for strategic question (3 options: 4-sp pancreas dual MMD pivot, refine E9.2 expectation to be data-adaptive, or Tier-3 FQR baseline redesign).

**Earlier (iter119 REJECTED, narrative CORRECTED 2026-04-26)** — per-gene bias hybrid (`--use-per-gene-bias`) on testis H+Mac, 200ep, s42, no LR fix: E9.2 push<-0.05 = **0.0068%** (HARD FAIL, predicted 5-15%, essentially unchanged from iter109's 0.007%). **Original "bias never moved from zero / bias_share=0.0%" claim was WRONG** — diagnostic bug in `iter119_bias_extract.py` (numpy view shared storage with tensor.data, was zeroed and not refreshed). After bug fix in iter120 invocation: corrected `bias_share_of_total_magnitude=2.7%`, mean_per_gene_bias_on_neg_S_g=**-0.0267**, mean_per_gene_bias_on_pos_S_g=**+0.0192**, abs_max=0.0565. **Bias DID engage at LR=1e-4 in the CORRECT direction**, just too weakly (~10× below equilibrium target). Root cause refined: at LR=1e-4 with per-gene mean-reduced gradient ~3e-7, Adam's normalized update is LR-bounded (eps=1e-8 doesn't dominate); cumulative movement over 20K steps is bounded by ~LR×N → max abs_max=0.057 reached, vs the equilibrium target ~±0.5. iter120 (this invocation) tested the LR=1e-3 fix and CONFIRMED the LR fix works mechanistically (bias engaged 8× more) but did NOT break the testis dataset's S_g_ema architectural ceiling. E8 6/6 vs scVI(donor=sample_batch) MAINTAINED; LT-acc 0.873 (vs iter109 0.899, within E8.6 tol); UMAP aspect 1.17 round; z PC1/PC2 4.43 (REGRESSED, same 200ep over-fit pattern as iter117/118). Code change preserved opt-in (default OFF = byte-identical). New/fixed diagnostic scripts: `iter119_bias_extract.py` (bug-fixed, `.copy()` after `.numpy()`), `gin_signed_diagnostics.py` extended with bias-share metrics. **iter109 retained as testis H+Mac best-known** (no registry change).

**Earlier (iter118 REJECTED):** asymmetric L_anchor (`--lambda-anchor-neg-factor 0.3`) on testis H+Mac, 200ep, s42: E9.2 push<-0.05 = **0.39%** (FAIL, predicted 1.5-5%). Falsifying probe failed: mean corr on neg-S_g = +0.279 (target ≤ +0.15). E8 6/6 MAINTAINED. iter118 produced the FIRST partial directional shift on rare class (mean corr 0.353→0.279, frac_corr_neg_on_neg_S_g 5%→16%, Δα<-0.02 1.6%→7.1%) — proving anchor symmetry contributes but is NOT the dominant binding constraint. **MLP feature-space smoothness is the dominant constraint.** New CLI flag `--lambda-anchor-neg-factor` preserved as opt-in (default 1.0 = byte-identical).

**Earlier (iter117 REJECTED):** testis_unified iter115 recipe at 200ep (s42) gave E9.2 down 1.04% (FAIL: predicted 1.5-2.5%, below baseline 1.57%, below 2% PASS line). Three stochastic basins observed for the same recipe + seed: {1.04, 1.57, 2.43}% for E9.2 down, mean ~1.7% < 2% PASS. **The killed-200ep run's 2.43% was a stochastic fluke.** Architectural ceiling on E9.2 down CONFIRMED for 3-sp testis_unified, joining 2-sp testis H+Mac and H+Mouse. iter115 100ep retained as testis_unified best-known. NO CODE CHANGE from iter117 (deliberate no-code reproducibility test).

**Last accepted:** 2026-04-25, iter115 ACCEPTED & PROMOTED (testis_unified, 100ep, applies iter111 coupled recipe to 3-sp).

- **★ promotion: testis_unified iter115-100ep replaces iter102 as best-known.** Recipe: prior-clamp 0.85 + CT-cond MMD on cell_ontology_class (5 shared groups across all 3 species, 100% coverage). 30/33 expectations + **6/6 E8 vs scVI(donor=sample_batch)** + **6/6 E8 vs scVI(species=batch)**.
- **E3.2 HARD FAIL CURED**: frac(α>0.95) = 29.06% → **1.32%** (22× cure; well under refinement.md target ≤ 5%). The registry's last hard-fail is now resolved.
- **E5.3** also cured (was marginal on iter102).
- **Integration metrics improved across the board** vs iter102: kNN-sp 0.792 → 0.730 (better), LT-acc 0.989 → 0.992 (BEATS scVI(species) 0.901 by +0.092 — best ever for testis_unified), sil-ct 0.347 → 0.320 (slight). Per-celltype kNN-sp Δ vs scVI(donor): -0.093 worst (Chimera better-mixed than scVI on every shared CT).
- **Alpha distribution smooth + bidirectional**: mean=0.850, std=0.055, frac>0.95=1.32%, frac<0.05=0%, Δα>0.02=41.16%, Δα<-0.02=10.56%, range [-0.121, +0.392]. Bimodality coefficient BC=0.361 (healthy, not collapsed).
- **UMAP aspect 0.91** (round, NOT thin/elongated; well below 1.5 limit). All 3 species well-mixed across all 5 cell-ontology groups.
- **E9.2 down at 1.57%** — below 2% PASS threshold but **200× better than iter102's ~0%**. The killed 200ep run reached 2.43% PASS, indicating the LR-scheduler difference between 100ep and 200ep targets matters; 100ep produces a more aggressive but slightly less downward-skewed equilibrium. Future iteration: re-run at 200ep for full E9.2 down PASS.
- **E6.1/E6.2 z-leakage gates fail** (adv_acc=0.609 vs chance=0.333). Not new from this iteration — long-standing testis pattern; iter102 also failed these.
- ★ status for testis_unified: this entry is now **best-known with E9.2 down architectural ceiling at 2-sp**, joining iter109 H+Mac and iter111 H+Mouse in the same status (the testis triad is now consistent: integration done, E9.2 down arch ceiling).

**Mechanistic finding (iter115 + brainstorm)**: 3-species training reduces per-gene S_g_ema noise AND increases the count of genes with non-trivial neg-S_g (more pairwise contrasts surface more divergent genes). The GIN MLP's discrimination problem is easier on 3-sp than 2-sp. This refines the prior "gradient signal structure" hypothesis (iter114): the primary issue is signal **DENSITY**, not signal **STRUCTURE**. **Implication for testis 2-sp**: future attacks on E9.2 down should target signal density (slower S_g_ema, larger batch, joint 3-sp model with per-2-sp evaluation) BEFORE code-changing class-balanced FQR / per-gene bias hybrid.

**Earlier (iter114 REJECTED) — preserved for historical reference:**
- iter114 testis H+Mac GIN width 32 → 64: terminated early at ep60 (verdict reached). The wider MLP (17,665 params, 2.60×) was consistently 0.01-0.02 BELOW iter109's Spearman r at every measured epoch. E9.2 push<-0.05 stayed at 0.000%. iter109 stays as testis H+Mac best-known. Combined with iter113 (λ_fit ×3, Δρ = +0.020) and the iter114_input_separability finding (input AUC=0.84 linear/0.80 nonlinear), the testis 2-sp ρ ceiling is robust to BOTH loss-weight scaling AND MLP capacity. Workflow pivoted to 3-sp unified for iter115 (chose proven recipe transfer over speculative 3rd same-family attack).

**Earlier (iter112) changes:**
- **E9.1 threshold revised: ≥ 0.50** (down from ≥ 0.90), per user direction this invocation: "cross-seed α robustness rho above 0.5 is success." Rationale: the 0.9 bar treated per-gene α as a deterministic fingerprint; with VAMP K=500 random init + adversarial training, ρ ≈ 0.6–0.85 is the attainable range. The dominant rank-ordering of conserved-vs-divergent genes is preserved at ρ ≥ 0.5 (top-10% high-α Jaccard ≈ 0.44 verifies headline-scientific-story stability). Updated in `workflow.md` E9.1 row with full rationale.
- **iter112 result on H+Mouse pancreas (3 seeds: 42, 43, 44) — under revised E9.1**:
  - Each seed independently: **33/33 + 6/6 E8 + E9.1 PASS ✓** (E1–E7+E9 33-gate suite all pass each seed; E8 vs scVI(batch=batch) 6/6 each seed; E9.1 cross-seed Spearman ρ min = 0.762 ≥ 0.50 PASS).
  - Multi-seed metrics: LT-acc 0.971 ± 0.005; sil-ct 0.444 ± 0.036; kNN-sp 0.853 ± 0.038; α mean 0.762 ± 0.007; α std 0.150 ± 0.021; UMAP aspect 1.34 ± 0.12 (round, not thin).
  - **★ awarded** to H+Mouse iter90 (3-seed) — the first config in the registry to pass the full expectation suite (including multi-seed E9.1) under refinement.md "definition of done."
  - The "stronger adv + free_bits → tighter cross-seed" hypothesis is still **FALSIFIED** at the magnitude level (H+Mouse 0.762 < H+Mo 0.83) but no longer matters for the gate's pass/fail decision; both pass under revised threshold.
- **H+Mo iter90 also re-gated**: cross-seed ρ = 0.83 ≥ 0.50 → E9.1 PASS → **★ awarded** to H+Mo iter90 (3-seed). Both 2-species pancreas configs now meet the full bar.
- **Identifiability audit (user-prompted)**: pancreas H+Mouse / H+Zebrafish have `batch == species` (1-to-1). H+Mo has 3 batches (Monkey has 2 within-species batches; ConstrainedBatchOffset has ~1 dof). NO `sample_batch` column in pancreas datasets. Testis datasets DO have `sample_batch` with 5–9 donor batches per species, but Chimera uses `batch_key=batch` while scVI baseline uses `batch_key=sample_batch` — this is the asymmetry the user flagged. iter94 tested `batch_key=sample_batch` on Chimera, rolled back (z contracted 4×). The interaction between Chimera's ConstrainedBatchOffset / adversary / MMD and a multi-batch-per-species `batch_key` is an **open question deferred to iter114+**.
- **iter113 plan (next invocation)**: revert to **the workflow-original iter112 plan** (now iter113): apply iter111 coupled recipe (prior-clamp 0.85 + CT-cond MMD on cell_ontology_class) to testis unified. Closes the registry's only remaining hard-fail (testis unified E3.2 sat 29%) and completes the testis-tissue story across all 3 datasets.

(Prior history retained from iter111 below.)

- **iter111 PROMOTED — testis H+Mouse**: applied --prior-clamp-max 0.85 +
- **iter111 PROMOTED — testis H+Mouse**: applied --prior-clamp-max 0.85 +
  --celltype-conditional-mmd --celltype-key **cell_ontology_class** (NEW: not
  cell_type) to iter91b recipe. **E3.2 saturation cured: 35% → 0.231%**
  (152× reduction). z stays HEALTHY at 10/10 active dims (vs iter110 collapse
  to 3/10). Integration metrics dramatically improved over iter91b:
  - kNN-sp 0.991 → **0.813** (Chimera 0.173 BELOW scVI(donor)=0.986)
  - LT-acc 0.863 → **0.993** (Chimera +0.097 ABOVE scVI(donor)=0.896 — best ever for testis H+Mouse)
  - sil-ct 0.255 → 0.244 (held)
  - All 3 shared CTs better mixed (Elongated_spermatids near chance)
  - E6.1 newly passes (was failing on iter91b)
  - E8 6/6 maintained
  Cost: E1.5/E3.2 std/E3.4 mag-path now FAIL marginally (α.std=0.04952,
  0.0005 below 0.05 floor — same clamp-narrowing edge as iter109 H+Mac).
  E9.2 push<-0.05 still 0% (universal GIN ceiling, Tier 3 work).
  **Net**: 28/33 vs iter91b's 29/33 (one fewer pass), but qualitatively a
  strict upgrade — the headline E3.2 35% saturation gate is cured, and
  integration metrics are much better.
- **KEY ENGINEERING INSIGHT (iter111)**: When 2-species datasets have
  asymmetric celltype-label granularity (Mouse testis labels are coarser
  than Human/Macaque), CT-conditional MMD on `cell_type` covers only the
  cells with shared fine labels (38% on H+Mouse). The `cell_ontology_class`
  column harmonizes labels at a coarser level (5 shared groups, 100%
  coverage on H+Mouse), restoring CT-cond MMD's celltype-preserving effect.
  **General rule: when shared-CT coverage on `cell_type` is < 80%, default
  to `cell_ontology_class` for CT-cond MMD.**
- **Testis-tissue coupled-pair confirmed across H+Mac (iter109) and H+Mouse
  (iter111)**: prior-clamp at 0.85 + CT-cond MMD on a label that covers
  100% of cells is the testis-tissue recipe. Both flags are required —
  iter110 demonstrated the clamp alone collapses posterior.
- **iter112 plan**: apply same coupled recipe to testis unified (3-species,
  5 shared cell_ontology_class groups across all 3 species, 100% coverage).
  Predicted to cure E3.2 sat 29% → ~0.5%, maintain E8 6/6.

(Prior history retained from iter110 below.)

- **iter110 ROLLBACK on testis H+Mouse**: applied --prior-clamp-max 0.85 to
  iter91b recipe (without --celltype-conditional-mmd) → catastrophic celltype
  collapse: LT-acc 0.863 → 0.278, sil-ct 0.255 → 0.091, z active dims 10/10
  → 3/10 (posterior collapse). E8 dropped 6/6 → 4/6.
- **iter110 FINDING (now superseded by iter111)**: The iter109 recipe is a
  COUPLED PAIR (prior-clamp + CT-conditional MMD), NOT two independent
  overrides. Without CT-cond MMD, the prior-clamp inflates the residual's
  role, and the residual eats celltype info from z (since global MMD is
  celltype-agnostic). iter111 confirmed this — and additionally identified
  that for asymmetric-label datasets, the celltype-key choice (cell_type vs
  cell_ontology_class) determines whether CT-cond MMD covers enough cells
  to be celltype-preserving.

(Prior history retained from iter109 below.)

- **iter109 — Testis H+Mac prior-cap fix**: `--prior-clamp-max 0.85` cures
  E3.2 saturation from 26.17% to **0.06%** (clamping the prior at 0.85 prevents
  the 32% of testis genes pre-clamped at 0.95 from saturating α from the
  prior alone). E9.2 push>+0.05 also fixed (3.4% → 8.2%). E8 maintained 6/6
  with **LT-acc improvement** (0.880 → 0.899) and **better UMAP** (aspect
  1.30 → 1.17). Cost: E1.5 marginal numerical FAIL (std=0.0499 vs 0.05).
- **iter109 still does NOT pass E9.2 push<-0.05** (0.01% measured, ≥2% needed).
  Mechanism diagnosed: GIN MLP architectural ceiling — only 1.09% of testis
  genes have S_g_ema<-0.05, and their gene features overlap with positive-S_g
  genes' features, so the smooth MLP pulls them positive. Spearman r(S_g, corr)
  = 0.518 (not the ~1.0 expected at α-detach equilibrium). corr_min plateaus at
  -0.304 across epochs 30→200. Not solvable by tuning the prior cap or anchor;
  needs Tier 3 FQR-baseline redesign.
- **★ remains UNRESTORED on testis H+Mac** because refinement.md "definition
  of done" requires BOTH integration AND α biology passing E9.2. iter109 closes
  the upward direction and saturation; downward push pending iter111+ Tier 3.
- **WARNING for the iter109 recipe**: do NOT apply --prior-clamp-max alone on
  testis datasets without --celltype-conditional-mmd. The coupling is required
  to prevent posterior collapse + LT-acc destruction (see iter110).

(Prior history retained from iter108 below.)

- **workflow.md E3.2 tightened** to refinement.md target (frac sat ≤ 5%, was 50%).
- **NEW E9.1** (cross-seed α robustness, Spearman ρ **≥ 0.50** — *revised 2026-04-25 from prior ≥ 0.90 per user direction; see workflow.md E9.1 row*).
- **NEW E9.2** (GIN bidirectional push, both push>+0.05 ≥ 2% AND push<-0.05 ≥ 2%).
- **Testis ★ REVOKED** on H+Mac iter98 and unified iter102 — both fail tightened E3.2
  (26.17% / 29.06% sat) and E9.2 (0.01% / 0.5% downward push). They remain best-known
  for INTEGRATION; α biology is an open gap. Star is conditional on BOTH integration
  AND α biology passing per refinement.md "definition of done".
- **Pancreas baselines re-trained** with batch_key="batch" (donors) for fair comparison;
  Chimera margins are larger vs scVI(donor) than vs scVI(species). All H+Mo / H+Mouse
  E8 gates still pass. 4sp pancreas + testis unified results pending (sequential MPS).
- **α cross-seed ρ on H+Mo (3 seeds) = 0.83** — under prior E9.1 ≥ 0.90 was MARGINAL FAIL; under **revised E9.1 ≥ 0.50** (2026-04-25) → **PASS**.
- **iter106 line preserved**: E2.7 lambda_res bug corrected (50× inflation); testis
  baselines updated to scVI(donor) at iter106.

---

## Quick reference — what to run

| Scenario | CLI overrides (from defaults) | Status (post iter111) |
|----------|-------------------------------|--------|
| H+Mo (close, 2-sp, pancreas) | *none* | ★ **32/33 + 6/6 E8 + E9.1 PASS** (3 seeds vs scVI(donor); E3.2 marginal 6.0–6.7% sat; **E9.1 ρ=0.83 PASS** ≥ 0.50 revised). |
| H+Mouse (mid, 2-sp, pancreas) | `--adv-weight 200 --free-bits 1.0` | ★ **33/33 + 6/6 E8 + E9.1 PASS** (3 seeds 42, 43, 44, iter112 multi-seed; **E9.1 ρ_min=0.762 PASS** ≥ 0.50 revised; LT-acc 0.971 ± 0.005; sil-ct 0.444 ± 0.036; kNN-sp 0.853 ± 0.038; UMAP aspect 1.34 ± 0.12). |
| H+Zebrafish (very, 2-sp, pancreas) | `--lambda-mmd 10` | **28/33 + 4/6 E8** structural (E1.3, E2.7, E6.1, E6.2 + E9.2 fail at 450 Myr) |
| 4-species pancreas | `--adv-weight 200 --lambda-anchor 0.001 --lambda-mmd 100 --free-bits 1.0` | **31/33 + 3/6 E8** (iter101; kNN-sp=0.979 vs scVI(donor)=0.930; STRUCTURAL — only 2 shared CTs across 4 species; iter108 confirmed) |
| Testis H+Mac (close, 2-sp) | `--adv-weight 100 --free-bits 1.0 --lambda-mmd 50 --celltype-conditional-mmd --celltype-key cell_type --prior-clamp-max 0.85 --max-epochs 200` | **30/33 + 6/6 E8** (iter109; **E3.2 cured: 0.06% sat**, E9.2 upward PASS 8.2%, downward still 0.01% — GIN ceiling. E1.5 marginal FAIL std=0.0499. LT-acc 0.899 > iter98 0.880; star NOT restored until E9.2 downward passes.) |
| Testis H+Mouse (mid, 2-sp) | `--adv-weight 200 --free-bits 1.0 --lambda-mmd 50 --celltype-conditional-mmd --celltype-key cell_ontology_class --prior-clamp-max 0.85 --max-epochs 200` | **28/33 + 6/6 E8** (iter111; **E3.2 saturation cured: 35% → 0.231%**, E6.1 newly passes, kNN-sp 0.991→**0.813** (BELOW scVI 0.986!), LT-acc 0.863→**0.993** (+0.097 BEATS scVI). E1.5/E3.2/E3.4 marginal FAIL on α.std=0.0495 (clamp narrowing). E9.2 push<-0.05=0% architectural. ★ NOT restored until E9.2 down passes.) |
| Testis unified (3-sp) | `--adv-weight 100 --lambda-anchor 0.001 --free-bits 1.0 --lambda-mmd 50 --celltype-conditional-mmd --celltype-key cell_ontology_class --prior-clamp-max 0.85 --max-epochs 100` | **30/33 + 6/6 E8 vs scVI(donor) + 6/6 E8 vs scVI(species)** (iter115; **E3.2 HARD FAIL CURED 29.06% → 1.32%**, E9.2 down 1.57% close to 2% PASS, LT-acc 0.992 BEATS scVI(species) 0.901 by +0.092, UMAP aspect 0.91, Δα bidirectional 41%/11%). E9.2 down still architectural at 2% threshold; 200ep schedule reaches 2.43% PASS. Best-known with E9.2 down arch ceiling, same status as iter109 H+Mac and iter111 H+Mouse. |
| van Zyl (2-organism) | *not yet evaluated* | No processed data; prep needed |

---

## Per-scenario details

### 1. H+Mo — 2-species close-primate (camex liver)

**Iteration:** iter90 | **Data:** `camex_human_monkey.h5ad` (**corrected**, all 72 GIN features active)

```bash
python training/run_experiment.py \
  --run-name <name> \
  --config A --max-epochs 150 \
  --adata data/processed/camex_human_monkey.h5ad \
  --seed 42 --accelerator mps
```

No overrides — model defaults match this recipe (max_epochs now defaults to 150).

**Config:** `adv_weight=100, lambda_mmd=50, free_bits=0.5, n_latent=10,
n_hidden=128, VAMP K=500, warmup_adv=5, warmup_fqr=20, warmup_kl=20,
lambda_fit=0.1, lambda_anchor=0.003, lambda_res=0.001, lambda_z_balance=1.0,
max_epochs=150`

**Changes vs iter85:** `max_epochs` 100 → 150. Longer training is the single
highest-leverage lever for both integration AND Δα: kNN-sp 0.781→0.718 (s42),
0.912→0.760 (s43); Δα- 22.4%→39.1% (s43 negative tail nearly doubles).

**E9.1 cross-seed α (3 seeds, Spearman ρ): min ρ = 0.83 → PASS ≥ 0.50 (revised 2026-04-25). ★ awarded.**

| Metric | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|--------|---------|---------|---------|------------|
| E1-E7 | **32/32** | **32/32** | 31/32 | — |
| E8 (vs scVI) | **6/6** | **6/6** | **6/6** | **6/6** |
| Δα > 0.02 | 32.2% | 31.8% | 31.8% | 31.9 ± 0.2% |
| Δα < -0.02 | 37.4% | 39.1% | 37.7% | 38.1 ± 0.9% |
| alpha mean | 0.784 | 0.784 | 0.783 | 0.784 ± 0.001 |
| alpha std | 0.142 | 0.140 | 0.143 | 0.142 ± 0.002 |
| frac α>0.95 | 6.2% | 5.8% | 6.5% | 6.2 ± 0.4% |
| z_min_var | 9.68 | 7.27 | 10.92 | 9.29 ± 1.84 |
| kNN-sp (↓) | **0.718** | **0.760** | **0.793** | **0.757 ± 0.038** |
| sil-ct (↑) | 0.552 | 0.507 | 0.549 | 0.536 ± 0.025 |
| LT-acc (↑) | 0.989 | 0.989 | 0.989 | 0.989 ± 0.000 |

**Checkpoints (corrected data):**
- `training/runs/iter90-150ep-s42/model/model.pt`
- `training/runs/iter90-150ep-s43/model/model.pt`
- `training/runs/iter90-150ep-s44/model/model.pt`

---

### 2. H+Mouse — 2-species mid-divergent (camex liver)

**Iteration:** iter90 (xds) | **Data:** `camex_human_mouse.h5ad` (**corrected**)

```bash
python training/run_experiment.py \
  --run-name <name> \
  --config A --max-epochs 150 \
  --adata data/processed/camex_human_mouse.h5ad \
  --adv-weight 200 --free-bits 1.0 \
  --seed 42 --accelerator mps
```

**Overrides vs default:** `--adv-weight 200 --free-bits 1.0`
(mid-divergent needs stronger adversary + higher KL floor)

**Multi-seed (3-seed verification iter112, 2026-04-25):**

| Metric | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|--------|---------|---------|---------|------------|
| E1-E7+E9.2 | **33/33** | **33/33** | **33/33** | — |
| E8 (vs scVI) | **6/6** | **6/6** | **6/6** | **6/6** |
| Δα > 0.02 | 12.1% | 26.6% | 25.5% | 21.4 ± 6.6% |
| Δα < -0.02 | 50.0% | 47.8% | 48.4% | 48.7 ± 1.0% |
| Δα > 0.05 | 3.81% | 14.28% | 12.73% | 10.27 ± 4.59% |
| Δα < -0.05 | 25.46% | 33.14% | 33.26% | 30.62 ± 3.65% |
| alpha mean | 0.771 | 0.753 | 0.761 | 0.762 ± 0.007 |
| alpha std | 0.122 | 0.174 | 0.153 | 0.150 ± 0.021 |
| frac α<0.05 | 0.000% | 0.836% | 0.097% | 0.31 ± 0.36% |
| frac α>0.95 | 1.671% | 4.881% | 4.227% | 3.59 ± 1.39% |
| z_min_var | 4.53 | 10.16 | 7.18 | 7.29 ± 2.30 |
| kNN-sp (↓) | 0.901 | 0.808 | 0.851 | 0.853 ± 0.038 |
| sil-ct (↑) | 0.402 | 0.438 | 0.491 | 0.444 ± 0.036 |
| LT-acc (↑) | 0.964 | 0.972 | 0.976 | 0.971 ± 0.005 |
| adv_acc | 0.516 | 0.506 | 0.518 | 0.513 ± 0.005 |
| residual_rms | 4.01 | 3.82 | 3.90 | 3.91 ± 0.08 |
| UMAP aspect | 1.25 | 1.51 | 1.25 | 1.34 ± 0.12 |

**E9.1 cross-seed α robustness (3-seed)**: min Spearman ρ = **0.762** → **PASS ≥ 0.50** (revised threshold, 2026-04-25).
- ρ(s42, s43) = 0.762
- ρ(s42, s44) = 0.828
- ρ(s43, s44) = 0.793

Cross-seed Δα Spearman ρ (the GIN correction signal): 0.28–0.40 — weaker
than α correlation, indicating the small-magnitude long-tail per-gene
GIN corrections are seed-noisy at this regularization regime. Top-K α
gene overlap (mean Jaccard): top-1% = 0.24, top-5% = 0.30, top-10% = 0.44.
**Headline scientific story (broad ordering of conserved-vs-divergent
genes, top-K most-conserved/most-divergent gene-sets) is preserved
across seeds; long-tail per-gene rank ordering is approximate.**

**Checkpoints (corrected data, 3 seeds):**
- `training/runs/xds-hmouse-iter90-150ep-s42/model/model.pt`
- `training/runs/xds-hmouse-iter90-150ep-s43/model/model.pt`
- `training/runs/xds-hmouse-iter90-150ep-s44/model/model.pt`

UMAP cross-seed comparison: `training/iter112_diagnostics/umaps_hmouse_3seeds.png`.

---

### 3. H+Zebrafish — 2-species very-divergent (camex liver)

**Iteration:** iter85 (xds) | **Data:** `camex_human_zebrafish.h5ad` (**corrected**)

```bash
python training/run_experiment.py \
  --run-name <name> \
  --config A --max-epochs 100 \
  --adata data/processed/camex_human_zebrafish.h5ad \
  --lambda-mmd 10 \
  --seed 42 --accelerator mps
```

**Overrides vs default:** `--lambda-mmd 10` (strong MMD over-integrates
very-divergent species; 10 preserves celltype structure)

Known structural limits: E1.3 (p_null range), E2.7 (residual L1 too small
at 450 Myr — structural), E6.1/E6.2 (adversary — species too divergent to
mix in z), E8.3 (sil-ct structurally negative at 450 Myr divergence),
E8.6 (LT-acc gap structural).

| Metric | Seed 42 |
|--------|---------|
| E1-E7 | 28/32 (E1.3, E2.7, E6.1, E6.2 — all structural at 450 Myr) |
| E8 (vs scVI) | 4/6 (E8.3, E8.6 structural) |
| Δα > 0.02 | 23.2% |
| Δα < -0.02 | 2.4% |
| alpha mean | 0.815 |
| alpha std | 0.096 |
| kNN-sp (↓) | 1.000 (species not mixable at this divergence) |
| sil-ct (↑) | -0.093 (structural limit) |
| LT-acc (↑) | 0.704 |
| adv_acc | 1.000 |

**Checkpoint (corrected data):**
`training/runs/xds-hzeb-iter85-s42/model/model.pt`

---

### 4. 4-species pancreas (H+Mo+Mouse+Zebrafish)

**Iteration:** iter101 (best) | **Data:** `camex_pancreas_unified.h5ad` (**corrected**)

```bash
python training/run_experiment.py \
  --run-name <name> \
  --config A --max-epochs 200 \
  --adata data/processed/camex_pancreas_unified.h5ad \
  --adv-weight 200 --lambda-anchor 0.001 --free-bits 1.0 \
  --lambda-mmd 100 \
  --seed 42 --accelerator mps
```

**Overrides vs default:** `--adv-weight 200 --lambda-anchor 0.001 --lambda-mmd 100 --free-bits 1.0`

**Status: 3/6 E8 (iter101).** kNN-sp=0.979 is the binding constraint.
Only 2 shared CTs (Hepatocytes 16222, Endothelial 2733). Species form
distinct local manifolds with limited bridges. scVI also at 0.917 (not
great mixing). This may be a structural limit at 4-species scale with
only 2 shared celltypes.

Note: 4-species requires `--lambda-anchor 0.001` (not the default 0.003) to
get meaningful Δα. With default anchor, Δα collapses to <10% on both sides.

| Metric | iter101 (adv=200, mmd=100, fb=1.0) |
|--------|-------------------------------|
| E1-E7 | 30/32 (E6.1, E6.2) |
| E8 | 3/6 (E8.2, E8.3, E8.6 pass) |
| Δα > 0.02 | 6.5% |
| Δα < -0.02 | 53.8% |
| alpha mean | 0.754 |
| alpha std | 0.130 |
| kNN-sp (↓) | 0.979 (scVI: 0.917) |
| sil-ct (↑) | **0.408** (best ever for 4sp) |
| LT-acc (↑) | **0.985** (scVI: 0.916, Chimera beats) |
| adv_acc | 0.765 |

**Checkpoint (corrected data):**
`training/runs/xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42/model/model.pt`

**Iteration history:**
- iter89 (adv=80, anch=0.001): 2/6 E8 — kNN-sp 0.981
- iter92 (mmd=200): z COLLAPSED (9/10 dead dims)
- iter95 (mmd=100, no fb): z COLLAPSED
- iter97 (mmd=75, fb=0.5): z collapsed
- iter99 (mmd=75, fb=1.0): z survived, 2/6 E8 (no mixing improvement)
- iter100 (mmd=100, fb=1.0): 3/6 E8, z healthy, sil-ct=0.330
- **iter101 (adv=200, mmd=100, fb=1.0)**: 3/6 E8, sil-ct=0.408, LT-acc=0.985
**Next:** iter105 (CT-cond MMD on 4sp — pre-change template in workflow.md).

---

### 5. van Zyl — 2-organism (aqueous humor)

**Not yet evaluated.** No processed `.h5ad` for van Zyl exists in
`data/processed/`. Data prep via `01_data_preparation.ipynb` +
`build_subset.py` is needed before this scenario can be tested.

Expected recipe: start from H+Mo defaults (no overrides) and evaluate.

---

### 6. Testis H+Macaque — 2-species close-primate (CAMEX testis)

**Iteration:** iter109 (CT-cond MMD + prior cap 0.85) | **Data:** `camex_testis_human_macaque.h5ad`

```bash
python training/run_experiment.py \
  --run-name <name> \
  --config A --max-epochs 200 \
  --adata data/processed/camex_testis_human_macaque.h5ad \
  --adv-weight 100 --free-bits 1.0 --lambda-mmd 50 \
  --celltype-conditional-mmd --celltype-key cell_type \
  --prior-clamp-max 0.85 \
  --seed 42 --accelerator mps
```

Recipe: adv=100, lambda_mmd=50 (celltype-conditional), free_bits=1.0,
prior_clamp_max=0.85, 200 epochs. CT-conditional MMD targets within-celltype
species mixing while preserving celltype boundaries. iter109 added the 0.85
prior cap because 32% of testis genes have prior at the 0.95 data cap, which
forced α to saturate from the prior alone (E3.2 26% in iter98). The cap drops
saturation to 0.06% with no regressions to E8.

**Dataset:** 23,593 cells (Human: 15,980, Macaque: 7,613), 14,640 genes,
10 shared cell types (spermatogenesis lineage + somatic). Conservation prior
mean=0.890, std=0.079 (higher than pancreas 0.800 — testis genes more conserved).
Note: `batch == species` (ConstrainedBatchOffset dead). Do NOT use `sample_batch`
as batch_key — iter94 proved harmful (z contracts 4×, spermatocyte mixing destroyed).

**scVI baseline (donor, batch_key=sample_batch):** kNN-sp=0.933, sil-sp=-0.006, sil-ct=0.265, LT-acc=0.914.
*(Species baseline preserved in integration_vs_scvi_bk_species.json)*

**Results (iter109 s42, with --prior-clamp-max 0.85):**

| Metric | Seed 42 |
|--------|---------|
| E1-E7 + E9 | **30/33** (E1.5 marginal std=0.0499, E6.1, E9.2 down — E3.2 NOW PASS 0.06%) |
| E8 (vs scVI(donor)) | **6/6** |
| kNN-sp (↓) | **0.686** (scVI: 0.933, Δ=-0.247 — Chimera far better) |
| sil-ct (↑) | **0.321** (scVI: 0.265 — Chimera better) |
| LT-acc (↑) | **0.899** (scVI: 0.914, Δ=-0.015 — improved over iter98 0.880) |
| E8.4 worst gap | **-0.154** on Undif_Sperm (all CTs below scVI) |
| E8.5 worst-ct excess | **+0.211** on Other_somatic (was +0.295 iter98) |
| frac(α>0.95) — E3.2 | **0.06%** (was 26.17% iter98 — cured) |
| frac(α-prior>+0.05) — E9.2 up | **8.20%** (was 3.44% iter98 — PASS) |
| frac(α-prior<-0.05) — E9.2 down | **0.01%** (FAIL — GIN ceiling) |
| Δα > 0.02 (vs clamped prior) | 29.4% (vs raw 10.3%) |
| Δα < -0.02 (vs clamped prior) | 1.6% (vs raw 66.3%) |
| α mean | 0.847 |
| α std | 0.050 |
| z min_var | 4.46 |
| z var max/min | 2.66 (was 3.43 iter98 — more balanced) |
| UMAP aspect | 1.17 (was 1.30 iter98 — more circular) |
| adv_acc | 0.691 (was 0.78 iter98 — more species-mixed) |

**Checkpoint (iter109):** `training/runs/testis-hmac-iter109-pclamp085-200ep-s42/model/model.pt`
**Raw prior (unclamped):** `training/runs/testis-hmac-iter109-pclamp085-200ep-s42/conservation_prior_raw.npy`
**iter98 checkpoint preserved:** `training/runs/testis-hmac-iter98-ctmmd50-200ep-s42/model/model.pt`

**Iteration history:**
- iter93 (adv=100, 200ep): 29/32 + 3/6 E8 — Sertoli +0.029, Undiff_Sperm +0.060
- iter94 (sample_batch): ROLLED BACK — z contracted 4×
- iter96 (adv=200, fb=1.0): 29/32 + 4/6 E8 — Undiff_Sperm fixed, Sertoli +0.021, E8.6 regressed
- iter98 (CT-cond MMD): 30/32 + 6/6 E8 — all celltypes pass, LT-acc 0.880
- iter106 (eval fix): 31/32 + 6/6 E8 — E2.7 bug fixed, donor baseline verified
- **iter109 (prior cap 0.85)**: 30/33 + **6/6 E8** — E3.2 saturation cured (26%→0.06%),
  upward push doubled (3.4%→8.2%), LT-acc improved (0.880→0.899). Downward push
  still 0.01% (GIN MLP ceiling on rare negative-S_g genes). E1.5 marginal numerical
  FAIL (std=0.0499 vs 0.05) — known cost of clamping 77% of testis genes.

**Key insights:**
- (iter98) Global adversarial pressure trades off species mixing vs celltype
  separation. CT-conditional MMD breaks this tradeoff by targeting mixing within
  each celltype while preserving between-celltype structure.
- (iter109) Testis-tissue ESM-2 conservation priors are 8× more often pre-clamped
  at the data prep ceiling (0.95) than pancreas datasets (32% vs 4%). This pre-
  clamp lock causes α to saturate from the prior alone, independent of GIN
  action. Capping the prior at 0.85 in-memory before training removes the lock
  with no integration cost. **This is a testis-tissue-specific override**, not a
  global default — pancreas datasets continue to use 0.95 cap with no regression.
- (iter109) The downward GIN push limit is architectural, not a tuning knob:
  Spearman r(S_g_ema, correction) = 0.518 (would be ~1.0 if GIN had full
  expressive power). The ~1% of negative-S_g genes have feature-space neighbors
  that are positive-S_g, so the smooth MLP pulls them positive. Future Tier 3
  work on FQR baseline (replace null with mu_full_ema for naturally bidirectional
  S_g) is needed to fully pass E9.2.

---

### 7. Testis H+Mouse — 2-species mid-divergent (CAMEX testis)

**Iteration:** iter111 (coupled prior-clamp + CT-cond MMD on cell_ontology_class) | **Data:** `camex_testis_human_mouse.h5ad`

```bash
python training/run_experiment.py \
  --run-name <name> \
  --config A --max-epochs 200 \
  --adata data/processed/camex_testis_human_mouse.h5ad \
  --adv-weight 200 --free-bits 1.0 \
  --celltype-conditional-mmd --celltype-key cell_ontology_class \
  --prior-clamp-max 0.85 \
  --seed 42 --accelerator mps
```

**Recipe:** adv=200, lambda_mmd=50 (celltype-conditional, default), free_bits=1.0,
prior_clamp_max=0.85, **celltype_key=cell_ontology_class** (NEW vs cell_type), 200 epochs.
The cell_ontology_class column harmonizes Mouse/Human labels at a coarser
level (5 shared groups, 100% cell coverage) where cell_type only has 3 shared
groups (38% coverage). Without this label switch, iter110 catastrophically
collapsed posterior. iter111 demonstrates that the coupled iter109 recipe
transfers cleanly across testis 2-species datasets when the celltype_key
provides ≥80% cell coverage.

**Dataset:** 21,241 cells (Human: 15,980, Mouse: 5,261), 14,731 genes.
3 shared cell_type labels (Elongated_spermatids, Other_somatic, Sertoli, 38%
coverage). 5 shared cell_ontology_class labels (Sertoli, Spermatocytes,
Spermatogonia, somatic, spermatids, 100% coverage). cell_type→
cell_ontology_class is a clean hierarchical mapping in the dataset.
Conservation prior mean=0.890, std=0.079 (same as H+Mac — testis genes
more conserved than pancreas).

**scVI baseline (donor, batch_key=sample_batch):** kNN-sp=0.986, sil-sp=0.039,
sil-ct=0.201, LT-acc=0.896.

**Results (iter111 s42):**

| Metric | Seed 42 | iter91b baseline | Δ |
|--------|---------|-----------------|---|
| E1-E7 + E9 | **28/33** (E1.5, E3.2 std, E3.4, E6.2, E9.2 down — all marginal/architectural) | 29/33 | -1 numerical, +1 qualitative (sat cured) |
| E8 (vs scVI(donor)) | **6/6** | 6/6 | maintained ✓ |
| **kNN-sp (↓)** | **0.813** (scVI: 0.986, Δ=**-0.173** — Chimera FAR better) | 0.991 (Δ=+0.005) | -0.178 ✓✓ |
| **sil-sp** | 0.040 (scVI: 0.039) | 0.013 | +0.027 |
| **sil-ct (↑)** | **0.244** (scVI: 0.201, Δ=+0.043) | 0.255 (Δ=+0.054) | -0.011 (held) |
| **kNN-ct (↑)** | 0.799 (scVI: 0.929) | 0.940 | -0.141 (CT-cond MMD trades local CT purity for species mixing; see sil-ct + LT-acc) |
| **LT-acc (↑)** | **0.993** (scVI: 0.896, Δ=**+0.097** — Chimera BEATS scVI) | 0.863 (Δ=-0.033) | +0.130 ✓✓ |
| **E3.2 frac(α>0.95)** | **0.231%** (was 35% on iter91b, 152× reduction) | 35.00% | **CURED** ✓ |
| Δα > 0.02 (vs clamped prior) | 26.97% | 30.8% | -3.83 |
| Δα < -0.02 (vs clamped prior) | 0.18% | 3.48% | -3.30 (architectural, GIN ceiling) |
| Δα > 0.02 (vs RAW prior) | 9.76% | n/a | new measure |
| Δα < -0.02 (vs RAW prior) | **66.34%** | n/a | strong biological bidirectionality vs RAW prior |
| α mean | 0.847 | 0.907 | -0.060 (clamped) |
| α std | 0.0495 | 0.076 | -0.026 (clamp narrowing — root cause of E1.5/E3.4 marginal fails) |
| z_min_var (E6.3) | **8.59** | 0.371 | +8.22 ✓ much larger z volume |
| z active dims (var > 0.1) | **10/10** | 10/10 | maintained ✓ (vs iter110's 3/10 collapse) |
| adv_acc | **0.741** | 0.843 | -0.102 ✓ closer to chance |
| residual_rms_mean | 2.99 | 1.09 | +1.90 (residual carries clamped genes' variance) |

**Per-celltype kNN species purity (chance = species ratio):**

| Celltype | n | chance | iter111 (excess) | iter91b (excess) | scVI (excess) | iter111 vs iter91b |
|---|---|---|---|---|---|---|
| Elongated_spermatids | 3497 | 0.687 | **0.726 (+0.039)** | 0.998 (+0.311) | 0.996 (+0.309) | -0.272 ✓ near-chance mixing |
| Other_somatic | 2832 | 0.508 | 0.850 (+0.343) | 0.956 (+0.449) | 0.986 (+0.479) | -0.106 ✓ |
| Sertoli | 1833 | 0.723 | 0.878 (+0.155) | 0.998 (+0.275) | 0.996 (+0.273) | -0.120 ✓ |

**Checkpoint (iter111):** `training/runs/testis-hmouse-iter111-pclamp-coc-200ep-s42/model/model.pt`
**Raw prior (unclamped):** `training/runs/testis-hmouse-iter111-pclamp-coc-200ep-s42/conservation_prior_raw.npy`
**iter91b checkpoint preserved:** `training/runs/testis-hmouse-iter91b-s42/model/model.pt`

**Iteration history (testis H+Mouse):**
- iter91b (adv=200, 150ep): 29/33 + 6/6 E8 — E3.2 35% sat (HARD FAIL), E6.1, E6.2, E9.2 down fail
- iter110 (clamp 0.85, no CT-cond MMD): ROLLED BACK — z collapsed 3/10, LT-acc 0.86→0.28
- **iter111 (clamp 0.85 + CT-cond MMD on cell_ontology_class)**: 28/33 + 6/6 E8 —
  E3.2 saturation cured (35% → 0.23%), kNN-sp 0.991 → 0.813, LT-acc 0.863 →
  0.993, E6.1 newly passes. E9.2 push<-0.05 still 0% (universal GIN ceiling
  on testis-tissue prior). ★ NOT restored until E9.2 down passes.

**Key insights (iter111):**
- The cell_ontology_class label is the harmonized version of cell_type for
  testis Mouse/Human/Macaque data. cell_type has asymmetric granularity (Mouse
  is coarser); cell_ontology_class fixes this with 5 shared groups covering
  100% of cells.
- CT-cond MMD effectiveness depends on shared-CT cell coverage. iter110
  failed because cell_type only covered 38% of cells. iter111 succeeded
  because cell_ontology_class covers 100%. **Rule: when shared-CT coverage
  on `cell_type` is < 80%, use `cell_ontology_class` for CT-cond MMD.**
- Coarse ontology grouping (5 groups) provides STRONGER per-group MMD signal
  than fine cell_type grouping (10 on H+Mac, 3 on H+Mouse) because each
  group has more cells. iter111's kNN-sp 0.813 is far below iter109 H+Mac's
  0.686 (different datasets) but the trade between species mixing and CT
  purity is well-balanced (sil-ct+LT-acc held).
- LT-acc improvement to 0.993 (best ever on testis H+Mouse) is the
  signature that z-encoded ontology-level information generalizes
  cleanly to fine-cell_type-level inference.

---

### 8. Testis unified — 3-species (CAMEX testis)

**Iteration:** iter115 (coupled prior-clamp + CT-cond MMD on cell_ontology_class) | **Data:** `camex_testis_unified.h5ad`

```bash
python training/run_experiment.py \
  --run-name <name> \
  --config A --max-epochs 100 \
  --adata data/processed/camex_testis_unified.h5ad \
  --adv-weight 100 --lambda-anchor 0.001 --free-bits 1.0 \
  --lambda-mmd 50 --celltype-conditional-mmd --celltype-key cell_ontology_class \
  --prior-clamp-max 0.85 \
  --seed 42 --accelerator mps
```

Recipe: adv=100, lambda_mmd=50 (celltype-conditional on **cell_ontology_class**),
free_bits=1.0, lambda_anchor=0.001, **prior_clamp_max=0.85** (NEW vs iter102),
**celltype_key=cell_ontology_class** (NEW vs iter102's cell_type), 100 epochs.

**Dataset:** 28,854 cells (Human: 15,980, Macaque: 7,613, Mouse: 5,261),
15,732 genes. 3 CTs shared on cell_type; **5 CTs shared on cell_ontology_class
(Sertoli, Spermatocytes, Spermatogonia, somatic, spermatids — 100% cell coverage
across all 3 species)**. The cell_ontology_class column harmonizes labels at the
coarser level matching iter111 H+Mouse's success pattern. Conservation prior
mean=0.890, std=0.079. `batch == species` (1-to-1); donor info lives in
`sample_batch` (5 Human + 2 Mac + 2 Mouse donors) — Chimera uses batch=species
following the canonical recipe.

**scVI baselines:**
- batch=sample_batch (donor): kNN-sp=0.920, sil-sp=-0.021, sil-ct=0.211, LT-acc=0.991.
- batch=species: kNN-sp=0.945, sil-sp=-0.027, sil-ct=0.171, LT-acc=0.901.

**Results (iter115 s42, with prior-clamp 0.85 + CT-cond MMD on cell_ontology_class, 100 ep):**

| Metric | Seed 42 | iter102 baseline | Δ |
|--------|---------|-----------------|---|
| E1-E7 + E9 | **30/33** (E6.1, E6.2, E9.2 fail — all known/architectural) | 28/32 (under old suite) | +2 (incl. tightened E3.2 + new E9 gates) |
| **E8 vs scVI(donor=sample_batch)** | **6/6** ★ | not measured for iter102 | new |
| **E8 vs scVI(species=batch)** | **6/6** ★ | 6/6 | maintained |
| **E3.2 frac(α>0.95)** | **1.32%** ✓ (was 29.06% HARD FAIL) | 29.06% | **CURED 22×** |
| α std | 0.0553 ✓ (above 0.05 floor) | 0.073 | -0.018 (clamp narrowing) |
| α mean | 0.850 (matches clamped prior) | 0.904 | -0.054 (clamp) |
| Δα > 0.02 | **41.16%** | 32.0% | +9.16 (more bidirectional) |
| Δα < -0.02 | **10.56%** | 4.7% | +5.86 (more bidirectional) |
| Δα > 0.05 (E9.2 up) | **18.68%** ✓ PASS | ~0.5% | +18 (37×) |
| Δα < -0.05 (E9.2 down) | **1.57%** ✗ FAIL (target 2%) | ~0% | +1.57 (200×; close to PASS) |
| corr range | [-0.121, +0.392] | n/a | bidirectional |
| BC (bimodality) | 0.361 | n/a | healthy (< 0.85) |
| **kNN-sp (↓)** | **0.730** (vs scVI(donor) 0.920, Δ=-0.190 — Chimera far better) | 0.792 | -0.062 (improved) |
| **sil-ct (↑)** | **0.320** (vs scVI(donor) 0.211, Δ=+0.109) | 0.347 | -0.027 (slight) |
| **kNN-ct (↑)** | 0.825 (vs scVI(donor) 0.910) | n/a | trade vs species mixing |
| **LT-acc (↑)** | **0.992** (vs scVI(donor) 0.991 ties; vs scVI(species) 0.901 BEATS by +0.092) | 0.989 | +0.003 |
| E8.4 worst gap | -0.093 on Other_somatic (vs scVI donor) | -0.116 on Sertoli | both PASS |
| z active dims | **10/10** ✓ | 10/10 | held |
| z_min_var | 5.04 ✓ | 2.51 | +2.53 (more balanced) |
| residual_rms | 3.92 ✓ | n/a | in [0.01, 5] PASS |
| adv_acc | 0.609 (peak 0.634, chance 0.333) | n/a | E6.1/E6.2 fail (z-leakage; long-standing testis pattern) |
| **UMAP aspect** | **0.91** ✓ (round, NOT thin/elongated) | n/a | well below 1.5 limit |

**Per-celltype kNN species purity (iter115 vs scVI(donor)):**

| Celltype | n | chance | iter115 (excess) | scVI(donor) (excess) |
|---|---|---|---|---|
| Elongated_spermatids | 4331 | 0.555 | 0.691 (+0.136) | 0.970 (+0.416) |
| Other_somatic | 4517 | 0.373 | 0.835 (+0.462) | 0.928 (+0.555) |
| Sertoli | 2406 | 0.551 | 0.672 (+0.121) | 0.916 (+0.364) |

Chimera better-mixed than scVI on EVERY shared cell-ontology group.

**Checkpoint (iter115):** `training/runs/testis-unified-iter115-pclamp-coc-100ep-s42/model/model.pt`
**Final state:** `training/runs/testis-unified-iter115-pclamp-coc-100ep-s42/final_state.npz`
**Raw prior (unclamped):** `training/runs/testis-unified-iter115-pclamp-coc-100ep-s42/conservation_prior_raw.npy`
**UMAP plot:** `training/runs/testis-unified-iter115-pclamp-coc-100ep-s42/plots/07_umap.png`
**iter102 checkpoint preserved:** `training/runs/testis-unified-iter102-ctmmd50-200ep-s42/model/model.pt`

**Iteration history:**
- iter91c (adv=80, anch=0.001): 28/32 + 5/6 E8 — E8.6 fail (LT-acc 0.809 vs scVI 0.901)
- **iter102 (CT-cond MMD on cell_type)**: 28/32 + 6/6 E8 (species baseline) — LT-acc 0.989 outperforms scVI 0.901, but E3.2 sat 29.06% HARD FAIL
- iter106 (eval fix): 29/32 + 6/6 E8 — E2.7 scaling bug fixed
- **iter115 (clamp 0.85 + CT-cond MMD on cell_ontology_class, 100ep)**: 30/33 + 6/6 E8 (donor) + 6/6 E8 (species) —
  E3.2 saturation cured (29.06% → 1.32%, **22×**), kNN-sp 0.792 → 0.730, LT-acc
  0.989 → 0.992 (BEATS scVI(species) 0.901 by +0.092), UMAP aspect 0.91 (round).
  E9.2 push<-0.05 reached **1.57%** — 200× iter102's ~0%, just below 2% PASS.
  E6.1/E6.2 z-leakage gates still failing (adv_acc=0.609 vs chance 0.333) — long-standing testis pattern, not new from this iteration.

**Key insights (iter115)**:
- The iter111 H+Mouse coupled recipe (clamp + CT-cond MMD on cell_ontology_class)
  transfers cleanly to 3-species unified. Same testis-tissue-specific override
  works across H+Mac (iter109), H+Mouse (iter111), and unified (iter115).
- 5 cell_ontology_class groups across all 3 species (100% coverage) gives
  stronger per-group MMD signal than iter102's 3 cell_type groups.
- The 100ep LR schedule is more aggressive than 200ep — produces wider corrections
  but slightly less downward-skewed equilibrium. Killed 200ep run (preserved at
  `runs/testis-unified-iter115-pclamp-coc-200ep-s42-killed-ep100/`) reached E9.2
  down 2.43% PASS at ep 100 of a 200ep schedule, vs 1.57% on the 100ep schedule.
- **Mechanistic finding**: 3-species training reduces per-gene S_g_ema noise AND
  surfaces more genes with non-trivial neg-S_g (because pairwise contrasts span
  more divergence). The GIN MLP's discrimination problem is easier on 3-sp than
  2-sp — gradient-signal **DENSITY** is the primary issue, not signal STRUCTURE.
- ★ status: **integration done, E9.2 down architectural ceiling at 2% threshold**.
  Same status as iter109 H+Mac and iter111 H+Mouse — testis triad is now
  consistent.

---

## Divergence-adaptive recipe lookup

| Divergence | λ_mmd | adv_weight | free_bits | CLI overrides |
|------------|-------|------------|-----------|---------------|
| Close (H+Mo) | 50 (default) | 100 (default) | 0.5 (default) | *none* |
| Mid (H+Mouse) | 50 (default) | **200** | **1.0** | `--adv-weight 200 --free-bits 1.0` |
| Very (H+Zeb) | **10** | 100 (default) | 0.5 (default) | `--lambda-mmd 10` |
| 4-species | 50 (default) | **80** | 0.5 (default) | `--adv-weight 80 --max-epochs 150` |

---

## Code changes active in iter81b (vs iter50)

All committed and active by default. No CLI flags needed.

1. **`alpha.detach()` in mu_full** (`_module.py`): L_recon gradient does NOT
   flow to GIN. GIN learns only from FQR + anchor.
2. **Unrectified S_g** (`_module.py`): ReLU clamp removed. Negative S_g pushes
   divergent genes' corrections negative → bidirectional Δα.
3. **`LAMBDA_MMD = 50.0`** (`_constants.py`): up from 10.0. Compensates for
   high-α regime where z carries more species info.
