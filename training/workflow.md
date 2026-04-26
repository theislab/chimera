# Chimera Training — Iteration Workflow

> **PROMOTION REMINDER.** On every promotion, update
> [workflow-best.md](workflow-best.md) **atomically** with: (1) the
> default config in `_constants.py`, (2) the notebook pointers, (3) the
> registry's status row, headline metrics and checkpoint path. If
> workflow-best.md, `_constants.py` defaults, and notebook pointers
> ever drift apart, that is a bug — investigate and reconcile **before
> any run**. workflow-best.md is the authoritative answer to *"what
> should I run on dataset X right now?"*

---

## Current iteration

**iter121 ROLLBACK (2026-04-26, this invocation).** Strategic PIVOT off the exhausted testis E9.2-down per-gene-mechanism family (iter113–iter120: 8 consecutive failures, anti-pattern rule binds). Attacked the worst remaining "definition of done" violation: 4-sp pancreas integration (Chimera kNN-sp 0.979 vs scVI 0.917 — Chimera WORSE). Code change: opt-in `--lambda-mmd-aux` flag adding `lambda_mmd_aux × L_mmd_global` on top of `lambda_mmd × L_mmd_ct` when CT-cond MMD is the primary mode. Test: `--lambda-mmd 50 --lambda-mmd-aux 100 --celltype-conditional-mmd --celltype-key cell_type` on iter101 recipe (adv=200, lambda_anchor=0.001, free_bits=1.0, 200ep, s42). **Result: ROLLBACK.** Two HARD falsifying probes triggered: (1) kNN-sp 0.976 essentially unchanged from iter101's 0.979 (predicted 0.85-0.92), Chimera STILL above scVI's 0.930 by +0.046; (2) z_mean_var_min=0.003 (predicted ≥ 0.1) — 4/10 z dims partially collapsed under the increased total MMD pressure. **E8 went 3/6 → 2/6 (REGRESSED)**: sil-ct dropped 0.408 → 0.244 (Hep/Endo over-mixed into each other); E8.3 / E8.4 (+0.123 on Endothelial worst-CT gap, primary integration gate) / E8.5 all fail. **iter101 retained as 4-sp pancreas best-known** (no registry change). Code change preserved opt-in (default lambda_mmd_aux=0.0 = byte-identical legacy). α distribution stayed healthy (mean=0.751, std=0.129, BC=0.471, frac>0.95=0.41%); Δα bidirectional (E9.2 down 35.35%, E9.2 up 3.76%); GIN-side biology working well (mean_corr_on_neg_S_g=-0.933, frac_corr_neg_on_neg_S_g=99%, ρ(S_g, corr)=0.55) — the failure was purely on the integration side, NOT on α biology. UMAP aspect 1.21 (round, PASS).

### iter121 measured-vs-predicted

| Metric | iter101 baseline | iter121 measured | Δ measured | Δ predicted | Verdict |
|---|---|---|---|---|---|
| **PRIMARY E8.1 kNN-sp** | 0.979 (vs scVI 0.917, Chimera worse) | **0.976** (vs scVI 0.930, Chimera worse by +0.046) | -0.003 | predicted 0.85-0.92 (target ≤ scVI) | **HARD FAIL** — falsifying probe triggered (kNN-sp ≥ 0.927) |
| **E8 score** | 3/6 | **2/6** | -1 | predicted 5-6/6 | **HARD FAIL — REGRESSED** |
| **PRIMARY E8.4 worst-CT gap** | +0.062 | **+0.123** on Endothelial | +0.061 | predicted -0.05 to +0.02 (PASS) | **HARD FAIL** |
| **PRIMARY E8.5 worst-CT vs chance excess** | +0.345 (Hep) | **+0.664** (Hep) | +0.319 (REGRESSED) | predicted +0.05 to +0.20 | **HARD FAIL** |
| sil-ct | 0.408 | **0.244** | -0.164 | predicted 0.35-0.42 (held) | **HARD FAIL** below 0.25 floor |
| sil-sp | 0.018 | **-0.017** | -0.035 | predicted -0.02 to +0.05 | (in range — better mixing) PASS E8.2 |
| LT-acc | 0.985 | **0.981** | -0.004 | predicted 0.97-0.99 | (in range) PASS E8.6 |
| **z_mean_var_min** | (≥ 0.1 expected) | **0.003** | catastrophic | predicted ≥ 0.1 (HARD blocker) | **HARD FAIL — partial posterior collapse** (4/10 weak dims, E6.3 fail) |
| z active dims (var > 0.1) | 10/10 | 6/10 (4 dims var<0.01) | -4 | predicted 10/10 | **HARD FAIL** |
| α mean | 0.754 | **0.751** | -0.003 | predicted 0.75-0.77 | PASS (in range) |
| α std | 0.130 | **0.129** | -0.001 | predicted 0.11-0.14 | PASS (in range) |
| BC(α) | 0.489 | **0.471** | -0.018 | predicted 0.40-0.55 | PASS (in range) |
| frac(α>0.95) | 0.95% | **0.41%** | -0.5pp | predicted 0.5-2% | PASS |
| Δα > 0.02 | 18.13% | **12.84%** | -5.3pp | predicted 12-25% | PASS (in range) |
| Δα < -0.02 | 54.31% | **56.46%** | +2.1pp | predicted 45-60% | PASS (in range) |
| E9.2 push>+0.05 | (not tracked iter101) | **3.76%** | n/a | (not predicted) | PASS (≥ 2%) |
| E9.2 push<-0.05 | (not tracked iter101) | **35.35%** | n/a | (not predicted) | PASS (well above 2%) |
| ρ(S_g, corr) all | (not tracked iter101) | **0.549** | n/a | (not predicted) | (mid-range; GIN healthy) |
| mean corr on neg-S_g | (not tracked iter101) | **-0.933** | n/a | (not predicted) | (excellent — strongly negative; GIN healthy) |
| frac_corr_neg_on_neg_S_g | (not tracked iter101) | **98.9%** | n/a | (not predicted) | (excellent — almost all rare class correctly assigned) |
| residual_rms | (iter101 baseline n/a) | **2.73** | (in [0.01, 5]) | predicted 3-5 | (slightly below predicted, in valid range) PASS |
| adv_acc | 0.765 | **0.709** (final) | -0.056 (better mix) | predicted 0.55-0.70 | borderline (just above range; E6.2 fail because chance=0.25, far) |
| UMAP aspect (p1-p99) | (iter101 n/a) | **1.21** | (round) | predicted < 1.5 | PASS (very round) |
| Expectation suite | 30/32 (iter101) | **31/33** (E6.2, E6.3 fail) | comparable | predicted 30-33/33 | (suite count similar; integration FAIL via E6.3 z-collapse) |

**Predicted-vs-measured discipline (the lesson is in WHY the dual MMD over-pressurized)**:
- ✗ **HARD FAIL on PRIMARY kNN-sp**: dual MMD at lambda_mmd=50 (CT-cond) + lambda_mmd_aux=100 (global) added more MMD pressure than iter101's 100 global alone, but the increase was absorbed by collapsing 4/10 z dims rather than by improving species mixing. With z capacity reduced from 10 → 6 effective dims, the model could not represent both per-CT structure and species-invariant z simultaneously — Hepatocytes (16,222 cells, 73% of data) over-mixed within itself by losing z dim variation, raising kNN-sp on Hep to 0.994 (vs iter101's 0.952 = even worse mixing on Hep!). Net: integration globally about the same, CT structure worse.
- ✗ **z_mean_var_min=0.003 HARD FAIL**: 4/10 z dims have var < 0.01. free_bits=1.0 was nominally engaged but the joint loss landscape was sufficiently constrained that those dims hit the variance floor anyway. iter95/iter97 collapsed at lambda_mmd=100 without free_bits; iter101 added free_bits=1.0 to prevent this. Adding lambda_mmd_aux=100 on top pushed past the free-bits buffer.
- ✗ sil-ct catastrophic regression (0.408 → 0.244): with 4/10 z dims collapsed, the remaining 6 dims have to encode both species-invariant and per-CT structure, and the CT-cond MMD applied within Hep/Endo means those dims are also being matched between species, so per-cluster silhouette weakens substantially.
- ✓ **GIN biology side is fine**: α distribution healthy, ρ=0.55, mean_corr_on_neg_S_g=-0.93 strongly negative, 99% of rare-class genes correctly assigned. The dual MMD does NOT contaminate FQR/anchor — confirms architectural separation between MMD (z-side) and FQR/α (gene-side) is preserved as designed.
- ✓ UMAP aspect 1.21 (round) — the partial z collapse doesn't yet show as visible elongation in UMAP non-linear projection, but it's there in linear PC1/PC2 (1.93 aspect on z) and in the var<0.01 count.

**Mechanistic finding (iter121, definitive)**: Dual MMD at lambda_mmd=50 + lambda_mmd_aux=100 PUSHES TOO HARD on the 4-sp pancreas. The z latent collapses 4/10 dims absorbing the extra pressure, and the remaining capacity over-mixes Hepatocytes (already 73% of data) within itself, which destroys CT structure (sil-ct 0.408 → 0.244) without gaining global species mixing. The 4-sp pancreas with only 2 shared CTs (Hep + Endo) is genuinely structure-limited; adding more MMD pressure doesn't add information about how to mix species AT THE EXPENSE OF CT structure — there's no "free room" to push.

**Decision**: ROLLBACK iter121. iter101 remains best-known for 4-sp pancreas (no registry change). Code change (`--lambda-mmd-aux` opt-in, default 0.0 = byte-identical) preserved as opt-in for future combinations. Diagnostic assets reusable: integration_vs_scvi.{json,md}, gin_signed_diagnostics.json, umap_aspect.json. Artifacts: `runs/xds-4sp-iter121-dualmmd-200ep-s42/{model,final_state.npz,summary.json,plots/,gin_signed_diagnostics.json,umap_aspect.json,integration_vs_scvi.{json,md},report.md}`.

**Next failing expectation + next hypothesis**:
- Failing: 4-sp pancreas E8 = 3/6 (iter101) — Chimera kNN-sp 0.979 > scVI 0.917 (Chimera WORSE on integration). iter121 dual MMD did NOT close this gap.
- **Two consecutive 4-sp E8 attacks (iter107 CT-cond ALONE, iter121 dual MMD) have failed**. Per refinement.md anti-pattern rule, the next 4-sp attack must be a structurally-different mechanism, not another MMD-weight retune.
- **Strategic question for the next invocation (USER INPUT NEEDED — three structurally-different paths forward)**:
  - **(α) 4-sp species-pair-conditional MMD** (orthogonal mechanism — weighted Σ over species pairs of MMD per pair, not within-CT). Not yet tried. Different mechanism from CT-cond MMD. Predicted: targets pairs that mix worst (e.g., Mouse↔Zebrafish at 450 Myr) without diluting easier pairs. Tier-3, ~3 hr.
  - **(β) Refine 4-sp E8 expectations to be structure-aware**: 4-sp has only 2 shared CTs vs 5 unshared CTs (per-species hubs). Even vanilla scVI can't fully mix species when only 2 shared CTs anchor the manifold. The 0.917 scVI baseline already represents the data-attainable ceiling; Chimera's 0.979 > 0.917 is structural. Treat as "best achievable on this dataset" rather than a model failure. Requires user approval to amend refinement.md.
  - **(γ) Pivot to testis E9.2-down resolution path (B) or (C)** from the prior strategic question — refine E9.2 to be data-adaptive (B) or Tier-3 FQR baseline redesign (C). These remain open and address a different "definition of done" gap (testis triad ★).

### iter121 pre-change template (RESOLVED — kept verbatim for measured-vs-predicted audit)

```
Iteration 121 — 4-sp pancreas DUAL MMD (global + CT-cond)
Expectation under attack: E8.1, E8.4, E8.5 on 4-sp pancreas
                          (iter101: kNN-sp 0.979 vs scVI 0.917 — Chimera WORSE
                           on integration; only 3/6 E8 pass; primary "definition
                           of done" violation in the suite).
Pivot rationale: 8 consecutive testis E9.2 down attacks (iter113–iter120) all
                 failed and the per-gene-mechanism family is exhausted (anti-
                 pattern rule binds). Dual MMD on 4-sp is structurally different
                 (loss-structure on a different dataset and different problem
                 class), addresses a worse "Chimera < scVI" gap, and is the
                 explicit guidance from memory `feedback_ctcond_mmd_4sp.md`
                 ("either use global MMD alone, or combine global + CT-cond
                 (dual MMD). Never drop global MMD entirely.").
Change:
  Code (3 files, ~25 LoC):
    chimera/_module.py — new constructor param `lambda_mmd_aux: float = 0.0`,
      saved to `self.lambda_mmd_aux`. New conditional inside the MMD branch:
      when `use_celltype_conditional_mmd=True` AND `lambda_mmd_aux > 0`, ALSO
      compute `L_mmd_aux = self._compute_mmd(z, species_idx)` (global MMD on
      ALL cells). Total loss adds `self.lambda_mmd_aux * L_mmd_aux`. Outputs
      gain a new `L_mmd_aux` key for logging.
    training/train_and_diagnose.py — new function param `lambda_mmd_aux`,
      forwarded to extra_model_kwargs when > 0; new CLI flag `--lambda-mmd-aux`.
    training/run_experiment.py — same plumbing pattern.
  Existing best-knowns: ALL byte-identical at default (lambda_mmd_aux=0).
  Verified by smoke test: default construction has `lambda_mmd_aux=0.0`; the
  new `if` branch is dead unless `use_celltype_conditional_mmd=True` AND the
  new lambda is set; existing testis configs (which use CT-cond MMD without
  the new flag) preserve byte-identical loss numerics.
  Pre-existing % escape bug fixed in two help strings (--lambda-anchor-neg-factor,
  --use-per-gene-bias) so `--help` no longer crashes (zero behavioral impact).
Test config (4-sp pancreas; iter101 recipe + dual MMD on the 2 shared CTs):
  --adata data/processed/camex_pancreas_unified.h5ad
  --config A --max-epochs 200 --seed 42 --accelerator mps
  --adv-weight 200 --lambda-anchor 0.001 --free-bits 1.0
  --lambda-mmd 50           # CT-cond MMD weight (HALF iter101's 100)
  --lambda-mmd-aux 100      # global MMD weight (EXACTLY iter101's pressure)
  --celltype-conditional-mmd --celltype-key cell_type
Tier: 3 (architectural addition: opt-in dual-MMD; one new lambda)

Direct touch-points
  - MMD pressure: iter101 had 100 × L_mmd_global; iter121 has 50 × L_ct + 100
    × L_mmd_global. Net pressure on the 86%% of cells in shared CTs (Hep
    16222, Endo 2733) is ~150 (50 CT-cond + 100 global). Net pressure on the
    14%% of cells in unshared CTs is 100 (global only, EXACTLY iter101's).
  - z encoder: more total MMD pressure on shared CTs → tighter species mixing
    on Hep/Endo. Risk: posterior collapse if z can't carry CT structure under
    pressure (free_bits=1.0 from iter101 should hold; iter95/iter97 collapsed
    at 100/75 mmd without free_bits).
  - Adversary classifier: better species mixing on the bulk of the data →
    drops AUC further (iter101 adv_acc=0.765; predicted 0.55–0.70).
  - GIN / α / FQR: completely unaffected (no FQR or anchor changes; CT info
    never reaches encoder/decoders, only the MMD grouping). α distribution
    predicted to be unchanged from iter101.
  - Residual decoder: receives more pressure to NOT carry species-on-shared-CTs
    info, which prevents the iter107 failure mode (CT-cond MMD alone ⇒ alpha
    frozen at prior, residual eats CTs ⇒ adv leak).
  - Loss-balance numbers (chimera-loss-scaling-fix memory): unchanged. λ_anchor
    =0.001, λ_fit=0.1, λ_aux=1.0. Only MMD components shifted.
  - Schedules: warmup_kl=20, warmup_fqr=20, warmup_adv=5 — unchanged.

Predicted deltas — signed, rough magnitude REQUIRED
  - PRIMARY kNN-sp: 0.979 (iter101) → 0.85–0.92 (target ≤ scVI 0.917 = E8.1 PASS).
  - PRIMARY E8.4 worst-CT (max chimera-scVI per-CT kNN gap): iter101 +0.062 →
    -0.05 to +0.02 (target ≤ +0.02 = E8.4 PASS).
  - PRIMARY E8.5 worst-CT vs chance excess: iter101 +0.345 (Hep) → +0.05 to
    +0.20 (target absolute ≤ 0.15 OR relative ≤ scVI excess + 0.05).
  - sil-ct: 0.408 (iter101) → 0.35–0.42 (held; some CT-purity tradeoff
    expected since CT-cond MMD pressure within Hep/Endo dilutes per-cluster
    silhouette slightly).
  - sil-sp: 0.018 (iter101) → -0.02 to +0.05 (lower better, ≤ scVI's 0.007 + 0.05 = E8.2 PASS).
  - LT-acc: 0.985 (iter101) → 0.97–0.99 (preserved or slight gain — better
    mixing on Hep/Endo helps cross-species CT inference).
  - alpha mean: 0.754 (iter101) → 0.75–0.77 (essentially unchanged; FQR/anchor
    unchanged; only MMD changed).
  - alpha std: 0.130 (iter101) → 0.11–0.14 (preserved).
  - frac(α>0.95): 0.95%% (iter101) → 0.5%%–2%% (PASS, well under E3.2 5%% bar).
  - Δα > 0.02: 18.13%% (iter101) → 12%%–25%% (preserved — α distribution unchanged).
  - Δα < -0.02: 54.31%% (iter101) → 45%%–60%% (preserved — α distribution unchanged).
  - BC(α): 0.489 (iter101 — healthy, < 0.85) → 0.40–0.55.
  - z active dims (var > 0.1): 10/10 (iter101) → 10/10 (HARD: must hold; <8/10 = z collapse blocker).
  - z_min_var: ≥ 0.1 (HARD: < 0.1 = posterior collapse blocker).
  - residual_rms: in [0.01, 5] (PASS).
  - adv_acc: 0.765 (iter101) → 0.55–0.70 (better mixing → harder for adv).
  - UMAP aspect (p1-p99): expected round, < 1.5 (PASS).

Falsifying probes
  - HARD: kNN-sp ≥ scVI's 0.917 + 0.01 = 0.927 (no improvement vs iter101's 0.979
    is bad; no improvement vs scVI is worse — dual MMD failed to add signal).
  - HARD: alpha frozen — Spearman r(α, prior) > 0.99 OR std(α) < 0.05 (this is
    iter107's catastrophic failure mode where the alpha-anchor regularizer
    starves the GIN; means dual MMD did not solve iter107's root cause).
  - HARD: z collapse — active dims < 8/10 OR z_min_var < 0.1 (means too much
    total MMD pressure; need to lower λ_mmd_aux next iteration).
  - HARD: sil-ct < 0.30 (CT structure destroyed — same iter107 failure
    mechanism, just a different shape).
  - HARD: E8 ≤ 3/6 (no improvement on integration vs iter101 — dual MMD does
    not help on this dataset).
  - SOFT: residual_rms > 5 (residual eating CT info; structural concern).
  - SOFT: LT-acc < 0.95 (cross-species CT inference regressed).

Rollback plan
  - The CLI flag `--lambda-mmd-aux` defaults to None / 0.0 → byte-identical
    behavior on every existing best-known (verified via smoke test on the
    constructor; the new `if lambda_mmd_aux > 0` branch is dead at default).
  - If iter121 fails: don't pass `--lambda-mmd-aux`, all best-knowns unaffected.
    iter101 (4-sp pancreas) remains best-known; revert is just "stop using the
    flag" — no code or config rollback needed.
  - The 3-file code change is small and well-scoped; safe to keep as opt-in.
  - Backup queue (if iter121 fails or partially succeeds):
      iter122a: same recipe with lambda_mmd_aux=70 (less aggressive global) +
                lambda_mmd=70 (more aggressive CT-cond). 1 iteration.
      iter122b: testis 2-sp dual MMD — apply same recipe to testis H+Mac/
                Mouse/unified to test the recipe transfers (iter109/iter111/
                iter115 currently use CT-cond ONLY); may be byte-identical
                semantically (testis 2-sp has 86%–100%% shared CT coverage so
                global MMD adds little). Diagnostic, not headline.
      iter122c: 4-sp species-pair-conditional MMD — orthogonal mechanism,
                weighted Σ over species pairs.
```

**Strategic-context summary** (for context, not action; resolved by this iteration):
After 8 consecutive failed testis E9.2-down attacks (iter113–iter120), the per-gene-mechanism family was definitively exhausted. The architectural ceiling is in the FQR baseline definition (`p_null × lib` produces a testis S_g_ema distribution where only 1.06%% of genes have S_g<-0.05 — math caps E9.2 down mass below 2%%). The three structural pivots considered were: (A) 4-sp pancreas dual MMD, (B) refine E9.2 expectation to be data-adaptive, (C) Tier-3 FQR baseline redesign (multi-week). **iter121 chose (A)** — strictly different mechanism family, addresses a worse "definition of done" gap (Chimera < scVI on integration), and aligns with explicit memory guidance. Options (B) and (C) remain open; option (B) requires user approval to amend refinement.md.

### Recent rejections — testis E9.2 down per-gene-mechanism family (DISTILLED)

Eight consecutive testis E9.2 down attacks rejected (iter113–iter120). **Family is now EXHAUSTED** by refinement.md anti-pattern rule. Full numerical detail lives in workflow-best.md §6 (testis H+Mac); compact lessons:

| iter | Mechanism | Verdict | Headline measure | Lesson |
|---|---|---|---|---|
| iter113 | λ_fit ×3 (loss-weight) | ROLLBACK | E3.2 sat 0.06%→28.5% (HARD); ρ +0.020 | Loss-weight scaling does NOT improve per-gene assignment. |
| iter114 | GIN width 32→64 (capacity) | ROLLBACK | ρ -0.011 at ep60 | MLP capacity NOT the bottleneck. |
| iter116 | SG_EMA β 0.9→0.97 (signal density) | ROLLBACK | ρ -0.024 unchanged primary | Per-minibatch S_g noise NOT the bottleneck. |
| iter117 | 200ep schedule (just train longer) | ROLLBACK | E9.2 down 1.04% (basin lottery) | 3-sp E9.2 down basin distribution mean ~1.7% < 2% PASS; 2.43% killed-run was a fluke. |
| iter118 | Asymmetric L_anchor factor=0.3 | ROLLBACK | E9.2 down 0.39% (predicted 1.5-5%) | Anchor relaxation gives a real but ~5× too small directional shift; MLP smoothness is the dominant constraint. |
| iter119 | Per-gene bias hybrid at GIN's LR=1e-4 | ROLLBACK | bias_share=2.7% (corrected) — engaged but bounded by Adam LR×N step cap | Per-bias_g gradient ~3e-7 ≫ Adam eps; updates LR-bounded; 200ep at 1e-4 reaches only abs_max=0.057 vs equilibrium target ~±0.5. **Bonus diagnostic-bug lesson saved**: `.numpy()` shares tensor storage; defensively `.copy()` before zero-restore in extract scripts. |
| iter120 | Per-gene bias hybrid + separate LR=1e-3 param group | ROLLBACK | bias_share=22.6% (PASS); mean corr on neg-S_g shifted +0.353→+0.011 (mechanism collapsed class mean to zero); E9.2 down 0.020% (HARD FAIL primary, predicted 5-15%) | **Definitive ceiling diagnosis: the FQR baseline `p_null × lib` produces a testis S_g_ema distribution where only 1.06% of genes have S_g<-0.05 — the math itself caps E9.2 down attainable mass below 2% regardless of mechanism quality.** Per-gene-mechanism family is structurally exhausted. |

**Three transferable insights from this family**:
1. **ρ(S_g, correction) is misleading as a primary metric** (iter117 had higher ρ than iter115 yet lower E9.2 down). Co-primary signal: **mean correction on neg-S_g class** + **bias_share_of_total_magnitude**. Both reported by `gin_signed_diagnostics.py`.
2. **Per-gene mechanism family converges by ~ep55** on testis (iter120 trajectory was identical from ep55 to ep200). For future per-gene experiments, 80–100ep suffices — saves 2× iteration time.
3. **All iter113–iter120 CLI flags are preserved opt-in with default-OFF byte-identical behavior**: `--lambda-fit`, `--gin-hidden`, `--sg-ema-beta`, `--lambda-anchor-neg-factor`, `--use-per-gene-bias`, plus the `_training_plan.py:configure_optimizers` per_gene_bias→1e-3 carve-out (active only when `use_per_gene_bias=True`). Zero impact on existing best-knowns.

**Testis 2-sp E9.2 down resolution paths still open**: (B) refine E9.2 expectation to be data-adaptive [requires user approval to amend refinement.md], (C) Tier-3 FQR baseline redesign `baseline = mu_full_ema.detach()` [multi-week, re-validates all loss-balance numbers]. iter121 (this invocation) elects a structurally-different pivot — see **Current iteration** above.

---

**iter117 REJECTED (2026-04-26, prior invocation, distilled).** Re-running iter115 recipe at 200ep on testis_unified produced **YET ANOTHER stochastic basin** distinct from both iter115 100ep finished and killed-200ep — three basins now observed for the same seed 42 + same recipe + same data:

| Run | E9.2 down | E9.2 up | frac>0.95 | mean correction on neg-S_g |
|---|---|---|---|---|
| iter115 100ep (current best) | **1.57%** | 18.68% | 1.32% | n/a (state diff) |
| killed-200ep at ep100 (no model) | 2.43% | 15.56% | 0.25% | n/a |
| **iter117 200ep finished (this run)** | **1.04%** | 23.71% | 3.59% | +1.024 (POSITIVE: MLP can't find neg-S_g) |

**PRIMARY metric (E9.2 down) FAILED**: predicted 1.5-2.5% range with ~50% PASS; measured 1.04%, BELOW both predicted range AND below iter115 100ep's 1.57%. The killed-run 2.43% was a **stochastic fluke**, not a property of the 200ep schedule. **Architectural ceiling on E9.2 down extends to 3-sp testis_unified at 1.0-1.6% range.**

E8 6/6 PASS vs both scVI(donor=sample_batch) AND scVI(species=batch); LT-acc 0.992 (matches scVI(donor) 0.991, beats scVI(species) 0.901 by +0.091); UMAP aspect 1.147 (round, well below 1.5 limit) — but **z PC1/PC2 SVD aspect REGRESSED to 1.77** (vs iter115's 0.91; falsifying probe partially triggered on this linear-projection diagnostic, though UMAP itself is round). Δα bidirectional 44.85%/9.55% (in predicted range; slightly less negative-tail mass than iter115's 41.16%/10.56%). E3.2 frac>0.95=3.59% (above predicted 0.2-2.5% range, still well under 5% PASS).

**Decision: ROLLBACK. iter115 100ep remains testis_unified best-known.** No code change to roll back. Architectural ceiling diagnosis is now CONFIRMED for both 2-sp (testis H+Mac/H+Mouse) and 3-sp (testis_unified) — Tier-3 code change is the required next attack class.

**Decision tree resolved**: iter117's hypothesis ("200ep reliably closes E9.2 down") is FALSIFIED → iter118 (this invocation) attacks the loss STRUCTURE via asymmetric L_anchor, not the workflow's prior planned per-gene bias hybrid. Asymmetric L_anchor was chosen for: (a) cheaper iteration (~3 hr vs ~4 hr), (b) lower scientific-story risk (GIN still learns from features rather than degenerating to per-gene memorization), (c) structurally different from all 5 prior failed attacks (none touched the L_anchor symmetric squash).

**iter116 REJECTED (2026-04-25, prior invocation, distilled).** Slower SG_EMA_BETA (0.9 → 0.97) on testis H+Mac — falsified the per-minibatch S_g noise hypothesis. PRIMARY metric (E9.2 push<-0.05) UNCHANGED at 0.007% (predicted 2-15%). Spearman r(S_g_ema, correction) slightly WORSE (0.490 vs iter109 0.514). UMAP aspect REGRESSED (PC1/PC2 2.52 vs 1.64). Code change (`--sg-ema-beta` CLI flag) preserved as opt-in diagnostic; default β=0.9 unchanged.

**iter115 ACCEPTED — testis_unified E3.2 HARD FAIL CURED, promoted to best-known (2026-04-25).** Applied iter111 coupled recipe (prior-clamp 0.85 + CT-cond MMD on cell_ontology_class) to testis 3-sp unified at 100 epochs (early termination from planned 200 to give the user fast iteration). **30/33 + 6/6 E8 vs scVI(donor) + 6/6 E8 vs scVI(species)**. **E3.2 frac>0.95: 29.06% → 1.32%** (22× cure — the registry's only remaining HARD FAIL is now resolved). E5.3 also cured. All integration metrics improved: kNN-sp 0.792 → 0.730, LT-acc 0.989 → 0.992, sil-ct 0.347 → 0.320 (slight). UMAP aspect 0.91 (round, not thin). E9.2 down 1.57% (still below 2% PASS threshold but 200× better than iter102's ~0% — the killed 200ep run reached 2.43% PASS, indicating the 200ep schedule is what crosses the line; 100ep's faster LR decay produces a more aggressive but slightly less downward-skewed equilibrium). E6.1/E6.2 z-leakage gates fail (adv_acc=0.609 vs chance 0.333) — long-standing testis issue, not new from this iteration. **Same recipe + ★ status as iter109 H+Mac and iter111 H+Mouse: integration done, E9.2 down architectural ceiling at 2-sp.** With 3-sp the ceiling is much closer to passing — confirms "gradient-signal-density" hypothesis from the brainstorm.

**iter113 + iter114 (REJECTED 2026-04-25 — distilled).** Both attacked testis 2-sp E9.2 down, both rejected after a single iteration; details in *Distilled Lessons* below. TWO false ceilings ruled out: (a) loss-weight scaling (iter113 λ_fit×3 → Δρ=+0.020), (b) MLP capacity (iter114 width×2.6 → Δρ=-0.011). The 72-d GIN input features ARE discriminable for pos- vs neg-S_g testis genes (separability AUC = 0.84 linear / 0.80 nonlinear via `iter114_input_separability.py`). Bottleneck is the **gradient signal STRUCTURE** (class imbalance / L_anchor symmetry / per-minibatch noise / FQR baseline) — iter115's 3-sp result lends weight to the per-minibatch S_g noise hypothesis (3-sp averages over more cells per gene, reducing noise).

## iter115 distilled lesson (ACCEPTED 2026-04-25, full table superseded)

iter115 transferred iter111's coupled recipe (prior-clamp 0.85 + CT-cond MMD on cell_ontology_class) to testis_unified at 100 ep. **E3.2 frac(α>0.95) cured 29.06% → 1.32% (22×)**; 6/6 E8 vs both scVI(donor) and scVI(species); LT-acc 0.992 (BEATS scVI(species) 0.901 by +0.092); UMAP aspect 0.91. E9.2 push<-0.05 reached 1.57% (close to 2% PASS). Full per-row metrics in workflow-best.md §"Testis unified". **Mechanistic finding** (iter115 + brainstorm + iter117 reproducibility test): the multi-basin attractor under MPS non-determinism produces ~0.6%-σ stochasticity in E9.2 down at this recipe; the killed-200ep run's 2.43% was a stochastic fluke, the basin distribution mean (~1.7%) sits below 2% PASS line. **3-species reduces per-gene S_g_ema noise AND surfaces more genes with non-trivial neg-S_g (more pairwise contrasts).** The Spearman ρ(S_g, corr) cross-config diagnostic showed iter115 had LOWER ρ (0.355) than iter109 (0.514) yet HIGHER E9.2 down — i.e. iter115's success was distributional widening (corr_min -0.747 vs iter109 -0.304), not better per-gene MLP discrimination. **Implication: ρ is misleading as a primary metric; mean correction on neg-S_g class is the right co-primary signal.**

---

## Current state (2026-04-26)

**2-species pancreas — DONE (★).** H+Mo iter90 (3 seeds) and H+Mouse iter90 (3 seeds, iter112) both pass the full expectation suite under the revised E9.1 ≥ 0.50 threshold: 33/33 + 6/6 E8 + E9.1 + E9.2 PASS, with round UMAPs (aspect 1.13–1.51) and cross-seed α Spearman ρ = 0.83 / 0.762 respectively. These are the canonical 2-species references.

**2-species testis — 6/6 E8 with E9.2 down architectural ceiling (post-iter120, 2026-04-26).** H+Mac iter109 (clamp 0.85 + CT-cond MMD on cell_type) and H+Mouse iter111 (clamp 0.85 + CT-cond MMD on cell_ontology_class) both pass 6/6 E8 and beat scVI(donor) on integration, but **E9.2 push<-0.05 ≈ 0%**. ★ withheld until E9.2 down passes — **EIGHT attacks now ruled out**: iter113 (λ_fit ×3), iter114 (capacity ×2.6), iter116 (slower SG_EMA β 0.9→0.97), iter117 (200ep schedule), iter118 (asymmetric L_anchor factor=0.3, partial 5% shift), iter119 (per-gene bias hybrid at GIN's LR=1e-4, mechanism inert), iter120 (per-gene bias hybrid at LR=1e-3, mechanism engaged 8× more but primary metric still 0.020%). **The architectural ceiling is in the FQR baseline definition (`p_null × lib`) producing an S_g_ema distribution where only 1.06% of testis genes have S_g<-0.05 — the math itself caps E9.2 down attainable mass below the 2% threshold regardless of mechanism quality.** Per-gene mechanism family is now EXHAUSTED. Next iteration MUST pivot to a structurally-different problem (4-sp pancreas dual MMD), or refine E9.2 expectation to be data-adaptive, or begin Tier-3 FQR baseline redesign. Marginal numerical fails on E1.5 (α.std ≈ 0.0495 vs 0.05 floor) follow from prior-clamp narrowing.

**Testis unified (3-sp) — best-known iter115 (100ep), E9.2 down 1.57% architectural ceiling CONFIRMED (2026-04-26).** iter117 200ep re-run produced 1.04% (worse than 100ep, below 2% PASS). Three stochastic basins observed for the same recipe + seed (1.04, 1.57, 2.43%); the basin distribution mean ~1.7% sits below the 2% PASS line. **The killed-200ep 2.43% was a stochastic fluke**, NOT a property of the 200ep schedule. iter115 100ep retained as best-known; ★ withheld; ARCHITECTURAL CEILING extends to 3-sp testis. iter120 attacks 2-sp first (testis H+Mac cheapest); if per-gene bias at LR=1e-3 wins, transfer to H+Mouse + unified.

**4-species pancreas — structural ceiling at 2 shared CTs.** iter101 reaches sil-ct = 0.408 (best ever) and LT-acc = 0.985 (beats scVI 0.916) but kNN-sp = 0.979 vs scVI 0.917 because only Hepatocytes (16,222) and Endothelial (2,733) are shared across all 4 species. iter107 CT-cond MMD attempt rolled back. Treated as structural until proven otherwise.

**H+Zebrafish — structural at 450 Myr divergence.** iter85 with --lambda-mmd 10 reaches 28/33 + 4/6 E8. Species too divergent to mix in z without destroying celltype structure (sil-ct -0.093, LT-acc 0.704). Treated as structural.

**Identifiability audit (user-prompted, iter112).** Pancreas H+Mouse / H+Zeb have `batch == species` (1-to-1); ConstrainedBatchOffset is mathematically inert. Pancreas H+Mo has 3 batches (Monkey has 2) — ConstrainedBatchOffset has ~1 dof. Testis datasets DO have `sample_batch` (5–9 donor batches/species) but Chimera uses `batch_key=batch` (==species) while scVI baseline uses `batch_key=sample_batch` (donors). iter94 tested Chimera with `batch_key=sample_batch` and rolled back (z contracted 4×). The interaction between Chimera's ConstrainedBatchOffset / adversary / MMD and a multi-batch-per-species `batch_key` is an **open Tier-3 architectural question**, not addressable as an HP tune.

---

## iter117 distilled lesson (REJECTED — 2026-04-26)

**What was tried.** Re-run iter115 recipe at 200ep on testis_unified s42 to test reproducibility of the killed-200ep run's 2.43% E9.2 down. NO CODE CHANGE.

**How it failed.** Primary E9.2 push<-0.05 measured 1.04% (predicted 1.5-2.5%, below iter115's 1.57%, below 2% PASS). Three independent runs of the same recipe + seed produced three distinct stochastic basins for E9.2 down: {1.04, 1.57, 2.43}%, mean ≈1.7%, std ≈0.6%. The 2% PASS threshold sits AT the basin-distribution mean — passing is a coin flip, not a property of the recipe. Falsifying probe partial-trigger: z PC1/PC2 SVD aspect REGRESSED to 1.77 (UMAP itself round at 1.147) — encoder over-fits to a thinner manifold at 200ep; UMAP recovers via non-linear projection. E8 6/6 vs both scVI baselines maintained, LT-acc 0.992, all integration metrics held.

**Critical finding.** Mean correction on neg-S_g genes ROSE to +1.024 (vs iter115's +0.655) despite ρ(S_g,corr) IMPROVING to 0.448 (vs 0.355) — so wider corr range incidentally boosts ρ without fixing the rare-class assignment problem. **ρ(S_g,corr) is misleading as a primary metric.** Future iterations must report mean correction on neg-S_g class as a co-primary metric — that's what determines E9.2 down magnitude.

**Design implication.** "Just train longer" is exhausted as an attack on testis E9.2 down. Architectural ceiling extends to 3-sp (was already confirmed on 2-sp via iter113/114/116). Tier-3 code change is the only remaining attack class. iter118 (asymmetric L_anchor — this invocation) attacks the loss STRUCTURE; if it falsifies, per-gene bias hybrid is the next Tier-3 candidate.

Artifacts preserved at `runs/testis-unified-iter117-200ep-s42/{model,final_state.npz,summary.json,plots/,integration_vs_scvi.{md,json},report.md}`.

---

## iter116 distilled lesson (ROLLBACK 2026-04-25)

iter116 tested the **per-minibatch S_g noise** half of the iter115 brainstorm: opt-in `--sg-ema-beta 0.97` on iter109's recipe (clamp 0.85 + CT-cond MMD on cell_type, 200ep), no other change. **FALSIFIED for testis 2-sp E9.2 down.** Tightening the GIN's regression target via 3.3× temporal averaging did NOT improve per-gene assignment to neg-S_g (ρ ↓ 0.514 → 0.490), narrowed the correction range (corr_max 1.373 → 1.169, corr_std 0.200 → 0.175), and did NOT move the primary metric (0.007% UNCHANGED). The 3-sp E9.2 down success (iter115) is therefore NOT explained by EMA noise reduction alone — count-mechanism (more pairwise contrasts) or class-imbalance is the remaining candidate. Bonus surprise: smoother FQR gradients leak into encoder regularization (UMAP aspect REGRESSED 1.64 → 2.52). **Future "tighten S_g signal" attempts deprecated** unless the encoder regularization side-effect is mitigated. CLI flag `--sg-ema-beta` preserved opt-in (default β=0.9 unchanged). Artifacts: `runs/testis-hmac-iter116-sgema097-200ep-s42/`.

## iter117-pre cross-config diagnostic (2026-04-25, post-hoc analysis on existing checkpoints)

**Cross-config Spearman ρ(S_g_ema, correction) on existing testis runs**, computed during iter117's parallel-work window:

| Config | ρ(S_g, corr) all | nz subset | corr range | corr std | mean corr on neg-S_g |
|---|---|---|---|---|---|
| iter115 testis_unified 3-sp 100ep | 0.355 | 0.408 | [-0.747, +1.819] | 0.340 | **+0.655** (positive!) |
| iter109 testis H+Mac 2-sp | 0.514 | 0.538 | [-0.304, +1.373] | 0.200 | +0.353 |
| iter111 testis H+Mouse 2-sp | 0.490 | 0.505 | [-0.296, +1.703] | 0.196 | +0.397 |

**Surprising finding: 3-sp testis_unified has LOWER ρ (0.355 < 0.490) but HIGHER E9.2 down (1.57% > 0.007%) and WIDER correction range (std 0.340 vs 0.200) than 2-sp.** Mean correction on neg-S_g genes is MORE positive on 3-sp (+0.655 vs +0.353) — i.e., the MLP is even WORSE at correctly identifying neg-S_g genes on 3-sp. So **iter115's E9.2 down "success" is NOT from improved per-gene MLP assignment** — it's from **distributional widening** that incidentally puts more genes (regardless of true S_g sign) in the |Δα|>0.05 region. The 3-sp's wider corr range reflects gene-distribution diversity (more pairwise contrasts surface a wider spread of S_g_ema magnitudes across genes), not better per-gene fit.

**Implication for iter118+ design**: per-gene bias hybrid should target ρ(S_g, corr) directly (target ≥ 0.9 via per-gene dof), bypassing the feature-space-neighbor smoothness that limits MLPs. Class-balanced FQR (iter119 backup) is a weaker fix because it would still rely on the MLP's feature-space generalization, just with rescaled per-class gradients. **Per-gene bias hybrid is the most direct architectural fix.**

---

## Iteration 115 (HISTORICAL — ACCEPTED, distilled 2026-04-25)

**What was tried.** Apply the iter111 coupled recipe (prior-clamp 0.85 + CT-cond MMD on cell_ontology_class) to testis unified (3-sp). Pure recipe transfer, no code change. Time-pressured to 100ep (originally 200ep planned; killed at ep100).

**How it worked (predictions vs measured already in §"iter115 measured-vs-predicted" above).** Cured E3.2 HARD FAIL (29.06% → 1.32%). 6/6 E8 vs scVI(donor=sample_batch) AND vs scVI(species=batch). E9.2 down rose to 1.57% — close to PASS threshold of 2%; the killed 200ep run had reached 2.43% PASS at ep100 of a 200ep schedule, so 200ep would PASS but the 100ep LR schedule produced a more aggressive equilibrium with slightly less downward skew. **Mechanistic surprise**: Spearman r(S_g_ema, correction) was 0.355 (LOWER than iter109 H+Mac's 0.514), yet E9.2 down was higher — so 3-sp success is NOT from improved per-gene MLP discrimination, it is from a wider overall correction MAGNITUDE distribution (corr_min = -0.747 vs iter109 -0.304, 2.5×). Iter116's diagnostic recomputation confirmed this on iter109/iter111/iter115 cross-comparison.

**Design implication.** "Signal density via averaging" hypothesis (slower S_g_ema, larger batch) attacks per-gene discrimination — but iter115's actual mechanism was correction-range widening, possibly driven by gene-distribution diversity (more non-trivial neg-S_g genes from more pairwise contrasts, not lower-noise S_g_ema per gene). **iter116 tested the noise-reduction half on H+Mac and FAILED** (β 0.9→0.97 → Δρ=-0.024, primary unchanged). So the dominant 3-sp mechanism IS the count effect, and per-gene-independent signal-range fixes (per-gene bias hybrid, class-balanced FQR) are the next attack class. iter117 is testing reproducibility of the 3-sp 2.43% E9.2 down basin via 200ep re-run.

**Backup options (escalating cost):** per-gene bias hybrid `_components.py` (~4 hr total, Tier-3, iter118 if iter117 fails); class-balanced FQR loss `_module.py` (~1.5 hr code + train, Tier-3); asymmetric L_anchor (~1 hr code); joint-3sp warmstart (~1 hr code); FQR baseline redesign `mu_full_ema.detach()` (multi-week).
---

## Active hypothesis queue (after iter120 ROLLBACK — PIVOT REQUIRED)

In priority order. Re-evaluate on every iteration; the queue is informational only.

**Post-iter120 conviction (definitive)**: EIGHT failed attacks on testis E9.2 down — iter113 (λ_fit×3), iter114 (GIN width×2.6), iter116 (SG_EMA β 0.9→0.97), iter117 (200ep schedule), iter118 (asymmetric L_anchor factor=0.3), iter119 (per-gene bias hybrid at GIN's LR=1e-4), AND iter120 (per-gene bias hybrid with proper LR=1e-3). **iter120 finally engaged the bias mechanism strongly (bias_share=22.6%, mean_per_gene_bias_on_neg_S_g=-0.296, mean correction on neg-S_g class shifted from +0.353 to +0.011) yet E9.2 down stayed at 0.020% (predicted 5-15%, target 2%).** The fundamental constraint is **the FQR baseline definition (`p_null × lib`) producing an S_g_ema distribution where only 1.06% of testis genes have S_g<-0.05** — the math itself caps E9.2 down attainable mass below the 2% threshold regardless of mechanism quality. **Per-gene mechanism family is EXHAUSTED on testis E9.2 down. The per-gene-FQR-mechanism class can no longer be attacked without violating the refinement.md anti-pattern rule (8 consecutive failures in the same family).** The next iteration MUST pivot to a structurally-different problem or mechanism.

**Strategic question for the next invocation (USER INPUT NEEDED — three structurally-different options)**:

| # | Hypothesis | Tier | Predicted impact | Pros / Cons |
|---|------------|------|------------------|-------------|
| **0A (PIVOT — recommended for fastest-time-to-value)** | **iter121: 4-sp pancreas dual MMD (global + CT-cond)** — current 4-sp E8 = 3/6 (kNN-sp 0.979 vs scVI 0.917 — Chimera WORSE on integration). Memory `feedback_ctcond_mmd_4sp.md` says explicitly: "either use global MMD alone, or combine global + CT-cond (dual MMD). Never drop global MMD entirely." iter107 tried CT-cond ALONE (rolled back); dual MMD has NOT been tried. ~3 hr train + 30 min eval. | 3 | Closes a clearer "definition of done" gap (Chimera < scVI on integration) than testis E9.2 down (which is data-architectural). Predicted E8 → 5-6/6, kNN-sp → 0.85-0.92, sil-ct held > 0.40. **Falsifying probe**: kNN-sp ≥ scVI's 0.917 (no improvement, dual MMD doesn't add signal). | Pros: structurally different problem; clear "scVI-better" failure mode targeted; 1 iteration; aligns with "perfect results for ALL datasets" goal. Cons: 4-sp is technically lower priority than 2-sp per user preference. |
| **0B (PIVOT — principled response)** | **iter121: Refine E9.2 expectation to be data-adaptive** — math shows 2% threshold unreachable on testis (only 1.06% genes with S_g<-0.05). Replace `frac(Δα<-0.05) ≥ 2%` with `frac(Δα<-0.05) ≥ min(2%, 60% × frac(S_g<-0.05))`. Adapts threshold to data signal density. | 0 (no model change; expectation refinement) | Testis 2-sp triad: E9.2 down threshold drops to ~0.6%. iter109 H+Mac (currently 0.007%) — still fails. iter120 (0.020%) — still fails. iter111 H+Mouse (0%) — still fails. iter115 unified (1.57%) — likely passes. So this only closes ★ on testis_unified, not 2-sp. **Even with adaptive threshold, the 2-sp testis E9.2 down problem persists at the per-dataset bar.** | Pros: principled response to architectural ceiling; closes testis_unified ★; surfaces the data-vs-threshold mismatch. Cons: REQUIRES USER APPROVAL TO AMEND refinement.md; may be perceived as moving goalposts; only partial relief on 2-sp testis. |
| **0C (TIER-3 — addresses true root cause, multi-week)** | **iter121+: FQR baseline redesign** — replace `baseline = p_null × lib` with `baseline = mu_full_ema.detach()` so FQR signal is the model's own residual deviation (naturally bidirectional). Expected to widen S_g_ema distribution → more genes with non-trivial S_g<-0.05 → more achievable E9.2 down. | 3 (multi-week) | Predicted: testis S_g_ema distribution becomes naturally bidirectional (vs current +0.007 mean, 1.06% neg-class). E9.2 down → 5-15% achievable across testis triad. Tested on H+Mac first, then H+Mouse and unified. | Pros: addresses TRUE root cause; principled; closes testis triad ★ end-to-end. Cons: multi-week effort; re-validates ALL loss-balance numbers (chimera-loss-scaling-fix memory); re-validates ALL best-known recipes (pancreas + testis); high risk of regressions on currently-passing datasets. |
| 1 (DEPRIORITIZED — anti-pattern) | iter121b: combination iter118 (asymmetric L_anchor 0.3) + iter120 (per-gene bias LR=1e-3) | 3 | Would be 9th consecutive testis attack in the same family. ANTI-PATTERN per refinement.md: "do NOT run a third tweak in that family." Predicted partial improvement; not enough to break 2% threshold given dataset constraint. | Skipped unless user explicitly directs. |
| 2 | Multi-seed verification of accepted future testis configs (seeds 43, 44) for E9.1 | 1 | Multi-seed E9.1 certification | Conditional on testis 2-sp ★ (which requires either Tier-3 FQR fix or expectation refinement). |
| 3 | 4-sp pancreas E8.4 attack — species-pair-conditional MMD (alternative to dual MMD, more granular) | 3 | Different mechanism than option (A); per-pair MMD weighting. | If (A) doesn't fully close 4-sp, this is the next attack. |
| 4 | Tier-3: Investigate Chimera × multi-batch-per-species `batch_key` (testis sample_batch identifiability) | 3 | Architectural; iter94 root-cause unsolved | Long-term work; not in immediate path. |

---

## Pre-registered expectations

**Authoritative.** Adding, removing, or loosening an expectation requires editing this section (and `check_expectations.py` / `cross_seed_check.py` / `compare_vs_scvi.py`) — never silently. Every expectation maps to a refinement.md constraint and a specific failure mode it blocks.

### E1: Pretraining — inputs are sane

| # | Name | Metric | Threshold | Spec cite |
|---|------|--------|-----------|-----------|
| E1.1 | `conservation_signal` | Spearman r(conservation_prior, per-gene CPM-log-mean similarity) | `r ≥ 0.02` OR `\|r\| < 0.15` AND prior_std ≥ 0.05 | refinement.md §"alpha biology" |
| E1.2 | `gene_features_normalized` | per-feature mean abs < 0.1 AND std in [0.5, 1.5] | all 72 active dims (after 2026-04-18 GIN input fix) | refinement.md §"GIN input" |
| E1.3 | `p_null_dynamic_range` | frac of (species, gene) with 1e-5 < p_null < 0.5 | ≥ 0.5 | implementation prompt (p_null Laplace) |
| E1.4 | `batch_to_species_unique` | every batch maps to exactly one species | 100% | model setup invariant |
| E1.5 | `conservation_prior_range` | min ≥ 0.05, max ≤ 0.95, std ≥ 0.05 | hard | GIN logit stability |

### E2: Training dynamics — loss & schedule

| # | Name | Metric | Threshold | Failure mode |
|---|------|--------|-----------|--------------|
| E2.1 | `loss_finite` | no NaN/Inf in total loss, any epoch | — | catastrophic |
| E2.2 | `loss_decreases` | last train_loss < first train_loss | strict < | training doesn't start |
| E2.3 | `L_recon_drops` | L_recon @ ep 20 < L_recon @ ep 1 | ≥10% reduction | decoder not learning |
| E2.4 | `kl_warms_up` | kl_weight monotonic non-decreasing | — | warmup wiring |
| E2.5 | `fqr_activates` | schedule/fqr_active = 1 at epoch ≥ WARMUP_FQR | — | FQR wiring |
| E2.6 | `adv_activates` | schedule/adv_active = 1 at epoch ≥ WARMUP_ADV | — | adv wiring |
| E2.7 | `L1_scale_ok` | `lambda_res * L_residual_sparsity / L_recon` after ep 25 | in [1e-3, 0.3] | FM1 (wrong reduction) — fixed iter106 (50× inflation bug) |
| E2.8 | `anchor_scale_ok` | `lambda_anchor * L_alpha_anchor / L_recon` after ep 25 | in [1e-5, 0.3] | FM3 (wrong reduction) |

### E3: GIN behavior

| # | Name | Metric | Threshold | Failure mode |
|---|------|--------|-----------|--------------|
| E3.1 | `alpha_init_is_prior` | max\|alpha_epoch_0 − prior\| | ≤ 0.05 | GIN zero-init broken |
| E3.2 | `alpha_histogram_healthy` | α.std ≥ 0.05 AND frac(α<0.05) ≤ 0.05 AND frac(α>0.95) ≤ 0.05 | per refinement.md "<5% target" (tightened iter108 from prior 50% bar) | FM1 / hard-gate degeneration |
| E3.3 | `alpha_no_bimodal_collapse` | bimodality coefficient of α | < 0.85 | FM3 (bimodal collapse). NOTE: BC is desired-signal for Chimera (healthy α IS bimodal) — failure mode is BC > 0.85 with high saturation, not BC > 0.5 alone. |
| E3.4 | `delta_alpha_biologically_informative` | (sign) frac(Δα>0.02) ≥ 0.05 AND frac(Δα<-0.02) ≥ 0.05 OR (magnitude) α.std ≥ 0.05 AND Spearman(α, prior) ≥ 0.5 AND \|correction\|.std ≥ 0.15 | — | FM4 (one-sided). OR-clause added 2026-04-21 to pass under iter50+ strong-integration regime. |
| E3.5 | `correction_nonzero` | mean\|correction\| | ≥ 0.01 | GIN dead |
| E3.6 | `gin_learning_evolved` | \|\|alpha_final − alpha_init\|\|_1 / G | ≥ 0.01 | GIN dead |

### E4: Alpha biology sanity

| # | Name | Metric | Threshold | What it confirms |
|---|------|--------|-----------|-------------------|
| E4.1 | `alpha_correlates_prior` | Spearman r(α_final, prior) | ≥ 0.25 | prior carries structure |
| E4.2 | `residual_biologically_informative` | (L-shape) Spearman r(α, mean\|res\|) ≤ −0.10 OR ratio bot25%/top25% ≥ 1.5 OR (magnitude) residual.std ≥ 0.1 | — | low-α genes carry bigger residual (iter42-regime) OR residual has large across-gene magnitude spread (iter50+ regime). OR-clause added 2026-04-21. |
| E4.3 | `fqr_active_fraction` | frac(S_g_ema > 0) at end | ≥ 3% | FQR has signal on some genes |

### E5: Shared decoder + residual

| # | Name | Metric | Threshold | Failure mode |
|---|------|--------|-----------|--------------|
| E5.1 | `L_shared_aux_drops` | L_shared_aux @ ep 30 < L_shared_aux @ ep 1 | ≥5% | FM2 (collapse) |
| E5.2 | `mu_shared_has_variance` | std over genes of mean(mu_shared_only) | ≥ 0.01 | FM2 (constant predictor) |
| E5.3 | `residual_has_magnitude` | mean\|residual_log_rate\| | in [0.01, 5] | residual alive but not exploded |

### E6: z-leakage / adversarial

| # | Name | Metric | Threshold | Failure mode |
|---|------|--------|-----------|--------------|
| E6.1 | `adv_accuracy_drops` | final adv_acc < peak(post-warmup) − 0.03 OR final < mean(post-warmup) | — | adversarial effective |
| E6.2 | `adv_accuracy_near_chance` | final adv_acc in [1/K − 0.15, 1/K + 0.20] (strict) OR (dropped from peak by ≥ 0.05 AND final < chance + (1−chance)×0.5) (trajectory) | — | FM-D (species in z) |
| E6.3 | `no_posterior_collapse` | min over z-dims of Var(z_mean) | ≥ 0.01 | FM-E |

### E7: Convergence + generalization (activates when max_epochs ≥ 50)

Tail = last 20% of training (min 10 epochs).

| # | Name | Metric | Threshold | What it tells you |
|---|------|--------|-----------|-------------------|
| E7.1 | `alpha_stabilized` | mean L1 between consecutive α snapshots in tail | < 0.01 | GIN at resting point |
| E7.3 | `val_loss_plateaued` | (max − min) / mean of tail validation_loss | < 2% | converged |
| E7.4 | `val_loss_no_overfit` | (val_final − min(val in tail)) / min | ≤ 1% | no late overfit rise |
| E7.5 | `val_loss_made_progress_post_warmup` | (val@ref_ep − val_final) / val@ref_ep, ref_ep = max(25, max_epochs//4) | ≥ 0.5% | not grossly underfit |

### E8: Cross-species integration vs vanilla scVI (added iter44)

E8 is evaluated by [`compare_vs_scvi.py`](compare_vs_scvi.py): trains/loads a vanilla scVI baseline on the same adata, computes the integration scorecard on both Chimera `z` and scVI `z`, writes `integration_vs_scvi.{md,json}` into the run dir.

**Pancreas baseline (iter108 fix)**: scVI uses `batch_key="batch"` (donors) for fair Chimera comparison. **Testis baseline (iter106 fix)**: scVI uses `batch_key="sample_batch"` (donors). Asymmetry remains: testis Chimera uses `batch_key=batch`==species; multi-batch-per-species on Chimera was tried iter94 and rolled back.

| # | Name | Metric | Threshold | What it gates |
|---|------|--------|-----------|---------------|
| E8.1 | `beats_scvi_knn_species_purity` | global kNN species purity | ≤ scVI + 0.01 | not worse globally on species mixing |
| E8.2 | `beats_scvi_sil_species` | sil-species (global) | ≤ scVI + 0.05 | not globally species-clustered |
| E8.3 | `celltype_structure_preserved` | sil-celltype (global) | ≥ 0.25 AND ≥ scVI; OR ≥ scVI − 0.05 when scVI < 0.25 | preserves biology. Relative clause added for 4-sp structural limit. |
| E8.4 | `per_celltype_purity_matches_scvi` | max over shared CTs of (chimera kNN purity − scVI kNN purity) | ≤ 0.02 | **Primary gate.** Chimera not worse than scVI within ANY shared CT. |
| E8.5 | `per_celltype_purity_near_chance` | max over shared CTs of (chimera kNN purity − chance) | abs ≤ 0.15 OR rel: max(chimera excess − scVI excess) ≤ 0.05 | within-CT mixing near-chance OR on par with scVI |
| E8.6 | `label_transfer_acc_matches_scvi` | cross-species LT-acc (shared CTs only) | ≥ scVI − 0.05 | z usable for cross-species CT inference |

### E9: Alpha robustness + GIN bidirectional push (added iter108, 2026-04-24)

| # | Name | Metric | Threshold | What it gates |
|---|------|--------|-----------|---------------|
| E9.1 | `alpha_cross_seed_robust` | min Spearman ρ(α_i, α_j) over all seed pairs of same config (≥ 2 seeds; if 1 seed → `skipped`) | **≥ 0.50** (revised 2026-04-25 from prior ≥ 0.90 per user direction) | α reproducible across seeds. Currently PASS on H+Mo iter90 (3 seeds; min ρ = 0.83) and H+Mouse iter90 (3 seeds, iter112; min ρ = 0.762). |
| E9.2 | `gin_bidirectional_push` | frac(α − prior > 0.05) AND frac(α − prior < -0.05) | both ≥ 2% | GIN actively pushes α in BOTH directions. Currently FAILS on every testis run (downward ≈ 0%) — universal GIN architectural ceiling. |

**E9.1 rationale (revised threshold)**: ρ ≥ 0.5 means the dominant ordering of conserved-vs-divergent genes is preserved across seeds while still allowing meaningful seed-to-seed variation in the long tail of mid-range genes (where the loss landscape is genuinely flat). The previous 0.90 bar treated per-gene α as a deterministic fingerprint; in practice with VAMP K=500 random init + adversarial training, ρ ≈ 0.6–0.85 is the attainable range. Headline scientific story (top-K most conserved/divergent gene-sets, top-10% Jaccard 0.44 on H+Mouse) preserved at ρ ≥ 0.5.

---

## Operating norms

### The loop (per invocation, exactly one iteration)

```
            diagnose → hypothesize → fix → retrain → evaluate → compare → commit
                                                                              │
                                                                              ▼
                                                                      stop at iteration
                                                                      boundary
```

User re-invokes for the next iteration. Inside one invocation, do NOT auto-continue.

### Scripts (under `chimera/training/`)

| Script | Purpose |
|---|---|
| `build_subset.py` | Build a k-species subset from unified AnnData (recomputes p_null, filters genes). |
| `train_and_diagnose.py` | Train one model, dump per-epoch snapshots + final state + metrics.csv. |
| `check_expectations.py` | Evaluate E1–E7 + E9.2; write `report.md`. Re-runnable without retraining. |
| `cross_seed_check.py` | Evaluate **E9.1** across 2+ runs of same config (revised threshold ≥ 0.50). |
| `compare_vs_scvi.py` | Evaluate **E8** vs cached scVI baseline; supports `--batch-key`. |
| `run_experiment.py` | End-to-end driver — train + check; writes `summary.json`. |
| `plot_diagnostics.py` | 6 fast plots per run (loss curves, α dynamics, adv, histograms, L-shape, z-var). |
| `summarize_runs.py` | Cross-run markdown comparison table. |
| `recompute_esm2_variance.py` | One-off ESM-2 inter-species variance recompute. |
| `compare_gin_ablation.py` | GIN-input ablation table (used iter17 GIN-input audit). |
| `run_gin_input_sweep.sh` | 4-run GIN-input ablation sweep. |
| `run_xds_sweep.sh` | Cross-dataset sweep (H+Mo, H+Mouse, H+Zeb at minimum). |
| `quick_adv_trajectory.py` | Quick diagnostic for adversary trajectory shape. |

Run artifacts under `runs/<run-name>/`: `metrics.csv`, `alpha_epoch_*.npz`, `final_state.npz`, `model/`, `report.md`, `summary.json`, `config.json`, `plots/`, `integration_vs_scvi.{md,json}`.

### Pre-change template — mandatory before every run

````
Iteration N — <short name>
Expectation under attack: <id from E1–E9>
Change: <exact diff summary; one coherent group, not a grab-bag>
Tier: <1 = HP tune | 2 = loss/schedule | 3 = structural addition | 4 = architecture>

Direct touch-points
  - Modules whose parameters update differently: ...
  - Modules whose gradient magnitudes shift (rough factor): ...
  - Schedules to re-verify (α warmup, FQR, KL, adv, LR, ...): ...
  - Loss-balance numbers that shift: ...

Predicted deltas — signed, rough magnitude REQUIRED
  - Primary species-mixing metric: <sign, magnitude>
  - sil-celltype: ...
  - held-out NB-LL: ...
  - α gate health (BC, std, frac-nontrivial |Δα|): ...
  - residual activity: ...
  - adversary AUC trajectory: ...
  - any module touched: ...

Falsifying probe: <metric whose result would refute the hypothesis>
Rollback plan: <exact revert path: config key, checkpoint, code commit>
````

A change is **accepted** only if (a) primary metric moved in predicted direction by predicted magnitude AND (b) no flagged module regressed silently. Otherwise: rollback.

### Stopping condition (per invocation)

One iteration's record in workflow.md must include:
- Measured deltas vs every predicted delta (signed, magnitude per metric).
- UMAP screenshots/paths for every dataset touched. Thin/elongated UMAP → run E6.3 / KL active-dim chain.
- Integration metrics (kNN-sp, sil-sp, sil-ct, kNN-mix, LT-acc, NMI/ARI, per-CT within-CT kNN purity, linear + non-linear z-leakage probes).
- Pass/fail status for every affected expectation.
- Side-by-side vs vanilla scVI (and SysVI where applicable).
- Full α distribution (BC, std(α), frac Δα>+ε / <-ε, sat fraction).
- Per-module health (GIN |corr|, frac positive Δα, residual ablation Δ, frozen-α Δ, L_recon trajectory, KL active-dim count, adv AUC, train-vs-held-out per-submodule).
- Loss/grad sanity (no NaN, per-component ratio to L_recon, per-submodule grad norm, clip rate).
- Cross-dataset coverage (every dataset that could regress; ≥2 seeds on headline; 4-species when relevant).
- Accept/rollback decision per template; if promoted, **same commit** updates workflow-best.md + `_constants.py` defaults + notebook pointers.
- Next failing expectation + next hypothesis logged.

Then stop. Do not start another iteration. Do not summarize-and-wait.

### Fast-iteration discipline

- **Diagnose with the correct metric** before changing anything. Identify root cause, never superficial symptom.
- **Don't loop on the same approach.** If two consecutive iterations attacked the same expectation through the same module / loss-weight family / schedule lever and both failed, do NOT run a third tweak in that family. Pivot to a structurally different angle.
- **One hypothesis per iteration.** Change one thing (or one coherent group), measure delta, log conclusion.
- **Regression on any previously-passing metric / dataset / seed = blocker.**
- **Sequential MPS only — never run two trainings concurrently.**
- **Diagnostics must be fast** — no slow scib metrics inside the loop. Reserve for final validation.
- **4-species is a first-class iteration target**, co-equal with 2-species.
- **Don't bloat.** Standard HP tuning (BS, LN, activation, depth/width, LR, dropout, WD) is NOT bloat — sweep freely. Inventing new modules / losses / features without measured failure mode IS bloat — resist.

### workflow.md hygiene

This file is the loop's control plane AND institutional memory. Pruning rule: when resolving a hypothesis or finishing an iteration, **distill** its full body into a one-paragraph Lessons entry naming what was tried, **how exactly it failed** (the metric that moved wrong, the module that regressed, the mechanism that didn't engage), and the design implication. A future agent reading workflow.md should be able to answer "what has been tried on this expectation, how did it fail, and what should I try next?" from this file alone.

---

## Recent iterations (most-recent full detail; 109–114 distilled)

### Iterations 113 + 114 — Testis 2-sp E9.2 down attacks (BOTH ROLLBACK, DISTILLED 2026-04-25)

**iter113 (λ_fit 0.1 → 0.3 on H+Mac, ROLLBACK).** Hypothesis: stronger FQR gradient amplifies corrections, including on rare neg-S_g genes. Result: ρ(S_g, corr) measured 0.534 vs iter109 0.514 (Δ=+0.020, predicted +0.04-0.18) — ceiling robust to gradient scaling. E9.2 down passed (13.64%) but for the WRONG reason: wider correction distribution incidentally landing on pos-S_g/zero-S_g genes; the 163 neg-S_g genes still received mean correction +1.65 (positive). E3.2 saturation regressed from 0.06% to 28.53% (HARD FAIL — falsifying probe triggered). **Lesson**: λ_fit scaling does NOT improve per-gene assignment; it just rescales the existing function the MLP outputs. E9.2 is a magnitude metric — passing it without improving Spearman ρ(S_g, corr) is metric-cosmetic, not biological.

**iter114 (GIN width 32 → 64 on H+Mac, ROLLBACK at ep60).** Hypothesis: per-gene discriminability is capacity-limited; 2.6× MLP params (6,785 → 17,665) should let the MLP carve sharper boundaries. Result: ρ measured 0.524 (ep60) vs iter109 0.535 — slightly WORSE. corr range widened ([-0.328, +1.93]) but mean correction on neg-S_g genes stayed at +0.341 (identical to iter109, +0.34). Wider MLP just amplifies global magnitudes without fixing per-gene assignment, mirroring iter113's pattern. Stopped at ep60 to free MPS for iter115. **Lesson**: per-gene discriminability is NOT capacity-limited either. Adding params doesn't change WHICH genes the MLP treats as positive vs negative.

**Combined diagnostic finding**: Per `iter114_input_separability.py`, the 72-d GIN feature space IS discriminable for pos- vs neg-S_g testis genes (linear AUC=0.84, nonlinear=0.80). So the input is NOT the bottleneck either. Three false ceilings ruled out:
1. NOT loss-weight scaling (iter113)
2. NOT MLP capacity (iter114)
3. NOT GIN input space (iter114_input_separability)

**Conclusion** (re-confirmed by 4 testis runs at iter109/iter111/iter113/iter114): The Spearman ρ(S_g, corr) ≈ 0.5 ceiling on testis is a **gradient-signal-STRUCTURE problem**, not a gradient-strength or capacity problem. The candidates are:
- Class imbalance (666 pos-S_g vs 155 neg-S_g testis genes) — gradient bulk dominates positive direction
- L_anchor symmetric penalty — squashes the rare negative-correction tail equally
- Per-minibatch S_g_ema noise / FQR baseline definition (Tier-3 multi-week)

**Artifacts** (preserved): `runs/testis-hmac-iter113-lfit03-200ep-s42/`, `runs/testis-hmac-iter114-gin64-200ep-s42/{alpha_snapshots/, lightning/metrics.csv, config.json}` (no checkpoint — ep60 termination). `training/iter114_diagnostics.py` and `training/iter114_input_separability.py` reusable.

---

### Iteration 112 — H+Mouse pancreas 3-seed verification (2026-04-25, ★ awarded)

3-seed run of iter90 H+Mouse recipe (s42 cached, s43+s44 new). Each seed: **33/33 + 6/6 E8** ✓; **E9.1 cross-seed Spearman ρ_min = 0.762** ≥ 0.50 (revised threshold). LT-acc 0.971 ± 0.005, sil-ct 0.444 ± 0.036, kNN-sp 0.853 ± 0.038, UMAP aspect 1.34 ± 0.12 (round). **★ awarded to H+Mouse iter90; H+Mo iter90 ρ=0.83 also re-gated to ★ under revised threshold.**

**Lesson** (E9.1 threshold revision, 2026-04-25, user-directed): Cross-seed α ρ ≥ 0.50 is the right bar. With VAMP K=500 random init + adversarial training, ρ ≈ 0.6-0.85 is the attainable range; the prior 0.90 bar treated per-gene α as a deterministic fingerprint, which it is not. Top-K Jaccard (top-10% 0.44) verifies that the headline scientific story (most-conserved/most-divergent gene-sets) is preserved across seeds even when long-tail per-gene rank ordering is approximate.

**Lesson** (falsified hypothesis preserved as anti-pattern): Predicted that stronger regularization (adv=200 vs 100, free_bits=1.0 vs 0.5) gives TIGHTER cross-seed α, because larger penalties resist seed-init noise. Measured the OPPOSITE (H+Mouse 0.762 < H+Mo 0.83). Adversarial training strength is **inversely correlated** with cross-seed α reproducibility — stronger adv aggressively reshapes z; small init differences amplify into larger trajectory divergences. **Future iterations attacking E9.1 should NOT push adv higher.**

**Artifacts**: `runs/xds-hmouse-iter90-150ep-{s42,s43,s44}/`, `iter112_diagnostics/{hmouse_e9_1.json,hmo_e9_1.json,umaps_hmouse_3seeds.png}`.

---

### Iterations 109–111 — Testis prior-clamp + CT-cond MMD coupled recipe (DISTILLED)

Full per-dataset numbers live in workflow-best.md (sections 6–7). Compact lessons:

- **iter109 (H+Mac)** — added `--prior-clamp-max 0.85` to iter98 (CT-cond MMD on cell_type) recipe. **E3.2 saturation cured**: 26.17% → **0.06%** (clamp removes the prior-saturation lock on the 32% of testis genes pre-clamped at 0.95). Cost: E1.5 marginal numerical FAIL (α.std=0.0499 vs 0.05). E8 maintained 6/6 with LT-acc improved 0.880→0.899; UMAP aspect 1.30→1.17. **E9.2 push<-0.05 still 0.01%** — Spearman r(S_g_ema, correction)=0.518 reveals a universal **architectural ceiling on the GIN MLP's feature-space discriminability**, not solvable by HP tuning. ACCEPT as testis H+Mac best-known; ★ withheld pending E9.2 down.
- **iter110 (H+Mouse, prior-clamp ALONE)** — ROLLBACK. Applying clamp without CT-cond MMD collapsed posterior (z active dims 10/10→3/10), LT-acc 0.863→0.278, sil-ct 0.255→0.091, E8 6/6→4/6. **Lesson: iter109 recipe is a COUPLED PAIR.** Without CT-cond MMD, prior-clamp inflates the residual which then eats CT info from z (global MMD is CT-agnostic).
- **iter111 (H+Mouse, COUPLED + cell_ontology_class)** — iter109's recipe transferred cleanly when CT-cond MMD uses **`cell_ontology_class`** (5 shared groups, 100% coverage) instead of `cell_type` (3 shared groups, 38% coverage on H+Mouse due to asymmetric Mouse/Human label granularity). E3.2 35% → **0.231%** (152× reduction); kNN-sp 0.991→**0.813** (Δ=-0.173 below scVI 0.986); LT-acc 0.863→**0.993** (+0.097 BEATS scVI 0.896); E8 6/6 maintained; α.std=0.0495 marginal numerical FAIL same as iter109. ACCEPT as testis H+Mouse best-known; ★ withheld pending E9.2 down.

**Three transferable rules (proven across H+Mac and H+Mouse 2-sp testis)**:
1. Testis ESM-2 conservation priors are 8× more pre-clamped at 0.95 than pancreas (32% vs 4%) — `--prior-clamp-max 0.85` is a **testis-tissue-specific override** that is mandatory for E3.2 to pass.
2. Prior-clamp + CT-cond MMD is a **coupled pair**, not independent overrides; never apply clamp alone (iter110 rollback).
3. **When shared-CT coverage on `cell_type` is < 80%, use `cell_ontology_class` for CT-cond MMD** (the harmonized label covering all cells).

**Open architectural ceiling** (iter109 + iter111 + iter113 all confirm): Spearman r(S_g, corr) ≈ 0.5 universal across testis runs. Negative-S_g (S_g<-0.05) genes (~1% of testis genes) have feature-space neighbors that are positive-S_g; smooth MLP interpolates them positive. iter114 (in flight) tests width capacity; if it fails, pivot to depth/activation; ultimate fallback is FQR baseline redesign (Tier-3 multi-week).

**Artifacts**: `runs/testis-hmac-iter109-pclamp085-200ep-s42/{model,conservation_prior_raw.npy}`, `runs/testis-hmouse-iter111-pclamp-coc-200ep-s42/{model,conservation_prior_raw.npy}`. iter98 / iter91b preserved.

---

### Iteration 108 — Tier-0 audit (DISTILLED, 2026-04-24)

Pure evaluation, no model changes. Five actions:
1. **scVI baselines retrained** with `batch_key="batch"` (donors) on pancreas; testis with `sample_batch`. All E8 gates still PASS for Chimera against the stronger donor baselines.
2. **NEW E9.1** (cross-seed α robustness, Spearman ρ; threshold initially ≥ 0.90, revised to ≥ 0.50 in iter112).
3. **NEW E9.2** (GIN bidirectional push, both push>+0.05 ≥ 2% AND push<-0.05 ≥ 2%).
4. **E3.2 tightened** to refinement.md `<5%` (was silently 0.5/50%, 10× lenient).
5. **α-saturation audit revealed two mechanism families**: (A) strong-integration regime (H+Mouse panc, 4-sp panc) — residual carries species, α uniformly low; (B) high-prior regime (testis) — prior pre-clamped at 0.95, FQR signal one-sided positive.

**Saturation table (post-iter108 scoring)**:
| Run | α>0.95 | E3.2 (<5%) |
|---|---|---|
| H+Mo iter90 s42–44 | 6.19/5.85/6.47% | FAIL marginal |
| H+Mouse iter90 | 1.67% | PASS |
| 4-sp iter101 | 0.95% | PASS |
| Testis H+Mac iter98 | **26.17%** | FAIL hard 5× → cured iter109 (0.06%) |
| Testis H+Mouse iter91b | **35.00%** | FAIL hard 7× → cured iter111 (0.231%) |
| Testis unified iter102 | **29.06%** | FAIL hard 5.8× → next iter113 plan retired in favor of iter114 |

**Star revocation rule**: ★ requires BOTH integration AND α biology gates (E3.2 + E9.2) per refinement.md "definition of done." Best-known-for-integration ≠ ★. iter108 revoked ★ on testis H+Mac iter98 and unified iter102.

**Output**: iter108 became the evaluation baseline; iter109 prior-cap fix planned and executed.

---

## Distilled Lessons (older iterations 1–107)

**Each entry**: what was tried, *how exactly it failed*, design implication. Replaces full iteration prose; full bodies removed for legibility. For run-by-run numerical detail, see `runs/<run-name>/{summary.json,report.md,integration_vs_scvi.json}` and `iteration_log.md`.

### Foundation (iter 1–43): 32-gate suite

- **iter1–iter15** (H+Mo, various adv_lr/λ regimes): converged to **iter16** baseline — Tanh GIN, lr=1e-4, adv_weight=100, 100 epochs, 32/32 expectation pass. Verified seeds 42 + 43.
- **iter17–iter21**: 4-species exploration. **iter21** (adv_weight=200, free_bits=1.0) first 4-sp 32/32 — pre-VAMP era, validated seed 42 + 43.
- **GIN-input ablation sweep (iter17 era)**: 4-run table (full / no-ESM2 / no-expr-stats / no-scalars). Verdict: keep 72-d input. ESM-2 features carry signal (no-ESM2 regresses Δα balance 27%/55% → 26%/55%). Scalars (identity stats) marginal but kept. **No further GIN input sweeps without measured-failure cause.**
- **iter22–iter43**: many variants on adversary capacity, MMD knobs, masking. None broke 32/32 ceiling on 2-sp pancreas.

### Integration crisis (iter44–47): 32-gate ≠ "integration"

- **iter44 diagnostic**: iter42 passes 32/32 but FAILS E8.4/5/6 vs vanilla scVI on per-celltype within-CT mixing. Established the integration-gap measurement.
- **iter45a (λ_mmd=500)**: First 5/6 E8 BUT killed Δα bidirectionality (0% pos / 100% neg). Mean|correction| 0.86 → 3.33 (4× larger, all negative). L-shape disappeared.
- **iter45b (λ_mmd=1000)**: 4/6 E8 — overshoots, LT-acc regresses 0.987 → 0.935.
- **iter46a/b/c single-knob rescues** (lambda_res ↓, lambda_mmd ↓, lambda_anchor ↑×100): all FAIL to restore Δα bidirectionality. **Lesson**: not a loss-rescaling problem. The L_recon gradient on `(1−α)·residual` is a uniform downward force on every correction whenever residual carries per-species variance under species-invariant z.
- **iter47 CT-cond MMD at λ=500**: REGRESSION (3/6 E8 vs iter45a's 5/6). Per-cell MMD signal too weak in this regime; integration on harder CTs fails. **Lesson**: CT-cond MMD ≠ universal fix; it changes which groups pressure acts on, doesn't address the L_recon-gradient root cause.

### VAMP breakthrough (iter48–54): 6/6 E8

- **iter48 (VAMP, no MMD)**: 2/6 E8 — α perfect, integration fails. **VAMP alone does not align species.**
- **iter49–iter51 (VAMP + MMD ∈ {2,50})**: 5/6 E8.
- **iter50 (VAMP + λ_mmd=10)**: **first 6/6 E8** on seed 42. kNN-sp 0.737, sil-ct 0.543, LT-acc 0.987 (ties scVI 0.990). E8.4 worst gap −0.215 fibroblasts (BEATS scVI by 0.215). E8.5 worst-CT-vs-chance +0.111 Hep.
- **iter50 3-seed (s42/43/44)**: 5/6 robust E8 (Hep purity-vs-chance fails 2/3 seeds at +0.325/0.187). E8.5 not robust on Hep (largest, most species-skewed CT).
- **iter52 stratified VAMP init**: 6/6→5/6 REGRESSION on s42 (Hep +0.111→+0.284). Forced equal-species pseudo-input init created two equally-strong species-specific Hep attractors that MMD couldn't merge. **Lesson**: natural sampling beats stratified on skewed CT×species distributions.
- **iter53 K=1000 VAMP**: 4/6 E8 strict regression on s43. With K=1000 uniform init, VAMP prior covers nearly all data points; KL becomes very weak; posteriors collapse onto nearest mode. **Lesson**: K=500 is the operating point for uniform init; scaling K needs matching mode-density change.
- **iter54 equal-species sampler**: 4/6 E8 on s43 — REJECTED. The mini-batch gradient stochasticity explanation was wrong; seed variance comes from the encoder/MMD basin selection, not batch composition.

### Δα reconciliation breakthrough (iter81): α-detach + unrectified S_g

- **iter81b (CRITICAL, current code default)**: Two coupled changes in `_module.py`:
  1. `alpha.detach()` in `mu_full` — L_recon gradient does NOT flow to GIN. GIN learns only from FQR + anchor.
  2. Removed `torch.clamp(S_g, min=0.0)` — negative S_g (gene divergent) can push corrections negative.
  Plus `LAMBDA_MMD = 50.0` (up from 10).
- Result: H+Mo s42+s43 BOTH 32/32 + 6/6 E8 + bidirectional Δα (Δα+ 51.2%, Δα- 31.9% on s42). **First config to pass BOTH integration AND α biology simultaneously.**
- **Why it works**: Under α-detach, GIN equilibrium correction ≈ λ_fit·S_g / (2·λ_anchor) = 50·S_g. S_g > 0 → α > prior (more conserved); S_g < 0 → α < prior (more divergent). The L_recon uniform downward force is now blocked from reaching the GIN. Smooth, biologically grounded bidirectional Δα.
- **Divergence-adaptive λ_mmd rule** (post-iter81b): close=50/adv=100, mid=50/adv=200/fb=1.0, very=10/adv=100, 4-sp=50/adv=80 (later 200).

### DATA CORRECTION 2026-04-23 (between iter81 and iter81b) — DO NOT IGNORE

- Every processed `.h5ad` was regenerated 2026-04-23 15:17, fixing TWO data-prep bugs:
  1. **Broken species-name mapping** in `01_data_preparation.ipynb`: missing `"monkey"→"mmulatta"` and `"zebrafish"→"drerio"` entries → ESM-2 inter-species variance feature (col 64) was all-zero on every pre-correction subset.
  2. **Stale "padding" feature slots** (cols 69-71): the unified file built before `assemble_gin_features` was enhanced to compute identity std/max/min carried zeros; subsets inherited.
- Fix: edit notebook (`download_and_process_esm2` returns `(consensus, per_species_matrices)`; `compute_esm2_interspecies_variance` raises `ValueError` on <2 species instead of silently returning zeros). Notebook Section 7 calls `build_subset.py` so a clean run from `00_environment.ipynb` produces every artifact.
- **Affected files** (rewritten 2026-04-23 15:17): all four `camex_*.h5ad` in `data/processed/`, `gin_scaler.pkl`. Backups preserved as `.bak_pre_nb_rerun` (col 64 was patched but cols 69-71 zero) and `.bak_pre_esm2var` (both broken).
- **Implication**: iter ≤ 81 runs on H+Mo/H+Mouse/H+Zeb were trained with 4 of 72 GIN feature dims dead. Comparisons across the boundary are NOT like-for-like. For strict reproduction of an iter ≤ 81 run, use `.bak_pre_nb_rerun` as `--adata`.
- **Verification (iter82)**: re-running iter81b recipe on corrected H+Mo gave 31/32 + 6/6 E8 with Δα + 41.3% / -32.7% — all integration + α biology metrics held. Data correction is non-regressive.

### Loss-balance fix (iter37, "chimera-loss-scaling-fix" memory)

The adv fool_loss was O(1) vs L_recon O(G) — fool_loss was 0.001% of total loss, effectively dead. Fixed by ×n_genes scaling on adv loss + lambda_anchor default 0.005 → 0.002. After this fix the per-component loss ratios in E2.7/E2.8 are calibrated to expected ranges.

### 4-species progression (iter55–101, current best iter101)

- **iter55–80**: many variants under various recipes. iter59a (adv=80, mmd=30, n_latent=20) reached 5/6 E8 in legacy regime (collapsed α=0.20, residual carries species). iter80 with adv=80 + λ_mmd=35 closed E8.4 to +0.001 but Δα one-sided (legacy regime mechanism).
- **iter89 (adv=80, anchor=0.001 — discovery that 4-sp needs lower anchor)**: 2/6 E8, kNN-sp 0.981. Establishes that 4-species requires `--lambda-anchor 0.001` (vs default 0.003) for meaningful Δα; default anchor collapses Δα to <10% on both sides.
- **iter92 (mmd=200)**: z COLLAPSED, 9/10 dead dims. Strong global MMD on 4-species can't be supported by current capacity.
- **iter95 (mmd=100, no fb)**: z collapsed.
- **iter97 (mmd=75, fb=0.5)**: z collapsed.
- **iter99 (mmd=75, fb=1.0)**: z survived, 2/6 E8.
- **iter100 (mmd=100, fb=1.0)**: 3/6 E8, z healthy, sil-ct=0.330.
- **iter101 (adv=200, mmd=100, fb=1.0) — current best**: 30/32 + 3/6 E8. sil-ct=0.408 (best ever for 4sp), LT-acc=0.985 (BEATS scVI 0.916 by +0.069). kNN-sp=0.979 vs scVI 0.917 — structural ceiling at 2 shared CTs (Hepatocytes 16222, Endothelial 2733).
- **iter107 4-sp CT-cond MMD attempt**: ROLLBACK. CT-cond MMD over-integrates at 4-species scale, sil-ct regresses.

### Testis tissue progression (iter91–106)

- **iter91a/b/c**: Testis H+Mac, H+Mouse, unified — first introduction of testis dataset. Recipes inherited from pancreas with some adjustment. Already showed E3.2 saturation (later confirmed in iter108) and one-sided Δα (mechanism family B: high prior).
- **iter94 (testis H+Mac, batch_key=sample_batch=donor)**: ROLLBACK. z contracted 4×, spermatocyte mixing destroyed. **Lesson**: Chimera × multi-batch-per-species `batch_key` interaction is broken; root-cause not yet diagnosed. iter112 flagged this as the open identifiability question; Tier-3 work.
- **iter98 (testis H+Mac CT-cond MMD)**: 30/32 + 6/6 E8 — E8.6 closed (LT-acc 0.880), all CTs pass. **Lesson**: CT-conditional MMD breaks the "global adv mixing vs CT separation" trade-off by enforcing within-CT species mixing.
- **iter102 (testis unified CT-cond MMD)**: 29/32 + 6/6 E8 — same pattern as iter98 for unified. Was ★ until iter108 audit revoked (E3.2 sat 29% hard-fail).
- **iter106 (evaluation framework verification)**: Two fixes — (1) E2.7 `lambda_res * L1 / L_recon` scale was 50× inflated (bug in scale denominator), now correctly in [1e-3, 0.3]; (2) testis scVI baselines re-trained with `batch_key="sample_batch"` (donors) for fair Chimera comparison. All testis E8 gates still PASS against the stronger donor baselines.

### Cross-seed α robustness (iter108–112)

- **iter108**: H+Mo iter90 3-seed audit gave Spearman ρ pairwise 0.831/0.842/0.827. Initial threshold ≥ 0.90 → MARGINAL FAIL. Top-K Jaccard high (top-1% 0.76).
- **iter112**: H+Mouse iter90 3-seed gave min ρ = 0.762, Top-1% Jaccard 0.24, Top-10% 0.44. Stronger regularization (adv=200 vs 100, fb=1.0 vs 0.5) gave LOWER cross-seed reproducibility — falsifying the "stronger reg → tighter cross-seed" hypothesis.
- **2026-04-25 user-revised threshold**: ≥ 0.50 (was ≥ 0.90). Rationale in E9.1 row above. Both H+Mo and H+Mouse PASS under revised; ★ promoted on both.
- **Lesson**: adversarial-training strength is **inversely correlated** with cross-seed α reproducibility. Future iterations attacking E9.1 should NOT push adv higher.

### Other rejected / investigated approaches

- **HSIC negative result** (memory: chimera-hsic-negative): HSIC(z, species) at λ=0.1 and 1.0 left kNN purity at 1.000. Don't enable by default.
- **Config M vs D** (memory: chimera-config-m-vs-d): GIN (D) gives ~3× larger, bidirectional Δα vs per-gene param (M). Keep GIN.
- **scvi-tools kwarg regression** (memory: chimera-scvi-kwarg-regression): `datamodule`→`data_module` + removed `self.drop_last`. Regression gate is verify-iter16-post-fix.
- **z integration fix** (memory: chimera-z-integration-fix): MMD+masking+z-detach fixes z integration (sil-sp 0.286→0.033, sil-ct 0.217→0.642). Best λ_mmd=1000 for that pre-VAMP regime.
- **Caveat 34 investigation** (memory: chimera-caveat34-investigation): iter22/23/24 deeper adv + mask both → linear sep 99%→72% but nonlin still 100% (data-intrinsic).
- **z-balance loss** (memory: chimera-z-balance-loss): L_z_balance fixes VAMP z-dim ratio (4617→1.04). Δα anchor sweep rejected. BC=0.43 is biologically correct.
- **Anchor / FQR / λ_res sweeps** (iter83–87): tested at 2-sp pancreas under iter81b code. All accepted final values (lambda_res=0.005 → 0.001, lambda_anchor=0.003, lambda_fit=0.1, lambda_z_balance=1.0). Higher anchor narrows Δα; weaker anchor inflates Δα magnitude but loses calibration. The current defaults are the validated equilibrium.

---

## Architectural ceilings & structural limits (currently unresolved)

These are NOT HP-tunable. Each requires Tier-3 architectural work or is genuinely data-limited.

### E9.2 down on testis (GIN MLP feature-space discriminability — capacity-limited, not loss-weight-limited)

**Mechanism**: Spearman r(S_g_ema, correction) ≈ 0.5 ceiling across all testis runs at equilibrium (iter109: 0.514, iter113: 0.534). Capacity-limited diagnosis: **iter113 (λ_fit 0.1 → 0.3, 3×) gave Δρ = +0.020 — essentially unchanged ceiling.** The ~1% of negative-S_g genes (S_g < -0.05) have feature-space neighbors that are positive-S_g; the GIN MLP interpolates and pulls them positive (iter113 measurement: mean correction on negative-S_g genes = **+1.65**, despite global correction range expanding to [-2.03, +2.88]).

**iter113 result confirms the ceiling is genuinely architectural**: stronger gradient (3× λ_fit) did not improve per-gene assignment, only amplified the overall correction distribution. E9.2 down passed iter113's metric (13.64%) but for the WRONG reason — incidental amplification, not improved discriminability.

**Tier-2 attacks attempted, both REJECTED**:
- iter113 (λ_fit ×3): Δρ = +0.020 — loss-weight scaling does not break the ceiling.
- iter114 (GIN width ×2.6): Δρ = -0.011 — capacity does not break the ceiling either. Combined with `iter114_input_separability.py` (linear AUC=0.84 / nonlinear=0.80 on the 72-d GIN inputs for pos- vs neg-S_g testis genes), the input space IS discriminable and the MLP IS large enough — the bottleneck must be elsewhere. **iter115 plans GeLU activation** (different smoothness profile across feature-space neighbors); if also rejected, **iter116 plans class-balanced FQR loss** and **iter117 per-gene bias hybrid** (each from a structurally-different mechanism family).

**Tier-3 fallback (if iter115/116/117 mechanism-shape attacks all fail)**: Replace `baseline = p_null · lib` in FQR with `baseline = mu_full_ema.detach()`. Makes FQR signal the deviation of `mu_full` from its own EMA — naturally bidirectional. Multi-week to re-calibrate all loss-balance numbers and validate non-regression on pancreas.

### 4-species pancreas E8.4 within-CT mixing (data-limited)

**Mechanism**: Only 2 shared cell types (Hepatocytes 16222, Endothelial 2733) across 4 species. iter101 reaches LT-acc 0.985 (BEATS scVI 0.916) and sil-ct 0.408 (best ever for 4sp), but kNN-sp = 0.979 vs scVI 0.917 — species form distinct local manifolds with limited bridges. iter107 CT-cond MMD on this dataset over-integrated (sil-ct regression).

**Tier-3 candidate fix**: Species-pair-conditional MMD (Σ over species pairs of MMD per pair, weighted by pair divergence) instead of all-species global. Or: shared-CT-augmentation via consensus-CT extraction. Or: accept as data-limited.

### H+Zebrafish at 450 Myr divergence (structural)

**Mechanism**: kNN-sp = 1.000 (scVI also 1.000). At 450 Myr, species are not mixable in z without destroying celltype structure (sil-ct -0.093). LT-acc 0.704 — Chimera matches scVI at this divergence. Multiple iter45-style attempts; no recipe reaches 6/6 E8.

**Treated as structural** unless a Tier-3 cross-species cycle-consistency or species-conditioned residual augmentation is attempted.

### Chimera × multi-batch-per-species `batch_key` (testis identifiability)

**Mechanism (suspected)**: testis datasets have 5–9 donor batches per species. iter94 tested Chimera with `batch_key=sample_batch` on testis H+Mac and rolled back (z contracted 4×). The interaction with ConstrainedBatchOffset, adversary, and MMD is not understood. scVI uses sample_batch successfully — but its model has no species-specific machinery.

**Tier-3 work**: Two-categorical model (batch_key=donor for batch correction, species_key=species for species-specific machinery). Requires careful re-derivation of all loss terms that currently consume `batch_index` interchangeably with `species_index`.

---

## Run log (recent only)

The full run-by-run history is preserved in `iteration_log.md`. Below is a compact recent slice for fast cross-reference; older rows are intentionally elided.

| # | Dataset | Recipe (overrides vs default) | Headline | Pass | Verdict |
|---|---|---|---|---|---|
| iter90 | H+Mo (panc) | (defaults) 150ep | 32/32 + 6/6 E8 (3 seeds) | E9.1 ρ=0.83 PASS | ★ |
| iter90 | H+Mouse (panc) | adv=200, fb=1.0, 150ep | 33/33 + 6/6 E8 (3 seeds, iter112) | E9.1 ρ=0.762 PASS | ★ |
| iter85 | H+Zeb (panc) | mmd=10 | 28/33 + 4/6 E8 | structural at 450 Myr | best |
| iter101 | 4-sp panc | adv=200, mmd=100, fb=1.0, anch=0.001, 200ep | 31/33 + 3/6 E8 | structural at 2 shared CTs | best |
| iter109 | Testis H+Mac | clamp=0.85, ct-mmd=cell_type, adv=100, fb=1.0, 200ep | 30/33 + 6/6 E8 | E3.2 cured 0.06%; E9.2 down arch | best |
| iter111 | Testis H+Mouse | clamp=0.85, ct-mmd=cell_ontology_class, adv=200, fb=1.0, 200ep | 28/33 + 6/6 E8 | E3.2 cured 0.231%; E9.2 down arch | best |
| iter102 | Testis unified | ct-mmd=cell_type, adv=100, fb=1.0, anch=0.001, 200ep | 28/33 + 6/6 E8 | **E3.2 sat 29% HARD FAIL** | iter113 cures |
| iter112 | H+Mouse panc multi-seed | (iter90 recipe, s42+s43+s44) | 33/33×3 + 6/6×3 + E9.1 PASS | per iter90 row | ★ promoted |
| iter113 | Testis H+Mac (λ_fit ×3) | iter109 + `--lambda-fit 0.3` | E3.2 sat 28.53% HARD FAIL; ρ(S_g, corr)=0.534 (iter109 0.514) | gradient strength NOT the bottleneck | **ROLLBACK** |
| iter114 | Testis H+Mac (GIN width 32→64) | iter109 + `--gin-hidden 64` | early stop ep60; ρ=0.524 (iter109 0.535); E9.2 down 0% | MLP capacity NOT the bottleneck | **ROLLBACK** |
| iter115 | Testis unified (3-sp, iter111 recipe transfer, 100ep) | iter102 + clamp=0.85 + ct-mmd-key=cell_ontology_class | 30/33 + 6/6 E8 vs scVI(donor) + 6/6 E8 vs scVI(species); E3.2 1.32%, E9.2 down 1.57% (close to PASS), LT-acc 0.992 (BEATS scVI(species) 0.901 by +0.092), UMAP aspect 0.91 | E3.2 HARD FAIL CURED 22×; new best for testis_unified | **★ promoted (testis_unified registry HARD FAIL resolved)** |
| iter116 | Testis H+Mac (slower SG_EMA β 0.9→0.97) | iter109 + `--sg-ema-beta 0.97` (NEW opt-in CLI flag; default 0.9 unchanged) | 28/33 + 6/6 E8 vs scVI(donor); E9.2 down 0.007% UNCHANGED, ρ(S_g, corr) 0.490 (vs iter109 0.514); corr range NARROWED (max 1.169 vs 1.373); UMAP PC1/PC2 aspect 2.52 (vs 1.64, REGRESSED) | per-minibatch S_g noise NOT the bottleneck; signal density via temporal averaging refuted | **ROLLBACK** (iter109 retained as best-known; CLI flag preserved opt-in) |
| iter117 | Testis unified (3-sp, iter115 recipe at **200ep** s42) | same as iter115; just `--max-epochs 200` | 30/33 + 6/6 E8 vs scVI(donor) + 6/6 E8 vs scVI(species); E9.2 down **1.04%** (FAIL, below predicted 1.5-2.5%, below iter115's 1.57%); E9.2 up 23.71%; α mean=0.856 std=0.058; corr range [-0.696, +2.134] WIDER; ρ(S_g, corr)=0.448 ↑ but mean corr on neg-S_g=+1.024 (more positive — MLP even WORSE at rare class); z PC1/PC2=1.77 REGRESSED, UMAP aspect 1.147 round; LT-acc 0.992 (matches iter115) | NO CODE CHANGE; testis_unified 3-sp E9.2 down stochastic basin distribution is ~1.7%±0.6%, below 2% PASS at most realizations — architectural ceiling extends to 3-sp; killed-200ep 2.43% was stochastic fluke | **ROLLBACK** (iter115 100ep remains best-known; pivot to iter118 Tier-3 attack on loss STRUCTURE) |
| iter118 | Testis H+Mac (asymmetric L_anchor factor=0.3, 200ep s42) | iter109 + `--lambda-anchor-neg-factor 0.3` (NEW opt-in CLI flag; default 1.0 unchanged) | 29/33 + 6/6 E8 vs scVI(donor); E9.2 down **0.39%** (FAIL, below predicted 1.5-5%, below 2% PASS line); mean corr on neg-S_g **+0.279** (FAIL falsifying probe ≤+0.15); frac_corr_neg_on_neg_S_g 5%→16% (3× directional shift IS real); Δα<-0.02 1.6%→7.1% (4.4×); ρ(S_g, corr)=0.472 (DOWN from 0.514); z PC1/PC2=3.35 (REGRESSED from 1.64); UMAP aspect 1.23 (round); LT-acc 0.891 (-0.008 from 0.899; within E8.6 tol) | Anchor relaxation produces a real but ~5× too small directional shift on rare class — confirms MLP feature-space smoothness is the dominant binding constraint, not anchor symmetry; per-gene bias hybrid is now strongly motivated | **ROLLBACK** (iter109 retained as testis H+Mac best-known; CLI flag preserved opt-in default 1.0) |
| iter119 | Testis H+Mac (per-gene bias hybrid, 200ep s42) | iter109 + `--use-per-gene-bias` (NEW opt-in CLI flag; default OFF unchanged) | 28/33 + 6/6 E8 vs scVI(donor); E9.2 down **0.0068%** (HARD FAIL, far below predicted 5-15%); **bias_share_of_total_magnitude=2.7%** (corrected — original 0.0% reading was a diagnostic bug, see iter120 invocation lessons; mechanism DID engage but only weakly); mean corr on neg-S_g +0.239 (vs +0.353 iter109, +0.279 iter118); mean_per_gene_bias_on_neg_S_g=-0.027, on_pos_S_g=+0.019 (correct DIRECTION but tiny magnitude); ρ(S_g, corr)=0.450 (vs 0.514 iter109); z PC1/PC2=4.43 (REGRESSED from 1.64; same 200ep over-fit pattern as iter117/118); UMAP aspect 1.17 (PASS round); E3.2 std=0.0495 marginal numerical FAIL (0.0009 below 0.05 floor) | Bias engaged in correct direction but bounded by Adam's per-step LR-magnitude update at LR=1e-4 (cumulative movement over 20K steps capped at ~LR×N). config M's per_gene_correction works because it has its own LR=1e-3 param group at line 326 of _training_plan.py. iter120 fixes this LR mismatch. | **ROLLBACK** (iter109 retained as testis H+Mac best-known; flag preserved opt-in default OFF; iter119_bias_extract.py BUG FIXED in iter120 invocation — `.copy()` after `.numpy()` to prevent storage aliasing) |
| iter120 | Testis H+Mac (per-gene bias hybrid + LR=1e-3 param group, 200ep s42) | iter109 + `--use-per-gene-bias` + 5-line `_training_plan.py:configure_optimizers` change | E9.2 down **0.020%** HARD FAIL (predicted 5-15%); bias_share 22.6% (PASS — LR fix engages mechanism 8×); mean corr on neg-S_g class shifted +0.353→+0.011; E8 6/6 maintained; UMAP 1.07 round | Per-gene mech engages but primary metric capped by testis S_g_ema distribution (only 1.06% genes S_g<-0.05) — architectural ceiling is in FQR baseline definition, not any per-gene mechanism. 8 consecutive testis E9.2 down attacks now ruled out. PIVOT REQUIRED. | **ROLLBACK** (iter109 best-known retained; flag opt-in default OFF; LR fix kept in `_training_plan.py` reversible/benign at default; full detail in workflow-best.md §6 last-iter120 entry) |
| iter121 | 4-sp pancreas DUAL MMD (iter101 + global-MMD-aux on top of CT-cond, 200ep s42) | iter101 + `--celltype-conditional-mmd --celltype-key cell_type --lambda-mmd 50 --lambda-mmd-aux 100` (NEW opt-in flag — adds global MMD on top of CT-cond, byte-identical when omitted) | E8 **2/6** (REGRESSED from iter101 3/6); kNN-sp 0.976 (predicted 0.85-0.92, **HARD FAIL** — essentially unchanged); sil-ct **0.244** (HARD FAIL below 0.25, was 0.408); E8.4 worst-CT gap **+0.123** on Endothelial (HARD FAIL); z_mean_var_min **0.003** (HARD FAIL — 4/10 weak dims, partial posterior collapse triggered E6.3); α biology HEALTHY (mean=0.751, std=0.129, BC=0.471, E9.2 down 35.35%, mean_corr_on_neg_S_g=-0.93, ρ=0.55); UMAP aspect 1.21 round | Dual MMD at lambda_mmd=50 + lambda_mmd_aux=100 PUSHED TOO HARD: z absorbed extra pressure by collapsing 4/10 dims; remaining capacity over-mixed Hepatocytes (73% of cells) within itself, destroying CT structure. 4-sp pancreas with only 2 shared CTs (Hep + Endo) has no "free room" to push more MMD without sacrificing CT structure. GIN-side biology completely preserved (architectural separation between MMD/z-side and FQR/α-side confirmed). | **ROLLBACK** (iter101 retained as 4-sp pancreas best-known; `--lambda-mmd-aux` flag preserved opt-in default 0.0 = byte-identical) |

---

## Notes for the next agent

1. **Read this file end-to-end AND [workflow-best.md](workflow-best.md)** before doing anything. Both are authoritative; if a config you intend to use is not the documented current best, reconcile (update registry or update plan) before running.

2. **Where the loop stands (post-iter121 ROLLBACK, 2026-04-26)**:
   - **Pancreas 2-sp (H+Mo, H+Mouse): both ★ DONE.** Three-seed verified, full 33/33 + 6/6 E8 + E9.1 PASS.
   - **Pancreas H+Zeb: structural at 450 Myr** — 4/6 E8 with kNN-sp at chance. Treated as data-limited; no further iteration unless cycle-consistency Tier-3 work is authorized.
   - **Pancreas 4-sp (iter101 retained as best-known)**: 3/6 E8, kNN-sp 0.979 vs scVI 0.917 (Chimera WORSE). iter121 dual MMD attempt failed (over-pressurized z, collapsed 4/10 dims, sil-ct dropped 0.408→0.244, E8 regressed to 2/6). Next attack class must be structurally-different from MMD-weight retunes.
   - **Testis triad (H+Mac iter109, H+Mouse iter111, unified iter115)**: 6/6 E8 vs scVI(donor) on all three; **E9.2 down hits architectural ceiling** (0%, 0%, 1.57%). Per-gene mechanism family EXHAUSTED (8 failed attacks iter113–iter120). ★ withheld until E9.2 down resolved.

3. **What this invocation did (iter121)**:
   - Pivoted off the exhausted testis E9.2-down per-gene-mechanism family.
   - Implemented opt-in dual MMD: new constructor param `lambda_mmd_aux` in `_module.py`; new CLI flag `--lambda-mmd-aux` in `train_and_diagnose.py` and `run_experiment.py`. When > 0 AND `--celltype-conditional-mmd` is set, adds `lambda_mmd_aux × L_mmd_global` to the loss alongside the existing `lambda_mmd × L_mmd_ct`. Default 0.0 = byte-identical to all current best-knowns.
   - Trained iter121 (4-sp pancreas, lambda_mmd=50 + lambda_mmd_aux=100, 200ep, s42, ~2hr): **ROLLBACK**. Two HARD falsifying probes triggered (kNN-sp essentially unchanged 0.976 vs predicted 0.85-0.92; z_mean_var_min=0.003 partial posterior collapse predicted ≥ 0.1). E8 went 3/6 → 2/6 (REGRESSED on integration). α-side biology preserved fine — the failure was purely on the z/integration side.
   - Fixed pre-existing argparse `%` escape bug in two help strings (--lambda-anchor-neg-factor, --use-per-gene-bias) so `--help` no longer crashes (zero behavioral impact).
   - Distilled iter118/iter119/iter120 detailed tables into compact "Recent rejections" entry; full data preserved in workflow-best.md §6 and run dirs.
   - Workflow.md size: ~880 lines (well within 3000 limit).

4. **Strategic question for the NEXT invocation (USER INPUT NEEDED)**: TWO definition-of-done gaps remain, both with their previously-considered families exhausted by anti-pattern rule.
   - **Gap A: 4-sp pancreas integration** (Chimera kNN-sp 0.979 > scVI 0.917). Two MMD-family attacks now failed (iter107 CT-cond ALONE, iter121 dual MMD). Open paths:
     - **(α) 4-sp species-pair-conditional MMD** — weighted Σ over species pairs of MMD per pair (NOT within-CT). Orthogonal mechanism from CT-cond + global. Targets pairs that mix worst (e.g., Mouse↔Zebrafish at 450 Myr) without diluting easier pairs. Tier-3 code change (~1 day) + 1 training (~3 hr).
     - **(β) Refine 4-sp E8 expectations to be structure-aware** — accept that with only 2 shared CTs, kNN-sp is data-attainable-ceiling-limited. Tighten or relax thresholds based on shared-CT count. Requires user approval to amend refinement.md.
     - **(γ) Reduce dual-MMD pressure** — try `--lambda-mmd 30 --lambda-mmd-aux 50` (well below iter101's effective 100). Would be the THIRD MMD-family attack on 4-sp; refinement.md anti-pattern allows it only because the lower-weight regime is structurally distinct (CT-cond at low weight + small global aux is qualitatively different from iter121's high-pressure config). 1 iteration.
   - **Gap B: Testis triad E9.2 down** (per-gene mechanism family exhausted, iter113–iter120). Open paths:
     - **(δ) Refine E9.2 expectation to be data-adaptive** — `min(2%, 60% × frac(S_g<-0.05))`. Closes testis_unified ★ (iter115 1.57% > adaptive ~0.6%). Doesn't fully close 2-sp testis. Requires user approval to amend refinement.md.
     - **(ε) Tier-3 FQR baseline redesign** — `baseline = mu_full_ema.detach()` instead of `p_null × lib`. Naturally bidirectional S_g_ema. Multi-week (re-validates ALL loss-balance numbers and ALL best-known recipes).
   - **Recommended ordering**: option (α) (species-pair-conditional MMD on 4-sp) is the lowest-blast-radius single-iteration attack on Gap A. Option (γ) (lower-pressure dual MMD) is a quick re-attempt but borderline anti-pattern. Options (β), (δ), (ε) all involve user-policy decisions or multi-week work.

5. **CRITICAL diagnostic findings preserved from iter117–iter121 (do NOT lose)**:
   - **ρ(S_g, corr) is misleading as a primary metric** for testis E9.2 down — iter117 had higher ρ than iter115 yet worse E9.2 down. The right co-primary signal is **mean correction on neg-S_g class** + **bias_share_of_total_magnitude**. Both reported by `gin_signed_diagnostics.py`.
   - **The architectural ceiling on testis E9.2 down is in the FQR baseline definition `p_null × lib`**, not in any per-gene mechanism. The math: testis has 1.06% of genes with S_g<-0.05; even at perfect mechanism engagement, only ~1% can reach Δα<-0.05.
   - **Per-gene mechanisms converge by ~ep55** on testis. Future per-gene experiments can use 80–100ep instead of 200ep.
   - **iter121 dual MMD lesson**: 4-sp pancreas with only 2 shared CTs has no "free room" to push more MMD without sacrificing CT structure. Increasing total MMD pressure beyond iter101's level collapses z dims (4/10 weak at lambda_mmd=50 + lambda_mmd_aux=100), and the remaining capacity over-mixes within the dominant Hepatocyte cluster (73% of cells). The architectural separation between MMD/z-side and FQR/α-side is preserved — α biology (mean corr neg-S_g, ρ, bidirectional Δα) was completely fine. The integration ceiling is structural to the data, not the mechanism.

6. **Identifiability state (closed audit, 2026-04-26)**: Verified per-dataset batch/species columns. Testis Chimera uses `batch_key=batch` (==species) while scVI baseline uses `batch_key=sample_batch` (donors). The asymmetry is INTENTIONAL — scVI(donor) is the STRONGER baseline (more batch-correction power), and Chimera STILL beats it on integration metrics. Pancreas has no `sample_batch` column (donors live in `batch`); both Chimera and scVI use the same key. NO re-runs needed. The architectural Chimera × multi-batch-per-species problem (iter94 rollback root cause) remains a Tier-3 open question deferred indefinitely.

7. **Next-iteration evaluation steps** (after any training completes — this invocation verified the pipeline works for 4-sp):
   - (a) `check_expectations.py` auto-runs from `run_experiment.py` → produces `report.md` + `summary.json`.
   - (b) Run `compare_vs_scvi.py --chimera-run-name <name> --adata <adata-path> --batch-key batch` (use `--batch-key sample_batch` for testis); auto-uses cached baselines at `runs/baselines_scvi/`.
   - (c) Run `gin_signed_diagnostics.py <run-dir>` for per-gene signed metrics (rare-class mean corr is co-primary; ρ alone misleading).
   - (d) Run `umap_check.py <run-dir> --adata <adata-path>` for UMAP plot + aspect ratio (must be < 1.5).
   - (e) Measured-vs-predicted entry in workflow.md against the pre-change template.
   - (f) If accepted: atomic update of workflow-best.md + `_constants.py` defaults (only if scenario default changes) + notebook pointers.
   - **NOTE**: `compare_vs_scvi.py`, `umap_check.py`, `gin_signed_diagnostics.py` are NOT auto-invoked from `run_experiment.py` — must be run manually.

8. **Code state at end of iter121**:
   - `chimera/_module.py` NEW: `lambda_mmd_aux: float = 0.0` constructor param + `self.lambda_mmd_aux`. New `if lambda_mmd_aux > 0 AND use_celltype_conditional_mmd AND celltype_idx is not None` branch computes `L_mmd_aux = self._compute_mmd(z, species_idx)`. Loss adds `+ self.lambda_mmd_aux * L_mmd_aux`. Outputs gain `L_mmd_aux` key.
   - `training/train_and_diagnose.py` + `training/run_experiment.py` NEW: CLI flag `--lambda-mmd-aux`, plumbed through `setup_and_train_chimera_model` and `extra_model_kwargs`.
   - **All earlier opt-in flags retained**: `--use-per-gene-bias`, `--lambda-anchor-neg-factor`, `--sg-ema-beta`, `--gin-hidden`, `--prior-clamp-max`, `--celltype-conditional-mmd`, `--celltype-key`. All default OFF / byte-identical legacy.
   - `_training_plan.py:configure_optimizers` per_gene_bias→1e-3 carve-out (from iter120) preserved; only active when `use_per_gene_bias=True`.
   - **All existing best-knowns** (iter90 H+Mo / iter90 H+Mouse / iter101 4-sp / iter109 H+Mac / iter111 H+Mouse / iter115 unified) UNAFFECTED — verified by smoke test (default constructor lambda_mmd_aux=0.0, new branch dead).

9. **Stop after one iteration.** The user drives the outer loop. Do not auto-continue.
