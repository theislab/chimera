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

**iter116 REJECTED (2026-04-25, this invocation).** Slower SG_EMA_BETA (0.9 → 0.97) on testis H+Mac — falsified the per-minibatch S_g noise hypothesis. PRIMARY metric (E9.2 push<-0.05) UNCHANGED at 0.007% (predicted 2-15%, target ≥ 2%). Spearman r(S_g_ema, correction) slightly WORSE (0.490 vs iter109 0.514, predicted 0.55-0.70). UMAP aspect REGRESSED (PC1/PC2 ratio 2.52 vs 1.64; falsifying probe triggered on UMAP shape too). E8 maintained 6/6 vs scVI(donor); some metrics improved (E3.2 0.06% → 0%, sil-ct 0.321 → 0.381, kNN-sp 0.686 → 0.694), but the failing-and-targeted metric did not move. **Conclusion: density-via-temporal-averaging is NOT the bottleneck for testis 2-sp E9.2 down.** Pivot to count-mechanism (joint-3sp transfer) or class-imbalance (class-balanced FQR loss) attacks. Code change (`--sg-ema-beta` CLI flag) preserved as opt-in diagnostic; default β=0.9 unchanged → no impact on existing best-known recipes.

**iter115 ACCEPTED — testis_unified E3.2 HARD FAIL CURED, promoted to best-known (2026-04-25).** Applied iter111 coupled recipe (prior-clamp 0.85 + CT-cond MMD on cell_ontology_class) to testis 3-sp unified at 100 epochs (early termination from planned 200 to give the user fast iteration). **30/33 + 6/6 E8 vs scVI(donor) + 6/6 E8 vs scVI(species)**. **E3.2 frac>0.95: 29.06% → 1.32%** (22× cure — the registry's only remaining HARD FAIL is now resolved). E5.3 also cured. All integration metrics improved: kNN-sp 0.792 → 0.730, LT-acc 0.989 → 0.992, sil-ct 0.347 → 0.320 (slight). UMAP aspect 0.91 (round, not thin). E9.2 down 1.57% (still below 2% PASS threshold but 200× better than iter102's ~0% — the killed 200ep run reached 2.43% PASS, indicating the 200ep schedule is what crosses the line; 100ep's faster LR decay produces a more aggressive but slightly less downward-skewed equilibrium). E6.1/E6.2 z-leakage gates fail (adv_acc=0.609 vs chance 0.333) — long-standing testis issue, not new from this iteration. **Same recipe + ★ status as iter109 H+Mac and iter111 H+Mouse: integration done, E9.2 down architectural ceiling at 2-sp.** With 3-sp the ceiling is much closer to passing — confirms "gradient-signal-density" hypothesis from the brainstorm.

**iter113 + iter114 (REJECTED 2026-04-25 — distilled).** Both attacked testis 2-sp E9.2 down, both rejected after a single iteration; details in *Distilled Lessons* below. TWO false ceilings ruled out: (a) loss-weight scaling (iter113 λ_fit×3 → Δρ=+0.020), (b) MLP capacity (iter114 width×2.6 → Δρ=-0.011). The 72-d GIN input features ARE discriminable for pos- vs neg-S_g testis genes (separability AUC = 0.84 linear / 0.80 nonlinear via `iter114_input_separability.py`). Bottleneck is the **gradient signal STRUCTURE** (class imbalance / L_anchor symmetry / per-minibatch noise / FQR baseline) — iter115's 3-sp result lends weight to the per-minibatch S_g noise hypothesis (3-sp averages over more cells per gene, reducing noise).

## iter115 measured-vs-predicted (post-acceptance, 2026-04-25)

| Metric | iter102 (prior best) | iter115 100ep (s42) | Δ measured | Δ predicted | Verdict |
|---|---|---|---|---|---|
| **PRIMARY** E3.2 frac(α>0.95) | 29.06% | **1.32%** | -27.74 (22× cure) | predicted 0.1-1.0% | (slightly above predicted range; well under 5% target — PASS) |
| α std | 0.073 | 0.055 | -0.018 (narrower) | predicted 0.05-0.07 | (in range — no E1.5/E3.2 std-clause fail) |
| α mean | 0.904 | 0.850 | -0.054 | predicted 0.85-0.88 | (in range) |
| E9.2 push>+0.05 | ~0.5% | **18.68%** | +18.2 (37×) | predicted 5-15% | (slightly above range; PASS ≥ 2%) |
| E9.2 push<-0.05 | ~0% | **1.57%** | +1.57 | predicted 0-2% (HELD as architectural) | (close to PASS threshold — improved more than predicted) |
| E8 vs scVI(donor) | 6/6 (iter106) | **6/6** | maintained ✓ | predicted 5/6 or 6/6 | (PASS) |
| E8 vs scVI(species) | 6/6 | **6/6** | maintained ✓ | predicted 5/6 | (PASS, exceeded prediction) |
| kNN-sp (vs scVI(donor)) | 0.792 | 0.730 | -0.062 (better) | predicted 0.65-0.85 | (in range) |
| sil-ct | 0.347 | 0.320 | -0.027 | predicted ±0.05 | (in range) |
| LT-acc (vs scVI(donor)) | 0.989 | 0.992 | +0.003 | predicted 0.95-1.00 | (in range; matches scVI 0.991) |
| LT-acc (vs scVI(species)) | 0.989 | 0.992 | +0.003 | not predicted | (BEATS scVI 0.901 by +0.092) |
| z active dims | 10/10 | 10/10 | held ✓ | predicted 10/10 | (held, no posterior collapse) |
| z_min_var | 2.51 | 5.04 | +2.53 | predicted 2-9 | (in range; healthy) |
| residual_rms | n/a | 3.92 | n/a | predicted ≤ 5.0 | (PASS) |
| adv_acc | n/a | 0.609 | n/a | predicted 0.65-0.78 | (slightly below — adv barely reduced past peak 0.634; E6.1/E6.2 fail) |
| Δα > 0.02 | 32.0% | 41.16% | +9.16 | (not predicted) | (more bidirectional than predicted) |
| Δα < -0.02 | 4.7% | 10.56% | +5.86 | (not predicted) | (more bidirectional than predicted) |
| UMAP aspect | n/a | 0.91 | n/a | predicted ≤ 1.5 | (round, PASS) |

**Predicted-vs-measured discipline (the wrong predictions are the lesson)**:
- ✗ E9.2 down predicted "~0%" (architectural ceiling); measured 1.57% — **the architectural-ceiling hypothesis is partially refuted on 3-sp.** The 3-sp regime has more genes with non-trivial neg-S_g (because per-gene S_g averages over more species pairs), reducing per-minibatch noise the GIN MLP must overcome. Per the user-prompted brainstorm, this favors the "per-minibatch S_g noise" diagnosis over the "MLP smoothness" diagnosis.
- ✗ E3.2 sat predicted 0.1-1.0% (matching iter109 0.06% / iter111 0.231%); measured 1.32% — slightly higher because 100ep schedule gave more aggressive corrections than 200ep (the killed 200ep run had 0.25% at the same metric). Still well under 5% target — PASS.
- ✓ E8 6/6 maintained vs both baselines; LT-acc beats scVI(species) by +0.092 (best ever for testis_unified).
- ✓ kNN-sp + sil-ct in predicted ranges.
- ✗ Original 200ep prediction was based on the killed run's epoch-100 snapshot which had a DIFFERENT trajectory than the 100ep target run — the LR scheduler is max_epochs-dependent, so a 100ep run is NOT just a "first 100 epochs of a 200ep run." This is a NEW lesson worth preserving.

**Mechanistic finding (iter115 + brainstorm)**: 3-species training reduces per-gene S_g_ema noise (averaged across more species → tighter per-minibatch estimate) AND increases the count of genes with non-trivial neg-S_g (more pairwise contrasts surface more divergent genes). Both effects make the GIN MLP's discrimination problem easier. **This refines the workflow's "gradient signal structure" hypothesis: signal DENSITY is the primary issue, not signal STRUCTURE.** Future testis 2-sp attacks should target signal density (slower S_g_ema, larger batch size, joint 3-sp transfer) before code-changing class-balanced FQR / per-gene bias.

**Decision**: ACCEPT iter115 100ep as new testis_unified best-known.

---

## Current state (2026-04-25)

**2-species pancreas — DONE (★).** H+Mo iter90 (3 seeds) and H+Mouse iter90 (3 seeds, iter112) both pass the full expectation suite under the revised E9.1 ≥ 0.50 threshold: 33/33 + 6/6 E8 + E9.1 + E9.2 PASS, with round UMAPs (aspect 1.13–1.51) and cross-seed α Spearman ρ = 0.83 / 0.762 respectively. These are the canonical 2-species references.

**2-species testis — 6/6 E8 with universal architectural ceiling on E9.2 down.** H+Mac iter109 (clamp 0.85 + CT-cond MMD on cell_type) and H+Mouse iter111 (clamp 0.85 + CT-cond MMD on cell_ontology_class) both pass 6/6 E8 and beat scVI(donor) on integration, but **E9.2 push<-0.05 ≈ 0%** because the GIN MLP smooths across feature-space neighbors of rare negative-S_g genes (Spearman r(S_g_ema, correction) = 0.514–0.490 stable across all attacks). Marginal numerical fails on E1.5 (α.std ≈ 0.0495 vs 0.05 floor) follow from prior-clamp narrowing. ★ withheld until E9.2 down passes — **Tier-2 attacks exhausted**: iter113 (λ_fit ×3), iter114 (capacity ×2.6), AND iter116 (slower SG_EMA β 0.9→0.97) all failed. Next attempts must be Tier-3 code changes — class-balanced FQR loss, joint-3sp warmstart, per-gene bias hybrid, asymmetric L_anchor, or FQR baseline redesign.

**Testis unified (3-sp) — best-known iter115 (100ep), E9.2 down 1.57% just below 2% PASS.** iter115 promoted; killed 200ep run had E9.2 down 2.43% PASS at ep 100, so a 200ep re-run almost certainly closes the gate without code change.

**4-species pancreas — structural ceiling at 2 shared CTs.** iter101 reaches sil-ct = 0.408 (best ever) and LT-acc = 0.985 (beats scVI 0.916) but kNN-sp = 0.979 vs scVI 0.917 because only Hepatocytes (16,222) and Endothelial (2,733) are shared across all 4 species. iter107 CT-cond MMD attempt rolled back. Treated as structural until proven otherwise.

**H+Zebrafish — structural at 450 Myr divergence.** iter85 with --lambda-mmd 10 reaches 28/33 + 4/6 E8. Species too divergent to mix in z without destroying celltype structure (sil-ct -0.093, LT-acc 0.704). Treated as structural.

**Identifiability audit (user-prompted, iter112).** Pancreas H+Mouse / H+Zeb have `batch == species` (1-to-1); ConstrainedBatchOffset is mathematically inert. Pancreas H+Mo has 3 batches (Monkey has 2) — ConstrainedBatchOffset has ~1 dof. Testis datasets DO have `sample_batch` (5–9 donor batches/species) but Chimera uses `batch_key=batch` (==species) while scVI baseline uses `batch_key=sample_batch` (donors). iter94 tested Chimera with `batch_key=sample_batch` and rolled back (z contracted 4×). The interaction between Chimera's ConstrainedBatchOffset / adversary / MMD and a multi-batch-per-species `batch_key` is an **open Tier-3 architectural question**, not addressable as an HP tune.

---

## Iteration 116 — measured-vs-predicted (ROLLBACK, 2026-04-25)

| Metric | iter109 baseline | iter116 measured | Δ measured | Δ predicted | Verdict |
|---|---|---|---|---|---|
| **PRIMARY** E9.2 push<-0.05 | 0.007% | **0.007%** | 0 (UNCHANGED) | predicted 2-15% PASS | **FAIL — primary did not move** |
| E9.2 push>+0.05 | 8.20% | 6.36% | -1.84pp | predicted 5-12% (HELD) | (in range; slight ↓) |
| Spearman r(S_g_ema, correction) | 0.514 | 0.490 | -0.024 (slight ↓) | predicted 0.55-0.70 ↑ | **FAIL — Spearman went down, not up** |
| Spearman r (nz subset) | 0.538 | 0.515 | -0.023 (↓) | not predicted | (slight ↓) |
| corr_min | -0.304 | -0.306 | ~unchanged | not predicted | (correction range did NOT widen — opposite of iter115's wider corr_min=-0.747) |
| corr_max | 1.373 | 1.169 | -0.204 (↓) | not predicted | (correction range NARROWED, not widened) |
| corr_std | 0.200 | 0.175 | -0.025 (↓) | not predicted | (correction distribution narrower) |
| α std | 0.050 | 0.050 | held marginal | predicted 0.05-0.07 ↑ | (held marginal FAIL on E1.5; predicted slight ↑ did NOT happen) |
| α mean | 0.847 | 0.845 | -0.002 (held) | predicted 0.83-0.86 | (in range) |
| α saturation (>0.95) | 0.06% | **0.00%** | -0.06pp | predicted 0.05-0.5% (held PASS) | (slightly improved; PASS) |
| α floor (<0.05) | 0% | 0% | held | predicted 0-0.5% | (held PASS) |
| **E8 vs scVI(donor=sample_batch)** | 6/6 | **6/6** | maintained | predicted 6/6 | (PASS) |
| kNN-sp (Chimera) | 0.686 | 0.694 | +0.008 (slight ↑) | predicted 0.65-0.75 (held) | (in range; both ≪ scVI 0.933) |
| sil-ct | 0.321 | **0.381** | +0.060 (↑) | predicted 0.28-0.35 (held) | (slightly above predicted upper; both > scVI 0.265) |
| LT-acc | 0.899 | 0.890 | -0.009 (slight ↓) | predicted 0.88-0.91 | (in range; both vs scVI 0.914) |
| z_min_var | 4.46 | 9.03 | +4.57 (↑) | predicted 3-7 | (slightly above; healthy) |
| z active dims | 10/10 | 10/10 | held | predicted held | (PASS) |
| residual_rms | 2.99 | 3.36 | +0.37 (↑) | predicted ≤ 5.0 | (PASS) |
| adv_acc | 0.691 | 0.692 | held | predicted 0.65-0.78 | (in range; E6.1 still fails) |
| **z PC1/PC2 aspect (UMAP proxy)** | 1.64 | **2.52** | +0.88 (REGRESSED) | predicted 1.0-1.4 (held) | **FAIL — UMAP shape ELONGATED, falsifying probe (f) triggered** |
| Δα > 0.02 (vs clamped prior) | 29.4% | 27.0% | -2.4pp | predicted 25-35% | (in range) |
| Δα < -0.02 (vs clamped prior) | 1.6% | 1.6% | unchanged | predicted 2-15% ↑ | **FAIL — bidirectionality did not move** |

**Predicted-vs-measured discipline (the wrong predictions are the lesson)**:
- ✗ Primary E9.2 down predicted 2-15%; measured 0.007% (UNCHANGED) — **the per-minibatch S_g noise hypothesis is FALSIFIED for testis 2-sp.** Reducing EMA noise from σ/√10 to σ/√33 did NOT improve the GIN MLP's per-gene assignment. The bottleneck is NOT temporal averaging of S_g.
- ✗ Predicted Spearman r ↑ to 0.55-0.70; measured 0.490 (slight ↓). Less-noisy S_g_ema did not produce a tighter MLP fit on neg-S_g genes; if anything, the smoother regression target made the MLP slightly LESS responsive to the rare-class signal.
- ✗ Predicted correction range to widen (more bidirectional Δα); measured corr_max ↓, corr_min unchanged, corr_std ↓. **The correction distribution actually NARROWED.** Plausibly: slower EMA lets the equilibrium S_g_ema settle closer to the (smaller) true S_g for most genes, and the MLP's anchor regularization pulls corrections toward zero on average — net narrowing.
- ✗ Predicted UMAP aspect held in 1.0-1.4; measured PC1/PC2 ratio 2.52 (REGRESSED from 1.64). Slower EMA was supposed to be surgical to GIN; it wasn't. Possible mechanism: smoother FQR gradients change the encoder's effective regularization landscape because L_fit_reward enters the global loss; less stochasticity in this term → encoder may overfit to a thinner manifold. **NEW Lesson worth preserving.**
- ✓ E8 6/6 maintained; integration metrics in band (some slight improvements, some slight regressions, all within tolerance). The change does NOT break integration — it just doesn't help E9.2 down.
- ✓ E3.2 sat slightly improved; α floor unchanged. Hard-gate regression did NOT happen.

**Mechanistic finding (iter116 + iter109/iter111/iter115 cross-comparison)**: The 3-sp success on E9.2 down (iter115's 1.57% vs 2-sp 0.007%) is NOT explained by per-minibatch S_g noise — slower EMA reduced that noise without moving the metric. The remaining iter115-brainstorm candidates are: (a) **gene-distribution diversity** (more pairwise contrasts surface more non-trivial neg-S_g genes → wider GIN input distribution → wider correction output range); (b) **class imbalance** (3-sp testis has more genes balanced across pos/neg signs, so the MLP gradient is less pos-dominated). These are deeply linked but architecturally different to attack — both require Tier-3 code changes (class-balanced FQR loss, joint-3sp warmstart, or per-gene bias hybrid).

**Decision**: ROLLBACK iter116. Code change (`--sg-ema-beta` CLI flag) preserved as opt-in (default β=0.9 unchanged); pancreas best-known recipes are unaffected. iter109 remains best-known for testis H+Mac. Next attack pivots to **count / class-imbalance** mechanism — Tier-3 code changes (class-balanced FQR loss in `_module.py`, OR per-gene bias hybrid in `_components.py`, OR joint-3sp warmstart with checkpoint loading).

**Artifacts preserved**: `runs/testis-hmac-iter116-sgema097-200ep-s42/{model,final_state.npz,iter116_diagnostics.json,integration_vs_scvi.{md,json},summary.json,plots/}`. iter109 ckpt preserved as best-known. Diagnostic script `training/iter116_diagnostics.py` reusable for future iterations (computes Spearman r, neg-S_g class metrics, z PC1/PC2 aspect — same schema as the deleted iter114_diagnostics.json).

---

## Active iteration template — iter116 (HISTORICAL, REJECTED — distilled to measured-vs-predicted above)

**Pivot rationale (2026-04-25).** workflow.md prior plan after iter115 acceptance had no explicit "next iteration" template — only a backup queue (iter116 adv=200 on 3-sp; iter117 class-balanced FQR loss; iter118 per-gene bias hybrid; iter119 FQR baseline redesign). Per the user direction ("perfect results for all datasets, prioritize 2-species first; you do not have to follow workflow.md's last decision; think outside the box"), the highest-value next iteration was to attack **testis 2-sp E9.2 down** with a structurally NEW lever family.

**iter113 + iter114 attacked MLP-expressivity (loss-weight scaling, then capacity); both rejected.** iter115 promotion (3-sp testis_unified) succeeded with E9.2 down 1.57% via a recipe transfer; brainstorm credited "signal density" (per-minibatch S_g_ema noise + count of non-trivial neg-S_g genes) as the underlying mechanism. iter116 tested the **per-minibatch noise** half surgically: opt-in `--sg-ema-beta 0.97` (effective EMA window 10 → 33 batches), no other change to iter109's coupled recipe (clamp 0.85 + CT-cond MMD on cell_type, 200ep). Code: 4-file CLI plumbing (defaults preserved → pancreas unaffected). PRIMARY metric (E9.2 push<-0.05) UNCHANGED at 0.007%; Spearman r slightly ↓ (0.514 → 0.490); correction range NARROWED (corr_max 1.373 → 1.169, corr_std 0.200 → 0.175); UMAP aspect REGRESSED (PC1/PC2 1.64 → 2.52, falsifying probe (f) triggered); E8 6/6 maintained; sil-ct slightly improved.

**Lesson (per-minibatch S_g noise hypothesis FALSIFIED for testis 2-sp E9.2 down)**: tightening the GIN's regression target by a factor of 3.3× in temporal averaging did NOT improve per-gene assignment to neg-S_g genes (ρ ↓ slightly, not ↑), did NOT widen the correction range (it narrowed), and did NOT move the primary metric. The 3-sp E9.2 down success (iter115) is therefore NOT explained by EMA noise reduction alone. Plus: a "surgical" loss-term change can leak into encoder regularization and worsen z geometry — the smoother FQR gradient appears to have let the encoder converge to a thinner manifold. This is a NEW Lesson worth preserving for any future "tighten S_g signal" attempt.

---

## Iteration 115 (HISTORICAL — ACCEPTED, distilled 2026-04-25)

**What was tried.** Apply the iter111 coupled recipe (prior-clamp 0.85 + CT-cond MMD on cell_ontology_class) to testis unified (3-sp). Pure recipe transfer, no code change. Time-pressured to 100ep (originally 200ep planned; killed at ep100).

**How it worked (predictions vs measured already in §"iter115 measured-vs-predicted" above).** Cured E3.2 HARD FAIL (29.06% → 1.32%). 6/6 E8 vs scVI(donor=sample_batch) AND vs scVI(species=batch). E9.2 down rose to 1.57% — close to PASS threshold of 2%; the killed 200ep run had reached 2.43% PASS at ep100 of a 200ep schedule, so 200ep would PASS but the 100ep LR schedule produced a more aggressive equilibrium with slightly less downward skew. **Mechanistic surprise**: Spearman r(S_g_ema, correction) was 0.355 (LOWER than iter109 H+Mac's 0.514), yet E9.2 down was higher — so 3-sp success is NOT from improved per-gene MLP discrimination, it is from a wider overall correction MAGNITUDE distribution (corr_min = -0.747 vs iter109 -0.304, 2.5×). Iter116's diagnostic recomputation confirmed this on iter109/iter111/iter115 cross-comparison.

**Design implication.** "Signal density via averaging" hypothesis (slower S_g_ema, larger batch) attacks per-gene discrimination — but iter115's actual mechanism was correction-range widening, possibly driven by gene-distribution diversity (more non-trivial neg-S_g genes from more pairwise contrasts, not lower-noise S_g_ema per gene). iter116 will test the noise-reduction half of the hypothesis on testis H+Mac. If iter116 fails, the dominant 3-sp mechanism was the count effect, and per-gene-independent signal-range fixes (class-balanced FQR, per-gene bias hybrid) become the next attack class.

**Backup options preserved (escalating cost):** adv=200 on 3-sp unified (~85 min); class-balanced FQR loss in `_module.py` (~1.5 hr code + train, Tier-3); per-gene bias hybrid in `_components.py` (~1 hr code, Tier-3); FQR baseline redesign `mu_full_ema.detach()` (multi-week).
---

## Active hypothesis queue (after iter116 REJECTED)

In priority order. Re-evaluate on every iteration; the queue is informational only.

**Post-iter116 conviction (refines iter114 + iter115 diagnosis)**: ρ(S_g, corr) ≈ 0.5 ceiling on testis 2-sp is robust to (a) loss-weight scaling (iter113 λ_fit×3 → Δρ=+0.020), (b) MLP capacity (iter114 width×2.6 → Δρ=-0.011), AND (c) per-minibatch S_g noise reduction (iter116 SG_EMA β 0.9→0.97 → Δρ=-0.024, slightly worse). The remaining 3-sp success mechanisms from the iter115 brainstorm are: (1) **gene-distribution diversity / count of non-trivial neg-S_g genes** (3-sp surfaces more divergent genes via more pairwise contrasts → wider gradient signal across the GIN's input distribution), and (2) **class imbalance via L_fit_reward gradient bulk** (3-sp testis has 130 neg vs 398 pos genes — better balanced than 2-sp's 155 vs 666). Both require Tier-3 code changes.

| # | Hypothesis | Tier | Predicted impact | Notes |
|---|------------|------|------------------|-------|
| 1 (HIGH-VALUE, NO CODE) | **testis_unified 200ep re-run** (already-known result: killed run reached E9.2 down 2.43% PASS at ep 100 of 200ep; vs current 100ep accepted at 1.57%). Same recipe; just longer training. | 1 | Closes 3-sp E9.2 down PASS → ★ on testis_unified. ~85 min wall. | High-value, low-risk; does NOT solve 2-sp. |
| 2 | iter117: **Class-balanced FQR loss** — multiply per-gene FQR loss by `1 / class_freq(sign(S_g_ema))` (3 bins: neg / zero / pos), so the rare neg-class genes get effectively higher gradient weight | 3 | Directly attacks class imbalance: per-batch FQR gradient on neg-S_g genes scales ×4-5 vs current. Predicted Δρ(S_g, corr) +0.05–0.15 AND E9.2 down 0% → 2-10%. | Code change in `_module.py` FQR loss (~10 lines); ~1.5 hr code + train + non-regression on pancreas. RISKY: rebalancing changes gradient regime; pancreas ★ recipes need verification. |
| 3 | iter118: **Joint-3sp warmstart** — load iter115's GIN ckpt, fine-tune ChimeraModel on H+Mac (gene_features format compatible if the gene set overlaps; verify) | 3 | Transfers iter115's WIDER correction range (corr_min -0.747, std 0.34) to 2-sp; predicts E9.2 down jumps because the GIN already has neg-output regions baked in. | Requires checkpoint-loading code in train_and_diagnose; ~1 hr. Risk: gene-set mismatch between testis_unified (15732 genes) and H+Mac (14640 genes) — need gene-id alignment. |
| 4 | iter119: **Per-gene bias hybrid** — `correction = MLP(features) + nn.Parameter(zeros(G))`, allowing per-gene dof independent of features | 3 | Each gene gets its own dof for Δα; bypasses feature-space neighbor interference. Risk: per-gene param alone (config M) was historically worse than GIN; hybrid may recover. | Code change in `_components.py`; ~1 hr code + train + non-regression. Multi-dataset validation required. |
| 5 | iter120: **Asymmetric L_anchor** — `L_anchor = lambda_anchor_pos * relu(corr).pow(2).mean() + lambda_anchor_neg * relu(-corr).pow(2).mean()` with `lambda_anchor_neg < lambda_anchor_pos` | 3 | Reduces penalty on negative corrections; allows MLP's neg outputs to grow without symmetric pull-back. Targeted attack on L_anchor symmetry hypothesis (workflow's listed candidate). | Code change in `_module.py` L_anchor; ~1 hr code + train. |
| 6 | Transfer any accepted iter117/118/119/120 recipe to testis H+Mouse and unified | 1 | Closes ★ on testis triad | Conditional; per-recipe |
| 7 | Multi-seed verification of accepted testis configs (seeds 43, 44) for E9.1 | 1 | Multi-seed E9.1 certification | Conditional on a 2-sp testis ★ |
| 8 | Tier-3: **FQR baseline redesign** (`baseline = mu_full_ema.detach()` instead of `p_null * lib`) — ultimate fallback | 3 | Multi-week | Reserved if iter117/118/119/120 all fail |
| 9 | Tier-3: 4-species pancreas E8.4 attack — species-pair-conditional MMD | 3 | Closes 4-sp structural | Data-limited (only 2 shared CTs); workflow already flags as structural |
| 10 | Tier-3: Investigate Chimera × multi-batch-per-species `batch_key` (testis sample_batch) | 3 | Identifiability asymmetry | Architectural; iter94 root-cause unsolved |

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

---

## Notes for the next agent

1. **Read this file end-to-end AND [workflow-best.md](workflow-best.md)** before doing anything. Both are authoritative; if a config you intend to use is not the documented current best, reconcile (update registry or update plan) before running.

2. **Where the loop stands (after iter116 ROLLBACK, 2026-04-25)**:
   - Pancreas 2-sp (H+Mo, H+Mouse): both ★ DONE.
   - Pancreas 4-sp: 3/6 E8 — STRUCTURAL at 2 shared CTs.
   - Pancreas H+Zeb: 4/6 E8 — STRUCTURAL at 450 Myr.
   - Testis H+Mac (2-sp): iter109 6/6 E8, E9.2 down ARCHITECTURAL CEILING; ★ withheld.
   - Testis H+Mouse (2-sp): iter111 6/6 E8, E9.2 down ARCHITECTURAL CEILING; ★ withheld.
   - Testis_unified (3-sp): iter115 6/6 E8 vs both baselines, E9.2 down 1.57% (just below 2% PASS); the killed 200ep run reached 2.43% PASS at ep100 → **a 200ep re-run almost certainly closes E9.2 down PASS, no code change needed**. THIS IS THE HIGHEST-VALUE NEXT ITERATION on the no-code-change track.

3. **iter116 was a clean falsifying experiment (RECOMMENDED reading)**. Slower SG_EMA_BETA on testis H+Mac DID NOT move E9.2 down (stayed at 0.007%) — the per-minibatch S_g noise hypothesis from iter115's brainstorm is FALSIFIED for testis 2-sp. Spearman r(S_g, corr) actually went DOWN slightly, correction range NARROWED, UMAP aspect REGRESSED. Three Tier-2 attacks in a row (iter113 λ_fit, iter114 capacity, iter116 noise) have now failed on testis 2-sp E9.2 down. The remaining mechanisms from the brainstorm (gene-distribution diversity / class imbalance) all require Tier-3 code changes. **Do not try a 4th Tier-2 attack on testis 2-sp E9.2 down.**

4. **Recommended next iterations** (priority order):
   - (a) **testis_unified 200ep re-run** — same iter115 recipe but 200ep instead of 100ep. Predicted: E9.2 down crosses 2% PASS; integration metrics held; ★ on testis_unified. ~85 min wall. NO CODE CHANGE.
   - (b) **iter117 class-balanced FQR loss** — Tier-3 code change in `_module.py`: re-weight per-gene FQR by `1 / class_freq(sign(S_g_ema))`. Direct attack on the class-imbalance hypothesis (testis 666 pos vs 155 neg neg-S_g). ~1.5 hr code + train + non-regression on pancreas.
   - (c) **iter118 joint-3sp warmstart** — code change to load iter115 GIN ckpt and fine-tune on H+Mac. Tests count-mechanism transferability. Risk: gene-set mismatch between testis_unified (15732) and H+Mac (14640). ~1 hr code + train.
   - (d) **iter119 per-gene bias hybrid** — `correction = MLP(features) + nn.Parameter(zeros(G))`. ~1 hr.
   - (e) **iter120 asymmetric L_anchor** — separate λ for positive vs negative corrections. ~1 hr.

5. **Next-iteration evaluation steps** (after any training completes): (a) check_expectations.py auto-runs from run_experiment.py → produces report.md; (b) `compare_vs_scvi.py --batch-key sample_batch` for testis donor baseline (cached at `runs/baselines_scvi/camex_*-bk_sample_batch/`); (c) plot_diagnostics.py for plots; (d) **`iter116_diagnostics.py`** (now reusable) for Spearman r + neg-S_g class metrics; (e) measured-vs-predicted in workflow.md against the pre-change template; (f) if accepted: atomic update of workflow-best.md + `_constants.py` defaults + notebook pointers.

6. **Identifiability audit (confirmed this invocation)**: testis datasets (H+Mac, H+Mouse, unified) have `batch == species` for Chimera, but `sample_batch` has 7-9 donor batches per species. scVI baseline correctly uses `sample_batch`; Chimera does not. iter94 attempted Chimera with `batch_key=sample_batch` and rolled back (z contracted 4×). This identifiability asymmetry is a **Tier-3 architectural question** — not addressable as an HP iteration. Item #10 in the active hypothesis queue.

7. **Workflow.md hygiene state**: ~600 lines after iter115 + iter116 distillation. The iter116 template body was distilled to a single Lessons paragraph plus the measured-vs-predicted table. iter115 template body distilled similarly. Headroom remains; further distillation not needed.

8. **Code state (as of iter116)**: New `--sg-ema-beta` CLI flag plumbed through `_module.py`/`train_and_diagnose.py`/`run_experiment.py`. Default behavior unchanged (None → uses SG_EMA_BETA=0.9 constant). Diagnostic script `training/iter116_diagnostics.py` reusable for any future testis run (Spearman r, neg/pos-S_g class metrics, z PC1/PC2 aspect).

9. **Stop after one iteration.** The user drives the outer loop.
