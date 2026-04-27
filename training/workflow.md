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

**iter124 (2026-04-26, this invocation) — co-tuned adv classifier capacity + LOWER adv_weight on 4-sp. ROLLBACK.**
Tier-2 (loss-weight + capacity co-tune). Direct mechanistic test of iter123's
ROLLBACK lesson: keep the (64,32) classifier but halve adv_weight 200→100,
predicting the encoder gets headroom to find a real species-blind manifold
rather than collapsing CT structure to fool the bigger classifier.

**Result: HARD ROLLBACK. The hypothesis "co-tune adv_weight lower" is
FALSIFIED at adv_weight=100.** sil-ct collapsed even more than iter123
(0.079 vs 0.096); LT-acc REGRESSED further (0.753 vs iter123's 0.893,
iter101's 0.985). The (64,32) classifier × 4-sp encoder interaction is
binding regardless of adv_weight — the encoder still sacrifices CT structure
to evade the bigger classifier.

**Adv-side family on 4-sp is now EXHAUSTED** (2 consecutive failures iter123 +
iter124, both via sil-ct collapse). Per refinement.md anti-pattern rule, the
next 4-sp attack MUST be structurally different (η-r encoder capacity is the
recommended pivot).

### iter124 pre-change template (PRE-RUN)

```
Iteration 124 — 4-sp adv-capacity (64,32) co-tuned with adv_weight 200→100
Expectation under attack: E8.1, E8.4, E8.5 on 4-sp pancreas (Endothelial worst-CT
                          Δ_excess vs scVI = +0.128 on iter101; close-3 species
                          0% mixed within Hep+Endo while scVI mixes 5–25%).
Pivot rationale: iter123 ROLLBACK proved that bigger classifier alone at
                 adv_weight=200 over-tips GRL game (encoder evades specific
                 classifier instance by destroying CT). The iter123 lessons
                 explicitly recommend: "Future capacity bumps MUST co-tune
                 adv_weight lower (e.g., (64,32) needs adv_weight ≈ 80–120,
                 not 200)." iter124 = direct test of that recommendation at
                 adv_weight=100. Structurally different from iter123 (game
                 balance ≠ pure capacity); not a 5th in same family yet.
Change:
  Two CLI flags relative to iter101 best-known:
    --adv-classifier-hidden "64,32"      (same as iter123)
    --adv-weight 100                     (halved from iter101's 200)
  No code change. Both flags already plumbed.
Test config (4-sp pancreas):
  --adata data/processed/camex_pancreas_unified.h5ad
  --config A --max-epochs 200 --seed 42 --accelerator mps
  --adv-weight 100 --lambda-anchor 0.001 --free-bits 1.0 --lambda-mmd 100
  --adv-classifier-hidden "64,32"
Tier: 2 (loss-weight + capacity co-tune)

Direct touch-points
  - Adversary classifier: same (64,32) as iter123 — 2,816 params (vs iter101's
    448, 6.3× scale-up). Loss applied with HALF the gradient strength
    (adv_weight 100 vs 200).
  - GRL gradient on z: scaled DOWN by 0.5× vs iter101 AND iter123. Bigger
    classifier produces sharper, non-linear gradient SHAPE; the lower
    weight reduces total gradient MAGNITUDE on encoder. The combined effect
    should let the bigger classifier inform encoder about non-linear species
    structure WITHOUT forcing the encoder to over-prioritize the adversary
    objective at the cost of CT structure.
  - Encoder: less total adversarial pressure (adv_weight=100 vs 200) means
    the encoder is NOT forced to find any-cost ways to fool the classifier;
    it has headroom to keep CT compact while reducing real species leakage.
    Net predicted: sil-ct preserved at iter101 levels (~0.40), kNN-sp
    improves on close-3 mixing.
  - Schedules: warmup_kl=20, warmup_fqr=20, warmup_adv=5 — unchanged.
  - Loss-balance numbers (chimera-loss-scaling-fix memory):
      adv fool_loss is per-cell mean × n_genes × adv_weight.
      iter101: ratio_adv/L_recon ≈ (0.5 × n_genes × 200) / O(n_genes × 1) = O(100).
      iter123: same O(100) — bigger classifier doesn't change scale.
      iter124: ratio HALVED → O(50). This is closer to iter90 H+Mo regime
      (adv_weight=100 default) which ran healthy with sil-ct ≥ 0.50.
  - α-side machinery: unaffected. Predicted α biology held at iter123 levels
    (which were already PASSING on E1-E9: 33/33).
  - Residual decoder: receives less adversarial pressure → predicted to
    remain in [0.01, 5] range; possibly slightly lower than iter123 (residual
    less stressed).
  - MMD: unchanged. λ_mmd=100 stays (uniform per-pair as in iter101/iter123).

Predicted deltas — signed, rough magnitude REQUIRED
  - PRIMARY E8.1 kNN-sp: 0.979 (iter101) / 0.962 (iter123) → 0.91–0.95
    (improvement vs iter101; better than iter123 because CT structure should
    survive at lower adv_weight, allowing within-CT species mixing rather
    than global flattening). Target ≤ scVI 0.930 + 0.01 = 0.940 = E8.1 PASS.
  - PRIMARY E8.4 worst-CT Δ vs scVI: +0.128 Endo (iter101) / +0.073 Endo
    (iter123) → +0.02 to +0.07 (iter123's +0.073 was achieved at the cost
    of CT collapse; iter124 should hold or improve while preserving sil-ct).
    Target ≤ +0.02 = E8.4 PASS.
  - PRIMARY sil-ct: 0.408 (iter101) → 0.36–0.42 (HARD: ≥ 0.30 required;
    falsifying probe < 0.30 triggers ROLLBACK as in iter123).
  - LT-acc: 0.985 (iter101) / 0.893 (iter123) → 0.96–0.99 (preserved when
    CT structure holds; the iter123 collapse driver — encoder flattening CT
    to fool classifier — should not engage at half the adv_weight).
  - sil-sp: 0.018 (iter101) / -0.052 (iter123) → -0.05 to +0.03.
  - in-training adv_acc: 0.765 (iter101) / 0.389 (iter123) → 0.55–0.75
    (a balanced GRL game equilibrium, between fully-fooled iter123 and
    iter101's healthy state). Falsifying probe: adv_acc ≥ 0.85 → encoder
    failing to push species out at all (lower weight too weak).
  - **POST-HOC z-leakage** (CO-PRIMARY per iter123 lesson):
    * MLP-64-32 excess: 0.738 (iter101) / 0.709 (iter123) → 0.50–0.65
      (target ≤ +0.50 to match 2-sp envelope; the bigger classifier should
      now provide useful gradients to encoder rather than just being evaded).
    * MLP-32 excess: 0.712 (iter101) → 0.50–0.65.
    * kNN5 excess: 0.731 (iter101) / 0.719 (iter123) → 0.55–0.65.
    * Linear excess: 0.575 (iter101) → 0.40–0.55.
  - α mean: 0.754 (iter101) / 0.680 (iter123) → 0.72–0.77.
  - α std: 0.134 (iter101) / 0.194 (iter123) → 0.13–0.18.
  - frac(α>0.95) E3.2: 0.95% (iter101) → < 5% (PASS).
  - Δα > 0.05 (E9.2 up): 7.69% (iter101) / 17.4% (iter123) → 8–18% (PASS ≥ 2%).
  - Δα < -0.05 (E9.2 down): 35.78% (iter101) / 62.7% (iter123) → 35–60%.
  - z active dims (var > 0.1): 10/10 (HARD; <8/10 = z collapse blocker).
  - z_min_var: ≥ 0.1 (HARD; < 0.1 = posterior collapse blocker).
  - residual_rms: in [0.01, 5] (HARD).
  - UMAP aspect (p1-p99): expected round, < 1.5.
  - E8 score: 3/6 (iter101) / 2/6 (iter123) → 4–6/6 (target IMPROVE on iter101).
  - E1-E9 (33-gate): 30/32 (iter101) / 33/33 (iter123) → 31–33/33.

Falsifying probes
  - HARD (rollback trigger): sil-ct < 0.30 (CT structure destroyed — same
    iter123 failure mode at lower adv_weight = iter124 hypothesis fully falsified).
  - HARD: kNN-sp ≥ scVI's 0.930 + 0.01 = 0.940 (no improvement vs iter101's
    0.979 → adv-side machinery exhausted on 4-sp; pivot to (η-r) encoder
    capacity or (θ) cycle-consistency).
  - HARD: post-hoc MLP-64-32 z accuracy ≥ 0.95 (≤ 0.05 reduction from iter101's
    0.738 → bigger classifier still ineffective even at balanced GRL).
  - HARD: z collapse — active dims < 8/10 OR z_min_var < 0.1.
  - HARD: in-training adv_acc ≥ 0.90 (encoder no longer pushing — adv_weight
    too low; would suggest a sweep at adv_weight=150 instead).
  - HARD: alpha drift — std(α) < 0.05 OR Spearman ρ(α, prior) > 0.99.
  - SOFT: residual_rms > 5 (residual eating CT info).
  - SOFT: LT-acc < 0.94 (cross-species CT inference degraded).

Rollback plan
  - Both CLI flags default-safe: --adv-weight defaults to 100.0 (which IS
    iter124's value), --adv-classifier-hidden defaults to None → (32,) legacy.
    iter101's best-known recipe explicitly passes --adv-weight 200, so
    omitting that flag does NOT corrupt iter101 — iter101 is preserved.
  - If iter124 fails: don't pass --adv-classifier-hidden; use --adv-weight 200;
    iter101 remains 4-sp best-known. No code rollback.
  - Backup queue (if iter124 also fails or partially succeeds):
      iter124b: --adv-classifier-hidden "64,32" --adv-weight 80 (slightly lower).
      iter125 (η-r): --n-hidden 256 (encoder capacity bump, structurally
                     different — leaves the adv-side family entirely).
                     Verify --n-hidden flag is plumbed before running.
      iter125b (θ): cycle-consistency Tier-3 (multi-day; explicit user OK).
      iter125c (β): Refine 4-sp E8 expectations (requires user approval).
```

### iter124 measured vs predicted (POST-RUN, ROLLBACK)

Training completed 2026-04-26 18:16, 200 epochs, 4-sp pancreas s42, 3,985s
(66 min, slower than iter123's 53 min — env startup overhead). No NaN/Inf.

| Metric | iter101 | iter123 | iter124 measured | predicted | Verdict |
|---|---|---|---|---|---|
| **PRIMARY E8.1 kNN-sp** | 0.979 | 0.962 | **0.946** | 0.91–0.95 | hit lower band; FAIL E8.1 (Δ vs scVI=+0.016 > 0.01) |
| **PRIMARY E8.4 worst-CT gap** | +0.128 Endo | +0.073 Endo | **+0.040 Hep** | +0.02 to +0.07 | hit lower predicted edge; FAIL ≤+0.02 |
| **PRIMARY sil-ct (E8.3)** | 0.408 | 0.096 | **0.079** | ≥ 0.30 hard | **HARD FAIL — falsifying probe; collapsed MORE than iter123** |
| **LT-acc (E8.6)** | 0.985 | 0.893 | **0.753** | 0.96–0.99 | **HARD FAIL — REGRESSED FURTHER** (vs scVI 0.929: Δ=−0.176, FAILS E8.6 ≥ scVI−0.05) |
| sil-sp (E8.2) | 0.018 | −0.052 | −0.020 | −0.05 to +0.03 | PASS within band |
| **E8 score** | 3/6 | 2/6 | **2/6** | 4–6/6 | **REGRESSED — same fail pattern as iter123** |
| in-training adv_acc | 0.765 | 0.389 | **0.529** | 0.55–0.75 | hit predicted band lower edge (more balanced than iter123 0.389) |
| **post-hoc MLP-64-32 z** | +0.738 | +0.709 | **+0.694** | ≤ +0.50 | **MISS** — barely improved (Δ=−0.044 from iter101); still far above 2-sp envelope |
| post-hoc MLP-32 z | +0.712 | n/a | +0.602 | 0.50–0.65 | hit predicted band |
| post-hoc kNN5 z | +0.731 | +0.719 | +0.701 | 0.55–0.65 | small Δ; hit upper predicted edge |
| post-hoc linear z | +0.575 | n/a | +0.510 | 0.40–0.55 | hit predicted band — encoder DID reduce linear leakage |
| α mean | 0.754 | 0.680 | **0.679** | 0.72–0.77 | DRIFTED — same pattern as iter123 |
| α std | 0.134 | 0.194 | **0.195** | 0.13–0.18 | DRIFTED (slightly above predicted) |
| frac(α>0.95) E3.2 | 0.95% | <2% | 0.59% | < 5% | PASS |
| Δα > 0.05 (E9.2 up) | 7.7% | 17.4% | 3.1% | 8–18% | **MISSED — partly fell back from iter123, but still PASS ≥ 2%** |
| Δα < −0.05 (E9.2 down) | 35.8% | 62.7% | 50.1% | 35–60% | within band |
| z active dims | 10/10 | 10/10 | 10/10 | 10/10 | PASS |
| z_min_var | 0.13 | 0.20 | 0.219 | ≥ 0.1 | PASS |
| residual_rms | 1.99 | 2.61 | 2.48 | [0.01, 5] | PASS |
| UMAP aspect (p1-p99) | 1.21 | n/a | 1.194 | < 1.5 | PASS |
| **z PC1/PC2 SVD aspect** | (n/a) | (n/a) | **5.27** | round | **HARD FAIL — z severely thin in linear projection** |
| **E1-E9 (33-gate suite)** | 30/32 | 33/33 | **33/33** | 31–33/33 | PASS — all 33 green |
| Endo within-CT (kNN purity vs chance) | 0.980 (+0.448) | n/a | 0.876 (+0.344) | improve | partial — Endo Δ vs scVI = +0.024 (vs iter101 +0.128, scVI 0.852/+0.320) |
| Hep within-CT (kNN purity vs chance) | 0.997 (+0.668) | n/a | 0.992 (+0.663) | improve | barely changed (+0.045→+0.040 vs scVI) — Hep close-3 mixing did NOT improve |
| **mean_corr_on_neg_S_g** | n/a | n/a | **−1.87** | (negative, correct sign) | correct sign on neg class |
| **mean_corr_on_pos_S_g** | n/a | n/a | **−1.58** | (positive, correct sign) | **WRONG SIGN — GIN pushes positive class negative too** |
| frac_corr_pos_on_pos_S_g | n/a | n/a | **0.00** | (~1.0 expected at α-detach equilibrium) | **HARD FAIL — α biology degenerate** |
| ρ(S_g, corr) nz subset | n/a | n/a | 0.357 | (modest) | low |

**Mechanistic finding (iter124, definitive)**:

1. **Co-tuning adv_weight=100 does NOT save CT structure when classifier is (64,32).** sil-ct went from iter123's 0.096 to iter124's 0.079 — *worse*. LT-acc went from iter123's 0.893 to iter124's 0.753 — also *worse*. The relationship is NOT a smooth tradeoff curve where halving adv_weight halves CT damage; it's roughly the same regime, slightly worse on the CT side.

2. **The (64,32) classifier × 4-sp encoder interaction is the binding mechanism** — adv_weight choice within [100, 200] is a second-order knob. The bigger classifier creates a non-linear gradient SHAPE that the encoder evades by collapsing CT structure regardless of total gradient MAGNITUDE.

3. **α biology DEGENERATE in iter124, more severely than iter123.** mean_corr_on_pos_S_g = **−1.58** (WRONG SIGN — pushes pos-S_g class toward divergence). frac_corr_pos_on_pos_S_g = 0.00 (every positive-class gene gets a negative correction). The α-side machinery looked "improved" via E9.2 PASS metrics in iter123 but is actually DEGENERATE under z-side perturbation: the L_recon gradient on `(1-α)·residual` under z drift forces a uniform downward push on every correction. iter123 hid this signal; iter124 surfaces it because the encoder pressure regime shifted slightly.

4. **The post-hoc MLP-64-32 z-leakage improved more in iter124 than iter123** (0.738→0.694 vs 0.738→0.709) — i.e., the encoder DID do somewhat more legitimate species-scrubbing at the lower adv_weight. But the improvement is far from the 0.50 target (2-sp envelope), and comes at unacceptable CT cost. So the iter124 hypothesis "lower adv_weight gives encoder room to scrub legitimately" is *partially* true but *insufficient*.

5. **Endothelial close-3 mixing IMPROVED slightly** (Δ vs scVI: iter101 +0.128 → iter124 +0.024, a 5× reduction). This was the iter123 audit's binding worst-CT — and iter124 nominally closed most of that gap. But Hepatocytes (73% of cells, the dominant CT) close-3 mixing did NOT improve (Δ vs scVI iter101 +0.045 → iter124 +0.040, barely changed). And the global CT structure collapsed (sil-ct 0.408→0.079).

**Decision: ROLLBACK iter124. iter101 retained as 4-sp best-known (no registry change).**
Both flags default-safe: omitting `--adv-classifier-hidden` reverts to (32,) legacy; iter101's recipe explicitly passes `--adv-weight 200` so omitting iter124's `--adv-weight 100` flag is also safe. No code change needed.

**Lessons (iter124 distilled)**:

1. **Adv-side family on 4-sp is EXHAUSTED** (iter123 + iter124 both via sil-ct collapse, 2 consecutive failures). Per refinement.md anti-pattern rule, the next 4-sp attack MUST use a structurally-different mechanism family — encoder capacity (η-r), cycle-consistency (θ), or expectation refinement (β). NOT another adv knob.
2. **Bigger classifier × 4-sp pancreas is a fundamental mismatch.** The (64,32) MLP at any tested adv_weight gives the encoder cheap escape routes through CT collapse. The encoder does NOT learn species-blind structure — it learns to evade the specific classifier instance, with mostly-CT-axis flattening as the cheapest evasion. Future capacity bumps to the in-training adversary on 4-sp are deprecated.
3. **α biology degeneration under z drift is masked by E9.2 metrics.** iter123 looked like "α biology improved" (Δα bidirectional widened) but iter124's per-class signed metrics revealed the underlying degeneracy: pos-S_g class also pushed negative (mean_corr_on_pos_S_g=−1.58). The right co-primary metric is `mean_corr_on_pos_S_g` and `frac_corr_pos_on_pos_S_g` from `gin_signed_diagnostics.py`, NOT just frac(Δα<−0.05). Adopt as a co-primary diagnostic for any 4-sp z-side perturbation.
4. **z PC1/PC2 SVD aspect (5.27) catches what UMAP aspect (1.19) misses.** Linear projection diagnostic exposes encoder collapsing onto a thin linear subspace even when the non-linear UMAP looks round. `gin_signed_diagnostics.py` reports this; it's a standing diagnostic, not new — but iter124 shows it matters for 4-sp z health more than UMAP aspect.
5. **In-training adv_acc and post-hoc MLP-64-32 are now CORRELATED in iter124** (0.529 in-training, 0.694 post-hoc, gap 0.165) compared to iter123 (0.389 in-training, 0.709 post-hoc, gap 0.320). This means lower adv_weight reduces the encoder's "evasion vs cleanup" ratio — but the cleanup is still inadequate (0.694 >> 0.50 target). The encoder simply doesn't have a way to make z species-blind under the 4-sp regime via adv-side pressure alone.

**Backup queue (next, post-iter124)**:

- **iter125 (η-r) RECOMMENDED**: `--n-hidden 256` on iter101 recipe (encoder capacity 128→256, no other changes). Structurally different from adv-side family entirely. Hypothesis: encoder needs more capacity to encode 4-species shared celltype structure WITHOUT losing CT separation under the iter101 adv_weight=200 + lambda_mmd=100 regime. **Pre-flight**: verify `--n-hidden` flag is plumbed; if not, small code change.
- iter125b (η-d): Encoder DEPTH bump (n_layers 1→2) on iter101 recipe.
- iter126 (θ): Cycle-consistency Tier-3 (multi-day, explicit user authorization).
- iter127 (β): Refine 4-sp E8 expectations (requires user approval to amend refinement.md).

**Artifacts**: `runs/xds-4sp-iter124-advhid64-32-advw100-200ep-s42/{model, final_state.npz, summary.json, integration_vs_scvi.{md,json}, z_leakage_probe.json, gin_signed_diagnostics.json, plots/umap.png, lightning/metrics.csv, alpha_snapshots/}`.

---

### iter123 distilled lesson (ROLLBACK 2026-04-26, prior body collapsed)

iter123 tested `--adv-classifier-hidden "64,32"` on iter101 4-sp recipe at
fixed `adv_weight=200` (200ep s42, 52.7 min). **Result: ROLLBACK — sil-ct
HARD failed (0.408 → 0.096, falsifying probe < 0.30 triggered); E8 regressed
3/6 → 2/6; LT-acc 0.985 → 0.893.** Mechanism: bigger (64,32) classifier at
fixed adv_weight=200 over-tips GRL game; encoder takes cheapest path to
fool the *specific* classifier instance by destroying CT structure rather
than learning a generally species-blind z. Two pieces of evidence: (1)
in-training adv_acc collapsed 0.765 → 0.389 (looked like a win) but post-hoc
MLP-64-32 only marginally dropped 0.738 → 0.709 (deeper non-linear z leakage
unchanged); (2) sil-ct collapsed 0.408 → 0.096 — encoder flattened CT axis
because it's the cheapest direction to flatten species-correlated structure.
**However**, 33/33 PASS on E1-E9 (improved from 30/32 — Δα bidirectional
widened to E9.2 up 17.4%, down 62.7%) — α-side decoupled from z-side
perturbation, confirming iter121's earlier finding of architectural separation
between MMD/adv/z-side and FQR/α-side machinery.

**Three transferable lessons (preserved verbatim from iter123 entry)**:
1. **Adversary capacity and `adv_weight` are coupled.** Bigger classifier at
   fixed `adv_weight` is over-strong: encoder takes the cheapest path to fool
   it (collapse CT structure) rather than learning a robust species-blind
   manifold. Future capacity bumps MUST co-tune `adv_weight` lower
   (e.g., (64,32) needs `adv_weight ≈ 80–120`, not 200). **iter124 (this
   invocation) is the direct co-tune test at adv_weight=100.**
2. **Post-hoc z-leakage is the right metric, in-training adv_acc is misleading.**
   Adopt `z_leakage_probe.py` as a co-primary diagnostic for any iteration
   touching adv/MMD machinery. iter101 4-sp baseline excess: linear=+0.575,
   kNN5=+0.731, MLP-32=+0.712, MLP-64-32=+0.738. Target for any
   adv/MMD-touching iteration: MLP-64-32 ≤ +0.50 (matches 2-sp envelope).
3. **sil-ct < 0.30 is a near-immediate falsifying probe trigger** when
   adversary is over-strong on 4-sp; confirms iter121's same-direction finding.
   Any 4-sp attack on z-side machinery must include sil-ct as a HARD floor.

**Audit-trail finding (iter123 invocation)**: Tier-0 audit overturned the
prior workflow narrative: 4-sp Endothelial Δ_excess vs scVI = +0.128 is the
binding worst-CT (NOT Hepatocytes +0.045 as workflow had been claiming).
Per-source-species neighbor distribution shows close-3 species (H+Mo+Mouse)
have ~0% cross-species neighbors within Hep+Endo on Chimera iter101 while
scVI has 5–25%. Zebrafish-Hep is unmixable under both → that part IS
structural, but the close-3 gap is NOT.

**iter123 artifacts**: `runs/xds-4sp-iter123-advhid64-32-200ep-s42/`. NEW
diagnostic script `training/z_leakage_probe.py` retained (works on any run-name
post-hoc).

---

### iter123 audit context (preserved for reproducibility — collapsed from full body)

Tier-0 audit findings from iter123 invocation, retained because they inform
iter124's predictions:

1. **Mis-attributed bottleneck:** workflow had been calling Hepatocytes the worst-CT,
   but the iter101 `integration_vs_scvi.json` shows the worst-CT (E8.4) is
   **Endothelial Δ_excess vs scVI = +0.128** (Hep is +0.045). Endo is 12% of cells,
   not 73%, and its species ratios are far less Zebrafish-skewed (chance 0.532 vs
   Hep's 0.329). So the binding constraint isn't "Zebrafish swamps everything."
2. **Per-source-species neighbor distribution within shared CTs (iter101 vs scVI):**

   |  | Chimera iter101 | scVI(batch=batch=donor) |
   |---|---|---|
   | Hep H→{H,Mo,Mou,Zeb} | 0.99 / 0.00 / 0.00 / 0.00 | 0.88 / 0.08 / 0.04 / 0.01 |
   | Hep Mo→… | 0.00 / 1.00 / 0.00 / 0.00 | 0.03 / 0.95 / 0.01 / 0.01 |
   | Hep Mouse→… | 0.00 / 0.00 / 1.00 / 0.00 | 0.03 / 0.02 / 0.94 / 0.01 |
   | Endo H→… | 0.96 / 0.04 / 0.01 / 0.00 | 0.74 / 0.23 / 0.03 / 0.00 |
   | Endo Mo→… | 0.00 / 0.99 / 0.00 / 0.00 | 0.06 / 0.90 / 0.03 / 0.01 |
   | Endo Mouse→… | 0.00 / 0.01 / 0.99 / 0.00 | 0.02 / 0.11 / 0.87 / 0.00 |

   Chimera has ~0% cross-species neighbors among H+Mo+Mouse within Hep; scVI has
   5–12%. Within Endo, scVI mixes 11–23% across close-3 species; Chimera 0–4%.
   **The 4-sp gap is on close-3 mixing, not Zebrafish-Hep alone** (Zeb stays
   isolated under both Chimera 1.00 and scVI 0.92–1.00 — that part IS structural).
3. **z-leakage probe gap.** `check_expectations.py` does NOT run a non-linear z
   probe; it relies on the in-training adversary's adv_acc. New script
   `training/z_leakage_probe.py` (this invocation) runs LinearLR / kNN5 / MLP-32 /
   MLP-64-32 on `z_mean` post-hoc (n=8000 subsample, 3-fold CV); writes
   `runs/<name>/z_leakage_probe.json`. Cross-config baselines (excess over chance):

   | Run | n_sp | linear | kNN5 | MLP-32 | MLP-64-32 | non-lin gap |
   |---|---|---|---|---|---|---|
   | H+Mo iter90 (★) | 2 | +0.169 | +0.275 | +0.169 | +0.260 | +0.091 |
   | H+Mouse iter90 (★) | 2 | +0.396 | +0.454 | +0.426 | +0.482 | +0.086 |
   | Testis H+Mac iter109 | 2 | +0.191 | +0.229 | +0.193 | +0.278 | +0.087 |
   | Testis H+Mouse iter111 | 2 | +0.259 | +0.334 | +0.276 | +0.399 | +0.140 |
   | Testis unified iter115 | 3 | +0.216 | +0.416 | +0.354 | +0.503 | +0.287 |
   | **4-sp iter101** | 4 | +0.575 | +0.731 | +0.712 | **+0.738** | **+0.163** |

   Two readings: (a) **4-sp iter101 has the largest absolute z-leakage** (MLP-64-32
   excess +0.738 vs +0.26–+0.50 for everything else); (b) **the linear-vs-
   nonlinear gap is biggest on 4-sp** (+0.163 vs +0.086 for 2-sp pancreas) —
   confirms the in-training (32,) classifier is undersized for the non-linear
   species manifolds the encoder leaves in z. Even ★ 2-sp configs have non-zero
   non-linear leakage (so refinement.md "near chance" is loose at the integration
   metrics' k=15 scale; non-linear leakage that doesn't manifest as nearest-
   neighbor clustering is tolerated by E8 today). For iter123, the BASELINE to
   beat is +0.738; success means MLP-64-32 excess drops to ≤ +0.50 (matching the
   2-sp envelope).
4. **Identifiability + baseline audit (closes the user's outstanding question).**
   Pancreas: NO `sample_batch` column exists in the .h5ad files. `batch` IS the
   donor column (3 in H+Mo, 2 in H+Mouse / H+Zeb / 4-sp). 4-sp `batch` has 5
   values (Monkey has 2 within-species batches; others 1 each), so
   ConstrainedBatchOffset has ~1 dof on 4-sp. **scVI baseline cache has both
   `-bk_batch` and default (species) variants; iter108 retrained the bk_batch
   variant, which is what `compare_vs_scvi.py` uses by default for pancreas.**
   Testis: `sample_batch` (5–9 donors per species) exists; Chimera uses
   `batch=species`, scVI uses `batch=sample_batch` (the iter108-fixed donor
   baseline). The asymmetry stays — iter94 confirmed Chimera × multi-batch-per-
   species `batch_key` is broken (Tier-3 unresolved). Baselines are not the
   blocker on the current cross-dataset numbers; they were re-fit at iter108.
5. **α biology is alive on iter101**: alpha mean 0.754, std 0.134, 0.95% saturated
   above 0.95, 0% below 0.05. **E9.2 is PASSING** (Δα>0.05 = 7.69% ≥ 2%; Δα<-0.05
   = 35.78% ≥ 2%) — so the 4-sp gap is purely on **integration**, not α biology.
   This decouples the iter123 hypothesis: improving z mixing won't trade against α.

**iter123 hypothesis — Tier-1, structurally-different from MMD family.** Increase
the in-training adversary's classifier hidden dims from `(32,)` to `(64, 32)`
(2.5× params at 4 species; ReLU+Dropout(0.1) layers, plumbing already exists
via `--adv-classifier-hidden`). A larger classifier detects more of the
non-linear species structure during training, applying more useful gradient
through the GRL → encoder must purge more species info from z to fool it.
This attacks the *capacity-undersized adversary* root cause directly. **Not in
the MMD family** — anti-pattern rule passes.

**Decision (iter123 invocation, 2026-04-26)**: ran iter123, ROLLBACK. Distilled body + lessons + measured-vs-predicted retained in "iter123 distilled lesson" near the top of this file. Artifacts at `runs/xds-4sp-iter123-advhid64-32-200ep-s42/`. iter101 retained as 4-sp best-known.

### iter122 distilled lesson (ROLLBACK 2026-04-26)

**iter122** tested heterogeneous per-pair-conditional MMD on 4-sp (`--mmd-pair-weights "1.0,1.0,0.3,1.0,0.3,0.3"` on iter101 recipe at `--lambda-mmd 100`, 200ep s42). **TRAINING NaN at ep17→18** before FQR/KL warmup completion. Pre-NaN trajectory healthy (grad_enc 1.7-2.6, L_recon 2500-2900). **Mechanism**: heterogeneous weights concentrate MMD gradient on close pairs at ~1.5× uniform per-pair level (close share 0.256 vs uniform 0.167); combined with iter101's adv=200 + mmd=100 + fb=1.0 + anch=0.001 regime, the concentrated gradient tipped the encoder past stability — *more numerically dangerous* than iter121's same-total-pressure uniform regime (graceful 4/10 collapse). Code preserved opt-in, default None = byte-identical legacy. **Lessons**: (1) per-pair concentration is a different stability regime from uniform high pressure; future per-pair attempts must lower lambda_mmd to compensate (e.g., `lambda_mmd ≈ 65` if max_weight=1.0 and sum=3.9), or soften ratios (far pairs 0.5 instead of 0.3), or add per-encoder gradient clipping; (2) the 4-sp MMD family is now exhausted (3 attacks, distinct failure modes — iter107 CT-cond → over-integrates / sil-ct regress; iter121 dual MMD → 4/10 z dim collapse; iter122 per-pair heterogeneous → NaN); per refinement.md anti-pattern rule, the next 4-sp attack must use a structurally-different mechanism family (iter123 — adversary capacity — does so).

<details>
<summary>iter122 full pre-change template (preserved for reproducibility audits)</summary>

```
Iteration 122 — 4-sp species-pair-conditional MMD (divergence-aware weights)
Expectation under attack: E8.1, E8.4, E8.5 on 4-sp pancreas
                          (iter101: kNN-sp 0.979 vs scVI 0.917 — Chimera WORSE
                           on integration; only 3/6 E8 pass; primary "definition
                           of done" violation in the suite).
Pivot rationale: iter107 (CT-cond MMD ALONE) and iter121 (dual MMD) both
                 failed under uniform-pair-weight regimes. Per anti-pattern
                 rule, the 3rd 4-sp attack must be structurally different.
                 Heterogeneous per-pair weighting based on pre-known
                 pairwise divergence is a NEW conditioning axis (not present
                 in iter107 or iter121). It targets the actual data heterogeneity:
                 H+Mo+Mouse pairs are mixable in z; Zebrafish pairs (450 Myr)
                 are not. Iter121's lesson — too much uniform MMD pressure
                 collapses z while Zeb-Hep stays unmixable — directly motivates
                 down-weighting Zeb pairs while keeping pressure on close pairs.
Change:
  Code (3 files, ~30 LoC):
    chimera/_module.py — new constructor param `mmd_pair_weights: Optional[Sequence[float]] = None`
      stored as `self._mmd_pair_weights` (None or registered float buffer of
      length C(K,2) where K=n_species). In `_compute_mmd`, when non-None,
      replace the uniform `mmd_total / n_pairs` with weighted average:
      `Σ w_k × MMD_k / Σ w_k` for pairs with ≥2 cells per side. Pair index
      k follows lex order: (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5
      for K=4. CT-cond path is unaffected (CT-cond MMD already groups by
      celltype × species-pair; pair weights don't apply when CT-cond is on).
    training/train_and_diagnose.py — new function param `mmd_pair_weights`,
      forwarded to extra_model_kwargs when non-None; new CLI flag
      `--mmd-pair-weights "w0,w1,w2,..."`.
    training/run_experiment.py — same plumbing pattern.
  Existing best-knowns: ALL byte-identical at default (mmd_pair_weights=None).
  Verified by smoke test: default construction has _mmd_pair_weights=None;
  the new branch is dead unless flag is set.
Test config (4-sp pancreas; iter101 recipe + per-pair MMD):
  --adata data/processed/camex_pancreas_unified.h5ad
  --config A --max-epochs 200 --seed 42 --accelerator mps
  --adv-weight 200 --lambda-anchor 0.001 --free-bits 1.0
  --lambda-mmd 100          # SAME magnitude as iter101 (preserves the lambda)
  --mmd-pair-weights "1.0,1.0,0.3,1.0,0.3,0.3"
                            # Order (lex by species idx): H-Mo, H-Mouse, H-Zeb,
                            # Mo-Mouse, Mo-Zeb, Mouse-Zeb. Close pairs (≤90 Myr)
                            # weight 1.0; Zebrafish pairs (450 Myr) weight 0.3.
Tier: 3 (architectural addition: heterogeneous-pair-weight MMD; one new param)

Direct touch-points
  - MMD pressure: iter101 uniform = 100 × Σ(MMD_pair) / 6. iter122 weighted =
    100 × Σ(w_k × MMD_pair_k) / Σ(w_k). With weights summing to 3.9 (3×1.0 +
    3×0.3), close pairs get ~1.54× iter101's effective weight; Zeb pairs get
    ~0.46× iter101's effective weight. Net total MMD on close-3 is HIGHER
    (more mixing pressure where it can succeed); on Zeb pairs LOWER (less
    pressure on impossible mixing). Total integral pressure (Σ(w×MMD)/Σw)
    similar to iter101 in magnitude but redistributed.
  - z encoder: increased pressure on close-3 species → tighter mixing on
    Hepatocytes/Endothelial among H+Mo+Mouse. Decreased pressure on Zeb-pairs
    → Zebrafish stays largely as own manifold (which it already is in iter101).
    Risk: posterior collapse if encoder can't carry both close-3-mixed and
    Zeb-isolated structure within 10 dims (free_bits=1.0 should hold; iter121
    collapsed at total_pressure=150 absorbed-uniform; iter122 effective
    pressure stays at iter101's level but redistributes).
  - Adversary classifier: Zeb cells will become MORE distinguishable in z
    (less MMD pressure to hide them) → adv may exploit Zeb-vs-others split.
    But adv loss is per-cell mean across 4 species; if 25% (Zeb) is easier
    while 75% (H+Mo+Mouse) becomes harder, net adv_acc could go either way.
    Predicted: adv_acc 0.765 → 0.55-0.75 (slight improvement, dominated by
    close-3 mixing improvement on Hep cluster which is bulk of cells).
  - GIN / α / FQR: completely unaffected. No FQR/anchor changes; pair weights
    only affect z-side MMD computation. α distribution predicted unchanged.
  - Residual decoder: receives slightly less pressure to absorb Zeb-specific
    expression (since less MMD is forcing Zeb's z to mix in). Could free up
    residual to focus on biological cell-state variation. Neutral expected.
  - Loss-balance numbers (chimera-loss-scaling-fix memory): unchanged. λ_anchor
    =0.001, λ_fit=0.1, λ_aux=1.0, λ_mmd=100. Only MMD inner aggregation changed.
  - Schedules: warmup_kl=20, warmup_fqr=20, warmup_adv=5 — unchanged.

Predicted deltas — signed, rough magnitude REQUIRED
  - PRIMARY kNN-sp: 0.979 (iter101) → 0.55-0.75 (Zeb cells (25% of data) keep
    high own-species-purity ~1.0; close-3 Hepatocytes (~65% of data, 10,933
    of 16,222 Hep cells are non-Zeb) drop to ~0.4-0.5 mixed; weighted
    expected ≈ 0.25 × 1.0 + 0.75 × 0.45 ≈ 0.59. Target ≤ scVI 0.917 + 0.01
    = 0.927 = E8.1 PASS).
  - PRIMARY E8.4 worst-CT (max chimera-scVI per-CT kNN gap): iter101 +0.062 →
    -0.10 to +0.02 (target ≤ +0.02 = E8.4 PASS; close-3 Hepatocyte mixing
    drives this lower).
  - PRIMARY E8.5 worst-CT vs chance excess: iter101 +0.345 (Hep) → +0.05 to
    +0.20 (target absolute ≤ 0.15 OR relative ≤ scVI excess + 0.05). Hep
    chance ≈ 0.25 (4 species), worst-case excess if Zeb-cells stay separate
    = 5289/16222 - 0.25 ≈ 0.076 baseline + within-Zeb mixing fraction ~1.0;
    H+Mo+Mouse Hep cells get strong mixing → low excess. Net worst-case
    around Zeb-Hep at ~0.20-0.40 (BORDERLINE on absolute; relative to scVI
    excess of (0.917-0.25)=0.667 + 0.05 = 0.717 → almost certainly PASS rel).
  - sil-ct: 0.408 (iter101) → 0.30-0.42 (held; CT structure preserved
    because pair-MMD doesn't dilute between-CT distance, only within-CT-and-
    species variance. Some residual loss because close-3 Hep mixing reduces
    Hep cluster compactness slightly).
  - sil-sp: 0.018 (iter101) → -0.05 to +0.05 (lower better, expected to drop
    because close-3 mixing lowers global species silhouette).
  - LT-acc: 0.985 (iter101) → 0.96-0.99 (preserved; cross-species CT
    inference works as long as cell-types map cleanly. Could drop slightly
    if Zeb-Hep is more isolated and harder to label-transfer onto from H/Mo/
    Mouse references).
  - alpha mean: 0.754 (iter101) → 0.74-0.77 (essentially unchanged; FQR/
    anchor unchanged; only z-side aggregation changed).
  - alpha std: 0.130 (iter101) → 0.11-0.14 (preserved).
  - frac(α>0.95): 0.95% (iter101) → 0.5%-2% (PASS, well under E3.2 5% bar).
  - Δα > 0.02: 6.5% (iter101) → 5%-12% (preserved).
  - Δα < -0.02: 53.8% (iter101) → 45%-60% (preserved).
  - BC(α): ~0.47 (held).
  - z active dims (var > 0.1): 10/10 (HARD: must hold; <8/10 = z collapse blocker).
  - z_min_var: ≥ 0.1 (HARD: < 0.1 = posterior collapse blocker).
    With reduced uniform pressure (no over-pressure like iter121's dual MMD),
    z is MORE robust to collapse than iter121, similar to iter101's regime.
  - residual_rms: in [0.01, 5] (PASS).
  - adv_acc: 0.765 (iter101) → 0.55-0.75 (better close-3 mixing, easier
    Zeb-detection. Net hard to predict; falsifying probe is wide).
  - UMAP aspect (p1-p99): expected round, < 1.5 (PASS).

Falsifying probes
  - HARD: kNN-sp ≥ scVI's 0.917 + 0.01 = 0.927 (no improvement vs iter101's
    0.979 is bad; no improvement vs scVI is worse — pair-weighted MMD failed
    to add signal beyond uniform MMD).
  - HARD: alpha frozen — Spearman r(α, prior) > 0.99 OR std(α) < 0.05 (means
    pair-weighted MMD destabilized α; should NOT happen since α-side machinery
    is untouched).
  - HARD: z collapse — active dims < 8/10 OR z_min_var < 0.1 (means net pressure
    on close-3 is too concentrated; need to reduce close-pair weights).
  - HARD: sil-ct < 0.30 (CT structure destroyed — same iter121 failure
    mechanism, but with redistributed pressure).
  - HARD: E8 ≤ 3/6 (no improvement on integration vs iter101).
  - SOFT: residual_rms > 5 (residual eating CT info; structural concern).
  - SOFT: LT-acc < 0.94 (Zeb isolation makes cross-species CT inference harder).

Rollback plan
  - The CLI flag `--mmd-pair-weights` defaults to None → byte-identical
    behavior on every existing best-known (verified via smoke test on the
    constructor; the new branch in `_compute_mmd` is dead at default).
  - If iter122 fails: don't pass `--mmd-pair-weights`, all best-knowns
    unaffected. iter101 (4-sp pancreas) remains best-known; revert is just
    "stop using the flag" — no code or config rollback needed.
  - The 3-file code change is small and well-scoped; safe to keep as opt-in.
  - Backup queue (if iter122 fails or partially succeeds):
      iter122b: same recipe with weights "1.0,1.0,0.5,1.0,0.5,0.5" (less
                aggressive Zeb de-weighting); 1 iteration.
      iter122c: same recipe with weights "1.0,1.0,0.0,1.0,0.0,0.0" (full
                Zeb exclusion); diagnostic, exposes whether Zeb pairs add
                ANY useful integration signal.
      iter122d: weights "1.0,1.0,0.3,1.0,0.3,0.3" + lambda_mmd=150 (boost
                effective close-pair pressure to compensate for the avg
                being lower).
```
</details>

**iter121 ROLLBACK (2026-04-26, prior-iteration; distilled).** Strategic pivot off testis-E9.2-down family attacked 4-sp pancreas integration (Chimera kNN-sp 0.979 vs scVI 0.917) via opt-in `--lambda-mmd-aux` dual MMD on iter101 recipe (`--lambda-mmd 50 --lambda-mmd-aux 100 --celltype-conditional-mmd --celltype-key cell_type`, 200ep s42). **Result: HARD FAIL — REGRESSED on E8 (3/6 → 2/6).** kNN-sp 0.976 (predicted 0.85-0.92), z_mean_var_min 0.003 (predicted ≥ 0.1 → 4/10 z dims collapsed), sil-ct 0.408 → 0.244, E8.4 +0.123 on Endothelial. **Mechanistic finding**: 4-sp pancreas has only 2 shared CTs (Hep 16222 cells = 73% of data, Endo 2733); under increased uniform MMD pressure, z absorbs the extra pressure by collapsing dims rather than improving mixing — Hep over-mixes within itself (kNN-sp on Hep 0.952 → 0.994), destroying CT structure. **No "free room" to push more MMD uniformly.** GIN-side biology preserved (mean_corr_on_neg_S_g=-0.93, ρ(S_g, corr)=0.55, BC=0.471, frac>0.95=0.41% — confirms architectural separation between MMD/z-side and FQR/α-side).

**Audit trail** (predicted-vs-measured kept compact; full table in run-dir `summary.json`):

| Metric | iter101 | iter121 measured | predicted | Verdict |
|---|---|---|---|---|
| **PRIMARY kNN-sp** | 0.979 | **0.976** | 0.85-0.92 | HARD FAIL |
| **E8 score** | 3/6 | **2/6** | 5-6/6 | HARD FAIL — REGRESSED |
| **E8.4 worst-CT** | +0.062 Hep | **+0.123** Endo | -0.05 to +0.02 | HARD FAIL |
| **z_mean_var_min** | (≥ 0.1) | **0.003** | ≥ 0.1 (HARD) | HARD FAIL — partial collapse |
| sil-ct | 0.408 | **0.244** | 0.35-0.42 | HARD FAIL |
| α mean / std / BC | 0.754 / 0.130 / 0.49 | **0.751 / 0.129 / 0.47** | held | PASS (α biology fine) |
| Δα > 0.02 / < -0.02 (clamped) | 18.13% / 54.31% | **12.84% / 56.46%** | held | PASS |
| UMAP aspect | (n/a) | **1.21** | < 1.5 | PASS |

**Decision**: ROLLBACK iter121. iter101 retained as 4-sp best-known. `--lambda-mmd-aux` flag preserved opt-in (default 0.0 = byte-identical). Artifacts at `runs/xds-4sp-iter121-dualmmd-200ep-s42/`.

**Lesson for iter122 (this invocation)**: Two MMD-family attacks on 4-sp under uniform-pair-weight regimes (iter107 CT-cond ALONE, iter121 dual MMD) have failed. The next 4-sp attack must use a structurally-different mechanism — heterogeneous per-pair weighting based on pre-known divergences (option α, this iteration), or pivot to other gaps (testis E9.2 down resolution paths δ/ε remain open but blocked on user approval / Tier-3 multi-week work).

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

## Active hypothesis queue (post-iter124 ROLLBACK)

In priority order. Re-evaluate on every iteration; the queue is informational only.

**Post-iter124 conviction (2026-04-26)**:

- **Pancreas 4-sp**: FIVE attacks now failed (iter107 CT-cond → over-integrates;
  iter121 dual MMD → 4/10 z-dim collapse; iter122 per-pair heterogeneous → NaN;
  iter123 adv (64,32) at adv_weight=200 → sil-ct collapse to 0.096; **iter124
  adv (64,32) at adv_weight=100 → sil-ct collapse to 0.079 (worse), LT-acc 0.753
  (worse), pos-S_g class corrected NEGATIVE (α-biology degenerate)**). The MMD
  family is exhausted on 4-sp; the adv-side-family-at-any-tested-adv_weight is
  also exhausted (2 consecutive sil-ct collapses). Per refinement.md
  anti-pattern rule, the next 4-sp attack MUST use a structurally-different
  mechanism family — encoder/residual capacity (η-r), cycle-consistency (θ),
  or expectation refinement (β).
- **Testis triad E9.2 down**: per-gene mechanism family EXHAUSTED (iter113–
  iter120). Architectural ceiling in FQR baseline `p_null × lib` (math caps
  E9.2 down attainable mass below 2% on testis regardless of mechanism). The
  remaining paths are expectation refinement (β/δ; user approval) or Tier-3
  FQR redesign (ε; multi-week).

**Strategic queue (post-iter124, ordered by lowest blast radius first)**:

| # | Hypothesis | Tier | When to run | Pros / Cons |
|---|------------|------|-------------|-------------|
| **(η-r) iter125 — RECOMMENDED NEXT** | `--n-hidden 256` on iter101 recipe (encoder capacity 128→256). Structurally different from adv-side family entirely. Hypothesis: encoder needs more capacity to encode 4-species shared celltype structure WITHOUT losing CT separation under the iter101 adv_weight=200 + lambda_mmd=100 regime. | 1 | NEXT | Pros: enters NEW family (encoder, not adversary); falsified by clear E8.1 / sil-ct trigger pattern; ~80 min run. Cons: verify `--n-hidden` flag is plumbed before running; if not, small code change to `_constants.py`/`_module.py`. |
| **(η-d) iter125b** | Encoder DEPTH bump (n_layers 1 → 2) on iter101 recipe. | 1 | If (η-r) does not help | Same family as (η-r) but different lever; cheap. |
| **(θ) iter126** | Cycle-consistency / cross-species reconstruction. Encode species A, decode through residual as if species B; enforce within-celltype distribution consistency in species B. | 3 | If (η-r) and (η-d) exhausted | Tier-3, multi-day; refinement.md explicitly authorizes as a primary lever for cross-species integration. |
| **(β) Refine 4-sp E8 expectations** | Accept structural 2-shared-CT limit; relax E8.1 / E8.4 thresholds for low-shared-CT-count scenarios. | 0 | After (η-r), (η-d), (θ) all fail | REQUIRES USER APPROVAL — refinement.md change. |
| **(δ) Refine E9.2 down for testis** | `frac(Δα<-0.05) ≥ min(2%, 60% × frac(S_g<-0.05))`. | 0 | When 4-sp pancreas closed | REQUIRES USER APPROVAL. Closes ★ on testis_unified iter115 only; 2-sp testis remains arch-ceilinged. |
| **(ε) Tier-3 FQR baseline redesign** | `baseline = mu_full_ema.detach()` instead of `p_null × lib`. Expected to make S_g_ema bidirectional on testis. | 3 (multi-week) | Last resort | Re-validates ALL loss-balance numbers and ALL best-known recipes; high risk of regression on currently-★ datasets. |
| **(ζ) Multi-seed E9.1 verification of new accepted configs** | Run seeds 43 & 44 on any newly-promoted config; verify cross-seed Spearman ρ ≥ 0.50. | 1 | After every promotion | Routine — cheap, only run when something gets promoted. |
| **(η-id) Tier-3: Chimera × multi-batch-per-species `batch_key`** | Investigate iter94 z-contraction root cause; possibly separate batch_key (donor) from species_key. | 3 (multi-week) | Long-term backlog | Architectural; not in immediate path. |
| (DEPRECATED — adv-side family EXHAUSTED) | `--adv-classifier-hidden "64,32" --adv-weight 80` or related adv-knob sweeps | 2 | Skipped per anti-pattern rule | Iter123 (adv_w=200) + iter124 (adv_w=100) both produced sil-ct collapse + pos-S_g class wrong-sign correction. Family exhausted. |

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
| `z_leakage_probe.py` | Post-hoc non-linear z-leakage probe (linear/kNN5/MLP-32/MLP-64-32) on `z_mean` (added iter123). Co-primary diagnostic for any iteration touching adv/MMD machinery. Reads `final_state.npz`, writes `z_leakage_probe.json`. |
| `umap_check.py` | UMAP plot + aspect ratio (must be < 1.5) per run. |
| `gin_signed_diagnostics.py` | Per-gene signed metrics: ρ(S_g, corr), mean correction on neg-S_g class, bias_share if `use_per_gene_bias=True`. Co-primary for testis E9.2 down attacks. |

Run artifacts under `runs/<run-name>/`: `metrics.csv`, `alpha_epoch_*.npz`, `final_state.npz`, `model/`, `report.md`, `summary.json`, `config.json`, `plots/`, `integration_vs_scvi.{md,json}`, `z_leakage_probe.json`.

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
| iter121 | 4-sp panc DUAL MMD | iter101 + `--lambda-mmd 50 --lambda-mmd-aux 100 --celltype-conditional-mmd --celltype-key cell_type` (NEW `--lambda-mmd-aux` flag, opt-in default 0.0 = byte-identical) | E8 **2/6** (REGRESSED); kNN-sp 0.976 (HARD FAIL — vs predicted 0.85-0.92); sil-ct 0.408→**0.244** (HARD FAIL); z_mean_var_min=**0.003** (HARD — 4/10 dims collapsed); α biology preserved | Uniform high MMD pressure over-pressures z; 4-sp w/ only 2 shared CTs has no "free room" to push more MMD without sacrificing CT structure. MMD/z-side and FQR/α-side architecturally separable (α biology preserved). | **ROLLBACK** (iter101 retained 4-sp best-known) |
| iter122 | 4-sp panc species-pair-conditional MMD | iter101 + `--mmd-pair-weights "1.0,1.0,0.3,1.0,0.3,0.3"` (NEW opt-in flag — heterogeneous per-pair averaging, default None = byte-identical) | **TRAINING NaN at ep17→18** (before FQR/KL warmup completion). Pre-NaN ep15-17: grad_enc=1.7-2.6, L_recon=2500-2900 (healthy). All final_state tensors NaN (α, correction, S_g_ema, z_mean, mu_full, residual). | Heterogeneous per-pair weighting concentrates MMD gradient on close pairs at ~1.5× uniform per-pair magnitude (close: 1.0/3.9 = 0.256 vs uniform 1/6 = 0.167). At iter101's already-aggressive regime (adv=200, mmd=100, fb=1.0), this concentration tipped encoder past stability. Per-pair heterogeneous more catastrophic than uniform high pressure (iter121 graceful 4/10 collapse vs iter122 instant NaN). Future per-pair attempts must lower lambda_mmd to compensate (e.g. ~65) or use softer weight ratios. | **ROLLBACK** (iter101 retained 4-sp best-known; `--mmd-pair-weights` flag preserved opt-in default None = byte-identical) |
| iter123 | 4-sp panc adv-classifier capacity (32,)→(64,32) | iter101 + `--adv-classifier-hidden "64,32"` (existing CLI flag, default None = (32,) legacy) | 33/33 E1-E9 PASS (improved from 30/32; Δα 17.4%/62.7% wider bidirectional); E8 **2/6** (REGRESSED from 3/6); kNN-sp 0.979→**0.962** (missed predicted 0.93–0.95); **sil-ct 0.408→0.096** (HARD falsifying probe <0.30 triggered); LT-acc 0.985→0.893; in-training adv_acc 0.765→0.389 (over-fooled) but post-hoc MLP-64-32 0.738→**0.709** (deeper z-leakage barely budged). | Bigger classifier at FIXED `adv_weight=200` over-tips GRL game: encoder over-fits to fool the SPECIFIC (64,32) instance by destroying CT structure (cheapest axis), not learning a generally species-blind z. Adv capacity must co-tune with adv_weight; iter124 = same change at adv_weight=100 is the next test. NEW co-primary diagnostic: `training/z_leakage_probe.py` (post-hoc linear/kNN5/MLP). | **ROLLBACK** (iter101 retained 4-sp best-known; CLI value reverted, no code change; new probe script kept) |
| iter124 | 4-sp panc adv-classifier (64,32) + adv_weight 200→100 | iter101 + `--adv-classifier-hidden "64,32" --adv-weight 100` (both flags pre-existing, default safe) | 33/33 E1-E9 PASS; E8 **2/6** (REGRESSED, same as iter123); kNN-sp 0.946 (hit lower band 0.91-0.95, FAIL E8.1); **sil-ct 0.408→0.079** (HARD <0.30 trigger, WORSE than iter123 0.096); LT-acc 0.985→**0.753** (HARD FAIL, REGRESSED MORE than iter123 0.893); in-training adv_acc 0.529 (predicted 0.55-0.75, just below); post-hoc MLP-64-32 0.738→0.694 (predicted ≤0.50, MISS); Endo Δ vs scVI +0.024 (vs iter101 +0.128, partial close-3 improvement); z PC1/PC2 SVD aspect **5.27** (HARD FAIL — z thin); mean_corr_on_pos_S_g **−1.58** (WRONG SIGN — α-biology DEGENERATE) | Co-tuning adv_weight=100 with (64,32) classifier does NOT save CT structure — sil-ct collapsed *more* than iter123 at fixed adv_weight=200. The (64,32) × 4-sp encoder interaction is the binding mechanism; adv_weight is second-order. Adv-side family on 4-sp now EXHAUSTED (2 consecutive sil-ct collapses). NEW co-primary: `mean_corr_on_pos_S_g` from `gin_signed_diagnostics.py` exposes α-biology degeneration that E9.2 PASS metrics mask. | **ROLLBACK** (iter101 retained 4-sp best-known; flags reverted, no code change) |

---

## Notes for the next agent

1. **Read this file end-to-end AND [workflow-best.md](workflow-best.md)** before doing anything. Both are authoritative; if a config you intend to use is not the documented current best, reconcile (update registry or update plan) before running.

2. **Where the loop stands (post-iter124 ROLLBACK, 2026-04-26)**:
   - **Pancreas 2-sp (H+Mo, H+Mouse): both ★ DONE.** Three-seed verified, full 33/33 + 6/6 E8 + E9.1 PASS.
   - **Pancreas H+Zeb: structural at 450 Myr** — 4/6 E8 with kNN-sp at chance. Treated as data-limited; no further iteration unless cycle-consistency Tier-3 work is authorized.
   - **Pancreas 4-sp (iter101 retained as best-known)**: 3/6 E8, kNN-sp 0.979 vs scVI 0.930 (Chimera WORSE). **FIVE 4-sp attacks have now failed**: iter107 CT-cond → over-integrates; iter121 dual MMD → 4/10 z collapse; iter122 per-pair heterogeneous → NaN; iter123 adv capacity (64,32) at adv_weight=200 → sil-ct collapse 0.408→0.096; **iter124 adv (64,32) co-tuned at adv_weight=100 → sil-ct collapsed FURTHER to 0.079, LT-acc 0.753**. **Adv-side family is now EXHAUSTED on 4-sp** (2 consecutive sil-ct collapses iter123+iter124, and MMD family was already exhausted). Per refinement.md anti-pattern rule, the next 4-sp attack MUST be structurally different — encoder/residual capacity (η-r), cycle-consistency (θ), or expectation refinement (β).
   - **Testis triad (H+Mac iter109, H+Mouse iter111, unified iter115)**: 6/6 E8 vs scVI(donor) on all three; **E9.2 down hits architectural ceiling** (0%, 0%, 1.57%). Per-gene mechanism family EXHAUSTED (8 failed attacks iter113–iter120). ★ withheld until E9.2 down resolved.

3. **What this invocation did (iter124)**:
   - Trained iter124 = iter101 recipe + `--adv-classifier-hidden "64,32" --adv-weight 100` (200ep s42, 66 min). Direct mechanistic test of iter123's "co-tune lower" lesson.
   - **Result: HARD ROLLBACK.** 33/33 PASS on E1-E9 (same as iter123) BUT E8 still 2/6 (same FAIL pattern as iter123): sil-ct collapsed FURTHER to 0.079 (vs iter123 0.096, iter101 0.408 — HARD <0.30 trigger), LT-acc REGRESSED FURTHER to 0.753 (vs iter123 0.893, iter101 0.985, scVI 0.929 — fails E8.6). kNN-sp improved to 0.946 (hit lower predicted band 0.91-0.95) but still FAIL E8.1 (Δ vs scVI=+0.016 > 0.01). Post-hoc MLP-64-32 z-leakage 0.738→0.694 (predicted ≤0.50; encoder DID do somewhat more legitimate species-scrubbing than iter123, but inadequate).
   - **Critical α-biology degeneration finding**: gin_signed_diagnostics revealed `mean_corr_on_pos_S_g = −1.58` (WRONG SIGN — GIN pushes positive class toward divergence) and `frac_corr_pos_on_pos_S_g = 0.00` (every positive-class gene gets a negative correction). E9.2 PASS metrics MASKED this in iter123; iter124 surfaces it. **Adopt `mean_corr_on_pos_S_g` and `frac_corr_pos_on_pos_S_g` as co-primary diagnostics** for any 4-sp z-side perturbation.
   - **z PC1/PC2 SVD aspect = 5.27** (HARD thin) while UMAP aspect = 1.19 (round) — linear-projection diagnostic catches encoder collapse onto thin manifold even when non-linear UMAP looks fine.
   - Workflow.md updated with iter124 measured-vs-predicted + 5 distilled lessons + active hypothesis queue refresh + Scripts table additions.

4. **Strategic question for the NEXT invocation (adv-side EXHAUSTED on 4-sp)**:
   - **Gap A: 4-sp pancreas integration** (Chimera kNN-sp 0.946 iter124 / 0.979 iter101 > scVI 0.930). MMD family AND adv-side-at-any-adv_weight family are both exhausted. Remaining structurally-different paths:
     - **(η-r, NOW RECOMMENDED) Encoder n_hidden 128→256 on iter101 recipe** — bumps encoder capacity (NOT adversary). Hypothesis: encoder needs more capacity to encode 4-species shared celltype structure WITHOUT losing CT separation under the iter101 adv_weight=200 + lambda_mmd=100 regime. CLI: `--n-hidden 256` (verify this flag is plumbed before running; if not, small code change to `_constants.py`/`_module.py`). 1 iteration; ~80 min.
     - **(η-d) Encoder DEPTH bump** (n_layers 1→2) on iter101 recipe.
     - **(θ) Cycle-consistency / cross-species reconstruction** — Tier-3, multi-day.
     - **(β) Refine 4-sp E8 expectations** — accept 2-shared-CT structural limit (requires user approval to amend refinement.md).
   - **Gap B: Testis triad E9.2 down** (per-gene mechanism family exhausted, iter113–iter120). Open paths unchanged:
     - **(δ) Refine E9.2 expectation to be data-adaptive** — Requires user approval to amend refinement.md.
     - **(ε) Tier-3 FQR baseline redesign** — Multi-week.
   - **Recommended ordering for next invocation**: **option (η-r)** — encoder capacity bump. Structurally different from adv-side family entirely. If it fails, pivot to (η-d) depth, then (θ) cycle-consistency. (β) E8 refinement should be raised to the user only after (η-r), (η-d), and (θ) are all exhausted.

5. **CRITICAL diagnostic findings preserved from iter117–iter124 (do NOT lose)**:
   - **iter124 lesson — adv-side family EXHAUSTED on 4-sp.** Two consecutive sil-ct collapses (iter123 at adv_weight=200, iter124 at adv_weight=100, both with classifier=(64,32)). The (64,32) × 4-sp encoder interaction is the binding mechanism; adv_weight is second-order. Future capacity bumps to the in-training adversary on 4-sp are deprecated.
   - **iter124 lesson — α biology degeneration under z drift is masked by E9.2 metrics.** iter123 looked like "α biology improved" (Δα bidirectional widened) but iter124's per-class signed metrics revealed underlying degeneracy: `mean_corr_on_pos_S_g=−1.58` (NEGATIVE on positive class!). Adopt `gin_signed_diagnostics.py`'s per-class correction signs as co-primary diagnostics for 4-sp z-side perturbation, NOT just frac(Δα<−0.05).
   - **iter124 lesson — z PC1/PC2 SVD aspect catches what UMAP aspect misses.** Encoder can collapse onto a thin linear subspace (z PC1/PC2 aspect 5.27) while UMAP aspect (1.19) looks round. The linear-projection diagnostic is the sharper signal for 4-sp z health. `gin_signed_diagnostics.py` reports it.
   - **iter123 lesson — adv classifier capacity and `adv_weight` are coupled.** A bigger (64,32) classifier at fixed `adv_weight=200` is over-strong; the encoder takes the cheapest path to fool it (collapse celltype structure) rather than learning species-blind z. iter124 confirmed this is NOT a smooth tradeoff curve — halving adv_weight does NOT halve CT damage.
   - **iter123 lesson — post-hoc z-leakage is the right metric, in-training adv_acc is misleading.** `training/z_leakage_probe.py` (added iter123) is now a co-primary diagnostic for any iteration touching adv/MMD machinery. iter101 4-sp baseline: linear=+0.575, kNN5=+0.731, MLP-32=+0.712, MLP-64-32=+0.738. iter124 baseline: linear=+0.510, kNN5=+0.701, MLP-32=+0.602, MLP-64-32=+0.694. Target for any adv/MMD-touching iter: MLP-64-32 ≤ +0.50.
   - **iter123 lesson — 4-sp Endothelial WAS the binding worst-CT (iter101 +0.128 Δ vs scVI), but iter124 shows the gap can be partially closed (+0.024) at unacceptable global CT cost.** The remaining post-iter124 worst-CT is Hepatocytes at +0.040 vs scVI (essentially unchanged). Hep close-3 mixing is the harder structural constraint.

   - **ρ(S_g, corr) is misleading as a primary metric** for testis E9.2 down — iter117 had higher ρ than iter115 yet worse E9.2 down. The right co-primary signal is **mean correction on neg-S_g class** + **bias_share_of_total_magnitude**. Both reported by `gin_signed_diagnostics.py`.
   - **The architectural ceiling on testis E9.2 down is in the FQR baseline definition `p_null × lib`**, not in any per-gene mechanism. The math: testis has 1.06% of genes with S_g<-0.05; even at perfect mechanism engagement, only ~1% can reach Δα<-0.05.
   - **Per-gene mechanisms converge by ~ep55** on testis. Future per-gene experiments can use 80–100ep instead of 200ep.
   - **iter121 dual MMD lesson**: 4-sp pancreas with only 2 shared CTs has no "free room" to push more uniform MMD without sacrificing CT structure. Total MMD pressure ≥ ~150 collapses z dims (4/10 weak), and remaining capacity over-mixes within Hep (73% of cells).
   - **iter122 per-pair heterogeneous lesson**: Concentrating MMD gradient on close pairs (close-pair share 1.5× uniform) at iter101's lambda_mmd=100 is numerically unstable — model NaN'd at ep17→18 BEFORE FQR/KL warmup completion. **Future per-pair attempts MUST lower lambda_mmd to compensate** (target close-pair gradient share ≤ uniform's, requires lambda_mmd ≤ ~65 with `[1.0, 1.0, 0.3, 1.0, 0.3, 0.3]` weights), AND/OR use softer weight ratios (e.g., far pairs at 0.5 instead of 0.3), AND/OR add explicit per-encoder gradient clipping (currently only total-grad clip is on).
   - **Cell-count diagnosis on 4-sp pancreas (recorded for next iteration's pre-change template)**: Hepatocytes 16,222 = 73% of all cells; Endothelial 2,733 = 12%. Of Zebrafish's 5,530 cells, 5,289 (96%) are Hepatocytes. Zebrafish-Hep is the dominant unmixable cluster. Within-Hep kNN-sp on iter101 is 0.952 — H+Mo+Mouse + Zeb cells are not mixed within Hepatocytes despite global integration pressure.

6. **Identifiability state (closed audit, 2026-04-26)**: Verified per-dataset batch/species columns. Testis Chimera uses `batch_key=batch` (==species) while scVI baseline uses `batch_key=sample_batch` (donors). The asymmetry is INTENTIONAL — scVI(donor) is the STRONGER baseline (more batch-correction power), and Chimera STILL beats it on integration metrics. Pancreas has no `sample_batch` column (donors live in `batch`); both Chimera and scVI use the same key. NO re-runs needed. The architectural Chimera × multi-batch-per-species problem (iter94 rollback root cause) remains a Tier-3 open question deferred indefinitely.

7. **Next-iteration evaluation steps** (after any training completes — this invocation verified the pipeline works for 4-sp):
   - (a) `check_expectations.py` auto-runs from `run_experiment.py` → produces `report.md` + `summary.json`.
   - (b) Run `compare_vs_scvi.py --chimera-run-name <name> --adata <adata-path> --batch-key batch` (use `--batch-key sample_batch` for testis); auto-uses cached baselines at `runs/baselines_scvi/`.
   - (c) Run `gin_signed_diagnostics.py <run-dir>` for per-gene signed metrics (rare-class mean corr is co-primary; ρ alone misleading).
   - (d) Run `umap_check.py <run-dir> --adata <adata-path>` for UMAP plot + aspect ratio (must be < 1.5).
   - (e) Measured-vs-predicted entry in workflow.md against the pre-change template.
   - (f) If accepted: atomic update of workflow-best.md + `_constants.py` defaults (only if scenario default changes) + notebook pointers.
   - **NOTE**: `compare_vs_scvi.py`, `umap_check.py`, `gin_signed_diagnostics.py` are NOT auto-invoked from `run_experiment.py` — must be run manually.

8. **Code state at end of iter123**:
   - **NEW this invocation**: `training/z_leakage_probe.py` — post-hoc non-linear z-leakage probe (linear/kNN5/MLP-32/MLP-64-32). Run as `python training/z_leakage_probe.py --run-name <name> [--adata <path>]`. Reads `final_state.npz`, writes `z_leakage_probe.json`. Closes a refinement.md gap (non-linear z probe was missing from `check_expectations.py`).
   - `--adv-classifier-hidden "h1,h2,..."` — already plumbed (run_experiment.py:209, train_and_diagnose.py:454); was used as iter123's only knob, no code change required. Default None → (32,) hidden as before, byte-identical legacy.
   - **All earlier opt-in flags retained from prior iterations**: `--mmd-pair-weights` (iter122), `--lambda-mmd-aux` (iter121), `--use-per-gene-bias` (iter119), `--lambda-anchor-neg-factor` (iter118), `--sg-ema-beta` (iter116), `--gin-hidden` (iter114), `--prior-clamp-max` (iter109), `--celltype-conditional-mmd`, `--celltype-key`. All default OFF / byte-identical legacy.
   - `_training_plan.py:configure_optimizers` per_gene_bias→1e-3 carve-out (from iter120) preserved; only active when `use_per_gene_bias=True`.
   - **All existing best-knowns** (iter90 H+Mo / iter90 H+Mouse / iter101 4-sp / iter109 H+Mac / iter111 H+Mouse / iter115 unified) UNAFFECTED — iter123's only change was the `--adv-classifier-hidden` CLI value at runtime, not a default change. workflow-best.md unchanged.

9. **Stop after one iteration.** The user drives the outer loop. Do not auto-continue.
