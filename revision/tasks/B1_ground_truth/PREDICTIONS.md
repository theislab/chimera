# B1 Track A: expected sign and detection, stated before any altered run was scored

All scores are oriented so that higher means more conserved. Every alteration is divergence in one species, so the expected direction on any readout that can see it is **altered below its matched control**. Reasons below follow `_module.py` and `_components.py`.

## S_g (log-likelihood gain of the species-blind shared decoder over the species-specific flat null)
| Alteration | Expected | Why |
|---|---|---|
| level f | Negative, growing with f, larger at higher expression | Null p_null[s] is recomputed, so it absorbs the species' new level; the shared decoder cannot, because z carries no species. It must place one level between two, so both species pay: the level cost. Cost grows with the log fold and with counts per cell. A mean log-likelihood difference is larger for count-rich genes. Weakened when z retains species: the shared decoder can then fit each level (mechanism note: species-blindness depends on integration). |
| pattern q | Negative, growing with q | The pattern gain falls: the shared profile cannot match both the original and the swapped species. Total conserved in the altered species by construction, so there is no level cost beyond rounding. |
| loss e | Negative, growing with e | Pattern gain falls in the altered species' top type; the gene's total also falls there, so a level cost joins. |
| sham_swap, thin_both, none | No shift for sham and none. thin_both: \|S\| shrinks with expression, S moves toward 0, and more genes are masked, because both species thin equally there is no level cost | |

**Masking.** A gene needs 20 expressing cells in a 256-cell batch. The low band is masked in 100% of genes in all three cohorts (midlow 35%, 63%, 73%; midhigh 0-2%; high 0%, on unaltered fits). So S_g is exactly 0 in the low band on every alteration: AUROC 0.5 and power at or below 5%. Prediction: **no power in the low band; this is the known limit**. Thinning can also push midlow genes under the mask, moving S toward 0 rather than lower, which lowers power for level in midlow. Thinning one species only raises this less than thin_both, because the other species still expresses.

**S_fwd** (no mask, exploratory): same direction; should show some signal in the low band if the limit is the mask rather than the counts, but with little count evidence.

## Δα (S_g projected on gene features, logit units around π)
Direction, when any: same as S_g, because L_fit = -mean(c_g S_g_ema). Magnitude: **the plan's forecast is that Δα inherits detection through the features; the mechanism predicts only partial inheritance, at most.** α never sees the reconstruction. The MLP is a smooth function of 72 fixed features, and altered genes are drawn at random with respect to protein sequence, so it can only express an altered gene's S_g through what it memorises about that feature vector, in competition with its feature neighbours. Expected: AUROC well below S_g's on every alteration, ordered as S_g, and largest where rho(D, S_g) is highest (kidney about 0.55-0.61, testis about 0.42-0.48, liver bimodal 0.22-0.63 across seeds).

Second channel, through the three expression columns of the features, which move with the data: thinning lowers them, so Δα shifts by the cohort's relation of Δα to expression. Testis rho(D, expression) +0.71 to +0.73 and kidney +0.35 to +0.52: thinning lowers Δα, the expected sign, for a reason that is not divergence (thin_both will show it). Liver rho between -0.24 and +0.15 across seeds: no consistent push. Therefore **thin_both is the control for Δα on level**: Δα should detect level above thin_both's false signal only if the S_g channel works. Low band: Δα is defined (feature extrapolation), not a count call; a level signal there is the expression channel.

## Others
| Method | level | pattern | loss | Why |
|---|---|---|---|---|
| B1, B1b (profile correlation) | none | strong | strong | Species-centred log profile; a uniform thinning leaves it unchanged except near the detection floor, where low-count noise gives a small drop |
| B2b (specificity cosine) | none | strong | strong | Normalised profile |
| B1c (rank level) | strong | partial, falling with K (4 liver, 6 testis, 16 kidney): the swap moves ranks in 2 of K types | partial | Rank distance per type |
| B1z | none | strong, some loss to cluster purity | strong | As B1 on clusters of z; depends on the fit |
| Sdir (labelled direct margin) | strong | strong | strong | The same statistic as S_g with labels instead of a model: expected to match or beat S_g on clean labels |
| Lfc (species fold change) | strongest | none, total conserved | strong on loss in a dominant type | Direct measure of the level gap |
| B3a, π, dN/dS, LOEUF | none, AUROC about 0.5 | none | none | Do not read the altered counts, or only as a noisy target of a ridge on sequence features |
| expr (control) | Detects level and loss by the fall in expression | none | | Reference: any level signal no better than this adds nothing beyond expression |
| SATURN S_deploy, S_cos | weak | weak to partial | weak to partial | Scalars derived from macrogene weights; weights respond to expression through the autoencoder, sequence initialises them |
| CAMEX C_cos | weak; low coverage (HVG and DEG genes) | weak | weak | Gene node embedding; altered genes are often not nodes, and such genes tie |

The plan's "correlation-type baselines miss level" and "rank-level baselines miss pattern" are kept as stated, with the refinement above for B1c. Thresholds are not set; the metrics report what happens.

## Type I
- Placebo sets on unaltered fits: reject05 about 0.05 for every method. Above that means the matching leaves imbalance.
- sham_swap: as placebo for count-only scores. For z-based and Chimera scores a shift would mean the moved packets themselves are seen.
- thin_both: S_g, S_fwd, Δα, B1c, expr can move through expression, B1 and B2b through the floor. These are false calls of divergence on a pure expression fall, reported as such.

## Gate arms, stated before any arm fit was scored (added after the interim reading, so post hoc as a design, not as a prediction of data seen)
The interim explanation of why Δα misses isolated changes says the limit is the smooth gate. Two model forms lack it: **free** (Config M, no GIN, one free c_g per gene) and **hybrid** (GIN plus per-gene bias, iter119). Both run on the same altered objects as the main fits, recipe otherwise unchanged, tier 0 central alterations and nulls x 2 seeds, unaltered fits x 2 seeds for reference and placebo sets, then tier 1 main effects. New readout c_g (the correction in logit units).

- **Free.** With L = λ_anchor mean(c²) − λ_fit mean(c S_ema) the equilibrium is c* = (λ_fit / 2λ_anchor) S_ema, about 16.7 S_ema on these recipes. So c_g detects as S_g does (AUROC within noise of S_g on every alteration), if the run converges. Δα = α − π is that value through the sigmoid, so it compresses by π(1−π) and may saturate: detection close to S_g's but somewhat lower, most lost for genes near the prior clamp. Type I near nominal on placebo sets. On thin_both it inherits S_g's false shift, no more: the expression-feature channel is gone, so testis Δα on thin_both should fall from .80 to about S_g's .74, with power near 0.
- **Masked genes.** S_ema = 0 gives c* = 0, so free Δα is exactly at the prior in the low band: AUROC .50, power 0, same as S_g. The free gate therefore gives up the feature extrapolation, which was never a count call.
- **Convergence, read from unaltered Config M fits before any arm fit was scored.** rev-ctl-liv_hmouse-m-s42, 45, 46: ρ(c, S_ema) .98, .91, .96 on unmasked genes, slope 4.9, 2.2, 1.3 against 16.7, SD of masked c below .002. So c ranks with S_g but stops well short of equilibrium, and the size of c varies by seed. Detection by rank should hold; Δα's scale will be small and seed-dependent, which also limits saturation.
- **Hybrid.** The bias is also free, so its equilibrium is the same, but it trains at the optimiser rate its parameter group gets (`_training_plan.py`) and shares the correction with the MLP. iter119 on testis H+Macaque found the bias engaged at 2.7% of the total correction, abs max .057, bounded by Adam's step. So: between the canonical gate and the free gate, nearer the canonical gate unless its parameter group has since moved to the faster rate, in which case nearer free. In the low band it keeps MLP extrapolation.
- **Relation to the prior.** π stays the centre by construction. Reported per fit: ρ(α, π), ρ(Δα, π), ρ(Δα, S_g), SD of c, saturation, mixing, against canonical fits on the same objects. Expected: ρ(Δα, S_g) near 1 for free, ρ(α, π) lower than canonical if c is large against logit π; integration unchanged, because α is detached from the reconstruction only for its own gradient, yet it scales the residual, so a change is possible and is reported.
- **Failure of the explanation.** If free Δα and c_g stay well below S_g, the limit is not the gate.
- **Expression control.** On level, every arm is read beside thin_both and the expression control, as for the main fits.
