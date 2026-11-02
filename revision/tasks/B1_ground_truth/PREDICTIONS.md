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
