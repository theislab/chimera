# Chimera

**Cross-species Harmonization via Integrated Mixing of Evolutionary Residual Alphas**

---

## Why this model exists

Every tool for integrating single-cell RNA-seq data across species treats species as a batch effect — a uniform technical artifact to be erased. But species divergence is not uniform. It is per-gene. Histone H3 is 98% identical between human and zebrafish; 500 million years of evolution barely touched it. Some immune receptors share less than 30% sequence identity across the same species pair. When a model is told "species = batch," it applies the same correction everywhere: it overcorrects the immune receptors, forcing alignment where biology says there should be none, and undercorrects the histones, failing to exploit a near-perfect conservation signal that evolution has handed to it for free.

The deeper problem is not that current tools integrate poorly — at the cell level, they often integrate well enough. The problem is that after integration, you cannot ask the question that matters most to evolutionary biology: *which genes drove the alignment, and which were dragged along against their biology?* scVI, Harmony, and SysVI produce integrated cell embeddings with no per-gene accounting. SATURN learns macrogenes from protein language model embeddings, and CAMEX builds heterogeneous graphs that handle many-to-many orthologs; their macrogene weights and gene modules are genuine gene-level evidence. What they do not give is one signed number per one-to-one ortholog that is itself the gate of the integration and is read against a protein-identity prior. Chimera provides that number, and gives up cell-type resolution and many-to-many homology to do so.

That per-gene number is what Chimera was built to provide.

## What Chimera is

Chimera is a variational autoencoder built on scvi-tools that integrates multi-species scRNA-seq data while simultaneously learning a **per-gene integration coefficient — alpha — that is both the mechanism of integration and a readable scientific output.** Alpha is not a post-hoc diagnostic computed after the model finishes. It is the gate through which evolutionary information enters the decoder during training: for each gene, alpha controls how much species-specific correction is applied on top of a species-agnostic shared prediction. When training is done, the same alpha that controlled the integration is directly readable as a per-gene statement about how much cross-species alignment that gene needed.

What makes alpha scientifically useful rather than just mechanistically convenient is that it is **bidirectional around the conservation prior.** Each gene enters training with an alpha initialized from its protein sequence conservation — an 85%-conserved gene starts at 0.85, a 40%-conserved gene at 0.40. During training, expression data pushes alpha above or below that prior. A gene whose alpha settles above its prior is one whose expression behavior is *more conserved across species than its protein sequence alone would predict* — functional conservation beyond what evolution left in the amino acids. A gene whose alpha settles below its prior is one whose expression is *more species-specific than its sequence would predict* — regulatory divergence despite structural similarity. The delta between alpha and the prior is the model's per-gene readout: not sequence conservation, not expression correlation, but the discrepancy between the two as estimated by a model that must reconstruct the data with it.

## What you can do with the outputs

Chimera produces integrated cell embeddings for the usual tasks — clustering, visualization, cross-species label transfer — and its per-gene outputs support analyses such as:

- **Discover functional conservation that sequence misses.** Genes with `delta_alpha > 0` are conserved in expression beyond what their protein identity predicts. These are candidates for expression programs that are more conserved than sequence identity alone predicts. Such calls are hypotheses to test against independent data, not evidence of selection.

- **Flag regulatory divergence in structurally similar genes.** Genes with `delta_alpha < 0` have species-specific expression patterns despite high sequence conservation. These are candidates for species-specific deployment (same protein, different expression), which sequence-based conservation scores do not show.

- **Prioritise ortholog pairs for inspection.** A pair annotated as one-to-one orthologs that gets a low alpha despite high sequence conservation may reflect a misannotation, an unresolved duplication, or diverged regulation. Alpha flags such pairs; it does not decide between these explanations.

- **Compare ortholog resources.** The ortholog table is an input. Running Chimera with tables from different resources would show where they disagree in ways that matter for expression; this repository uses Ensembl release 110 only.

- **Identify which genes are safe to transfer across species.** For any downstream analysis that assumes cross-species equivalence — differential expression between human and mouse cell types, gene regulatory network inference from a joint atlas, imputation of missing species — alpha tells you, per gene, how much the model trusted that equivalence. Filter to high-alpha genes and you are working with the subset that the data, not just the annotation, says are comparable.

## How it works

### Conservation features and the Gene Integration Network

Before training, Chimera assembles a 72-dimensional feature vector for each gene that encodes everything the model should know about that gene's evolutionary context:

| Feature | What it captures |
|---------|-----------------|
| Protein sequence identity (mean and std across species pairs) | Raw amino acid conservation, and whether it is uniform or patchy across the species in the dataset |
| ESM-2 structural embeddings (PCA'd to 64 dims) | What the protein looks like structurally and functionally, as encoded by a protein language model trained on millions of sequences |
| ESM-2 inter-species variance | How much the structural embedding diverges across species for this gene — the pairwise conservation signal that ESM-2 encodes implicitly but that no tool before Chimera has surfaced explicitly |
| Expression level and variance | How highly expressed the gene is on average and how much it discriminates between cell types |
| Detection rate | How sparse the gene is across cells |

The ESM-2 inter-species variance deserves emphasis because it closes a specific gap. SATURN uses ESM-2 embeddings to place genes in a shared space, which captures what each protein looks like individually. But ESM-2 processes one sequence at a time — it produces a vector for human TP53 and a separate vector for mouse Trp53 without any concept of how those two vectors relate. Chimera takes precomputed ESM-2 embeddings of each species' protein (the SATURN release, `esm2_t48_15B_UR50D`) and computes, per gene, the mean pairwise cosine distance between the species' embeddings. Two proteins can share only 60% amino acid identity but fold into nearly the same structure — low ESM-2 variance despite moderate sequence divergence. Conversely, a few key mutations can alter structure substantially even at high sequence identity. The variance captures what raw identity percentage cannot: whether the proteins are functionally similar *as proteins*, not just similar *as strings*.

The Gene Integration Network (GIN) is a small neural network (a few thousand parameters, in a model of several million) that takes this feature vector and computes alpha through a prior-plus-correction architecture: `alpha = sigmoid(logit(conservation_prior) + MLP(features))`. The logit of the conservation prior is the anchor — before the model sees expression data, alpha matches the gene's protein conservation. The MLP correction starts near zero and learns to adjust alpha based on what the expression data reveals. After training, the correction *is* the delta-alpha: the discrepancy between what evolution wrote in the sequence and what the cells actually do.

### The residual decoder

The decoder reconstructs gene expression through two pathways that are always combined:

The **shared pathway** is species-agnostic — it processes the latent cell representation through hidden layers that see no species label and produces a predicted log-expression for each gene. Because this pathway cannot distinguish species, it must learn expression patterns that hold across all of them. It also carries a per-batch technical offset constrained to sum to zero within each species, which allows within-species batch correction (different donors, different labs) without leaking species identity into the shared prediction.

The **residual pathway** adds a species-specific correction on top of the shared prediction, gated per gene by `(1 - alpha)`. When alpha is high, the correction is suppressed. When alpha is low, the full correction is applied.

The critical design decision is that the shared prediction is always present at full strength — there is no setting of alpha that bypasses it. This is what makes alpha *honest*: the model cannot route around the shared pathway by setting alpha to zero, because the shared pathway's output is always there regardless. Alpha controls only how much additional species-specific correction is layered on top. A high alpha does not mean "use the shared pathway" — it means "the shared pathway is already enough, no correction needed." That structural guarantee is what makes alpha readable as a scientific measurement rather than just an optimization artifact: the architecture forces the model to reveal, per gene, whether the species-agnostic prediction suffices.

### Why four structural loss terms, not one

The residual decoder is the right architecture, but left to standard reconstruction loss alone, it will silently fail in at least four distinct ways — each of which was discovered through careful gradient-flow analysis and each of which requires its own structural fix. The intellectual substance of Chimera is not the decoder design alone but the understanding of *why* each failure mode arises from the mathematics of variational inference under a constrained decoder, and the minimal structural intervention that prevents it.

**Failure mode 1: the residual absorbs everything.** The species-aware residual pathway is strictly more expressive than the species-agnostic shared pathway. Under reconstruction loss alone, the optimal strategy is for the residual to fit the data entirely and alpha to drift to zero for every gene. The L1 sparsity penalty on the residual prevents this by making the correction expensive — it activates only where the shared pathway genuinely cannot fit.

**Failure mode 2: the shared decoder collapses.** Even with the L1, the shared decoder has no loss term that depends only on its own output. Once the residual fits the data well, the joint reconstruction loss is satisfied and the shared decoder receives vanishing gradient — it can stay at noise indefinitely while the residual does all the work. An auxiliary shared-only reconstruction loss gives the shared decoder an independent training signal, guaranteeing it learns the species-averaged expression pattern.

**Failure mode 3: alpha collapses to zero for divergent genes.** A derivation of the alpha gradient at the L1 first-order condition reveals that the reconstruction gradient is strictly non-negative for any gene with a nonzero residual — both the reconstruction loss and the L1 cost push alpha down monotonically, with no opposing force. Without intervention, alpha for every divergent gene asymptotes to the sigmoid lower bound, and the continuous-alpha story collapses into a binary indicator. An alpha anchor penalty in logit space provides the only opposing force, producing a continuous interior equilibrium below the prior.

**Failure mode 4: alpha is one-directional.** Even with the first three fixes, the alpha gradient is non-negative everywhere — alpha can be pushed below the prior but never above it. The `delta_alpha > 0` regime is architecturally unreachable, and applications that depend on bidirectional delta-alpha produce one-sided outputs while looking superficially correct. The fit-quality reward (FQR) adds the missing upward force by comparing, per gene, how well the shared prediction outperforms a per-species gene-mean null baseline, and rewarding alpha when the shared prediction wins. The comparison is computed under `torch.no_grad()` so the signal flows only through alpha and never trains the shared decoder — a surgical, one-directional intervention that creates the upward force without destabilizing anything else.

Each of these four terms protects against a specific failure mode. Remove any one and the corresponding failure reopens — silently, because cell-level integration metrics remain unaffected while the per-gene outputs quietly become meaningless. The full loss, with all four structural terms mandatory, is what makes alpha a trustworthy scientific measurement rather than an optimization artifact that happens to live in the right range.

### Adversarial training and z-leakage

Because the shared decoder does not receive a species label, the encoder has no incentive to remove species information from the latent space — and the auxiliary loss actively rewards keeping it, since the shared decoder could use leaked species information to fit each species individually. This z-leakage is structurally incentivized, not merely a possibility to monitor, and it directly contaminates the delta-alpha signal: if the shared decoder exploits species information in the latent space to fit divergent genes well, FQR falsely rewards alpha for those genes and the `delta_alpha > 0` distribution mixes genuine conservation signal with leakage artifact. The fix is mandatory adversarial training — a species classifier on the latent representation with gradient reversal — complemented by detaching the latent representation on the shared pathway input to prevent the feedback loop that creates new leakage. Both are required; neither alone is sufficient.

## Datasets

Chimera operates on multi-species scRNA-seq count matrices linked by one-to-one ortholog mappings from Ensembl BioMart, which gives 7,998–8,879 shared genes for the liver cohorts used here and 14,640–15,732 for the testis cohorts. Each cell carries a species label and a batch label (donor, sample, or sequencing run). Cell-type annotations are used for downstream evaluation but are not required for training.

## Installation

Python 3.12 is used for all results. The supported route is the environment notebook
[`notebooks/00_environment.ipynb`](notebooks/00_environment.ipynb), which creates the
`chimera_env` mamba environment, installs PyTorch for the platform (CPU, CUDA or Apple MPS)
and then this package in editable mode with its extras. Manually:

```bash
mamba create -n chimera_env python=3.12 pip
mamba activate chimera_env
pip install torch==2.11.0
pip install -e ".[dev,graph,attributions,protein,evolution,data,baselines]"
pytest -q
```

Versions of the core dependencies are pinned in [`pyproject.toml`](pyproject.toml) to those
the checkpoints were trained with. On Linux x86-64,
`bash revision/envs/build_env_from_lock.sh chimera_env` builds the CPU environment with every
package at the version of the lock file used for the revision analyses. Training defaults to `--accelerator mps`; on Linux pass
`--accelerator cpu` or `--accelerator gpu`.

## Data

All analysed data are public; no new data were generated.

| Data | Source |
|---|---|
| Liver, human, macaque, mouse | Guilliams et al. 2022, *Cell* 185:379, GEO GSE192742 |
| Liver, zebrafish | Wang et al. 2023, *Nucleic Acids Res.*, doi:10.1093/nar/gkac633, GEO GSE198832 |
| Testis, human, macaque, mouse | Murat et al. 2023, *Nature*, doi:10.1038/s41586-022-05547-7, E-MTAB-11063, E-MTAB-11068, E-MTAB-11071 |
| Processed CAMEX bundle (raw counts of the above) | Guo, Huang and Zhang 2026, *Nat. Commun.* 17:3017, figshare doi:10.6084/m9.figshare.31131808.v1 |
| ESM-2 protein embeddings (`esm2_t48_15B_UR50D`) | SATURN, Rosen et al. 2024, *Nat. Methods* |
| Ortholog tables | Ensembl BioMart, release 110 |

The processed AnnData files, ortholog tables, trained checkpoints and per-gene outputs used in
the paper are archived at [DOI]. See [`REPRODUCE.md`](REPRODUCE.md) for the expected layout.
The `panc_*` and `pancreas_*` names in code and file names refer to the **liver** cohorts.

## Reproducing the results

[`REPRODUCE.md`](REPRODUCE.md) describes how to regenerate every analysis from the archived
artefacts. The canonical training recipes are listed in
[`configs/canonical_recipes.tsv`](configs/canonical_recipes.tsv), and
[`revision/reproduce_smoke.sh`](revision/reproduce_smoke.sh) checks a checkout against the
archived runs.

## Citation

If you use Chimera, please cite the paper (reference added on publication) and the software;
see [`CITATION.cff`](CITATION.cff).

## Built on scvi-tools

Chimera is built on top of [scvi-tools](https://scvi-tools.org/) and follows its conventions for model registration, AnnData handling, and training plans. If you have used scVI, the API will feel familiar.

## License

BSD 3-Clause; see [LICENSE](LICENSE).
