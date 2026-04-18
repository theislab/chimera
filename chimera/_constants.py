"""Chimera v2.4 — Hyperparameter defaults, feature dimensions, architectural constants.

All values sourced from:
  - ortholog_aware_integration_guide_initial.md, "Hyperparameter ranges (v2.4)" table
  - critical_reexamination_modules.md, Component 10 "Updated hyperparameter table"
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------
LAMBDA_AUX: float = 0.3
"""Weight for L_shared_aux (shared-only reconstruction). Collapse-prevention #2."""

LAMBDA_RES: float = 0.05
"""Weight for L_residual_sparsity (L1 on residual). Collapse-prevention #1."""

LAMBDA_ANCHOR: float = 0.005
"""Weight for L_alpha_anchor (correction.pow(2).sum()). Collapse-prevention #3."""

LAMBDA_FIT: float = 0.01
"""Weight for L_fit_reward (FQR). Asymmetry-repair #4."""

LAMBDA_DISP: float = 1e-4
"""Weight for dispersion regularization."""

LAMBDA_MMD: float = 0.0
"""Weight for MMD (Maximum Mean Discrepancy) loss on z between species.
When > 0, directly penalizes species-stratified latent distributions,
producing species-mixed z embeddings. Recommended range: [10, 200].
Set to 0 to disable (backward compatible)."""

# ---------------------------------------------------------------------------
# Adversarial training
# ---------------------------------------------------------------------------
ADV_WEIGHT: float = 0.1
"""Adversarial species classifier weight. Mandatory when lambda_aux > 0.1 AND lambda_fit > 0."""

# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------
GRAD_CLIP_MAX_NORM: float = 1.0
"""Max gradient norm for gradient clipping."""

# ---------------------------------------------------------------------------
# GIN (Gene Integration Network) architecture
# ---------------------------------------------------------------------------
GIN_N_FEATURES: int = 72
"""Default GIN input features (64 ESM-2 PCA + 8 scalars). Overridden by actual gene_features.shape[1]."""

GIN_N_HIDDEN: int = 32
"""GIN hidden layer size. 72 -> 32 -> 1. Total params: ~2,369."""

GIN_LR_FACTOR: float = 0.1
"""GIN learning rate = base_lr * GIN_LR_FACTOR. Default: 1e-4 when base_lr=1e-3."""

# ---------------------------------------------------------------------------
# FQR (Fit-Quality Reward) parameters
# ---------------------------------------------------------------------------
MIN_CELLS_FQR: int = 20
"""Minimum expressing cells per gene per minibatch to compute reliable S_g."""

SG_EMA_BETA: float = 0.9
"""Exponential moving average decay for S_g. Smooths per-minibatch ReLU-clip bias."""

# ---------------------------------------------------------------------------
# KL parameters
# ---------------------------------------------------------------------------
FREE_BITS: float = 0.5
"""Free-bits threshold for KL per dimension. Below this, KL is not penalized."""

# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------
WARMUP_ADV_EPOCHS: int = 15
"""Start adversarial training after this many epochs."""

WARMUP_KL_EPOCHS: int = 20
"""Number of epochs to linearly ramp KL weight from 0 to 1."""

WARMUP_FQR_EPOCHS: int = 20
"""Start FQR after this many epochs. Must be >= WARMUP_KL_EPOCHS."""

MIN_TRAINING_EPOCHS: int = 150
"""Minimum epochs before early stopping can trigger."""

LR_WARMUP_EPOCHS: int = 5
"""Number of epochs for linear learning rate warmup."""

# ---------------------------------------------------------------------------
# Batch offset
# ---------------------------------------------------------------------------
BATCH_OFFSET_WEIGHT_DECAY: float = 1e-4
"""Weight decay on batch_offset_raw to damp null-space drift."""

# ---------------------------------------------------------------------------
# Model architecture defaults
# ---------------------------------------------------------------------------
N_LATENT: int = 10
"""Latent space dimensionality."""

N_HIDDEN: int = 128
"""Hidden layer size for encoder and decoder MLPs."""

N_LAYERS_ENCODER: int = 2
"""Number of hidden layers in the encoder."""

N_LAYERS_DECODER: int = 2
"""Number of hidden layers in the decoder."""

DROPOUT_RATE: float = 0.1
"""Dropout rate for encoder."""

# ---------------------------------------------------------------------------
# Learning rates
# ---------------------------------------------------------------------------
BASE_LR: float = 1e-3
"""Base learning rate for encoder, decoder, dispersion."""

# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------


@dataclass
class AblationConfig:
    """Configuration for ablation experiments.

    Each config disables or modifies specific components to test their necessity.
    """

    name: str
    description: str
    lambda_aux: float = LAMBDA_AUX
    lambda_res: float = LAMBDA_RES
    lambda_anchor: float = LAMBDA_ANCHOR
    lambda_fit: float = LAMBDA_FIT
    adv_weight: float = ADV_WEIGHT
    use_gin: bool = True
    use_fqr: bool = True
    use_anchor: bool = True
    use_adversarial: bool = True
    use_z_detach: bool = True
    use_shared_aux: bool = True
    use_residual_sparsity: bool = True
    extra: Dict[str, object] = field(default_factory=dict)


# Ablation configs A-M per round2 spec (dataclass format, kept for backward compat)
ABLATION_CONFIGS_DC: Dict[str, AblationConfig] = {
    "A": AblationConfig(
        name="A",
        description="Full v2.4 (default)",
    ),
    "B": AblationConfig(
        name="B",
        description="No shared-aux → Failure mode 2 (shared decoder collapse)",
        use_shared_aux=False,
        lambda_aux=0.0,
    ),
    "C": AblationConfig(
        name="C",
        description="No L1 sparsity → Failure mode 1 (alpha drift)",
        use_residual_sparsity=False,
        lambda_res=0.0,
    ),
    "D": AblationConfig(
        name="D",
        description="No anchor → Failure mode 3 (alpha-collapse for divergent genes)",
        use_anchor=False,
        lambda_anchor=0.0,
    ),
    "F": AblationConfig(
        name="F",
        description="No adversarial → z-leakage contaminates FQR",
        use_adversarial=False,
        adv_weight=0.0,
    ),
    "G": AblationConfig(
        name="G",
        description="No z.detach() on shared path → L_shared_aux trains encoder to leak species",
        use_z_detach=False,
    ),
    "H": AblationConfig(
        name="H",
        description="Higher lambda_aux=0.5",
        lambda_aux=0.5,
    ),
    "I": AblationConfig(
        name="I",
        description="Higher lambda_res=0.1",
        lambda_res=0.1,
    ),
    "J": AblationConfig(
        name="J",
        description="No FQR → Failure mode 4 (delta_alpha > 0 unreachable)",
        use_fqr=False,
        lambda_fit=0.0,
    ),
    "K": AblationConfig(
        name="K",
        description="Fixed alpha (no GIN, no FQR, no anchor) — minimal variant",
        use_gin=False,
        use_fqr=False,
        use_anchor=False,
        lambda_fit=0.0,
        lambda_anchor=0.0,
    ),
    "L": AblationConfig(
        name="L",
        description="Unified adaptive-target anchor (replaces separate anchor + FQR)",
        extra={"use_unified_anchor": True, "unified_anchor_c": 1.0, "unified_anchor_lambda": 0.005},
    ),
    "M": AblationConfig(
        name="M",
        description="Per-gene nn.Parameter alpha (no GIN) — tests GIN value",
        use_gin=False,
        extra={"use_per_gene_parameter_alpha": True},
    ),
}

# ---------------------------------------------------------------------------
# Dict-based ablation configs for from_config factory (Task 7.7)
# ---------------------------------------------------------------------------
# Each config is a dict with core loss keys (lambda_aux, lambda_res,
# lambda_anchor, lambda_fit) and optional meta-keys that from_config pops
# before constructing the model:
#   freeze_alpha, random_gene_features, no_esm2, per_gene_alpha,
#   fqr_null_variant, unified_anchor_lambda, note
# ---------------------------------------------------------------------------

ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    # A: Full v2.4 (default) — all components enabled at default lambdas
    "A": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": LAMBDA_FIT,
        "note": "Full v2.4 (default)",
    },
    # B: No shared-aux loss — tests failure mode 2 (shared decoder collapse)
    "B": {
        "lambda_aux": 0.0,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": LAMBDA_FIT,
        "note": "No shared-aux (lambda_aux=0)",
    },
    # B': Frozen alpha at prior — tests whether GIN adaptation matters
    "B_prime": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": LAMBDA_FIT,
        "freeze_alpha": "prior",
        "note": "Alpha frozen at conservation prior (no GIN correction)",
    },
    # C: No L1 sparsity — tests failure mode 1 (alpha drift)
    "C": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": 0.0,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": LAMBDA_FIT,
        "note": "No L1 sparsity (lambda_res=0)",
    },
    # C': Alpha frozen at 0.5 — tests prior information value
    "C_prime": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": LAMBDA_FIT,
        "freeze_alpha": 0.5,
        "note": "Alpha frozen at 0.5 (uninformative prior)",
    },
    # C'': Random gene features — tests ESM-2 embedding value
    "C_double_prime": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": LAMBDA_FIT,
        "random_gene_features": True,
        "note": "Random gene features (ESM-2 replaced with N(0,1))",
    },
    # C''': No ESM-2 dims — tests ESM-2 PCA contribution vs scalar features
    "C_triple_prime": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": LAMBDA_FIT,
        "no_esm2": True,
        "note": "No ESM-2 (first 64 dims zeroed, scalar features only)",
    },
    # D: No anchor — tests failure mode 3 (alpha-collapse for divergent genes)
    "D": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": 0.0,
        "lambda_fit": LAMBDA_FIT,
        "note": "No anchor (lambda_anchor=0)",
    },
    # F: No adversarial — z-leakage contaminates FQR
    "F": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": LAMBDA_FIT,
        "note": "No adversarial (adv_weight handled by training plan)",
    },
    # G: No z.detach() — L_shared_aux trains encoder to leak species
    "G": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": LAMBDA_FIT,
        "note": "No z.detach() on shared path",
    },
    # G': FQR null variant — tests FQR signal quality
    "G_prime": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": LAMBDA_FIT,
        "fqr_null_variant": True,
        "note": "FQR null variant (S_g zeroed, correction still penalized)",
    },
    # H: Higher lambda_aux=0.5
    "H": {
        "lambda_aux": 0.5,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": LAMBDA_FIT,
        "note": "Higher lambda_aux=0.5",
    },
    # J: No FQR — failure mode 4 (delta_alpha > 0 unreachable)
    "J": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": 0.0,
        "note": "No FQR (lambda_fit=0)",
    },
    # K: Fixed alpha (no GIN, no FQR, no anchor) — minimal variant
    "K": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": 0.0,
        "lambda_fit": 0.0,
        "freeze_alpha": "prior",
        "note": "Fixed alpha, no FQR, no anchor — minimal variant",
    },
    # L: Unified adaptive-target anchor (replaces separate anchor + FQR)
    "L": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": LAMBDA_FIT,
        "unified_anchor_lambda": 0.005,
        "note": "Unified adaptive-target anchor (replaces separate anchor + FQR)",
    },
    # M: Per-gene nn.Parameter alpha (no GIN) — tests GIN value
    "M": {
        "lambda_aux": LAMBDA_AUX,
        "lambda_res": LAMBDA_RES,
        "lambda_anchor": LAMBDA_ANCHOR,
        "lambda_fit": LAMBDA_FIT,
        "per_gene_alpha": True,
        "note": "Per-gene nn.Parameter alpha (no GIN) — tests GIN value",
    },
}
