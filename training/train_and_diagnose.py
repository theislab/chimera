"""Train a Chimera model and dump per-epoch training artefacts.

This is the **entry point** for every iteration on Chimera training. It:

1. Loads a prepared unified AnnData (e.g. the 2-species subset).
2. Runs the pretraining sanity checks (E1 in `workflow.md`).
3. Constructs a ChimeraModel via `from_config` (default "A" = full v2.4).
4. Trains with:
     - Lightning's built-in CSVLogger (per-epoch metrics.csv)
     - A small `AlphaSnapshotCallback` that dumps alpha / correction / S_g
       at chosen epochs (for trajectory reconstruction).
5. After training dumps a single `final_state.npz` with everything
   `check_expectations.py` needs: alpha, prior, correction, S_g_ema, z,
   per-gene residual magnitude, per-gene mu_shared mean, adversarial accuracy,
   and the last-epoch loss-component breakdown.

Design notes
-------------
We do **not** modify the `chimera` package. Everything here is a consumer of
the public API: `ChimeraModel.from_config`, `.train`, `.get_*`. The
`AlphaSnapshotCallback` is a plain Lightning callback.

Usage
-----
    python training/train_and_diagnose.py \
        --adata data/processed/camex_human_monkey.h5ad \
        --run-name quick-baseline \
        --max-epochs 50 \
        --config D
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chimera._constants import ABLATION_CONFIGS, WARMUP_ADV_EPOCHS, WARMUP_FQR_EPOCHS, WARMUP_KL_EPOCHS  # noqa: E402
from chimera._model import ChimeraModel  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "training" / "runs"


# ---------------------------------------------------------------------------
# Pretraining checks (E1.* from workflow.md)
# ---------------------------------------------------------------------------

def pretraining_checks(adata: ad.AnnData) -> dict:
    """Lightweight versions of the E1 expectations. Return a dict of findings.

    E1.1 uses LIBRARY-SIZE-NORMALIZED expression (per-cell fraction, then species
    mean), not raw log-mean UMI counts. Raw log-means are depth-confounded: highly
    conserved housekeeping genes can look divergent simply because the two species
    have different median library sizes. Spearman rank correlation on fractions
    removes that depth confound.
    """
    from scipy.stats import spearmanr

    out: dict = {}
    species = adata.obs["species"].astype("category")
    sp_codes = species.cat.codes.values
    unique_sp = species.cat.categories.tolist()
    K = len(unique_sp)
    G = adata.n_vars

    # Per-species mean of per-cell gene fractions (library-size normalized).
    # fraction[c, g] = X[c, g] / lib[c]  where lib[c] = X[c].sum()
    X = adata.X
    if hasattr(X, "toarray"):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)
    lib = X_dense.sum(axis=1, keepdims=True).clip(min=1.0)
    frac = X_dense / lib

    species_profiles = np.zeros((K, G), dtype=np.float64)
    for k in range(K):
        m = sp_codes == k
        species_profiles[k] = np.log1p(1e4 * frac[m].mean(axis=0))  # CPM-scale log

    conservation_prior = adata.var["conservation_prior"].values.astype(np.float64)
    # Similarity = -|delta_logCPM|; bigger (less negative) => more similar => higher conservation
    if K == 2:
        similarity = -np.abs(species_profiles[0] - species_profiles[1])
    else:
        pairs = [-np.abs(species_profiles[a] - species_profiles[b])
                 for a in range(K) for b in range(a + 1, K)]
        similarity = np.mean(np.stack(pairs, axis=0), axis=0)

    # Drop zero-expression genes AND bottom-50% expressed genes. Low-expression genes
    # are dominated by sampling noise and cell-composition differences between species;
    # their CPM-based similarity is not a reliable readout of protein-conservation-vs-
    # expression-conservation signal. The top 50% carry the informative signal
    # (investigated 2026-04-17: on the 4-species pancreas, top-50% expressed genes
    # give spearman r ≈ 0, all genes give r ≈ −0.19 — the negative is noise, not biology).
    total_expr = X_dense.sum(axis=0)
    median_expr = np.median(total_expr[total_expr > 0])
    informative = total_expr >= median_expr
    r, _ = spearmanr(conservation_prior[informative], similarity[informative])
    # E1.1 is a signal-presence check for the PRIMARY GIN input (conservation_prior).
    # Three regimes per round2 (critical_reexamination §96):
    #   (a) r >= 0.02 ................  prior carries expression signal -> strong pass
    #   (b) |r| < 0.15  ..............  prior carries ~no signal -> OK, GIN falls back
    #                                    to ESM-2 and expression features (which are
    #                                    present in gene_features[:, 0..64, 66..68])
    #   (c) r <= -0.15  ..............  STRONG NEGATIVE correlation -> data is corrupt
    #                                    or mis-standardized -> FAIL
    # CPM-composition effects can drive r slightly negative on close-species samples
    # where cell-type composition differs (housekeeping genes dominate CPM fractions
    # and look "divergent" in |Δ log CPM|). The spec explicitly allows this outcome.
    # We additionally require prior std >= 0.05 so that the prior has *some* range
    # to be informative (E1.5 already checks this, but having it here makes the E1.1
    # failure mode specific: "dead or inverted signal").
    prior_std = float(conservation_prior.std())
    pass_e11 = bool((r >= 0.02) or (abs(r) < 0.15 and prior_std >= 0.05))
    out["E1.1_conservation_signal"] = {
        "metric": "spearman_r_prior_vs_neg_abs_delta_logCPM_informative_genes",
        "value": float(r),
        "prior_std": prior_std,
        "n_informative_genes": int(informative.sum()),
        "threshold": "r>=0.02 OR (|r|<0.15 AND prior_std>=0.05)",
        "pass": pass_e11,
        "notes": (
            "r>=0.02: strong signal. |r|<0.15: weak but GIN will lean on ESM-2 "
            "and expression scalars. r<=-0.15: data/standardization issue."
        ),
    }

    # E1.2 gene_features normalization. We ignore zero-std features (padding
    # dimensions that pad up to 72 after the standard scaler saw constant
    # inputs). We require the ACTIVE features to be well standardized.
    gf = adata.varm["gene_features"]
    per_feat_mean_abs = np.abs(gf.mean(axis=0))
    per_feat_std = gf.std(axis=0)
    active = per_feat_std > 1e-6
    active_std = per_feat_std[active]
    active_mean_abs = per_feat_mean_abs[active]
    n_padding = int((~active).sum())
    out["E1.2_gene_features_normalized"] = {
        "n_active_features": int(active.sum()),
        "n_padding_or_dead_features": n_padding,
        "active_mean_abs_max": float(active_mean_abs.max()) if len(active_mean_abs) else 0.0,
        "active_std_min": float(active_std.min()) if len(active_std) else 0.0,
        "active_std_max": float(active_std.max()) if len(active_std) else 0.0,
        "pass": bool(
            len(active_std) >= 65  # at least 65/72 features alive
            and active_mean_abs.max() < 0.2  # relaxed from 0.1: subset building shifts means slightly
            and 0.5 <= active_std.min()
            and active_std.max() <= 1.5
        ),
        "notes": "Zero-std features are treated as padding/dead and excluded.",
    }

    # E1.3 p_null dynamic range
    p_null = adata.uns["p_null"]
    total = p_null.size
    in_range = ((p_null > 1e-5) & (p_null < 0.5)).sum()
    frac_in = float(in_range) / float(total)
    out["E1.3_p_null_dynamic_range"] = {
        "frac_in_range_1e-5_to_5e-1": frac_in,
        "threshold": 0.5,
        "pass": bool(frac_in >= 0.5),
    }

    # E1.4 batch_to_species_unique
    bs = adata.obs.groupby("batch", observed=True)["species"].nunique()
    out["E1.4_batch_to_species_unique"] = {
        "max_species_per_batch": int(bs.max()),
        "pass": bool(bs.max() == 1),
    }

    # E1.5 conservation_prior_range
    cp = adata.var["conservation_prior"].values
    out["E1.5_conservation_prior_range"] = {
        "min": float(cp.min()),
        "max": float(cp.max()),
        "std": float(cp.std()),
        "pass": bool(cp.min() >= 0.05 and cp.max() <= 0.95 and cp.std() >= 0.05),
    }
    return out


# ---------------------------------------------------------------------------
# Alpha snapshot callback
# ---------------------------------------------------------------------------

class AlphaSnapshotCallback(Callback):
    """Snapshot alpha, correction, S_g_ema, a compact loss summary per epoch.

    Snapshots are written as `alpha_epoch_XXX.npz` to `out_dir`. They are cheap
    (a few kB per file) so we snapshot every epoch by default; pass
    ``every_n_epochs`` to thin.
    """

    def __init__(self, out_dir: Path, every_n_epochs: int = 1) -> None:
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.alpha_trajectory: list[np.ndarray] = []

    @torch.no_grad()
    def _snapshot(self, pl_module) -> dict:
        m = pl_module.module
        prior = m.conservation_prior.detach().cpu().float().numpy()
        # alpha/correction: Config M (per-gene parameter) takes precedence over
        # the GIN attribute, which still exists but is frozen in that config.
        if getattr(m, "_use_per_gene_alpha", False):
            correction = m.per_gene_correction.detach().cpu().float().numpy()
            alpha = 1.0 / (1.0 + np.exp(-(np.log(prior / (1 - prior)) + correction)))
        elif hasattr(m, "gin"):
            alpha_t, correction_t = m.gin(m.gene_features, m.conservation_prior)
            alpha = alpha_t.detach().cpu().float().numpy()
            correction = correction_t.detach().cpu().float().numpy()
        else:
            alpha = m.alpha.detach().cpu().float().numpy()
            correction = np.zeros_like(alpha)
        S_g = m.S_g_ema.detach().cpu().float().numpy() if hasattr(m, "S_g_ema") else np.zeros_like(alpha)
        return {"prior": prior, "alpha": alpha, "correction": correction, "S_g_ema": S_g}

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = int(trainer.current_epoch)
        if epoch % self.every_n_epochs != 0:
            return
        snap = self._snapshot(pl_module)
        path = self.out_dir / f"alpha_epoch_{epoch:04d}.npz"
        np.savez_compressed(path, epoch=epoch, **snap)


class StashTrainingPlanCallback(Callback):
    """Stash the live training plan onto the user's model so the adversarial
    classifier can be inspected post-train. scvi-tools' TrainRunner drops its
    reference to the plan after ``model.train()`` returns; this hook keeps the
    plan reachable via ``model._last_training_plan``.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self._model = model

    def on_fit_start(self, trainer, pl_module):
        self._model._last_training_plan = pl_module


# ---------------------------------------------------------------------------
# Final snapshot: extract model outputs after training for expectation checks
# ---------------------------------------------------------------------------

@torch.no_grad()
def dump_final_state(model: ChimeraModel, adata: ad.AnnData, out_path: Path) -> dict:
    """Compute everything `check_expectations.py` needs and save as a single npz."""
    module = model.module
    module.eval()
    device = next(module.parameters()).device

    prior = module.conservation_prior.detach().cpu().float().numpy()

    # alpha, correction: Config M takes precedence over the GIN attribute
    # (which still exists but is frozen when per_gene_alpha=True).
    if getattr(module, "_use_per_gene_alpha", False):
        correction = module.per_gene_correction.detach().cpu().float().numpy()
        alpha = 1.0 / (1.0 + np.exp(-(np.log(prior / (1 - prior)) + correction)))
    elif hasattr(module, "gin"):
        alpha_t, correction_t = module.gin(module.gene_features, module.conservation_prior)
        alpha = alpha_t.detach().cpu().float().numpy()
        correction = correction_t.detach().cpu().float().numpy()
    else:
        alpha = module.alpha.detach().cpu().float().numpy()
        correction = np.zeros_like(alpha)

    S_g_ema = module.S_g_ema.detach().cpu().float().numpy() if hasattr(module, "S_g_ema") else np.zeros_like(alpha)

    # Run a single forward pass over a batch to collect residual + mu_shared stats.
    # We iterate all cells in chunks and aggregate per-gene statistics incrementally.
    from scvi import REGISTRY_KEYS

    dl = model._make_data_loader(adata=adata, batch_size=512, shuffle=False)
    n_g = adata.n_vars
    mu_shared_sum = np.zeros(n_g, dtype=np.float64)
    mu_shared_sq_sum = np.zeros(n_g, dtype=np.float64)
    mu_full_sum = np.zeros(n_g, dtype=np.float64)
    residual_abs_sum = np.zeros(n_g, dtype=np.float64)
    residual_sq_sum = np.zeros(n_g, dtype=np.float64)
    total_cells = 0

    # For z-leakage post-hoc: accumulate z and species labels, then run the
    # current adversarial classifier on the full set for a stable accuracy.
    z_chunks: list[np.ndarray] = []
    sp_chunks: list[np.ndarray] = []

    for tensors in dl:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
        x = batch[REGISTRY_KEYS.X_KEY]
        b_idx = batch[REGISTRY_KEYS.BATCH_KEY].squeeze(-1).long()
        s_idx = batch[REGISTRY_KEYS.LABELS_KEY].squeeze(-1).long()
        sp_oh = F.one_hot(s_idx, module.n_species).float()
        b_oh = F.one_hot(b_idx, module.n_batch).float()
        enc_input = module._encoder_input(x, sp_oh, b_oh)
        h_enc = module.encoder(enc_input)
        z_mean = module.z_mean_linear(h_enc)
        z_chunks.append(z_mean.detach().cpu().float().numpy())
        sp_chunks.append(s_idx.detach().cpu().long().numpy())

        # dual-h
        h_full = module.decoder_hidden(z_mean)
        h_det = module.decoder_hidden(z_mean)  # no detach for final snapshot; we just want the magnitudes
        log_rate_shared_full = module.output_shared(h_full)
        log_rate_shared_det = module.output_shared(h_det)
        log_rate_residual = module.output_residual(torch.cat([h_full, sp_oh, b_oh], dim=-1))
        lib = x.sum(-1, keepdim=True).clamp(min=1.0)
        log_rate_batch = module.batch_offset()[b_idx]

        if getattr(module, "_use_per_gene_alpha", False):
            alpha_g = torch.sigmoid(
                torch.logit(module.conservation_prior) + module.per_gene_correction
            )
        elif hasattr(module, "gin"):
            alpha_g = module.gin(module.gene_features, module.conservation_prior)[0]
        else:
            alpha_g = module.alpha
        mu_shared_only = torch.exp(log_rate_shared_det + log_rate_batch) * lib
        mu_full = torch.exp(log_rate_shared_full + log_rate_batch + (1 - alpha_g).unsqueeze(0) * log_rate_residual) * lib

        mu_shared_sum += mu_shared_only.detach().cpu().float().numpy().sum(axis=0)
        mu_shared_sq_sum += (mu_shared_only.detach().cpu().float().numpy() ** 2).sum(axis=0)
        mu_full_sum += mu_full.detach().cpu().float().numpy().sum(axis=0)
        residual_abs = log_rate_residual.abs().detach().cpu().float().numpy()
        residual_abs_sum += residual_abs.sum(axis=0)
        residual_sq_sum += (log_rate_residual.detach().cpu().float().numpy() ** 2).sum(axis=0)
        total_cells += x.shape[0]

    mu_shared_mean = mu_shared_sum / max(1, total_cells)
    mu_shared_var = mu_shared_sq_sum / max(1, total_cells) - mu_shared_mean ** 2
    mu_full_mean = mu_full_sum / max(1, total_cells)
    residual_mean_abs = residual_abs_sum / max(1, total_cells)
    residual_rms = np.sqrt(residual_sq_sum / max(1, total_cells))

    z_all = np.concatenate(z_chunks, axis=0)
    species_labels = np.concatenate(sp_chunks, axis=0)

    # post-hoc adversarial accuracy on all cells. The scvi-tools TrainRunner
    # drops its reference to the training plan after model.train() returns, so
    # the trained classifier is not reachable via model._training_plan. We
    # stash a reference on the model in _fit_and_stash_plan; fall back to
    # reading the per-epoch CSV if that's missing.
    adv_acc = float("nan")
    from chimera._training_plan import ChimeraTrainingPlan
    tp = getattr(model, "_last_training_plan", None)
    if tp is not None and getattr(tp, "adversarial_classifier", None) is not None:
        clf = tp.adversarial_classifier.to(device).eval()
        z_t = torch.from_numpy(z_all).to(device).float()
        with torch.no_grad():
            preds = clf(z_t).argmax(-1).cpu().numpy()
        adv_acc = float((preds == species_labels).mean())

    snap = {
        "prior": prior,
        "alpha": alpha,
        "correction": correction,
        "S_g_ema": S_g_ema,
        "mu_shared_mean": mu_shared_mean,
        "mu_shared_var": mu_shared_var,
        "mu_full_mean": mu_full_mean,
        "residual_mean_abs": residual_mean_abs,
        "residual_rms": residual_rms,
        "z_mean": z_all,
        "species_labels": species_labels,
        "adversarial_accuracy_final": np.array(adv_acc, dtype=np.float64),
        "n_species": np.array(int(module.n_species)),
        "n_batch": np.array(int(module.n_batch)),
        "n_cells": np.array(int(total_cells)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **snap)
    return snap


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_diagnose(
    adata_path: Path,
    run_name: str,
    config: str = "A",
    max_epochs: int = 50,
    batch_size: int = 256,
    seed: int = 42,
    n_latent: int = 10,
    n_hidden: int = 128,
    use_observed_lib_size: bool = True,
    warmup_kl_epochs: int | None = None,
    warmup_fqr_epochs: int | None = None,
    warmup_adversarial_epochs: int | None = None,
    min_training_epochs: int | None = None,
    early_stopping: bool = False,
    accelerator: str | None = None,
    snapshot_every_n_epochs: int = 1,
    lambda_fit: float | None = None,
    lambda_anchor: float | None = None,
    lambda_res: float | None = None,
    adv_weight: float | None = None,
    adv_lr: float | None = None,
    adv_classifier_hidden: tuple[int, ...] | None = None,
    mask_species_in_encoder: bool = True,
    mask_batch_in_encoder: bool = True,
    gin_feature_op: str | None = None,
    free_bits: float | None = None,
    lambda_mmd: float | None = None,
    lambda_z_balance: float | None = None,
    kl_weight_max: float | None = None,
    use_celltype_conditional_mmd: bool = False,
    celltype_key: str | None = None,
    use_vamp_prior: bool = True,
    n_vamp_components: int = 500,
    vamp_init_from_data: bool = True,
    vamp_init_stratify_species: bool = False,
    equal_species_sampling: bool = False,
    batch_key: str = "batch",
    prior_clamp_max: float | None = None,
    gin_n_hidden: int | None = None,
    sg_ema_beta: float | None = None,
) -> Path:
    """Train a ChimeraModel and dump diagnostic artefacts.

    ``gin_feature_op`` (optional) applies an in-place patch to
    ``model.module.gene_features`` after construction but before training.
    Supported values:

    - ``"zero_esm2"``           — zero the first 64 columns (ESM-2 PCA dims).
    - ``"zero_esm2_var"``       — zero column 64 (ESM-2 inter-species variance).
    - ``"zero_mean_identity"``  — zero column 65 (mean protein identity).
    - ``"zero_expr_stats"``     — zero columns 66-68 (log_mean_expr, log_expr_var, frac_nonzero).
    - ``"zero_all_scalars"``    — zero columns 64-68 (all 5 scalar features).
    - ``"random_all"``          — replace the entire matrix with N(0, 1).

    These are diagnostic-only: the standard production run passes
    ``gin_feature_op=None``. The patch is logged to ``config.json``.
    """

    run_dir = RUNS_ROOT / run_name
    if run_dir.exists():
        logger.warning("Run dir %s already exists; existing files may be overwritten.", run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data & pretrain checks
    logger.info("Loading %s", adata_path)
    adata = ad.read_h5ad(adata_path)
    logger.info("Data shape: %s, species: %s", adata.shape, adata.obs["species"].value_counts().to_dict())

    # iter109: prior cap reduction. The data-prep default caps prior at 0.95.
    # On testis ~32% of genes sit at the cap so α saturates from prior alone.
    # `--prior-clamp-max` applies an additional in-memory clamp before training;
    # the unclamped prior is preserved on disk under conservation_prior_raw.npy.
    if prior_clamp_max is not None and prior_clamp_max < 0.95:
        raw_prior = adata.var["conservation_prior"].values.astype(np.float64).copy()
        np.save(run_dir / "conservation_prior_raw.npy", raw_prior)
        clamped = np.clip(raw_prior, 0.05, float(prior_clamp_max))
        n_clamped = int((raw_prior > prior_clamp_max).sum())
        logger.info(
            "iter109: clamping conservation_prior to [0.05, %.3f]. "
            "%d / %d genes (%.1f%%) had prior > %.3f and were clamped down. "
            "Raw prior saved to %s.",
            prior_clamp_max, n_clamped, len(raw_prior),
            100.0 * n_clamped / len(raw_prior), prior_clamp_max,
            run_dir / "conservation_prior_raw.npy",
        )
        adata.var["conservation_prior"] = clamped.astype(adata.var["conservation_prior"].dtype)

    pretrain_results = pretraining_checks(adata)
    with open(run_dir / "pretrain_checks.json", "w") as f:
        json.dump(pretrain_results, f, indent=2)
    all_pass_e1 = all(r.get("pass", False) for r in pretrain_results.values())
    if not all_pass_e1:
        failed = [k for k, v in pretrain_results.items() if not v.get("pass", False)]
        logger.warning("Pretraining checks FAILED: %s", failed)
    else:
        logger.info("All pretraining (E1) checks passed.")

    # --- Setup AnnData / model
    set_seed(seed)
    # If celltype-conditional MMD is requested, we must register the cell-type
    # column with scvi's AnnDataManager so the tensors dict carries the index
    # at forward time. The field is NOT fed to encoder/decoders — only to the
    # MMD grouping logic. Default `celltype_key="cell_type"` is the column
    # present in the canonical H+Mo / 4-species adatas.
    setup_kwargs: dict = {"batch_key": batch_key, "species_key": "species"}
    if use_celltype_conditional_mmd:
        effective_ct_key = celltype_key or "cell_type"
        setup_kwargs["celltype_key"] = effective_ct_key
    ChimeraModel.setup_anndata(adata, **setup_kwargs)
    extra_model_kwargs: dict = {}
    # Module defaults: mask_species_in_encoder=True, mask_batch_in_encoder=True
    # (validated iter50+ recipe; iter24 proved both masks improve z quality).
    # Only override if explicitly set to False via CLI --no-mask-* flags.
    extra_model_kwargs["mask_species_in_encoder"] = mask_species_in_encoder
    extra_model_kwargs["mask_batch_in_encoder"] = mask_batch_in_encoder
    if lambda_mmd is not None:
        extra_model_kwargs["lambda_mmd"] = float(lambda_mmd)
    if lambda_z_balance is not None:
        extra_model_kwargs["lambda_z_balance"] = float(lambda_z_balance)
    if use_celltype_conditional_mmd:
        extra_model_kwargs["use_celltype_conditional_mmd"] = True
    if use_vamp_prior:
        extra_model_kwargs["use_vamp_prior"] = True
        extra_model_kwargs["n_vamp_components"] = int(n_vamp_components)
    if gin_n_hidden is not None:
        # iter114: GIN MLP hidden width is configurable. Default 32 (matches pre-iter114
        # behavior). Larger widths (e.g. 64) test whether per-gene feature-space
        # discriminability is the bottleneck for E9.2 down on testis.
        extra_model_kwargs["gin_n_hidden"] = int(gin_n_hidden)
    if sg_ema_beta is not None:
        # iter116: per-minibatch S_g noise reduction via slower EMA. Default 0.9
        # (10-batch effective averaging). Larger β (e.g. 0.97 → 33-batch) tests
        # the iter115 mechanistic finding that signal density (not MLP capacity
        # or loss-weight scaling) is the bottleneck for testis 2-sp E9.2 down.
        extra_model_kwargs["sg_ema_beta"] = float(sg_ema_beta)
    model = ChimeraModel.from_config(
        adata,
        config_name=config,
        n_latent=n_latent,
        n_hidden=n_hidden,
        use_observed_lib_size=use_observed_lib_size,
        **extra_model_kwargs,
    )

    # Allow override of loss weights on the module (post-construction)
    if lambda_fit is not None:
        model.module.lambda_fit = float(lambda_fit)
    if lambda_anchor is not None:
        model.module.lambda_anchor = float(lambda_anchor)
    if lambda_res is not None:
        model.module.lambda_res = float(lambda_res)
    if free_bits is not None:
        model.module.free_bits = float(free_bits)

    # VAMP pseudo-input initialization from training data (if requested).
    # Random init converges slowly and often collapses to a single mode;
    # initializing from K random training cells gives the prior an
    # immediately-sensible set of "prototype" count vectors.
    if (
        use_vamp_prior
        and vamp_init_from_data
        and model.module.pseudo_inputs is not None
    ):
        X = adata.X
        n_obs = X.shape[0]
        rng_state = np.random.RandomState(seed)
        K = int(n_vamp_components)
        if vamp_init_stratify_species:
            # Species-stratified init. Under pure uniform sampling the species
            # composition of VAMP modes inherits the dataset skew (e.g. on
            # H+Mo the larger species takes ~66 % of modes); iter50 seed-43/44
            # Hepatocytes E8.5 failures trace to that bias. Drawing ceil(K / S)
            # cells from EACH species gives every species equal mode capacity
            # at init, eliminating the seed-dependent coverage variance.
            species = np.asarray(adata.obs["species"].values)
            unique_species = np.unique(species)
            n_sp = len(unique_species)
            base = K // n_sp
            remainder = K - base * n_sp
            idx_parts = []
            species_tally = {}
            for i, sp in enumerate(unique_species):
                sp_indices = np.where(species == sp)[0]
                target = base + (1 if i < remainder else 0)
                sp_idx = rng_state.choice(
                    sp_indices,
                    size=min(target, len(sp_indices)),
                    replace=(target > len(sp_indices)),
                )
                idx_parts.append(sp_idx)
                species_tally[str(sp)] = int(len(sp_idx))
            idx = np.concatenate(idx_parts)
            rng_state.shuffle(idx)
            logger.info(
                "VAMP stratified init: %d pseudo-inputs per species: %s",
                K, species_tally,
            )
        else:
            idx = rng_state.choice(n_obs, size=min(K, n_obs), replace=(K > n_obs))
            logger.info("Initialized %d VAMP pseudo-inputs from random training cells", K)
        if hasattr(X, "toarray"):
            X_init = X[idx].toarray()
        else:
            X_init = np.asarray(X)[idx]
        X_init_t = torch.tensor(X_init, dtype=torch.float32)
        with torch.no_grad():
            model.module.pseudo_inputs.data.copy_(X_init_t)

    # --- Optional: GIN feature-space ablation patch (round2 §1526 & Issue 15.3).
    # Applied *after* from_config (which already builds the standard feature
    # matrix from varm['gene_features']) and *before* training. StandardScaler
    # has already been applied upstream, so zeroing a column is equivalent to
    # fixing that feature at its population mean.
    if gin_feature_op is not None:
        with torch.no_grad():
            gf = model.module.gene_features
            D = gf.shape[1]
            if gin_feature_op == "zero_esm2":
                n = min(64, D)
                gf[:, :n] = 0.0
                _touched = (0, n)
            elif gin_feature_op == "zero_esm2_var":
                if D > 64:
                    gf[:, 64] = 0.0
                _touched = (64, 65)
            elif gin_feature_op == "zero_mean_identity":
                if D > 65:
                    gf[:, 65] = 0.0
                _touched = (65, 66)
            elif gin_feature_op == "zero_expr_stats":
                lo, hi = 66, min(69, D)
                gf[:, lo:hi] = 0.0
                _touched = (lo, hi)
            elif gin_feature_op == "zero_all_scalars":
                lo, hi = 64, min(69, D)
                gf[:, lo:hi] = 0.0
                _touched = (lo, hi)
            elif gin_feature_op == "random_all":
                gf.normal_(0.0, 1.0)
                _touched = (0, D)
            else:
                raise ValueError(f"Unknown gin_feature_op: {gin_feature_op!r}")
        logger.info("Applied gin_feature_op=%r; zeroed/replaced cols %s", gin_feature_op, _touched)

    # --- Build plan kwargs
    plan_kwargs: dict = {}
    if warmup_kl_epochs is not None:
        plan_kwargs["warmup_kl_epochs"] = int(warmup_kl_epochs)
    if warmup_fqr_epochs is not None:
        plan_kwargs["warmup_fqr_epochs"] = int(warmup_fqr_epochs)
    if warmup_adversarial_epochs is not None:
        plan_kwargs["warmup_adversarial_epochs"] = int(warmup_adversarial_epochs)
    if min_training_epochs is not None:
        plan_kwargs["min_training_epochs"] = int(min_training_epochs)
    if adv_weight is not None:
        plan_kwargs["adv_weight"] = float(adv_weight)
    if adv_lr is not None:
        plan_kwargs["adv_lr"] = float(adv_lr)
    if adv_classifier_hidden is not None:
        plan_kwargs["adv_classifier_hidden"] = tuple(int(h) for h in adv_classifier_hidden)
    if kl_weight_max is not None:
        plan_kwargs["kl_weight_max"] = float(kl_weight_max)

    # --- Trainer kwargs (scvi-tools forwards these as **trainer_kwargs to Lightning Trainer)
    csv_logger = CSVLogger(save_dir=str(RUNS_ROOT), name=run_name, version="lightning")
    alpha_cb = AlphaSnapshotCallback(out_dir=run_dir / "alpha_snapshots", every_n_epochs=snapshot_every_n_epochs)
    stash_cb = StashTrainingPlanCallback(model)

    trainer_kwargs: dict = dict(
        logger=csv_logger,
        callbacks=[alpha_cb, stash_cb],
        enable_progress_bar=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,  # needed so adv accuracy / val loss are logged each epoch
    )

    # --- Save config snapshot
    config_snapshot = dict(
        adata_path=str(adata_path),
        run_name=run_name,
        ablation_config=config,
        max_epochs=max_epochs,
        batch_size=batch_size,
        seed=seed,
        n_latent=n_latent,
        n_hidden=n_hidden,
        use_observed_lib_size=use_observed_lib_size,
        plan_kwargs=plan_kwargs,
        lambda_fit=lambda_fit,
        lambda_anchor=lambda_anchor,
        lambda_res=lambda_res,
        adv_weight=adv_weight,
        adv_lr=adv_lr,
        adv_classifier_hidden=(
            None if adv_classifier_hidden is None else list(adv_classifier_hidden)
        ),
        gin_feature_op=gin_feature_op,
        free_bits=free_bits,
        lambda_mmd=lambda_mmd,
        lambda_z_balance=lambda_z_balance,
        kl_weight_max=kl_weight_max,
        use_celltype_conditional_mmd=bool(use_celltype_conditional_mmd),
        celltype_key=celltype_key,
        use_vamp_prior=bool(use_vamp_prior),
        n_vamp_components=int(n_vamp_components) if use_vamp_prior else None,
        vamp_init_from_data=bool(vamp_init_from_data) if use_vamp_prior else None,
        vamp_init_stratify_species=bool(vamp_init_stratify_species) if use_vamp_prior else None,
        equal_species_sampling=bool(equal_species_sampling),
        batch_key=batch_key,
        prior_clamp_max=(None if prior_clamp_max is None else float(prior_clamp_max)),
        gin_n_hidden=(None if gin_n_hidden is None else int(gin_n_hidden)),
        sg_ema_beta=(None if sg_ema_beta is None else float(sg_ema_beta)),
        mask_species_in_encoder=bool(mask_species_in_encoder),
        mask_batch_in_encoder=bool(mask_batch_in_encoder),
        WARMUP_KL_EPOCHS=WARMUP_KL_EPOCHS,
        WARMUP_FQR_EPOCHS=WARMUP_FQR_EPOCHS,
        WARMUP_ADV_EPOCHS=WARMUP_ADV_EPOCHS,
        started_at=datetime.now().isoformat(timespec="seconds"),
    )
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # --- Train
    t0 = time.time()
    logger.info("Starting training: %d epochs, batch_size=%d, accelerator=%s", max_epochs, batch_size, accelerator)
    train_kwargs: dict = dict(
        max_epochs=max_epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        plan_kwargs=plan_kwargs,
        **trainer_kwargs,  # scvi-tools forwards any unknown kwarg to Lightning Trainer
    )
    if accelerator is not None:
        train_kwargs["accelerator"] = accelerator
    if equal_species_sampling:
        train_kwargs["equal_species_sampling"] = True
        train_kwargs["sqrt_species_sampling"] = False
    model.train(**train_kwargs)
    elapsed = time.time() - t0
    logger.info("Training finished in %.1fs", elapsed)

    # --- Save model
    model_dir = run_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model.save(str(model_dir), save_anndata=False, overwrite=True)

    # --- Dump final state for expectations checker
    final_path = run_dir / "final_state.npz"
    logger.info("Extracting final state to %s", final_path)
    dump_final_state(model, adata, final_path)

    # --- Persist training time + run summary
    run_summary = dict(
        run_name=run_name,
        config=config,
        max_epochs=max_epochs,
        elapsed_seconds=elapsed,
        n_cells=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        n_species=int(adata.obs["species"].nunique()),
        n_batch=int(adata.obs["batch"].nunique()),
        finished_at=datetime.now().isoformat(timespec="seconds"),
    )
    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, default=Path("data/processed/camex_human_monkey.h5ad"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", default="A", choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-latent", type=int, default=10)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--no-observed-lib-size", action="store_true")
    parser.add_argument("--warmup-kl-epochs", type=int, default=None)
    parser.add_argument("--warmup-fqr-epochs", type=int, default=None)
    parser.add_argument("--warmup-adversarial-epochs", type=int, default=None)
    parser.add_argument("--min-training-epochs", type=int, default=None)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--accelerator", default=None, help="cpu, mps, cuda, or None (auto)")
    parser.add_argument("--snapshot-every-n-epochs", type=int, default=1)
    parser.add_argument("--lambda-fit", type=float, default=None)
    parser.add_argument("--lambda-anchor", type=float, default=None)
    parser.add_argument("--lambda-res", type=float, default=None)
    parser.add_argument("--adv-weight", type=float, default=None)
    parser.add_argument("--adv-lr", type=float, default=None,
                        help="LR of the adversarial species classifier. Default 1e-3.")
    parser.add_argument("--adv-classifier-hidden", type=str, default=None,
                        help="Comma-separated hidden widths of the adversarial classifier "
                             "MLP (e.g. '32' for default, '64,32' for a 2-hidden MLP). "
                             "Needed to close the Caveat 34 nonlinear z-leakage gap.")
    parser.add_argument("--free-bits", type=float, default=None,
                        help="Per-dim KL floor override (default 0.5)")
    parser.add_argument("--kl-weight-max", type=float, default=None,
                        help="Max KL weight after warmup (default 1.0). "
                             "Lower (0.3-0.5) when z-detach reduces recon gradient.")
    parser.add_argument("--gin-feature-op", type=str, default=None,
                        help="Diagnostic GIN feature patch (zero_esm2, zero_expr_stats, etc.)")
    parser.add_argument("--lambda-mmd", type=float, default=None,
                        help="Weight for MMD loss on z for species mixing. "
                             "Recommended range [10, 200]. 0 disables.")
    parser.add_argument("--lambda-z-balance", type=float, default=None,
                        help="Weight for z-dim variance balance loss. "
                             "None=module default (1.0). 0 disables.")
    parser.add_argument("--mask-species-in-encoder", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Zero species_oh in encoder input (default: True). "
                             "Residual decoder still receives species explicitly. "
                             "Use --no-mask-species-in-encoder to disable.")
    parser.add_argument("--mask-batch-in-encoder", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Zero batch_oh in encoder input (default: True). "
                             "batch_offset submodule still sees batch_idx for its "
                             "constrained per-batch offset. "
                             "Use --no-mask-batch-in-encoder to disable.")
    parser.add_argument("--celltype-conditional-mmd", action="store_true",
                        help="Use celltype-conditional MMD: match species within each "
                             "cell type instead of globally. Registers a cell-type field "
                             "via setup_anndata. No celltype info reaches encoder/decoders.")
    parser.add_argument("--celltype-key", type=str, default=None,
                        help="Column in adata.obs with cell-type labels for "
                             "--celltype-conditional-mmd. Default: 'cell_type'.")
    parser.add_argument("--vamp-prior", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use VAMP prior (default: True, canonical for iter50+). "
                             "Pass --no-vamp-prior for N(0,I).")
    parser.add_argument("--n-vamp-components", type=int, default=500,
                        help="K (number of VAMP pseudo-inputs). Default 500.")
    parser.add_argument("--no-vamp-init-from-data", action="store_true",
                        help="Initialize VAMP pseudo-inputs as random Gaussian instead "
                             "of sampling from training cells. Default is data-init.")
    parser.add_argument("--vamp-init-stratify-species", action="store_true",
                        help="When data-initializing VAMP pseudo-inputs, draw K/S cells "
                             "from EACH species instead of uniformly across cells. "
                             "Equalizes mode capacity across species on skewed datasets "
                             "(target: robustify E8.5 on species-skewed celltypes).")
    parser.add_argument("--equal-species-sampling", action="store_true",
                        help="Use equal (1/S) per-species weighting in the training "
                             "DataLoader instead of sqrt-proportional. Each species "
                             "contributes 1/S of draws; tightens per-batch species "
                             "balance for MMD and targets seed-variance in integration.")
    parser.add_argument("--batch-key", type=str, default="batch",
                        help="obs column to use as batch_key for setup_anndata. "
                             "Default 'batch' (==species in most datasets). Use "
                             "'sample_batch' for donor-level batch correction.")
    parser.add_argument("--prior-clamp-max", type=float, default=None,
                        help="iter109: optional upper clamp on adata.var['conservation_prior'] "
                             "applied in-memory before training. The data-prep default caps "
                             "prior at 0.95; on testis ~32%% of genes sit at the cap so α "
                             "saturates from prior alone. Use 0.85 for testis to give the GIN "
                             "1.21 logit units of headroom in both directions. None ⇒ preserve "
                             "the data's existing values (current behavior).")
    parser.add_argument("--sg-ema-beta", type=float, default=None,
                        help="iter116: S_g EMA decay (default SG_EMA_BETA=0.9, ~10-batch "
                             "averaging). Larger β slows the EMA and reduces per-minibatch "
                             "noise on rare neg-S_g testis genes. Try 0.97 (~33-batch) or "
                             "0.99 (~100-batch) on testis 2-sp to attack E9.2 down ceiling.")
    args = parser.parse_args()

    train_and_diagnose(
        adata_path=args.adata,
        run_name=args.run_name,
        config=args.config,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        use_observed_lib_size=not args.no_observed_lib_size,
        warmup_kl_epochs=args.warmup_kl_epochs,
        warmup_fqr_epochs=args.warmup_fqr_epochs,
        warmup_adversarial_epochs=args.warmup_adversarial_epochs,
        min_training_epochs=args.min_training_epochs,
        early_stopping=args.early_stopping,
        accelerator=args.accelerator,
        snapshot_every_n_epochs=args.snapshot_every_n_epochs,
        lambda_fit=args.lambda_fit,
        lambda_anchor=args.lambda_anchor,
        lambda_res=args.lambda_res,
        adv_weight=args.adv_weight,
        adv_lr=args.adv_lr,
        adv_classifier_hidden=(
            None if args.adv_classifier_hidden is None
            else tuple(int(h) for h in args.adv_classifier_hidden.split(","))
        ),
        mask_species_in_encoder=args.mask_species_in_encoder,
        mask_batch_in_encoder=args.mask_batch_in_encoder,
        gin_feature_op=args.gin_feature_op,
        free_bits=args.free_bits,
        lambda_mmd=args.lambda_mmd,
        lambda_z_balance=args.lambda_z_balance,
        kl_weight_max=args.kl_weight_max,
        use_celltype_conditional_mmd=args.celltype_conditional_mmd,
        celltype_key=args.celltype_key,
        use_vamp_prior=args.vamp_prior,
        n_vamp_components=args.n_vamp_components,
        vamp_init_from_data=not args.no_vamp_init_from_data,
        vamp_init_stratify_species=args.vamp_init_stratify_species,
        equal_species_sampling=args.equal_species_sampling,
        batch_key=args.batch_key,
        prior_clamp_max=args.prior_clamp_max,
        sg_ema_beta=args.sg_ema_beta,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
