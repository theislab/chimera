"""Chimera v2.4 — Training plan with differential LR, multi-phase schedule, and adversarial training.

Extends scvi-tools AdversarialTrainingPlan with:
- Differential learning rates (GIN at 10x lower LR)
- Batch offset weight decay (null-space damping)
- KL warmup (0 → 1 over 20 epochs)
- FQR warmup (off for first 20 epochs)
- Adversarial delayed start (off for first 15 epochs)
- Linear LR warmup (5 epochs)
- ReduceOnPlateau scheduler
- Gradient clipping (max_norm=1.0)
- Composite convergence criteria
- wandb-compatible metric logging
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from chimera._constants import (
    ADV_WEIGHT,
    GRAD_CLIP_MAX_NORM,
    MIN_TRAINING_EPOCHS,
    WARMUP_ADV_EPOCHS,
    WARMUP_FQR_EPOCHS,
    WARMUP_KL_EPOCHS,
)

try:
    from scvi.train import AdversarialTrainingPlan
except ImportError:
    # Fallback for environments where scvi-tools is not installed
    # (allows module to be imported for type-checking)
    from scvi.train import TrainingPlan as AdversarialTrainingPlan


class CompositeConvergenceChecker:
    """Stop training only when ALL criteria are met simultaneously.

    Five criteria must hold:
    (a) ELBO plateau — validation loss stable for `patience` epochs
    (b) Alpha histogram stability — L1 distance < 0.01
    (c) S_g > 0 fraction stabilized
    (d) Adversarial accuracy near chance (1/K ± 0.05)
    (e) Minimum `min_epochs` reached

    Parameters
    ----------
    n_species
        Number of species in the dataset.
    min_epochs
        Minimum epochs before stopping is allowed.
    patience
        Number of epochs to look back for plateau detection.
    """

    def __init__(
        self,
        n_species: int,
        min_epochs: int = MIN_TRAINING_EPOCHS,
        patience: int = 10,
    ) -> None:
        self.min_epochs = min_epochs
        self.patience = patience
        self.n_species = n_species
        self.alpha_hist_buffer: list[torch.Tensor] = []
        self.sg_frac_buffer: list[float] = []
        self.val_loss_buffer: list[float] = []

    def _check_plateau(self, val_loss: float) -> bool:
        """Check if validation loss has plateaued."""
        self.val_loss_buffer.append(float(val_loss))
        if len(self.val_loss_buffer) < self.patience:
            return False
        recent = self.val_loss_buffer[-self.patience :]
        loss_range = max(recent) - min(recent)
        mean_loss = sum(recent) / len(recent)
        return loss_range / (abs(mean_loss) + 1e-8) < 0.01

    def should_stop(
        self,
        epoch: int,
        val_loss: float,
        alpha: torch.Tensor,
        S_g: torch.Tensor,
        adv_accuracy: float,
    ) -> bool:
        """Check all convergence criteria.

        Parameters
        ----------
        epoch
            Current training epoch.
        val_loss
            Validation ELBO.
        alpha
            Current alpha values, shape (G,).
        S_g
            Current S_g_ema values, shape (G,).
        adv_accuracy
            Adversarial classifier accuracy on validation data.

        Returns
        -------
        True if training should stop.
        """
        # (e) Minimum epochs MUST be reached
        if epoch < self.min_epochs:
            return False

        # (a) ELBO plateau
        elbo_plateau = self._check_plateau(val_loss)

        # (b) Alpha histogram stability
        hist = torch.histc(alpha.detach().float(), bins=50, min=0.0, max=1.0)
        hist = hist / hist.sum()
        self.alpha_hist_buffer.append(hist)
        alpha_stable = False
        if len(self.alpha_hist_buffer) > self.patience:
            dists = [
                (self.alpha_hist_buffer[-i] - self.alpha_hist_buffer[-i - 1]).abs().sum()
                for i in range(1, self.patience + 1)
            ]
            alpha_stable = all(d < 0.01 for d in dists)

        # (c) S_g > 0 fraction stabilized
        sg_frac = (S_g > 0).float().mean().item()
        self.sg_frac_buffer.append(sg_frac)
        sg_stable = (
            len(self.sg_frac_buffer) > self.patience
            and max(self.sg_frac_buffer[-self.patience :])
            - min(self.sg_frac_buffer[-self.patience :])
            < 0.02
        )

        # (d) Adversarial accuracy near chance
        chance = 1.0 / self.n_species
        adv_near_chance = abs(adv_accuracy - chance) < 0.05

        return elbo_plateau and alpha_stable and sg_stable and adv_near_chance


class ChimeraTrainingPlan(AdversarialTrainingPlan):
    """Custom training plan for Chimera v2.4.

    Extends AdversarialTrainingPlan with differential learning rates,
    multi-phase training schedule, and composite convergence criteria.

    Parameters
    ----------
    module
        ChimeraModule instance.
    **kwargs
        Passed to AdversarialTrainingPlan after popping Chimera-specific args.
    """

    def __init__(self, module: nn.Module, **kwargs: Any) -> None:
        # Pop custom kwargs BEFORE passing to super().__init__()
        self.warmup_adversarial_epochs: int = kwargs.pop(
            "warmup_adversarial_epochs", WARMUP_ADV_EPOCHS
        )
        self.warmup_kl_epochs: int = kwargs.pop("warmup_kl_epochs", WARMUP_KL_EPOCHS)
        self.warmup_fqr_epochs: int = kwargs.pop("warmup_fqr_epochs", WARMUP_FQR_EPOCHS)
        self.warmup_lr_epochs: int = kwargs.pop("warmup_lr_epochs", 5)
        self.min_training_epochs: int = kwargs.pop("min_training_epochs", MIN_TRAINING_EPOCHS)
        self.grad_clip_max_norm: float = kwargs.pop("grad_clip_max_norm", GRAD_CLIP_MAX_NORM)
        # Target adversarial weight after warmup; the Lightning attribute
        # ``scale_adversarial_loss`` is swapped between 0 and this value
        # per-epoch in ``training_step``.
        self._adv_weight_target: float = float(kwargs.pop("adv_weight", ADV_WEIGHT))
        # Learning rate for the adversarial species classifier. Default is 1e-3
        # (matches encoder) which is the scvi-tools default. Lower values (1e-4)
        # slow the classifier so the encoder can fool it for more of training —
        # useful on datasets where the classifier trivially out-trains the
        # encoder (2-species close-primate). See workflow.md §"E6.2 analysis".
        self._adv_lr: float = float(kwargs.pop("adv_lr", 1e-3))
        # Hidden widths of the adversarial species classifier. Default ``(32,)``
        # preserves the 1-hidden-layer MLP used by iter1–21. Passing e.g.
        # ``(64, 32)`` makes the classifier a 2-hidden-layer MLP — useful when
        # the post-hoc nonlinear z-leakage probe (Caveat 34) reveals that the
        # encoder is hiding species in nonlinearly-separable directions the
        # linear/shallow classifier cannot catch.
        self._adv_classifier_hidden: tuple[int, ...] = tuple(
            kwargs.pop("adv_classifier_hidden", (32,))
        )

        # KL-FQR warmup constraint
        assert self.warmup_fqr_epochs >= self.warmup_kl_epochs, (
            f"warmup_fqr_epochs ({self.warmup_fqr_epochs}) must be >= "
            f"warmup_kl_epochs ({self.warmup_kl_epochs})"
        )

        # Adversarial classifier: arbitrary-depth MLP, latent_dim → H1 → … → K
        n_latent = getattr(module, "n_latent", 10)
        n_species = getattr(module, "n_species", 2)

        cls_layers: list[nn.Module] = []
        d = n_latent
        for h in self._adv_classifier_hidden:
            cls_layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        cls_layers.append(nn.Linear(d, n_species))

        super().__init__(
            module,
            adversarial_classifier=nn.Sequential(*cls_layers),
            scale_adversarial_loss=self._adv_weight_target,
            **kwargs,
        )

        # Override n_output_classifier: Chimera adversarially removes species
        # (not batch, which is scvi-tools' default).
        self.n_output_classifier = n_species

        # Pass schedule constants to module for forward() gating
        self.module.warmup_fqr_epochs = self.warmup_fqr_epochs

        # Composite convergence checker
        self.convergence_checker = CompositeConvergenceChecker(
            n_species=n_species,
            min_epochs=self.min_training_epochs,
        )

    def configure_optimizers(self) -> tuple[dict, dict]:
        """Differential LR + batch offset weight decay.

        Returns TWO optimizer configs: model optimizer + adversarial classifier optimizer.
        AdversarialTrainingPlan expects [model_opt, adv_opt].
        """
        param_groups = [
            # Encoder (including z_mean_linear and z_logvar_linear)
            {
                "params": list(self.module.encoder.parameters())
                + list(self.module.z_mean_linear.parameters())
                + list(self.module.z_logvar_linear.parameters()),
                "lr": 1e-3,
            },
            # Decoder hidden
            {"params": list(self.module.decoder_hidden.parameters()), "lr": 1e-3},
            # Shared output head
            {"params": list(self.module.output_shared.parameters()), "lr": 1e-3},
            # Residual output head
            {"params": list(self.module.output_residual.parameters()), "lr": 1e-3},
            # Batch offset with weight decay for null-space damping
            {
                "params": [self.module.batch_offset.batch_offset_raw],
                "lr": 1e-3,
                "weight_decay": 1e-4,
            },
            # Dispersion
            {"params": [self.module.dispersion], "lr": 1e-3},
            # GIN at 10x lower LR
            {"params": list(self.module.gin.parameters()), "lr": 1e-4},
        ]

        # Library encoder (conditional)
        if hasattr(self.module, "library_encoder"):
            param_groups.append(
                {
                    "params": list(self.module.library_encoder.parameters())
                    + list(self.module.ql_mean.parameters())
                    + list(self.module.ql_var.parameters()),
                    "lr": 1e-3,
                }
            )

        # Config M: per-gene learnable correction
        if hasattr(self.module, "per_gene_correction"):
            param_groups.append({"params": [self.module.per_gene_correction], "lr": 1e-3})

        optimizer = torch.optim.Adam(param_groups)

        # Store initial LRs for warmup
        for pg in optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        # Adversarial classifier optimizer. The LR is a hyperparameter set by
        # ``adv_lr`` (default 1e-3 matches scvi-tools). Slower classifiers
        # (1e-4) give the encoder more time to fool; see workflow.md §E6.2.
        adv_optimizer = torch.optim.Adam(
            self.adversarial_classifier.parameters(), lr=self._adv_lr
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "validation_loss"},
            },
            {"optimizer": adv_optimizer},
        )

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Gradient clipping — called even under manual optimization in PL >= 2.0."""
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            max_norm=self.grad_clip_max_norm,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Custom training step: bypasses scvi-tools' BaseModuleClass protocol.

        ChimeraModule extends nn.Module (not BaseModuleClass) and returns
        ``(loss, outputs)`` from ``forward()`` instead of the scvi-tools
        ``(inference_outputs, generative_outputs, LossOutput)`` triple.
        We override completely rather than calling ``super().training_step()``.
        """
        from scvi import REGISTRY_KEYS

        # Set epoch-dependent weights on module
        self.module.kl_weight = min(1.0, self.current_epoch / max(self.warmup_kl_epochs, 1))
        self.module.current_epoch = self.current_epoch

        # Adversarial delayed start
        if self.current_epoch < self.warmup_adversarial_epochs:
            self.scale_adversarial_loss = 0.0
        else:
            self.scale_adversarial_loss = self._adv_weight_target

        kappa = self.scale_adversarial_loss

        # Get optimizers
        opts = self.optimizers()
        if not isinstance(opts, list):
            opt1 = opts
            opt2 = None
        else:
            opt1, opt2 = opts

        # Forward pass through our custom module
        loss, outputs = self.module(batch)
        z = outputs["z"]

        # Adversarial fool loss (if active) — classifies SPECIES, not batch
        fool_loss_raw = None
        if kappa > 0 and self.adversarial_classifier is not False:
            species_tensor = batch.get(REGISTRY_KEYS.LABELS_KEY, batch.get("labels", batch.get("species")))
            fool_loss = self.loss_adversarial_classifier(z, species_tensor, False)
            fool_loss_raw = fool_loss.detach()
            loss = loss + fool_loss * kappa

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        # Model optimizer step
        opt1.zero_grad()
        self.manual_backward(loss)
        opt1.step()

        # Adversarial classifier step — classifies SPECIES
        adv_cls_loss_raw = None
        if opt2 is not None and kappa > 0:
            species_tensor = batch.get(REGISTRY_KEYS.LABELS_KEY, batch.get("labels", batch.get("species")))
            adv_loss = self.loss_adversarial_classifier(z.detach(), species_tensor, True)
            adv_cls_loss_raw = adv_loss.detach()
            adv_loss = adv_loss * kappa
            opt2.zero_grad()
            self.manual_backward(adv_loss)
            opt2.step()

        # Log adversarial-internal signals on the first batch of each epoch.
        if batch_idx == 0 and kappa > 0:
            if fool_loss_raw is not None:
                self.log("adv/fool_loss", fool_loss_raw, prog_bar=False)
            if adv_cls_loss_raw is not None:
                self.log("adv/classifier_loss", adv_cls_loss_raw, prog_bar=False)
            self.log("schedule/adv_weight_effective", float(kappa), prog_bar=False)

        # Log metrics once per epoch (first batch)
        if batch_idx == 0:
            self.module.eval()
            with torch.no_grad():
                _, out = self.module(batch)
            self.module.train()
            self._log_training_metrics(out, self.current_epoch)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Custom validation step: bypasses scvi-tools' BaseModuleClass protocol."""
        self.module.eval()
        with torch.no_grad():
            loss, outputs = self.module(batch)
        self.log("validation_loss", loss, on_epoch=True)

    def on_train_epoch_start(self) -> None:
        """Linear LR warmup over first warmup_lr_epochs."""
        if self.current_epoch < self.warmup_lr_epochs:
            warmup_factor = (self.current_epoch + 1) / self.warmup_lr_epochs
            optimizers = self.optimizers()
            model_opt = optimizers[0] if isinstance(optimizers, list) else optimizers
            for param_group in model_opt.param_groups:
                param_group["lr"] = param_group["initial_lr"] * warmup_factor

    def on_validation_epoch_end(self) -> None:
        """Step LR scheduler and check composite convergence."""
        if hasattr(super(), "on_validation_epoch_end"):
            super().on_validation_epoch_end()

        val_loss = self.trainer.callback_metrics.get("validation_loss", float("inf"))
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()

        # Step ReduceLROnPlateau manually (automatic_optimization=False)
        if self.current_epoch >= self.warmup_lr_epochs:
            sch = self.lr_schedulers()
            if sch is not None:
                scheduler = sch[0] if isinstance(sch, list) else sch
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)

        # Compute adversarial accuracy on validation data
        adv_accuracy = 0.0
        if hasattr(self, "adversarial_classifier") and self.adversarial_classifier is not None:
            correct, total = 0, 0
            with torch.no_grad():
                for val_batch in self.trainer.val_dataloaders:
                    val_batch = self.transfer_batch_to_device(val_batch, self.device, 0)
                    _, out = self.module(val_batch)
                    z = out["z"]
                    sp_key = "labels" if "labels" in val_batch else "species"
                    species = val_batch[sp_key].squeeze(-1)
                    preds = self.adversarial_classifier(z).argmax(dim=-1)
                    correct += (preds == species).sum().item()
                    total += species.shape[0]
            if total > 0:
                adv_accuracy = correct / total
        self.log("adversarial_accuracy", adv_accuracy, prog_bar=False)

        # Get current alpha
        if hasattr(self.module, "per_gene_correction"):
            alpha = torch.sigmoid(
                torch.logit(self.module.conservation_prior) + self.module.per_gene_correction
            )
        elif hasattr(self.module, "gin"):
            alpha, _ = self.module.gin(self.module.gene_features, self.module.conservation_prior)
        else:
            alpha = self.module.alpha  # minimal variant

        # Check composite convergence
        if self.convergence_checker.should_stop(
            epoch=self.current_epoch,
            val_loss=val_loss,
            alpha=alpha.detach(),
            S_g=self.module.S_g_ema.detach() if hasattr(self.module, "S_g_ema") else torch.zeros(1),
            adv_accuracy=adv_accuracy,
        ):
            self.trainer.should_stop = True

    def _log_training_metrics(self, loss_dict: dict[str, torch.Tensor], epoch: int) -> None:
        """Log all training metrics for wandb."""
        # Loss components
        for key in [
            "L_recon",
            "L_kl",
            "L_shared_aux",
            "L_residual_sparsity",
            "L_alpha_anchor",
            "L_fit_reward",
            "L_KL_l",
            "L_unified",
        ]:
            if key in loss_dict and loss_dict[key] is not None:
                val = loss_dict[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                self.log(f"train/{key}", val, prog_bar=False)

        if "L_total" in loss_dict:
            val = loss_dict["L_total"]
            if isinstance(val, torch.Tensor):
                val = val.item()
            self.log("train/L_total", val, prog_bar=True)

        # Training schedule state
        kl_weight = min(1.0, epoch / max(self.warmup_kl_epochs, 1))
        self.log("schedule/kl_weight", kl_weight)
        self.log("schedule/adv_active", float(epoch >= self.warmup_adversarial_epochs))
        self.log("schedule/fqr_active", float(epoch >= self.warmup_fqr_epochs))

        # Alpha diagnostics
        if "alpha" in loss_dict:
            alpha = loss_dict["alpha"].detach().cpu().float()
            self.log("alpha/mean", alpha.mean())
            self.log("alpha/std", alpha.std())
            self.log("alpha/frac_below_0.1", (alpha < 0.1).float().mean())
            self.log("alpha/frac_above_0.9", (alpha > 0.9).float().mean())
            if hasattr(self.module, "conservation_prior"):
                delta = alpha - self.module.conservation_prior.cpu()
                self.log("alpha/frac_delta_positive", (delta > 0).float().mean())
                self.log("alpha/mean_delta", delta.mean())

        # S_g diagnostics
        if "S_g" in loss_dict:
            S_g = loss_dict["S_g"].detach().cpu().float()
            self.log("fqr/sg_frac_positive", (S_g > 0).float().mean())
            self.log("fqr/sg_mean", S_g.mean())
            self.log("fqr/sg_max", S_g.max())

        # Learning rate (first param group)
        optimizers = self.optimizers()
        model_opt = optimizers[0] if isinstance(optimizers, list) else optimizers
        if hasattr(model_opt, "param_groups"):
            self.log("lr/encoder", model_opt.param_groups[0]["lr"])
            if len(model_opt.param_groups) > 6:
                self.log("lr/gin", model_opt.param_groups[6]["lr"])
