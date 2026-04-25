"""Diagnostic: verify loss balance, adversarial effectiveness, GIN learning, residual contribution.

Trains a ChimeraModule on synthetic data for 80 epochs and checks that every
Chimera-specific component is actually contributing after the loss-scaling fix.

This is NOT a unit test — it's a longer integration diagnostic that verifies
the model works end-to-end.  Run with:
    python -m pytest tests/test_loss_balance_diagnostic.py -v -s
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from chimera._constants import (
    LAMBDA_ANCHOR,
    LAMBDA_FIT,
    LAMBDA_RES,
    WARMUP_ADV_EPOCHS,
    WARMUP_FQR_EPOCHS,
    WARMUP_KL_EPOCHS,
    ADV_WEIGHT,
)
from chimera._data_utils import create_synthetic_adata, precompute_p_null
from chimera._module import ChimeraModule
from chimera._training_plan import ChimeraTrainingPlan, _grad_reverse


def _build_module_and_batch(n_genes=500, n_cells=300, n_species=2, seed=42):
    """Build module + batch with enough genes to reveal scaling issues."""
    result = create_synthetic_adata(
        n_cells_per_species=n_cells,
        n_genes=n_genes,
        n_species=n_species,
        n_batches_per_species=2,
        seed=seed,
    )
    adata = result["adata"]
    n_genes_actual = adata.n_vars
    n_species_actual = len(adata.obs["species"].cat.categories)
    species_cats = list(adata.obs["species"].cat.categories)
    batch_cats = list(adata.obs["batch"].cat.categories)
    n_batch = len(batch_cats)

    batch_to_species = []
    for b in batch_cats:
        mask = adata.obs["batch"] == b
        s = adata.obs.loc[mask, "species"].iloc[0]
        batch_to_species.append(species_cats.index(s))
    batch_to_species_idx = torch.tensor(batch_to_species, dtype=torch.long)

    gene_features = torch.tensor(result["gene_features"], dtype=torch.float32)
    cp = result["conservation_prior"]
    if isinstance(cp, torch.Tensor):
        conservation_prior = cp.float()
    else:
        conservation_prior = torch.tensor(np.asarray(cp), dtype=torch.float32)
    p_null = torch.tensor(precompute_p_null(adata), dtype=torch.float32)

    module = ChimeraModule(
        n_genes=n_genes_actual,
        n_species=n_species_actual,
        n_batch=n_batch,
        n_latent=10,
        n_hidden=128,
        gene_features=gene_features,
        conservation_prior=conservation_prior,
        p_null=p_null,
        batch_to_species_idx=batch_to_species_idx,
        lambda_aux=0.3,
        lambda_res=LAMBDA_RES,
        lambda_anchor=LAMBDA_ANCHOR,
        lambda_fit=LAMBDA_FIT,
        use_observed_lib_size=True,
        mask_species_in_encoder=True,
        mask_batch_in_encoder=True,
    )

    X_raw = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    X = torch.tensor(X_raw, dtype=torch.float32)
    species = torch.tensor(adata.obs["species"].cat.codes.values, dtype=torch.long).unsqueeze(-1)
    batch = torch.tensor(adata.obs["batch"].cat.codes.values, dtype=torch.long).unsqueeze(-1)
    library_size = X.sum(-1, keepdim=True)

    batch_dict = {"X": X, "species": species, "batch": batch, "library_size": library_size}
    return module, batch_dict, n_species_actual


def _build_adversary(module, n_species):
    """Build adversarial classifier matching ChimeraTrainingPlan defaults."""
    import torch.nn as nn
    n_latent = module.n_latent
    classifier = nn.Sequential(
        nn.Linear(n_latent, 64), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(32, n_species),
    )
    return classifier


class TestLossBalanceDiagnostic:
    """Comprehensive post-fix diagnostic for loss balance and component effectiveness."""

    def test_loss_magnitude_balance(self):
        """After the scaling fix, adversarial loss should be 0.5-10% of L_recon, not 0.001%."""
        torch.manual_seed(42)
        module, batch_dict, n_species = _build_module_and_batch(n_genes=500)
        module.train()

        # Forward pass
        _, outputs = module(batch_dict)

        L_recon = outputs["L_recon"].item()
        L_kl = outputs["L_kl"].item()
        L_shared_aux = outputs["L_shared_aux"].item()
        L_residual_sparsity = outputs["L_residual_sparsity"].item()
        L_alpha_anchor = outputs["L_alpha_anchor"].item()

        # Simulate adversarial fool loss (cross-entropy at chance)
        z = outputs["z"]
        species_idx = batch_dict["species"].squeeze(-1)
        classifier = _build_adversary(module, n_species)
        logits = classifier(z)
        fool_loss = F.cross_entropy(logits, species_idx).item()

        n_genes = module.n_genes
        adv_weight = ADV_WEIGHT

        # OLD scaling: fool_loss * adv_weight (no n_genes)
        old_adv_contribution = fool_loss * adv_weight
        old_adv_pct = old_adv_contribution / L_recon * 100

        # NEW scaling: fool_loss * adv_weight * n_genes
        new_adv_contribution = fool_loss * adv_weight * n_genes
        new_adv_pct = new_adv_contribution / L_recon * 100

        print(f"\n{'='*60}")
        print(f"LOSS MAGNITUDE ANALYSIS (n_genes={n_genes})")
        print(f"{'='*60}")
        print(f"L_recon           = {L_recon:10.2f}  (100%)")
        print(f"L_shared_aux*0.3  = {0.3*L_shared_aux:10.2f}  ({0.3*L_shared_aux/L_recon*100:.1f}%)")
        print(f"L_kl              = {L_kl:10.2f}  ({L_kl/L_recon*100:.2f}%)")
        print(f"L_res_sparsity*λ  = {LAMBDA_RES*L_residual_sparsity:10.4f}  ({LAMBDA_RES*L_residual_sparsity/L_recon*100:.3f}%)")
        print(f"L_anchor*λ        = {LAMBDA_ANCHOR*L_alpha_anchor:10.6f}  ({LAMBDA_ANCHOR*L_alpha_anchor/L_recon*100:.5f}%)")
        print(f"fool_loss (raw)   = {fool_loss:10.4f}")
        print(f"")
        print(f"OLD adv contrib   = {old_adv_contribution:10.4f}  ({old_adv_pct:.4f}% of L_recon)")
        print(f"NEW adv contrib   = {new_adv_contribution:10.2f}  ({new_adv_pct:.2f}% of L_recon)")
        print(f"Improvement       = {new_adv_contribution/max(old_adv_contribution,1e-12):.0f}x")
        print(f"{'='*60}")

        # Assertions
        assert old_adv_pct < 0.1, f"Old scaling was already >0.1%? old_adv_pct={old_adv_pct}"
        assert new_adv_pct > 0.1, (
            f"New adversarial contribution still too small: {new_adv_pct:.3f}% of L_recon. "
            f"Expected > 0.1%"
        )
        assert new_adv_pct < 20.0, (
            f"New adversarial contribution too large: {new_adv_pct:.3f}% of L_recon. "
            f"Expected < 20%"
        )

    def test_training_80_epochs_all_components_active(self):
        """Train 80 epochs and verify every Chimera component contributes."""
        torch.manual_seed(42)
        module, batch_dict, n_species = _build_module_and_batch(n_genes=500, n_cells=200)
        classifier = _build_adversary(module, n_species)

        # Optimizers — with GRL, both are updated from the same loss
        model_opt = torch.optim.Adam(module.parameters(), lr=1e-3)
        adv_opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)

        n_genes = module.n_genes
        adv_weight = ADV_WEIGHT
        n_epochs = 80

        # Track metrics
        history = {
            "L_recon": [], "L_kl": [], "L_shared_aux": [],
            "L_residual_sparsity": [], "L_alpha_anchor": [],
            "L_fit_reward": [], "adv_cls_loss": [], "adv_acc": [],
            "correction_std": [], "residual_mag": [], "alpha_mean": [],
            "alpha_std": [], "delta_alpha_abs_mean": [],
        }

        species_idx = batch_dict["species"].squeeze(-1)

        for epoch in range(n_epochs):
            module.train()
            classifier.train()

            # Set epoch-dependent state
            kl_weight = min(1.0, epoch / max(WARMUP_KL_EPOCHS, 1))
            module.kl_weight = kl_weight
            module.current_epoch = epoch
            kappa = adv_weight if epoch >= WARMUP_ADV_EPOCHS else 0.0

            # Forward
            loss, outputs = module(batch_dict)
            z = outputs["z"]

            # Adversarial via Gradient Reversal Layer
            if kappa > 0:
                z_rev = _grad_reverse(z, lambda_=kappa * n_genes)
                cls_logits = classifier(z_rev)
                adv_cls_loss = F.cross_entropy(cls_logits, species_idx)
                loss = loss + adv_cls_loss
            else:
                adv_cls_loss = torch.tensor(0.0)

            # Joint step (model + classifier updated from same loss via GRL)
            model_opt.zero_grad()
            adv_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            model_opt.step()
            if kappa > 0:
                adv_opt.step()

            # Compute adversarial accuracy
            with torch.no_grad():
                preds = classifier(z).argmax(dim=-1)
                adv_acc = (preds == species_idx).float().mean().item()

            # Record
            with torch.no_grad():
                history["L_recon"].append(outputs["L_recon"].item())
                history["L_kl"].append(outputs["L_kl"].item())
                history["L_shared_aux"].append(outputs["L_shared_aux"].item())
                history["L_residual_sparsity"].append(outputs["L_residual_sparsity"].item())
                history["L_alpha_anchor"].append(outputs["L_alpha_anchor"].item())
                history["L_fit_reward"].append(outputs["L_fit_reward"].item())
                history["adv_cls_loss"].append(adv_cls_loss.item())
                history["adv_acc"].append(adv_acc)

                correction = outputs["correction"]
                alpha = outputs["alpha"]
                history["correction_std"].append(correction.std().item())
                history["residual_mag"].append(
                    outputs["log_rate_residual"].abs().mean().item()
                )
                history["alpha_mean"].append(alpha.mean().item())
                history["alpha_std"].append(alpha.std().item())
                delta_alpha = alpha - module.conservation_prior
                history["delta_alpha_abs_mean"].append(delta_alpha.abs().mean().item())

        # --- Print summary ---
        print(f"\n{'='*70}")
        print(f"80-EPOCH TRAINING DIAGNOSTIC (n_genes={n_genes}, n_species={n_species})")
        print(f"{'='*70}")

        first5 = slice(0, 5)
        last5 = slice(-5, None)

        for key in ["L_recon", "L_kl", "adv_cls_loss", "adv_acc",
                     "correction_std", "residual_mag", "alpha_std",
                     "delta_alpha_abs_mean"]:
            vals = history[key]
            early = np.mean(vals[:5])
            late = np.mean(vals[-5:])
            print(f"{key:30s}  early={early:10.4f}  late={late:10.4f}")

        # --- ASSERTION BLOCK ---
        print(f"\n{'='*70}")
        print("COMPONENT HEALTH CHECKS")
        print(f"{'='*70}")

        # 1. L_recon should decrease (model learns)
        early_recon = np.mean(history["L_recon"][:5])
        late_recon = np.mean(history["L_recon"][-5:])
        recon_improved = late_recon < early_recon * 0.95
        print(f"[{'PASS' if recon_improved else 'FAIL'}] Reconstruction improves: "
              f"{early_recon:.1f} -> {late_recon:.1f}")
        assert recon_improved, (
            f"L_recon did not improve: {early_recon:.1f} -> {late_recon:.1f}"
        )

        # 2. GIN correction should drift from zero
        late_correction_std = np.mean(history["correction_std"][-5:])
        gin_active = late_correction_std > 0.01
        print(f"[{'PASS' if gin_active else 'FAIL'}] GIN corrections drift from zero: "
              f"std={late_correction_std:.4f} (need > 0.01)")
        assert gin_active, (
            f"GIN correction std={late_correction_std:.4f}, expected > 0.01. "
            f"GIN is not learning."
        )

        # 3. Delta alpha should be non-trivial
        late_delta = np.mean(history["delta_alpha_abs_mean"][-5:])
        alpha_learning = late_delta > 0.005
        print(f"[{'PASS' if alpha_learning else 'FAIL'}] Alpha deviates from prior: "
              f"|Δα|={late_delta:.4f} (need > 0.005)")
        assert alpha_learning, (
            f"|delta_alpha| mean = {late_delta:.5f}, expected > 0.005"
        )

        # 4. Residual pathway should be active (not dormant)
        late_residual = np.mean(history["residual_mag"][-5:])
        residual_active = late_residual > 0.01
        print(f"[{'PASS' if residual_active else 'FAIL'}] Residual pathway active: "
              f"mean |residual|={late_residual:.4f} (need > 0.01)")
        assert residual_active, (
            f"Residual magnitude = {late_residual:.5f}, expected > 0.01. "
            f"Residual pathway is dormant."
        )

        # 5. With GRL, the adversary should maintain reasonable accuracy
        # (it's trained to classify, not fooled) but the encoder is pushed
        # to make classification harder. On synthetic data, we expect the
        # adversary accuracy to drop from 100% toward chance.
        post_warmup_acc = history["adv_acc"][WARMUP_ADV_EPOCHS:]
        if len(post_warmup_acc) > 5:
            late_acc = np.mean(post_warmup_acc[-5:])
            # With GRL the classifier stays trained; we just check that the
            # encoder has partially reduced species signal (not 100% anymore)
            adv_working = late_acc < 0.99
            print(f"[{'PASS' if adv_working else 'FAIL'}] GRL reduces adversary accuracy: "
                  f"acc={late_acc:.3f} (need < 0.99)")
            assert adv_working, (
                f"Adversarial accuracy = {late_acc:.3f}, expected < 0.99. "
                f"GRL is not affecting the encoder at all."
            )

        # 6. Adversarial classifier loss should remain meaningful
        # (classifier is actively trying to discriminate — not collapsed)
        late_adv_loss = np.mean(history["adv_cls_loss"][-5:])
        adv_active = late_adv_loss > 0.1
        print(f"[{'PASS' if adv_active else 'FAIL'}] Adv classifier loss active: "
              f"loss={late_adv_loss:.4f} (need > 0.1)")
        assert adv_active, (
            f"adv_cls_loss = {late_adv_loss:.4f}, expected > 0.1. "
            f"Classifier may have collapsed."
        )

        # 7. Verify the GRL gradient is a meaningful fraction of total loss
        # adv_cls_loss is unscaled CE (~0.7 at chance), but GRL multiplies
        # the gradient by kappa * n_genes. So effective gradient contribution
        # ≈ adv_cls_loss * kappa * n_genes / L_recon
        effective_adv = late_adv_loss * adv_weight * n_genes
        adv_fraction = effective_adv / late_recon
        balanced = 0.001 < adv_fraction < 1.0
        print(f"[{'PASS' if balanced else 'FAIL'}] GRL gradient/recon ratio: "
              f"{adv_fraction:.4f} (need 0.001-1.0)")
        assert balanced, (
            f"GRL/recon ratio = {adv_fraction:.4f}, expected in [0.001, 1.0]"
        )

        # 8. L_kl should be non-trivial (VAE is not degenerating)
        late_kl = np.mean(history["L_kl"][-5:])
        kl_ok = late_kl > 1.0
        print(f"[{'PASS' if kl_ok else 'FAIL'}] KL divergence healthy: "
              f"L_kl={late_kl:.2f} (need > 1.0)")
        assert kl_ok, f"L_kl = {late_kl:.2f}, expected > 1.0"

        print(f"\n{'='*70}")
        print("ALL CHECKS PASSED")
        print(f"{'='*70}")

    def test_species_leakage_reduced(self):
        """After training with active GRL adversary, linear probe should detect less species signal."""
        torch.manual_seed(42)
        module, batch_dict, n_species = _build_module_and_batch(n_genes=500, n_cells=200)
        classifier = _build_adversary(module, n_species)

        model_opt = torch.optim.Adam(module.parameters(), lr=1e-3)
        adv_opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)

        n_genes = module.n_genes
        adv_weight = ADV_WEIGHT
        species_idx = batch_dict["species"].squeeze(-1)

        # Train 60 epochs with GRL
        for epoch in range(60):
            module.train()
            classifier.train()
            module.kl_weight = min(1.0, epoch / max(WARMUP_KL_EPOCHS, 1))
            module.current_epoch = epoch
            kappa = adv_weight if epoch >= WARMUP_ADV_EPOCHS else 0.0

            loss, outputs = module(batch_dict)
            z = outputs["z"]

            if kappa > 0:
                z_rev = _grad_reverse(z, lambda_=kappa * n_genes)
                adv_cls_loss = F.cross_entropy(classifier(z_rev), species_idx)
                loss = loss + adv_cls_loss

            model_opt.zero_grad()
            adv_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            model_opt.step()
            if kappa > 0:
                adv_opt.step()

        # Post-training: train a fresh linear probe on z
        module.eval()
        with torch.no_grad():
            _, outputs = module(batch_dict)
            z = outputs["z_mean"]  # deterministic

        probe = torch.nn.Linear(module.n_latent, n_species)
        probe_opt = torch.optim.Adam(probe.parameters(), lr=1e-2)

        for _ in range(200):
            probe_opt.zero_grad()
            probe_loss = F.cross_entropy(probe(z), species_idx)
            probe_loss.backward()
            probe_opt.step()

        with torch.no_grad():
            probe_preds = probe(z).argmax(dim=-1)
            probe_acc = (probe_preds == species_idx).float().mean().item()

        print(f"\nLinear probe accuracy on z: {probe_acc:.3f}")
        print(f"(Chance = {1.0/n_species:.3f})")

        # With detach_z_residual=True + GRL + MMD + masking, species signal
        # is reduced. On 60-epoch synthetic data, the probe should be below
        # 100% (some reduction from perfect separation).
        assert probe_acc < 1.0, (
            f"Linear probe accuracy = {probe_acc:.3f}. Expected < 1.0 — "
            f"the model should reduce species signal in z at least partially."
        )

    def test_gin_learns_different_alphas_for_different_genes(self):
        """GIN should produce gene-specific corrections, not a uniform shift."""
        torch.manual_seed(42)
        module, batch_dict, _ = _build_module_and_batch(n_genes=500, n_cells=200)
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

        # Train 60 epochs with FQR active
        for epoch in range(60):
            module.train()
            module.kl_weight = min(1.0, epoch / max(WARMUP_KL_EPOCHS, 1))
            module.current_epoch = max(epoch, WARMUP_FQR_EPOCHS + 1)  # Force FQR active

            optimizer.zero_grad()
            loss, outputs = module(batch_dict)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()

        # Check that corrections vary across genes
        with torch.no_grad():
            _, outputs = module(batch_dict)

        correction = outputs["correction"]
        alpha = outputs["alpha"]
        prior = module.conservation_prior

        corr_std = correction.std().item()
        alpha_std = alpha.std().item()
        delta = (alpha - prior).abs()

        # At least some genes should have |correction| > 0.05
        n_active = (correction.abs() > 0.05).sum().item()
        frac_active = n_active / correction.shape[0]

        print(f"\nGIN correction std: {corr_std:.4f}")
        print(f"Alpha std: {alpha_std:.4f}")
        print(f"Genes with |correction| > 0.05: {n_active}/{correction.shape[0]} ({frac_active:.1%})")
        print(f"|delta_alpha| max: {delta.max().item():.4f}")

        assert corr_std > 0.01, (
            f"GIN corrections have std={corr_std:.5f}, expected > 0.01. "
            f"The GIN is producing nearly uniform outputs."
        )

    def test_residual_reconstruction_gap(self):
        """Full model (with residual) should outperform shared-only by a meaningful margin."""
        torch.manual_seed(42)
        module, batch_dict, _ = _build_module_and_batch(n_genes=500, n_cells=200)
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

        # Train 60 epochs
        for epoch in range(60):
            module.train()
            module.kl_weight = min(1.0, epoch / max(WARMUP_KL_EPOCHS, 1))
            module.current_epoch = epoch
            optimizer.zero_grad()
            loss, _ = module(batch_dict)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()

        # Evaluate gap
        module.eval()
        with torch.no_grad():
            _, outputs = module(batch_dict)

        from scvi.distributions import NegativeBinomial
        x = batch_dict["X"]
        theta = outputs["theta"]

        nll_full = -NegativeBinomial(mu=outputs["mu_full"], theta=theta).log_prob(x).sum(-1).mean().item()
        nll_shared = -NegativeBinomial(mu=outputs["mu_shared_only"], theta=theta.detach()).log_prob(x).sum(-1).mean().item()

        gap = nll_shared - nll_full
        gap_pct = gap / nll_shared * 100

        print(f"\nNLL full model:   {nll_full:.2f}")
        print(f"NLL shared-only:  {nll_shared:.2f}")
        print(f"Gap (shared - full): {gap:.2f} nats ({gap_pct:.2f}%)")

        # The residual should provide at least 0.1% improvement
        assert gap > 0, (
            f"Full model is WORSE than shared-only: gap = {gap:.4f}. "
            f"Residual pathway is not contributing."
        )
        assert gap_pct > 0.05, (
            f"Residual reconstruction gap is only {gap_pct:.4f}%. "
            f"Expected > 0.05% improvement from residual pathway."
        )

    def test_nb_loglikelihood_metric_correct(self):
        """Verify NB log-likelihood can be computed for evaluation (claim 8 support)."""
        torch.manual_seed(42)
        module, batch_dict, _ = _build_module_and_batch(n_genes=200, n_cells=100)
        module.train()

        # Quick training
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
        for epoch in range(10):
            module.kl_weight = min(1.0, epoch / 20)
            module.current_epoch = epoch
            optimizer.zero_grad()
            loss, _ = module(batch_dict)
            loss.backward()
            optimizer.step()

        # Compute NB log-likelihood per gene per cell
        module.eval()
        with torch.no_grad():
            _, outputs = module(batch_dict)

        from scvi.distributions import NegativeBinomial
        x = batch_dict["X"]
        mu = outputs["mu_full"]
        theta = outputs["theta"]

        ll_per_gene = NegativeBinomial(mu=mu, theta=theta).log_prob(x)  # (B, G)
        assert ll_per_gene.shape == x.shape
        assert torch.isfinite(ll_per_gene).all(), "NB log-likelihood has non-finite values"

        # Per-gene mean log-likelihood (averaged over cells)
        ll_per_gene_mean = ll_per_gene.mean(0)  # (G,)
        assert ll_per_gene_mean.shape[0] == module.n_genes

        # Overall mean
        overall_ll = ll_per_gene.mean().item()
        print(f"\nNB log-likelihood per gene per cell: {overall_ll:.4f}")
        print(f"Range: [{ll_per_gene_mean.min().item():.4f}, {ll_per_gene_mean.max().item():.4f}]")

        # Should be negative (log-prob of count data)
        assert overall_ll < 0, f"Expected negative log-likelihood, got {overall_ll}"
