"""Post-training nonlinear z-leakage diagnostic for Chimera v2.4 (Task 9.12).

Trains a deep MLP species classifier on the latent z to detect species
information that the shallow adversarial classifier (z_dim -> 32 -> K)
cannot catch.  This is the safety net for the v2.4 bidirectional
delta_alpha scientific claim.

Round2: critical_reexamination_modules.md, Component 11 Caveat 34.

Relationship to Task 8.6 z-leakage diagnostics:
  - nonlinear_z_leakage_check (Task 8.6, MUST-RUN): sklearn MLP, 5-fold CV,
    quick (~30s). Use for routine checks.
  - adversarial_accuracy_check (Task 8.6, SHOULD-RUN): uses the model's own
    adversarial classifier.
  - posttraining_nonlinear_z_leakage (this task): PyTorch custom MLP with
    train/test split and early stopping, slower (~2min), provides detailed
    mitigation cascade. Use when Task 8.6 fails or for publication results.

Usage
-----
    python -m scripts.nonlinear_leakage \\
        --dataset data/processed/camex_pancreas.h5ad \\
        --model-dir results/chimera_v24_config_D \\
        --output-dir results/leakage \\
        --n-epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def posttraining_nonlinear_z_leakage(
    model,
    adata,
    species_key: str = "species",
    n_epochs: int = 50,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """Post-training deep classifier on z to detect nonlinear species encoding.

    The mandatory adversarial classifier is shallow (z_dim -> 32 -> K, Fix 20).
    This uses a deeper network to catch encoding the adversarial cannot see.

    Parameters
    ----------
    model
        Trained ``ChimeraModel``.
    adata
        AnnData registered with the model.
    species_key
        Column in ``adata.obs`` with species labels.
    n_epochs
        Maximum training epochs for the probe classifier.
    lr
        Learning rate for the probe classifier.

    Returns
    -------
    dict
        Test accuracy, chance level, whether leakage was detected, and
        mitigation advice if leakage is present.
    """
    z = model.get_latent_representation(adata)  # (N, z_dim) numpy
    species_labels = adata.obs[species_key].values
    species_cats = np.unique(species_labels)
    K = len(species_cats)
    sp_to_idx = {sp: i for i, sp in enumerate(species_cats)}
    y = np.array([sp_to_idx[s] for s in species_labels])
    chance = 1.0 / K

    # 80/20 train/test split with explicit seed for reproducibility
    N = len(z)
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(N)
    split = int(0.8 * N)
    z_tr, z_te = z[perm[:split]], z[perm[split:]]
    y_tr, y_te = y[perm[:split]], y[perm[split:]]

    # 3-layer MLP — deeper than adversarial (Caveat 34)
    device = next(model.module.parameters()).device
    z_dim = z.shape[1]
    clf = nn.Sequential(
        nn.Linear(z_dim, 64), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(32, K),
    ).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(
        torch.tensor(z_tr, dtype=torch.float32).to(device),
        torch.tensor(y_tr, dtype=torch.long).to(device),
    )
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    z_te_t = torch.tensor(z_te, dtype=torch.float32).to(device)

    best_acc, patience_count = 0.0, 0
    for epoch in range(n_epochs):
        clf.train()
        for zb, yb in dl:
            opt.zero_grad()
            loss_fn(clf(zb), yb).backward()
            opt.step()

        clf.eval()
        with torch.no_grad():
            preds = clf(z_te_t).argmax(dim=1).cpu().numpy()
            acc = float((preds == y_te).mean())
        if acc > best_acc:
            best_acc = acc
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 10:
                logger.info("Early stopping at epoch %d (best_acc=%.3f)", epoch, best_acc)
                break

    leaked = best_acc > chance + 0.05
    return {
        "test_accuracy": best_acc,
        "chance_level": chance,
        "threshold": chance + 0.05,
        "z_leakage_detected": leaked,
        "consequence": (
            "delta_alpha > 0 contaminated — z-leakage lets shared decoder "
            "fit per-species patterns via h(z), inflating S_g spuriously"
            if leaked
            else "z appears species-invariant"
        ),
        "mitigation": (
            "increase adv_weight 3x and retrain; if persistent, try "
            "MMD-based invariance loss; flag all delta_alpha > 0 as unreliable"
            if leaked
            else "none needed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-training nonlinear z-leakage diagnostic (Task 9.12)",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to prepared .h5ad")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to saved ChimeraModel")
    parser.add_argument("--output-dir", type=str, default="results/leakage")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    import anndata as ad
    from chimera import ChimeraModel

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset: %s", args.dataset)
    adata = ad.read_h5ad(args.dataset)

    logger.info("Loading model from: %s", args.model_dir)
    model = ChimeraModel.load(args.model_dir, adata=adata)

    result = posttraining_nonlinear_z_leakage(
        model, adata, n_epochs=args.n_epochs, lr=args.lr,
    )

    out_path = out / "z_leakage_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Results saved to %s", out_path)
    logger.info(
        "Accuracy=%.3f (chance=%.3f, threshold=%.3f) -> leakage=%s",
        result["test_accuracy"], result["chance_level"],
        result["threshold"], result["z_leakage_detected"],
    )


if __name__ == "__main__":
    main()
