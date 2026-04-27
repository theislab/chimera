"""Run a full Chimera training experiment: train + evaluate + summarize.

One command runs `train_and_diagnose.py`, then `check_expectations.py`, and emits
a compact `summary.json` in the run directory with the key metrics the workflow
tracks across iterations. The summary is the canonical "did this iteration
work?" artifact — `iteration_log.md` is populated from these.

Usage
-----
    # iter50 canonical recipe (VAMP + MMD, close-primate):
    # VAMP prior is on by default; pass --no-vamp-prior to disable.
    python training/run_experiment.py --run-name my-run \
        --adata data/processed/camex_human_monkey.h5ad \
        --max-epochs 100 --adv-weight 100 --lambda-mmd 10

    # mid-divergent (H+Mouse): add --free-bits 1.0 --adv-weight 200
    # very-divergent (H+Zebrafish): keep --adv-weight 100 (not 200)

Or via config file:
    python training/run_experiment.py --plan training/plans/iter50.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.check_expectations import check_all, format_report, load_run  # noqa: E402
from training.train_and_diagnose import train_and_diagnose  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "training" / "runs"


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """Summarize a finished run into a compact JSON-friendly dict.

    Keys are the headline metrics the workflow cross-references: per-category
    expectation pass counts, key scalars (alpha stats, adv_acc, residual RMS,
    loss components), and the dataset identity. Written to `summary.json`.
    """
    run = load_run(run_dir)
    results = check_all(run)

    cat_counts: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r["category"] or "other"
        c = cat_counts.setdefault(cat, {"pass": 0, "fail": 0, "skip": 0})
        if r["pass_"] is True:
            c["pass"] += 1
        elif r["pass_"] is False:
            c["fail"] += 1
        else:
            c["skip"] += 1

    fs = run.get("final_state", {})
    cfg = run.get("config", {})

    def _maybe_float(x) -> float | None:
        if x is None:
            return None
        try:
            return float(np.asarray(x).item()) if hasattr(x, "item") else float(x)
        except Exception:
            return None

    alpha = fs.get("alpha")
    prior = fs.get("prior")
    delta = alpha - prior if alpha is not None and prior is not None else None

    summary = {
        "run_name": run_dir.name,
        "config": cfg.get("ablation_config"),
        "adata_path": cfg.get("adata_path"),
        "max_epochs": cfg.get("max_epochs"),
        "lambda_fit": cfg.get("lambda_fit"),
        "lambda_anchor": cfg.get("plan_kwargs", {}).get("lambda_anchor"),
        "adv_weight": cfg.get("adv_weight"),
        "expectation_summary": {
            "pass": sum(c["pass"] for c in cat_counts.values()),
            "fail": sum(c["fail"] for c in cat_counts.values()),
            "skip": sum(c["skip"] for c in cat_counts.values()),
            "by_category": cat_counts,
        },
        "key_metrics": {
            "alpha_mean": _maybe_float(alpha.mean()) if alpha is not None else None,
            "alpha_std": _maybe_float(alpha.std()) if alpha is not None else None,
            "delta_pos_frac": _maybe_float((delta > 0.02).mean()) if delta is not None else None,
            "delta_neg_frac": _maybe_float((delta < -0.02).mean()) if delta is not None else None,
            "delta_max": _maybe_float(delta.max()) if delta is not None else None,
            "delta_min": _maybe_float(delta.min()) if delta is not None else None,
            "adv_acc_final": _maybe_float(fs.get("adversarial_accuracy_final")),
            "residual_rms_mean": _maybe_float(np.sqrt((fs["residual_rms"] ** 2).mean())) if "residual_rms" in fs else None,
            "z_mean_var_min": _maybe_float(fs["z_mean"].var(axis=0).min()) if "z_mean" in fs else None,
            "sg_ema_frac_positive": _maybe_float((fs["S_g_ema"] > 0).mean()) if "S_g_ema" in fs else None,
            "n_species": _maybe_float(fs.get("n_species")),
            "n_cells": _maybe_float(fs.get("n_cells")),
        },
        "failed_expectations": [r["id"] for r in results if r["pass_"] is False],
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "report.md").write_text(format_report(run_dir.name, run, results))
    return summary


def run_experiment(
    run_name: str,
    adata_path: Path,
    max_epochs: int = 150,
    config: str = "A",
    adv_weight: float | None = 100.0,
    adv_lr: float | None = None,
    adv_classifier_hidden: tuple[int, ...] | None = None,
    lambda_fit: float | None = 0.1,
    lambda_anchor: float | None = 0.003,
    lambda_res: float | None = 0.001,
    warmup_adversarial_epochs: int | None = 5,
    warmup_fqr_epochs: int | None = 20,
    warmup_kl_epochs: int | None = 20,
    n_latent: int = 10,
    n_hidden: int = 128,
    seed: int = 42,
    accelerator: str | None = "mps",
    snapshot_every_n_epochs: int = 5,
    mask_species_in_encoder: bool = True,
    mask_batch_in_encoder: bool = True,
    gin_feature_op: str | None = None,
    free_bits: float | None = None,
    lambda_mmd: float | None = None,
    lambda_z_balance: float | None = None,
    use_vamp_prior: bool = True,
    n_vamp_components: int = 500,
    vamp_init_from_data: bool = True,
    use_celltype_conditional_mmd: bool = False,
    celltype_key: str | None = None,
    batch_key: str = "batch",
    prior_clamp_max: float | None = None,
    gin_n_hidden: int | None = None,
    sg_ema_beta: float | None = None,
    lambda_anchor_neg_factor: float | None = None,
    use_per_gene_bias: bool = False,
    lambda_mmd_aux: float | None = None,
    mmd_pair_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Train a Chimera model end-to-end and return a summary dict."""
    run_dir = train_and_diagnose(
        adata_path=adata_path,
        run_name=run_name,
        config=config,
        max_epochs=max_epochs,
        seed=seed,
        n_latent=n_latent,
        n_hidden=n_hidden,
        warmup_kl_epochs=warmup_kl_epochs,
        warmup_fqr_epochs=warmup_fqr_epochs,
        warmup_adversarial_epochs=warmup_adversarial_epochs,
        accelerator=accelerator,
        snapshot_every_n_epochs=snapshot_every_n_epochs,
        lambda_fit=lambda_fit,
        lambda_anchor=lambda_anchor,
        lambda_res=lambda_res,
        adv_weight=adv_weight,
        adv_lr=adv_lr,
        adv_classifier_hidden=adv_classifier_hidden,
        mask_species_in_encoder=mask_species_in_encoder,
        mask_batch_in_encoder=mask_batch_in_encoder,
        gin_feature_op=gin_feature_op,
        free_bits=free_bits,
        lambda_mmd=lambda_mmd,
        lambda_z_balance=lambda_z_balance,
        use_vamp_prior=use_vamp_prior,
        n_vamp_components=n_vamp_components,
        vamp_init_from_data=vamp_init_from_data,
        use_celltype_conditional_mmd=use_celltype_conditional_mmd,
        celltype_key=celltype_key,
        batch_key=batch_key,
        prior_clamp_max=prior_clamp_max,
        gin_n_hidden=gin_n_hidden,
        sg_ema_beta=sg_ema_beta,
        lambda_anchor_neg_factor=lambda_anchor_neg_factor,
        use_per_gene_bias=use_per_gene_bias,
        lambda_mmd_aux=lambda_mmd_aux,
        mmd_pair_weights=mmd_pair_weights,
    )
    return summarize_run(run_dir)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--adata", type=Path, default=Path("data/processed/camex_human_monkey.h5ad"))
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--config", default="A")
    p.add_argument("--adv-weight", type=float, default=100.0)
    p.add_argument("--adv-lr", type=float, default=None,
                   help="LR of the adversarial species classifier. None=default 1e-3.")
    p.add_argument("--adv-classifier-hidden", type=str, default=None,
                   help="Comma-separated hidden widths of the adversarial classifier "
                        "MLP (e.g. '64,32'). Deeper classifier closes Caveat 34 "
                        "nonlinear z-leakage gap. None=default (32,).")
    p.add_argument("--lambda-fit", type=float, default=0.1)
    p.add_argument("--lambda-anchor", type=float, default=0.003)
    p.add_argument("--lambda-res", type=float, default=0.001)
    p.add_argument("--warmup-adversarial-epochs", type=int, default=5)
    p.add_argument("--warmup-fqr-epochs", type=int, default=20)
    p.add_argument("--warmup-kl-epochs", type=int, default=20)
    p.add_argument("--n-latent", type=int, default=10)
    p.add_argument("--n-hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--accelerator", default="mps")
    p.add_argument("--snapshot-every-n-epochs", type=int, default=5)
    p.add_argument("--mask-species-in-encoder", action=argparse.BooleanOptionalAction,
                   default=True, help="Zero species_oh in encoder input (default: True).")
    p.add_argument("--mask-batch-in-encoder", action=argparse.BooleanOptionalAction,
                   default=True, help="Zero batch_oh in encoder input (default: True).")
    p.add_argument(
        "--free-bits",
        type=float,
        default=None,
        help="Per-dim KL free-bits floor. None=module default (0.5). Raise (e.g. 1.0) "
             "to prevent posterior collapse under strong adversarial pressure.",
    )
    p.add_argument(
        "--gin-feature-op",
        default=None,
        choices=[None, "zero_esm2", "zero_esm2_var", "zero_mean_identity",
                 "zero_expr_stats", "zero_all_scalars", "random_all"],
        help="Diagnostic-only: patch GIN gene_features matrix after model construction. "
             "Used for the round2 GIN-input ablation sweep.",
    )
    p.add_argument("--lambda-mmd", type=float, default=None,
                   help="MMD loss weight. None=module default (50.0). "
                        "Set 0 to disable species mixing.")
    p.add_argument("--lambda-z-balance", type=float, default=None,
                   help="z-dim variance balance loss weight. None=module default (1.0). "
                        "0 disables. Penalizes imbalanced per-dim z variance.")
    p.add_argument("--vamp-prior", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Use VAMP prior (default: True, canonical for iter50+). "
                        "Pass --no-vamp-prior to use N(0,I) instead.")
    p.add_argument("--n-vamp-components", type=int, default=500,
                   help="Number of VAMP pseudo-input components.")
    p.add_argument("--no-vamp-init-from-data", action="store_true",
                   help="Initialize VAMP pseudo-inputs from Gaussian noise "
                        "instead of random training cells.")
    p.add_argument("--celltype-conditional-mmd", action="store_true",
                   help="Use celltype-conditional MMD: match species within each "
                        "cell type instead of globally. Directly targets E8.4.")
    p.add_argument("--batch-key", type=str, default="batch",
                   help="obs column to use as batch_key (default 'batch'; use "
                        "'sample_batch' for donor-level correction)")
    p.add_argument("--celltype-key", type=str, default=None,
                   help="Column in adata.obs with cell-type labels for "
                        "--celltype-conditional-mmd. Default: 'cell_type'.")
    p.add_argument("--prior-clamp-max", type=float, default=None,
                   help="iter109: optional upper clamp on conservation_prior "
                        "applied in-memory before training. Default 0.95 (data). "
                        "Use 0.85 on testis to give the GIN ±1.21 logit units of "
                        "headroom and break the prior-saturation lock.")
    p.add_argument("--gin-hidden", type=int, default=None,
                   help="iter114: GIN MLP hidden width. Default 32 "
                        "(architecture 72 -> 64 -> 32 -> 1, ~6.8K params). "
                        "Set 64 to widen to 72 -> 128 -> 64 -> 1 (~17.7K params, 2.6x). "
                        "Targets feature-space discriminability ceiling on testis E9.2 down.")
    p.add_argument("--sg-ema-beta", type=float, default=None,
                   help="iter116: S_g EMA decay (default SG_EMA_BETA=0.9, ~10-batch "
                        "averaging). Larger β slows the EMA and reduces per-minibatch "
                        "noise on rare neg-S_g testis genes. 0.97 → ~33-batch averaging "
                        "(3.3× the iter109 default), 0.99 → ~100-batch. Targets the "
                        "iter115 mechanistic finding (signal density, not MLP capacity).")
    p.add_argument("--lambda-anchor-neg-factor", type=float, default=None,
                   help="iter118: asymmetric L_alpha_anchor factor for the negative-"
                        "correction half-axis. Default 1.0 (symmetric, byte-identical "
                        "to all prior best-knowns). Set < 1.0 to relax the negative "
                        "anchor penalty (e.g. 0.3 → 3.33× more negative equilibrium "
                        "for correctly-identified neg-S_g genes). Targets the universal "
                        "testis E9.2 down architectural ceiling. Prediction (testis H+Mac, "
                        "factor=0.3): mean correction on neg-S_g genes shifts from +0.353 "
                        "toward 0 or negative; E9.2 push<-0.05 from 0.01%% to 1.5-5%%.")
    p.add_argument("--use-per-gene-bias", action="store_true",
                   help="iter119: opt-in per-gene bias hybrid in GIN. When set, "
                        "correction = MLP(features) + per_gene_bias[gene], where "
                        "per_gene_bias = nn.Parameter(torch.zeros(n_genes)) (zero-init). "
                        "Default OFF (byte-identical to all current best-knowns). "
                        "Targets the universal testis E9.2 push<-0.05 architectural "
                        "ceiling: iter118 proved MLP feature-space smoothness is the "
                        "dominant binding constraint. Per-gene scalar bias gives each "
                        "gene its own dof, bypassing MLP smoothness without discarding "
                        "feature-space generalization (the MLP still learns the smooth "
                        "bulk). Predicted: ρ(S_g, corr) → 0.85-0.97; mean corr on neg-S_g "
                        "→ -0.5 to -1.5; E9.2 push<-0.05 → 5-15%%.")
    p.add_argument("--lambda-mmd-aux", type=float, default=None,
                   help="iter121: opt-in dual-MMD weight. When >0 AND "
                        "--celltype-conditional-mmd is set, adds a global MMD term "
                        "alongside CT-cond MMD: total = lambda_mmd*L_ct + lambda_mmd_aux*L_global. "
                        "Restores species-mixing pressure on cells in UNSHARED CTs "
                        "(iter107 4-sp failure: alpha frozen, adv leaked because "
                        "6 of 8 CTs got zero MMD). Default unset/0.0 = byte-identical legacy.")
    p.add_argument("--mmd-pair-weights", type=str, default=None,
                   help="iter122: opt-in species-pair-conditional MMD weights "
                        "(comma-separated, length C(n_species, 2), in lex pair order). "
                        "For 4 species: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3). When set, "
                        "_compute_mmd uses weighted-pair averaging Sum_k w_k MMD_k / Sum_k w_k "
                        "instead of uniform. Down-weight 'impossible' pairs (e.g. Zebrafish "
                        "at 450 Myr) to concentrate pressure on close-pair mixing. Default "
                        "unset = uniform = byte-identical legacy.")
    p.add_argument("--plan", type=Path, default=None,
                   help="JSON file overriding all above args (single experiment).")
    args = p.parse_args()

    if args.plan is not None:
        plan = json.loads(args.plan.read_text())
        summary = run_experiment(**plan)
    else:
        summary = run_experiment(
            run_name=args.run_name,
            adata_path=args.adata,
            max_epochs=args.max_epochs,
            config=args.config,
            adv_weight=args.adv_weight,
            adv_lr=args.adv_lr,
            adv_classifier_hidden=(
                None if args.adv_classifier_hidden is None
                else tuple(int(h) for h in args.adv_classifier_hidden.split(","))
            ),
            lambda_fit=args.lambda_fit,
            lambda_anchor=args.lambda_anchor,
            lambda_res=args.lambda_res,
            warmup_adversarial_epochs=args.warmup_adversarial_epochs,
            warmup_fqr_epochs=args.warmup_fqr_epochs,
            warmup_kl_epochs=args.warmup_kl_epochs,
            n_latent=args.n_latent,
            n_hidden=args.n_hidden,
            seed=args.seed,
            accelerator=args.accelerator,
            snapshot_every_n_epochs=args.snapshot_every_n_epochs,
            mask_species_in_encoder=args.mask_species_in_encoder,
            mask_batch_in_encoder=args.mask_batch_in_encoder,
            gin_feature_op=args.gin_feature_op,
            free_bits=args.free_bits,
            lambda_mmd=args.lambda_mmd,
            lambda_z_balance=getattr(args, 'lambda_z_balance', None),
            use_vamp_prior=args.vamp_prior,
            n_vamp_components=args.n_vamp_components,
            vamp_init_from_data=not args.no_vamp_init_from_data,
            use_celltype_conditional_mmd=args.celltype_conditional_mmd,
            celltype_key=args.celltype_key,
            batch_key=args.batch_key,
            prior_clamp_max=args.prior_clamp_max,
            gin_n_hidden=args.gin_hidden,
            sg_ema_beta=args.sg_ema_beta,
            lambda_anchor_neg_factor=args.lambda_anchor_neg_factor,
            use_per_gene_bias=args.use_per_gene_bias,
            lambda_mmd_aux=args.lambda_mmd_aux,
            mmd_pair_weights=(
                None if args.mmd_pair_weights is None
                else [float(w) for w in args.mmd_pair_weights.split(",") if w.strip()]
            ),
        )

    es = summary["expectation_summary"]
    print(f"\n=== {summary['run_name']} ===")
    print(f"Expectations: pass={es['pass']}, fail={es['fail']}, skip={es['skip']}")
    print(f"Failed: {summary['failed_expectations']}")
    print(f"Key metrics: alpha_mean={summary['key_metrics']['alpha_mean']:.3f}, "
          f"delta+={summary['key_metrics']['delta_pos_frac']:.3f}, "
          f"delta-={summary['key_metrics']['delta_neg_frac']:.3f}, "
          f"adv={summary['key_metrics']['adv_acc_final']:.3f}")
    return 0 if summary["expectation_summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
