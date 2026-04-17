"""Run a full Chimera training experiment: train + evaluate + summarize.

One command runs `train_and_diagnose.py`, then `check_expectations.py`, and emits
a compact `summary.json` in the run directory with the key metrics the workflow
tracks across iterations. The summary is the canonical "did this iteration
work?" artifact — `iteration_log.md` is populated from these.

Usage
-----
    python training/run_experiment.py --run-name iter14-example \
        --adata data/processed/camex_human_monkey.h5ad \
        --max-epochs 50 --adv-weight 100 --lambda-fit 0.1 \
        --lambda-anchor 0.001 --lambda-res 0.01

Or via config file:
    python training/run_experiment.py --plan training/plans/iter10_reproduction.json
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
    max_epochs: int = 50,
    config: str = "D",
    adv_weight: float | None = 100.0,
    adv_lr: float | None = None,
    adv_classifier_hidden: tuple[int, ...] | None = None,
    lambda_fit: float | None = 0.1,
    lambda_anchor: float | None = 0.001,
    lambda_res: float | None = 0.01,
    warmup_adversarial_epochs: int | None = 5,
    warmup_fqr_epochs: int | None = 20,
    warmup_kl_epochs: int | None = 20,
    n_latent: int = 10,
    n_hidden: int = 128,
    seed: int = 42,
    accelerator: str | None = "mps",
    snapshot_every_n_epochs: int = 5,
    mask_species_in_encoder: bool = False,
    mask_batch_in_encoder: bool = False,
    gin_feature_op: str | None = None,
    free_bits: float | None = None,
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
    )
    return summarize_run(run_dir)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--adata", type=Path, default=Path("data/processed/camex_human_monkey.h5ad"))
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--config", default="D")
    p.add_argument("--adv-weight", type=float, default=100.0)
    p.add_argument("--adv-lr", type=float, default=None,
                   help="LR of the adversarial species classifier. None=default 1e-3.")
    p.add_argument("--adv-classifier-hidden", type=str, default=None,
                   help="Comma-separated hidden widths of the adversarial classifier "
                        "MLP (e.g. '64,32'). Deeper classifier closes Caveat 34 "
                        "nonlinear z-leakage gap. None=default (32,).")
    p.add_argument("--lambda-fit", type=float, default=0.1)
    p.add_argument("--lambda-anchor", type=float, default=0.001)
    p.add_argument("--lambda-res", type=float, default=0.01)
    p.add_argument("--warmup-adversarial-epochs", type=int, default=5)
    p.add_argument("--warmup-fqr-epochs", type=int, default=20)
    p.add_argument("--warmup-kl-epochs", type=int, default=20)
    p.add_argument("--n-latent", type=int, default=10)
    p.add_argument("--n-hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--accelerator", default="mps")
    p.add_argument("--snapshot-every-n-epochs", type=int, default=5)
    p.add_argument("--mask-species-in-encoder", action="store_true")
    p.add_argument("--mask-batch-in-encoder", action="store_true",
                   help="Also zero batch_oh in encoder input (Caveat 34 "
                        "nonlinear z-leakage when batch→species is 1:1).")
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
