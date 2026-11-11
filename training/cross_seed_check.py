"""Evaluate E9.1 cross-seed α robustness across multiple Chimera runs of the
same config.

E9.1 — `alpha_cross_seed_robust`: min Spearman ρ over all seed pairs of the
same config; threshold ≥ 0.50 (revised 2026-04-25; prior
bar was ≥ 0.90 — see workflow.md E9.1 row for rationale).

Usage
-----
    mamba run -n chimera_env python training/cross_seed_check.py \
        training/runs/xds-hmouse-iter90-150ep-s42 \
        training/runs/xds-hmouse-iter90-150ep-s43 \
        training/runs/xds-hmouse-iter90-150ep-s44

Reports per-pair ρ on α and Δα, top-K Jaccard for the high-α and low-α
gene sets, and the E9.1 pass/fail under both the revised (≥ 0.50) and
prior (≥ 0.90) thresholds for backward visibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

E9_1_THRESHOLD = 0.50  # revised 2026-04-25; prior was 0.90.
E9_1_LEGACY_THRESHOLD = 0.90


def load_alpha_prior(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load α and prior from a run's final_state.npz."""
    f = run_dir / "final_state.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found — run training first.")
    d = np.load(f)
    if "alpha" not in d or "prior" not in d:
        raise KeyError(f"{f}: expected 'alpha' and 'prior' keys, got {list(d.keys())}.")
    return d["alpha"], d["prior"]


def cross_seed_e9_1(run_dirs: list[Path]) -> dict:
    """Compute E9.1 numbers + auxiliary diagnostics over ≥ 2 run dirs."""
    if len(run_dirs) < 2:
        return {"status": "skipped", "reason": "fewer than 2 runs provided"}

    alphas, priors, deltas = [], [], []
    for r in run_dirs:
        a, p = load_alpha_prior(r)
        alphas.append(a)
        priors.append(p)
        deltas.append(a - p)

    G = len(alphas[0])
    for i, a in enumerate(alphas):
        if len(a) != G:
            raise ValueError(f"{run_dirs[i]} has {len(a)} genes; expected {G}.")

    pairs = [(i, j) for i in range(len(run_dirs)) for j in range(i + 1, len(run_dirs))]
    rhos_alpha = [float(spearmanr(alphas[i], alphas[j])[0]) for i, j in pairs]
    rhos_delta = [float(spearmanr(deltas[i], deltas[j])[0]) for i, j in pairs]

    # Top-K Jaccard for high-α and low-α sets at 1%, 5%, 10%.
    jaccs = {}
    for pct in (0.01, 0.05, 0.10):
        K = max(1, int(pct * G))
        sets_high = [set(np.argsort(a)[-K:]) for a in alphas]
        sets_low = [set(np.argsort(a)[:K]) for a in alphas]
        j_high = [len(sets_high[i] & sets_high[j]) / max(len(sets_high[i] | sets_high[j]), 1)
                  for i, j in pairs]
        j_low = [len(sets_low[i] & sets_low[j]) / max(len(sets_low[i] | sets_low[j]), 1)
                 for i, j in pairs]
        jaccs[f"top{int(pct * 100)}pct"] = {
            "high_alpha_mean_jaccard": float(np.mean(j_high)),
            "low_alpha_mean_jaccard": float(np.mean(j_low)),
        }

    min_rho_alpha = float(min(rhos_alpha))
    pass_revised = min_rho_alpha >= E9_1_THRESHOLD
    pass_legacy = min_rho_alpha >= E9_1_LEGACY_THRESHOLD

    return {
        "status": "evaluated",
        "n_runs": len(run_dirs),
        "n_genes": G,
        "pair_indices": pairs,
        "alpha_rho_per_pair": rhos_alpha,
        "delta_alpha_rho_per_pair": rhos_delta,
        "min_alpha_rho": min_rho_alpha,
        "topK_jaccard": jaccs,
        "E9_1_threshold_revised": E9_1_THRESHOLD,
        "E9_1_threshold_legacy": E9_1_LEGACY_THRESHOLD,
        "E9_1_pass_revised": pass_revised,
        "E9_1_pass_legacy": pass_legacy,
    }


def format_report(result: dict, run_dirs: list[Path]) -> str:
    if result["status"] == "skipped":
        return f"E9.1: SKIPPED — {result['reason']}"
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("E9.1 alpha_cross_seed_robust")
    lines.append("=" * 70)
    lines.append(f"Runs ({result['n_runs']}, {result['n_genes']} genes):")
    for r in run_dirs:
        lines.append(f"  - {r}")
    lines.append("")
    lines.append("Pairwise α Spearman ρ:")
    for (i, j), rho in zip(result["pair_indices"], result["alpha_rho_per_pair"]):
        lines.append(f"  ρ(run{i}, run{j}) = {rho:.4f}")
    lines.append(f"  min ρ           = {result['min_alpha_rho']:.4f}")
    lines.append("")
    lines.append("Pairwise Δα Spearman ρ (GIN correction signal):")
    for (i, j), rho in zip(result["pair_indices"], result["delta_alpha_rho_per_pair"]):
        lines.append(f"  ρ(run{i}, run{j}) = {rho:.4f}")
    lines.append("")
    lines.append("Top-K α gene-set Jaccard:")
    for k, v in result["topK_jaccard"].items():
        lines.append(f"  {k}: high-α mean={v['high_alpha_mean_jaccard']:.3f}  "
                     f"low-α mean={v['low_alpha_mean_jaccard']:.3f}")
    lines.append("")
    rev = "PASS" if result["E9_1_pass_revised"] else "FAIL"
    leg = "PASS" if result["E9_1_pass_legacy"] else "FAIL"
    lines.append(f"E9.1 verdict (revised threshold ≥ {result['E9_1_threshold_revised']}): {rev}")
    lines.append(f"E9.1 verdict (legacy   threshold ≥ {result['E9_1_threshold_legacy']}): {leg}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", type=Path,
                    help="Two or more Chimera run dirs containing final_state.npz.")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON only (machine-readable).")
    ap.add_argument("--out", type=Path, default=None,
                    help="If set, write the JSON result to this path.")
    args = ap.parse_args()

    result = cross_seed_e9_1(args.run_dirs)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2))

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_report(result, args.run_dirs))

    if result["status"] == "skipped":
        return 2
    return 0 if result["E9_1_pass_revised"] else 1


if __name__ == "__main__":
    sys.exit(main())
