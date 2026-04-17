"""Compare GIN-input ablation runs side-by-side.

Reads `runs/<name>/summary.json` + `runs/<name>/final_state.npz` for each run
and prints a compact markdown table on the key metrics affected by GIN-input
quality (alpha dynamics, correction magnitude, expectation pass rate).

Also writes a markdown summary to `runs/gin_input_ablation_report.md` for
inclusion in the iteration log.

Usage
-----
    python training/compare_gin_ablation.py \
        --runs gin-abl-baseline gin-abl-no-esm2 gin-abl-no-expr gin-abl-no-scalars
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "training" / "runs"


def load_run(name: str) -> dict:
    d = RUNS_ROOT / name
    summary = json.loads((d / "summary.json").read_text())
    fs = np.load(d / "final_state.npz")
    corr = np.abs(fs["correction"])
    alpha = fs["alpha"]
    prior = fs["prior"]
    delta = alpha - prior
    return {
        "name": name,
        "pass": summary["expectation_summary"]["pass"],
        "fail": summary["expectation_summary"]["fail"],
        "failed": summary["failed_expectations"],
        "alpha_mean": float(alpha.mean()),
        "alpha_std": float(alpha.std()),
        "delta_pos_frac": float((delta > 0.02).mean()),
        "delta_neg_frac": float((delta < -0.02).mean()),
        "delta_absmax": float(np.abs(delta).max()),
        "corr_mean_abs": float(corr.mean()),
        "corr_frac_lt_0_1": float((corr < 0.1).mean()),
        "corr_median": float(np.median(corr)),
        "residual_rms": float(np.sqrt((fs["residual_rms"] ** 2).mean())),
        "adv_acc": float(fs.get("adversarial_accuracy_final", np.array(np.nan))),
        "sg_frac_positive": float((fs["S_g_ema"] > 0).mean()),
    }


def format_table(rows: list[dict]) -> str:
    cols = [
        ("name", "Run", "s"),
        ("pass", "Pass/32", "d"),
        ("alpha_mean", "α̅", ".3f"),
        ("alpha_std", "α std", ".3f"),
        ("delta_pos_frac", "Δα>0.02", ".1%"),
        ("delta_neg_frac", "Δα<-0.02", ".1%"),
        ("corr_mean_abs", "mean|corr|", ".3f"),
        ("corr_median", "median|corr|", ".3f"),
        ("corr_frac_lt_0_1", "%|c|<0.1", ".1%"),
        ("residual_rms", "RMS resid", ".3f"),
        ("adv_acc", "adv_acc", ".3f"),
        ("sg_frac_positive", "S_g>0", ".1%"),
    ]
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    divider = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, divider]
    for r in rows:
        cells = []
        for key, _, fmt in cols:
            val = r[key]
            if fmt == "s":
                cells.append(str(val))
            elif fmt == "d":
                cells.append(f"{val}" if val is not None else "—")
            elif fmt == ".1%":
                cells.append(f"{val*100:.1f}%")
            else:
                cells.append(f"{val:{fmt}}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=RUNS_ROOT / "gin_input_ablation_report.md",
    )
    args = p.parse_args()

    rows = [load_run(n) for n in args.runs]
    table = format_table(rows)

    md = [
        "# GIN input-ablation sweep",
        "",
        "Diagnostic sweep requested by round2 critical_reexamination "
        "§1526 (ESM-2 ablation) and Issue 15.3 (per-gene resolution).",
        "",
        "All runs on Human+Monkey subset (12,608 cells, 8,430 genes), 100 epochs, "
        "seed 42, iter16 config (adv_lr=1e-4, λ_res=0.01, λ_fit=0.1, λ_anchor=0.001).",
        "",
        table,
        "",
        "## Failed expectations per run",
        "",
    ]
    for r in rows:
        failed = ", ".join(r["failed"]) if r["failed"] else "(none)"
        md.append(f"- **{r['name']}**: {failed}")

    out = "\n".join(md) + "\n"
    print(out)
    args.out.write_text(out)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
