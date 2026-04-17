"""Build a markdown comparison table across all runs under training/runs.

Reads `summary.json` from every run and prints a sorted side-by-side comparison.

Usage
-----
    python training/summarize_runs.py                      # all runs
    python training/summarize_runs.py --names iter10 iter12 # specific runs
    python training/summarize_runs.py --pattern 'iter*'    # glob pattern
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "training" / "runs"


def _fmt(x, spec: str = ".3f") -> str:
    if x is None:
        return "-"
    try:
        if isinstance(x, bool):
            return str(x)
        return format(float(x), spec)
    except Exception:
        return str(x)


def load_summaries(run_dirs: list[Path]) -> list[dict]:
    out = []
    for d in run_dirs:
        sp = d / "summary.json"
        if not sp.exists():
            continue
        try:
            out.append(json.loads(sp.read_text()))
        except Exception:
            continue
    return out


def print_table(summaries: list[dict]) -> str:
    if not summaries:
        return "(no summaries found)"
    lines: list[str] = []
    lines.append("| Run | Data | Epochs | Pass/Fail | δ+% | δ-% | adv_acc | resid_RMS | FQR>0% |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    # sort by pass count desc, then name
    summaries = sorted(summaries, key=lambda s: (-s["expectation_summary"]["pass"], s["run_name"]))
    for s in summaries:
        adata = Path(s.get("adata_path", "")).name.replace("camex_", "").replace(".h5ad", "")
        es = s["expectation_summary"]
        km = s["key_metrics"]
        lines.append(
            "| {rn} | {ad} | {ep} | {p}/{t}F{f}S{sk} | {dp} | {dn} | {ad_acc} | {rr} | {sg} |".format(
                rn=s["run_name"],
                ad=adata,
                ep=s.get("max_epochs", "?"),
                p=es["pass"],
                t=es["pass"] + es["fail"] + es["skip"],
                f=es["fail"],
                sk=es["skip"],
                dp=_fmt(km.get("delta_pos_frac"), ".2%"),
                dn=_fmt(km.get("delta_neg_frac"), ".2%"),
                ad_acc=_fmt(km.get("adv_acc_final")),
                rr=_fmt(km.get("residual_rms_mean")),
                sg=_fmt(km.get("sg_ema_frac_positive"), ".1%"),
            )
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--pattern", default=None)
    args = p.parse_args()

    if args.names:
        run_dirs = [RUNS_ROOT / n for n in args.names]
    elif args.pattern:
        run_dirs = sorted(RUNS_ROOT.glob(args.pattern))
    else:
        run_dirs = sorted([d for d in RUNS_ROOT.iterdir() if d.is_dir()])

    summaries = load_summaries(run_dirs)
    print(print_table(summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
