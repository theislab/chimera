"""Same-seed repeats of the liver Human+Mouse fit, beside the prior-swap control.

For seeds 46 and 47 three fits share the recipe and the seed: the seed run of the paper
(``rev-liv_hmouse-s<seed>``), its repeat with unchanged arguments (``rev-rep-liv_hmouse-s<seed>``,
launched from ``manifest.tsv`` of this folder), and the fit with the smoothed protein prior of
``analysis/tasks/T6_8_priorswap`` (``rev-ctl-liv_hmouse-priorswap-s<seed>``). Per seed the table gives

- the Spearman correlation across genes between each pair of the three runs, for the correction c,
  for Delta-alpha (alpha minus the run's own prior) and for the count evidence S_g
  (columns ``rho_c``, ``rho_D``, ``rho_S``);
- the Spearman correlation between the change of c and the change of the prior, swap run minus
  original run (in column ``rho_c``; the two other columns are empty).

The repeat-against-original row is the yardstick: a prior swap that moves the readouts no more than
a repeat of the same fit has no effect that can be told from run-to-run variation.

Output: ``results/analysis/T6_8b_repeat.csv`` (seed, pair, rho_c, rho_D, rho_S).

    python -m analysis.tasks.T6_8b_repeat.compare            # write the table
    python -m analysis.tasks.T6_8b_repeat.compare --check    # recompute, compare with the stored table, write nothing

The archived runs give the stored table again, digit for digit; ``--check`` allows 1e-12 because
the last digits of a correlation can depend on the machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.common import controls as K
from analysis.common import gene_ids

OUT = gene_ids.REPO_ROOT / "results/analysis/T6_8b_repeat.csv"
SEEDS = [46, 47]
RUNS = {"original": "rev-liv_hmouse-s{seed}", "repeat": "rev-rep-liv_hmouse-s{seed}", "swap": "rev-ctl-liv_hmouse-priorswap-s{seed}"}
PAIRS = {"repeat vs original (no swap)": ("repeat", "original"), "swap vs original": ("swap", "original"), "swap vs repeat": ("swap", "repeat")}
READOUTS = {"rho_c": "correction", "rho_D": "D", "rho_S": "S_g_ema"}
TOLERANCE = 1e-12


def spear(a, b) -> float:
    return float(spearmanr(a, b)[0])


def table() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        run = {name: K.read_run(pattern.format(seed=seed)) for name, pattern in RUNS.items()}
        for pair, (a, b) in PAIRS.items():
            rows.append(dict(seed=seed, pair=pair, **{col: spear(run[a][key], run[b][key]) for col, key in READOUTS.items()}))
        d_c = run["swap"]["correction"] - run["original"]["correction"]
        d_pi = run["swap"]["prior"] - run["original"]["prior"]
        rows.append(dict(seed=seed, pair="rho(delta c, delta pi), swap - original", rho_c=spear(d_c, d_pi), rho_D=np.nan, rho_S=np.nan))
    return pd.DataFrame(rows)


def check(new: pd.DataFrame, path: Path) -> int:
    """Compare the recomputed table with the stored one; 0 when every cell agrees within TOLERANCE."""
    old = pd.read_csv(path, float_precision="round_trip")
    if list(old.columns) != list(new.columns) or not old[["seed", "pair"]].equals(new[["seed", "pair"]]):
        print(f"{path}: rows or columns differ from the recomputed table")
        return 1
    cols = list(READOUTS)
    a, b = old[cols].to_numpy(float), new[cols].to_numpy(float)
    same_blank = np.array_equal(np.isnan(a), np.isnan(b))
    diff = float(np.nanmax(np.abs(a - b)))
    print(f"{path}: {int(np.isfinite(a).sum())} values, largest difference {diff:.3g}, tolerance {TOLERANCE:g}")
    return 0 if same_blank and diff <= TOLERANCE else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="file to write (default: results/analysis/T6_8b_repeat.csv)")
    ap.add_argument("--check", action="store_true", help="recompute and compare with the stored table (--out); write nothing")
    a = ap.parse_args()
    new = table()
    if a.check:
        return check(new, a.out)
    K.write_csv(new, a.out)
    print(new.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
