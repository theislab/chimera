"""Compose X8's supplementary figures with F1's tool, writing only under results/revision/X8/composed/ (not the manuscript, not F1's report).

    PYTHONPATH=$PWD python -m revision.tasks.X8_gate_divergence.compose_x8 [names]
"""
from __future__ import annotations

import sys

from revision.tasks.F1_figures import compose as CO
from revision.tasks.X8_gate_divergence import rules as RU
from revision.tasks.X8_gate_divergence import specs_x8 as SP

DST = RU.OUT / "composed"


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    CO.FIG_DIR = DST
    CO.OUT = DST
    names = sys.argv[1:] or list(SP.FIGS)
    fails = 0
    for n in names:
        fails += not CO.compose(n, SP.FIGS[n])["ok"]
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
