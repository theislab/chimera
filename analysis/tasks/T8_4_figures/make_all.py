"""Render every panel of this folder into ``figures/panels/`` and check each PDF.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m analysis.tasks.T8_4_figures.make_all

Writes ``results/analysis/T8_4/checks.csv`` (fonts embedded, no Type 3, no internal run name) and exits non-zero if any
panel fails. The figures are then composed with ``bash figures/build.sh``.
"""

from __future__ import annotations

import sys

import pandas as pd

from analysis.tasks.T8_4_figures import (common, fig3_integration, fig4_additions, fig5_controls,
                                         fig7_validation, supp_figures)


def main() -> int:
    panels = []
    for mod in (fig3_integration, fig4_additions, fig5_controls, fig7_validation, supp_figures):
        panels += mod.main()
    rows = [common.check_pdf(p) for p in sorted(panels)]
    df = pd.DataFrame(rows)
    out = common.RES / "T8_4" / "checks.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    return 0 if df.ok.all() else 1


if __name__ == "__main__":
    sys.exit(main())
