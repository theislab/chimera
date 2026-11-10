"""Re-render two supplementary panels that their own task scripts draw at their size on the page.

The ten-method integration panel is written by T5_5 into ``notebooks/figures/baselines_master_ext/``, where its figure
spec reads it; the closure panel is written by T2_5 into its results folder and taken from there as a panel.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m analysis.tasks.F1_figures.regen_supp
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from analysis.tasks.T8_4_figures import common as cm


def purity_view() -> None:
    sys.path.insert(0, str(cm.REPO / "analysis" / "tasks" / "T5_5_benchmark"))
    import views

    cm.fd.apply_design()
    cm.fd.set_figure_dir(cm.REPO / "notebooks" / "figures" / "baselines_master_ext")
    T = pd.read_csv(cm.REPO / "notebooks" / "cache" / "baselines_master_ext" / "master_table_ext.csv")
    views.figure(T, cm.fd)


def closure() -> None:
    from analysis.tasks.T2_5_celltype_margin import run as R25

    tables = {c: pd.read_parquet(cm.RES / "T2_5" / f"{c}__genes.parquet") for c in R25.C.CANONICAL
              if (cm.RES / "T2_5" / f"{c}__genes.parquet").exists()}
    print({k: (len(v), np.corrcoef(v.ppass_S_samp, v.S_g_ema)[0, 1].round(3)) for k, v in tables.items()})
    fig = R25.fig_closure(tables)
    fig.savefig(cm.RES / "T2_5" / "fig_supp_closure.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(cm.RES / "T2_5" / "fig_supp_closure.png", dpi=200, bbox_inches="tight", pad_inches=0.01)
    cm.copy_pdf(cm.RES / "T2_5" / "fig_supp_closure.pdf", "supp_celltype_closure")


if __name__ == "__main__":
    cm.setup()
    purity_view()
    closure()
