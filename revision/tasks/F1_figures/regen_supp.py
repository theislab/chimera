"""Re-render the supplementary panels whose own task scripts now draw them at their size on the page.

Fig 8 (T7_10), bone marrow (T12_3), functional layer (T6_9) and the panels of T8_4 keep their own commands;
these three write outputs that are only copied by T8_4, so this runs them and refreshes the copies.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m revision.tasks.F1_figures.regen_supp
"""

from __future__ import annotations

import shutil
import sys

import numpy as np
import pandas as pd

from revision.tasks.T8_4_figures import common as cm


def purity_view() -> None:
    sys.path.insert(0, str(cm.REPO / "revision" / "tasks" / "T5_5_benchmark"))
    import views

    cm.fd.apply_design()
    cm.fd.set_figure_dir(cm.REPO / "notebooks" / "figures" / "baselines_master_ext")
    T = pd.read_csv(cm.REPO / "notebooks" / "cache" / "baselines_master_ext" / "master_table_ext.csv")
    views.figure(T, cm.fd)
    shutil.copyfile(cm.REPO / "notebooks/figures/baselines_master_ext/composite_mixing_bio_10models.pdf",
                    cm.FIG_OUT / "supp_integration_purity_view.pdf")


def closure() -> None:
    from revision.tasks.T2_5_celltype_margin import run as R25

    tables = {c: pd.read_parquet(cm.RES / "T2_5" / f"{c}__genes.parquet") for c in R25.C.CANONICAL
              if (cm.RES / "T2_5" / f"{c}__genes.parquet").exists()}
    print({k: (len(v), np.corrcoef(v.ppass_S_samp, v.S_g_ema)[0, 1].round(3)) for k, v in tables.items()})
    fig = R25.fig_closure(tables)
    fig.savefig(cm.RES / "T2_5" / "fig_supp_closure.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(cm.RES / "T2_5" / "fig_supp_closure.png", dpi=200, bbox_inches="tight", pad_inches=0.01)
    shutil.copyfile(cm.RES / "T2_5" / "fig_supp_closure.pdf", cm.FIG_OUT / "supp_celltype_closure.pdf")


if __name__ == "__main__":
    cm.setup()
    purity_view()
    closure()
