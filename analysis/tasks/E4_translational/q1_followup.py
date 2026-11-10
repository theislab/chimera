"""E4 Q1 follow-up (post hoc, written after reading q1_concordance.csv): does mouse lethality or phenotype breadth explain the inverse
association of Delta-alpha with curated mouse disease models? Adds MGI/IMPC lethality and the number of MP systems (Y2 pheno) to the
pooled model. Prints only; numbers are copied into the ledger.

    PYTHONPATH=$PWD mamba run -n chimera_env python -m analysis.tasks.E4_translational.q1_followup
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.tasks.E4_translational import analysis as A
from analysis.tasks.E4_translational import rules as RU


def main() -> int:
    T = pd.read_parquet(A.TAB / "cohorts.parquet")
    ph = pd.read_parquet(RU.REUSED["pheno"])
    for o, org in [("Y_mgi", False), ("Y_impc", False)]:
        F = A.q1_frame(T, o, org)
        F = F[F.cohort != "muscle_ageing"]
        F["leth"] = F.hsapiens_gene_id.map(ph.mouse_lethal).fillna(False).astype(float)
        F["nsys"] = np.log1p(F.hsapiens_gene_id.map(ph.n_mp_sys).fillna(0))
        for r in ["D_free", "D_can", "B1"]:
            for extra in [[], ["leth"], ["leth", "nsys"]]:
                b, se = A.logit_fit(F.dropna(subset=["INT_" + r]), "INT_" + r, A.COVS_Q1 + extra, True)
                print(o, r, extra, round(b, 3), round(se, 3))
        print(o, "lethal share by D_free tertile", F.groupby(pd.qcut(F.INT_D_free, 3, labels=False)).leth.mean().round(3).to_dict(),
              "outcome by lethal", F.groupby("leth").Y.mean().round(3).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
