"""Can the gold measure sparse genes? Split-half reliability of TS vs TMS and Murat, masked against unmasked genes (T4_3_scoreboard splits.npz, Spearman-Brown on the median over splits), gold-side detection
in the TS and TMS liver atlases against training detection. Reads no Chimera readout value beyond the mask.
    python -m analysis.tasks.D1_sparse_genes.gold_check
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.tasks.D1_sparse_genes import common as CM
from analysis.tasks.D1_sparse_genes import rules as RU

G = RU.REPO / "results/analysis/T4_3/golds"


def main():
    z = np.load(G / "splits.npz", allow_pickle=True)
    print({k: z[k].shape for k in z.files}, flush=True)
    rows = []
    for gold, cohorts in (("h3", ["liv_hmouse", "liv_4sp", "liv_hmonkey", "liv_hzeb"]), ("h4", ["test_hmouse", "test_hmac", "test_3sp"])):
        genes = pd.Index(z[f"{gold}_genes"] if f"{gold}_genes" in z.files else z["genes"]).astype(str)
        A, B = z[f"{gold}_a"], z[f"{gold}_b"]
        for c in cohorts:
            t = CM.load(c).set_index("hsapiens_gene_id")
            for name, m in (("masked", t.masked & t.clean), ("unmasked", ~t.masked & t.clean)):
                ids = t.index[m]
                p = genes.get_indexer(ids)
                p = p[p >= 0]
                rs = []
                for i in range(A.shape[0]):
                    a, b = A[i, p], B[i, p]
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.sum() >= 30:
                        rs.append(spearmanr(a[ok], b[ok])[0])
                med = float(np.median(rs)) if rs else np.nan
                rows.append(dict(gold=gold, cohort=c, stratum=name, n=int(len(p)), n_splits=len(rs), r_half=med, r_sb=2 * med / (1 + med) if rs else np.nan))
    d = pd.DataFrame(rows)
    d.to_csv(RU.RES / "gold_reliability.csv", index=False)
    print(d.to_string(), flush=True)
    h = np.load(G / "h3_profiles.npz", allow_pickle=True)
    ids = pd.Index(h["genes"]).astype(str)
    f = pd.Series(np.minimum(h["TS_f"].max(1) if h["TS_f"].shape[0] == len(ids) else h["TS_f"].max(0), h["TMS_f"].max(1) if h["TMS_f"].shape[0] == len(ids) else h["TMS_f"].max(0)), index=ids)
    rows = []
    for c in ("liv_hmouse", "liv_4sp"):
        t = CM.load(c).set_index("hsapiens_gene_id")
        for name, m in (("masked", t.masked & t.clean), ("unmasked", ~t.masked & t.clean)):
            ff = f.reindex(t.index[m]).dropna()
            rows.append(dict(cohort=c, stratum=name, n=len(ff), gold_max_type_det_median=float(ff.median()), training_det_median=float(t.loc[m, "detection_spmean"].median()), expressing_cells_median=float(t.loc[m, "n_expr_total"].median())))
    pd.DataFrame(rows).to_csv(RU.RES / "gold_depth.csv", index=False)
    print(pd.DataFrame(rows).to_string(), flush=True)


if __name__ == "__main__":
    main()
