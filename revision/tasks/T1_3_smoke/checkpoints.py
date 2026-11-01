"""T1.3: load the 11 canonical checkpoints on CPU and compare alpha with final_state.npz."""
import sys
import time

import anndata as ad
import numpy as np
from scipy.stats import spearmanr

from chimera import ChimeraModel
from revision.common.gene_ids import H5ADS, REPO_ROOT

RUNS = [
    ("liv_hmonkey", "iter90-150ep-s42"),
    ("liv_hmonkey", "iter90-150ep-s43"),
    ("liv_hmonkey", "iter90-150ep-s44"),
    ("liv_hmouse", "xds-hmouse-iter90-150ep-s42"),
    ("liv_hmouse", "xds-hmouse-iter90-150ep-s43"),
    ("liv_hmouse", "xds-hmouse-iter90-150ep-s44"),
    ("liv_hzeb", "xds-hzeb-iter85-s42"),
    ("liv_4sp", "xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42"),
    ("test_hmac", "testis-hmac-iter109-pclamp085-200ep-s42"),
    ("test_hmouse", "testis-hmouse-iter111-pclamp-coc-200ep-s42"),
    ("test_3sp", "testis-unified-iter115-pclamp-coc-100ep-s42"),
]

out = open(sys.argv[1] + ".tmp", "w")
out.write("cohort\trun\tloaded\tn_genes\tspearman_alpha\tmax_abs_diff\tseconds\terror\n")
for cohort, run in RUNS:
    t = time.time()
    rd = REPO_ROOT / "training" / "runs" / run
    try:
        adata = ad.read_h5ad(REPO_ROOT / "data" / "processed" / f"{H5ADS[cohort]}.h5ad")
        model = ChimeraModel.load(str(rd / "model"), adata=adata, accelerator="cpu")
        a = model.get_alpha()
        ref = np.load(rd / "final_state.npz")["alpha"]
        rho = spearmanr(a, ref)[0]
        row = f"{cohort}\t{run}\tyes\t{a.size}\t{rho:.6f}\t{np.abs(a - ref).max():.3g}\t{time.time() - t:.0f}\t"
    except Exception as e:  # recorded, triaged in the ledger
        row = f"{cohort}\t{run}\tno\t\t\t\t{time.time() - t:.0f}\t{type(e).__name__}: {str(e)[:200]}"
    print(row, flush=True)
    out.write(row + "\n")
out.close()
import os
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
