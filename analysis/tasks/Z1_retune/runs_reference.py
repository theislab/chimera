"""Write results/analysis/Z1/runs_reference.tsv: the canonical and first-pass runs of the diagnostics table."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "training" / "runs"
C = {
    "liv_hmonkey": ["iter90-150ep-s42", "iter90-150ep-s43", "iter90-150ep-s44"],
    "liv_hmouse": ["xds-hmouse-iter90-150ep-s42", "xds-hmouse-iter90-150ep-s43", "xds-hmouse-iter90-150ep-s44"]
    + [f"rev-liv_hmouse-s{s}" for s in range(45, 52)],
    "liv_hzeb": ["xds-hzeb-iter85-s42"],
    "liv_4sp": ["xds-4sp-iter101-adv200-mmd100-fb10-200ep-s42"] + [f"rev-liv_4sp-s{s}" for s in range(43, 47)],
    "test_hmac": ["testis-hmac-iter109-pclamp085-200ep-s42"],
    "test_hmouse": ["testis-hmouse-iter111-pclamp-coc-200ep-s42"] + [f"rev-test_hmouse-s{s}" for s in range(43, 52)],
    "test_3sp": ["testis-unified-iter115-pclamp-coc-100ep-s42"] + [f"rev-test_3sp-s{s}" for s in range(43, 47)],
    "bone_marrow": ["rev-bm2-grid5-s42", "rev-bm2-s46"],
    "kidney": ["rev-kid-grid3-s42"] + [f"rev-kid-s{s}" for s in range(43, 47)],
    "islet": ["rev-isl-grid14-s42"] + [f"rev-isl-s{s}" for s in range(43, 47)],
    "vz_full": ["rev-vz-grid1-s42", "rev-vz-s44", "rev-vz-s45", "rev-vz-s46"],
    "vz_hmmul": ["rev-vz-hmmul-grid1-s42", "rev-vz-hmmul-s43", "rev-vz-hmmul-s44"],
    "vz_hmouse": ["rev-vz-hmouse-grid1-s42", "rev-vz-hmouse-s43"],
    "vz_primates": ["rev-vz-primates-grid2-s42", "rev-vz-primates-s43", "rev-vz-primates-s44"],
    "vz_hpig": [f"rev-vz-hpig-grid{k}-s42" for k in range(1, 5)],
    "vz_hmfas": [f"rev-vz-hmfas-grid{k}-s42" for k in range(1, 5)],
}
rows = [dict(cohort=k, run=r) for k, v in C.items() for r in v if (RUNS / r / "final_state.npz").exists()]
missing = [r for v in C.values() for r in v if not (RUNS / r / "final_state.npz").exists()]
print("missing", missing)
out = REPO / "results" / "analysis" / "Z1" / "runs_reference.tsv"
pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
print(len(rows), out)
