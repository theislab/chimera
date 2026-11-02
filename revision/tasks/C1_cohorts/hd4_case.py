"""C1: program transfer per type (H-D4 of P13, its code unchanged) on a disease case; judged against the label null.

Universe: human-mouse pairs of the C1 ortholog rule (Compara 110, both dumps, high confidence, protein coding, no retrocopy target), independent of any model vocabulary.

    python revision/tasks/C1_cohorts/hd4_case.py --case ad [--null --null-type microglia] [--null-combine]
"""
from __future__ import annotations

import sys

import pandas as pd

from revision.common import gene_ids
from revision.tasks.P13_kidney import hd4 as H

HM = gene_ids.REPO_ROOT / "data/revision/new_usecases/c1_build/objects/ortholog_table_hm_all.csv"


def main() -> int:
    case = sys.argv[sys.argv.index("--case") + 1]
    argv = [x for i, x in enumerate(sys.argv) if x != "--case" and sys.argv[i - 1] != "--case"]
    res = gene_ids.REPO_ROOT / "results" / "revision" / "C1" / case
    H.RES, H.DE, H.OUT = res, res / "de", res / "hd4"
    H.OUT.mkdir(parents=True, exist_ok=True)
    H.one2one = lambda: pd.read_csv(HM)[["hsapiens_gene_id", "mmusculus_gene_id"]].rename(columns={"hsapiens_gene_id": "human_id", "mmusculus_gene_id": "mouse_id"})
    sys.argv = argv
    return H.main()


if __name__ == "__main__":
    raise SystemExit(main())
