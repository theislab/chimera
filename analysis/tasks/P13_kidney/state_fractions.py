"""Author injury-state fractions (CellStateLevel2) of the kidney case, per type x role x species
(pre-specified analysis plan, section 8c).

Descriptive, and partly circular: the collection's shared state labels were made by mapping mouse
onto human. Pooled over units of each role (multiome primary arms; Kirita and human v3 apart).
Reads ``results/analysis/P13/states_{human,mouse}_unit_type.csv`` and ``unit_roles.tsv``.

    PYTHONPATH=<chimera repo> python -I analysis/tasks/P13_kidney/state_fractions.py
"""

from __future__ import annotations

import sys

import pandas as pd

from analysis.common import gene_ids

ROOT = gene_ids.REPO_ROOT
RES = ROOT / "results" / "analysis" / "P13"
ROLES = ROOT / "analysis" / "tasks" / "P13_kidney" / "unit_roles.tsv"


def main() -> int:
    roles = pd.read_csv(ROLES, sep="\t", dtype=str).set_index("unit")
    parts = []
    for s in ("human", "mouse"):
        t = pd.read_csv(RES / f"states_{s}_unit_type.csv")
        r = roles.loc[t["unit"]]
        t["group"] = (r["species"] + "|" + r["role"] + "|" + r["assay"]).to_numpy()
        parts.append(t)
    t = pd.concat(parts)
    tab = t.pivot_table(index=["cell_ontology_class", "group"], columns="CellStateLevel2", values="n_nuclei",
                        aggfunc="sum", fill_value=0)
    n = tab.sum(axis=1)
    frac = tab.div(n, axis=0).round(4)
    frac.insert(0, "n_nuclei", n)
    frac["altered"] = (1 - frac.get("reference", 0)).round(4)
    frac.reset_index().to_csv(RES / "state_fractions.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(frac.loc[["pt", "tal", "fib", "ec"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
