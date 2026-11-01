"""T6.6 step 1: fix the liver-lineage experiment filter and the TF panel. Reads only ChIP-Atlas
column headers; no Chimera readout and no binding value. Run before ``run.py``.

    python revision/tasks/T6_6_regulatory/fix_panel.py
"""

from __future__ import annotations

import json
import re

import pandas as pd

from revision.common import gene_ids

FUNC = gene_ids.REPO_ROOT / "data/revision/functional/chipatlas"
OUT = gene_ids.REPO_ROOT / "results/revision/T6_6"
TFS = ["HNF4A", "CEBPA", "FOXA1", "FOXA2", "HNF1A", "ONECUT1", "RXRA", "PPARA", "NR1H3"]
LIVER_LINEAGE = re.compile(r"liver|hepat|^hep_g2$|^hep_3b$|^huh|^lo2$|^hepa_1-6$", re.I)
MIN_TFS = 3


def mouse_name(tf: str) -> str:
    return tf[0] + tf[1:].lower()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, used, unused = [], {}, {}
    for tf in TFS:
        r = dict(tf=tf)
        for g, name in (("hg38", tf), ("mm10", mouse_name(tf))):
            hdr = (FUNC / g / f"{name}.5.tsv").open().readline().rstrip("\n").split("\t")
            cols = [c for c in hdr[1:] if "|" in c and not c.endswith("|Average")]
            keep = [c for c in cols if LIVER_LINEAGE.search(c.split("|", 1)[1])]
            for c in cols:
                (used if c in keep else unused).setdefault(g, set()).add(c.split("|", 1)[1])
            r[f"{g}_experiments"] = len(cols)
            r[f"{g}_liver_lineage"] = len(keep)
            r[f"{g}_columns"] = ";".join(keep)
        rows.append(r)
    t = pd.DataFrame(rows)
    t["in_panel"] = (t.hg38_liver_lineage > 0) & (t.mm10_liver_lineage > 0)
    t.to_csv(OUT / "panel_filter.csv", index=False)
    panel = t.loc[t.in_panel, "tf"].tolist()
    (OUT / "panel.json").write_text(json.dumps(dict(
        panel=panel, n=len(panel), stop=len(panel) < MIN_TFS, filter=LIVER_LINEAGE.pattern,
        labels_used={g: sorted(v) for g, v in used.items()}), indent=1))
    print(t[["tf", "hg38_experiments", "hg38_liver_lineage", "mm10_experiments", "mm10_liver_lineage", "in_panel"]].to_string(index=False))
    print("used", {g: sorted(v) for g, v in used.items()})
    print("not used, labels containing hep/liv:", {g: sorted(x for x in v if re.search("hep|liv", x, re.I)) for g, v in unused.items()})


if __name__ == "__main__":
    main()
