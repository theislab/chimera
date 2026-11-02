"""T7.9: gate G-D3h (plan section 8b). Reads ``integration_vs_scvi.json`` of each seed run, written by
``training/compare_vs_scvi.py --seed 42`` (scVI with batch_key = species, one cached seed-42
reference for all seeds). Pass: >= 3 of 5 seeds meet E8.6 and E8.3. A seed without the file, or
with a non-finite latent, counts as failing.

    python revision/tasks/T7_9_islet/gd3h.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "training" / "runs"
SEED_MAP = REPO / "revision" / "tasks" / "T7_9_recipe_b" / "seed_map.tsv"
OUT = REPO / "results" / "revision" / "T7_9" / "gd3h"
NEED = 3


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in pd.read_csv(SEED_MAP, sep="\t").iterrows():
        name = str(r["source"]).split()[0].replace("training/runs/", "")
        f = RUNS / name / "integration_vs_scvi.json"
        row = dict(seed=int(r["seed"]), run=name, found=f.exists())
        if f.exists():
            d = json.loads(f.read_text())
            e8 = {e["id"]: e for e in d["e8"]}
            ch, sc = d["rows"][0]["global"], d["rows"][1]["global"]
            row.update(scvi_name=d["rows"][1]["name"],
                       sil_celltype=ch["sil_celltype"], sil_celltype_scvi=sc["sil_celltype"],
                       label_transfer=ch["label_transfer_acc"], label_transfer_scvi=sc["label_transfer_acc"],
                       knn_mixing_entropy=ch["knn_mixing_entropy"], knn_mixing_entropy_scvi=sc["knn_mixing_entropy"],
                       knn_species_purity=ch["knn_species_purity"], knn_species_purity_scvi=sc["knn_species_purity"],
                       **{f"{k}_pass": bool(v["pass"]) for k, v in e8.items()})
        row["meets"] = bool(row.get("E8.3_pass", False) and row.get("E8.6_pass", False))
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "gd3h.tsv", sep="\t", index=False)
    res = dict(n_meet=int(t["meets"].sum()), n_seeds=len(t), need=NEED, pass_=bool(t["meets"].sum() >= NEED))
    (OUT / "gd3h.json").write_text(json.dumps(res, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.to_string(index=False))
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
