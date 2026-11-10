"""T11.1: check the Van Zyl object against the raw GEO matrices, gene by gene.

For every species and every ortholog group:
* the object's per-gene total equals the total of the CSV row the group maps to (its Ensembl
  ID row when the matrix has one, else its symbol row);
* false zeros: the group is all-zero in the object although the CSV has counts under the
  group's Ensembl ID, or under its (unique) symbol when the matrix has no ID row;
* for groups whose ID and symbol are both rows of the matrix (audit R4 n1), the object column
  equals the ID row cell by cell.

Writes ``verify_build.json`` beside the object and exits 1 on any mismatch.

    python analysis/tasks/T11_1_vanzyl/verify_build.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "analysis" / "tasks" / "T11_1_vanzyl"))

from build_vanzyl import DEFAULT_OUT, ENSEMBL, OWN_MATRIX, RAW  # noqa: E402


def main() -> int:
    import anndata as ad

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    out = ap.parse_args().out
    a = ad.read_h5ad(out / "van_zyl_unified.h5ad")
    table = pd.read_csv(out / "ortholog_table_van_zyl.csv")
    both = pd.DataFrame(json.loads((out / "build_report.json").read_text())["measured_genes"]["symbol_and_id_rows"])
    src2new = dict(zip(table["source_ortholog_group_id"], table["ortholog_group_id"]))
    sp_of = {v: k for k, v in ENSEMBL.items()}
    rep: dict = {}
    bad = 0
    for sp, label in ((s, sp_of[s]) for s in sorted(ENSEMBL.values())):
        csv = pd.read_csv(RAW / OWN_MATRIX[sp], index_col=0)
        csv.index = csv.index.astype(str)
        cells = a.obs_names[a.obs["species"].astype(str) == label]
        csv = csv[cells]  # all cells kept, same barcodes
        totals = csv.sum(axis=1)
        sub = a[cells]
        obj_tot = pd.Series(np.asarray(sub.X.sum(axis=0)).ravel(), index=table["ortholog_group_id"].to_numpy())
        gid, name = table[f"{sp}_gene_id"].astype(str), table[f"{sp}_gene_name"]
        name_unique = ~name.duplicated(keep=False) & name.notna()
        row = np.where(gid.isin(totals.index), gid, np.where(name_unique & name.isin(totals.index), name, None))
        expect = pd.Series([totals.get(r, np.nan) if r is not None else np.nan for r in row],
                           index=table["ortholog_group_id"].to_numpy())
        mism = int((~np.isclose(obj_tot.to_numpy(), expect.to_numpy())).sum())
        id_tot = gid.map(totals).fillna(0).to_numpy()
        nm_tot = np.where(name_unique, name.map(totals).fillna(0), 0)
        has_id = gid.isin(totals.index).to_numpy()
        # A symbol row counts only when the matrix has no row for the ID (it may be another gene).
        false_zero = (obj_tot.to_numpy() == 0) & ((id_tot > 0) | (~has_id & (nm_tot > 0)))
        r = {"n_groups": int(len(table)), "n_total_mismatch": mism, "n_false_zeros": int(false_zero.sum()),
             "false_zero_genes": table.loc[false_zero, "hsapiens_gene_name"].tolist()[:20],
             "n_all_zero": int((obj_tot == 0).sum())}
        b = both[both["species"] == sp] if len(both) else both
        if len(b):
            b = b[b["source_ortholog_group_id"].isin(src2new)]  # groups kept in the object
            ok = 0
            for _, x in b.iterrows():
                col = np.asarray(sub.X[:, int(src2new[x["source_ortholog_group_id"]])].todense()).ravel()
                ok += int(np.array_equal(col, csv.loc[x["id_row_used"]].to_numpy(dtype=col.dtype)))
            r["n_symbol_and_id_groups_kept"] = int(len(b))
            r["n_equal_to_id_row_cell_by_cell"] = ok
            r["genes"] = table.set_index("source_ortholog_group_id").loc[b["source_ortholog_group_id"], "hsapiens_gene_name"].tolist()
            bad += int(len(b) - ok)
        bad += mism + int(false_zero.sum())
        rep[sp] = r
        print(sp, json.dumps(r))
    (out / "verify_build.json").write_text(json.dumps(rep, indent=2))
    print("OK" if bad == 0 else f"{bad} problem(s)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
