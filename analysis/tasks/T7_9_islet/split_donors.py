"""T7.9: fix every donor's and library's role for the islet T2D case, from obs metadata only.

No expression value is read: only ``obs`` of the two CELLxGENE files.

Human (48-donor islet atlas). The 17 non-diabetic ("normal") donors are split into a
training half (healthy islet Chimera model only) and a DE half (control arm of the human
T2D vs non-diabetic contrast), so that no donor enters both. The split is stratified by
sex x 10x chemistry (v2 / v3); within each stratum donors are sorted by ID, permuted with
``numpy.random.default_rng(SEED)`` and the first ceil(n/2) go to the DE half. T2D donors
are DE; prediabetes donors are sensitivity only.

Mouse (Hrovatin 2023 atlas). The disease contrasts use only untreated libraries:
db/db (GSE174194) ``sham_Lepr-/-`` vs ``chow_WT``, mSTZ (GSE128565) ``STZ`` vs ``control``.
There are 3 control libraries in all, too few to split, so the mouse half of the healthy
model uses the atlas's other healthy adult in vivo samples, disjoint from every DE library:
``4m`` (GSE211796, 4 mice, 4 months) and ``aged`` (GSE211795, 3 libraries, 2 years);
together they span adult life, as the human donors (21-66 years) do. Excluded by
definition: embryonic, P16 (juvenile), chem (islets cultured ex vivo), NOD (T1D model).
Treated db/db (pair-fed, VSG) and treated mSTZ libraries are sensitivity only.

    python -I analysis/tasks/T7_9_islet/split_donors.py <human.h5ad> <mouse.h5ad> <out.tsv>
"""

from __future__ import annotations

import math
import sys

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

SEED = 20261101


def _obs(path: str, cols: list[str]) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        return pd.DataFrame({c: read_elem(f["obs"][c]) for c in cols}).astype(str)


def human_roles(path: str) -> pd.DataFrame:
    o = _obs(path, ["donor_id", "disease", "sex", "assay"])
    d = o.groupby("donor_id").agg(disease=("disease", "first"), sex=("sex", "first"),
                                  assay=("assay", lambda s: ",".join(sorted(s.unique()))),
                                  n_cells=("sex", "size"))
    if (d["assay"].str.contains(",")).any():
        raise ValueError("a donor spans two chemistries")
    rng = np.random.default_rng(SEED)
    role = pd.Series("", index=d.index)
    role[d["disease"] == "type 2 diabetes mellitus"] = "de_disease"
    role[d["disease"] == "prediabetes syndrome"] = "sensitivity_prediabetes"
    nd = d[d["disease"] == "normal"]
    for (sex, assay), g in nd.groupby(["sex", "assay"], sort=True):
        ids = np.array(sorted(g.index))
        perm = rng.permutation(ids)
        k = math.ceil(len(ids) / 2)
        role[perm[:k]] = "de_control"
        role[perm[k:]] = "train_healthy"
    if (role == "").any():
        raise ValueError("unassigned human donor")
    out = d.assign(role=role, species="Human", unit=d.index, study="human48",
                   condition=np.where(d["disease"] == "normal", "control", "disease"))
    return out.reset_index(drop=True)


DE_MOUSE = {  # dataset, design -> condition
    ("db/db", "chow_WT"): "control", ("db/db", "sham_Lepr-/-"): "disease",
    ("mSTZ", "control"): "control", ("mSTZ", "STZ"): "disease",
}
SENS_MOUSE = {("db/db", "PF_Lepr-/-"), ("db/db", "VSG_Lepr-/-"), ("mSTZ", "STZ_insulin"),
              ("mSTZ", "STZ_GLP-1"), ("mSTZ", "STZ_estrogen"), ("mSTZ", "STZ_GLP-1_estrogen"),
              ("mSTZ", "STZ_GLP-1_estrogen+insulin")}
TRAIN_MOUSE = {"4m", "aged"}


def mouse_roles(path: str) -> pd.DataFrame:
    o = _obs(path, ["dataset__design__sample", "dataset", "design", "sex", "assay", "age", "strain",
                    "disease", "GEO_accession"])
    d = o.groupby("dataset__design__sample").agg(
        dataset=("dataset", "first"), design=("design", "first"),
        sex=("sex", lambda s: ",".join(sorted(s.unique()))), assay=("assay", "first"),
        age=("age", "first"), strain=("strain", "first"), disease=("disease", "first"),
        geo=("GEO_accession", "first"), n_cells=("sex", "size"))
    rows = []
    for unit, r in d.iterrows():
        key = (r.dataset, r.design)
        if key in DE_MOUSE:
            role, cond = "de_" + DE_MOUSE[key], DE_MOUSE[key]
        elif key in SENS_MOUSE:
            role, cond = "sensitivity_treated", "disease"
        elif r.dataset in TRAIN_MOUSE:
            if r.disease != "normal":
                raise ValueError(f"{unit}: training library not normal")
            role, cond = "train_healthy", "control"
        else:
            continue
        rows.append(dict(unit=unit, species="Mouse", study=r.dataset, role=role, condition=cond,
                         disease=r.disease, sex=r.sex, assay=r.assay, age=r.age, strain=r.strain,
                         geo=r.geo, design=r.design, n_cells=int(r.n_cells)))
    out = pd.DataFrame(rows)
    for key in DE_MOUSE:
        if not ((out["study"] == key[0]) & (out["design"] == key[1])).any():
            raise ValueError(f"no library for {key}")
    return out


def main() -> int:
    hp, mp, outp = sys.argv[1:4]
    h, m = human_roles(hp), mouse_roles(mp)
    cols = ["species", "unit", "study", "role", "condition", "disease", "sex", "assay", "age", "strain",
            "geo", "design", "n_cells"]
    tab = pd.concat([h, m], ignore_index=True).reindex(columns=cols)
    tab = tab.sort_values(["species", "role", "unit"]).reset_index(drop=True)
    tab.to_csv(outp, sep="\t", index=False)
    print(tab.groupby(["species", "role"]).size().to_string())
    print(tab[(tab.species == "Human") & tab.role.isin(["de_control", "train_healthy"])]
          [["unit", "role", "sex", "assay"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
