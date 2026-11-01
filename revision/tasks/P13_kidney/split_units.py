"""P13: fix every unit's role for the kidney AKI case (plan section 8c), from obs metadata only.

No expression value is read: only ``obs`` columns of the two CELLxGENE files.

Units. Human: a donor within one chemistry (``assay``); a donor with both chemistries gives one
unit per chemistry, and its role (fixed per donor) covers both. Mouse: an animal; Kirita 2020
libraries ``<name>b1`` / ``<name>b2`` (``experiment``) of one animal are merged, every other
library is one animal.

Human. Healthy pool = ``Conditions.Subtype`` in HEALTHY. Donors with multiome nuclei are split
within collection (biopsy / nephrectomy) x sex strata: IDs sorted, permuted with
``numpy.random.default_rng(SEED)`` (one generator, strata in sorted order), the first ceil(n/2)
go to the DE half (``de_control``), the rest to the training half (``train_healthy``). Healthy
donors with only 10x 3' v3 nuclei are ``de_control`` (v3 sensitivity only). AKI donors are
``de_disease``. All other subtypes are not listed.

Mouse. KPMP multiome (``source == current study``): ``Ref`` -> ``train_healthy``; ``Sham`` ->
``de_control``; IRI -> ``de_disease``; stage = acute (2 d) or late (4-24 wk). Kirita 2020
(10x 3' v2): Sham -> ``sens_control``, IRI -> ``sens_disease``. Gerhardt 2023 is not listed
(lineage-sorted GFP fractions).

    python -I revision/tasks/P13_kidney/split_units.py <human.h5ad> <mouse.h5ad> <out.tsv>
"""

from __future__ import annotations

import math
import re
import sys

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

SEED = 20261101
HEALTHY = ("Healthy Reference Tissue", "Living Donor Healthy Reference Tissue (KPMP)")
AKI = "Acute Kidney Injury"
MULTIOME, V3, V2 = "10x multiome", "10x 3' v3", "10x 3' v2"
BIOPSY = {"Percutaneous Needle Biopsy": "biopsy", "Transplant Pre-perfusion Biopsy": "biopsy",
          "Intra-operative Biopsy": "biopsy", "Nephrectomy": "nephrectomy",
          "Deceased Donor Nephrectomy": "nephrectomy"}
STAGE = {"2d": "acute", "4hrs": "acute", "12hrs": "acute", "2wk": "late", "4-6wk": "late", "20-24wk": "late"}
COLS = ["species", "unit", "donor", "role", "condition", "assay", "source", "collection", "tissue_collection",
        "sex", "age", "stage", "time", "subtype", "n_libraries", "n_nuclei"]


def _obs(path: str, cols: list[str]) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        return pd.DataFrame({c: read_elem(f["obs"][c]) for c in cols}).astype(str)


def human_roles(path: str) -> pd.DataFrame:
    o = _obs(path, ["donor_id", "Conditions.Subtype", "TissueCollection", "assay", "sex", "Age", "library"])
    o = o[o["Conditions.Subtype"].isin(HEALTHY + (AKI,))]
    d = o.groupby("donor_id").agg(subtype=("Conditions.Subtype", "first"), sex=("sex", "first"),
                                  age=("Age", "first"),
                                  tc=("TissueCollection", lambda s: "|".join(sorted(s.unique()))),
                                  assays=("assay", lambda s: "|".join(sorted(s.unique()))))
    for c in ("Conditions.Subtype", "sex", "TissueCollection"):
        if (o.groupby("donor_id")[c].nunique() > 1).any():
            raise ValueError(f"a donor has more than one {c}")
    d["collection"] = d["tc"].map(BIOPSY)
    if d["collection"].isna().any():
        raise ValueError(f"unmapped collection: {sorted(d.loc[d.collection.isna(), 'tc'].unique())}")
    role = pd.Series("", index=d.index)
    role[d["subtype"] == AKI] = "de_disease"
    healthy = d[d["subtype"].isin(HEALTHY)]
    role[healthy.index[~healthy["assays"].str.contains(MULTIOME, regex=False)]] = "de_control"
    multi = healthy[healthy["assays"].str.contains(MULTIOME, regex=False)]
    rng = np.random.default_rng(SEED)
    for _, g in multi.groupby(["collection", "sex"], sort=True):
        perm = rng.permutation(np.array(sorted(g.index)))
        k = math.ceil(len(perm) / 2)
        role[perm[:k]] = "de_control"
        role[perm[k:]] = "train_healthy"
    if (role == "").any():
        raise ValueError("unassigned human donor")
    d["role"] = role
    per = o.groupby(["donor_id", "assay"]).agg(n_nuclei=("sex", "size"), n_libraries=("library", "nunique"))
    rows = []
    for (donor, assay), r in per.iterrows():
        x = d.loc[donor]
        rows.append(dict(species="Human", unit=f"{donor}|{assay}", donor=donor, role=x.role,
                         condition="disease" if x.subtype == AKI else "control", assay=assay, source="KPMP",
                         collection=x.collection, tissue_collection=x.tc, sex=x.sex, age=x.age, stage="NA",
                         time="NA", subtype=x.subtype, n_libraries=int(r.n_libraries), n_nuclei=int(r.n_nuclei)))
    return pd.DataFrame(rows)


def _animal(source: str, experiment: str, donor: str) -> str:
    if source == "Kirita2020":
        m = re.match(r"^(.*)b\d$", experiment)
        return m.group(1) if m else experiment
    return donor


def mouse_roles(path: str) -> pd.DataFrame:
    o = _obs(path, ["donor_id", "experiment", "source", "assay", "condition_level2", "condition_level3", "sex",
                    "development_stage", "library"])
    o["animal"] = [_animal(s, e, d) for s, e, d in zip(o["source"], o["experiment"], o["donor_id"])]
    g = o.groupby("animal")
    d = g.agg(source=("source", "first"), assay=("assay", "first"), cond=("condition_level2", "first"),
              time=("condition_level3", "first"), sex=("sex", "first"), age=("development_stage", "first"),
              n_libraries=("library", "nunique"), n_nuclei=("sex", "size"))
    for c in ("source", "assay", "condition_level2", "condition_level3", "sex"):
        if (g[c].nunique() > 1).any():
            raise ValueError(f"an animal has more than one {c}")
    rows = []
    for animal, r in d.iterrows():
        if r.source == "current study":
            if r.cond == "Ref":
                role, cond, stage = "train_healthy", "control", "NA"
            elif r.cond == "Sham":
                role, cond, stage = "de_control", "control", "NA"  # set from injury_condition below
            else:
                role, cond, stage = "de_disease", "disease", STAGE[r.time]
        elif r.source == "Kirita2020":
            role = "sens_control" if r.cond == "Sham" else "sens_disease"
            cond = "control" if r.cond == "Sham" else "disease"
            stage = "late" if r.cond == "Sham" else STAGE[r.time]
        else:
            continue
        rows.append(dict(species="Mouse", unit=animal, donor=animal, role=role, condition=cond, assay=r.assay,
                         source=r.source, collection="NA", tissue_collection="NA", sex=r.sex, age=r.age,
                         stage=stage, time=r.time, subtype=r.cond, n_libraries=int(r.n_libraries),
                         n_nuclei=int(r.n_nuclei)))
    out = pd.DataFrame(rows)
    # Sham stage from the injury_condition label (Sham_2d / Sham_20wk), read separately.
    ic = _obs(path, ["donor_id", "injury_condition"]).drop_duplicates().set_index("donor_id")["injury_condition"]
    sham = (out["source"] == "current study") & (out["role"] == "de_control")
    lab = out.loc[sham, "unit"].map(ic)
    st = lab.map({"Sham_2d": "acute", "Sham_20wk": "late"})
    if st.isna().any():
        raise ValueError(f"sham without stage: {lab[st.isna()].tolist()}")
    out.loc[sham, "stage"] = st.to_numpy()
    out.loc[sham, "time"] = lab.to_numpy()
    return out


def main() -> int:
    hp, mp, outp = sys.argv[1:4]
    tab = pd.concat([human_roles(hp), mouse_roles(mp)], ignore_index=True).reindex(columns=COLS)
    tab = tab.sort_values(["species", "role", "unit"]).reset_index(drop=True)
    tab.to_csv(outp, sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(tab.groupby(["species", "role", "assay"]).size().to_string())
    h = tab[(tab.species == "Human") & (tab.assay == MULTIOME)]
    print(h.groupby(["role", "collection", "sex"]).size().to_string())
    print(tab[tab.species == "Mouse"].groupby(["role", "source", "stage"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
