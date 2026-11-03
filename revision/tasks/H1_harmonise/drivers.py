"""H1: cell-type contributions to S_g per cohort, the T2.5 split (C_gk, posterior-mean z), from existing forward passes. No forward pass is run here.

Canonical: T2.5 tables, the runs it covers. New cohorts: fw files of the owning tasks, mean over kept seeds with a file. Islet: Y1's cell-half forward pass, halves summed back to classes
(the sum over a class's two halves is the class's sum, by construction). Eye: T11 fw files; a cohort of the registry with no file yet is skipped and reported.

    PYTHONPATH=<chimera repo> python revision/tasks/H1_harmonise/drivers.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from revision.tasks.H1_harmonise import rules as RU
from revision.tasks.T2_5_celltype_margin import margin as MG

OUT = RU.OUT / "drivers"
RES = RU.RES


def fw_files(c: str, seeds: list[int]) -> tuple[list, str]:
    if c == "bone_marrow":
        return [(s, RES / f"P12/drivers/fw_s{s}.npz") for s in seeds], "cell_ontology_class"
    if c == "kidney":
        return [(s, RES / f"P13/drivers/fw_s{s}.npz") for s in seeds], "cell_ontology_class"
    if c == "islet":
        return [(s, RES / f"Y1/forward/fw_islet_s{s}.npz") for s in seeds], "y1half"
    return [(s, RES / f"T11/drivers/fw_{c[3:]}_s{s}.npz") for s in seeds], "cell_ontology_class"


def fold_halves(fw: dict, key: str, obs_label: np.ndarray | None = None) -> tuple[dict, str]:
    """y1half labels look like '<class>|<half>' or similar: fold them into classes."""
    g = fw[f"{key}__groups"]
    lab = np.asarray(g["label"]).astype(str)
    base = np.array([x.rsplit("|", 1)[0] for x in lab])
    sp = np.asarray(g["species_idx"], int)
    pairs = sorted({(int(a), b) for a, b in zip(sp, base)})
    idx = {p: i for i, p in enumerate(pairs)}
    to = np.array([idx[(int(a), b)] for a, b in zip(sp, base)])
    out = {k: v for k, v in fw.items() if not k.startswith(f"{key}__")}
    n = np.zeros(len(pairs))
    np.add.at(n, to, np.asarray(g["n"], float))
    out["cls__groups"] = np.array([(s, l, c) for (s, l), c in zip(pairs, n.astype(int))], dtype=[("species_idx", "i4"), ("label", "U64"), ("n", "i8")])
    for mode in ("meanz", "samp", "within"):
        M = np.zeros((len(pairs), fw[f"{key}__sum_{mode}"].shape[1]))
        np.add.at(M, to, fw[f"{key}__sum_{mode}"])
        out[f"cls__sum_{mode}"] = M
    return out, "cls"


def new_cohort(c: str, seeds: list[int]) -> pd.DataFrame | None:
    files, key = fw_files(c, seeds)
    frames, weights = [], None
    used = []
    for s, f in files:
        if not f.exists():
            continue
        fw = dict(np.load(f, allow_pickle=False))
        k = key if f"{key}__groups" in fw else str(fw["keys"][0])
        if k == "y1half":
            fw, k = fold_halves(fw, k)
        dec = MG.decompose(fw, k)
        frames.append(pd.DataFrame(dec["C_gk_meanz"], index=fw["var_names"].astype(str)))
        w = pd.Series(dec["w"]).groupby(dec["labels"]).sum()
        weights = w
        used.append(s)
    if not frames:
        return None
    m = sum(frames) / len(frames)
    m.attrs["weights"] = weights
    m.attrs["seeds"] = used
    return m


def canonical(c: str) -> pd.DataFrame | None:
    base = RES / "T2_5"
    t = "__cell_ontology_class" if c.startswith("test") else ""
    fs = [base / f"{c}{t}__C_gk.parquet"] + [base / f"{c}_s{s}{t}__C_gk.parquet" for s in (43, 44)]
    frames, seeds, weights = [], [], None
    for f, s in zip(fs, (42, 43, 44)):
        if f.exists():
            d = pd.read_parquet(f)
            frames.append(d.pivot(index="var_name", columns="cell_type", values="C_gk_meanz"))
            weights = d.groupby("cell_type").weight_k.first()
            seeds.append(s)
    if not frames:
        return None
    m = sum(frames) / len(frames)
    m.attrs["weights"], m.attrs["seeds"] = weights, seeds
    return m


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RU.OUT / "runs.tsv", sep="\t")
    meta = []
    for c, d in RU.cohorts().items():
        seeds = sorted(runs[(runs.cohort == c) & runs.kept].seed)
        if d.get("source") == "C1":  # C1 ran H1's drivers code on its own forward passes; taken as written
            src = RU.RES / "C1" / "h1" / "drivers"
            if (src / f"{c}.parquet").exists():
                pd.read_parquet(src / f"{c}.parquet").to_parquet(OUT / f"{c}.parquet")
                pd.read_csv(src / f"{c}__shares.csv").to_csv(OUT / f"{c}__shares.csv", index=False)
                mm = pd.read_csv(src / "meta.csv").set_index("cohort")
                meta.append(dict(cohort=c, seeds_with_forward=str(mm.loc[c, "seeds_with_forward"]), n_types=int(mm.loc[c, "n_types"]),
                                 closure_C_sum_vs_S_ema_pearson=float(mm.loc[c, "closure_C_sum_vs_S_ema_pearson"]), note="C1 forward passes"))
            else:
                meta.append(dict(cohort=c, seeds_with_forward="", note="no forward pass yet"))
            continue
        m = canonical(c) if d["role"] == "reference" else new_cohort(c, seeds)
        if m is None:
            meta.append(dict(cohort=c, seeds_with_forward="", note="no forward pass yet"))
            continue
        w, sd = m.attrs["weights"], m.attrs["seeds"]
        g = pd.read_parquet(RU.OUT / "genes" / f"{c}.parquet").reindex(m.index)
        m2 = m.copy()
        m2.attrs = {}
        m2.to_parquet(OUT / f"{c}.parquet")
        cs = m.sum(1)
        un = g.count_support.to_numpy()
        closure = float(np.corrcoef(cs[un], g.S_ema[un])[0, 1])
        A = m.to_numpy()
        ty = np.array(m.columns)
        dom = ty[np.abs(A).argmax(1)]
        strict = g.strict_support.to_numpy()
        share_neg = np.abs(np.where(A < 0, A, 0)).sum(0)
        share_pos = np.where(A > 0, A, 0).sum(0)
        s = pd.DataFrame(dict(cohort=c, cell_type=ty, weight=w.reindex(ty).to_numpy(), share_abs_S=np.abs(A[un]).sum(0) / np.abs(A[un]).sum(), share_negative=share_neg / share_neg.sum(),
                              share_positive=share_pos / share_pos.sum(), dominant_in_strict_up=[(dom[strict & (g.D.to_numpy() > 0)] == k).mean() for k in ty],
                              dominant_in_strict_down=[(dom[strict & (g.D.to_numpy() < 0)] == k).mean() for k in ty]))
        s.to_csv(OUT / f"{c}__shares.csv", index=False)
        meta.append(dict(cohort=c, seeds_with_forward=",".join(map(str, sd)), n_types=len(ty), closure_C_sum_vs_S_ema_pearson=closure))
    pd.DataFrame(meta).to_csv(OUT / "meta.csv", index=False)
    print(pd.DataFrame(meta).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
