"""C1: recipe grid and blind selection (plan section 13.0). No reference, test or disease file is read here.

    python revision/tasks/C1_cohorts/grid.py manifest <cohort> [--round 1]   # writes grids/<cohort>_r<k>.tsv (run_name, args)
    python revision/tasks/C1_cohorts/grid.py report <cohort>                 # health, rho, embedding per run; selection
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from revision.tasks.C1_cohorts import configs as CF

ROOT = CF.ROOT
GRIDS = CF.TASK / "grids"
RUNS = ROOT / "training" / "runs"
FORMS = {2: {"a": (100, 50, 0.5, 150), "b": (200, 50, 1.0, 200)},
         3: {"a": (133, 100, 1.0, 200), "b": (200, 100, 1.0, 200)},
         4: {"a": (150, 150, 1.0, 200), "b": (200, 100, 1.0, 200)}}
TOP_SEED_BAND = 0.03
FLAG_BAND = 0.05


def tag(cohort: str) -> str:
    return cohort.replace("_", "-")


def depth_ratio(cohort: str) -> tuple[float, int]:
    rep = json.loads((CF.RES / cohort / "build_report.json").read_text())
    return float(rep["depth_ratio_to_testis3sp"]), len(rep["median_library"])


def anchor_base(r: float, k: int) -> float:
    return round(0.003 * math.sqrt(max(r, 1.0)), 5) if k == 2 else round(0.001 * max(r, 1.0), 5)


def args_for(cohort: str, form: str, anchor: float, seed: int, k: int) -> str:
    adv, mmd, fb, ep = FORMS[k][form]
    a = str(CF.h5ad(cohort).relative_to(ROOT))
    return (f"--config A --max-epochs {ep} --adata {a} --seed {seed} --batch-key batch --adv-weight {adv} --lambda-anchor {anchor} "
            f"--lambda-mmd {mmd} --free-bits {fb} --celltype-conditional-mmd --celltype-key cell_ontology_class --prior-clamp-max 0.85")


def manifest(cohort: str, rnd: int = 1, anchors: list[float] | None = None, forms: tuple = ("a", "b")) -> Path:
    r, k = depth_ratio(cohort)
    A = anchor_base(r, k)
    rows = []
    GRIDS.mkdir(exist_ok=True)
    for form in forms:
        for lab, an in (("A", A), ("2A", round(2 * A, 5))) if anchors is None else [(f"x{an_}", an_) for an_ in anchors]:
            rows.append(dict(run_name=f"rev-c1-{tag(cohort)}-{form}{lab}-s42", args=args_for(cohort, form, an, 42, k)))
    p = GRIDS / f"{cohort}_r{rnd}.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    print(p, len(rows), "depth ratio", round(r, 2), "A", A)
    return p


def seeds_manifest(cohort: str, row: str, seeds=(43, 44)) -> Path:
    rows = []
    base = RUNS / f"rev-c1-{tag(cohort)}-{row}-s42" / "_provenance.json"
    argv = json.loads(base.read_text())["argv"]
    for s in seeds:
        v = argv.copy()
        v[v.index("--seed") + 1] = str(s)
        rows.append(dict(run_name=f"rev-c1-{tag(cohort)}-{row}-s{s}", args=" ".join(v)))
    p = GRIDS / f"{cohort}_seeds_{row}.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return p


def runs_of(cohort: str) -> list[str]:
    import re

    rx = re.compile(rf"^rev-c1-{re.escape(tag(cohort))}-[ab](A|2A|x[0-9.]+)-s\d+$")
    return sorted(d.name for d in RUNS.glob(f"rev-c1-{tag(cohort)}-*-s*") if d.is_dir() and rx.match(d.name))


def embedding(run: str, a: ad.AnnData) -> dict:
    sys.path.insert(0, str(ROOT / "training"))
    from compare_vs_scvi import compute_integration_metrics  # noqa: PLC0415

    with np.load(RUNS / run / "final_state.npz", allow_pickle=True) as z:
        zm = np.asarray(z["z_mean"], float)
    if not np.isfinite(zm).all():
        return {}
    m = compute_integration_metrics(zm, a.obs["species"].astype(str).to_numpy(), a.obs["cell_ontology_class"].astype(str).to_numpy(), k=15, seed=42)["global"]
    return {k: float(v) for k, v in m.items() if np.isscalar(v)}


def scvi_reference(cohort: str) -> dict:
    """scVI(species) on the same object, the cached latent of training/compare_vs_scvi.py (trained there, seed 42)."""
    f = ROOT / "training" / "runs" / "baselines_scvi" / f"{CF.h5ad(cohort).stem}-s42"
    sum_f = CF.RES / cohort / "scvi_reference.json"
    return json.loads(sum_f.read_text()) if sum_f.exists() else {"missing": str(f)}


def canonical_embedding_floor(k: int) -> dict:
    t = pd.read_csv(ROOT / "results/revision/H1/cohort_table.csv")
    return t


CANON_LT_FLOOR = {2: 0.704, 3: 0.956}  # lowest label transfer of the canonical cohorts per K class, results/revision/H1/cohort_table.csv


def select(t: pd.DataFrame, k: int, scvi: dict) -> dict:
    """Plan 13.0: among healthy s42 rows, rows within 0.03 of the best rho, higher composite wins; a row below the canonical floor is flagged and a healthy unflagged row within 0.05 replaces it."""
    h = t[(t["healthy"] == True) & t["run"].str.endswith("-s42")].copy()  # noqa: E712
    if h.empty:
        return dict(selected=None, reason="no healthy row")
    floor = CANON_LT_FLOOR[2 if k == 2 else 3]
    h["flag_embedding"] = h["emb_label_transfer_acc"] < floor
    best = h["rho_DS"].max()
    band = h[h["rho_DS"] >= best - TOP_SEED_BAND].copy()

    def mm(x):
        return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x * 0 + 0.5

    band["composite"] = 0.4 * mm(band["emb_knn_mixing_entropy"]) + 0.6 * (mm(band["emb_label_transfer_acc"]) + mm(band["emb_sil_celltype"])) / 2
    pick = band.sort_values(["composite", "rho_DS"], ascending=False).iloc[0]
    if pick["flag_embedding"]:
        alt = h[(~h["flag_embedding"]) & (h["rho_DS"] >= pick["rho_DS"] - FLAG_BAND)]
        if len(alt):
            pick = alt.sort_values("rho_DS", ascending=False).iloc[0]
    return dict(selected=pick["run"], rho_DS=float(pick["rho_DS"]), label_transfer=float(pick["emb_label_transfer_acc"]), floor=floor, flagged=bool(pick["flag_embedding"]), n_healthy=len(h), scvi=scvi)


def anchored_tails(cohort: str) -> bool:
    """The ruling's row names primate cortex. Its threshold (more than half of genes at pi >= 0.90) also catches lung (0.54) and M1 (0.59),
    where the raw upper tail is reachable (5-11% of genes) and both raw tails pass; the rule is applied as the row states it, to the DLPFC cohorts
    (0.89-0.94), where it is out of reach."""
    return cohort.startswith("brain_dlpfc")


def report(cohort: str) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "revision/tasks/Z1_retune"))
    import diagnostics as ZD  # noqa: PLC0415

    a = ad.read_h5ad(CF.h5ad(cohort), backed="r")
    a2 = ad.AnnData(obs=a.obs.copy())
    a.file.close()
    rows = []
    for run in runs_of(cohort):
        if not (RUNS / run / "final_state.npz").exists():
            rows.append(dict(run=run, finished=False))
            continue
        r = ZD.one_run(run, with_mixing=False)
        if r.get("no_nan"):
            r.update({f"emb_{k}": v for k, v in embedding(run, a2).items()})
        rows.append(r)
    t = pd.DataFrame(rows)
    t.insert(0, "cohort", cohort)
    if anchored_tails(cohort):
        # Coordinator ruling (amendment log): more than half of genes at pi >= 0.90: both tails against the anchored prior; saturation and every other criterion unchanged.
        t["healthy_raw_rule"] = t["healthy"]
        t["healthy"] = (t.get("no_nan") == True) & (t["n_active_z"] == 10) & (t["tail_up_tr"] >= ZD.TAIL_MIN) & (t["tail_dn_tr"] >= ZD.TAIL_MIN) & (t["sat_hi"] <= ZD.SAT_MAX) & (t["sat_lo"] <= ZD.SAT_MAX)  # noqa: E712
    out = CF.RES / cohort
    t.to_csv(out / "grid_report.tsv", sep="\t", index=False)
    k = int(depth_ratio(cohort)[1])
    if "healthy" in t:
        sel = select(t, k, scvi_reference(cohort))
        (out / "selection.json").write_text(json.dumps(sel, indent=2, default=float))
        print(sel)
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "report", "seeds"])
    ap.add_argument("cohort")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--anchors", default=None)
    ap.add_argument("--row", default=None)
    ap.add_argument("--forms", default="ab")
    a = ap.parse_args()
    if a.cmd == "manifest":
        manifest(a.cohort, a.round, [float(x) for x in a.anchors.split(",")] if a.anchors else None, tuple(a.forms))
    elif a.cmd == "seeds":
        print(seeds_manifest(a.cohort, a.row))
    else:
        t = report(a.cohort)
        cols = [c for c in ["run", "healthy", "rho_DS", "rho_DS_partial", "frac_support", "sat_hi", "tail_up_raw", "tail_dn_raw", "n_active_z", "adv_acc", "last_epoch", "lr_freeze_epoch",
                            "emb_knn_mixing_entropy", "emb_label_transfer_acc", "emb_sil_celltype", "emb_knn_species_purity"] if c in t]
        print(t[cols].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
