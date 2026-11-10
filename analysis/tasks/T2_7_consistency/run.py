"""T2.7: measured values for the internal-consistency repairs of the manuscript tables.

One row per value: the item, its manuscript location, the value the manuscript states,
the value measured here from the h5ads, ``final_state.npz``, checkpoint configs and code,
and the source of the measurement. Items:

* ``tables/per-dataset-recipes.tex``: N (and per species), G, B, shared cell types on
  ``cell_type`` (and ``cell_ontology_class`` on testis) with coverage, alpha and delta-alpha
  summaries, E9.1 minimum cross-seed Spearman.
* ``tables/assumptions-summary.tex:56`` (prior clamp binding), ``:72``, ``:135``, ``:169``
  (cross-seed gate: what it measures and its values for alpha and delta-alpha).
* ESM-2 model: embedding dimension of the SATURN ESM-2 files and the code constant.
* M/F ortholog-QC flags: the rule as written, counts per cohort, and whether the
  hand-labelled attribution candidates satisfy it.

Usage::

    python analysis/tasks/T2_7_consistency/run.py [--outdir DIR] [--prerun]
"""

from __future__ import annotations

import argparse
import itertools
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.common import gene_ids
from analysis.common.outputs import TaskOutput, add_common_args

ROOT = gene_ids.REPO_ROOT
RECIPES = "tables/per-dataset-recipes.tex"
ASSUME = "tables/assumptions-summary.tex"

# What the submitted tables state, transcribed (manuscript location -> text).
STATED = {
    "liv_hmonkey": dict(row="Liver, H+Mac", N="≈12 k", G="≈12 k", B="3", shared_ct="6",
                        alpha_mean="0.784 ± 0.001 (3 seeds)", alpha_std="0.142 ± 0.002",
                        tail_pos_02="31.9 ± 0.2 %", tail_neg_02="38.1 ± 0.9 %", e91="0.83"),
    "liv_hmouse": dict(row="Liver, H+Mouse", N="≈14 k", G="≈10 k", B="2", shared_ct="5",
                       alpha_mean="0.762 ± 0.007 (3 seeds)", alpha_std="0.150 ± 0.021",
                       tail_pos_05="10.27 ± 4.59 %", tail_neg_05="30.62 ± 3.65 %", e91="0.762"),
    "liv_hzeb": dict(row="Liver, H+Zebrafish", N="≈10 k", G="≈6 k", B="2", shared_ct="4", alpha_mean="0.815",
                     alpha_std="0.096", tail_pos_02="23.2 %", tail_neg_02="2.4 %"),
    "liv_4sp": dict(row="Liver, four-species", N="22,104 (H 4,125 / Mac 8,483 / Mouse 3,966 / Zeb 5,530)",
                    G="8,879", B="5", shared_ct="2 (Hepatocytes 16,222; Endothelial 2,733)", alpha_mean="0.754",
                    alpha_std="0.134", tail_pos_05="7.7 %", tail_neg_05="35.8 %", tail_pos_02="18.1 %",
                    tail_neg_02="54.3 %"),
    "test_hmac": dict(row="Testis, H+Mac", N="23,593 (H 15,980 / Mac 7,613)", G="14,640", B="2", shared_ct="10",
                      alpha_mean="0.847", alpha_std="0.0499", tail_pos_02="29.4 %", tail_neg_02="1.6 %",
                      tail_pos_05="8.20 %", tail_neg_05="0.01 %"),
    "test_hmouse": dict(row="Testis, H+Mouse", N="21,241 (H 15,980 / Mouse 5,261)", G="14,731", B="2",
                        shared_ct="3 (38 % coverage)", shared_coc="5 (100 % coverage)", alpha_mean="0.847",
                        alpha_std="0.0495", tail_pos_02="26.97 %", tail_neg_02="0.18 %", tail_neg_05="0 %"),
    "test_3sp": dict(row="Testis, three-species", N="28,854 (H 15,980 / Mac 7,613 / Mouse 5,261)", G="15,732",
                     B="3", shared_ct="3", shared_coc="5 (100 % coverage)", alpha_mean="0.850", alpha_std="0.055",
                     tail_pos_05="18.68 %", tail_neg_05="1.57 %"),
}


def row(item, location, stated, measured, source, match=None, note=""):
    return dict(item=item, location=location, manuscript_value=stated, measured_value=measured,
                matches=match, source=source, note=note)


def shared_types(obs: pd.DataFrame, key: str) -> tuple[list[str], dict]:
    t = pd.crosstab(obs[key].astype(str), obs["species"].astype(str))
    shared = t.index[(t > 0).all(1)].tolist()
    cov = {s: float(t.loc[shared, s].sum() / t[s].sum()) for s in t.columns}
    return shared, cov, t


def recipe_rows(cohort: str) -> list[dict]:
    run = C.CANONICAL[cohort]
    st = C.load_state(run)
    cfg = st["config"]
    a = C.read_adata(cohort, backed=True)
    obs = a.obs.copy()
    G = a.n_vars
    a.file.close()
    s = STATED[cohort]
    loc = f"{RECIPES}, row '{s['row']}'"
    src_h5 = f"data/processed/{gene_ids.H5ADS[cohort]}.h5ad"
    rows = []
    per_sp = obs["species"].astype(str).value_counts().sort_index()
    rows.append(row(f"{cohort}: N cells", loc, s["N"],
                    f"{len(obs):,} ({' / '.join(f'{k} {v:,}' for k, v in per_sp.items())})", f"{src_h5} obs"))
    rows.append(row(f"{cohort}: G orthologs", loc, s["G"], f"{G:,}", f"{src_h5} var",
                    match=s["G"].replace(",", "") == str(G) if "≈" not in s["G"] else None))
    bkey = cfg.get("batch_key") or "batch"
    nb = obs[bkey].astype(str).nunique()
    rows.append(row(f"{cohort}: B batches (key '{bkey}')", loc, s["B"], str(nb),
                    f"{src_h5} obs['{bkey}']; training/runs/{run}/config.json batch_key",
                    match=str(nb) == s["B"], note=f"n_batch in final_state: {int(st['n_batch'])}"))
    for key, stated in (("cell_type", s["shared_ct"]), ("cell_ontology_class", s.get("shared_coc"))):
        if key == "cell_ontology_class" and not C.is_testis(cohort):
            continue
        sh, cov, t = shared_types(obs, key)
        counts = "; ".join(f"{k} {int(t.loc[k].sum()):,}" for k in sh)
        covs = ", ".join(f"{k} {v:.0%}" for k, v in cov.items())
        rows.append(row(f"{cohort}: shared types on {key}", loc, stated or "not stated",
                        f"{len(sh)} ({counts}); cells in shared types: {covs}", f"{src_h5} obs['{key}'] x species",
                        match=(stated or "").split(" ")[0] == str(len(sh)) if stated else None))
    da = st["alpha"] - st["prior"]
    meas = dict(alpha_mean=f"{st['alpha'].mean():.3f}", alpha_std=f"{st['alpha'].std():.4f}",
                tail_pos_02=f"{(da > 0.02).mean():.2%}", tail_neg_02=f"{(da < -0.02).mean():.2%}",
                tail_pos_05=f"{(da > 0.05).mean():.2%}", tail_neg_05=f"{(da < -0.05).mean():.2%}")
    for k, v in meas.items():
        if k in s:
            rows.append(row(f"{cohort}: {k} (s42)", loc, s[k], v, f"training/runs/{run}/final_state.npz"))
    if cohort in C.SEED_RUNS:
        sts = [C.load_state(r) for r in C.SEED_RUNS[cohort]]
        al = np.array([x["alpha"].mean() for x in sts])
        sd = np.array([x["alpha"].std() for x in sts])
        rows.append(row(f"{cohort}: alpha mean ± std over 3 seeds", loc, s["alpha_mean"],
                        f"{al.mean():.3f} ± {al.std(ddof=1):.3f}; std {sd.mean():.3f} ± {sd.std(ddof=1):.3f}",
                        "final_state.npz of " + ", ".join(C.SEED_RUNS[cohort]), note="± is the SD across seeds (ddof=1)"))
        for thr, name in ((0.02, "02"), (0.05, "05")):
            p = np.array([((x["alpha"] - x["prior"]) > thr).mean() for x in sts])
            n = np.array([((x["alpha"] - x["prior"]) < -thr).mean() for x in sts])
            stated = s.get(f"tail_pos_{name}", "")
            rows.append(row(f"{cohort}: Δα > +{thr} / < −{thr}, 3 seeds", loc,
                            f"{stated} / {s.get(f'tail_neg_{name}', '')}" if stated else "not stated",
                            f"{p.mean():.2%} ± {p.std(ddof=1):.2%} / {n.mean():.2%} ± {n.std(ddof=1):.2%}",
                            "final_state.npz of the three seeds"))
    return rows


def seed_rows() -> list[dict]:
    rows = []
    src = "training/cross_seed_check.py (E9.1 passes on min Spearman of alpha: rhos_alpha, min_rho_alpha)"
    stated = {"liv_hmonkey": "ρ_min 0.83 (recipe table); 'ρ ∈ [0.76, 0.83]' (:135)",
              "liv_hmouse": "ρ_min 0.762 (recipe table)"}
    for cohort, runs in C.SEED_RUNS.items():
        sts = [C.load_state(r) for r in runs]
        ra, rd, sg, se = [], [], [], []
        for i, j in itertools.combinations(range(3), 2):
            a, b = sts[i], sts[j]
            ra.append(spearmanr(a["alpha"], b["alpha"])[0])
            da, db = a["alpha"] - a["prior"], b["alpha"] - b["prior"]
            rd.append(spearmanr(da, db)[0])
            sg.append(float((np.sign(da) == np.sign(db)).mean()))
            se.append(spearmanr(a["S_g_ema"], b["S_g_ema"])[0])
        rows.append(row(f"{cohort}: cross-seed Spearman of alpha (3 pairs)", f"{ASSUME}:72,135,169; {RECIPES}",
                        stated[cohort], f"min {min(ra):.3f}, max {max(ra):.3f}", src))
        rows.append(row(f"{cohort}: cross-seed Spearman of Δα (3 pairs)", f"{ASSUME}:72",
                        "ρ(Δα^(ω1), Δα^(ω2)) ≥ 0.50 at the E9.1 gate", f"min {min(rd):.3f}, max {max(rd):.3f}",
                        "final_state.npz alpha - prior of the three seeds", match=min(rd) >= 0.5,
                        note="the gate code reads alpha, not Δα"))
        rows.append(row(f"{cohort}: cross-seed Δα sign agreement", f"{ASSUME}:72", "not stated",
                        f"min {min(sg):.3f}, max {max(sg):.3f}", "final_state.npz"))
        rows.append(row(f"{cohort}: cross-seed Spearman of S_g_ema", "-", "not stated",
                        f"min {min(se):.3f}, max {max(se):.3f}", "final_state.npz"))
    code = (ROOT / "training" / "cross_seed_check.py").read_text()
    reads_alpha = "min_rho_alpha >= E9_1_THRESHOLD" in code
    rows.append(row("E9.1 gate quantity", f"{ASSUME}:72,135,169", "cross-seed rank stability of Δα (:72)",
                    "alpha" if reads_alpha else "see code", "training/cross_seed_check.py: pass_revised = "
                    "min_rho_alpha >= E9_1_THRESHOLD", match=False if reads_alpha else None))
    m = re.search(r"E9_1_THRESHOLD\s*=\s*([0-9.]+)", code)
    rows.append(row("E9.1 threshold", f"{ASSUME}:72,135", "0.50", m.group(1) if m else "not found",
                    "training/cross_seed_check.py", match=(m.group(1) == "0.50" or m.group(1) == "0.5") if m else None))
    return rows


def clamp_rows() -> list[dict]:
    rows = []
    for tissue, cohorts in (("liver", ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp"]),
                            ("testis", ["test_hmac", "test_hmouse", "test_3sp"])):
        stated = "~4 % of liver genes" if tissue == "liver" else "~32 % of testis genes"
        vals, clamp = [], []
        for c in cohorts:
            run = C.CANONICAL[c]
            st = C.load_state(run)
            a = C.read_adata(c, backed=True)
            raw = C.raw_prior(c, run, a.var["conservation_prior"].to_numpy(float))
            a.file.close()
            vals.append(f"{c} {(raw >= 0.95 - 1e-6).mean():.1%}")
            if C.is_testis(c):
                clamp.append(f"{c} {(st['prior'] >= st['prior'].max() - 1e-6).mean():.1%} at {st['prior'].max():.2f}")
        rows.append(row(f"{tissue}: genes at the 0.95 data ceiling of the raw prior", f"{ASSUME}:56; {RECIPES} testis H+Mac",
                        stated, "; ".join(vals), "h5ad var['conservation_prior'] and conservation_prior_raw.npy"))
        if clamp:
            rows.append(row("testis: genes held at the training clamp (final_state prior)", f"{ASSUME}:56; main.tex:278",
                            "'π_g is held fixed by recipe' / ~32 % clamp binding", "; ".join(clamp),
                            "final_state.npz prior of the canonical testis runs",
                            note="the clamp the runs anchor on is 0.85, so the binding fraction is the 0.85 one"))
    return rows


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class _ShapeUnpickler(pickle.Unpickler):
    """Reads tensor shapes from a torch zip checkpoint without importing torch."""

    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, *a, **k: ("tensor", tuple(size))
        if module.startswith("torch") or module.startswith("collections"):
            if name == "OrderedDict":
                import collections
                return collections.OrderedDict
            return _Stub
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return None


def _shapes(obj, out):
    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "tensor":
        out.append(obj[1])
    elif isinstance(obj, dict):
        for v in obj.values():
            _shapes(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _shapes(v, out)


def esm_rows() -> list[dict]:
    rows = []
    d = ROOT / "data" / "raw" / "saturn_esm2" / "protein_embeddings_export" / "ESM2"
    dims = {}
    for f in sorted(d.glob("*_embedding.torch")) if d.exists() else []:
        try:
            with zipfile.ZipFile(f) as z:
                pk = next(n for n in z.namelist() if n.endswith("data.pkl"))
                obj = _ShapeUnpickler(z.open(pk)).load()
            shp = []
            _shapes(obj, shp)
            dims[f.name] = sorted({s[-1] for s in shp if len(s) >= 1})
        except Exception as exc:  # noqa: BLE001
            dims[f.name] = f"unreadable: {type(exc).__name__}"
    rows.append(row("ESM-2 embedding dimension (SATURN ESM2 files)", "extended_methods:139,1616; main.tex:209",
                    "5120 dimensions, esm2_t48_15B_UR50D (corrected in T9.2)",
                    "; ".join(f"{k}: {v}" for k, v in dims.items()) or "files not found",
                    "data/raw/saturn_esm2/protein_embeddings_export/ESM2/*.torch (tensor shapes read from data.pkl)",
                    note="esm2_t36_3B_UR50D has 2560 dimensions; esm2_t48_15B_UR50D has 5120"))
    src = (ROOT / "chimera" / "_data_utils.py").read_text().splitlines()
    for i, line in enumerate(src, 1):
        if "emb_dim = 5120" in line or ("ESM-2 3B" in line):
            rows.append(row("ESM-2 in package code", f"chimera/_data_utils.py:{i}", "-", line.strip(),
                            "package source (not modified)",
                            note="docstring still names 3B" if "3B" in line else ""))
    return rows


def mf_rows() -> list[dict]:
    rows = []
    rule = "M: π ≥ 0.80 and Δα ≤ −0.10; F: π ≤ 0.65 and Δα ≥ +0.10; then drop genes whose Δα sign differs across tissues"
    rows.append(row("M/F flag rule", "extended_methods:1523-1545 (alg:ortholog-qc)", rule,
                    "notebooks/_ortholog_qc_builder.py: PRIOR_HIGH 0.80, PRIOR_LOW 0.65, DELTA_FLOOR 0.10",
                    "notebooks/_ortholog_qc_builder.py:308-310"))
    tables = {}
    for c in C.COHORT_ORDER:
        g = C.load_cohort(c, covariates=False).genes
        tables[c] = g
        for prior_col in ("prior", "prior_raw"):
            da = g["alpha"] - g[prior_col]
            M = (g[prior_col] >= 0.80) & (da <= -0.10)
            F = (g[prior_col] <= 0.65) & (da >= 0.10)
            rows.append(row(f"{c}: raw M / F flag counts ({prior_col})", "extended_methods:1523", "-",
                            f"M {int(M.sum())}, F {int(F.sum())}", f"final_state.npz alpha, {prior_col}",
                            note="prior = clamped prior the run anchors on" if prior_col == "prior" else ""))
    # cross-tissue sign-consistency filter on the true join
    idx = {c: g.dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
           for c, g in tables.items()}
    sign = pd.DataFrame({c: np.sign(t["delta_alpha"]) for c, t in idx.items()})
    consistent = sign.apply(lambda r: r.dropna().nunique() <= 1, axis=1)
    tot_m, tot_f = set(), set()
    for c, t in idx.items():
        M = t.index[(t["prior"] >= 0.80) & (t["delta_alpha"] <= -0.10)]
        F = t.index[(t["prior"] <= 0.65) & (t["delta_alpha"] >= 0.10)]
        tot_m |= set(M[consistent.reindex(M).fillna(True).to_numpy()])
        tot_f |= set(F[consistent.reindex(F).fillna(True).to_numpy()])
    rows.append(row("M / F flags after the cross-tissue consistency filter (7 cohorts, union)",
                    "extended_methods:1531-1545", "-", f"M {len(tot_m)}, F {len(tot_f)}",
                    "final_state.npz, joined on hsapiens_gene_id"))
    # hand-labelled attribution candidates
    text = (ROOT / "notebooks" / "_gin_attribution_builder.py").read_text()
    cand = re.findall(r'\("([A-Za-z0-9.-]+)",\s+"(M|F|ref)",\s+"([a-z_0-9]+)"', text)
    dsmap = {"panc_hmonkey": "liv_hmonkey", "panc_hmouse": "liv_hmouse", "panc_hzeb": "liv_hzeb", "panc_4sp": "liv_4sp"}
    detail = []
    for sym, lab, ds in cand:
        c = dsmap.get(ds, "liv_hmonkey" if ds == "ref" else None)
        t = tables.get(c)
        hit = t[t["symbol"] == sym] if t is not None else pd.DataFrame()
        if hit.empty:
            detail.append((sym, lab, ds, np.nan, np.nan, "absent"))
            continue
        p, d = float(hit["prior"].iloc[0]), float(hit["delta_alpha"].iloc[0])
        rule_lab = "M" if (p >= 0.80 and d <= -0.10) else ("F" if (p <= 0.65 and d >= 0.10) else "none")
        detail.append((sym, lab, ds, p, d, rule_lab))
    det = pd.DataFrame(detail, columns=["symbol", "label", "dataset", "prior", "delta_alpha", "rule_label"])
    for lab in ("M", "F", "ref"):
        s = det[det.label == lab]
        ok = (s.rule_label == lab).sum() if lab != "ref" else (s.rule_label == "none").sum()
        rows.append(row(f"attribution candidates labelled {lab}: meet the QC rule", "extended_methods:1651; extended_results:44",
                        "classes are 'the M-flag, F-flag, and reference partition of the ortholog QC'",
                        f"{ok} of {len(s)} (in vocabulary {s.prior.notna().sum()})",
                        "notebooks/_gin_attribution_builder.py CANDIDATE_GENES vs final_state.npz of the named run",
                        match=bool(ok == len(s)), note="pre-HVG-filter candidate pool; the paper's 70 proteins are a subset"))
    for sym in ("ETV1", "RUNX1", "HPRT1"):
        s = det[det.symbol == sym]
        if len(s):
            r = s.iloc[0]
            rows.append(row(f"{sym}: label vs rule", "supplementary_figures:391,413", f"{r.label}-flag",
                            f"π {r.prior:.2f}, Δα {r.delta_alpha:+.2f} in {r.dataset}; rule gives {r.rule_label}",
                            "final_state.npz", match=r.label == r.rule_label))
    return rows, det


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap)
    args = ap.parse_args()
    out = TaskOutput("T2_7", args.outdir, prerun=args.prerun, force=args.force)
    rows = []
    for c in C.COHORT_ORDER:
        print("==", c, flush=True)
        rows += recipe_rows(c)
    rows += seed_rows()
    rows += clamp_rows()
    rows += esm_rows()
    mf, det = mf_rows()
    rows += mf
    df = pd.DataFrame(rows)
    out.csv(df, "consistency_values.csv")
    out.csv(det, "attribution_candidates_vs_rule.csv")
    pd.set_option("display.width", 250, "display.max_colwidth", 110, "display.max_rows", 300)
    print(df[["item", "manuscript_value", "measured_value", "matches"]].to_string())
    out.finish(dict(n_values=len(df)))


if __name__ == "__main__":
    main()
