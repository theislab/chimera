"""Annotation table, per-seed forward margins and gene tables on healthy seeds. No association is computed here.

    python -m analysis.tasks.Y5_aspects.build annot
    python -m analysis.tasks.Y5_aspects.build forward --threads 10
    python -m analysis.tasks.Y5_aspects.build genes
"""
from __future__ import annotations

import argparse
import gzip
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from analysis.common import cohorts as C
from analysis.common import gene_ids
from analysis.tasks.Y5_aspects import rules as RU

REPO = gene_ids.REPO_ROOT
OUT = REPO / "results" / "analysis" / "Y5"
FWD = OUT / "forward"
GEN = OUT / "genes"
ANN = REPO / "data" / "analysis" / "y5_annotations"
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8,
          P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
DEEP = ["gallus_gallus", "anolis_carolinensis", "xenopus_tropicalis", "danio_rerio", "latimeria_chalumnae",
        "lepisosteus_oculatus", "petromyzon_marinus"]
IDSP = {"opossum": "monodelphis_domestica", "platypus": "ornithorhynchus_anatinus", "chicken": "gallus_gallus",
        "mouse": "mus_musculus", "zebrafish": "danio_rerio"}


def healthy(cohort: str, rawtail: bool = False) -> dict[int, str]:
    h = pd.read_csv(OUT / "seed_health.csv")
    h = h[h.cohort == cohort]
    h = h[h.pass_GV_rawtail if rawtail else h.pass_GV]
    return {int(r.seed): r.run for r in h.itertuples()}


def seed_policy(cohort: str) -> dict[int, str]:
    """Literal gate; testis pairs on the raw-prior tail (labeled sensitivity); descriptive cohorts: all finished seeds."""
    if cohort in RU.PRIMARY:
        return healthy(cohort)
    if cohort in RU.SENS_RAWTAIL or cohort == "test_hmac":
        return healthy(cohort, rawtail=True)
    h = pd.read_csv(OUT / "seed_health.csv")
    return {int(r.seed): r.run for r in h[h.cohort == cohort].itertuples()}


def _span(field: str, want: str = "") -> list[tuple[int, int]]:
    out = []
    if not isinstance(field, str):
        return out
    for m in re.finditer(r"([A-Z_]+) <?(\d+)\.\.>?(\d+)(;[^;]*)?(; /note=\"([^\"]*)\")?", field):
        if want and want not in (m.group(0) + (m.group(6) or "")):
            continue
        out.append((int(m.group(2)), int(m.group(3))))
    return out


def _regions(field: str, want: str) -> list[tuple[int, int]]:
    out = []
    if not isinstance(field, str):
        return out
    for item in field.split("REGION ")[1:]:
        m = re.match(r"<?(\d+)\.\.>?(\d+)", item)
        if m and want in item:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


def _cover(spans, L):
    if not spans or not L:
        return 0.0
    v = np.zeros(L + 1, bool)
    for a, b in spans:
        v[a:b + 1] = True
    return float(v.sum() / L)


def annot():
    u = pd.read_csv(ANN / "uniprot_human_features.tsv.gz", sep="\t", dtype=str)
    u["Length"] = u["Length"].astype(int)
    hg = pd.read_csv(REPO / "data/analysis/x3_annotations/hgnc_complete_set.txt", sep="\t", dtype=str,
                     usecols=["ensembl_gene_id", "uniprot_ids", "symbol"])
    hg = hg.dropna(subset=["ensembl_gene_id", "uniprot_ids"])
    e = hg.assign(acc=hg.uniprot_ids.str.split("|")).explode("acc")[["ensembl_gene_id", "acc"]]
    u = u.merge(e, left_on="Entry", right_on="acc").sort_values("Length", ascending=False).drop_duplicates("ensembl_gene_id")
    t = pd.DataFrame({"hsapiens_gene_id": u.ensembl_gene_id.values, "uniprot": u.Entry.values})
    L = u["Length"].to_numpy()
    t["log_length"] = np.log(L)
    t["signal_peptide"] = u["Signal peptide"].notna().astype(float).values
    t["transmembrane"] = u["Transmembrane"].notna().astype(float).values
    t["disorder_frac"] = [_cover(_regions(f, "Disordered"), n) for f, n in zip(u["Region"], L)]
    t["domain_dens"] = [100 * f.count("DOMAIN ") / n if isinstance(f, str) else 0.0 for f, n in zip(u["Domain [FT]"], L)]
    t["compbias_frac"] = [_cover(_span(f), n) for f, n in zip(u["Compositional bias"], L)]
    t["coiled_frac"] = [_cover(_span(f), n) for f, n in zip(u["Coiled coil"], L)]
    t["ptm_dens"] = [100 * f.count("MOD_RES ") / n if isinstance(f, str) else 0.0 for f, n in zip(u["Modified residue"], L)]
    t["active_site"] = u["Active site"].notna().astype(float).values
    t["dna_zn_binding"] = (u["DNA binding"].notna() | u["Zinc finger"].notna()).astype(float).values
    loc = u["Subcellular location [CC]"].fillna("")
    known = loc.str.len() > 0
    for name, pat in (("loc_nucleus", "Nucleus"), ("loc_cytoplasm", "Cytoplasm"), ("loc_mitochondrion", "Mitochondrion"),
                      ("loc_secreted", "Secreted"), ("loc_plasma_membrane", "Cell membrane"),
                      ("loc_er_golgi", "Endoplasmic reticulum|Golgi")):
        v = loc.str.contains(pat).astype(float).where(known)
        t[name] = v.values
    seqs = u["Sequence"].astype(str)
    t["gravy"] = [np.mean([KD.get(a, 0.0) for a in s]) for s in seqs]
    t["ncpr"] = [(s.count("K") + s.count("R") - s.count("D") - s.count("E")) / len(s) for s in seqs]
    # Compara: paralogs, deep orthologs, identities
    keep = "\\|".join(["homo_sapiens"] + DEEP)
    cmd = ("zcat data/analysis/ensembl110/Compara.110.protein_default.homologies.tsv.gz | "
           "awk -F'\\t' '$3==\"homo_sapiens\" && ($8==\"homo_sapiens\" || $8==\"mus_musculus\" || "
           "$8==\"monodelphis_domestica\" || $8==\"ornithorhynchus_anatinus\" || "
           + " || ".join(f'$8==\"{d}\"' for d in DEEP) + ") {print $1,$4,$5,$6,$8,$9}' OFS='\\t' > "
           + str(OUT / "compara_human.tsv"))
    subprocess.run(cmd, shell=True, check=True, cwd=REPO)
    co = pd.read_csv(OUT / "compara_human.tsv", sep="\t", names=["g", "id", "typ", "h", "sp", "hid"])
    par = co[(co.sp == "homo_sapiens") & co.typ.str.contains("paralog")]
    t["paralog_count"] = np.log1p(t.hsapiens_gene_id.map(par.groupby("g").h.nunique()).fillna(0)).values
    par[par.id >= 40][["g", "h"]].drop_duplicates().to_parquet(OUT / "paralog_pairs_id40.parquet")
    orth = co[(co.sp != "homo_sapiens") & co.typ.str.startswith("ortholog")]
    br = orth[orth.sp.isin(DEEP)].groupby("g").sp.nunique()
    t["ortholog_breadth"] = t.hsapiens_gene_id.map(br).fillna(0).values
    for nm, sp in IDSP.items():
        o = orth[orth.sp == sp].groupby("g")["id"].max()
        t[f"identity_{nm}"] = t.hsapiens_gene_id.map(o).values
    from analysis.tasks.T6_common import functional as FN
    ed = FN.string_edges("9606", RU.STRING_THR)
    deg = pd.concat([ed.a, ed.b]).value_counts()
    t["ppi_degree"] = np.log1p(t.hsapiens_gene_id.map(deg).fillna(0)).values
    # chromosome
    chrom = {}
    with gzip.open(REPO / "data/analysis/ensembl110/Homo_sapiens.GRCh38.110.gtf.gz", "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.split("\t", 3)
            if p[2] == "gene":
                g = re.search(r'gene_id "([^"]+)"', line).group(1)
                chrom[g] = p[0]
    pd.Series(chrom, name="chrom").rename_axis("hsapiens_gene_id").reset_index().to_parquet(OUT / "chromosomes.parquet")
    t["chrom"] = t.hsapiens_gene_id.map(chrom)
    t.to_parquet(OUT / "annot.parquet")
    print(t.shape, t.drop(columns=["hsapiens_gene_id", "uniprot", "chrom"]).describe().T[["count", "mean", "min", "max"]].round(3).to_string())


def forward(threads: int):
    sys.path.insert(0, str(REPO / "analysis" / "tasks" / "T2_5_celltype_margin"))
    import run as T25RUN
    FWD.mkdir(parents=True, exist_ok=True)
    runs = {}
    for c in RU.PRIMARY:
        runs.update({f"{c}_s{s}": (c, r) for s, r in seed_policy(c).items()})
    runs.update({f"test_hmouse_s{s}": ("test_hmouse", seed_policy("test_hmouse")[s]) for s in (42, 43, 45)})
    for lab, (c, r) in runs.items():
        f = FWD / f"{lab}.npz"
        if f.exists():
            continue
        fw = T25RUN.forward_sums(lab, c, r, threads, 512)
        fw["run"] = np.array(r)
        tmp = FWD / f"{lab}.tmp.npz"
        np.savez_compressed(tmp, **fw)
        tmp.replace(f)


def gin_responses(run: str) -> dict:
    """c_full, c_esm (expr3 at mean), c_expr (esm64 + esm_var at mean); alpha check against final_state.npz."""
    import torch
    from chimera._components import GeneIntegrationNetwork
    d = torch.load(C.RUNS_DIR / run / "model" / "model.pt", map_location="cpu", weights_only=False)
    sd = d["model_state_dict"]
    gs = {k[len("gin."):]: v for k, v in sd.items() if k.startswith("gin.")}
    F = sd["gene_features"].float()
    prior = sd["conservation_prior"].float()
    w = [v for k, v in gs.items() if k.endswith("weight")]
    n_h = w[1].shape[0] if len(w) > 1 else 32
    n_layers = len(w) - 1
    cfg = C.load_state(run)["config"]
    act = cfg.get("gin_hidden_activation", cfg.get("hidden_activation", "tanh"))
    net = GeneIntegrationNetwork(n_features=F.shape[1], n_hidden=w[0].shape[0] // 2, hidden_activation=str(act),
                                 n_hidden_layers=n_layers)
    net.load_state_dict({k: v for k, v in gs.items() if k != "per_gene_bias"}, strict=True)
    net.eval()
    fs = C.load_state(run)
    with torch.no_grad():
        a, c = net(F, prior)
        if np.abs(a.numpy() - fs["alpha"]).max() > 1e-4:
            raise AssertionError(f"{run}: GIN alpha differs from final_state")
        Fe = F.clone()
        Fe[:, C.GIN_COLUMNS["expr3"]] = 0
        ce = net.correction_mlp(Fe).squeeze(-1)
        Fx = F.clone()
        Fx[:, C.GIN_COLUMNS["esm64"] + C.GIN_COLUMNS["esm_var"]] = 0
        cx = net.correction_mlp(Fx).squeeze(-1)
    return dict(c_full=c.numpy().astype(float), c_esm=ce.numpy().astype(float), c_expr=cx.numpy().astype(float))


def genes():
    GEN.mkdir(parents=True, exist_ok=True)
    ann = pd.read_parquet(OUT / "annot.parquet")
    chrom = pd.read_parquet(OUT / "chromosomes.parquet").set_index("hsapiens_gene_id").chrom
    pp = pd.read_parquet(OUT / "paralog_pairs_id40.parquet")
    pp["cg"], pp["ch"] = pp.g.map(chrom), pp.h.map(chrom)
    odd_par = set(pp[pp.ch.isin(RU.ODD)].g)        # gene has a >= 40 % paralog on an odd chromosome
    for c in list(C.CANONICAL):
        x3 = pd.read_parquet(REPO / f"results/analysis/X3/genes/{c}.parquet")
        sd = pd.read_parquet(REPO / f"results/analysis/X3/seeds/{c}.parquet")
        seeds = seed_policy(c)
        dr = sd[[f"D_raw_s{s}" for s in seeds if f"D_raw_s{s}" in sd]].mean(1)
        ss = sd[[f"S_s{s}" for s in seeds if f"S_s{s}" in sd]]
        Sm = ss.mean(1)
        g = pd.DataFrame({"var_name": x3.var_name, "hsapiens_gene_id": x3.hsapiens_gene_id, "symbol": x3.symbol,
                          "prior_raw": x3.prior_raw, "log_cp10k_spmean": x3.log_cp10k_spmean,
                          "detection_spmean": x3.detection_spmean, "log_cds_length": x3.log_cds_length,
                          "tau": x3.tau})
        g = g.merge(pd.DataFrame({"var_name": sd.var_name, "D_raw": dr - np.nanmedian(dr), "S": Sm,
                                  "masked": (ss == 0).all(1)}), on="var_name", how="left")
        resp = {k: [] for k in ("c_full", "c_esm", "c_expr")}
        for s, r in seeds.items():
            q = gin_responses(r)
            for k in resp:
                resp[k].append(q[k])
        for k, v in resp.items():
            g[k] = np.mean(v, 0)
            g[k + "_rank_sd"] = np.nan
        bl = pd.read_csv(REPO / f"results/analysis/T4_2/baselines__{c}.csv")[["var_name", "B1b", "B1b_raw", "B1c", "B1c_raw", "B1z", "B1z_raw", "B2b", "B3a"]]
        g = g.merge(bl, on="var_name", how="left")
        fx = REPO / f"results/analysis/X4/genes/{c}.parquet"
        if fx.exists():
            xt = pd.read_parquet(fx)
            keep = ["hsapiens_gene_id", "tau_shared", "block", "G_H1", "G_H2", "G_H3", "G_H4", "H4_nsp", "G_H4_primates", "nameable", "dNdS", "LOEUF"] + [k for k in xt.columns if k.startswith("cls_") or k.startswith("C_") or k.startswith("Csp_")] + [k for k in xt.columns if k in ("Sdir", "gain", "cost", "pattern_gain", "level_cost")]
            g = g.merge(xt[keep], on="hsapiens_gene_id", how="left")
            sd2 = pd.read_parquet(REPO / f"results/analysis/X4/cache/Sdir__{c}.parquet")
            g = g.merge(sd2[[k for k in sd2.columns if k in ("hsapiens_gene_id", "var_name", "Sdir", "Sdir_pattern", "Sdir_level")]].drop(columns=["hsapiens_gene_id"], errors="ignore"), on="var_name", how="left") if "var_name" in sd2 else g.merge(sd2[[k for k in sd2.columns if k in ("hsapiens_gene_id", "Sdir", "Sdir_pattern", "Sdir_level")]], on="hsapiens_gene_id", how="left")
        g = g.merge(ann.drop(columns=["chrom"]), on="hsapiens_gene_id", how="left")
        g["chrom"] = g.hsapiens_gene_id.map(chrom)
        g["half"] = np.where(g.chrom.isin(RU.ODD), "odd", np.where(g.chrom.isin(RU.EVEN), "even", "other"))
        g["has_odd_paralog"] = g.hsapiens_gene_id.isin(odd_par)
        g["cohort"] = c
        g.to_parquet(GEN / f"{c}.parquet")
        print(c, g.shape, g.half.value_counts().to_dict(), "seeds", list(seeds), "masked", round(float(g.masked.mean()), 3), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what")
    p.add_argument("--threads", type=int, default=10)
    a = p.parse_args()
    dict(annot=annot, genes=genes, forward=lambda: forward(a.threads))[a.what]()
