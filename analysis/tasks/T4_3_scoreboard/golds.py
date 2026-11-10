"""Independent references ("golds") G of the validation, built from the files of analysis/tasks/T4_1_gold. No Chimera
readout is read.

Every G is oriented so that higher means more conserved (pre-specified analysis plan, section 4).

- H1, H2: G = -log EVEE ``var`` (liver, testis); symbols through the map of T4_1_gold, var = -1 is a
  missing-value sentinel and is dropped. EVEE ``mean`` is kept for the sensitivity of plan, section 4.
- H3: Tabula Sapiens liver (10x 3' v3 cells only) vs Tabula Muris Senis liver 10x, the
  K = 6 label map (``h3_label_map.tsv``, fixed with the plan), ages pooled. Per atlas and type, counts are pooled over all cells
  (the >= 30-cell rule of the plan is at atlas level), ``x = log2(1 + 1e6 C / sum C)`` with the
  library size over the human-mouse 1:1 liver vocabulary present in both atlases. A gene is
  scorable if it is detected in >= 1% of the cells of at least one type in both atlases (the
  1% detection rule of the direct baselines). G = Pearson across the K types of the atlas-centered profiles;
  NaN for a constant profile.
- H4: the 8 Murat held-out species, coarse types present in all 8 (Spermatogonia,
  Spermatocytes, spermatids, somatic), Ensembl 1:1 orthologs (map of T4_1_gold; 110, chicken 105).
  Per species: library pseudobulk with >= 30 cells, mean over libraries of log2(1 + CPM),
  library size over the species' mapped genes; detection over the cells of those libraries.
  A species counts as present for a gene if the gene is mapped, detected in >= 1% of the cells
  of at least one type and its profile is not constant. Pairwise Pearson of species-centered
  profiles, Fisher z; G = unweighted mean of the three tier means (primate-primate,
  primate-outgroup, outgroup-outgroup); a gene needs >= 3 primates and >= 2 outgroups.
  ``H4_nsp`` (species present) joins C_pat. Primates-only G (>= 3 primates) is the sensitivity.
- Split halves for the noise ceilings (plan, section 3): H3, random halves of donors in each atlas
  (TS has 4 donors, so donor halves, not cell halves); H4, library halves in each species;
  100 splits each, seeded. Each half goes through the same pipeline.
- H8: LOEUF (gnomAD v4.1, canonical transcript, ``lof.oe_ci.upper``) and dN/dS (Ensembl
  Compara 99 human-mouse 1:1, union of the human and mouse files, dn/ds with ds in (0, 3]).
  Oriented as -LOEUF and -dN/dS (higher = more constrained); H8 is two-sided.
- Sets (H5, H6): C1 Eisenberg housekeeping; C2 HGNC cytosolic ribosomal (728, 729), proteasome
  (690), spliceosome core (Sm 1255, LSm 1505, U1 1499, U2 1501, U4/U6 1502, U5 1503, tri-snRNP
  1506; groups with their subgroups); V1 Hallmark xenobiotic + bile acid (liver only); V2
  Hallmark IFN-alpha, IFN-gamma, inflammatory; V3 human GOA, aspect C, not NOT, any term in the
  is_a/part_of closure of GO:0005576, all evidence codes except ND.

Usage: python -m analysis.tasks.T4_3_scoreboard.golds [--outdir results/analysis/T4_3/golds]
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import gene_ids
from analysis.tasks.T4_1_gold import gold_checks as GC

REPO = gene_ids.REPO_ROOT
GOLD = REPO / "data" / "analysis" / "gold"
T41 = REPO / "results" / "analysis" / "T4_1"
# H3 cell-type label map; fixed with the pre-specified analysis plan, before any readout was compared with these data.
LABEL_MAP = Path(__file__).resolve().with_name("h3_label_map.tsv")
HALLMARK = Path.home() / ".cache" / "gseapy" / "Enrichr.MSigDB_Hallmark_2020.gmt"
MIN_CELLS = 30
DET_MIN = 0.01
N_SPLITS = 100
SPLIT_SEED = 20261101
PRIMATES = GC.PRIMATES
OUTGROUPS = GC.OUTGROUPS
H4_TYPES = ["Spermatogonia", "Spermatocytes", "spermatids", "somatic"]
C2_FAMILIES = {"ribosomal": ["728", "729"], "proteasome": ["690"],
               "spliceosome_core": ["1255", "1505", "1499", "1501", "1502", "1503", "1506"]}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- shared helpers
def profile_corr(A: np.ndarray, B: np.ndarray, okA: np.ndarray, okB: np.ndarray) -> np.ndarray:
    """Row-wise Pearson of centered profiles (genes x types); NaN where not ok or constant."""
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    na, nb = np.sqrt((A ** 2).sum(1)), np.sqrt((B ** 2).sum(1))
    ok = okA & okB & (na > 1e-12) & (nb > 1e-12)
    r = np.full(A.shape[0], np.nan)
    r[ok] = (A[ok] * B[ok]).sum(1) / (na[ok] * nb[ok])
    return np.clip(r, -1, 1)


def group_sums(X, keys: np.ndarray):
    """Count sums, expressing-cell counts and cell numbers per key (rows of X are cells)."""
    uk, inv = np.unique(keys, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(keys)), (inv, np.arange(len(keys)))), shape=(len(uk), len(keys)))
    Xb = X.copy()
    Xb.data = (Xb.data > 0).astype(np.float64)
    return uk, np.asarray((G @ X).todense()), np.asarray((G @ Xb).todense()), np.bincount(inv)


# ---------------------------------------------------------------- H1, H2, EVEE mean
def evee() -> pd.DataFrame:
    import io
    import zipfile

    z = zipfile.ZipFile(GOLD / "G1" / "evolutionary_statistics.zip")
    var = pd.read_csv(io.BytesIO(z.read("evolutionary_statistics/var.txt")), sep="\t")
    mean = pd.read_csv(io.BytesIO(z.read("evolutionary_statistics/mean.txt")), sep="\t")
    m = pd.read_csv(T41 / "evee_symbol_to_ensg.tsv", sep="\t")
    if not var.gene_name.equals(m.symbol.astype(str)):
        raise AssertionError("EVEE symbol order differs from the T4.1 map")
    out = pd.DataFrame({"hsapiens_gene_id": m.ensg})
    for t, h in (("liver", "H1"), ("testis", "H2")):
        v = var[t].to_numpy(dtype=float)
        g = np.where(v > 0, -np.log(np.where(v > 0, v, 1.0)), np.nan)
        out[f"G_{h}"] = g
        mn = mean[t].to_numpy(dtype=float)
        out[f"evee_mean_{t}"] = np.where(v > 0, mn, np.nan)
    out = out[out.hsapiens_gene_id.notna()]
    if out.hsapiens_gene_id.duplicated().any():
        raise AssertionError("EVEE: duplicated ENSG after mapping")
    return out.set_index("hsapiens_gene_id")


# ---------------------------------------------------------------- H3
def h3_inputs():
    """Per (atlas, type, donor) sums on the shared 1:1 vocabulary. Returns a dict."""
    import anndata as ad

    lm = pd.read_csv(LABEL_MAP, sep="\t")
    tab = gene_ids.load_table("liver")
    tab = tab[tab.hsapiens_gene_id.notna() & tab.mmusculus_gene_id.notna()]
    ts = ad.read_h5ad(GOLD / "G2" / "ts_liver.h5ad")
    tms = ad.read_h5ad(GOLD / "G3" / "tms_liver_10x.h5ad")
    ts_genes = pd.Index(ts.raw.var_names)
    tms_genes = pd.Index(tms.raw.var_names)
    keep = tab.hsapiens_gene_id.isin(ts_genes) & tab.mmusculus_gene_id.isin(tms_genes)
    tab = tab[keep]
    hs, mm = tab.hsapiens_gene_id.to_numpy(), tab.mmusculus_gene_id.to_numpy()
    out = dict(genes=hs, types=list(lm.coarse_type))
    for name, a, col, ids in (("TS", ts, "ts_cell_type", hs), ("TMS", tms, "tms_cell_type", mm)):
        obs = a.obs
        mk = np.ones(a.n_obs, bool)
        if name == "TS":
            mk = obs["assay"].astype(str).to_numpy() == "10x 3' v3"
        lab = obs["cell_type"].astype(str).to_numpy()
        tmap = dict(zip(lm[col], lm.coarse_type))
        ct = np.array([tmap.get(x, "") for x in lab])
        mk &= ct != ""
        X = a.raw.X[np.flatnonzero(mk)]
        X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        cols = pd.Index(a.raw.var_names).get_indexer(ids)
        X = X[:, cols].astype(np.float64)
        d = X.data
        if not (np.all(d >= 0) and np.all(d == np.round(d))):
            raise AssertionError(f"{name}: raw.X is not raw counts")
        donor = obs["donor_id"].astype(str).to_numpy()[mk]
        keys = np.array([f"{t}|{dd}" for t, dd in zip(ct[mk], donor)])
        uk, S, Nexp, Nc = group_sums(X, keys)
        out[name] = dict(keys=uk, S=S, Nexp=Nexp, Nc=Nc,
                         cells_per_type={t: int(Nc[[k.split("|")[0] == t for k in uk]].sum()) for t in out["types"]})
    return out


def h3_gold(inp: dict, donors: dict | None = None, profiles: bool = False):
    """G for H3 from the donor-level sums; ``donors`` restricts each atlas to a donor subset."""
    types = inp["types"]
    profs, oks, ncell = {}, {}, {}
    for name in ("TS", "TMS"):
        d = inp[name]
        t_of = np.array([k.split("|")[0] for k in d["keys"]])
        dn_of = np.array([k.split("|")[1] for k in d["keys"]])
        use = np.ones(len(t_of), bool) if donors is None else np.isin(dn_of, list(donors[name]))
        P, F, Ncs = [], [], []
        for t in types:
            m = use & (t_of == t)
            S = d["S"][m].sum(0)
            n = d["Nc"][m].sum()
            Ncs.append(int(n))
            P.append(np.log2(1 + 1e6 * S / max(S.sum(), 1)))
            F.append(d["Nexp"][m].sum(0) / max(n, 1))
        profs[name] = np.column_stack(P)
        oks[name] = np.column_stack(F).max(1) >= DET_MIN
        ncell[name] = Ncs
        if profiles:
            ncell[f"{name}_x"] = profs[name]
            ncell[f"{name}_f"] = np.column_stack(F)
    G = profile_corr(profs["TS"], profs["TMS"], oks["TS"], oks["TMS"])
    return G, ncell


def h3_splits(inp: dict, rng: np.random.Generator):
    """100 random donor halves in each atlas; each half through the same pipeline."""
    donors = {n: sorted({k.split("|")[1] for k in inp[n]["keys"]}) for n in ("TS", "TMS")}
    A, B, info = [], [], []
    for i in range(N_SPLITS):
        ha, hb = {}, {}
        for n, ds in donors.items():
            perm = rng.permutation(ds)
            ha[n], hb[n] = set(perm[: len(ds) // 2]), set(perm[len(ds) // 2:])
        ga, na = h3_gold(inp, ha)
        gb, nb = h3_gold(inp, hb)
        A.append(ga)
        B.append(gb)
        info.append(dict(split=i, ts_a=" ".join(sorted(ha["TS"])), tms_a=" ".join(sorted(ha["TMS"])),
                         min_cells_a=min(na["TS"] + na["TMS"]), min_cells_b=min(nb["TS"] + nb["TMS"])))
    return np.array(A, dtype=np.float32), np.array(B, dtype=np.float32), pd.DataFrame(info), donors


# ---------------------------------------------------------------- H4
def h4_inputs():
    import anndata as ad

    m = pd.read_csv(T41 / "h4_ortholog_map_one2one_union.tsv.gz", sep="\t", dtype=str)
    human = np.array(sorted(m.hsapiens_gene_id.unique()))
    hidx = pd.Index(human)
    out = dict(genes=human, species={})
    for spc in PRIMATES + OUTGROUPS:
        a = ad.read_h5ad(REPO / f"data/raw/camex/2testis/dataset/raw-testis-{spc}-Murat.h5ad")
        ids = a.var["gene"].astype(str).str.replace(r"\.\d+$", "", regex=True).to_numpy()
        mm = m[m.murat_species == spc]
        s2h = dict(zip(mm.species_gene_id, mm.hsapiens_gene_id))
        cols = np.array([i for i, g in enumerate(ids) if g in s2h])
        rows_h = hidx.get_indexer([s2h[ids[i]] for i in cols])
        X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
        X = X[:, cols].astype(np.float64)
        d = X.data
        if not (np.all(d >= 0) and np.all(d == np.round(d))):
            raise AssertionError(f"{spc}: X is not raw counts")
        ct = a.obs["cell_ontology_class"].astype(str).to_numpy()
        lib = a.obs["sample_batch"].astype(str).to_numpy()
        mk = np.isin(ct, H4_TYPES)
        keys = np.array([f"{t}|{l}" for t, l in zip(ct[mk], lib[mk])])
        uk, S, Nexp, Nc = group_sums(X[np.flatnonzero(mk)], keys)
        out["species"][spc] = dict(keys=uk, S=S, Nexp=Nexp, Nc=Nc, rows=rows_h,
                                   libraries=sorted(set(lib)))
    return out


def h4_species_profiles(inp: dict, libs: dict | None = None):
    """Per species: profile (genes x 4 types, human rows) and the 'present' mask."""
    n = len(inp["genes"])
    res = {}
    for spc, d in inp["species"].items():
        t_of = np.array([k.split("|")[0] for k in d["keys"]])
        l_of = np.array([k.split("|")[1] for k in d["keys"]])
        use = np.ones(len(t_of), bool) if libs is None else np.isin(l_of, list(libs[spc]))
        P = np.zeros((d["S"].shape[1], len(H4_TYPES)))
        F = np.zeros_like(P)
        okt = True
        for j, t in enumerate(H4_TYPES):
            m = use & (t_of == t) & (d["Nc"] >= MIN_CELLS)
            if not m.any():
                okt = False
                continue
            S = d["S"][m]
            P[:, j] = np.mean(np.log2(1 + 1e6 * S / S.sum(1, keepdims=True)), 0)
            F[:, j] = d["Nexp"][m].sum(0) / d["Nc"][m].sum()
        prof = np.full((n, len(H4_TYPES)), np.nan)
        prof[d["rows"]] = P
        present = np.zeros(n, bool)
        if okt:
            Pc = P - P.mean(1, keepdims=True)
            pres = (F.max(1) >= DET_MIN) & (np.sqrt((Pc ** 2).sum(1)) > 1e-12)
            present[d["rows"]] = pres
        res[spc] = (prof, present)
    return res


def h4_gold(inp: dict, libs: dict | None = None) -> pd.DataFrame:
    res = h4_species_profiles(inp, libs)
    n = len(inp["genes"])
    tiers = {"PP": [], "PO": [], "OO": []}
    for s1, s2 in itertools.combinations(PRIMATES + OUTGROUPS, 2):
        p1, ok1 = res[s1]
        p2, ok2 = res[s2]
        ok = ok1 & ok2
        r = np.full(n, np.nan)
        if ok.any():
            r[ok] = profile_corr(p1[ok], p2[ok], np.ones(ok.sum(), bool), np.ones(ok.sum(), bool))
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        tier = ("PP" if s1 in PRIMATES and s2 in PRIMATES else
                "OO" if s1 in OUTGROUPS and s2 in OUTGROUPS else "PO")
        tiers[tier].append(z)
    npri = np.sum([res[s][1] for s in PRIMATES], 0)
    nout = np.sum([res[s][1] for s in OUTGROUPS], 0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tm = {k: np.nanmean(np.column_stack(v), 1) for k, v in tiers.items()}
    G = (tm["PP"] + tm["PO"] + tm["OO"]) / 3
    G[~((npri >= 3) & (nout >= 2))] = np.nan
    Gp = tm["PP"].copy()
    Gp[~(npri >= 3)] = np.nan
    return pd.DataFrame({"G_H4": G, "H4_nsp": (npri + nout).astype(float), "H4_npri": npri, "H4_nout": nout,
                         "G_H4_primates": Gp}, index=pd.Index(inp["genes"], name="hsapiens_gene_id"))


def h4_splits(inp: dict, rng: np.random.Generator):
    A, B, info = [], [], []
    for i in range(N_SPLITS):
        la, lb = {}, {}
        for spc, d in inp["species"].items():
            libs = rng.permutation(d["libraries"])
            k = len(libs) // 2 if rng.random() < 0.5 or len(libs) % 2 == 0 else (len(libs) + 1) // 2
            la[spc], lb[spc] = set(libs[:k]), set(libs[k:])
        A.append(h4_gold(inp, la)["G_H4"].to_numpy())
        B.append(h4_gold(inp, lb)["G_H4"].to_numpy())
        info.append(dict(split=i, **{f"{s}_a": " ".join(sorted(la[s])) for s in la}))
    return np.array(A, dtype=np.float32), np.array(B, dtype=np.float32), pd.DataFrame(info)


# ---------------------------------------------------------------- H8
def h8() -> pd.DataFrame:
    g = pd.read_csv(GOLD / "G7" / "gnomad.v4.1.constraint_metrics.tsv", sep="\t", dtype=str,
                    usecols=["gene_id", "canonical", "lof.oe_ci.upper"])
    g = g[g.gene_id.str.startswith("ENSG", na=False) & (g.canonical == "true")]
    l = pd.to_numeric(g["lof.oe_ci.upper"], errors="coerce")
    lo = pd.Series(l.to_numpy(), index=g.gene_id).dropna()
    if lo.index.duplicated().any():
        raise AssertionError("LOEUF: duplicated canonical genes")
    d = pd.read_csv(T41 / "dnds99_hmouse_union.tsv", sep="\t")
    d = d[(d.ds > 0) & (d.ds <= 3) & d.dn.notna()]
    if d.hsapiens_gene_id.duplicated().any():
        raise AssertionError("dN/dS: duplicated human genes")
    dn = pd.Series((d.dn / d.ds).to_numpy(), index=d.hsapiens_gene_id)
    out = pd.DataFrame({"G_LOEUF": -lo}).join(pd.DataFrame({"G_dNdS": -dn}), how="outer")
    out["LOEUF"] = -out["G_LOEUF"]
    out["dNdS"] = -out["G_dNdS"]
    out.index.name = "hsapiens_gene_id"
    return out


# ---------------------------------------------------------------- sets
def sets() -> tuple[pd.DataFrame, dict]:
    names, _ = GC.gtf_gene_names(REPO / "data/analysis/ensembl110/Homo_sapiens.GRCh38.110.gtf.gz")
    mapper = GC.SymbolMapper(GOLD / "G10" / "hgnc_complete_set.txt", names)
    S, info = {}, {}
    hk = pd.read_csv(T41 / "hk_symbol_to_ensg.tsv", sep="\t")
    S["C1"] = set(hk.ensg.dropna())
    fam = pd.read_csv(GOLD / "G10/hgnc_family.csv", dtype=str)
    hier = pd.read_csv(GOLD / "G10/hgnc_hierarchy.csv", dtype=str)
    ghf = pd.read_csv(GOLD / "G10/hgnc_gene_has_family.csv", dtype=str)
    hg = pd.read_csv(GOLD / "G10/hgnc_complete_set.txt", sep="\t", dtype=str, usecols=["hgnc_id", "ensembl_gene_id"])
    hid2ensg = dict(zip(hg.hgnc_id.str.replace("HGNC:", "", regex=False), hg.ensembl_gene_id))
    kids = {}
    for p, c in zip(hier.parent_fam_id, hier.child_fam_id):
        kids.setdefault(p, set()).add(c)

    def family_genes(fid):
        ids, stack = {fid}, [fid]
        while stack:
            for c in kids.get(stack.pop(), ()):
                if c not in ids:
                    ids.add(c)
                    stack.append(c)
        h = set(ghf.hgnc_id[ghf.family_id.isin(ids)])
        return {hid2ensg[x] for x in h if isinstance(hid2ensg.get(x), str)}

    c2 = set()
    for k, fids in C2_FAMILIES.items():
        gs = set().union(*[family_genes(f) for f in fids])
        info[f"C2_{k}"] = len(gs)
        c2 |= gs
    S["C2"] = c2
    fname = dict(zip(fam.id, fam.name))
    info["C2_families"] = {f: fname.get(f) for v in C2_FAMILIES.values() for f in v}
    hm = GC.read_gmt(HALLMARK)
    for lab, terms in (("V1", ["Xenobiotic Metabolism", "Bile Acid Metabolism"]),
                       ("V2", ["Interferon Alpha Response", "Interferon Gamma Response", "Inflammatory Response"])):
        miss = [t for t in terms if t not in hm]
        if miss:
            raise KeyError(f"Hallmark terms missing: {miss}")
        syms = sorted(set().union(*[hm[t] for t in terms]))
        S[lab] = set(mapper.map(syms).ensg.dropna())
    desc = GC.go_descendants(GOLD / "G10/go-basic.obo", "GO:0005576")
    v3 = set()
    with gzip.open(GOLD / "G10/goa_human.gaf.gz", "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            p = line.split("\t")
            if p[8] != "C" or "NOT" in p[3] or p[6] == "ND":
                continue
            if p[4] in desc:
                v3.add(p[2])
    S["V3"] = set(mapper.map(sorted(v3)).ensg.dropna())
    info.update({k: len(v) for k, v in S.items()})
    allg = sorted(set().union(*S.values()))
    df = pd.DataFrame({k: [g in v for g in allg] for k, v in S.items()}, index=pd.Index(allg, name="hsapiens_gene_id"))
    return df, info


# ---------------------------------------------------------------- chromosomes
def chromosomes() -> pd.Series:
    import re

    rx = re.compile(r'gene_id "([^"]+)"')
    d = {}
    with gzip.open(REPO / "data/analysis/ensembl110/Homo_sapiens.GRCh38.110.gtf.gz", "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 9)
            if p[2] == "gene":
                d[rx.search(p[8]).group(1)] = p[0]
    return pd.Series(d, name="chromosome")


# ---------------------------------------------------------------- main
def build(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SPLIT_SEED)
    meta = {}
    log("EVEE")
    ev = evee()
    log("H8")
    h8t = h8()
    log("sets")
    st, sinfo = sets()
    meta["sets"] = sinfo
    log("H3 inputs")
    inp3 = h3_inputs()
    G3, nc3 = h3_gold(inp3, profiles=True)
    prof3 = {k: nc3.pop(k) for k in ("TS_x", "TS_f", "TMS_x", "TMS_f")}
    np.savez_compressed(outdir / "h3_profiles.tmp.npz", genes=inp3["genes"], types=np.array(inp3["types"]), **prof3)
    (outdir / "h3_profiles.tmp.npz").replace(outdir / "h3_profiles.npz")
    meta["H3"] = dict(K=len(inp3["types"]), types=inp3["types"], cells=nc3,
                      cells_per_type={n: inp3[n]["cells_per_type"] for n in ("TS", "TMS")},
                      K_ge30_both=int(sum(min(a, b) >= MIN_CELLS for a, b in zip(nc3["TS"], nc3["TMS"]))),
                      n_vocabulary=len(inp3["genes"]), n_defined=int(np.isfinite(G3).sum()))
    log("H3 splits")
    A3, B3, i3, donors = h3_splits(inp3, rng)
    meta["H3"]["donors"] = donors
    log("H4 inputs")
    inp4 = h4_inputs()
    g4 = h4_gold(inp4)
    meta["H4"] = dict(types=H4_TYPES, n_human_genes=len(inp4["genes"]), n_defined=int(g4.G_H4.notna().sum()),
                      n_primates_only=int(g4.G_H4_primates.notna().sum()),
                      libraries={s: d["libraries"] for s, d in inp4["species"].items()})
    log("H4 splits")
    A4, B4, i4 = h4_splits(inp4, rng)
    chrom = chromosomes()
    g3 = pd.DataFrame({"G_H3": G3}, index=pd.Index(inp3["genes"], name="hsapiens_gene_id"))
    tab = ev.join(g3, how="outer").join(g4, how="outer").join(h8t, how="outer").join(st, how="outer")
    for c in st.columns:
        tab[c] = tab[c].eq(True)
    tab["chromosome"] = chrom.reindex(tab.index).to_numpy()
    tab.index.name = "hsapiens_gene_id"
    for k, v in (("gold_table.csv", tab),):
        tmp = outdir / (k + ".tmp")
        v.to_csv(tmp)
        tmp.replace(outdir / k)
    np.savez_compressed(outdir / "splits.tmp.npz", h3_genes=inp3["genes"], h3_a=A3, h3_b=B3,
                        h4_genes=inp4["genes"], h4_a=A4, h4_b=B4)
    (outdir / "splits.tmp.npz").replace(outdir / "splits.npz")
    for name, df in (("h3_splits.csv", i3), ("h4_splits.csv", i4)):
        df.to_csv(outdir / (name + ".tmp"), index=False)
        (outdir / (name + ".tmp")).replace(outdir / name)
    meta["n_defined"] = {c: int(tab[c].notna().sum()) for c in tab.columns if c.startswith("G_")}
    (outdir / "gold_meta.json.tmp").write_text(json.dumps(meta, indent=2, default=str))
    (outdir / "gold_meta.json.tmp").replace(outdir / "gold_meta.json")
    log(json.dumps(meta, indent=1, default=str)[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=REPO / "results/analysis/T4_3/golds")
    build(ap.parse_args().outdir)


if __name__ == "__main__":
    main()
