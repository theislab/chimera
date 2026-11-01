"""T4.1 checks on the gold standards: identifiers, scales and coverage only.

No Chimera readout is read here, and no gold value is related to any readout.
Every number this script writes is a count, a scale summary or an ID overlap.

Usage:
    python -I gold_checks.py <chimera_repo> <out_dir> [--part all|dnds|evee|sets|h3|h4]

Parts:
    dnds  Ensembl Compara 102 (and 110): are dn, ds populated for human-mouse 1:1?
    evee  EVEE var.txt scale and symbol-to-ENSG mapping, ambiguous symbols dropped.
    sets  LOEUF, housekeeping, HGNC groups, Hallmark, GO extracellular region coverage.
    h3    Tabula Sapiens (10x 3' v3 only) and TMS liver counts under the committed map.
    h4    Ensembl 1:1 ortholog map for the 8 Murat held-out species (110; chicken 105); coarse types.
"""

from __future__ import annotations

import argparse
import gzip
import io
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PRIMATES = ["bonobo", "chimpanzee", "gorilla", "gibbon", "marmoset"]
OUTGROUPS = ["opossum", "platypus", "chicken"]
# Ensembl species whose gene IDs the Murat var 'gene' column carries. Bonobo reads were
# mapped to the chimpanzee genome (var IDs are ENSPTRG), so bonobo uses pan_troglodytes.
ENSEMBL_SPECIES = {
    "bonobo": "pan_troglodytes",
    "chimpanzee": "pan_troglodytes",
    "gorilla": "gorilla_gorilla",
    "gibbon": "nomascus_leucogenys",
    "marmoset": "callithrix_jacchus",
    "opossum": "monodelphis_domestica",
    "platypus": "ornithorhynchus_anatinus",
    "chicken": "gallus_gallus",
}
ID_PREFIX = {
    "pan_troglodytes": "ENSPTRG",
    "gorilla_gorilla": "ENSGGOG",
    "nomascus_leucogenys": "ENSNLEG",
    "callithrix_jacchus": "ENSCJAG",
    "monodelphis_domestica": "ENSMODG",
    "ornithorhynchus_anatinus": "ENSOANG",
    "gallus_gallus": "ENSGALG",
}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- vocabularies
def vocabularies(repo: Path) -> dict[str, set[str]]:
    """Human ENSG sets of the ortholog tables and of the two primary cohorts' var."""
    import h5py

    sys.path.insert(0, str(repo))
    from revision.common import gene_ids  # noqa: E402

    P = repo / "data" / "processed"
    out = {
        "liver_table": set(pd.read_csv(P / "ortholog_table.csv").hsapiens_gene_id.dropna()),
        "testis_table": set(pd.read_csv(P / "ortholog_table_testis.csv").hsapiens_gene_id.dropna()),
    }
    for cohort in ("liv_hmouse", "test_hmouse"):
        with h5py.File(P / f"{gene_ids.H5ADS[cohort]}.h5ad", "r") as f:
            key = f["var"].attrs["_index"]
            v = [x.decode() if isinstance(x, bytes) else x for x in f["var"][key][()]]
        out[cohort] = set(gene_ids.ensembl(cohort, v).dropna())
    return out


def overlap_row(name: str, ids: set[str], voc: dict[str, set[str]], note: str = "") -> dict:
    row = {"source": name, "n_ids": len(ids)}
    for k, v in voc.items():
        row[f"in_{k}"] = len(ids & v)
    row["note"] = note
    return row


# ---------------------------------------------------------------- symbol mapping
class SymbolMapper:
    """HGNC symbol -> ENSG. Approved symbols first, then previous symbols, then aliases.

    A symbol is ambiguous, and dropped, when it is not approved and its previous or alias
    hits point to more than one ENSG, or when its single hit is an ENSG that another input
    symbol already claims through an approved match. Ensembl 110 GTF gene_name is used
    as a cross-check and for symbols HGNC lacks.
    """

    def __init__(self, hgnc: Path, gtf_names: dict[str, set[str]]):
        h = pd.read_csv(hgnc, sep="\t", dtype=str, usecols=[
            "symbol", "status", "ensembl_gene_id", "prev_symbol", "alias_symbol"])
        h = h[h.status == "Approved"]
        self.approved = {s: e for s, e in zip(h.symbol, h.ensembl_gene_id) if isinstance(e, str)}
        self.approved_noensg = set(h.symbol[h.ensembl_gene_id.isna()])
        self.prev, self.alias = defaultdict(set), defaultdict(set)
        for s, e, p, a in zip(h.symbol, h.ensembl_gene_id, h.prev_symbol, h.alias_symbol):
            if not isinstance(e, str):
                continue
            for col, d in ((p, self.prev), (a, self.alias)):
                if isinstance(col, str):
                    for x in col.split("|"):
                        d[x.strip()].add(e)
        self.gtf = gtf_names

    def map(self, symbols) -> pd.DataFrame:
        syms = pd.Index(pd.Series(list(symbols), dtype=str).str.strip())
        rows = []
        for s in syms:
            if s in self.approved:
                rows.append((s, self.approved[s], "approved"))
            elif s in self.approved_noensg:
                rows.append((s, None, "approved_without_ensg"))
            elif len(self.prev.get(s, ())) == 1:
                rows.append((s, next(iter(self.prev[s])), "previous"))
            elif len(self.prev.get(s, ())) > 1:
                rows.append((s, None, "ambiguous_previous"))
            elif len(self.alias.get(s, ())) == 1:
                rows.append((s, next(iter(self.alias[s])), "alias"))
            elif len(self.alias.get(s, ())) > 1:
                rows.append((s, None, "ambiguous_alias"))
            elif len(self.gtf.get(s, ())) == 1:
                rows.append((s, next(iter(self.gtf[s])), "ensembl110_gene_name"))
            elif len(self.gtf.get(s, ())) > 1:
                rows.append((s, None, "ambiguous_ensembl110_gene_name"))
            else:
                rows.append((s, None, "unmapped"))
        df = pd.DataFrame(rows, columns=["symbol", "ensg", "route"])
        # duplicated input symbols
        dup_sym = df.symbol.duplicated(keep=False)
        df.loc[dup_sym, ["ensg", "route"]] = [None, "duplicated_input_symbol"]
        # one ENSG reached by more than one input symbol: keep the approved one if unique,
        # drop the rest; if none is approved, drop all of them
        hit = df.ensg.notna()
        for e, idx in df[hit].groupby("ensg").groups.items():
            if len(idx) < 2:
                continue
            appr = [i for i in idx if df.at[i, "route"] == "approved"]
            keep = appr[0] if len(appr) == 1 else None
            for i in idx:
                if i != keep:
                    df.at[i, "ensg"] = None
                    df.at[i, "route"] = "ambiguous_shared_ensg"
        # GTF cross-check for approved hits
        df["gtf_agrees"] = [
            (e in self.gtf.get(s, ())) if isinstance(e, str) else np.nan
            for s, e in zip(df.symbol, df.ensg)
        ]
        return df


def gtf_gene_names(gtf: Path) -> tuple[dict[str, set[str]], dict[str, str], dict[str, int]]:
    """Ensembl 110 GTF: gene_name -> {ENSG}, ENSG -> biotype, ENSG -> longest CDS (bp)."""
    names, biotype = defaultdict(set), {}
    rx_id = re.compile(r'gene_id "([^"]+)"')
    rx_name = re.compile(r'gene_name "([^"]+)"')
    rx_bt = re.compile(r'gene_biotype "([^"]+)"')
    with gzip.open(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 9)
            if p[2] != "gene":
                continue
            gid = rx_id.search(p[8]).group(1)
            m = rx_name.search(p[8])
            if m:
                names[m.group(1)].add(gid)
            b = rx_bt.search(p[8])
            biotype[gid] = b.group(1) if b else ""
    return names, biotype


# ---------------------------------------------------------------- parts
def part_dnds(repo: Path, out: Path, voc) -> list[dict]:
    rows, ov = [], []
    for rel, path in (("102", repo / "data/revision/gold/G8/Compara.102.protein_default.homologies.tsv.gz"),
                      ("110", repo / "data/revision/ensembl110/Compara.110.protein_default.homologies.tsv.gz")):
        cnt = Counter()
        mouse_1to1 = {}
        for ch in pd.read_csv(path, sep="\t", chunksize=2_000_000, dtype=str,
                              usecols=["gene_stable_id", "homology_type", "homology_species", "dn", "ds"]):
            ch = ch[ch.homology_species.isin(["mus_musculus", "macaca_mulatta", "danio_rerio"])]
            for (sp, ht), g in ch.groupby(["homology_species", "homology_type"]):
                dn_ok = g.dn.ne("NULL") & g.dn.notna()
                ds_ok = g.ds.ne("NULL") & g.ds.notna()
                cnt[(sp, ht, "rows")] += len(g)
                cnt[(sp, ht, "dn_populated")] += int(dn_ok.sum())
                cnt[(sp, ht, "ds_populated")] += int(ds_ok.sum())
                cnt[(sp, ht, "both_populated")] += int((dn_ok & ds_ok).sum())
                cnt[(sp, ht, "ds_positive")] += int((ds_ok & (pd.to_numeric(g.ds, errors="coerce") > 0)).sum())
                if sp == "mus_musculus" and ht == "ortholog_one2one":
                    ok = g[dn_ok & ds_ok]
                    mouse_1to1.update(zip(ok.gene_stable_id, ok.ds))
        for (sp, ht, k), v in sorted(cnt.items()):
            rows.append({"release": rel, "species": sp, "homology_type": ht, "measure": k, "n": v})
        ds = pd.to_numeric(pd.Series(mouse_1to1), errors="coerce")
        log(f"Compara {rel}: human-mouse 1:1 with dn and ds: {len(mouse_1to1)} genes; ds quantiles",
            ds.quantile([0, .5, .99, 1]).round(3).to_dict() if len(ds) else {})
        ov.append(overlap_row(f"G8_compara{rel}_hmouse_1to1_dn_ds", set(mouse_1to1), voc,
                              "human-mouse ortholog_one2one with dn and ds non-NULL"))
    pd.DataFrame(rows).to_csv(out / "dnds_population.tsv", sep="\t", index=False)
    return ov


def part_evee(repo: Path, out: Path, voc, mapper: SymbolMapper) -> list[dict]:
    z = zipfile.ZipFile(repo / "data/revision/gold/G1/evolutionary_statistics.zip")
    tabs = {k: pd.read_csv(io.BytesIO(z.read(f"evolutionary_statistics/{k}.txt")), sep="\t")
            for k in ("var", "mean", "fdr", "perc")}
    v = tabs["var"]
    scale = []
    for k, t in tabs.items():
        for tissue in ("liver", "testis"):
            x = t[tissue]
            scale.append({"file": k, "tissue": tissue, "n": int(x.notna().sum()), "n_nan": int(x.isna().sum()),
                          "min": x.min(), "q01": x.quantile(.01), "median": x.median(), "q99": x.quantile(.99),
                          "max": x.max(), "n_le0": int((x <= 0).sum()),
                          "log10_range": float(np.log10(x[x > 0].max() / x[x > 0].min())) if (x > 0).any() else np.nan})
    pd.DataFrame(scale).to_csv(out / "evee_scale.tsv", sep="\t", index=False)
    same_genes = all(t.gene_name.equals(v.gene_name) for t in tabs.values())
    log("EVEE rows", len(v), "files share gene order", same_genes, "dup symbols", int(v.gene_name.duplicated().sum()))
    m = mapper.map(v.gene_name)
    m.to_csv(out / "evee_symbol_to_ensg.tsv", sep="\t", index=False)
    log("EVEE mapping routes", m.route.value_counts().to_dict(), "gtf disagrees", int((m.gtf_agrees == False).sum()))  # noqa: E712
    ids = set(m.ensg.dropna())
    return [overlap_row("G1_EVEE_var_mapped", ids, voc,
                        f"{len(v)} symbols; {len(ids)} kept after dropping ambiguous; routes {m.route.value_counts().to_dict()}")]


def read_gmt(path: Path) -> dict[str, list[str]]:
    d = {}
    for line in path.read_text().splitlines():
        p = line.split("\t")
        d[p[0]] = [g for g in p[2:] if g]
    return d


def go_descendants(obo: Path, root: str) -> set[str]:
    """Terms under ``root`` through is_a and part_of (true-path rule)."""
    children = defaultdict(set)
    cur, obsolete = None, False
    for line in obo.read_text().splitlines():
        if line == "[Term]":
            cur, obsolete = None, False
        elif line.startswith("id: GO:"):
            cur = line[4:]
        elif line.startswith("is_obsolete: true"):
            obsolete = True
        elif cur and not obsolete and line.startswith("is_a: "):
            children[line[6:16]].add(cur)
        elif cur and not obsolete and line.startswith("relationship: part_of "):
            children[line[22:32]].add(cur)
    seen, stack = {root}, [root]
    while stack:
        for c in children.get(stack.pop(), ()):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def part_sets(repo: Path, out: Path, voc, mapper: SymbolMapper) -> list[dict]:
    G = repo / "data/revision/gold"
    ov = []
    # G7 LOEUF
    g = pd.read_csv(G / "G7/gnomad.v4.1.constraint_metrics.tsv", sep="\t", dtype=str,
                    usecols=["gene", "gene_id", "transcript", "canonical", "mane_select", "lof.oe_ci.upper"])
    g = g[g.gene_id.str.startswith("ENSG", na=False)]
    loeuf = pd.to_numeric(g["lof.oe_ci.upper"], errors="coerce")
    can = g[(g.canonical == "true") & loeuf.notna()]
    mane = g[(g.mane_select == "true") & loeuf.notna()]
    log("gnomAD ENSG rows", len(g), "canonical with LOEUF", can.gene_id.nunique(),
        "MANE with LOEUF", mane.gene_id.nunique(), "canonical dup genes", int(can.gene_id.duplicated().sum()))
    ov.append(overlap_row("G7_LOEUF_canonical", set(can.gene_id), voc, "lof.oe_ci.upper, canonical transcript, ENSG rows"))
    ov.append(overlap_row("G7_LOEUF_MANE", set(mane.gene_id), voc, "lof.oe_ci.upper, MANE Select transcript"))
    # G9 housekeeping
    hk = pd.read_csv(G / "G9/HK_genes.txt", sep="\t", header=None, names=["symbol", "refseq"], dtype=str)
    m = mapper.map(hk.symbol.str.strip())
    m.to_csv(out / "hk_symbol_to_ensg.tsv", sep="\t", index=False)
    ov.append(overlap_row("G9_C1_housekeeping", set(m.ensg.dropna()), voc,
                          f"{len(hk)} symbols; routes {m.route.value_counts().to_dict()}"))
    # G10 HGNC groups
    fam = pd.read_csv(G / "G10/hgnc_family.csv", dtype=str)
    hier = pd.read_csv(G / "G10/hgnc_hierarchy.csv", dtype=str)
    ghf = pd.read_csv(G / "G10/hgnc_gene_has_family.csv", dtype=str)
    hg = pd.read_csv(G / "G10/hgnc_complete_set.txt", sep="\t", dtype=str, usecols=["hgnc_id", "symbol", "ensembl_gene_id", "status"])
    hg["hid"] = hg.hgnc_id.str.replace("HGNC:", "", regex=False)
    hid2ensg = dict(zip(hg.hid, hg.ensembl_gene_id))
    kids = defaultdict(set)
    for p, c in zip(hier.parent_fam_id, hier.child_fam_id):
        kids[p].add(c)

    def family_genes(fid: str) -> set[str]:
        ids, stack = {fid}, [fid]
        while stack:
            for c in kids.get(stack.pop(), ()):
                if c not in ids:
                    ids.add(c)
                    stack.append(c)
        h = set(ghf.hgnc_id[ghf.family_id.isin(ids)])
        return {hid2ensg[x] for x in h if isinstance(hid2ensg.get(x), str)}

    fname = dict(zip(fam.id, fam.name))
    groups = {
        "C2_cytosolic_ribosomal (728 S + 729 L)": ["728", "729"],
        "C2_proteasome (690)": ["690"],
        "C2_spliceosome_core option A: Sm spliceosomal (1255) + LSm (1505)": ["1255", "1505"],
        "C2_spliceosome_core option B: U1, U2, U4/U6, U5 snRNPs (1499, 1501, 1502, 1503)": ["1499", "1501", "1502", "1503"],
        "C2_spliceosome option C: Major spliceosome (1518)": ["1518"],
    }
    for label, fids in groups.items():
        ids = set().union(*[family_genes(f) for f in fids])
        ov.append(overlap_row(f"G10_{label}", ids, voc, "; ".join(f"{f}={fname.get(f)}" for f in fids)))
    # Hallmark (Enrichr cache copy in the functional folder)
    hm = read_gmt(Path.home() / ".cache/gseapy/Enrichr.MSigDB_Hallmark_2020.gmt")
    sets = {
        "V1_xenobiotic+bile_acid": ["Xenobiotic Metabolism", "Bile Acid Metabolism"],
        "V2_IFNa+IFNg+inflammatory": ["Interferon Alpha Response", "Interferon Gamma Response", "Inflammatory Response"],
    }
    for label, terms in sets.items():
        missing = [t for t in terms if t not in hm]
        syms = sorted(set().union(*[hm[t] for t in terms if t in hm]))
        mm = mapper.map(syms)
        ov.append(overlap_row(f"G10_{label}", set(mm.ensg.dropna()), voc,
                              f"{len(syms)} symbols; missing terms {missing}; routes {mm.route.value_counts().to_dict()}"))
    # GO extracellular region GO:0005576 from the human GAF
    desc = go_descendants(G / "G10/go-basic.obo", "GO:0005576")
    direct, prop = set(), set()
    with gzip.open(G / "G10/goa_human.gaf.gz", "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            p = line.split("\t")
            if p[8] != "C" or "NOT" in p[3]:
                continue
            if p[4] == "GO:0005576":
                direct.add(p[2])
            if p[4] in desc:
                prop.add(p[2])
    for label, s in (("V3_GO0005576_direct", direct), ("V3_GO0005576_with_descendants", prop)):
        mm = mapper.map(sorted(s))
        ov.append(overlap_row(f"G10_{label}", set(mm.ensg.dropna()), voc,
                              f"{len(s)} GAF symbols (NOT excluded); {len(desc)} terms in closure; routes {mm.route.value_counts().to_dict()}"))
    return ov


def part_h3(repo: Path, out: Path, label_map: Path) -> None:
    import anndata as ad

    lm = pd.read_csv(label_map, sep="\t")
    ts = ad.read_h5ad(repo / "data/revision/gold/G2/ts_liver.h5ad", backed="r")
    tms = ad.read_h5ad(repo / "data/revision/gold/G3/tms_liver_10x.h5ad", backed="r")
    to, mo = ts.obs.copy(), tms.obs.copy()
    ts.file.close()
    tms.file.close()
    log("TS assays", to["assay"].astype(str).value_counts().to_dict())
    log("TMS assays", mo["assay"].astype(str).value_counts().to_dict())
    keep = to["assay"].astype(str) == "10x 3' v3"
    tc = to.loc[keep, "cell_type"].astype(str).value_counts()
    tc_all = to["cell_type"].astype(str).value_counts()
    mc = mo["cell_type"].astype(str).value_counts()
    rows = []
    for r in lm.itertuples():
        a, b = int(tc.get(r.ts_cell_type, 0)), int(mc.get(r.tms_cell_type, 0))
        rows.append({"coarse_type": r.coarse_type, "ts_cell_type": r.ts_cell_type, "tms_cell_type": r.tms_cell_type,
                     "ts_cells_map": r.ts_cells, "tms_cells_map": r.tms_cells,
                     "ts_cells_all_assays": int(tc_all.get(r.ts_cell_type, 0)),
                     "ts_cells_10x3v3": a, "tms_cells": b, "ge30_both": (a >= 30) and (b >= 30)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "h3_counts.tsv", sep="\t", index=False)
    pd.DataFrame({"ts_cell_type_10x3v3": tc}).to_csv(out / "h3_ts_all_types_10x3v3.tsv", sep="\t")
    pd.DataFrame({"tms_cell_type": mc}).to_csv(out / "h3_tms_all_types.tsv", sep="\t")
    # donors per type, needed later for split-half ceilings
    dcol_ts = next(c for c in ("donor_id", "donor") if c in to.columns)
    dcol_ms = next(c for c in ("donor_id", "mouse.id", "donor") if c in mo.columns)
    dn = []
    for r in lm.itertuples():
        dn.append({"coarse_type": r.coarse_type,
                   "ts_donors_10x3v3": int(to.loc[keep & (to.cell_type.astype(str) == r.ts_cell_type), dcol_ts].nunique()),
                   "tms_donors": int(mo.loc[mo.cell_type.astype(str) == r.tms_cell_type, dcol_ms].nunique())})
    pd.DataFrame(dn).to_csv(out / "h3_donors.tsv", sep="\t", index=False)
    log(df.to_string(index=False))
    log(pd.DataFrame(dn).to_string(index=False))
    log("K (types with >=30 cells in both)", int(df.ge30_both.sum()))


def part_h4(repo: Path, out: Path, voc) -> None:
    import anndata as ad

    dset = repo / "data/raw/camex/2testis/dataset"
    var, coarse = {}, {}
    for sp in PRIMATES + OUTGROUPS:
        a = ad.read_h5ad(dset / f"raw-testis-{sp}-Murat.h5ad", backed="r")
        var[sp] = a.var[["gene", "gene_symbol"]].copy()
        o = a.obs
        coarse[sp] = pd.crosstab(o.cell_ontology_class.astype(str), o.sample_batch.astype(str))
        a.file.close()
    # coarse types
    allc = sorted(set().union(*[set(c.index) for c in coarse.values()]))
    ct = pd.DataFrame({sp: [int(coarse[sp].loc[c].sum()) if c in coarse[sp].index else 0 for c in allc]
                       for sp in coarse}, index=allc)
    minlib = pd.DataFrame({sp: [int(coarse[sp].loc[c].min()) if c in coarse[sp].index else 0 for c in allc]
                           for sp in coarse}, index=allc)
    ct.to_csv(out / "h4_coarse_cells.tsv", sep="\t")
    minlib.to_csv(out / "h4_coarse_min_cells_per_library.tsv", sep="\t")
    in_all = [c for c in allc if (ct.loc[c] > 0).all()]
    log("H4 coarse cells per species\n" + ct.to_string())
    log("H4 min cells per library\n" + minlib.to_string())
    log("H4 coarse types present in all 8:", in_all)
    # one2one orthologs: Ensembl 110 for every species except chicken, whose Murat IDs are
    # GRCg6a IDs that Ensembl 110 (GRCg7b) no longer carries; chicken uses release 105,
    # the last release in which gallus_gallus is GRCg6a.
    parts = []
    for rel, species in (("110", set(ENSEMBL_SPECIES.values()) - {"gallus_gallus"}), ("105", {"gallus_gallus"})):
        for ch in pd.read_csv(repo / f"data/revision/ensembl110/Compara.{rel}.protein_default.homologies.tsv.gz",
                              sep="\t", chunksize=2_000_000, dtype=str,
                              usecols=["gene_stable_id", "homology_type", "homology_gene_stable_id",
                                       "homology_species", "is_high_confidence"]):
            ch = ch[ch.homology_species.isin(species) & (ch.homology_type == "ortholog_one2one")].copy()
            ch["release"] = rel
            parts.append(ch)
    o = pd.concat(parts)
    rows, cov = [], []
    for sp in PRIMATES + OUTGROUPS:
        es = ENSEMBL_SPECIES[sp]
        oo = o[o.homology_species == es]
        ids = var[sp].gene.astype(str)
        stripped = ids.str.replace(r"\.\d+$", "", regex=True)
        hit = stripped.isin(set(oo.homology_gene_stable_id))
        prefix_ok = stripped.str.startswith(ID_PREFIX[es]).mean()
        sub = oo[oo.homology_gene_stable_id.isin(set(stripped))]
        for h, s, hc, r in zip(sub.gene_stable_id, sub.homology_gene_stable_id, sub.is_high_confidence, sub.release):
            rows.append((h, sp, es, s, hc, r))
        cov.append({"species": sp, "ensembl_species": es, "var_genes": len(ids), "id_prefix_fraction": round(prefix_ok, 4),
                    "ensembl_release": "105" if es == "gallus_gallus" else "110",
                    "one2one_pairs_species": len(oo), "var_genes_with_one2one": int(hit.sum()),
                    "human_genes_mapped": sub.gene_stable_id.nunique()})
    m = pd.DataFrame(rows, columns=["hsapiens_gene_id", "murat_species", "ensembl_species", "species_gene_id", "is_high_confidence", "ensembl_release"])
    m.to_csv(out / "h4_ortholog_map_one2one.tsv.gz", sep="\t", index=False, compression="gzip")
    covdf = pd.DataFrame(cov)
    log(covdf.to_string(index=False))
    pres = m.groupby("hsapiens_gene_id").murat_species.apply(set)
    n_pri = pres.apply(lambda s: len(s & set(PRIMATES)))
    n_out = pres.apply(lambda s: len(s & set(OUTGROUPS)))
    eligible = set(pres.index[(n_pri >= 3) & (n_out >= 2)])
    prim_only = set(pres.index[n_pri >= 3])
    covdf.to_csv(out / "h4_species_coverage.tsv", sep="\t", index=False)
    tab = pd.crosstab(n_pri, n_out)
    tab.to_csv(out / "h4_primate_outgroup_presence.tsv", sep="\t")
    log("genes by (#primates, #outgroups) present\n" + tab.to_string())
    return [overlap_row("H4_eligible (>=3 primates, >=2 outgroups, 1:1, in Murat var)", eligible, voc),
            overlap_row("H4_primates_only (>=3 primates)", prim_only, voc)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--label-map", type=Path, required=True)
    ap.add_argument("--part", default="all")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    parts = {"dnds", "evee", "sets", "h3", "h4"} if a.part == "all" else set(a.part.split(","))
    voc = vocabularies(a.repo)
    log("vocabulary sizes", {k: len(v) for k, v in voc.items()})
    ov = []
    mapper = None
    if parts & {"evee", "sets"}:
        names, biotype = gtf_gene_names(a.repo / "data/revision/ensembl110/Homo_sapiens.GRCh38.110.gtf.gz")
        mapper = SymbolMapper(a.repo / "data/revision/gold/G10/hgnc_complete_set.txt", names)
        e110 = set(biotype)
        for k, v in voc.items():
            log(f"{k}: ENSG present in Ensembl 110 GTF {len(v & e110)} of {len(v)}")
    if "dnds" in parts:
        ov += part_dnds(a.repo, a.out, voc)
    if "evee" in parts:
        ov += part_evee(a.repo, a.out, voc, mapper)
    if "sets" in parts:
        ov += part_sets(a.repo, a.out, voc, mapper)
    if "h3" in parts:
        part_h3(a.repo, a.out, a.label_map)
    if "h4" in parts:
        ov += part_h4(a.repo, a.out, voc)
    if ov:
        df = pd.DataFrame(ov)
        f = a.out / f"id_overlap__{a.part.replace(',', '_')}.tsv"
        df.to_csv(f, sep="\t", index=False)
        log(df.drop(columns="note").to_string(index=False))


if __name__ == "__main__":
    main()
