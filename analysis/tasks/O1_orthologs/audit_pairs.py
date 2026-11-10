"""Step 1: every pair of every ortholog table against Compara 110 (protein and ncRNA dumps), 105, 116, the 110 GTFs; rebuild of each table from the dumps.
No count, readout or gold enters this script. Definitions: rules.py.

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.audit_pairs
"""

from __future__ import annotations

import gzip
import json
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from analysis.tasks.O1_orthologs import rules as RU

COLS = ["gene_stable_id", "species", "identity", "homology_type", "homology_gene_stable_id", "homology_species", "homology_identity", "goc_score", "wga_coverage", "is_high_confidence"]


def read_dump(args) -> pd.DataFrame:
    """Rows of one dump between human and any other species, oriented human first."""
    path, source = args
    awk = "awk -F'\\t' 'NR==1 || $3==\"homo_sapiens\" || $8==\"homo_sapiens\"'"
    out = subprocess.run(f"zcat '{path}' | {awk}", shell=True, capture_output=True, text=True, check=True).stdout
    from io import StringIO

    d = pd.read_csv(StringIO(out), sep="\t", dtype=str, usecols=COLS, na_values=["NULL"])
    fwd = d[(d.species == "homo_sapiens") & (d.homology_species != "homo_sapiens")]
    rev = d[(d.homology_species == "homo_sapiens") & (d.species != "homo_sapiens")].rename(columns={
        "gene_stable_id": "homology_gene_stable_id", "homology_gene_stable_id": "gene_stable_id", "species": "homology_species", "homology_species": "species",
        "identity": "homology_identity", "homology_identity": "identity"})
    d = pd.concat([fwd[COLS], rev[COLS]], ignore_index=True)
    d = d.rename(columns={"gene_stable_id": "hsapiens_gene_id", "homology_gene_stable_id": "gene_id", "homology_species": "species_ens"}).drop(columns="species")
    d["source"] = source
    return d


def compara_rows(files: list[tuple], cache) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    with ProcessPoolExecutor(8) as ex:
        parts = list(ex.map(read_dump, files))
    d = pd.concat(parts, ignore_index=True).drop_duplicates(["species_ens", "hsapiens_gene_id", "gene_id", "homology_type", "source"])
    for c in ["identity", "homology_identity", "goc_score", "wga_coverage", "is_high_confidence"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d.to_parquet(cache)
    return d


def gtf_info(sp: str) -> pd.DataFrame:
    genes, cds = {}, {}
    with gzip.open(RU.GTF_110[sp], "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 8)
            if p[2] == "gene":
                g = re.search(r'gene_id "([^"]+)"', p[8]).group(1)
                n = re.search(r'gene_name "([^"]+)"', p[8])
                b = re.search(r'gene_biotype "([^"]+)"', p[8])
                genes[g] = (n.group(1) if n else "", b.group(1) if b else "", p[0])
            elif p[2] == "CDS":
                g = re.search(r'gene_id "([^"]+)"', p[8]).group(1)
                t = re.search(r'transcript_id "([^"]+)"', p[8]).group(1)
                cds[(g, t)] = cds.get((g, t), 0) + 1
    d = pd.DataFrame.from_dict(genes, orient="index", columns=["name", "biotype", "contig"])
    s = pd.Series(cds)
    s.index = pd.MultiIndex.from_tuples(s.index)
    d["n_cds_exons"] = s.groupby(level=0).max().reindex(d.index)
    d.index.name = "gene_id"
    d.insert(0, "species", sp)
    return d.reset_index()


def table_pairs() -> pd.DataFrame:
    rows = []
    for name, path in RU.TABLES.items():
        t = pd.read_csv(path)
        for sp in RU.ENS:
            if f"{sp}_gene_id" in t:
                r = t[["ortholog_group_id", "hsapiens_gene_id", "hsapiens_gene_name", f"{sp}_gene_id", f"{sp}_gene_name", f"{sp}_identity"]].copy()
                r.columns = ["ortholog_group_id", "hsapiens_gene_id", "hsapiens_gene_name", "gene_id", "table_gene_name", "identity_table"]
                r.insert(0, "species", sp)
                r.insert(0, "table", name)
                rows.append(r)
    return pd.concat(rows, ignore_index=True)


def annotate(tp: pd.DataFrame, c110: pd.DataFrame, c105: pd.DataFrame, gtf: pd.DataFrame) -> pd.DataFrame:
    tp = tp.copy()
    tp["species_ens"] = tp.species.map(RU.ENS)
    prot = c110[c110.source == "protein"].drop_duplicates(["species_ens", "hsapiens_gene_id", "gene_id"])
    nc = c110[c110.source == "ncrna"].drop_duplicates(["species_ens", "hsapiens_gene_id", "gene_id"])
    key = ["species_ens", "hsapiens_gene_id", "gene_id"]
    m = tp.merge(prot[key + ["homology_type", "identity", "goc_score", "wga_coverage", "is_high_confidence"]], on=key, how="left")
    m = m.merge(nc[key + ["homology_type", "identity"]].rename(columns={"homology_type": "ncrna_type", "identity": "ncrna_identity"}), on=key, how="left")
    h = gtf[gtf.species == "hsapiens"].set_index("gene_id")
    s = gtf[gtf.species != "hsapiens"].set_index(["species", "gene_id"])
    m["human_biotype"] = m.hsapiens_gene_id.map(h.biotype)
    m["human_exons"] = m.hsapiens_gene_id.map(h.n_cds_exons)
    idx = pd.MultiIndex.from_arrays([m.species, m.gene_id])
    m["species_biotype"] = s.biotype.reindex(idx).to_numpy()
    m["species_exons"] = s.n_cds_exons.reindex(idx).to_numpy()
    m["species_gene_name"] = s.name.reindex(idx).to_numpy()
    m["human_symbol_110"] = m.hsapiens_gene_id.map(h.name)

    inv = {}
    for sp, x in gtf[(gtf.species != "hsapiens") & (gtf.name != "")].groupby("species"):
        tmp = {}
        for g, n in zip(x.gene_id, x.name):
            tmp.setdefault(n.upper(), set()).add(g)
        inv[sp] = tmp
    sym = m.human_symbol_110.fillna(m.hsapiens_gene_name).fillna("").str.upper().to_numpy()
    own = m.species_gene_name.fillna("").str.upper().to_numpy()
    other, f_named = [], []
    for sp, gid, y, o in zip(m.species, m.gene_id, sym, own):
        cand = sorted(inv.get(sp, {}).get(y, set()) - {gid}) if y else []
        other.append(";".join(cand))
        f_named.append(bool(y) and o != y and len(cand) > 0)
    m["named_gene_elsewhere"], m["F_named_raw"] = other, f_named

    # What Compara says about the named gene and about the human gene's other partners
    pp = prot[prot.homology_type.str.startswith("ortholog")]
    by_h = pp.groupby(["species_ens", "hsapiens_gene_id"]).gene_id.agg(lambda x: ";".join(sorted(x)))
    m["human_partners_110"] = by_h.reindex(pd.MultiIndex.from_arrays([m.species_ens, m.hsapiens_gene_id])).to_numpy()
    by_g = pp.groupby(["species_ens", "gene_id"]).hsapiens_gene_id.agg(lambda x: ";".join(sorted(x)))
    m["species_partners_110"] = by_g.reindex(pd.MultiIndex.from_arrays([m.species_ens, m.gene_id])).to_numpy()

    in_prot, in_nc = m.homology_type.notna(), m.ncrna_type.notna()
    m["F_absent"] = ~in_prot & ~in_nc
    m["F_ncrna"] = ~in_prot & in_nc
    m["F_type"] = (in_prot & (m.homology_type != "ortholog_one2one")) | (~in_prot & in_nc & (m.ncrna_type != "ortholog_one2one"))
    m["F_biotype"] = (m.human_biotype != "protein_coding") | (m.species_biotype != "protein_coding")
    m["F_lowconf"] = in_prot & (m.is_high_confidence == 0)
    m["F_nosynt"] = (m.goc_score == 0) & (m.wga_coverage == 0)
    m["F_retro"] = (m.species_exons <= 1) & (m.human_exons >= 3)
    m["F_named"] = m.F_named_raw & ~((m.goc_score > 50) & (m.wga_coverage > 50))

    lab = pd.read_parquet(RU.REPO / "results/analysis/Y6/annot/labels.parquet")[["species", "hsapiens_gene_id", "gene_id", "homology_type_116", "in_116"]].drop_duplicates(["species", "hsapiens_gene_id", "gene_id"])
    m = m.merge(lab, on=["species", "hsapiens_gene_id", "gene_id"], how="left")
    m["F_later"] = m.homology_type_116 != "ortholog_one2one"
    t105 = c105[c105.homology_type.str.startswith("ortholog")].drop_duplicates(key)[key + ["homology_type"]].rename(columns={"homology_type": "homology_type_105"})
    m = m.merge(t105, on=key, how="left")
    has105 = c105.species_ens.unique()
    m["F_earlier"] = m.species_ens.isin(has105) & (m.homology_type_105 != "ortholog_one2one")
    m["T1"] = m[RU.FLAGS_T1].any(axis=1)
    m["T2"] = m[RU.FLAGS_T2].any(axis=1)
    return m


def rebuild(tp: pd.DataFrame, c110: pd.DataFrame, c105: pd.DataFrame) -> pd.DataFrame:
    """Is each table the inner join on the human gene of Compara one-to-one pairs? Which release, which dump."""
    rows = []
    for name, path in RU.TABLES.items():
        t = pd.read_csv(path)
        sps = [sp for sp in RU.ENS if f"{sp}_gene_id" in t]
        for label, frame in (("110 protein", c110[c110.source == "protein"]), ("110 protein+ncRNA", c110), ("105 human dump", c105)):
            if label.startswith("105") and not all(RU.ENS[sp] in set(c105.species_ens) for sp in sps):
                continue
            j = None
            for sp in sps:
                d = frame[(frame.species_ens == RU.ENS[sp]) & (frame.homology_type == "ortholog_one2one")][["hsapiens_gene_id", "gene_id"]].drop_duplicates()
                d = d[~d.hsapiens_gene_id.duplicated(keep=False) & ~d.gene_id.duplicated(keep=False)].rename(columns={"gene_id": sp})
                j = d if j is None else j.merge(d, on="hsapiens_gene_id")
            jj = set(map(tuple, j[["hsapiens_gene_id"] + sps].to_numpy()))
            tt = set(map(tuple, t[["hsapiens_gene_id"] + [f"{sp}_gene_id" for sp in sps]].to_numpy()))
            rows.append(dict(table=name, source=label, n_table=len(tt), n_rebuilt=len(jj), both=len(jj & tt), table_only=len(tt - jj), rebuilt_only=len(jj - tt)))
        for sp in sps:  # identity equal to the release's value?
            for label, frame in (("110 protein", c110[c110.source == "protein"]), ("110 ncRNA", c110[c110.source == "ncrna"]), ("105 human dump", c105)):
                d = frame[frame.species_ens == RU.ENS[sp]].drop_duplicates(["hsapiens_gene_id", "gene_id"])
                x = t[["hsapiens_gene_id", f"{sp}_gene_id", f"{sp}_identity"]].merge(d, left_on=["hsapiens_gene_id", f"{sp}_gene_id"], right_on=["hsapiens_gene_id", "gene_id"])
                if len(x):
                    rows.append(dict(table=name, source=f"identity {label}", species=sp, n_table=len(t), both=len(x), identity_equal=int((np.abs(x[f"{sp}_identity"] - x.identity) < 1e-3).sum())))
    return pd.DataFrame(rows)


def main():
    RU.DATA.mkdir(parents=True, exist_ok=True)
    RU.RES.mkdir(parents=True, exist_ok=True)
    files = [(p, "protein") for p in RU.PROTEIN_110.values()] + [(p, "ncrna") for p in RU.NCRNA_110.values()]
    c110 = compara_rows(files, RU.DATA / "compara110_human_rows.parquet")
    c105 = compara_rows([(RU.PROTEIN_105_HUMAN, "protein")], RU.DATA / "compara105_human_rows.parquet")
    print("rows", len(c110), len(c105), c105.species_ens.nunique(), flush=True)
    with ProcessPoolExecutor(7) as ex:
        gtf = pd.concat(list(ex.map(gtf_info, RU.GTF_110)), ignore_index=True)
    gtf.to_parquet(RU.DATA / "gtf110_genes.parquet")
    tp = table_pairs()
    pairs = annotate(tp, c110, c105, gtf)
    pairs.to_parquet(RU.RES / "table_pairs_flags.parquet")
    pairs[(pairs.T2)].to_csv(RU.RES / "table_pairs_flagged_T2.csv.gz", index=False)
    rb = rebuild(tp, c110, c105)
    rb.to_csv(RU.RES / "table_rebuild_check.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    print(rb.to_string())
    fl = ["F_absent", "F_ncrna", "F_type", "F_biotype", "F_lowconf", "F_nosynt", "F_retro", "F_named_raw", "F_named", "F_later", "F_earlier", "T1", "T2"]
    print(pairs.groupby(["table", "species"])[fl].sum())
    print((pairs.groupby(["table", "species"])[fl].mean() * 100).round(2))
    print(pairs[pairs.F_biotype].groupby(["table", "species"]).species_biotype.value_counts().to_string())
    print(pairs[pairs.F_biotype].groupby(["table", "species"]).human_biotype.value_counts().to_string())
    (RU.RES / "audit_pairs_meta.json").write_text(json.dumps(dict(n_pairs=len(pairs)), indent=1))


if __name__ == "__main__":
    main()
