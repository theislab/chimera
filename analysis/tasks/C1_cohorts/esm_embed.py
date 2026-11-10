"""C1: ESM-2 gene embeddings for species outside the SATURN export, the SATURN recipe.

SATURN (``protein_embeddings/`` of its clone, see ``analysis/external_tools``): Ensembl ``pep.all``, proteins with a stop codon
dropped; ``esm2_t48_15B_UR50D``, sequences truncated at 1022 residues, mean of the last layer (48) over
residues; a gene's embedding is the mean over its proteins. The existing ``ESM2/*.torch`` files are
this recipe keyed by symbol. Here the output is keyed by Ensembl gene ID, which
``chimera._data_utils.download_and_process_esm2`` matches first. Only genes of ``--genes`` are embedded.
Identical sequences are embedded once (a deterministic model maps them to the same vector).

    esm_embed.py plan --species ptroglodytes --genes genes.txt     # writes the shard plan
    esm_embed.py shard --species ptroglodytes --index K --threads 16 # one array task, resumable
    esm_embed.py combine --species ptroglodytes                    # one .torch file
    esm_embed.py check                                             # recipe against the SATURN human file
    esm_embed.py pool --species a,b --workers 8 --threads 4        # one model, forked workers, shards in order
    plan --canonical: the protein Compara used, one per gene, in place of the mean over isoforms, to cut CPU; canoncheck measures it on human

Files read are Ensembl FASTA and text, as data. Run with ``python -I``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # the repository root
OUT = ROOT / "data" / "analysis" / "new_usecases" / "esm"
WEIGHTS = OUT / "hub" / "checkpoints" / "esm2_t48_15B_UR50D.pt"
NEW = ROOT / "data" / "analysis" / "new_usecases" / "ensembl110"
PEP = {"ptroglodytes": NEW / "Pan_troglodytes.Pan_tro_3.0.pep.all.fa.gz",
       "cjacchus": NEW / "Callithrix_jacchus.mCalJac1.pat.X.pep.all.fa.gz",
       "ggallus": NEW / "Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.pep.all.fa.gz",
       "hsapiens": NEW / "Homo_sapiens.GRCh38.pep.all.fa.gz"}
COMMON = {"ptroglodytes": "chimpanzee", "cjacchus": "marmoset", "ggallus": "chicken", "hsapiens": "human"}
N_SHARDS = 96
MAX_LEN = 1022
LAYER = 48


def read_pep(path: Path, genes: set | None) -> dict[str, list[str]]:
    """gene id (no version) -> sequences; proteins with a stop codon dropped, as SATURN's clean_fasta."""
    out: dict[str, list[str]] = {}
    gid, seq = None, []

    def flush():
        if gid and seq:
            s = "".join(seq)
            if "*" not in s and (genes is None or gid in genes):
                out.setdefault(gid, []).append(s)

    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                flush()
                gid, seq = None, []
                for tok in line.split():
                    if tok.startswith("gene:"):
                        gid = tok[5:].split(".")[0]
            else:
                seq.append(line.strip())
    flush()
    return out


def canonical_proteins(species: str, genes: set) -> dict[str, str]:
    """gene -> the protein Compara used for the human-species homology, the one the identity column reads."""
    import pandas as pd

    from_dump = {"ptroglodytes": ROOT / "data/analysis/ensembl110/by_species/pan_troglodytes/Compara.110.protein_default.homologies.tsv.gz",
                 "cjacchus": ROOT / "data/analysis/ensembl110/by_species/callithrix_jacchus/Compara.110.protein_default.homologies.tsv.gz",
                 "ggallus": NEW / "Compara.110.protein_default.homologies.gallus_gallus.tsv.gz"}
    human = ROOT / "data/analysis/ensembl110/Compara.110.protein_default.homologies.tsv.gz"
    out: dict[str, str] = {}
    cols = ["gene_stable_id", "protein_stable_id", "homology_gene_stable_id", "homology_protein_stable_id"]
    for path in (from_dump[species], human):
        for chunk in pd.read_csv(path, sep="\t", usecols=cols, dtype=str, chunksize=2_000_000):
            a = chunk[chunk["gene_stable_id"].isin(genes)]
            out.update(zip(a["gene_stable_id"], a["protein_stable_id"]))
            b = chunk[chunk["homology_gene_stable_id"].isin(genes)]
            out.update(zip(b["homology_gene_stable_id"], b["homology_protein_stable_id"]))
    return out


def read_pep_ids(path: Path, wanted: set) -> dict[str, str]:
    """protein id (no version) -> sequence."""
    out: dict[str, str] = {}
    pid, seq = None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if pid in wanted and seq:
                    out[pid] = "".join(seq)
                pid, seq = line[1:].split()[0].split(".")[0], []
            else:
                seq.append(line.strip())
    if pid in wanted and seq:
        out[pid] = "".join(seq)
    return out


def plan(species: str, genes_file: Path, canonical: bool = False) -> None:
    genes = {g.strip() for g in genes_file.read_text().split() if g.strip()}
    if canonical:
        c = canonical_proteins(species, genes)
        seqs_by_id = read_pep_ids(PEP[species], set(c.values()))
        pep = {g: [seqs_by_id[p]] for g, p in c.items() if p in seqs_by_id and "*" not in seqs_by_id[p]}
        # a gene whose Compara protein is not in pep.all, or has a stop codon, falls back to all its proteins
        rest = genes - set(pep)
        if rest:
            pep.update(read_pep(PEP[species], rest))
    else:
        pep = read_pep(PEP[species], genes)
    seqs = sorted({s[:MAX_LEN] for v in pep.values() for s in v}, key=len)
    n_iso = sum(len(v) for v in pep.values())
    d = OUT / species
    d.mkdir(parents=True, exist_ok=True)
    # Round-robin over length: shards take about the same time.
    shards = [seqs[i::N_SHARDS] for i in range(N_SHARDS)]
    for i, s in enumerate(shards):
        (d / f"shard_{i:02d}.json").write_text(json.dumps(s))
    (d / "gene_to_seqs.json").write_text(json.dumps({g: sorted({x[:MAX_LEN] for x in v}) for g, v in pep.items()}))
    print(json.dumps(dict(species=species, genes_asked=len(genes), genes_found=len(pep), proteins=n_iso, unique_truncated=len(seqs),
                          residues=int(sum(map(len, seqs))), median_len=int(np.median([len(s) for s in seqs])))))


def load_model(threads: int):
    import esm
    import torch

    torch.set_num_threads(threads)
    # Weights-only unpickling stays on; the checkpoint's config needs argparse.Namespace on the allow list.
    torch.serialization.add_safe_globals([argparse.Namespace])
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(WEIGHTS))
    model.eval()
    return torch, model, alphabet.get_batch_converter()


def embed_one(torch, model, conv, seq: str) -> np.ndarray:
    _, _, toks = conv([("p", seq)])
    with torch.inference_mode():
        rep = model(toks, repr_layers=[LAYER], return_contacts=False)["representations"][LAYER]
    return rep[0, 1: len(seq) + 1].mean(0).numpy().astype(np.float32)


def shard(species: str, k: int, threads: int, model_parts=None) -> None:
    d = OUT / species
    seqs = json.loads((d / f"shard_{k:02d}.json").read_text())
    f = d / f"emb_{k:02d}.npz"
    done: dict[str, np.ndarray] = {}
    if f.exists():
        z = np.load(f, allow_pickle=False)
        done = dict(zip(z["seq"].tolist(), z["emb"]))
    todo = [s for s in seqs if s not in done]
    print(f"{species} shard {k}: {len(seqs)} sequences, {len(todo)} to do", flush=True)
    if not todo:
        return
    t0 = time.time()
    torch, model, conv = model_parts if model_parts is not None else load_model(threads)
    print(f"model loaded in {time.time() - t0:.0f} s", flush=True)
    t1, res = time.time(), 0

    def save():
        tmp = d / f"emb_{k:02d}.tmp.npz"
        np.savez(tmp, seq=np.array(list(done)), emb=np.stack(list(done.values())))
        tmp.replace(f)

    for i, s in enumerate(todo):
        done[s] = embed_one(torch, model, conv, s)
        res += len(s)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(todo)}: {res / (time.time() - t1):.1f} residues/s", flush=True)
            save()
    save()
    (d / f"DONE_{k:02d}").write_text("ok\n")


def pool(species: list[str], workers: int, threads: int) -> None:
    """One model in memory, forked workers (copy on write), so memory stays about one model per job. Workers take
    shards in order; a claim file marks a shard taken, and a shard with a DONE file is skipped."""
    import multiprocessing as mp
    import os
    import socket

    torch, model, conv = load_model(1)

    def worker():
        torch.set_num_threads(threads)
        for sp in species:
            d = OUT / sp
            for k in range(N_SHARDS):
                if (d / f"DONE_{k:02d}").exists():
                    continue
                claim = d / f"claim_{k:02d}"
                try:
                    fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                os.write(fd, f"{socket.gethostname()} {os.getpid()}".encode())
                os.close(fd)
                shard(sp, k, threads, (torch, model, conv))

    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=worker) for _ in range(workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()


def canon_check(threads: int, n: int = 30) -> None:
    """Canonical protein alone against the SATURN mean over isoforms, human genes: is one protein close enough?"""
    import pandas as pd
    import torch as T

    tab = pd.read_csv(ROOT / "data" / "processed" / "ortholog_table_testis.csv")
    sat = T.load(ROOT / "data/raw/saturn_esm2/protein_embeddings_export/ESM2/human_embedding.torch", map_location="cpu", weights_only=False)
    rng = np.random.default_rng(1)
    pick = tab.iloc[rng.choice(len(tab), 300, replace=False)]
    genes = set(pick["hsapiens_gene_id"])
    d = pd.read_csv(ROOT / "data/analysis/ensembl110/Compara.110.protein_default.homologies.tsv.gz", sep="\t",
                    usecols=["gene_stable_id", "protein_stable_id"], dtype=str, nrows=3_000_000).drop_duplicates("gene_stable_id")
    c = dict(zip(d["gene_stable_id"], d["protein_stable_id"]))
    want = {c[g] for g in genes if g in c}
    seqs = read_pep_ids(PEP["hsapiens"], want)
    allp = read_pep(PEP["hsapiens"], genes)
    torch, model, conv = load_model(threads)
    rows = []
    for _, r in pick.iterrows():
        g, sym = r["hsapiens_gene_id"], r["hsapiens_gene_name"]
        if g not in c or c[g] not in seqs or sym not in sat or len(allp.get(g, [])) < 2:
            continue
        v = embed_one(torch, model, conv, seqs[c[g]][:MAX_LEN])
        w = np.asarray(sat[sym], dtype=np.float32)
        rows.append(dict(symbol=sym, n_iso=len(allp[g]), length=len(seqs[c[g]]), cos=float(v @ w / (np.linalg.norm(v) * np.linalg.norm(w)))))
        print(rows[-1], flush=True)
        if len(rows) == n:
            break
    print(pd.DataFrame(rows).describe().to_string())


def combine(species: str) -> None:
    import torch

    d = OUT / species
    emb: dict[str, np.ndarray] = {}
    for f in sorted(d.glob("emb_*.npz")):
        z = np.load(f, allow_pickle=False)
        emb.update(zip(z["seq"].tolist(), z["emb"]))
    g2s = json.loads((d / "gene_to_seqs.json").read_text())
    out, missing = {}, 0
    for g, ss in g2s.items():
        have = [emb[s] for s in ss if s in emb]
        if len(have) != len(ss):
            missing += 1
            continue
        out[g] = torch.tensor(np.mean(have, axis=0))
    (OUT / "ESM2").mkdir(exist_ok=True)
    torch.save(out, OUT / "ESM2" / f"{COMMON[species]}_embedding.torch")
    print(json.dumps(dict(species=species, genes=len(out), genes_incomplete=missing, dim=int(next(iter(out.values())).shape[0]))))


def check(threads: int, n: int = 12) -> None:
    """Recompute a sample of human genes and compare with the SATURN file, keyed by symbol; prints speed."""
    import pandas as pd
    import torch as T

    tab = pd.read_csv(ROOT / "data" / "processed" / "ortholog_table_testis.csv")
    sat = T.load(ROOT / "data/raw/saturn_esm2/protein_embeddings_export/ESM2/human_embedding.torch", map_location="cpu", weights_only=False)
    rng = np.random.default_rng(0)
    pick = tab.iloc[rng.choice(len(tab), 400, replace=False)]
    pep = read_pep(PEP["hsapiens"], set(pick["hsapiens_gene_id"]))
    rows = []
    torch, model, conv = load_model(threads)
    for _, r in pick.iterrows():
        g, sym = r["hsapiens_gene_id"], r["hsapiens_gene_name"]
        if g not in pep or sym not in sat or len(pep[g]) > 3:
            continue
        ss = [s[:MAX_LEN] for s in pep[g]]
        t0 = time.time()
        v = np.mean([embed_one(torch, model, conv, s) for s in ss], axis=0)
        w = np.asarray(sat[sym], dtype=np.float32)
        cos = float(v @ w / (np.linalg.norm(v) * np.linalg.norm(w)))
        rows.append(dict(symbol=sym, n_prot=len(ss), residues=sum(map(len, ss)), sec=time.time() - t0, cos=cos,
                         ))
        print(rows[-1], flush=True)
        if len(rows) == n:
            break
    t = pd.DataFrame(rows)
    print(t.describe().to_string())
    print("residues/s", t.residues.sum() / t.sec.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "shard", "combine", "check", "pool", "canoncheck"])
    ap.add_argument("--canonical", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--species")
    ap.add_argument("--genes", type=Path)
    ap.add_argument("--index", type=int)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a.species, a.genes, a.canonical)
    elif a.cmd == "shard":
        shard(a.species, a.index, a.threads)
    elif a.cmd == "pool":
        pool(a.species.split(","), a.workers, a.threads)
    elif a.cmd == "canoncheck":
        canon_check(a.threads)
    elif a.cmd == "combine":
        combine(a.species)
    else:
        check(a.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
