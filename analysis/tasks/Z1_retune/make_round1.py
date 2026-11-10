"""Write the manifest of round 1 (Van Zyl H+Pig, H+M. fascicularis). Rows differ from the K = 2 rule base only."""
from pathlib import Path

BASE = ("--config A --max-epochs 150 --adata data/analysis/van_zyl/van_zyl_{c}.h5ad --seed {s} --batch-key source_batch "
        "--celltype-key cell_ontology_class --adv-weight 100 --lambda-anchor {a} --lambda-mmd 50 --free-bits 0.5 "
        "--celltype-conditional-mmd --prior-clamp-max 0.85{x}")
ROWS = {"a": (0.003, "", (43, 44)), "b": (0.003, " --warmup-kl-epochs 5", (42, 43)),
        "c": (0.0045, "", (42, 43)), "d": (0.0045, " --warmup-kl-epochs 5", (42, 43))}
out = ["run_name\targs"]
for c in ("hpig", "hmfas"):
    for r, (a, x, seeds) in ROWS.items():
        for s in seeds:
            out.append(f"rev-z1-{c}-{r}-s{s}\t" + BASE.format(c=c, s=s, a=a, x=x))
Path(__file__).with_name("round1_vz_pairs.tsv").write_text("\n".join(out) + "\n")
print(len(out) - 1)
