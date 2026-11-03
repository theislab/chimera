#!/usr/bin/env bash
#SBATCH --job-name=b2t_blast
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=23:50:00
# Reciprocal blastp of the Ensembl 110 human and mouse proteomes with the tool's own map_genes.sh,
# then the gene-level homology graph that every SAMap run reuses. Once per species pair.
# Usage, from chimera/: sbatch --output=results/revision/B2/tools_pilot/_logs/%x_%j.out revision/tasks/B2_benchmark/tools/blast_samap.sbatch
set -euo pipefail
export PATH=/home/icb/kemal.inecik/tools/apps/mamba/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
TOOL=$REPO/../external_tools/SAMap
D=$REPO/data/revision/B2/tools/samap
cd "$D"
t0=$SECONDS
if [[ -f maps/hsmm/SHA256SUMS ]] && sha256sum -c --quiet maps/hsmm/SHA256SUMS; then
  echo "BLAST tables present and match SHA256SUMS: reused"
else
mamba run --no-capture-output -n samap_env bash "$TOOL/map_genes.sh" \
  --tr1 fasta/Homo_sapiens.GRCh38.pep.all.fa --t1 prot --n1 hs \
  --tr2 fasta/Mus_musculus.GRCm39.pep.all.fa --t2 prot --n2 mm --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "blast_elapsed_s=$((SECONDS - t0))"
sha256sum maps/hsmm/*.txt | tee maps/hsmm/SHA256SUMS
fi
cd "$REPO"
mamba run --no-capture-output -n samap_env python revision/tasks/B2_benchmark/tools/build_samap_graph.py
echo "total_elapsed_s=$((SECONDS - t0))"
