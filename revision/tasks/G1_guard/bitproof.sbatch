#!/usr/bin/env bash
#SBATCH --job-name=g1_bit
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --exclude=cpusrv114
#SBATCH --output=results/revision/G1/logs/%x_%j.out
# G1 step 2. usage: sbatch bitproof.sbatch <before tree> <after tree>. All fits in this one job, so on one node.
set -uo pipefail
export PATH=/home/icb/kemal.inecik/tools/apps/mamba/bin:$PATH
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
before=$1; after=$2
here=$PWD/revision/tasks/G1_guard
data=$PWD/data/processed
hostname; lscpu | grep "Model name"
fit() { # tree name threads args
  local tree=$1 name=$2 th=$3; shift 3
  OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th mamba run --no-capture-output -n chimera_env python "$here/bitproof_run.py" "$tree" --run-name "$name" --accelerator cpu --max-epochs 3 --seed 42 "$@" 2>&1 | tail -3
}
L="--config A --adv-weight 200 --free-bits 1.0 --adata $data/camex_human_mouse.h5ad"
T="--config A --adv-weight 200 --free-bits 1.0 --celltype-conditional-mmd --celltype-key cell_ontology_class --prior-clamp-max 0.85 --adata $data/camex_testis_human_mouse.h5ad"
for g in "liv-t1 1 $L" "test-t1 1 $T" "liv-t8 8 $L"; do
  set -- $g; name=g1bit-$1; th=$2; shift 2
  fit "$before" "$name-a" "$th" "$@"
  fit "$before" "$name-b" "$th" "$@"
  fit "$after"  "$name-a" "$th" "$@"
done
mamba run --no-capture-output -n chimera_env python "$here/bitproof_compare.py" "$before" "$after" results/revision/G1/bitproof.csv
