#!/usr/bin/env bash
#SBATCH --job-name=t73_mash
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=12:00:00
# T7.3 and T7.8, CPU only, revision_de_env.
# Usage: mkdir -p results/revision/T7_3/logs && sbatch --output=results/revision/T7_3/logs/%x_%j.out \
#          revision/tasks/T7_3_mash/run.sbatch <stage>
# Stages: fits (sensitivity, permuted, bulk, batch control), analysis (H-D1, H-D1b, H-D4), t78.
# guilliams.py runs first.
set -uo pipefail
export PATH=/home/icb/kemal.inecik/tools/apps/mamba/bin:$PATH
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${CHIMERA_ROOT:-/home/icb/kemal.inecik/lustre_workspace/master_chimera/chimera}" || exit 2
export PYTHONPATH=$PWD
py() { mamba run --no-capture-output -n revision_de_env python "$@"; }
case "$1" in
  fits)
    py revision/tasks/T7_3_mash/de.py --group primary --workers 2 || exit 3
    py revision/tasks/T7_3_mash/de.py --group sens --workers 30 || exit 3
    py revision/tasks/T7_3_mash/de.py --group source --workers 3 || exit 3
    py revision/tasks/T7_3_mash/de.py --group perm --workers 30 || exit 3
    py revision/tasks/T7_3_mash/de.py --group bulk --n-cpus 24 || exit 3
    py revision/tasks/T7_3_mash/de.py --group control --n-cpus 8 || exit 3
    ;;
  analysis)
    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    py revision/tasks/T7_3_mash/hd4.py --jobs 7 || exit 3
    py revision/tasks/T7_3_mash/hd1.py --jobs 4 || exit 3
    ;;
  t78)
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    py revision/tasks/T7_3_mash/t78_second_contrast.py || exit 3
    ;;
esac
