#!/usr/bin/env bash
#SBATCH --job-name=b2i
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --time=23:50:00
# Usage (from chimera/): sbatch -c 16 --mem=64G --nice=5 --output=results/revision/B2/logs/%x_%j.out revision/tasks/B2_benchmark/integration/tools.sbatch <cohort> <SATURN|SATURN-Leiden|CAMEX>
source revision/slurm/_body.sh
export DGLBACKEND=pytorch
c="$1"; m="$2"; R=$PWD/data/revision/B2/integration_inputs/$c; L=$PWD/results/revision/B2/integration/latents/$c/$m
if [[ ! -f $R/saturn/run.csv ]]; then mamba run --no-capture-output -n chimera_env python -m revision.tasks.B2_benchmark.integration.prep "$c" || exit 1; fi
unset PYTHONPATH
case "$m" in
  SATURN) mamba run --no-capture-output -n saturn_env python revision/tasks/B2_benchmark/tools/run_saturn_b2.py --run "integ_${c}_saturn" --seed 42 --root "$R" --latent-out "$L" ;;
  SATURN-Leiden) mamba run --no-capture-output -n saturn_env python revision/tasks/B2_benchmark/tools/run_saturn_b2.py --run "integ_${c}_saturnl" --seed 42 --root "$R" --label-col leiden_within --latent-out "$L" ;;
  CAMEX) mamba run --no-capture-output -n camex_env python revision/tasks/B2_benchmark/tools/run_camex_b2.py --run "integ_${c}_camex" --seed 42 --root "$R" --latent-out "$L" ;;
esac
