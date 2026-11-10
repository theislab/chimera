#!/usr/bin/env bash
#SBATCH --job-name=b1feed
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=23:50:00
# Runs feeder.sh as a job, so it outlives any login shell. Usage (from chimera/): sbatch -o results/analysis/B1/logs/%x_%j.out \
#   analysis/tasks/B1_ground_truth/feeder.sbatch <CPU_CAP> <MEM_CAP_GB>
cd "${CHIMERA_ROOT:-/home/icb/kemal.inecik/lustre_workspace/master_chimera/chimera}" || exit 2
bash analysis/tasks/B1_ground_truth/feeder.sh "$1" "$2"
