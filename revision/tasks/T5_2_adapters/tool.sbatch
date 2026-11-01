#!/usr/bin/env bash
#SBATCH --job-name=chimera_T5_tool
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=23:50:00
# Usage (from chimera/): sbatch --output=results/revision/T5_2/logs/%x_%j.out \
#   revision/tasks/T5_2_adapters/tool.sbatch <saturn|camex> <stem> <labels|-> <seed> <work> <dest>
source revision/tasks/T5_2_adapters/_tool_body.sh
run_tool "$@"
