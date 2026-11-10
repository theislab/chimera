#!/usr/bin/env bash
#SBATCH --job-name=b1trackd
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=23:50:00
# One timed job of jobs.tsv on 16 CPUs. Usage (from chimera/): sbatch --array=0-N --output=results/analysis/B1/logs/%x_%A_%a.out .../job.sbatch
source analysis/tasks/T5_2_adapters/_tool_body.sh
export PYTHONPATH=$PWD
M=analysis/tasks/B1_ground_truth/track_d/jobs.tsv
I=$SLURM_ARRAY_TASK_ID
get() { python3 -c "import csv;print(list(csv.DictReader(open('$M'),delimiter='\t'))[$I]['$1'])"; }
job=$(get job); method=$(get method); cohort=$(get cohort); object=$(get object); h5ad=$(get h5ad); recipe=$(get recipe)
OUT=results/analysis/B1/track_d; mkdir -p "$OUT"
T=$OUT/$job.time.txt
[[ -f "$T" ]] && { echo "done $job"; exit 0; }
TOOLPY=analysis/tasks/B1_ground_truth/track_d/tool_input.py
STEM=$(python3 -c "print({'liv_hmouse':'camex_human_mouse','test_hmouse':'camex_testis_human_mouse','kidney':'kidney_healthy_train'}.get('$cohort',''))")
W=data/analysis/B1/trackD/work/$job
rm -rf "$W"; mkdir -p "$W"
case $method in
  chimera)
    python3 analysis/tasks/B1_ground_truth/track_d/timed.py "$T" mamba run --no-capture-output -n chimera_env python training/run_experiment.py --run-name "b1trackd-$job" --accelerator cpu $recipe --adata "$h5ad" --seed 42
    rm -rf "training/runs/b1trackd-$job/model" "training/runs/b1trackd-$job/lightning" "training/runs/b1trackd-$job/alpha_snapshots" ;;
  direct)
    python3 analysis/tasks/B1_ground_truth/track_d/timed.py "$T" mamba run --no-capture-output -n chimera_env python -m analysis.tasks.B1_ground_truth.track_d.direct_timed "$job" "$h5ad" "$cohort" ;;
  saturn)
    mamba run --no-capture-output -n saturn_env python $TOOLPY saturn "$cohort" "$object" "$W/in"
    python3 analysis/tasks/B1_ground_truth/track_d/timed.py "$T" mamba run --no-capture-output -n saturn_env python analysis/tasks/T5_2_adapters/run_saturn.py --stem "$STEM" --labels cell_type --seed 42 \
      --work "$W/work" --dest "$W/final" --inputs "$W/in" --hv-genes 40000 --method-name saturn_1to1
    rm -rf "$W" ;;
  camex)
    mamba run --no-capture-output -n camex_env python $TOOLPY camex "$cohort" "$object" "$W/in"
    python3 analysis/tasks/B1_ground_truth/track_d/timed.py "$T" mamba run --no-capture-output -n camex_env python analysis/tasks/T5_2_adapters/run_camex.py --stem "$STEM" --seed 42 \
      --work "$W/work" --dest "$W/final" --inputs "$W/in"
    rm -rf "$W" ;;
esac
