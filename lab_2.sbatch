#!/bin/bash
#SBATCH --job-name=lab_2
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=lab_2_%j.out
#SBATCH --error=lab_2_%j.err

cd "$HOME/deep_learning/src/lab_2"
source "$HOME/deep_learning/env_dl/bin/activate"

echo "=== JOB INFORMATION ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPU allocation: $SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "=== PART I ==="
python part1_memory.py

echo "=== PART II ==="
python part2_precision.py

echo "=== PART III ==="
python part3_timing.py

echo "=== FINISHED ==="


