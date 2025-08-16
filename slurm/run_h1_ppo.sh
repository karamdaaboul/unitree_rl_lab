#!/bin/bash
#SBATCH --job-name=h1_velocity_train
#SBATCH --partition=H100-MIG
#SBATCH --gpus=3g.40gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

# Optional: Create logs directory if it doesn't exist
mkdir -p logs

# Load modules if required (uncomment if your cluster uses modules)
# module load cuda/12.1

# Activate your conda environment (replace with your own conda initialization if needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Run your training script
python scripts/rsl_rl/train.py --headless --task Unitree-H1-Velocity --logger wandb