#!/bin/bash
#SBATCH --job-name=dmrg_jm4
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH -p xhhctdnormal
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 96:00:00


# shellcheck disable=SC1083
cd {work_dir} || exit
module purge
# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate toolchain_env

uv run run_task_jm4.py