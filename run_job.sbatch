#!/bin/bash
#SBATCH --job-name=auto-qrels
#SBATCH --account=liangmingpan
#SBATCH --partition=gpu_standard              
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --time=100:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


module load python/3.11/3.11.4
module load cuda12/12.4.1             
module load cmake     

chmod +x run_agent_hpc.sh
sh run_agent_hpc.sh