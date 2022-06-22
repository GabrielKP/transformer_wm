#!/bin/bash
#SBATCH --job-name=run_lm_mem
#SBATCH --time=10:00:00
#SBATCH --mem 12gb
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=all
#SBATCH --mail-user=gabriel.kressin@fu-berlin.de
#SBATCH --output=/home-2/gkressi1@jhu.edu/experiments/%x_%j.out
$*
