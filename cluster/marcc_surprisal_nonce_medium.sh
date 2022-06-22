#!/bin/bash
#SBATCH --job-name=surprisal_test
#SBATCH --time=10:00:00
#SBATCH --mem 5gb
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=all
#SBATCH --mail-user=gabriel.kressin@fu-berlin.de
#SBATCH --output=/home-2/gkressi1@jhu.edu/experiments/nonce_surprisal_small/output.log
#SBATCH --error=/home-2/gkressi1@jhu.edu/experiments/nonce_surprisal_small/output.err

ROOT=$HOME/code/sentences

ml cuda/9.2
ml anaconda
conda activate $HOME/code/conda_envs/neural_lm_mem

python $ROOT/surprisal.py \
    $ROOT/data/nonce/sentences_small.json \
    $ROOT/data/vignettes_small.json \
    $HOME/experiments/nonce_surprisal_medium/output.csv \
    --debug \
    --device cuda
