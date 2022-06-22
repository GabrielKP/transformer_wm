#!/bin/bash
srun -K \
  --job-name=run_lm_mem \
  --time=10:00:00 \
  --mem 20gb \
  --partition=gpuk80 \
  --gres=gpu:1 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=6 \
  $*
