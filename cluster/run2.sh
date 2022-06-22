#!/bin/bash
WORKDIR="`pwd`"
ml cuda/9.2
ml anaconda
conda activate $HOME/code/conda_envs/neural_lm_mem
pip install .
$*
