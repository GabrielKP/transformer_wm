#!/bin/bash
WORKDIR="`pwd`"
ml cuda/9.2
ml anaconda
conda activate $HOME/code/conda_envs/neural_lm_mem
pip install .

# options are
# --device ["cuda", "cpu"] --batch_size int --model_name <model_name>
python $WORKDIR/scripts/run.py \
    run \
    $*
