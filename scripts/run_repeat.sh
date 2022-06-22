#!/bin/bash

python lm_mem/surprisal.py data/input/repeat/repeats.json data/output/repeat/repeats_gtp2.csv -f
python lm_mem/surprisal.py data/input/repeat/controls.json data/output/repeat/controls_gtp2.csv -f
