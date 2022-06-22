# Sentential

# Setup

Install dependencies

```bash
# clone project
git clone https://github.com/GabrielKP/lm_mem
cd lm_mem

# [OPTIONAL] create conda environment
conda create -n lm_mem python=3.9
conda activate lm_mem

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install package
pip install .

# for developing instead, install requirements.xt
pip install -r requirements.txt

# install pre-commit
pre-commit install

# run pre-commit against all files once
pre-commit run --all-files
```


# Run

## At home

```bash
# create input data
python scripts/run.py create

# run all
python scripts/run.py --batch_size 64

# run gpt2
python scripts/run.py --batch_size 64 --model_name gpt2
```

## Cluster

### Marcc

Connect to [marcc](https://www.marcc.jhu.edu/getting-started/connecting-to-marcc/) and run experiment.
* Enable multiplexing
```bash
# into .ssh/config
Host marcc
    Hostname login.marcc.jhu.edu
    User userid@jhu.edu
    ControlMaster auto
    ControlPath ~/.ssh/control:%h:%p:%r
```

* Activate multiplexing
```bash
# only once per session
ssh -fNM marcc -l userid@jhu.edu
```

* Connect to marcc
```bash
ssh marcc
```

* Close multiplexing (not required, but if you do not want it for some reason)
```bash
ssh -O stop marcc
```

* Setup conda environment
```bash
# Jokes on you, I forgot  how I did that
```

* Run experiment
```bash
# cd into project root dir

# Open screen
screen -S screenname

# reconnect to screen
screen -rd screenname

# for interactive output run script:
# gpt2
./cluster/usrun.sh cluster/run.sh --device cuda --batch_size 64 --model_name gpt2
# gpt-neo
./cluster/usrun.sh cluster/run.sh --device cuda --batch_size 64 --model_name 'EleutherAI/gpt-neo-1.3B'
```

## Baseline experiment: nonce data

Nonce data (small)
```bash

```

## RNN

Using the RNN requires downloaded trained RNN models from here:
https://doi.org/10.5281/zenodo.3559340.
Place the extracted models into a folder `rnn_models` in the root directory.

Alternatively use following commands:
```bash
mkdir rnn_models
cd rnn_models
wget https://zenodo.org/record/3559340/files/LSTM_40m.tar.gz?download=1 -O LSTM_40m.tar.gz
tar -xvzf LSTM_40.tar.gz
cd ../
python rnn/model2statedict.py rnn_models/LSTM_400_40m_a_10-d0.2.pt
# or
python rnn/model2statedict.py rnn_models
```

```
python surprisal.py \
    data/nonce/sentences_small.json \
    data/vignettes_small.json \
    data/rnn/nonce_surprisal_small.csv \
    --debug \
    --device cpu \
    --model_name rnn \
    --rnn_config_file data/rnn/config.json \
    --rnn_weights_file ./../rnn_models/LSTM_400_40m_a_10-d0.2_statedict.pt
```

```
python surprisal.py \
    data/nonce/sentences_correct.json \
    data/vignettes_small.json \
    data/rnn/nonce_surprisal_correct.csv \
    --debug \
    --device cuda \
    --model_name rnn \
    --rnn_config_file data/rnn/config.json \
    --rnn_weights_file ./../rnn_models/LSTM_400_40m_a_10-d0.2_statedict.pt
```
