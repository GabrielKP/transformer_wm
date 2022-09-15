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
