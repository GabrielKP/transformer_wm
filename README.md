# Repository: Working Memory for Repeated Sentences in Transformer Language Models

This is the accompanying code repository for the the bachelor's thesis ['Working Memory for Repeated Sentences in Transformer Language Models'](pdf/kressin-2022-working_memory_for_repeated_sentences_in_transformer_language_models.pdf) by Gabriel Kressin Palacios.

> **_NOTE:_** The original submitted thesis had the name 'Working Memory for Sentences in Transformer Language Models'. See [here](README.md#submitted-version).

# Setup

```bash
# clone project
git clone https://github.com/GabrielKP/transformer_wm
cd transformer_wm

# [OPTIONAL] create conda environment
conda create -n transformer_wm python=3.9
conda activate transformer_wm

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

The [`scripts/run.py`](scripts/run.py) is the main interface for the experiments. It provides you with different commands:
```bash
usage: run.py [-h] {thesis,create,run,plot} ...

optional arguments:
  -h, --help            show this help message and exit
```
You can combine any subcommand with -h, for example `run.py run -h`.

```bash

# Compute data, replicate all figures, plots and table data from the thesis
python scripts/run.py thesis

# Run experiments, standard batch size is 24, this should allow for running gpt-neo-1.6 on a RTX 3080. Adjust properly for your machine.
# Note, that you can get the output data online, to avoid recomuting, see section below
python scripts/run.py run --batch_size 64

# Run only one model
python scripts/run.py --batch_size 64 --model_name gpt2

# Recreate the input data
python scripts/run.py create

# Recreate plots
python scripts/run.py plot
```

You can invoke all scripts manually, they are located in:
* data creation: [transformer_wm/data](transformer_wm/data)
* running the models: [transformer_wm/surprisal.py](transformer_wm/surprisal.py)
* plotting/analysing: [transformer_wm/analysis](transformer_wm/analysis)

# Download output data

If you do not want to recompute the output data, you can download it separately, and extract the `output` folder into the `data` directory.

```bash
# Go to data dir
cd data

# Download
wget https://raw.githubusercontent.com/GabrielKP/transformer_wm_output/main/output.zip

# Unzip
unzip output.zip
```

# Submitted version

The originally submitted thesis can be found in [pdf/kressin_original.pdf](pdf/kressin_original.pdf)

Changelog of the updated version to the original:
* Added 'Repeated' in title.
* Corrected matriculation number.
* Corrected spelling in cover page.
