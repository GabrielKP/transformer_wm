# Repository: Working Memory for Repeated Sentences in Transformer Language Models

This is the accompanying code repository for the the bachelor's thesis ['Working Memory for Repeated Sentences in Transformer Language Models'](pdf/kressin-2022-working_memory_for_repeated_sentences_in_transformer_language_models.pdf) by Gabriel Kressin Palacios.

> **_NOTE:_** The original submitted thesis had the name 'Working Memory for Sentences in Transformer Language Models' which was unintentional and misleading. See [here](README.md#submitted-version).

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

```bash
# create input data
python scripts/run.py create

# run all
python scripts/run.py --batch_size 64

# run gpt2
python scripts/run.py --batch_size 64 --model_name gpt2
```

# Submitted version

The originally submitted thesis can be found in [pdf/kressin_original.pdf](pdf/kressin_original.pdf)

Changelog of the updated version to the original:
* Added 'Repeated' in title.
* Corrected matriculation number.
* Corrected spelling in cover page.
