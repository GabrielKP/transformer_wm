import argparse
from functools import partial
from typing import Dict, List

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from lm_mem import get_logger

log = get_logger(__name__)


def _encode(tokenizer: PreTrainedTokenizer, text: str) -> BatchEncoding:
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )


def _collate(tokenizer: PreTrainedTokenizer, samples: List[Dict[str, str]]) -> BatchEncoding:
    batch = {key: [input_[key] for input_ in samples] for key in samples[0].keys()}
    return tokenizer(
        batch["text"],
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )


def ppl(model_name: str, batch_size: int):
    log.info(f"PPL COMPUTATION FOR {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Set device to {device}")
    log.info(f"Batch size: {batch_size}")

    log.info("Load Dataset")
    dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", split="test")

    log.info("Init Tokenizer")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    log.info("Init Model")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name
    )
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    model.to(device)

    # log.info("Tokenize dataset")
    # encode = partial(_encode, tokenizer)
    # dataset = dataset.map(encode, input_columns="text", remove_columns="text", batched=True)
    # dataset = dataset.with_format("torch")

    log.info("Get Perplexity")
    loss_func = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    nlls: List[float] = []

    with torch.no_grad():
        # sampler = BatchSampler(RandomSampler(dataset), batch_size=batch_size, drop_last=False)
        collate = partial(_collate, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
        for batch in tqdm(dataloader, desc="Computing perlexity"):
            # only need input ids
            past_key_values = None
            input_ids: torch.Tensor = batch["input_ids"]
            for idx in range(0, input_ids.size(1) - 1):
                selected_input_id = input_ids[:, idx : idx + 1].to(device)
                selected_target_id = input_ids[:, idx + 1 : idx + 2].to(device)

                outputs = model(
                    input_ids=selected_input_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                # del selected_input_id

                logits = outputs.logits
                del outputs
                loss: torch.Tensor = loss_func(
                    logits.view(-1, logits.size(-1)), selected_target_id.view(-1)
                )
                del selected_target_id
                del logits
                nlls.append(loss.detach().cpu().item())
                del loss

        log.info(f"Final ppl: {sum(nlls) / len(nlls)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="surprisal_gpt2.py runs perplexity experiment")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Model type on which to run the experiment.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch Size.",
    )
    parser.add_argument("-f", action="store_true")
    args = parser.parse_args()
    ppl(args.model_name, args.batch_size)
