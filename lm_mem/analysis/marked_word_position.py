from typing import List

import numpy as np
from transformers import AutoTokenizer

from lm_mem.data.reader import read_sequences


def print_marked_word_position(
    file_path: str,
    tokenizer_name: str,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    inputs_ = read_sequences(
        file_path=file_path,
        tokenizer=tokenizer,
        return_marked_word_position=True,
    )
    positions: List[int] = []
    for marked_token, sectionIDs in zip(
        inputs_["marked_word_token_indices"], inputs_["sectionIDs"]
    ):
        marker_start = marked_token[0]
        beginning_secID = sectionIDs[marker_start]
        pos = 0
        while beginning_secID == sectionIDs[marker_start - pos]:
            pos += 1
        positions.append(pos)
    print(f"Items : {len(positions)}")
    print(f"Mean  : {np.mean(positions)}")
    print(f"Std   : {np.std(positions)}")
    print(f"Median: {np.median(positions)}")


if __name__ == "__main__":
    print("Nouns")
    print_marked_word_position("data/input/word_swap/seq1s_rpt_noun.json", "gpt2")
    print("Verbs")
    print_marked_word_position("data/input/word_swap/seq1s_rpt_verb.json", "gpt2")
