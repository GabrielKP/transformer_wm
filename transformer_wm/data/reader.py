"""
File to read sentences.json and vignettes.json, construct
and tokenize, and prepare them for model usage.

@date: 01/16/2022
@author: gabriel.kressin@fu-berlin.de
@comment: this file is a bit overloaded should be split into two.
"""

import json
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from transformer_wm import get_logger

logger = get_logger(__name__)


def load_sequences(path: str) -> List[Dict]:
    with open(path) as f_in:
        sequences = json.load(f_in)

    return sequences["sequences"]


def load_models(path: str = "data/models.txt") -> List[str]:
    """Loads models from model file."""
    models: List[str] = []
    with open(path, "r") as f_in:
        for line in f_in.readlines():
            if line.startswith("#"):
                continue
            if line.endswith("\n"):
                line = line[:-1]
            models.append(line)
    logger.info(f"Modelfile: {models}")
    return models


# def save_synonyms(path: str, synonyms: Dict[str, List[str]]) -> None:
#     """Save synonym fict to txt file.

#     Lines starting with '#' are ignored.
#     Lines starting with '--' indicate original word.
#     Lines below indicate synonyms for that word.
#     """
#     logger.info(f"Saving synonyms to {path}")
#     with open(path, "w") as f_out:
#         for word, word_synonyms in synonyms.items():
#             f_out.write(f"-- {word}\n")
#             for word_synonym in word_synonyms:
#                 f_out.write(f"{word_synonym}\n")


def read_sequences(
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    return_marked_word_position: bool = False,
) -> Dict[str, List[Union[List, torch.Tensor]]]:
    """
    Function to create all input data.

    Parameter
    ---------
    file_path : ``str``
        Path of file containing sequences in json:
        {
            "sequences":[
                {"sequenceID":, ..., "sentenceID": ..., "vignetteID": ...,
                "preface": ..., "first": ..., "intervention": ...,
                "prompt": ..., "second": ...},...
            ]
        }
    tokenizer : ``GPT2TokenizerFast | RNNTokenizer``
        Tokenizer to tokenize strings and convert into ids.
    subtokens : ``bool``, optional (default=``True``)
        Whether Tokenizer uses subtokens.
    add_special_tokens_separately : ``bool``, optional (default=``False``)
        If set to ``True`` the function will add the tokenizers special
        before_of_string and end_of_string tokens separately to the input strings.
        This prevents the tokenizer from encoding them wrongly.

    Returns
    -------
    data : ``Dict``
        data = {
            "sequenceID": [],   # unique id for each surprisal measurement
            "experimentID": [], # id for sequences that belong to the same "experiment" e.g. the matching repeat and control
            "sentenceID": [],   # id for sentence_first and sentence_second
            "vignetteID": [],   # id for preface, intervention and prompt
            "one_hots": [],     # onehot sequence
            "sectionIDs": [],   # indices of current section within sequence
            "positionIDs": [],  # indices where token is within its section
            "subtokIDs": [],    # indices of original token within its section
        }
        one_hots : List[torch.Tensor[1,seq_len]]

    Important: maps "sequenceID" to every created datapoint.
    """

    # Load data
    logger.info(f"Loading from {file_path}.")
    sequences = load_sequences(file_path)

    """
    A sequence is the entire input the model receives.
    It can be split into 4 different sections:
    [preface, first_sentence, intervention+prompt, second_sentence]
    """

    data = {
        "sequenceID": [],  # unique id for each surprisal measurement
        "experimentID": [],  # id for sequences that belong to the same "experiment" e.g. the matching repeat and control
        "sentenceID": [],  # id for sentence_first, sentence_second combination
        "vignetteID": [],  # id for preface, intervention and prompt
        "one_hots": [],  # onehot sequence
        "sectionIDs": [],  # indices of current section within sequence (see sections above)
        "positionIDs": [],  # indices where token is within its section
        "subtokIDs": [],  # indices of original token within its section
        "tokens": [],  # tokens as str
    }
    if return_marked_word_position:
        data["marked_word_token_indices"] = []

    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<bos>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<eos>"})

    # TODO: log output of this function
    for sequence_dict in sequences:

        # Because of this: https://github.com/huggingface/transformers/issues/3311
        # add special tokens manually.
        sec0 = f"{tokenizer.bos_token} {sequence_dict['preface']}".strip()
        sec1 = sequence_dict["first"].strip()
        sec2 = f"{sequence_dict['intervention']} {sequence_dict['prompt']}".strip()
        sec3 = f"{sequence_dict['second']}{tokenizer.eos_token}".strip()
        sequence_text = sec0 + sec1 + sec2 + sec3

        encodings = tokenizer(
            sequence_text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        section_ids = []
        position_ids = []
        subtok_ids = []

        section_lengths = np.cumsum(
            (len(sec0), len(sec1), len(sec2), len(sec3))
        ).tolist()  # lengths are in characters.
        current_section_id = 0
        current_position_id = 0
        current_subtok_id = -1
        previous_word_id = None
        # Iterate through all input ids to create position and subtoken markers.
        for index in range(len(encodings["input_ids"][0])):

            current_word_id = encodings.token_to_word(index)
            if current_word_id != previous_word_id:
                current_subtok_id += 1
            previous_word_id = current_word_id

            section_ids.append(current_section_id)
            position_ids.append(current_position_id)
            subtok_ids.append(current_subtok_id)

            current_position_id += 1

            # At end of section, reset counters
            if encodings.token_to_chars(index).end == section_lengths[current_section_id]:
                current_section_id += 1
                current_position_id = 0
                current_subtok_id = 0

        if return_marked_word_position:
            if sequence_dict.get("marked_word") is None:
                raise RuntimeError(
                    "Sequence requires a marked word if word position is to be recorded."
                )
            sec3_start, sec3_end = sequence_dict["marked_word"]
            len_until_3 = len(sec0 + sec1 + sec2)
            start_index_char = sec3_start + len_until_3  # in original word sequence
            end_index_char = sec3_end + len_until_3
            start_index_token = encodings.char_to_token(
                start_index_char
            )  # in tokenized id sequence
            end_index_token = encodings.char_to_token(
                end_index_char - 1
            )  # subtract one to be within the marked word. (end_index_char is one char further.)

        data["experimentID"].append(sequence_dict["experimentID"])
        data["sequenceID"].append(sequence_dict["sequenceID"])
        data["sentenceID"].append(sequence_dict["sentenceID"])
        data["vignetteID"].append(sequence_dict["vignetteID"])
        data["one_hots"].append(encodings["input_ids"])
        data["sectionIDs"].append(section_ids)
        data["positionIDs"].append(position_ids)
        data["subtokIDs"].append(subtok_ids)
        data["tokens"].append([tokenizer.decode(one_hot) for one_hot in encodings["input_ids"][0]])
        if return_marked_word_position:
            data["marked_word_token_indices"].append((start_index_token, end_index_token))

    return data
