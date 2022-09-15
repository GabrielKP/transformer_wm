"""Creates shuffled experiment input data
"""

import argparse
import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from nltk.tokenize import word_tokenize
from numpy.random import default_rng

from transformer_wm import get_logger
from transformer_wm.data.nonce import get_correct_nonce_combinations
from transformer_wm.data.utils import (
    compute_hash,
    read_vignettes,
    save_multiple_to_json,
    save_to_json,
)

logger = get_logger(__name__)


def global_shuffle(sentences: List[str]) -> List[str]:
    rng = default_rng()
    shuffles = []
    for sentence in sentences:
        # 1. Tokenize
        tokenized = word_tokenize(sentence)
        # 2. shuffle
        shuffle = rng.permutation(np.array(tokenized, dtype=np.object_))
        # 3. join
        joined = " ".join(shuffle.tolist())
        shuffles.append(joined)

    return shuffles


def choose_random(
    choices: List[Tuple[str, str]],
    invalid_choice: List[Tuple[str, str]],
    length_matched: bool = False,
) -> Tuple[str, str]:
    invalid_uid, invalid_text = invalid_choice
    length_text = len(invalid_text.split())
    ctrl_uid, ctrl_text = choices[random.randint(0, len(choices) - 1)]
    n_trials = 100
    for _ in range(n_trials):
        if ctrl_uid != invalid_uid:
            if not length_matched or abs(len(ctrl_text.split()) - length_text) < 7:
                return ctrl_uid, ctrl_text
        ctrl_uid, ctrl_text = choices[random.randint(0, len(choices) - 1)]
    raise RuntimeError(
        f"Cannot choose random element after {n_trials} trials for sentence:"
        "\n"
        f"{invalid_text}"
    )


def create_shuffle(
    contexts_file: str = "data/nonce/sentential_contexts.tsv",
    vignettes_file: str = "data/vignettes/single.json",
    output_dir: str = "data/input/shuffle/",
    mode: str = "global",
    length_matched: bool = False,
    overwrite: bool = False,
) -> None:
    output_path_first = os.path.join(output_dir, f"shuffled_{mode}_first.json")
    output_path_first_control = output_path_first.replace(".json", "_control.json")
    output_path_second = os.path.join(output_dir, f"shuffled_{mode}_second.json")
    output_path_second_control = output_path_second.replace(".json", "_control.json")
    output_path_both = os.path.join(output_dir, f"shuffled_{mode}_both.json")
    output_path_both_control = output_path_both.replace(".json", "_control.json")
    # output_path_both_independent = os.path.join(output_dir, f"shuffled_{mode}_both_independent.json")
    # output_
    if (
        os.path.exists(output_dir)
        and (
            os.path.isfile(output_path_first)
            or os.path.isfile(output_path_first_control)
            or os.path.isfile(output_path_second)
            or os.path.isfile(output_path_second_control)
            or os.path.isfile(output_path_both)
            or os.path.isfile(output_path_both_control)
        )
        and not overwrite
    ):
        raise ValueError(
            f"Output directory {output_dir} already contains files, use `-f` to overwrite."
        )
    elif not os.path.exists(output_dir):
        # create output_dir
        os.mkdir(output_dir)

    logger.info(f"Reading nonce contexts from {contexts_file}")
    sentences = get_correct_nonce_combinations(contexts_file)
    logger.info(f"Reading vignettes from {vignettes_file}")
    vignette_dicts = read_vignettes(vignettes_file)

    sentences_no_uid = [txt for _, txt in sentences]
    if mode == "global":
        shuffles = global_shuffle(sentences_no_uid)
    else:
        raise NotImplementedError(f"Shuffling mode {mode} invalid.")

    shuffles = [(compute_hash(txt), txt) for txt in shuffles]

    outputs_shuf_first: List[Dict] = []
    outputs_shuf_first_control: List[Dict] = []
    outputs_shuf_second: List[Dict] = []
    outputs_shuf_second_control: List[Dict] = []
    outputs_shuf_both: List[Dict] = []
    outputs_shuf_both_control: List[Dict] = []

    sID = 0
    for vignette_dict in vignette_dicts:
        pairs = enumerate(zip(sentences, shuffles))
        for idx, ((uid, text), (uid_shu, text_shu)) in pairs:
            # Select a random shuffled and nonshuffled example
            uid_shu_random, text_shu_random = choose_random(
                shuffles, (uid_shu, text_shu), length_matched
            )
            uid_random, text_random = choose_random(sentences, (uid, text), length_matched)
            outputs_shuf_second.append(
                {
                    "sequenceID": sID,
                    "experimentID": idx,
                    "sentenceID": uid + uid_shu,
                    "vignetteID": vignette_dict["id"],
                    "preface": vignette_dict["preface"],
                    "first": text,
                    "intervention": vignette_dict["intervention"],
                    "prompt": vignette_dict["prompt"],
                    "second": text_shu,
                }
            )
            outputs_shuf_second_control.append(
                {
                    "sequenceID": sID + 1,
                    "experimentID": idx,
                    "sentenceID": uid_random + uid_shu,
                    "vignetteID": vignette_dict["id"],
                    "preface": vignette_dict["preface"],
                    "first": text_random,
                    "intervention": vignette_dict["intervention"],
                    "prompt": vignette_dict["prompt"],
                    "second": text_shu,
                }
            )
            outputs_shuf_first.append(
                {
                    "sequenceID": sID + 2,
                    "experimentID": idx,
                    "sentenceID": uid_shu + uid,
                    "vignetteID": vignette_dict["id"],
                    "preface": vignette_dict["preface"],
                    "first": text_shu,
                    "intervention": vignette_dict["intervention"],
                    "prompt": vignette_dict["prompt"],
                    "second": text,
                }
            )
            outputs_shuf_first_control.append(
                {
                    "sequenceID": sID + 3,
                    "experimentID": idx,
                    "sentenceID": uid_shu_random + uid,
                    "vignetteID": vignette_dict["id"],
                    "preface": vignette_dict["preface"],
                    "first": text_shu_random,
                    "intervention": vignette_dict["intervention"],
                    "prompt": vignette_dict["prompt"],
                    "second": text,
                }
            )
            outputs_shuf_both.append(
                {
                    "sequenceID": sID + 4,
                    "experimentID": idx,
                    "sentenceID": uid_shu + uid_shu,
                    "vignetteID": vignette_dict["id"],
                    "preface": vignette_dict["preface"],
                    "first": text_shu,
                    "intervention": vignette_dict["intervention"],
                    "prompt": vignette_dict["prompt"],
                    "second": text_shu,
                }
            )
            outputs_shuf_both_control.append(
                {
                    "sequenceID": sID + 5,
                    "experimentID": idx,
                    "sentenceID": uid_shu_random + uid_shu,
                    "vignetteID": vignette_dict["id"],
                    "preface": vignette_dict["preface"],
                    "first": text_shu_random,
                    "intervention": vignette_dict["intervention"],
                    "prompt": vignette_dict["prompt"],
                    "second": text_shu,
                }
            )
            sID += 6

    logger.info(
        f"Writing shuffles to {output_path_first}, {output_path_second}, {output_path_both}"
    )
    logger.info(
        f"Writing control-shuffles to {output_path_first_control}, {output_path_second_control},"
        f" {output_path_both_control}"
    )
    save_multiple_to_json(
        [
            outputs_shuf_first,
            outputs_shuf_first_control,
            outputs_shuf_second,
            outputs_shuf_second_control,
            outputs_shuf_both,
            outputs_shuf_both_control,
        ],
        [
            output_path_first,
            output_path_first_control,
            output_path_second,
            output_path_second_control,
            output_path_both,
            output_path_both_control,
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts_file",
        type=str,
        default="data/nonce/sentential_contexts.tsv",
        help=".tsv file containing sentential contexts",
    )
    parser.add_argument(
        "--vignettes_file",
        type=str,
        default="data/vignettes/single.json",
        help=".json file containing preface, intervention and prompt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/input/shuffle/",
        help="dir in which experiment input files are placed into.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="global",
        choices=[
            "global",
        ],
        help="which shuffling mode",
    )
    parser.add_argument(
        "--length_matched",
        action="store_true",
        help="Whether to match controls in length to sequences",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If set, will overwrite existing content in output_dir.",
    )
    args = parser.parse_args()

    create_shuffle(
        args.contexts_file,
        args.vignettes_file,
        args.output_dir,
        args.mode,
        length_matched=args.length_matched,
        overwrite=args.force,
    )
