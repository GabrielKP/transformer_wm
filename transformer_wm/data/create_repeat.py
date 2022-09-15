"""Creates repeat experiment input data, repeat sentences and controls.
"""

import argparse
import logging
import os
import random
from typing import Dict, List

from transformer_wm import get_logger
from transformer_wm.data.sentences import get_sentence_combinations_revised_resampled
from transformer_wm.data.utils import read_vignettes, save_to_json, save_with_suffix

logger = get_logger(__name__)


def create_repeat(
    vignettes_file: str = "data/vignettes/single.json",
    output_dir: str = "data/input/repeat/",
    overwrite: bool = False,
    n_seq2s: int = 10,
) -> None:
    if os.path.exists(output_dir) and os.listdir(output_dir) and not overwrite:
        raise ValueError(f"Output directory {output_dir} is non-empty, use `-f` to overwrite.")
    elif not os.path.exists(output_dir):
        # create output_dir
        os.mkdir(output_dir)

    logger.info("Reading Sentences.")
    sentences, _, _ = get_sentence_combinations_revised_resampled(insert="both")
    logger.info(f"Reading vignettes from {vignettes_file}")
    vignette_dicts = read_vignettes(vignettes_file)

    repeats: List[Dict] = []
    controls: List[Dict] = []

    sID = 0
    eID = 0
    for vignette_dict in vignette_dicts:
        for idx, (uid, text) in enumerate(sentences):

            repeats.append(
                {
                    "sequenceID": sID,
                    "experimentID": eID,
                    "sentenceID": uid + uid,
                    "vignetteID": vignette_dict["id"],
                    "preface": vignette_dict["preface"],
                    "first": text,
                    "intervention": vignette_dict["intervention"],
                    "prompt": vignette_dict["prompt"],
                    "second": text,
                }
            )
            sID += 1

            sampled_sentences = random.sample(
                sentences,
                len(sentences),
            )
            non_added = 0
            for idx2, (uid_control, text_control) in enumerate(sampled_sentences):
                if idx2 - non_added >= n_seq2s:
                    break
                if uid == uid_control:
                    non_added += 1
                    continue

                controls.append(
                    {
                        "sequenceID": sID,
                        "experimentID": eID,
                        "sentenceID": uid_control + uid,
                        "vignetteID": vignette_dict["id"],
                        "preface": vignette_dict["preface"],
                        "first": text_control,
                        "intervention": vignette_dict["intervention"],
                        "prompt": vignette_dict["prompt"],
                        "second": text,
                    }
                )
                sID += 1
            eID += 1

    save_with_suffix(
        output_dir=output_dir,
        seq1s=repeats,
        seq2s=controls,
        suffix="repeat",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vignettes_file",
        type=str,
        default="data/vignettes/single.json",
        help=".json file containing preface, intervention and prompt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/input/repeat/",
        help="dir in which experiment input files are placed into.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If set, will overwrite existing content in output_dir.",
    )
    parser.add_argument(
        "--n_seq2s",
        type=int,
        default=10,
        help="Number of controls.",
    )
    args = parser.parse_args()

    create_repeat(
        args.vignettes_file,
        args.output_dir,
        overwrite=args.force,
        n_seq2s=args.n_seq2s,
    )
