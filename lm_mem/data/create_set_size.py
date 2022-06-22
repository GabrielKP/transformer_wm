"""Creates repeat sentences and controls with different amounts of sentences.
"""

import argparse
import logging
import os
import random
from typing import Dict, List, Tuple

from lm_mem import get_logger
from lm_mem.data.nonce import get_correct_nonce_combinations
from lm_mem.data.utils import compute_hash, read_vignettes, save_to_json

logger = get_logger(__name__)


def _create_sets(
    sentences: List[Tuple[str, str]],
    set_size: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    repeated_sentences = []
    control_sentences = []
    for (uid, txt) in sentences:
        used_uids = [
            uid,
        ]
        text = [
            txt,
        ]
        # Create the "main" set
        for _ in range(set_size - 1):
            while uid in used_uids:
                random_idx = random.randint(0, len(sentences) - 1)
                uid, txt = sentences[random_idx]
            used_uids.append(uid)
            text.append(txt)

        # Create the "control" set
        control_text = []
        for _ in range(set_size):
            while uid in used_uids:
                random_idx = random.randint(0, len(sentences) - 1)
                uid, txt = sentences[random_idx]
            used_uids.append(uid)
            control_text.append(txt)

        repeated_txt = " ".join(text)
        repeated_sentences.append(
            (
                compute_hash(repeated_txt),
                repeated_txt,
            )
        )
        control_txt = " ".join(control_text)
        control_sentences.append(
            (
                compute_hash(control_txt),
                control_txt,
            )
        )

    return repeated_sentences, control_sentences


def create_set_sizes(
    contexts_file: str = "data/nonce/sentential_contexts.tsv",
    vignettes_file: str = "data/vignettes/single.json",
    output_dir: str = "data/input/set_size/",
    max_set_size: int = 7,
    overwrite: bool = False,
) -> None:
    if os.path.exists(output_dir) and os.listdir(output_dir) and not overwrite:
        raise ValueError(f"Output directory {output_dir} is non-empty, use `-f` to overwrite.")
    elif not os.path.exists(output_dir):
        # create output_dir
        os.mkdir(output_dir)

    logger.info(f"Reading nonce contexts from {contexts_file}")
    sentences = get_correct_nonce_combinations(contexts_file)
    logger.info(f"Reading vignettes from {vignettes_file}")
    vignette_dicts = read_vignettes(vignettes_file)

    sID = 0
    for set_size in range(1, max_set_size + 1):
        # Repeat sentence
        repeated_sentences, control_sentences = _create_sets(sentences, set_size)

        repeats: List[Dict] = []
        controls: Dict[Dict] = []

        for vignette_dict in vignette_dicts:
            for idx, ((uid, text), (c_uid, c_text)) in enumerate(
                zip(repeated_sentences, control_sentences)
            ):

                repeats.append(
                    {
                        "sequenceID": sID,
                        "experimentID": idx,
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

                controls.append(
                    {
                        "sequenceID": sID,
                        "experimentID": idx,
                        "sentenceID": c_uid + uid,
                        "vignetteID": vignette_dict["id"],
                        "preface": vignette_dict["preface"],
                        "first": c_text,
                        "intervention": vignette_dict["intervention"],
                        "prompt": vignette_dict["prompt"],
                        "second": text,
                    },
                )
                sID += 1

        repeats_path = os.path.join(output_dir, f"repeats_set_size_{set_size}.json")
        logger.info(f"Writing repeats to {repeats_path}")
        save_to_json(
            {"sequences": repeats},
            repeats_path,
        )
        controls_path = os.path.join(output_dir, f"controls_set_size_{set_size}.json")
        logger.info(f"Writing controls to {controls_path}")
        save_to_json(
            {"sequences": controls},
            controls_path,
        )


def _repeat_sentences(
    sentences: List[Tuple[str, str]],
    set_size: int,
) -> List[Tuple[str, str]]:
    repeated_sentences = []
    for (_, txt) in sentences:
        repeated_txt = f"{txt} " * set_size
        repeated_txt = repeated_txt[:-1]  # get rid of last whitespace
        repeated_sentences.append(
            (
                compute_hash(repeated_txt),
                repeated_txt,
            )
        )
    return repeated_sentences


def create_set_sizes_same(
    contexts_file: str = "data/nonce/sentential_contexts.tsv",
    vignettes_file: str = "data/vignettes/single.json",
    output_dir: str = "data/input/set_size_same/",
    max_set_size: int = 10,
    overwrite: bool = False,
) -> None:
    """Repeats a sentence x times. Not really useful."""
    if os.path.exists(output_dir) and os.listdir(output_dir) and not overwrite:
        raise ValueError(f"Output directory {output_dir} is non-empty, use `-f` to overwrite.")
    elif not os.path.exists(output_dir):
        # create output_dir
        os.mkdir(output_dir)

    logger.info(f"Reading nonce contexts from {contexts_file}")
    sentences = get_correct_nonce_combinations(contexts_file)
    logger.info(f"Reading vignettes from {vignettes_file}")
    vignette_dicts = read_vignettes(vignettes_file)

    sID = 0
    for set_size in range(1, max_set_size + 1):
        # Repeat sentence
        repeated_sentences = _repeat_sentences(sentences, set_size)

        repeats: List[Dict] = []
        controls: Dict[Dict] = []

        for vignette_dict in vignette_dicts:
            for idx, (uid, text) in enumerate(repeated_sentences):

                repeats.append(
                    {
                        "sequenceID": sID,
                        "experimentID": idx,
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

                random_idx = random.randint(0, len(repeated_sentences) - 1)
                ctrl_uid, ctrl_text = repeated_sentences[random_idx]
                while ctrl_uid == uid:
                    random_idx = random.randint(0, len(repeated_sentences) - 1)
                    ctrl_uid, ctrl_text = repeated_sentences[random_idx]

                controls.append(
                    {
                        "sequenceID": sID,
                        "experimentID": idx,
                        "sentenceID": ctrl_uid + uid,
                        "vignetteID": vignette_dict["id"],
                        "preface": vignette_dict["preface"],
                        "first": ctrl_text,
                        "intervention": vignette_dict["intervention"],
                        "prompt": vignette_dict["prompt"],
                        "second": text,
                    }
                )
                sID += 1

        repeats_path = os.path.join(output_dir, f"repeats_set_size_same_{set_size}.json")
        logger.info(f"Writing repeats to {repeats_path}")
        save_to_json(
            {"sequences": repeats},
            repeats_path,
        )
        controls_path = os.path.join(output_dir, f"controls_set_size_same_{set_size}.json")
        logger.info(f"Writing controls to {controls_path}")
        save_to_json(
            {"sequences": controls},
            controls_path,
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
        default="data/input/set_size/",
        help="dir in which experiment input files are placed into.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If set, will overwrite existing content in output_dir.",
    )
    args = parser.parse_args()

    create_set_sizes(
        args.contexts_file,
        args.vignettes_file,
        args.output_dir,
        overwrite=args.force,
    )
