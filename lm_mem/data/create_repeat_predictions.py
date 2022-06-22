"""Creates repeat experiment input data, repeat sentences and controls.
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple, Union

from lm_mem import get_logger
from lm_mem.data.create_word_swap import word_swap
from lm_mem.data.sentences import get_sentence_combinations_revised_resampled
from lm_mem.data.utils import read_vignettes, save_with_suffix

logger = get_logger(__name__)


def create_repeat_predictions(
    vignettes_file: str = "data/vignettes/single.json",
    output_dir: str = "data/input/repeat_predictions/",
    overwrite: bool = False,
) -> None:
    if os.path.exists(output_dir) and os.listdir(output_dir) and not overwrite:
        raise ValueError(f"Output directory {output_dir} is non-empty, use `-f` to overwrite.")
    elif not os.path.exists(output_dir):
        # create output_dir
        os.mkdir(output_dir)

    logger.info(f"Reading vignettes from {vignettes_file}")
    vignette_dicts = read_vignettes(vignettes_file)

    _, correct_nouns, correct_verbs = get_sentence_combinations_revised_resampled()

    seq1s_repeat_noun, seq2s_repeat_noun = word_swap(
        vignette_dicts=vignette_dicts,
        word_lists=[[n] for n in correct_nouns],
        word_to_swap="noun",
    )
    seq1s_repeat_verb, seq2s_repeat_verb = word_swap(
        vignette_dicts=vignette_dicts,
        word_lists=[[v] for v in correct_verbs],
        word_to_swap="verb",
    )

    save_with_suffix(
        output_dir=output_dir,
        seq1s=seq1s_repeat_noun,
        seq2s=seq2s_repeat_noun,
        suffix="repeat_noun",
    )
    save_with_suffix(
        output_dir=output_dir,
        seq1s=seq1s_repeat_verb,
        seq2s=seq2s_repeat_verb,
        suffix="repeat_verb",
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
        default="data/input/repeat_predictions/",
        help="dir in which experiment input files are placed into.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If set, will overwrite existing content in output_dir.",
    )
    args = parser.parse_args()

    create_repeat_predictions(
        vignettes_file=args.vignettes_file,
        output_dir=args.output_dir,
        overwrite=args.force,
    )
