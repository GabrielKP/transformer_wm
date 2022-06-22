"""Creates paraphrased experiment input data
"""

import argparse
import logging
import os
import random
from typing import Dict, List, Tuple

from lm_mem import get_logger
from lm_mem.data.utils import compute_hash, read_txt, read_vignettes, save_to_json

logger = get_logger(__name__)


def create_paraphrase(
    sentences_txt: str = "data/input/paraphrase/sentences_base.txt",
    paraphrases_txt: str = "data/input/paraphrase/sentences_backtranslated.txt",
    vignettes_file: str = "data/vignettes/single.json",
    output_dir: str = "data/input/paraphrase/",
    overwrite: bool = False,
) -> None:
    output_path = os.path.join(output_dir, "paraphrases.json")
    output_path_controls = os.path.join(output_dir, "controls.json")
    if (
        os.path.exists(output_dir)
        and (os.path.isfile(output_path) or os.path.isfile(output_path_controls))
        and not overwrite
    ):
        raise ValueError(f"Output directory {output_dir} is non-empty, use `-f` to overwrite.")
    elif not os.path.exists(output_dir):
        # create output_dir
        os.mkdir(output_dir)

    logger.info(f"Reading original sentences from {sentences_txt}")
    sentences = read_txt(sentences_txt)
    logger.info(f"Reading paraphrased sentences from {sentences_txt}")
    paraphrases = read_txt(paraphrases_txt)
    logger.info(f"Reading vignettes from {vignettes_file}")
    vignette_dicts = read_vignettes(vignettes_file)

    sentences: Tuple[str, str] = [(compute_hash(txt), txt) for txt in sentences]
    paraphrases: Tuple[str, str] = [(compute_hash(txt), txt) for txt in paraphrases]

    outputs: List[Dict] = []
    controls: List[Dict] = []

    sID = 0
    eID = 0
    for vignette_dict in vignette_dicts:
        pairs = zip(sentences, paraphrases)
        for (uid, text), (uid_par, text_par) in pairs:
            # normal -> paraphrased
            outputs.append(
                {
                    "sequenceID": sID,
                    "experimentID": eID,
                    "sentenceID": uid + uid_par,
                    "vignetteID": vignette_dict["id"],
                    "preface": vignette_dict["preface"],
                    "first": text,
                    "intervention": vignette_dict["intervention"],
                    "prompt": vignette_dict["prompt"],
                    "second": text_par,
                }
            )
            sID += 1

            # get "normal" control
            ctrl_uid, ctrl_text = sentences[random.randint(0, len(sentences) - 1)]
            while ctrl_uid == uid:
                ctrl_uid, ctrl_text = sentences[random.randint(0, len(sentences) - 1)]

            controls.append(
                {
                    "sequenceID": sID,
                    "experimentID": eID,
                    "sentenceID": ctrl_uid + uid_par,
                    "vignetteID": vignette_dict["id"],
                    "preface": vignette_dict["preface"],
                    "first": ctrl_text,
                    "intervention": vignette_dict["intervention"],
                    "prompt": vignette_dict["prompt"],
                    "second": text_par,
                }
            )
            sID += 1
            eID += 1

            # paraphrased -> normal
            outputs.append(
                {
                    "sequenceID": sID,
                    "experimentID": eID,
                    "sentenceID": uid_par + uid,
                    "vignetteID": vignette_dict["id"],
                    "preface": vignette_dict["preface"],
                    "first": text_par,
                    "intervention": vignette_dict["intervention"],
                    "prompt": vignette_dict["prompt"],
                    "second": text,
                }
            )
            sID += 1

            # get "paraphrased" control
            ctrl_uid_par, ctrl_text_par = paraphrases[random.randint(0, len(sentences) - 1)]
            while ctrl_uid_par == uid_par:
                ctrl_uid_par, ctrl_text_par = paraphrases[random.randint(0, len(sentences) - 1)]

            controls.append(
                {
                    "sequenceID": sID,
                    "experimentID": eID,
                    "sentenceID": ctrl_uid_par + uid,
                    "vignetteID": vignette_dict["id"],
                    "preface": vignette_dict["preface"],
                    "first": ctrl_text_par,
                    "intervention": vignette_dict["intervention"],
                    "prompt": vignette_dict["prompt"],
                    "second": text,
                }
            )
            sID += 1
            eID += 1

    logger.info(f"Writing paraphrases to {output_path}")
    save_to_json(
        {"sequences": outputs},
        output_path,
    )
    logger.info(f"Writing controls to {output_path_controls}")
    save_to_json(
        {"sequences": controls},
        output_path_controls,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentences_txt",
        type=str,
        default="data/input/paraphrase/sentences_base.txt",
        help=".txt file containing original sentences.",
    )
    parser.add_argument(
        "--paraphrases_txt",
        type=str,
        default="data/input/paraphrase/sentences_backtranslated.txt",
        help=".txt file containing paraphrased sentences in matching order to" + "sentences_txt.",
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
        default="data/input/paraphrase/",
        help="dir in which experiment input files are placed into.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If set, will overwrite existing content in output_dir.",
    )
    args = parser.parse_args()

    create_paraphrase(
        args.sentences_txt,
        args.paraphrases_txt,
        args.vignettes_file,
        args.output_dir,
        overwrite=args.force,
    )
