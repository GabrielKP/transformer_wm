"""Creates repeat experiment input data, repeat sentences and controls.
"""

import argparse
import os
import random
from typing import Dict, List, Optional, Tuple, Union

from transformer_wm import get_logger
from transformer_wm.data.nonce import get_nouns, get_verbs
from transformer_wm.data.sentences import (
    SUBJECT_MARKER,
    VERB_MARKER,
    get_sentence_combinations_revised_resampled,
    load_synonyms_revised_resampled,
)
from transformer_wm.data.utils import compute_hash, read_vignettes, save_with_suffix

logger = get_logger(__name__)


def word_swap(
    vignette_dicts: List[Dict],
    word_lists: Optional[List[List[str]]] = None,
    named_word_lists: Optional[List[List[Union[str, List[str]]]]] = None,
    word_to_swap: str = "noun",
    n_seq2s: int = 10,
) -> Tuple[List[Dict], List[Dict]]:

    if not ((word_lists is None) ^ (named_word_lists is None)):
        # ^ logical xor
        raise ValueError(
            "Function only works with either word_list or named_word_lists, but neither with both or none"
        )

    if word_to_swap not in ["verb", "noun"]:
        raise ValueError(
            f"word-to-swap has to be either 'noun' or 'verb' instead of {word_to_swap}."
        )

    logger.info("Read sentences")
    correct_sentences, _, _ = get_sentence_combinations_revised_resampled(insert="both")

    (
        sentences_without_words,
        correct_nouns,
        correct_verbs,
    ) = get_sentence_combinations_revised_resampled(
        insert="verb" if word_to_swap == "noun" else "noun",
    )

    correct_words = correct_nouns if word_to_swap == "noun" else correct_verbs

    seq1s: List[Dict] = []
    seq2s: List[Dict] = []
    sID = 0
    eID = 0

    swap_word_marker = SUBJECT_MARKER if word_to_swap == "noun" else VERB_MARKER

    for vignette_dict in vignette_dicts:
        iterator_ = enumerate(zip(correct_sentences, sentences_without_words, correct_words))
        for idx, ((uid, text), (_, text_without_word), (correct_word)) in iterator_:

            if word_lists is not None:
                swap_list = word_lists[idx]
            else:
                syn_word, swap_list = named_word_lists[idx]
                if syn_word != correct_word:
                    raise RuntimeError("Named word list does not match with sentences.")
                # if there is no synonym list for the word, then just skip this sentence
                if swap_list is None:
                    continue
            for word in swap_list:
                # Get start and end index of swap word index, start: inclusive, end: exclusive
                start_index_word = text_without_word.index(swap_word_marker)
                end_index_word = start_index_word + len(word)
                # Swap word
                text_swapped = text_without_word.replace(
                    swap_word_marker,
                    word,
                )
                text_swapped = text_swapped.strip()
                uid_swapped = compute_hash(text=text_swapped)

                # seq1
                # encoding sentence: "normal"
                # test sentence: "swapped word"
                seq1s.append(
                    {
                        "sequenceID": sID,
                        "experimentID": eID,
                        "sentenceID": uid + uid_swapped,
                        "vignetteID": vignette_dict["id"],
                        "preface": vignette_dict["preface"],
                        "first": text,
                        "intervention": vignette_dict["intervention"],
                        "prompt": vignette_dict["prompt"],
                        "second": text_swapped,
                        "marked_word": (start_index_word, end_index_word),
                    }
                )
                sID += 1

                # seq2
                # control_sentence: "all other normal sentences" (possibly also swap them?)
                # test_sentence: "swapped word"
                sampled_sentences = random.sample(
                    correct_sentences,
                    len(correct_sentences),
                )
                non_added = 0
                for idx2, (uid_control, text_control) in enumerate(sampled_sentences):
                    if idx2 - non_added >= n_seq2s:
                        break
                    if correct_word in text_control or uid == uid_control:
                        non_added += 1
                        continue

                    seq2s.append(
                        {
                            "sequenceID": sID,
                            "experimentID": eID,
                            "sentenceID": uid_control + uid_swapped,
                            "vignetteID": vignette_dict["id"],
                            "preface": vignette_dict["preface"],
                            "first": text_control,
                            "intervention": vignette_dict["intervention"],
                            "prompt": vignette_dict["prompt"],
                            "second": text_swapped,
                            "marked_word": (start_index_word, end_index_word),
                        }
                    )
                    sID += 1
                eID += 1
    return seq1s, seq2s


def create_word_swap(
    vignettes_file: str = "data/vignettes/single.json",
    output_dir: str = "data/input/word_swap/",
    n_ARB_word_swaps: int = 3,
    overwrite: bool = False,
) -> None:
    if os.path.exists(output_dir) and os.listdir(output_dir) and not overwrite:
        raise ValueError(f"Output directory {output_dir} is non-empty, use `-f` to overwrite.")
    elif not os.path.exists(output_dir):
        # create output_dir
        os.mkdir(output_dir)

    logger.info(f"Reading vignettes from {vignettes_file}")
    vignette_dicts = read_vignettes(vignettes_file)

    # For RPT experiment
    logger.info("RPT experiment")
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
        suffix="rpt_noun",
    )
    save_with_suffix(
        output_dir=output_dir,
        seq1s=seq1s_repeat_verb,
        seq2s=seq2s_repeat_verb,
        suffix="rpt_verb",
    )

    # For ARB experiment
    logger.info("ARB experiment")
    nouns_file = "data/nonce/nouns.tsv"
    verbs_file = "data/nonce/verbs.tsv"
    nouns = get_nouns(nouns_file)
    verbs = get_verbs(verbs_file)
    nouns_list = [
        random.sample(nouns, min(n_ARB_word_swaps, len(nouns))) for _ in range(len(correct_nouns))
    ]
    verbs_list = [
        random.sample(verbs, min(n_ARB_word_swaps, len(verbs))) for _ in range(len(correct_verbs))
    ]
    seq1s_syntax_noun, seq2s_syntax_noun = word_swap(
        vignette_dicts=vignette_dicts,
        word_lists=nouns_list,
        word_to_swap="noun",
    )
    seq1s_syntax_verb, seq2s_syntax_verb = word_swap(
        vignette_dicts=vignette_dicts,
        word_lists=verbs_list,
        word_to_swap="verb",
    )
    save_with_suffix(
        output_dir=output_dir,
        seq1s=seq1s_syntax_noun,
        seq2s=seq2s_syntax_noun,
        suffix="arb_noun",
    )
    save_with_suffix(
        output_dir=output_dir,
        seq1s=seq1s_syntax_verb,
        seq2s=seq2s_syntax_verb,
        suffix="arb_verb",
    )

    # For SYN experiment
    logger.info("SYN experiment")

    noun_syns, verb_syns = load_synonyms_revised_resampled()
    seq1s_semantic_noun, seq2s_semantic_noun = word_swap(
        vignette_dicts=vignette_dicts,
        named_word_lists=noun_syns,
        word_to_swap="noun",
    )

    seq1s_semantic_verb, seq2s_semantic_verb = word_swap(
        vignette_dicts=vignette_dicts,
        named_word_lists=verb_syns,
        word_to_swap="verb",
    )
    save_with_suffix(
        output_dir=output_dir,
        seq1s=seq1s_semantic_verb,
        seq2s=seq2s_semantic_verb,
        suffix="syn_verb",
    )
    save_with_suffix(
        output_dir=output_dir,
        seq1s=seq1s_semantic_noun,
        seq2s=seq2s_semantic_noun,
        suffix="syn_noun",
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
        default="data/input/word_swap/",
        help="dir in which experiment input files are placed into.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If set, will overwrite existing content in output_dir.",
    )
    args = parser.parse_args()

    create_word_swap(
        vignettes_file=args.vignettes_file,
        output_dir=args.output_dir,
        overwrite=args.force,
    )
