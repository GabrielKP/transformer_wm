"""Reads nonce dataset and builds sentences in different ways.
"""

import itertools
import logging
import os
from typing import List, Optional, Tuple, Union

import pandas as pd

from lm_mem.data.utils import compute_hash

logging.basicConfig(format=("[INFO] %(message)s"), level=logging.INFO)


NOUN_COLUMN = "s_noun"
VERB_COLUMN = "s_verb"
CONTEXT_COLUMN = "template"
SUBJECT_MARKER = "[SUBJECT]"
VERB_MARKER = "[VERB]"


def _read_tsv(
    file_path: str,
) -> pd.DataFrame:
    file_df = pd.read_csv(os.path.join(file_path), sep="\t")
    return file_df


def _read_nouns_tsv(file_path: str) -> List[str]:
    return _read_tsv(file_path)[NOUN_COLUMN].to_numpy()


def _read_verbs_tsv(file_path: str) -> List[str]:
    return _read_tsv(file_path)[VERB_COLUMN].to_numpy()


def _read_contexts_tsv(file_path: str) -> List[str]:
    return _read_tsv(file_path)[CONTEXT_COLUMN].to_numpy()


def get_nouns(file_path: str) -> List[str]:
    return list(_read_nouns_tsv(file_path))


def get_verbs(file_path: str) -> List[str]:
    return list(_read_verbs_tsv(file_path))


def get_contexts(file_path: str) -> List[str]:
    return list(_read_contexts_tsv(file_path))


def get_correct_nouns(contexts_file: str) -> List[str]:
    return get_correct_nonce_combinations(
        contexts_file=contexts_file, return_correct_separately=True
    )[1]


def get_correct_verbs(contexts_file: str) -> List[str]:
    return get_correct_nonce_combinations(
        contexts_file=contexts_file, return_correct_separately=True
    )[2]


def get_all_nonce_combinations(
    nouns_file: str = "data/nonce/nouns.tsv",
    contexts_file: str = "data/nonce/sentential_contexts.tsv",
    verbs_file: str = "data/nonce/verbs.tsv",
    limit: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Returns all possible combinations of verbs, nounce and sentences
    templates.

    """

    logging.info("Creating all nonce sentence combinations sentences.")

    nouns = _read_nouns_tsv(nouns_file)
    verbs = _read_verbs_tsv(verbs_file)
    contexts = _read_contexts_tsv(contexts_file)

    def _fill_sentence(noun, verb, context):
        text = context.replace(SUBJECT_MARKER, noun).replace(VERB_MARKER, verb)
        uid = compute_hash(text)
        # create tuples for computational reasons
        return (uid, text)

    # creates all possible combinations between input lists
    combinations = list(itertools.product(nouns, verbs, contexts))

    if limit > -1:
        combinations = combinations[:limit]
        logging.info(f"Limited sentences to first {len(combinations)}")

    # merge combinations to sentences
    return [_fill_sentence(*combination) for combination in combinations]


def get_correct_nonce_combinations(
    contexts_file: str = "sentential_contexts.tsv",
    insert: str = "both",
    return_correct_separately: bool = False,
) -> Union[List[Tuple[str, str]], Tuple[List[Tuple[str, str]], List[str], List[str]]]:
    """Returns nonce sentences that are semantically coherent."""

    contexts_df = _read_tsv(contexts_file)

    sentences: List[Tuple[str, str]] = []
    correct_nouns: List[str] = []
    correct_verbs: List[str] = []
    for _, row in contexts_df.iterrows():
        sentence = row[CONTEXT_COLUMN]
        if insert in "nouns" or insert in "both":
            sentence = sentence.replace(SUBJECT_MARKER, row["subject"])
        if insert in "verbs" or insert in "both":
            sentence = sentence.replace(VERB_MARKER, row["correct_verb"])

        uid = compute_hash(sentence)
        sentences.append((uid, sentence))
        correct_nouns.append(row["subject"])
        correct_verbs.append(row["correct_verb"])

    if return_correct_separately:
        return sentences, correct_nouns, correct_verbs
    return sentences
