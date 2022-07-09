from typing import List, Tuple, Union

from transformer_wm import get_logger
from transformer_wm.data.utils import compute_hash, read_txt

logger = get_logger(__name__)


SUBJECT_MARKER = "[SUBJECT]"
VERB_MARKER = "[VERB]"
DATA_PATH = "data/sentences/"


def get_sentence_combinations_revised_resampled(
    **kwargs,
) -> Union[List[Tuple[str, str]], Tuple[List[Tuple[str, str]], List[str], List[str]]]:
    sentences_revised, nouns_revised, verbs_revised = get_sentence_combinations(
        sentences_file=f"{DATA_PATH}/sentences_revised.txt",
        nouns_file=f"{DATA_PATH}/nouns_revised.txt",
        verbs_file=f"{DATA_PATH}/verbs_revised.txt",
        **kwargs,
    )
    sentences_resampled, nouns_resampled, verbs_resampled = get_sentence_combinations(
        sentences_file=f"{DATA_PATH}/sentences_resampled.txt",
        nouns_file=f"{DATA_PATH}/nouns_resampled.txt",
        verbs_file=f"{DATA_PATH}/verbs_resampled.txt",
        **kwargs,
    )
    return (
        [*sentences_revised, *sentences_resampled],
        [*nouns_revised, *nouns_resampled],
        [*verbs_revised, *verbs_resampled],
    )


def get_sentence_combinations(
    sentences_file: str,
    nouns_file: str,
    verbs_file: str,
    insert: str = "both",
) -> Union[List[Tuple[str, str]], Tuple[List[Tuple[str, str]], List[str], List[str]]]:
    """Returns nonce sentences that are semantically coherent."""

    sentences_raw = read_txt(sentences_file)
    nouns = read_txt(nouns_file)
    verbs = read_txt(verbs_file)

    sentences: List[Tuple[str, str]] = []
    for sentence, noun, verb in zip(sentences_raw, nouns, verbs):
        if insert in "nouns" or insert in "both":
            sentence = sentence.replace(SUBJECT_MARKER, noun)
        if insert in "verbs" or insert in "both":
            sentence = sentence.replace(VERB_MARKER, verb)

        uid = compute_hash(sentence)
        sentences.append((uid, sentence))

    return sentences, nouns, verbs


def load_synonyms(path: str) -> List[Tuple[str, List[str]]]:
    """Loads synonym list from txt file.

    Lines starting with '#' are ignored.
    Lines starting with '--' indicate original word. Count as markers to start new list.
    Lines below indicate synonyms for that word.
    """
    logger.info(f"Loading synonyms from {path}")
    synonyms: List[Tuple[str, List[str]]] = []
    current_word = None
    with open(path, "r") as f_in:
        for line in f_in.readlines():
            if line.startswith("#"):
                continue
            if line.endswith("\n"):
                line = line[:-1]
            if line.startswith("--"):
                current_word = line.replace("--", "").strip()
                synonyms.append([current_word, []])
            elif current_word is None:
                raise ValueError("Missing base word in synonyms file.")
            else:
                synonyms[-1][1].append(line.strip())

    return synonyms


def load_synonyms_revised_resampled() -> Tuple[
    List[Tuple[str, List[str]]], List[Tuple[str, List[str]]]
]:
    """Loads synyonym list for nouns and verbs

    Lines starting with '#' are ignored.
    Lines starting with '--' indicate original word. Count as markers to start new list.
    Lines below indicate synonyms for that word.
    """
    return (
        [
            *load_synonyms("data/synonyms/nouns_revised.txt"),
            *load_synonyms("data/synonyms/nouns_resampled.txt"),
        ],
        [
            *load_synonyms("data/synonyms/verbs_revised.txt"),
            *load_synonyms("data/synonyms/verbs_resampled.txt"),
        ],
    )
