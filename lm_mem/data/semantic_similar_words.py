import os
import random
from typing import Dict, List

from nltk.corpus import wordnet as wn

from lm_mem.data.nonce import get_correct_nouns, get_correct_verbs
from lm_mem.data.reader import save_synonyms


def get_semantic_similar_words(
    words: List[str],
    pos: str = "noun",
    n_similar: int = 10,
) -> Dict[str, List[str]]:

    if pos not in ["verb", "noun"]:
        raise ValueError(f"word-to-swap has to be either 'noun' or 'verb' instead of {pos}.")

    pos = wn.NOUN if pos == "noun" else wn.VERB
    synonyms = dict()
    for word in words:
        # get synset
        synsets = wn.synsets(word, pos=pos)
        collected_synonyms = list()
        for synset in synsets:
            collected_synonyms.extend(synset.lemma_names())
        unique_synonyms = [syn.replace("_", " ") for syn in set(collected_synonyms)]
        filtered_synonyms = [syn for syn in unique_synonyms if syn != word]
        sampled = random.sample(filtered_synonyms, min(len(filtered_synonyms), n_similar))
        synonyms[word] = sampled

    return synonyms


def nonce_similar_words(
    contexts_file: str = "data/nonce/sentential_contexts.tsv",
    output_dir: str = "data/input/word_swap",
    n_similar: int = 10,
    overwrite: bool = False,
) -> None:

    # Usual file/path checks
    nouns_path = os.path.join(output_dir, "synonyms_noun.txt")
    verbs_path = os.path.join(output_dir, "synonyms_verb.txt")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not os.path.isdir(output_dir):
        raise ValueError(f"output_dir {output_dir} has to be dir.")
    elif not overwrite and (os.path.exists(nouns_path) or os.path.exists(verbs_path)):
        raise ValueError(
            f"Output directory {output_dir} contains synonym files, use `-f` to overwrite."
        )

    nouns = get_correct_nouns(contexts_file=contexts_file)
    verbs = get_correct_verbs(contexts_file=contexts_file)

    noun_synonyms = get_semantic_similar_words(nouns, pos="noun", n_similar=n_similar)
    verb_synonyms = get_semantic_similar_words(verbs, pos="verb", n_similar=n_similar)

    save_synonyms(nouns_path, noun_synonyms)
    save_synonyms(verbs_path, verb_synonyms)


if __name__ == "__main__":
    nonce_similar_words(overwrite=True)
