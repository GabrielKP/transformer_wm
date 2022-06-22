"""
Tokenizer for RNNModel

@date: 01.14.2022
@authors: gabriel.kressin@fu-berlin.de, karmeni1
"""

import os
from typing import Callable, List, Optional, Union

import nltk
import torch


class Dictionary(object):
    """Maps between observations and indices

    KA: from github.com/vansky/neural-complexity-master
    """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """Adds a new obs to the dictionary if needed."""

        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class RNNTokenizer:
    """Class that is able tokenize input text and convert
    it appropriatly into ids to be used by the RNNModel with
    previous vocabularies.

    Parameters
    ----------
    vocab_file : ``str``
        Path to vocabulary file.
    tokenizer : ``Callable[[str], List[str]], optional (default=``None``)
        Tokenizer to convert a string into a list of tokens.
        If ``None`` will use nltk.tokenize.
    unk_token : ``str``, optional (default=``"<unk>"``)
        Unknown token. Outside of Vocabulary tokens get converted into this.
    bos_token : ``str``, optional (default=``"<eos>"``)
        Token inserted before sequence.
    eos_token : ``str``, optional (default=``"<eos>"``)
        Token append behind sequence.
    num_token : ``str``, optional (default=``"<num>"``)
        Numeral token. Floatable numbers get converted into this token.
    """

    def __init__(
        self,
        vocab_file: str,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<eos>",
        eos_token: str = "<eos>",
        num_token: str = "<num>",
    ) -> None:
        super(RNNTokenizer, self).__init__()

        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.num_token = num_token
        self.dictionary = Dictionary()
        self._load_vocab(vocab_file)
        for token in (unk_token, bos_token, eos_token, num_token):
            if token not in self.dictionary.word2idx:
                self.dictionary.add_word(token)
        self.tokenizer = tokenizer or nltk.word_tokenize

    def _load_vocab(self, path_vocab_file: str) -> None:
        """Loads vocabulary as dictionary from disk"""

        assert os.path.exists(path_vocab_file), f"Bad path: {path_vocab_file}"
        if path_vocab_file[-3:] == "bin":
            # This check actually seems to be faster than passing in a binary flag
            # Assume dict is binarized
            import dill

            with open(path_vocab_file, "rb") as file_handle:
                fdata = torch.load(file_handle, pickle_module=dill)
                if isinstance(fdata, tuple):
                    # Compatibility with old pytorch LM saving
                    self.dictionary = fdata[3]
                self.dictionary = fdata
        else:
            # Assume dict is plaintext
            with open(path_vocab_file, "r", encoding="utf-8") as file_handle:
                for line in file_handle:
                    self.dictionary.add_word(line.strip())

    def encode(self, input_: str, return_tensors: str = "pt") -> torch.Tensor:
        """Tokenizes input_ and converts it into ids."""

        def _isfloat(string):
            """Returns whether a string is floatable."""
            try:
                _ = float(string)
                return True
            except ValueError:
                return False

        def _convert(token):
            # 3 Options
            if token not in self.dictionary.word2idx:
                # 1. is out of vocabulary
                return self.dictionary.word2idx[self.unk_token]
            elif _isfloat(token) and self.num_token in self.dictionary.word2idx:
                # 2. token is numeral
                return self.dictionary.word2idx[self.num_token]
            else:
                # 3. token is in vocabulary
                return self.dictionary.word2idx[token]

        if return_tensors != "pt":
            # argument is only for compability with Tokenizers library
            raise NotImplementedError("Pytorch only.")

        tokens = self.tokenizer(input_)

        return torch.tensor(
            [_convert(token) for token in tokens],
            dtype=torch.int64,
        ).unsqueeze(0)

    def decode(
        self,
        ids: Union[int, List[int], torch.Tensor],
    ) -> str:
        """Returns tokens for given ids."""

        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            return self.dictionary.idx2word[ids]

        return " ".join([self.dictionary.idx2word[id] for id in ids])
