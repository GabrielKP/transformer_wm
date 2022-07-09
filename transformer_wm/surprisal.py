"""
surprisal.py is used to run the perplexity experiment with GPT-2
it relies on the Experiment() class which is just a class with wrapper methods
around the Transformers library.

Use as:

python experiment.py
or from an ipython console:
%run experiment.py ""

"""

import argparse
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from transformer_wm import get_logger
from transformer_wm.data.reader import read_sequences
from transformer_wm.rnn.model import RNNModel
from transformer_wm.rnn.tokenizer import RNNTokenizer

logger = get_logger(__name__)


# ===== Datalaoder CLASS ===== #
class SimpleDataset(Dataset):
    def __init__(self, _list: List) -> None:
        super().__init__()
        self.items = list(_list)

    def __getitem__(self, index) -> Any:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)


# ===== EXPERIMENT CLASS ===== #
class Experiment(object):

    """
    Exp() class contains wrapper methods to run experiments with transformer models.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Union[str, int, torch.device],
        batch_size: int = 1,
        use_past_keys: bool = True,
        mem_instead_of_past_keys: bool = False,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        self.batch_size = batch_size
        self.use_past_keys = use_past_keys
        self.mem_instead_of_past_keys = mem_instead_of_past_keys

    def get_ppl(
        self,
        input_ids: torch.Tensor,
        seq_lens: List[int],
        targets: torch.Tensor,
    ) -> Tuple[float, List[float], List[str]]:
        """Returns average ppl, list of suprisal values (llhs)
        per token, and list of tokens for given input_ids.

        taken from: https://huggingface.co/transformers/perplexity.html
        """
        # input_ids = [batch_size, seq_len]
        # seq_lens = [batch_size]
        batch_size = len(seq_lens)

        # set model to evaluation mode
        self.model.eval()
        self.model.to(self.device)

        # to match every token need to insert <eos> token artificially at beginning
        llhs = [
            [np.nan] for _ in range(batch_size)
        ]  # variable storing token-by-token neg log likelihoods

        # loop over tokens in input sequence
        with torch.no_grad():
            past_key_values = None
            for idx in range(0, input_ids.size(1) - 1):
                # targets are shifted by 1
                target_ids = targets[:, idx + 1 : idx + 2].to(self.device)

                if self.use_past_keys:
                    # select the current input token
                    selected_input_ids = input_ids[:, idx : idx + 1].to(self.device)

                    # get model output
                    if self.mem_instead_of_past_keys:
                        outputs = self.model(
                            input_ids=selected_input_ids,
                            mems=past_key_values,
                            use_mems=True,
                        )
                        # Save past attention keys for speedup
                        past_key_values = outputs.mems
                    else:
                        outputs = self.model(
                            input_ids=selected_input_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        # Save past attention keys for speedup
                        past_key_values = outputs.past_key_values
                    del selected_input_ids

                    logits = outputs.logits
                else:
                    # select the current input index span
                    # TODO: add handling for seq_len > context_len
                    selected_input_ids = input_ids[:, : idx + 1].to(self.device)

                    # targets are shifted by 1
                    target_ids = targets[:, idx + 1 : idx + 2].to(self.device)
                    outputs = self.model(input_ids=selected_input_ids)
                    del selected_input_ids

                    logits = outputs.logits[:, -1]

                # compute loss
                losses = self.loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                # del outputs
                del target_ids

                for batch_idx, nll_val in enumerate(losses):
                    llhs[batch_idx].append(nll_val.item())

        # Handle padded sequences
        final_llhs = []
        for batch_idx, llh_vals in enumerate(llhs):
            # Cut them off at appropriate length
            final_llhs.append(llh_vals[: seq_lens[batch_idx]])

        # compute perplexity, divide by the lenth of the sequence
        # use np.nansum as token at position 0 will have -LL of nan
        ppls = []
        for batch_idx, llh_vals in enumerate(final_llhs):
            ppls.append(
                torch.exp(torch.tensor(np.nansum(llh_vals)) / (len(llh_vals) - 1)).cpu().item()
            )
        return ppls, final_llhs

    def start(self, input_sequences: List[torch.Tensor]) -> Dict[str, List[any]]:
        """
        experiment.start() will loop over prefaces, prompts, and word_lists and run the .ppl() method on them
        It returns a dict:
        outputs = {
            "sequence_ppls": [],
            "surprisals": [],
        }
        """
        # 1. collate_fn
        def make_batch(sequence_list: List[torch.Tensor]) -> Tuple[torch.Tensor]:
            """Converts list of sequences into a padded torch Tensor and its lengths"""
            sequence_lengths = [len(sequence[0]) for sequence in sequence_list]
            batched_sequence = torch.nn.utils.rnn.pad_sequence(
                [sequence[0] for sequence in sequence_list],
                batch_first=True,
                padding_value=self.tokenizer.encode(self.tokenizer.unk_token)[0],
            )
            target_sequence = torch.nn.utils.rnn.pad_sequence(
                [sequence[0] for sequence in sequence_list],
                batch_first=True,
                padding_value=self.loss_fct.ignore_index,
            )
            return batched_sequence, sequence_lengths, target_sequence

        # 2. make dataset and dataloader
        sequence_dataset = SimpleDataset(input_sequences)
        sequence_loader = DataLoader(
            sequence_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=make_batch
        )

        # 3. go through sequences
        outputs = {
            "sequence_ppls": [],
            "surprisals": [],
        }
        sequence_iterator = tqdm(sequence_loader, desc="Computing Surprisal values", colour="blue")
        for input_ids, sequence_lengths, targets in sequence_iterator:

            ppls, surprisals = self.get_ppl(input_ids, sequence_lengths, targets)
            # store the outputs and
            outputs["sequence_ppls"].extend(ppls)
            outputs["surprisals"].extend(surprisals)

        return outputs


def init_model_and_tokenizer(
    model_name: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Initialize a Model and its tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return (model, tokenizer)


def convert_to_word_row_df(merged_dict: Dict, return_marked_words: bool = False) -> pd.DataFrame:
    """Returns df in which each word has its row, given df in which
    each row contains a sequence.
    """

    # Create for every word a dataframe row
    f_experimentIDs = []
    f_sequenceIDs = []
    f_sentenceIDs = []
    f_vignetteIDs = []
    f_sectionIDs = []
    f_positionIDs = []
    f_words = []
    f_surprisal = []
    f_marked_word = []

    marked_word_token_indices_list = merged_dict.get(
        "marked_word_token_indices",
        [(-1, -1)] * len(merged_dict["experimentID"]),
    )

    # Iterate through sequences, convert tokens into own rows
    sequence_rows = zip(
        merged_dict["experimentID"],
        merged_dict["sequenceID"],
        merged_dict["sentenceID"],
        merged_dict["vignetteID"],
        merged_dict["sectionIDs"],
        merged_dict["positionIDs"],
        merged_dict["subtokIDs"],
        merged_dict["surprisals"],
        merged_dict["tokens"],
        marked_word_token_indices_list,
    )

    # Go through each sequence
    for (
        experiment_ID,
        sequenceID,
        sentenceID,
        vignetteID,
        sectionIDs,
        positionIDs,
        subtokIDs,
        surprisals,
        tokens,
        marked_word_token_indices,
    ) in sequence_rows:

        token_list = enumerate(
            zip(
                sectionIDs,
                positionIDs,
                subtokIDs,
                surprisals,
                tokens,
            )
        )

        start_index_token, end_index_token = marked_word_token_indices
        # to add some confusion: end_index_token is INCLUSIVE

        # Handle merging
        # remember last subtokID, if it is positive and the same as before
        # merge both words. (every subtokID is the index of the original token)
        last_subtokID = -4

        for sentence_posID, (sectionID, positionID, subtokID, surprisal, token) in token_list:
            if subtokID > -1 and subtokID == last_subtokID:
                # add surprisal
                f_surprisal[-1] += surprisal
                # merge tokens with _
                f_words[-1] = f"{f_words[-1]}_{token}"
                continue

            f_experimentIDs.append(experiment_ID)
            f_sequenceIDs.append(sequenceID)
            f_sentenceIDs.append(sentenceID)
            f_vignetteIDs.append(vignetteID)
            f_sectionIDs.append(sectionID)
            f_positionIDs.append(positionID)
            f_words.append(token)
            f_surprisal.append(surprisal)
            # update last_subtokID
            last_subtokID = subtokID

            if (
                sectionID == 3
                and sentence_posID >= start_index_token
                and sentence_posID <= end_index_token
            ):
                f_marked_word.append(1)
            else:
                f_marked_word.append(0)

    dict_for_df = {
        "experimentID": f_experimentIDs,
        "sequenceID": f_sequenceIDs,
        "sentenceID": f_sentenceIDs,
        "vignetteID": f_vignetteIDs,
        "sectionID": f_sectionIDs,
        "positionID": f_positionIDs,
        "word": f_words,
        "surprisal": f_surprisal,
    }
    if return_marked_words:
        dict_for_df["marked_word"] = f_marked_word

    return pd.DataFrame(dict_for_df)


# ===== RUNTIME CODE WRAPPER ===== #
def compute_surprisal(
    input_path: str,
    output_path: str,
    model_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 1,
    device: Optional[str] = None,
    overwrite: bool = False,
    return_marked_words: bool = False,
    mem_instead_of_past_keys: bool = False,
    experiment_class: object = Experiment,
    experiment_kwargs: Optional[Dict] = None,
    converter_func: Callable[[Dict, bool], pd.DataFrame] = convert_to_word_row_df,
    converter_kwargs: Optional[Dict] = None,
    start_experiment_with_dict: bool = False,
):
    """Compute surprisal values for input with model and save them to output.

    Parameters
    ----------
    input_path : str
    output_path : str
    model_name : str
        Name of Initialized Huggingface Model, used to determine output file name if
        output_path is a filename.
    model : PreTrainedModel
        Initialized Huggingface Model on which experiment is run on.
    tokenizer : PreTrainedTokenizer
        Initialized Huggingface Transformer
    batch_size : int, optional, default=1
    device : str, optional, default=None
    return_marked_words : bool, optional, default=False
        Whether to return a mask indicating words have been marked in the input files.
        Is required, as the tokenizer may split words it is not straightforward to keep
        track of words coming from the input_files going to the output_files. This will
        mark words in section3 indicated by the field "marked_word" with a 1 in the output, and
        the rest of words with 0.
        "marked_word" is the char_range (start, end) (start inclusive, end exclusive) of the word
        in section 3.
    """
    experiment_kwargs = experiment_kwargs or {}
    converter_kwargs = converter_kwargs or {}

    # Normalize model_name
    normalized_model_name = model_name.split("/")[-1]

    # Check input and output files/dir
    if not os.path.exists(input_path):
        raise ValueError(f"Path {input_path} does not exist.")
    if os.path.isdir(input_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        else:
            if not os.path.isdir(output_path):
                raise ValueError(
                    "input_path and output_path have to be both a path or both a file."
                )
        # Get input files
        input_filenames = [
            path
            for path in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, path)) and path.endswith(".json")
        ]
        input_files = [os.path.join(input_path, path) for path in input_filenames]
        # Create output files
        output_files = [
            os.path.join(output_path, path.replace(".json", f"_{normalized_model_name}.csv"))
            for path in input_filenames
        ]
    else:
        if os.path.isdir(output_path):
            raise ValueError("input_path and output_path have to be both a path or both a file.")
        input_files = [
            input_path,
        ]
        output_files = [
            output_path,
        ]

    if not overwrite:
        for output_file in output_files:
            if os.path.isfile(output_file):
                raise ValueError(
                    f"Output file {output_file} already exists, use `-f` to overwrite."
                )
    if not os.path.isdir(os.path.dirname(output_path)):
        raise ValueError(
            f"Will not be able to create output_files in {output_path}. Make sure"
            + f" the directory {os.path.dirname(output_path)} exists."
        )
    logger.info("Computing surprisal for:\n" + "\n".join(f"  - {f}" for f in input_files))

    # Device
    if device is not None:
        if device == "cuda" and not torch.cuda.is_available():
            logger.critical(f"Device {device} not available.")
            raise Exception(f"Device {device} not available.")
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device {device} | n_gpu: {torch.cuda.device_count()} | batch_size: {batch_size}")

    # Model and Tokenizer
    logger.info("Loading Model and Tokenizer.")
    logger.info(f"Full model name: {model_name}")
    logger.info(f"Normalized name: {normalized_model_name}")
    if mem_instead_of_past_keys:
        logger.info("Use Mems: True")
    else:
        logger.info(f"Use Past Keys: {True}")
    logger.info(f"Return marked word mask: {return_marked_words}")

    for input_file, output_file in zip(input_files, output_files):

        # Data
        logger.info("Loading and processing Inputs")
        data_dict = read_sequences(
            file_path=input_file,
            tokenizer=tokenizer,
            return_marked_word_position=return_marked_words,
        )

        # Experiment class
        model.eval()
        experiment = experiment_class(
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            mem_instead_of_past_keys=mem_instead_of_past_keys,
            **experiment_kwargs,
        )

        # Run experiment
        if start_experiment_with_dict:
            exp_output_dict = experiment.start(data_dict)
        else:
            exp_output_dict = experiment.start(data_dict["one_hots"])

        # Handle output data
        logger.info("Postprocessing outputs...")
        # Merge input data with output data
        merged_dict = {**data_dict, **exp_output_dict}
        # Convert every token prediction into its own row
        converter_kwargs["return_marked_words"] = return_marked_words
        out_df = converter_func(merged_dict, **converter_kwargs)
        # Save
        logger.info(f"Saving to {output_file}")
        out_df.to_csv(output_file, sep=",", index=False)

    return 0


def main():
    # == Parse Arguments == #
    parser = argparse.ArgumentParser(description="surprisal_gpt2.py runs perplexity experiment")
    parser.add_argument(
        "input_path",
        type=str,
        help="path/to/ dir or file.json containing sequences.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="path/to/ dir or file.csv, has to be dir if input_path is dir.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Toggle debug messages on",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="path/to/log_file.txt if not none logs will be written there",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device program is run on (cpu/cuda)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Model type on which to run the experiment.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch Size.",
    )
    parser.add_argument(
        "--return_marked_words",
        action="store_true",
        help="Whether to mark marked words.",
    )
    parser.add_argument(
        "--mem_instead_of_past_keys",
        action="store_true",
        help="Whether to mark marked words.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If set, will overwrite existing output_file.",
    )

    args = parser.parse_args()

    # Init Logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(log_level)

    if args.log_file is not None:
        file_handler = logging.FileHandler(args.log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    model, tokenizer = init_model_and_tokenizer(args.model_name)

    compute_surprisal(
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model_name,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.force,
        return_marked_words=args.return_marked_words,
        mem_instead_of_past_keys=args.mem_instead_of_past_keys,
    )


# ===== Main ===== #
if __name__ == "__main__":
    main()
