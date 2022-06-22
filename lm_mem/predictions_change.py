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
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from lm_mem import get_logger
from lm_mem.data.reader import read_sequences
from lm_mem.surprisal import init_model_and_tokenizer

logger = get_logger(__name__)


# ===== EXPERIMENT FUNCTIONS ===== #
def get_seq_nll(
    seq: Tuple[int, torch.Tensor, Tuple[int, int]],
    model: PreTrainedModel,
    device: Union[str, int, torch.device],
) -> torch.Tensor:
    """Get nlls for seq at first marked word position."""
    _, one_hots, (marked_word_token_start, marked_word_token_end) = seq
    # one_hots = [1, seq_len]

    # get input ids until marked word
    input_ids = one_hots[:, :marked_word_token_start].to(device)

    # get model output
    # model_outputs[0] = [1, seq_len, hidden_size]
    model_outputs = model(input_ids=input_ids)
    del input_ids
    # last_logits = [1, hidden_size]
    last_logits = model_outputs[0][:, -1]

    # return = [1, hidden_size]
    return -F.log_softmax(last_logits, dim=1)


def get_change_prob_abs(seq1_logits: np.ndarray, seq2_logits: np.ndarray) -> np.ndarray:
    return np.exp(-seq1_logits) - np.exp(-seq2_logits)


def topk(list_: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns biggest k indices and elements from list_ in decreasing order."""
    # https://stackoverflow.com/a/23734295
    indices_largest = np.argpartition(list_, kth=-k)[-k:]
    sorted_indices_largest = list(reversed(indices_largest[np.argsort(list_[indices_largest])]))
    return sorted_indices_largest, list_[sorted_indices_largest]


def lowk(list_: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns smallest k indices and elements from list_ in ascending order."""
    # https://stackoverflow.com/a/23734295
    indices_largest_neg = np.argpartition(list_, kth=k)[:k]
    sorted_indices_largest_neg = indices_largest_neg[np.argsort(list_[indices_largest_neg])]
    return sorted_indices_largest_neg, list_[sorted_indices_largest_neg]


def compute_predictions_change_experiment(
    seq1s_data_dict: Dict[str, Any],
    seq2s_data_dict: Dict[str, Any],
    change_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_predictions: int,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: Union[str, int, torch.device],
) -> Dict[str, List[Tuple[List[str], List[float]]]]:
    # quick data check
    count_eIDs = Counter(seq2s_data_dict["experimentID"])
    n_controls = list(count_eIDs.values())[0]
    for eID in seq1s_data_dict["experimentID"]:
        if count_eIDs[eID] != n_controls:
            raise ValueError(
                f"Seq2s for {eID} contains {count_eIDs[eID]} controls"
                f" instead of expected {n_controls}."
            )

    # convert dict of lists/tensors into list of tuples
    seq1s = np.array(
        list(
            zip(
                seq1s_data_dict["experimentID"],
                seq1s_data_dict["one_hots"],
                seq1s_data_dict["marked_word_token_indices"],
            )
        ),
        dtype=object,
    )
    seq1s_sort_indices = sorted(range(len(seq1s)), key=lambda i: seq1s[i][0])
    seq1s = seq1s[seq1s_sort_indices]

    seq1s_inv_indices = np.empty(len(seq1s_sort_indices), dtype=np.int_)
    seq1s_inv_indices[seq1s_sort_indices] = np.arange(len(seq1s_sort_indices), dtype=np.int_)

    seq2s = np.array(
        list(
            zip(
                seq2s_data_dict["experimentID"],
                seq2s_data_dict["one_hots"],
                seq2s_data_dict["marked_word_token_indices"],
            )
        ),
        dtype=object,
    )
    seq2s_sort_indices = sorted(range(len(seq2s)), key=lambda i: seq2s[i][0])
    seq2s = seq2s[seq2s_sort_indices]

    model.to(device)
    model.eval()

    # List of [positive_words], [biggest]
    changes_positive: List[Tuple[List[str], List[float]]] = []
    changes_negative: List[Tuple[List[str], List[float]]] = []
    with torch.no_grad():
        for seq1_idx, seq1 in tqdm(
            enumerate(seq1s),
            desc="Computing predictions",
            colour="blue",
        ):

            # compute average nll
            seq2s_nlls: List[torch.Tensor] = []
            for seq2 in seq2s[(seq1_idx * n_controls) : ((seq1_idx + 1) * n_controls)]:
                seq2s_nlls.append(get_seq_nll(seq2, model, device))

            # seq2s_nlls_tensor = [n_controls, hidden_size]
            seq2s_nlls_tensor: torch.Tensor = torch.cat(seq2s_nlls, dim=0)
            del seq2s_nlls
            # seq2s_nlls_avg = [hidden_size]
            seq2s_nlls_avg: np.ndarray = torch.mean(seq2s_nlls_tensor, dim=0).cpu().numpy()
            del seq2s_nlls_tensor

            # seq1_nll = [hidden_size]
            seq1_nll: np.ndarray = get_seq_nll(seq1, model, device).squeeze(0).cpu().numpy()

            change = change_func(seq1_nll, seq2s_nlls_avg)
            positive_change_indices, positive_change = topk(change, n_predictions)
            negative_change_indices, negative_change = lowk(change, n_predictions)

            positive_change_words = [
                tokenizer.clean_up_tokenization(tok)
                for tok in tokenizer.convert_ids_to_tokens(positive_change_indices)
            ]

            negative_change_words = [
                tokenizer.clean_up_tokenization(tok)
                for tok in tokenizer.convert_ids_to_tokens(negative_change_indices)
            ]

            changes_positive.append((positive_change_words, positive_change.tolist()))
            changes_negative.append((negative_change_words, negative_change.tolist()))

    return {
        "changes_positive": np.array(changes_positive)[seq1s_inv_indices].tolist(),
        "changes_negative": np.array(changes_negative)[seq1s_inv_indices].tolist(),
    }


def convert_to_df(merged_dict: Dict) -> pd.DataFrame:
    """Returns df with prediction changes in each row"""
    return pd.DataFrame(
        {
            "experimentID": merged_dict["experimentID"],
            "sequenceID": merged_dict["sequenceID"],
            "sentenceID": merged_dict["sentenceID"],
            "vignetteID": merged_dict["vignetteID"],
            "changes_positive": [
                list(zip(words, values)) for words, values in merged_dict["changes_positive"]
            ],
            "changes_negative": [
                list(zip(words, values)) for words, values in merged_dict["changes_negative"]
            ],
        }
    )


# ===== RUNTIME CODE WRAPPER ===== #
def compute_predictions_change(
    input_path: str,
    output_path: str,
    model_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: Optional[str] = None,
    overwrite: bool = False,
    experiment_function: Callable[[Any], Dict[str, Any]] = compute_predictions_change_experiment,
    experiment_kwargs: Optional[Dict] = None,
    n_predictions: int = 20,
    change_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = get_change_prob_abs,
    converter_func: Callable[[Dict], pd.DataFrame] = convert_to_df,
    converter_kwargs: Optional[Dict] = None,
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
    if not os.path.isdir(input_path):
        raise ValueError("Input path has to be directory.")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    elif not os.path.isdir(output_path):
        raise ValueError("Output path has to be dir.")

    # Get input files
    input_filenames = [
        path
        for path in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, path)) and path.endswith(".json")
    ]
    seq1s_filenames = sorted(
        [filename for filename in input_filenames if filename.startswith("seq1s")]
    )
    seq2s_filenames = sorted(
        [filename for filename in input_filenames if filename.startswith("seq2s")]
    )
    seq1s_files = [os.path.join(input_path, path) for path in seq1s_filenames]
    seq2s_files = [os.path.join(input_path, path) for path in seq2s_filenames]
    # Create output files
    output_files = [
        os.path.join(
            output_path,
            path.replace(".json", f"_{normalized_model_name}.csv").replace("seq1s", "pred_change"),
        )
        for path in seq1s_filenames
    ]

    if not overwrite:
        for output_file in output_files:
            if os.path.isfile(output_file):
                raise ValueError(
                    f"Output file {output_file} already exists, use `-f` to overwrite."
                )

    logger.info(
        "Computing Prediciton changes for for:\n"
        + "\n".join(f"  - {s2} to {s1}" for s1, s2 in zip(seq1s_files, seq2s_files))
    )

    # Device
    if device is not None:
        if device == "cuda" and not torch.cuda.is_available():
            logger.critical(f"Device {device} not available.")
            raise Exception(f"Device {device} not available.")
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device {device} | n_gpu: {torch.cuda.device_count()}")

    # Model and Tokenizer
    logger.info("Loading Model and Tokenizer.")
    logger.info(f"Full model name: {model_name}")
    logger.info(f"Normalized name: {normalized_model_name}")
    logger.info(f"Use Past Keys: {True}")

    for seq1s_file, seq2s_file, output_file in zip(seq1s_files, seq2s_files, output_files):

        # Data
        logger.info("Loading and processing Inputs")
        seq1s_data_dict = read_sequences(
            file_path=seq1s_file,
            tokenizer=tokenizer,
            return_marked_word_position=True,
        )
        seq2s_data_dict = read_sequences(
            file_path=seq2s_file,
            tokenizer=tokenizer,
            return_marked_word_position=True,
        )

        # Run experiment class
        logger.info("Computing predictions.")
        output_dict = experiment_function(
            seq1s_data_dict=seq1s_data_dict,
            seq2s_data_dict=seq2s_data_dict,
            change_func=change_func,
            n_predictions=n_predictions,
            model=model,
            tokenizer=tokenizer,
            device=device,
            **experiment_kwargs,
        )

        # Handle output data
        logger.info("Postprocessing outputs...")
        # Merge input data with output data
        merged_dict = {**seq1s_data_dict, **output_dict}
        # Convert every token prediction into its own row
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
        help="path/to/ dir containing sequences.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="path/to/ dir for outputs",
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

    compute_predictions_change(
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model_name,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        overwrite=args.force,
    )


# ===== Main ===== #
if __name__ == "__main__":
    main()
