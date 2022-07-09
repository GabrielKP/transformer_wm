"""Runs everything with default paths.
"""

import argparse
import re
from typing import Dict, List, Optional, Union

from transformer_wm import get_logger
from transformer_wm.analysis.paraphrase import (
    plot_paraphrase,
    plot_same_vs_different_words,
    plot_shared_unshared_followsshared,
    plot_surprisal_by_length,
)
from transformer_wm.analysis.plot_multiple import plot_multiple
from transformer_wm.analysis.plot_repeat import plot_repeat
from transformer_wm.analysis.set_size import plot_set_size
from transformer_wm.analysis.shuffle import plot_shuffle
from transformer_wm.data.create_paraphrase import create_paraphrase
from transformer_wm.data.create_repeat import create_repeat
from transformer_wm.data.create_set_size import create_set_sizes
from transformer_wm.data.create_shuffle import create_shuffle
from transformer_wm.data.reader import load_models
from transformer_wm.surprisal import compute_surprisal, init_model_and_tokenizer

logger = get_logger(__name__)


def create_all_data():
    """Creates data for all experiments."""
    pass


def single_experiment(
    sel_regex: re.Pattern,
    regex_to_match: str,
    input_path: str,
    output_path: str,
    surprisal_kwargs: Dict,
    current_model_name: str,
):
    try:
        if sel_regex.match(regex_to_match):
            # Baseline experiment: Correct nonce data
            compute_surprisal(
                input_path=input_path,
                output_path=output_path,
                **surprisal_kwargs,
            )
    except Exception as err:
        logger.warning(
            f"Running {regex_to_match} Experiment for {current_model_name} failed:"
            + "\n"
            + str(err)
            + "\nContinuing"
        )
        logger.warning("\nContinuing")


def run_custom(
    batch_size: int = 24,
    model_name: Union[str, List[str]] = "all",
    device: Optional[str] = None,
    selection: str = None,
):
    if not isinstance(model_name, list):
        if model_name == "all":
            models = load_models()
        else:
            models = [model_name]
    logger.info(f"Running experiments on {models}")

    if selection is None:
        selection = ".*"
    sel_regex = re.compile(selection)

    for current_model_name in models:
        logger.info(f"---- Running surprisal experiments on {current_model_name}")
        model, tokenizer = init_model_and_tokenizer(current_model_name)
        normalized_model_name = current_model_name.split("/")[-1]
        surprisal_kwargs = {
            "model_name": current_model_name,
            "model": model,
            "tokenizer": tokenizer,
            "batch_size": batch_size,
            "device": device,
            "overwrite": True,
        }
        for pos in ["noun", "verb"]:
            single_experiment(
                sel_regex,
                regex_to_match="word_swap",
                input_path=f"data/input/word_swap/seq1s_semantic_{pos}.json",
                output_path=f"data/output/word_swap/seq1s_semantic_{pos}_{normalized_model_name}.csv",
                surprisal_kwargs={**surprisal_kwargs, "return_marked_words": True},
                current_model_name=current_model_name,
            )
            single_experiment(
                sel_regex,
                regex_to_match="word_swap",
                input_path=f"data/input/word_swap/seq2s_semantic_{pos}.json",
                output_path=f"data/output/word_swap/seq2s_semantic_{pos}_{normalized_model_name}.csv",
                surprisal_kwargs={**surprisal_kwargs, "return_marked_words": True},
                current_model_name=current_model_name,
            )


def plot_everything(model_name):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="commands",
        description="commands you can execute with the script",
    )

    # Create subcommand
    parser_create = subparsers.add_parser("create")
    parser_create.set_defaults(func=create_all_data)

    # Run subcommand
    parser_run = subparsers.add_parser("run")
    parser_run.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device program is run on (cpu/cuda)",
    )
    parser_run.add_argument(
        "--model_name",
        type=str,
        default="all",
        help="Model type on which to run the experiment. If 'all' is given, loads models in model_file",
    )
    parser_run.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Batch size for all experiments.",
    )
    parser_run.add_argument(
        "-k",
        "--selection",
        type=str,
        default=None,
        help="regex to select experiment to run.",
    )
    parser_run.set_defaults(func=run_custom)

    # Plot subcommand
    parser_plot = subparsers.add_parser("plot")
    parser_plot.add_argument(
        "--model_name",
        type=str,
        default="all",
        help="Model name on which experiment ran. If 'all' is given, loads models in model_file",
    )
    parser_plot.set_defaults(func=plot_everything)

    args = parser.parse_args()
    args = args.__dict__
    func = args.pop("func")
    func(**args)
