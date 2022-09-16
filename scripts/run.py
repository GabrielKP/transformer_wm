"""Runs everything with default paths.
"""

import argparse
import re
from typing import Callable, Dict, List, Optional, Union

from transformer_wm import get_logger
from transformer_wm.analysis.plot_example_sequence import plot_examples
from transformer_wm.analysis.plot_multiple import plot_multiple
from transformer_wm.analysis.plot_repeat import plot_repeat
from transformer_wm.analysis.plot_surprisal_vs_nseq2 import (
    plot_repeat_surprisal_vs_nseq2,
    plot_surprisal_vs_nseq2,
)
from transformer_wm.analysis.plot_word_swap import plot_word_swap
from transformer_wm.analysis.predictions_pretty import predictions_pretty
from transformer_wm.analysis.prob_change_summary import print_summary
from transformer_wm.data.create_repeat import create_repeat
from transformer_wm.data.create_repeat_predictions import create_repeat_predictions
from transformer_wm.data.create_word_swap import create_word_swap
from transformer_wm.data.reader import load_models
from transformer_wm.predictions_change import compute_predictions_change
from transformer_wm.surprisal import compute_surprisal, init_model_and_tokenizer

logger = get_logger(__name__)


def create_all_data():
    """Creates data for all experiments."""
    create_repeat(overwrite=True)
    create_repeat(output_dir="data/input/repeat_50_seq2s", overwrite=True, n_seq2s=50)
    create_word_swap(overwrite=True)
    create_repeat_predictions(overwrite=True)


def single_experiment(
    sel_regex: re.Pattern,
    regex_to_match: str,
    input_path: str,
    output_path: str,
    function_kwargs: Dict,
    current_model_name: str,
    function: Callable = compute_surprisal,
):
    "wrapper to run a surprisal experiment"
    if sel_regex.match(regex_to_match):
        # Baseline experiment: Correct nonce data
        logger.info(f"Running {regex_to_match}-experiment for {current_model_name}")
        function(
            input_path=input_path,
            output_path=output_path,
            **function_kwargs,
        )


def run_everything(
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
        surprisal_kwargs = {
            "model_name": current_model_name,
            "model": model,
            "tokenizer": tokenizer,
            "batch_size": batch_size,
            "device": device,
            "overwrite": True,
        }
        if current_model_name == "xlnet-base-cased":
            surprisal_kwargs["mem_instead_of_past_keys"] = True
        single_experiment(
            sel_regex,
            regex_to_match="repeat",
            input_path="data/input/repeat",
            output_path="data/output/repeat/",
            function_kwargs=surprisal_kwargs,
            current_model_name=current_model_name,
        )
        single_experiment(
            sel_regex,
            regex_to_match="word_swap",
            input_path="data/input/word_swap/",
            output_path="data/output/word_swap/",
            function_kwargs={**surprisal_kwargs, "return_marked_words": True},
            current_model_name=current_model_name,
        )
        single_experiment(
            sel_regex,
            regex_to_match="repeat_predictions_change",
            input_path="data/input/repeat_predictions/",
            output_path="data/output/repeat_predictions_change/",
            function_kwargs={k: v for k, v in surprisal_kwargs.items() if k != "batch_size"},
            current_model_name=current_model_name,
            function=compute_predictions_change,
        )


def plot_everything():

    # Figure 2: example timeseries plots, have to be from gpt2
    plot_examples(output_dir="plots/", model_name="gpt2", overwrite=True)

    # Figure 3: gpt2 repeat condition repeat surprisal
    plot_repeat(output_dir="plots/", model_name="gpt2", overwrite=True, skip_examples=True)

    # Figure 5: gpt2 word swap experiment
    plot_word_swap(
        output_path="plots/word_swap_gpt2.png", model_names=["gpt2"], old=False, overwrite=True
    )

    # Table 2: Probability change data
    predictions_pretty(output_dir="plots/")

    # Table 3: Probability change summary
    # analysis was done by hand, results found in data/output/prob_change_analysis/
    print_summary(output_dir="plots/")

    # Appendix, Figure 7
    plot_surprisal_vs_nseq2(output_dir="plots/", model_name="gpt2", overwrite=True)
    plot_repeat_surprisal_vs_nseq2(output_dir="plots/", model_name="gpt2", overwrite=True)

    # Appendix, Figure 8
    plot_multiple(output_path="plots/repeat_across_transformers_plot.png", overwrite=True)

    # Appendix, Figure 9
    plot_word_swap(
        output_path="plots/word_swap_across_transformers.png",
        model_names="all",
        old=True,
        overwrite=True,
    )

    logger.info("All results have been saved to `plots/`")


def run_thesis(
    batch_size: int = 24,
    device: Optional[str] = None,
):
    """Replicates thesis results.

    Parameters
    ----------
    batch_size : int, default=24
        Determines batch size for model.
    device : str, optional, default=`None`
        torch device identifier (mostly `cuda` or `cpu`)
    """

    # 1. create the necessary data
    # create_all_data()
    # Note: Data is already in the repo, but can be recreated with 'python script/run.py create'

    # 2. Run experiments
    run_everything(
        batch_size=batch_size,
        model_name="all",
        device=device,
        selection=None,
    )

    # 2.5 run experiment with 50 controls for appendix Fig 7
    model, tokenizer = init_model_and_tokenizer("gpt2")
    compute_surprisal(
        input_path="data/input/repeat_50_seq2s/",
        output_path="data/output/repeat_50_seq2s/",
        model_name="gpt2",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
        overwrite=True,
    )

    # 3. Result analysis and plotting
    plot_everything()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="commands",
        description="commands you can execute with the script",
    )

    # ------------------ Thesis replication subcommand
    parser_thesis = subparsers.add_parser("thesis")
    parser_thesis.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device program is run on (cpu/cuda)",
    )
    parser_thesis.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Batch size for all experiments.",
    )
    parser_thesis.set_defaults(func=run_thesis)

    # ------------------ Create subcommand
    parser_create = subparsers.add_parser("create")
    parser_create.set_defaults(func=create_all_data)

    # ------------------ Run subcommand
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
    parser_run.set_defaults(func=run_everything)

    # ------------------ Plot subcommand
    parser_plot = subparsers.add_parser("plot")
    parser_plot.set_defaults(func=plot_everything)

    args = parser.parse_args()
    args = args.__dict__
    func = args.pop("func")
    func()
