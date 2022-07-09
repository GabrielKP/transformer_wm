import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

from transformer_wm import get_logger
from transformer_wm.analysis.plot import plot_boxplot, plot_catplot, plot_raincloud, plot_raincloud2
from transformer_wm.analysis.utils import get_repeat_surprisal_df, get_repeat_surprisal_df_across_models
from transformer_wm.data.reader import load_models

logger = get_logger(__name__)


def plot_word_swap(
    output_path: str = "plots/word_swap.png",
    model_names: str = "all",
    use_mean: bool = False,
    overwrite: bool = False,
    data_dir_word_swap: str = "data/output/word_swap",
    kind="bar",
    old=False,
) -> None:
    """Plots data from set_size experiment."""

    if model_names == "all":
        model_names = load_models()
    else:
        if not isinstance(model_names, list):
            raise ValueError(f"model_names has to be list, but is {type(model_names)}")

    # 0. Check if directory is empty
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    elif not overwrite and os.path.exists(output_path):
        raise ValueError(f"Output file {output_path} already exists, use `-f` to overwrite.")

    if old:
        experiments_labels = [
            ("rpt_noun", ("RPT_noun")),
            ("rpt_verb", ("RPT_verb")),
            ("syn_noun", ("SYN_noun")),
            ("syn_verb", ("SYN_verb")),
            ("arb_noun", ("ARB_noun")),
            ("arb_verb", ("ARB_verb")),
        ]

    else:
        experiments_labels = [
            ("rpt_noun", ("RPT", "noun")),
            ("rpt_verb", ("RPT", "verb")),
            ("syn_noun", ("SYN", "noun")),
            ("syn_verb", ("SYN", "verb")),
            ("arb_noun", ("ARB", "noun")),
            ("arb_verb", ("ARB", "verb")),
        ]

    # 1. Read data and accumulate repeat surprisal values
    word_surprisal_df = get_df_across_models(
        model_names=model_names,
        data_dir=data_dir_word_swap,
        experiments=experiments_labels,
        use_mean=use_mean,
    )

    if old:
        logger.info(
            "Repeat Surprisal Values\n"
            f'{word_surprisal_df.groupby(["condition"])["repeat surprisal"].aggregate(["mean", "median", "min", "max"])}'
        )
    else:
        logger.info(
            "Repeat Surprisal Values\n"
            f'{word_surprisal_df.groupby(["condition", "subcondition"])["repeat surprisal"].aggregate(["mean", "median", "min", "max"])}'
        )

    if old:
        # 2. plot
        plt.title("Recall of transformers for repeated, synonym and arbitrary word presentation.")
        ax = plot_catplot(
            plt,
            word_surprisal_df,
            # ymax=200,
            kind=kind,
        )
        # ax = plot_raincloud2(
        #     hue="model",
        #     data=word_surprisal_df,
        #     figsize=(10, 10),
        #     font_scale=1.1,
        #     ylim=260,
        # )
    else:
        ax = plot_raincloud2(
            hue="subcondition",
            data=word_surprisal_df,
            figsize=(10, 10),
            font_scale=1.1,
            ylim=260,
        )
        ax.get_legend().set_title("marked word")
        plt.title("Recall of transformers for repeated, synonym and arbitrary word presentation.")
        plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def load_df(data_dir: str, experiment: str, model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        seq1 = pd.read_csv(f"{data_dir}/seq1s_{experiment}_{model_name}.csv")
        seq2 = pd.read_csv(f"{data_dir}/seq2s_{experiment}_{model_name}.csv")
        return seq1, seq2
    except FileNotFoundError:
        logger.warning(f"Cannot find {experiment} output for {model_name}")
        return None, None


def compute_marked_repeat_surprisal(seq1_df: pd.DataFrame, seq2_df: pd.DataFrame, use_mean: bool):
    """Returns Series of repeat surprisals at markers

    seq1_df: DataFrame
        containing surprisal values for seq1. Each seq1 should have a unique experimentID
    seq2_df: DataFrame
        containing surprisal values for seq2. seq2s with same experimentIDs will be averaged.
        Has to contain exactly the same unique experimentIDs as seq1_df.

    """
    # Group by sequenceID and sectionID
    grouped_data = seq1_df.groupby(["sequenceID", "experimentID", "marked_word"])["surprisal"]
    grouped_control = seq2_df.groupby(["sequenceID", "experimentID", "marked_word"])["surprisal"]
    # Get surprisals over marked words
    if use_mean:
        m_surprisal_data = grouped_data.mean().droplevel(level=0)
        m_surprisal_control = grouped_control.mean()
    else:
        m_surprisal_data = grouped_data.median().droplevel(level=0)
        m_surprisal_control = grouped_control.median()
    # Average over controls belonging to same test_sequence
    m_surprisal_control = m_surprisal_control.groupby(["experimentID", "marked_word"]).mean()
    # TODO: sort by experimentID
    # Compute repeat surprisal
    repeat_surprisal = 100 * m_surprisal_data[:, 1] / m_surprisal_control[:, 1]

    return repeat_surprisal


def get_df_across_models(
    model_names: List[str],
    data_dir: str,
    experiments: List[Union[str, Tuple[str, Tuple[str, str]]]],
    use_mean: bool = False,
):
    """Returns repeat surprisal values for each experiment."""
    df_accumulator: List[pd.DataFrame] = []

    for model_name in model_names:
        model_name = model_name.split("/")[-1]
        seq1_dfs: List[pd.DataFrame] = []
        seq2_dfs: List[pd.DataFrame] = []
        condition_names: List[str] = []

        for experiment in experiments:
            if isinstance(experiment, tuple):
                experiment_name, condition_name = experiment
            else:
                experiment_name = experiment
                condition_name = experiment.replace("_", "\n")
            seq1, seq2 = load_df(
                data_dir=data_dir,
                experiment=experiment_name,
                model_name=model_name,
            )
            if seq1 is not None and seq2 is not None:
                seq1_dfs.append(seq1)
                seq2_dfs.append(seq2)
                condition_names.append(condition_name)

        # Get the models repeat surprisal values for each condition
        if len(seq1_dfs):
            repeat_surprisal_df = get_repeat_surprisal_df(
                seq1_dfs=seq1_dfs,
                seq2_dfs=seq2_dfs,
                condition_names=condition_names,
                use_mean=use_mean,
                compute_function=compute_marked_repeat_surprisal,
            )
            # Annotate with model name
            repeat_surprisal_df["model"] = model_name
            df_accumulator.append(repeat_surprisal_df)

    return pd.concat(df_accumulator)


if __name__ == "__main__":
    plot_word_swap(overwrite=True, model_names="all", old=True)
    plot_word_swap(overwrite=True, model_names=["gpt2"], old=False)
