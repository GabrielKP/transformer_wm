import argparse
import logging
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from transformer_wm import get_logger
from transformer_wm.analysis.plot import save_fig
from transformer_wm.analysis.utils import get_repeat_surprisal_df

logger = get_logger(__name__)


def plot_example(
    out: plt,
    x_positions,
    y_surprisals,
    sectionIDs,
    x_tokens,
    figure_height=1.5,
    figure_width=15,
    ylim=None,
    is_control=False,
) -> None:
    """Plots a single sequence with their surprisal values."""

    f, a = out.subplots(figsize=(figure_width, figure_height))

    a.plot(x_positions, y_surprisals, marker="o", linestyle="--", color="darkblue")

    if ylim is None:
        ylim = a.get_ylim()

    x_rect2 = np.where(sectionIDs == 3)[0][0]
    y_rect = ylim[0]

    a.add_patch(
        Rectangle(
            xy=(x_rect2 - 0.5, y_rect),
            width=len(x_tokens[sectionIDs == 3]),
            height=ylim[-1] + 0.5,
            edgecolor=None,
            facecolor="tab:blue",
            alpha=0.15,
        )
    )

    # Show tokens
    a.set_xticks(x_positions)
    a.set_xticklabels(x_tokens, rotation=40, fontsize=12, ha="right")

    # Color tokens belonging to sentences
    blue_tokens = np.isin(sectionIDs, [3])
    for idx, tick in enumerate(a.xaxis.get_ticklabels()):
        if blue_tokens[idx]:
            tick.set_color("tab:blue")

    color = "tab:red" if is_control else "tab:green"

    blue_tokens = np.isin(sectionIDs, [1])
    for idx, tick in enumerate(a.xaxis.get_ticklabels()):
        if blue_tokens[idx]:
            tick.set_color(color)

    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    # a.set(ylabel=ylabel, title=title)
    out.tight_layout()

    return f, a


def plot_examples(
    data_dir: str = "data/output/repeat",
    output_dir: str = "plots/$model_name/repeat",
    model_name: str = "gpt2",
    t_sequenceID: int = 0,
    c_sequenceID: int = 10,
    overwrite: bool = False,
) -> None:
    """Plots data from repeat experiment."""

    output_dir = output_dir.replace("$model_name", model_name)

    # 0. Make sure output files do not exist yet
    example_t_seq_path = os.path.join(output_dir, "example_t_sequence.png")
    example_c_seq_path = os.path.join(output_dir, "example_c_sequence.png")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not overwrite and (
        os.path.exists(example_t_seq_path) or os.path.exists(example_c_seq_path)
    ):
        raise ValueError(
            f"Output directory {output_dir} already contains plots, use `-f` to overwrite."
        )

    # 1. Read data
    seq1s = pd.read_csv(f"{data_dir}/seq1s_repeat_{model_name}.csv")
    seq2s = pd.read_csv(f"{data_dir}/seq2s_repeat_{model_name}.csv")

    # 2. Plot single sequence
    example_t_seq = seq1s[seq1s["sequenceID"] == t_sequenceID]
    example_c_seq = seq2s[seq2s["sequenceID"] == c_sequenceID]

    plot_example(
        plt,
        range(len(example_t_seq["surprisal"])),
        example_t_seq["surprisal"].to_numpy(),
        example_t_seq["sectionID"].to_numpy(),
        example_t_seq["word"].to_numpy(),
        figure_height=4,
        figure_width=25,
    )
    save_fig(plt, example_t_seq_path)

    plot_example(
        plt,
        range(len(example_c_seq["surprisal"])),
        example_c_seq["surprisal"].to_numpy(),
        example_c_seq["sectionID"].to_numpy(),
        example_c_seq["word"].to_numpy(),
        figure_height=4,
        figure_width=25,
        is_control=True,
    )
    save_fig(plt, example_c_seq_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="From which model the experiment data is used.",
    )
    parser.add_argument(
        "--t_sequenceID",
        type=int,
        default=0,
        help="ID of specific experiment for which single sequences are plotted.",
    )
    parser.add_argument(
        "--c_sequenceID",
        type=int,
        default=10,
        help="ID of specific experiment for which single sequences are plotted.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If set, will overwrite existing content in output_dir.",
    )
    args = parser.parse_args()

    plot_examples(
        model_name=args.model_name,
        t_sequenceID=args.t_sequenceID,
        c_sequenceID=args.c_sequenceID,
        overwrite=args.force,
    )
