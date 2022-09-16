import argparse
import logging
import os

import pandas as pd
from matplotlib import pyplot as plt

from transformer_wm import get_logger
from transformer_wm.analysis.plot import plot_raincloud, plot_single_sequence_from_df, save_fig
from transformer_wm.analysis.utils import get_repeat_surprisal_df

logger = get_logger(__name__)


def plot_repeat(
    data_dir: str = "data/output/repeat",
    output_dir: str = "plots/$model_name/repeat",
    model_name: str = "gpt2",
    experimentID: int = 0,
    use_mean: bool = False,
    overwrite: bool = False,
    skip_examples: bool = False,
) -> None:
    """Plots data from repeat experiment."""

    output_dir = output_dir.replace("$model_name", model_name)

    # 0. Make sure output files do not exist yet
    single_repeat_path = os.path.join(output_dir, "single_repeat.png")
    single_repeat_control_path = os.path.join(output_dir, "single_repeat_control.png")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not overwrite and (
        os.path.exists(single_repeat_path) or os.path.exists(single_repeat_control_path)
    ):
        raise ValueError(
            f"Output directory {output_dir} already contains plots, use `-f` to overwrite."
        )

    # 1. Read data
    repeats = pd.read_csv(f"{data_dir}/seq1s_repeat_{model_name}.csv")
    controls = pd.read_csv(f"{data_dir}/seq2s_repeat_{model_name}.csv")

    # 2. Plot single sequence
    if not skip_examples:
        repeat_experiment = repeats[repeats["experimentID"] == experimentID]
        control_sequenceIDs = (
            controls[controls["experimentID"] == experimentID]["sequenceID"].unique().tolist()
        )
        control_experiment = controls[controls["sequenceID"] == control_sequenceIDs[0]]
        plot_single_sequence_from_df(plt, repeat_experiment, title="seq1")
        save_fig(plt, single_repeat_path)
        plot_single_sequence_from_df(plt, control_experiment, title="seq2")
        save_fig(plt, single_repeat_control_path)

    # 3. plot violin plot
    rs_df = get_repeat_surprisal_df(
        [repeats],
        [controls],
        ["repeat"],
        use_mean,
    )
    logger.info(
        "Repeat Surprisal Values\n"
        f'{rs_df["repeat surprisal"].aggregate(["mean", "median", "min", "max"])}'
    )
    # plot_violin(plt, rs_df)
    ax = plot_raincloud(
        data=rs_df,
        figsize=(5, 7),
        font_scale=1.1,
    )
    plt.title("Verbatim Recall of sentences in transformers")
    plt.tight_layout()
    save_fig(plt, os.path.join(output_dir, "repeat_plot.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/output/repeat",
        help="dir containing repeats_{model}.csv and controls_{model}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/$model_name/repeat",
        help="dir in which plots are placed into.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="From which model the experiment data is used.",
    )
    parser.add_argument(
        "--experimentID",
        type=int,
        default=0,
        help="ID of specific experiment for which single sequences are plotted.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If set, will overwrite existing content in output_dir.",
    )
    args = parser.parse_args()

    plot_repeat(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        experimentID=args.experimentID,
        overwrite=args.force,
    )
