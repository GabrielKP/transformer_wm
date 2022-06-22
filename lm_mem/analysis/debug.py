import argparse
import logging
import os

import pandas as pd
from matplotlib import pyplot as plt

from lm_mem import get_logger
from lm_mem.analysis.plot import (
    plot_repeat_surprisal_violin_from_dfs,
    plot_single_sequence_from_df,
)

logger = get_logger(__name__)


def plot_debug(
    data_dir: str = "data/output/debug",
    output_dir: str = "plots/debug",
    model_name: str = "gpt2",
    experimentID: int = 0,
    overwrite: bool = False,
) -> None:
    """Plots debug."""

    model_name = model_name.split("/")[-1]

    output_dir = output_dir.replace("$model_name", model_name)

    # 0. Make sure output files do not exist yet
    single_repeat_path = os.path.join(output_dir, f"single_repeat_{model_name}.png")
    single_repeat_control_path = os.path.join(
        output_dir, f"single_repeat_control_{model_name}.png"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not overwrite and (
        os.path.exists(single_repeat_path) or os.path.exists(single_repeat_control_path)
    ):
        raise ValueError(
            f"Output directory {output_dir} already contains plots, use `-f` to overwrite."
        )

    # 1. Read data
    repeats = pd.read_csv(f"{data_dir}/repeats_{model_name}.csv")
    controls = pd.read_csv(f"{data_dir}/controls_{model_name}.csv")

    # 2. Plot single sequence
    repeat_experiment = repeats[repeats["experimentID"] == experimentID]
    control_experiment = controls[controls["experimentID"] == experimentID]
    plot_single_sequence_from_df(plt, repeat_experiment, title="Repeat Condition")
    plt.savefig(single_repeat_path, dpi=300)
    plot_single_sequence_from_df(plt, control_experiment, title="Repeat Condition: Control 1")
    plt.savefig(single_repeat_control_path, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/output/debug",
        help="dir containing repeats_{model}.csv and controls_{model}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/debug",
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

    plot_debug(
        args.data_dir,
        args.output_dir,
        args.model_name,
        args.experimentID,
        args.force,
    )
