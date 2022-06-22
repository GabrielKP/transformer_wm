import argparse
import os

import pandas as pd
from matplotlib import pyplot as plt

from lm_mem.analysis.plot import (
    plot_repeat_surprisal_violin_from_dfs,
    plot_single_sequence_from_df,
)


def plot_set_size(
    data_dir: str = "data/output/set_size",
    output_dir: str = "plots/$model_name/set_size",
    model_name: str = "gpt2",
    experimentID: int = 0,
    use_mean: bool = False,
    overwrite: bool = False,
) -> None:
    """Plots data from set_size experiment."""

    output_dir = output_dir.replace("$model_name", model_name)

    # 0. Check if directory is empty
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not overwrite and len(os.listdir(output_dir)):
        raise ValueError(f"Output directory {output_dir} is non-empty, use `-f` to overwrite.")

    # 1. Read data
    repeat_names = sorted(
        [
            os.path.join(data_dir, file_name)
            for file_name in os.listdir(data_dir)
            if ("repeat" in file_name and model_name in file_name)
        ]
    )
    repeats = [(pd.read_csv(filepath), filepath) for filepath in repeat_names]
    control_names = sorted(
        [
            os.path.join(data_dir, file_name)
            for file_name in os.listdir(data_dir)
            if ("control" in file_name and model_name in file_name)
        ]
    )
    controls = [(pd.read_csv(filepath), filepath) for filepath in control_names]

    # 2. Plot single sequence for each set size
    for repeat, filepath in repeats:
        repeat_experiment = repeat[repeat["experimentID"] == experimentID]
        name = f"set_size_{filepath.split('_')[-2]}"
        plot_single_sequence_from_df(plt, repeat_experiment, title=name)
        plt.savefig(os.path.join(output_dir, f"{name}.png"), dpi=300)
        plt.close()
    # 3. Plot single sequence for each control
    for control, filepath in controls:
        control_experiment = control[control["experimentID"] == experimentID]
        name = f"set_size_{filepath.split('_')[-2]}"
        plot_single_sequence_from_df(plt, control_experiment, title=f"{name}_control_1")
        plt.savefig(os.path.join(output_dir, f"{name}_control.png"), dpi=300)
        plt.close()

    repeat_dfs, repeat_filepaths = zip(*repeats)
    control_dfs, control_filepaths = zip(*controls)
    repeat_names = [f"{filepath.split('_')[-2]}" for filepath in repeat_filepaths]
    control_names = [f"{filepath.split('_')[-2]}" for filepath in control_filepaths]

    # 4. Plot violin plots
    plot_repeat_surprisal_violin_from_dfs(
        plt,
        repeat_dfs,
        control_dfs,
        [*repeat_names, *control_names],
        ymax=110,
        use_mean=use_mean,
        figsize=(7, 5),
        xlabel="set size",
    )
    plt.savefig(os.path.join(output_dir, "set_size_violin.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/output/set_size",
        help="dir containing only repeats_set_size_{model}.csv and controls_set_size_{model}.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/$model_name/set_size",
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
        "--use_mean",
        action="store_true",
        help="Whether to use mean instead of median in violin plots",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If set, will overwrite existing content in output_dir.",
    )
    args = parser.parse_args()

    plot_set_size(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        experimentID=args.experimentID,
        use_mean=args.use_mean,
        overwrite=args.force,
    )
