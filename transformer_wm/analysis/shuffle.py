import argparse
import os
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt

from transformer_wm.analysis.plot import (
    plot_repeat_surprisal_violin_from_dfs,
    plot_single_sequence_from_df,
)


def plot_shuffle(
    data_dir: str = "data/output/shuffle",
    data_dir_repeat: str = "data/output/repeat",
    data_dir_paraphrase: str = "data/output/paraphrase",
    output_dir: str = "plots/$model_name/shuffle",
    model_name: str = "gpt2",
    experimentID: int = 0,
    use_mean: bool = False,
    overwrite: bool = False,
) -> None:
    """Plots data from repeat experiment."""

    output_dir = output_dir.replace("$model_name", model_name)

    # 0. Check output dir for existing files
    output_path_shuffle_global_both = os.path.join(output_dir, "shuffle_global_both.png")
    output_path_shuffle_global_both_control = os.path.join(
        output_dir, "shuffle_global_both_control.png"
    )
    output_path_shuffle_global_first = os.path.join(output_dir, "shuffle_global_first.png")
    output_path_shuffle_global_first_control = os.path.join(
        output_dir, "shuffle_global_first_control.png"
    )
    output_path_shuffle_global_second = os.path.join(output_dir, "shuffle_global_second.png")
    output_path_shuffle_global_second_control = os.path.join(
        output_dir, "shuffle_global_second_control.png"
    )
    addition = "_mean" if use_mean else ""
    output_path_violin = os.path.join(output_dir, f"shuffle_violin{addition}.png")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not overwrite and (
        os.path.exists(output_path_shuffle_global_both)
        or os.path.exists(output_path_shuffle_global_both_control)
        or os.path.exists(output_path_shuffle_global_first)
        or os.path.exists(output_path_shuffle_global_first_control)
        or os.path.exists(output_path_shuffle_global_second)
        or os.path.exists(output_path_shuffle_global_second_control)
        or os.path.exists(output_path_violin)
    ):
        raise ValueError(
            f"Output directory {output_dir} already contains plots, use `-f` to overwrite."
        )

    # 1. Read data
    repeats = pd.read_csv(f"{data_dir_repeat}/repeats_{model_name}.csv")
    controls = pd.read_csv(f"{data_dir_repeat}/controls_{model_name}.csv")
    paraphrases = pd.read_csv(f"{data_dir_paraphrase}/paraphrases_{model_name}.csv")
    paraphrase_controls = pd.read_csv(f"{data_dir_paraphrase}/controls_{model_name}.csv")
    shuffles_global_both = pd.read_csv(f"{data_dir}/shuffled_global_both_{model_name}.csv")
    shuffles_global_both_control = pd.read_csv(
        f"{data_dir}/shuffled_global_both_control_{model_name}.csv"
    )
    shuffles_global_first = pd.read_csv(f"{data_dir}/shuffled_global_first_{model_name}.csv")
    shuffles_global_first_control = pd.read_csv(
        f"{data_dir}/shuffled_global_first_control_{model_name}.csv"
    )
    shuffles_global_second = pd.read_csv(f"{data_dir}/shuffled_global_second_{model_name}.csv")
    shuffles_global_second_control = pd.read_csv(
        f"{data_dir}/shuffled_global_second_control_{model_name}.csv"
    )

    # 2. plot single sequence
    shuffle_global_both_experiment = shuffles_global_both[
        shuffles_global_both["experimentID"] == experimentID
    ]
    shuffle_global_both_control_experiment = shuffles_global_both_control[
        shuffles_global_both_control["experimentID"] == experimentID
    ]
    shuffle_global_first_experiment = shuffles_global_first[
        shuffles_global_first["experimentID"] == experimentID
    ]
    shuffle_global_first_control_experiment = shuffles_global_first_control[
        shuffles_global_first_control["experimentID"] == experimentID
    ]
    shuffle_global_second_experiment = shuffles_global_second[
        shuffles_global_second["experimentID"] == experimentID
    ]
    shuffle_global_second_control_experiment = shuffles_global_second_control[
        shuffles_global_second_control["experimentID"] == experimentID
    ]
    plot_single_sequence_from_df(plt, shuffle_global_both_experiment, title="Shuffle Global both")
    plt.savefig(output_path_shuffle_global_both, dpi=300)
    plt.close()
    plot_single_sequence_from_df(
        plt, shuffle_global_both_control_experiment, title="Shuffle Global both control"
    )
    plt.savefig(output_path_shuffle_global_both_control, dpi=300)
    plt.close()
    plot_single_sequence_from_df(
        plt, shuffle_global_first_experiment, title="Shuffle Global first"
    )
    plt.savefig(output_path_shuffle_global_first, dpi=300)
    plt.close()
    plot_single_sequence_from_df(
        plt, shuffle_global_first_control_experiment, title="Shuffle Global first control"
    )
    plt.savefig(output_path_shuffle_global_first_control, dpi=300)
    plt.close()
    plot_single_sequence_from_df(
        plt, shuffle_global_second_experiment, title="Shuffle Global second"
    )
    plt.savefig(output_path_shuffle_global_second, dpi=300)
    plt.close()
    plot_single_sequence_from_df(
        plt, shuffle_global_second_control_experiment, title="Shuffle Global second control"
    )
    plt.savefig(output_path_shuffle_global_second_control, dpi=300)
    plt.close()

    # 3. plot violin plot
    plot_repeat_surprisal_violin_from_dfs(
        plt,
        [
            repeats,
            paraphrases,
            shuffles_global_both,
            shuffles_global_first,
            shuffles_global_second,
        ],
        [
            controls,
            paraphrase_controls,
            shuffles_global_both_control,
            shuffles_global_first_control,
            shuffles_global_second_control,
        ],
        [
            "repeat",
            "para-\nphrases",
            "both\nshuffled",
            "first\nshuffled",
            "second\nshuffled",
        ],
        ymax=110,
        use_mean=use_mean,
        figsize=(7, 5),
    )
    plt.tight_layout()
    plt.savefig(output_path_violin, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/output/shuffle",
        help="dir containing the 3 output csv files from shuffle experiment",
    )
    parser.add_argument(
        "--data_dir_repeat",
        type=str,
        default="data/output/repeat",
        help="dir containing repeats_{model}.csv and controls_{model}",
    )
    parser.add_argument(
        "--data_dir_paraphrase",
        type=str,
        default="data/output/paraphrase",
        help="dir containing paraphrases_{model}.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/$model_name/shuffle",
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

    plot_shuffle(
        data_dir=args.data_dir,
        data_dir_repeat=args.data_dir_repeat,
        data_dir_paraphrase=args.data_dir_paraphrase,
        output_dir=args.output_dir,
        model_name=args.model_name,
        experimentID=args.experimentID,
        use_mean=args.use_mean,
        overwrite=args.force,
    )
