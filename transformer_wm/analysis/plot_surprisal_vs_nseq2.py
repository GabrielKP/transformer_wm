import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from transformer_wm import get_logger
from transformer_wm.analysis.plot import save_fig

logger = get_logger(__name__)


def plot_surprisal_vs_nseq2(
    data_dir: str = "data/output/repeat_50_seq2s",
    output_dir: str = "plots",
    model_name: str = "gpt2",
    use_mean: bool = False,
    overwrite: bool = False,
    figsize: Tuple[int, int] = (7, 5),
) -> None:

    # 0. Make sure output files do not exist yet
    surprisal_nseq2_path = os.path.join(output_dir, "influence_50_control_seqs_on_surprisal.png")
    surprisal_nseq2_closeup_path = os.path.join(
        output_dir, "influence_50_control_seqs_on_surprisal_closeup.png"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not overwrite and (
        os.path.exists(surprisal_nseq2_path) or os.path.exists(surprisal_nseq2_closeup_path)
    ):
        raise ValueError(
            f"Output directory {output_dir} already contains plots, use `-f` to overwrite."
        )

    # 1. Read data
    controls = pd.read_csv(f"{data_dir}/seq2s_repeat_{model_name}.csv")

    # 2. get average surprisal for amount of controls
    # ASSUMES EVERY EXPERIMENT HAS THE SAME AMOUNT OF CONTROLS
    experimentIDs = controls["experimentID"].unique().tolist()
    first_seq2_group = controls[controls["experimentID"] == experimentIDs[0]]
    n_controls = len(first_seq2_group["sequenceID"].unique().tolist())

    groups = controls.groupby(by=["sequenceID", "experimentID", "sectionID"])["surprisal"]
    groups = groups.mean() if use_mean else groups.median()

    df_accumulator: List[pd.DataFrame] = []
    for _ in range(21):
        for num in range(1, n_controls + 1):
            avg_surprisals: List[float] = [
                groups[:, eID, 2].sample(num).mean() for eID in experimentIDs
            ]
            temporary_df = pd.DataFrame(
                {"average surprisal": avg_surprisals, "experimentID": experimentIDs}
            )
            temporary_df["n seq2s"] = num
            df_accumulator.append(temporary_df)

    logger.info("plotting")
    data_df = pd.concat(df_accumulator).reset_index()
    _, ax = plt.subplots(figsize=figsize)
    graph = sns.lineplot(
        x="n seq2s",
        y="average surprisal",
        hue="experimentID",
        data=data_df,
    )
    graph.axhline(100, ls=":", color=".5")
    ax.set_ybound(0, 110)
    save_fig(plt, surprisal_nseq2_path)
    plt.close()
    _, ax = plt.subplots(figsize=figsize)
    graph = sns.lineplot(
        x="n seq2s",
        y="average surprisal",
        hue="experimentID",
        data=data_df,
    )
    save_fig(plt, surprisal_nseq2_closeup_path)
    plt.close()


def plot_repeat_surprisal_vs_nseq2(
    data_dir: str = "data/output/repeat_50_seq2s",
    output_dir: str = "plots",
    model_name: str = "gpt2",
    use_mean: bool = False,
    overwrite: bool = False,
    figsize: Tuple[int, int] = (7, 5),
) -> None:

    # 0. Make sure output files do not exist yet
    rsurprisal_nseq2_path = os.path.join(
        output_dir, "influence_50_control_seqs_on_repeat_surprisal.png"
    )
    rsurprisal_nseq2_closeup_path = os.path.join(
        output_dir, "influence_50_control_seqs_on_repeat_surprisal_closeup.png"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not overwrite and (
        os.path.exists(rsurprisal_nseq2_path) or os.path.exists(rsurprisal_nseq2_closeup_path)
    ):
        raise ValueError(
            f"Output directory {output_dir} already contains plots, use `-f` to overwrite."
        )

    # 1. Read data
    repeats = pd.read_csv(f"{data_dir}/seq1s_repeat_{model_name}.csv")
    controls = pd.read_csv(f"{data_dir}/seq2s_repeat_{model_name}.csv")

    # 2. get average surprisal for amount of controls
    # ASSUMES EVERY EXPERIMENT HAS THE SAME AMOUNT OF CONTROLS
    experimentIDs = controls["experimentID"].unique().tolist()
    first_seq2_group = controls[controls["experimentID"] == experimentIDs[0]]
    n_controls = len(first_seq2_group["sequenceID"].unique().tolist())

    groups = controls.groupby(by=["sequenceID", "experimentID", "sectionID"])["surprisal"]
    groups = groups.mean() if use_mean else groups.median()
    seq1_groups = repeats.groupby(by=["sequenceID", "experimentID", "sectionID"])["surprisal"]
    seq1_groups = seq1_groups.mean() if use_mean else seq1_groups.median()

    df_accumulator: List[pd.DataFrame] = []
    for _ in range(21):
        for num in range(1, n_controls + 1):
            avg_surprisals: List[float] = [
                seq1_groups[:, eID, 2].mean() / groups[:, eID, 2].sample(num).mean()
                for eID in experimentIDs
            ]
            temporary_df = pd.DataFrame(
                {"average repeat surprisal": avg_surprisals, "experimentID": experimentIDs}
            )
            temporary_df["n seq2s"] = num
            df_accumulator.append(temporary_df)

    logger.info("plotting")
    data_df = pd.concat(df_accumulator).reset_index()
    _, ax = plt.subplots(figsize=figsize)
    graph = sns.lineplot(
        x="n seq2s",
        y="average repeat surprisal",
        # hue="experimentID",
        data=data_df,
    )
    graph.axhline(100, ls=":", color=".5")
    ax.set_ybound(0, 110)
    save_fig(plt, rsurprisal_nseq2_path)
    plt.close()
    _, ax = plt.subplots(figsize=figsize)
    graph = sns.lineplot(
        x="n seq2s",
        y="average repeat surprisal",
        # hue="experimentID",
        data=data_df,
    )
    save_fig(plt, rsurprisal_nseq2_closeup_path)
    plt.close()


if __name__ == "__main__":
    plot_surprisal_vs_nseq2(overwrite=True)
    plot_repeat_surprisal_vs_nseq2(overwrite=True)
