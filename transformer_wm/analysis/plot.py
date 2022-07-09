"""Plotting functions
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from transformer_wm import get_logger

logger = get_logger(__name__)


def plot_single_sequence_from_df(
    out: plt,
    sequence_df: pd.DataFrame,
    title: str = "",
) -> None:
    plot_single_sequence(
        out,
        range(len(sequence_df["surprisal"])),
        sequence_df["surprisal"].to_numpy(),
        sequence_df["sectionID"].to_numpy(),
        sequence_df["word"].to_numpy(),
        "surprisal",
        title=title,
        figure_height=4,
        figure_width=25,
    )


def plot_single_sequence(
    out: plt,
    x_positions,
    y_surprisals,
    sectionIDs,
    x_tokens,
    ylabel,
    figure_height=1.5,
    figure_width=15,
    ylim=None,
    title="Surprisal",
) -> None:
    """Plots a single sequence with their surprisal values."""

    f, a = out.subplots(figsize=(figure_width, figure_height))

    a.plot(x_positions, y_surprisals, marker="o", linestyle="--", color="darkblue")

    if ylim is None:
        ylim = a.get_ylim()

    x_rect1 = np.where(sectionIDs == 1)[0][0]
    x_rect2 = np.where(sectionIDs == 3)[0][0]
    y_rect = ylim[0]

    # Add Background blue rectangles
    # a.add_patch(
    #     Rectangle(
    #         xy=(x_rect1 - 0.5, y_rect),
    #         width=len(x_tokens[sectionIDs == 1]),
    #         height=ylim[-1] + 0.5,
    #         edgecolor=None,
    #         facecolor="tab:blue",
    #         alpha=0.15,
    #     )
    # )
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
    blue_tokens = np.isin(sectionIDs, [1, 3])
    for idx, tick in enumerate(a.xaxis.get_ticklabels()):
        if blue_tokens[idx]:
            tick.set_color("tab:blue")

    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.set(ylabel=ylabel, title=title)
    out.tight_layout()

    return f, a


def plot_repeat_surprisal_violin_from_dfs(
    out: plt,
    data_dfs: List[pd.DataFrame],
    control_dfs: List[pd.DataFrame],
    names: List[str],
    use_mean: bool = False,
    **plt_violin_kwargs,
) -> None:
    """Computes surprisal values from dfs and plots them."""

    df_accumulator: List[pd.DataFrame] = []
    for name, data_df, control_df in zip(names, data_dfs, control_dfs):
        # Group by sequenceID and sectionID
        grouped_data = data_df.groupby(["sequenceID", "sectionID", "experimentID"])["surprisal"]
        grouped_control = control_df.groupby(["sequenceID", "sectionID", "experimentID"])[
            "surprisal"
        ]
        # Get median surprisals
        if use_mean:
            m_surprisal_data = grouped_data.mean().droplevel(level=0)
            m_surprisal_control = grouped_control.mean().droplevel(level=0)
        else:
            m_surprisal_data = grouped_data.median().droplevel(level=0)
            m_surprisal_control = grouped_control.median().droplevel(level=0)
        # Compute repeat surprisal
        repeat_surprisal = 100 * m_surprisal_data[3] / m_surprisal_control[3]
        # Average same experiments
        repeat_averaged = repeat_surprisal.groupby("experimentID").mean()
        # Save df
        condition_df = pd.DataFrame({"repeat surprisal": repeat_averaged})
        condition_df["condition"] = name
        df_accumulator.append(condition_df)

    result_df = pd.concat(df_accumulator)

    plot_violin(out, result_df, **plt_violin_kwargs)


def plot_violin(
    out: plt,
    data: pd.DataFrame,
    ymax: int = 200,
    figsize: Tuple[int, int] = (5, 7),
    xlabel: str = "condition",
) -> None:
    """Plots data into a violin-plot.

    Parameters
    ----------
    out : ``plt``
        plt object to plot upon
    data : ``pd.DataFrame``
        Requires following columns: "condition", "repeat surprisal"
    """

    _, ax = out.subplots(figsize=figsize)
    graph = sns.violinplot(
        x="condition",
        y="repeat surprisal",
        data=data,
        scale="count",
        cut=0,
        width=0.9,
        # inner="quartile",
        # hue="condition",
    )
    graph.axhline(100, ls=":", color=".5")
    ax.set_ylabel("repeat surprisal (%)")
    ax.set_ybound(0, ymax)
    ax.set_xlabel(xlabel)


def plot_boxplot(
    out: plt,
    data: pd.DataFrame,
    ymax: int = 110,
    figsize: Tuple[int, int] = (5, 7),
    xlabel: str = "condition",
) -> None:

    _, ax = out.subplots(figsize=figsize)
    graph = sns.boxplot(
        x="condition",
        y="repeat surprisal",
        hue="model",
        data=data,
    )

    graph.axhline(100, ls=":", color=".5")
    ax.set_ylabel("repeat surprisal (%)")
    ax.set_ybound(0, ymax)
    ax.set_xlabel(xlabel)


def plot_catplot(
    out: plt,
    data: pd.DataFrame,
    ymax: int = 110,
    figsize: Tuple[int, int] = (5, 7),
    xlabel: str = "condition",
    kind: str = "bar",
) -> None:

    _, ax = out.subplots(figsize=figsize)
    graph = sns.catplot(
        kind=kind,
        x="condition",
        y="repeat surprisal",
        hue="model",
        data=data,
        ci=95,
        legend_out=True,
    )
    graph.refline(y=100)


def save_fig(out: plt, path: str):
    logger.info(f"Saving figure to {path}")
    out.savefig(path, dpi=300)


def plot_raincloud(
    data: pd.DataFrame,
    x="condition",
    y="repeat surprisal",
    hue=None,
    bw=0.3,
    scale="area",
    width=0.6,
    font_scale=1,
    ylim=110,
    **fig_kwargs,
) -> None:
    sns.set(style="whitegrid", font_scale=font_scale)
    palette = sns.color_palette(palette="Set2")
    cut = 0.0
    inner = None

    fig, ax = plt.subplots(**fig_kwargs)

    ax = pt.half_violinplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        palette=palette,
        bw=bw,
        cut=cut,
        scale=scale,
        width=width,
        inner=inner,
    )

    ax = pt.stripplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        palette=palette,
        edgecolor="white",
        size=3,
        jitter=0.1,
        zorder=1,
    )

    ax = sns.pointplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        # palette=palette,
        ci=95,
        scale=0.5,
        color="black",
        zorder=2,
    )

    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.axhline(100, ls=":", color="black")
    xlim = list(ax.get_xlim())
    xlim[-1] -= (width) / 2.0
    ax.set_xlim(xlim)
    ax.set_ylabel("repeat surprisal\n(%)")

    return ax


def plot_raincloud2(
    data: pd.DataFrame,
    x="condition",
    y="repeat surprisal",
    hue=None,
    bw=0.3,
    scale="area",
    width=0.6,
    font_scale=1,
    ylim=110,
    **fig_kwargs,
) -> None:
    sns.set(style="whitegrid", font_scale=font_scale)
    palette = sns.color_palette(palette="pastel")
    cut = 0.0
    inner = None
    width_strip = 0.2

    fig, ax = plt.subplots(**fig_kwargs)

    ax = pt.half_violinplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        palette=sns.color_palette(palette="Set2"),
        bw=bw,
        cut=cut,
        scale=scale,
        width=width,
        inner=inner,
        ax=ax,
        split=True,
        alpha=0.5,
    )

    ax = pt.stripplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        palette=sns.color_palette(palette="Set2"),
        edgecolor="white",
        size=3.4,
        jitter=0.1,
        zorder=1,
        dodge=True,
        width=width_strip,
    )

    ax = sns.pointplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        # palette=sns.color_palette(palette="Set2"),
        ci=95,
        scale=1,
        color="black",
        zorder=2,
        dodge=width_strip / 2.0,
        # linestyles="--",
        join=False,
    )

    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.axhline(100, ls=":", color="black")
    xlim = list(ax.get_xlim())
    xlim[-1] -= (width) / 2.0
    ax.set_xlim(xlim)
    ax.set_ylabel("repeat surprisal\n(%)")
    ax.set_xlabel("")

    return ax
