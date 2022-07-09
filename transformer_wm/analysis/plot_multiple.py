import os

import matplotlib.pyplot as plt

from transformer_wm import get_logger
from transformer_wm.analysis.plot import plot_boxplot, plot_catplot, plot_raincloud, plot_raincloud2
from transformer_wm.analysis.utils import (
    compute_repeat_surprisal,
    get_df_across_models,
    get_repeat_surprisal_df,
    get_repeat_surprisal_df_across_models,
)
from transformer_wm.data.reader import load_models

logger = get_logger(__name__)


def plot_multiple(
    output_path: str = "plots/repeat_surprisal_all.png",
    model_names: str = "all",
    use_mean: bool = False,
    overwrite: bool = False,
    data_dir_repeat: str = "data/output/repeat",
    data_dir_paraphrase: str = "data/output/paraphrase",
    data_dir_shuffle: str = "data/output/shuffle",
) -> None:
    """Plots results across models."""

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

    # 1. Read data and accumulate repeat surprisal values
    repeat_surprisal_df = get_df_across_models(
        model_names=model_names,
        experiments=["repeat"],
        data_dir=data_dir_repeat,
        compute_function=compute_repeat_surprisal,
    )
    # repeat_surprisal_df = get_repeat_surprisal_df_across_models(
    #     model_names=model_names,
    #     data_dir_repeat=data_dir_repeat,
    #     data_dir_paraphrase="no",
    #     data_dir_shuffle="no",
    #     use_mean=use_mean,
    # )

    # 2. plot
    # plot_catplot(
    #     plt,
    #     repeat_surprisal_df,
    #     kind="bar",
    # )
    ax = plot_raincloud2(
        data=repeat_surprisal_df,
        figsize=(5, 7),
        font_scale=1.1,
        x="model",
    )
    plt.title("Verbatim Recall of sentences in transformers")
    plt.tight_layout()
    # plt.tight_layout()
    logger.info(f"Saving to {output_path}")
    plt.savefig(output_path, dpi=300)


if __name__ == "__main__":
    plot_multiple(
        overwrite=True,
        output_path="plots/repeat_surprisal_gpt2.png",
        model_names=["gpt2"],
    )
    plot_multiple(
        overwrite=True,
        output_path="plots/repeat_surprisal_all.png",
        model_names="all",
    )
