import argparse
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from wordfreq import word_frequency

from lm_mem.analysis.plot import (
    plot_repeat_surprisal_violin_from_dfs,
    plot_single_sequence_from_df,
)
from lm_mem.analysis.utils import is_stopword, unspace


def plot_paraphrase_basic(
    data_dir: str = "data/output/paraphrase",
    data_dir_repeat: str = "data/output/repeat",
    output_dir: str = "plots/$model_name/paraphrase",
    model_name: str = "gpt2",
    experimentID: int = 0,
    overwrite: bool = False,
) -> None:
    """Plots data from paraphrase experiment together with repeat experiment."""

    output_dir = output_dir.replace("$model_name", model_name)

    # 0. Make sure data doesn't exist yet
    single_paraphrase_path = os.path.join(output_dir, "single_paraphrase.png")
    paraphrase_violin_path = os.path.join(output_dir, "paraphrase_violin.png")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not overwrite and (
        os.path.exists(single_paraphrase_path) or os.path.exists(paraphrase_violin_path)
    ):
        raise ValueError(
            f"Output directory {output_dir} already contains plots, use `-f` to overwrite."
        )

    # 1. Read data
    paraphrases = pd.read_csv(f"{data_dir}/paraphrases_{model_name}.csv")
    paraphrase_controls = pd.read_csv(f"{data_dir}/controls_{model_name}.csv")
    try:
        repeats = pd.read_csv(f"{data_dir_repeat}/repeats_{model_name}.csv")
        controls = pd.read_csv(f"{data_dir_repeat}/controls_{model_name}.csv")

        data_dfs = [repeats, paraphrases]
        control_dfs = [controls, paraphrase_controls]
        names = ["repeat", "paraphrase"]
    except FileNotFoundError:
        data_dfs = [paraphrases]
        names = ["paraphrase"]

    # 1. Plot single sequence
    paraphrase_experiment = paraphrases[paraphrases["experimentID"] == experimentID]
    plot_single_sequence_from_df(
        plt, paraphrase_experiment, title=f"Paraphrase Condition {model_name}"
    )
    plt.savefig(single_paraphrase_path, dpi=300)
    plt.close()

    # 2. Plot violin plot
    plot_repeat_surprisal_violin_from_dfs(
        plt,
        data_dfs,
        control_dfs,
        names,
    )
    plt.savefig(paraphrase_violin_path, dpi=300)
    plt.close()


def plot_same_vs_different_words(
    data_dir: str = "data/output/paraphrase",
    output_dir: str = "plots/$model_name/paraphrase",
    model_name: str = "gpt2",
    exclude_stopwords: bool = False,
    overwrite: bool = False,
) -> None:

    output_dir = output_dir.replace("$model_name", model_name)

    # 0. Check for existing plot
    stop_words = "_no_stopwords" if exclude_stopwords else ""
    output_path = os.path.join(
        output_dir, f"paraphrase_shared_vs_non_no_stopwords{stop_words}.png"
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not overwrite and os.path.exists(output_path):
        raise ValueError(
            f"Output directory {output_dir} already contains plots, use `-f` to overwrite."
        )

    # 1. Read data
    paraphrases = pd.read_csv(f"{data_dir}/paraphrases_{model_name}.csv")
    shared = []
    shared_freq = []
    not_shared = []
    not_shared_freq = []
    n_same_words = 0
    n_different_words = 0
    # TODO: do with groupby
    for sequenceID in paraphrases["sequenceID"].unique().tolist():
        sequence = paraphrases[paraphrases["sequenceID"] == sequenceID]
        first_sentence = sequence[sequence["sectionID"] == 1]
        second_sentence = sequence[sequence["sectionID"] == 3]
        used_words = first_sentence["word"].unique().tolist()
        for _, row in second_sentence.iterrows():

            word_second = row["word"]

            if exclude_stopwords:
                if is_stopword(word_second):
                    continue

            if row["word"] in used_words:
                shared.append(row["surprisal"])
                shared_freq.append(word_frequency(unspace(word_second), "en"))
                n_same_words += 1
            else:
                not_shared.append(row["surprisal"])
                not_shared_freq.append(word_frequency(unspace(word_second), "en"))
                n_different_words += 1

    shared_df = pd.DataFrame({"surprisal": shared, "frequency": shared_freq})
    shared_df["condition"] = f"Shared words (n={n_same_words})"
    not_shared_df = pd.DataFrame({"surprisal": not_shared, "frequency": not_shared_freq})
    not_shared_df["condition"] = f"Not shared words (n={n_different_words})"
    data = pd.concat([shared_df, not_shared_df])

    # Bin frequencies
    data["frequency"] = pd.cut(data["frequency"], bins=5)

    sns.stripplot(
        x="condition",
        y="surprisal",
        data=data,
        hue="frequency",
    )
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_shared_unshared_followsshared(
    data_dir: str = "data/output/paraphrase",
    output_dir: str = "plots/$model_name/paraphrase",
    model_name: str = "gpt2",
    exclude_stopwords: bool = False,
    overwrite: bool = False,
) -> None:

    output_dir = output_dir.replace("$model_name", model_name)

    # 0. Check for existing plot
    stop_words = "_no_stopwords" if exclude_stopwords else ""
    output_path = os.path.join(
        output_dir,
        f"paraphrase_scatter_different_shared_follows{stop_words}.png",
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not overwrite and os.path.exists(output_path):
        raise ValueError(
            f"Output directory {output_dir} already contains plots, use `-f` to overwrite."
        )

    # 1. Read data
    paraphrases = pd.read_csv(f"{data_dir}/paraphrases_{model_name}.csv")

    shared = []
    shared_freq = []
    not_shared = []
    not_shared_freq = []
    follows_shared = []
    follows_shared_freq = []
    n_same_words = 0
    n_different_words = 0
    n_follows_shared_words = 0
    # TODO: do with groupby
    for sequenceID in paraphrases["sequenceID"].unique().tolist():
        sequence = paraphrases[paraphrases["sequenceID"] == sequenceID]
        first_sentence = sequence[sequence["sectionID"] == 1]
        second_sentence = sequence[sequence["sectionID"] == 3]

        words_first = first_sentence["word"].tolist()
        words_second = second_sentence["word"].tolist()
        surprisals_second = second_sentence["surprisal"].tolist()
        previous_word = None
        for word_second, surprisal in zip(words_second, surprisals_second):

            if exclude_stopwords:
                if is_stopword(word_second):
                    previous_word = word_second
                    continue
            # get indices of the second sentence word in the first sentence
            indcs = [
                idx for idx, word_first in enumerate(words_first) if word_second == word_first
            ]
            # If there are indices -> word is shared
            if len(indcs):
                # For any index, check if previous shared word precedes it
                # somewhere in first_sentence
                prev_shared_word_precedes = [
                    previous_word == words_first[max(i - 1, 0)] for i in indcs if i > 0
                ]
                if any(prev_shared_word_precedes):
                    n_follows_shared_words += 1
                    follows_shared.append(surprisal)
                    follows_shared_freq.append(word_frequency(unspace(word_second), "en"))
                else:
                    shared.append(surprisal)
                    shared_freq.append(word_frequency(unspace(word_second), "en"))
                    n_same_words += 1
            # If there are no indices -> word is not shared
            else:
                not_shared.append(surprisal)
                not_shared_freq.append(word_frequency(unspace(word_second), "en"))
                n_different_words += 1

            previous_word = word_second

    shared_df = pd.DataFrame({"surprisal": shared, "frequency": shared_freq})
    shared_df["condition"] = f"Shared (n={n_same_words})"
    not_shared_df = pd.DataFrame({"surprisal": not_shared, "frequency": not_shared_freq})
    not_shared_df["condition"] = f"Not shared (n={n_different_words})"
    follows_shared_df = pd.DataFrame(
        {"surprisal": follows_shared, "frequency": follows_shared_freq}
    )
    follows_shared_df["condition"] = f"Follows (n={n_follows_shared_words})"
    data = pd.concat([shared_df, not_shared_df, follows_shared_df])

    sns.scatterplot(
        x="frequency",
        y="surprisal",
        hue="condition",
        data=data,
    )
    plt.savefig(output_path, dpi=300)
    plt.close()

    # Bin frequencies
    data["frequency"] = pd.cut(data["frequency"], bins=5)

    sns.stripplot(
        x="condition",
        y="surprisal",
        data=data,
        hue="frequency",
    )
    plt.savefig(
        os.path.join(output_dir, "paraphrase_different_shared_follows_no_stopwords.png"), dpi=300
    )
    plt.close()


def plot_surprisal_by_length(
    data_dir: str = "data/output/paraphrase",
    output_dir: str = "plots/$model_name/paraphrase",
    model_name: str = "gpt2",
    use_mean: bool = False,
    overwrite: bool = False,
) -> None:

    output_dir = output_dir.replace("$model_name", model_name)

    # 0. Check for existing plot
    mean_marker = "_mean" if use_mean else ""
    output_path = os.path.join(
        output_dir,
        f"paraphrase_n_shared_vs_repeat_surprisal{mean_marker}.png",
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not overwrite and os.path.exists(output_path):
        raise ValueError(
            f"Output directory {output_dir} already contains plots, use `-f` to overwrite."
        )

    paraphrases = pd.read_csv(f"{data_dir}/paraphrases_{model_name}.csv")
    paraphrase_controls = pd.read_csv(f"{data_dir}/controls_{model_name}.csv")
    grouped = paraphrases.groupby(["sequenceID"])
    # get repeat surprisals
    gp = paraphrases.groupby(["sequenceID", "sectionID"])["surprisal"]
    gp_controls = paraphrase_controls.groupby(["sequenceID", "sectionID"])["surprisal"]
    if use_mean:
        surprisal = gp.mean().droplevel(level=0)
        surprisal_controls = gp_controls.mean().droplevel(level=0)
    else:
        surprisal = gp.median().droplevel(level=0)
        surprisal_controls = gp_controls.median().droplevel(level=0)

    repeat_surprisal = 100 * surprisal[3] / surprisal_controls[3]

    x = []
    y = []
    for (_, sequence_df), rs in zip(grouped, repeat_surprisal):
        first_sentence_words = sequence_df[sequence_df["sectionID"] == 1]["word"].tolist()
        second_sentence_words = sequence_df[sequence_df["sectionID"] == 3]["word"].tolist()

        n_shared = 0
        for word in second_sentence_words:
            if word in first_sentence_words:
                n_shared += 1
        x.append(n_shared)
        y.append(rs)

    plt.scatter(x, y)
    plt.xlabel("number of words shared per sentence")
    plt.ylabel("repeat surprisal (%)")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_paraphrase(
    data_dir: str = "data/output/paraphrase",
    data_dir_repeat: str = "data/output/repeat",
    output_dir: str = "plots/$model_name/paraphrase",
    model_name: str = "gpt2",
    experimentID: int = 0,
    exclude_stopwords: bool = False,
    use_mean: bool = False,
    overwrite: bool = False,
):
    plot_paraphrase_basic(
        data_dir=data_dir,
        data_dir_repeat=data_dir_repeat,
        output_dir=output_dir,
        model_name=model_name,
        experimentID=experimentID,
        overwrite=overwrite,
    )
    plot_same_vs_different_words(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        exclude_stopwords=exclude_stopwords,
        overwrite=overwrite,
    )
    plot_shared_unshared_followsshared(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        exclude_stopwords=exclude_stopwords,
        overwrite=overwrite,
    )
    plot_surprisal_by_length(
        data_dir=data_dir,
        output_dir=output_dir,
        use_mean=use_mean,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/output/paraphrase",
        help="dir containing paraphrases_{model}.csv",
    )
    parser.add_argument(
        "--data_dir_repeat",
        type=str,
        default="data/output/repeat",
        help="dir containing repeats_{model}.csv and controls_{model}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots/$model_name/paraphrase",
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
        "--exclude_stopwords",
        action="store_true",
        help="Whether to exclude stopwords in the scatter plots",
    )
    parser.add_argument(
        "--use_mean",
        action="store_true",
        help="Whether to use mean instead of median in shared word vs surprisal plot",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="If set, will overwrite existing content in output_dir.",
    )
    args = parser.parse_args()

    plot_paraphrase(
        data_dir=args.data_dir,
        data_dir_repeat=args.data_dir_repeat,
        output_dir=args.output_dir,
        model_name=args.model_name,
        experimentID=args.experimentID,
        exclude_stopwords=args.exclude_stopwords,
        use_mean=args.use_mean,
        overwrite=args.force,
    )
