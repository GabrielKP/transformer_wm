from typing import Callable, List, Tuple, Union

import pandas as pd
from nltk.corpus import stopwords

from lm_mem import get_logger

logger = get_logger(__name__)


def remove_stopwords(word_list: List[str]) -> List[str]:
    try:
        return [word for word in word_list if unspace(word) not in stopwords.words("english")]
    except LookupError:
        import nltk

        nltk.download("stopwords")

    return [word for word in word_list if unspace(word) not in stopwords.words("english")]


def is_stopword(word: str) -> bool:
    try:
        return unspace(word) in stopwords.words("english")
    except LookupError:
        import nltk

        nltk.download("stopwords")
    return unspace(word) in stopwords.words("english")


def unspace(word: str) -> str:
    return word[1:] if word[0] == " " else word


def compute_repeat_surprisal(
    seq1_df: pd.DataFrame, seq2_df: pd.DataFrame, use_mean: bool
) -> pd.Series:
    """Returns Series of repeat surprisals.

    seq1_df: DataFrame
        containing surprisal values for seq1. Each seq1 should have a unique experimentID
    seq2_df: DataFrame
        containing surprisal values for seq2. seq2s with same experimentIDs will be averaged.
        Has to contain exactly the same unique experimentIDs as seq1_df.

    """
    # TODO: use this in all plotting functions.
    # Group by sequenceID and sectionID
    grouped_data = seq1_df.groupby(["sequenceID", "sectionID", "experimentID"])["surprisal"]
    grouped_control = seq2_df.groupby(["sequenceID", "sectionID", "experimentID"])["surprisal"]
    # Get median surprisals
    if use_mean:
        m_surprisal_data = grouped_data.mean().droplevel(level=0)
        m_surprisal_control = grouped_control.mean()
    else:
        m_surprisal_data = grouped_data.median().droplevel(level=0)
        m_surprisal_control = grouped_control.median()
    # Average over controls belonging to same test_sequence
    m_surprisal_control = m_surprisal_control.groupby(["sectionID", "experimentID"]).mean()
    # TODO: sort by experimentID
    # Compute repeat surprisal
    repeat_surprisal = 100 * m_surprisal_data[3] / m_surprisal_control[3]

    return repeat_surprisal


def get_repeat_surprisal_df(
    seq1_dfs: List[pd.DataFrame],
    seq2_dfs: List[pd.DataFrame],
    condition_names: Union[List[str], List[Tuple[str, str]]],
    use_mean: bool,
    compute_function: Callable[
        [pd.DataFrame, pd.DataFrame, bool], pd.Series
    ] = compute_repeat_surprisal,
):
    """Returns dataframe with repeat surprisals labeled with condition.

    Return dataframe has columns repeat_surprisal, condition
    """

    df_accumulator: List[pd.DataFrame] = []
    for name, seq1_df, seq2_df in zip(condition_names, seq1_dfs, seq2_dfs):
        repeat_surprisal = compute_function(seq1_df, seq2_df, use_mean)
        condition_df = pd.DataFrame({"repeat surprisal": repeat_surprisal})
        if isinstance(name, tuple):
            condition_df["condition"] = name[0]
            condition_df["subcondition"] = name[1]
        else:
            condition_df["condition"] = name
        df_accumulator.append(condition_df)

    return pd.concat(df_accumulator)


def get_repeat_surprisal_df_across_models(
    model_names: List[str],
    data_dir_repeat: str = "data/output/repeat",
    data_dir_paraphrase: str = "data/output/paraphrase",
    data_dir_shuffle: str = "data/output/shuffle",
    use_mean: bool = False,
):
    """Returns repeat surprisal values for each experiment."""
    df_accumulator: List[pd.DataFrame] = []
    for model_name in model_names:
        model_name = model_name.split("/")[-1]
        seq1_dfs: List[pd.DataFrame] = []
        seq2_dfs: List[pd.DataFrame] = []
        condition_names: List[str] = []

        # in case that some experiments are missing use try and except
        try:
            seq1_dfs.append(pd.read_csv(f"{data_dir_repeat}/repeats_{model_name}.csv"))
            seq2_dfs.append(pd.read_csv(f"{data_dir_repeat}/controls_{model_name}.csv"))
            condition_names.append("repeat")
        except FileNotFoundError:
            logger.warning(f"Cannot find repeat experiment output for {model_name}")
        try:
            seq1_dfs.append(pd.read_csv(f"{data_dir_paraphrase}/paraphrases_{model_name}.csv"))
            seq2_dfs.append(pd.read_csv(f"{data_dir_paraphrase}/controls_{model_name}.csv"))
            condition_names.append("paraphrases")
        except FileNotFoundError:
            logger.warning(f"Cannot find paraphrase experiment output for {model_name}")
        try:
            seq1_dfs.append(
                pd.read_csv(f"{data_dir_shuffle}/shuffled_global_both_{model_name}.csv")
            )
            seq2_dfs.append(
                pd.read_csv(f"{data_dir_shuffle}/shuffled_global_both_control_{model_name}.csv")
            )
            condition_names.append("both\nshuffled")
        except FileNotFoundError:
            logger.warning(f"Cannot find both shuffled experiment output for {model_name}")
        try:
            seq1_dfs.append(
                pd.read_csv(f"{data_dir_shuffle}/shuffled_global_first_{model_name}.csv")
            )
            seq2_dfs.append(
                pd.read_csv(f"{data_dir_shuffle}/shuffled_global_first_control_{model_name}.csv")
            )
            condition_names.append("first\nshuffled")
        except FileNotFoundError:
            logger.warning(f"Cannot find first shuffled experiment output for {model_name}")
        try:
            seq1_dfs.append(
                pd.read_csv(f"{data_dir_shuffle}/shuffled_global_second_{model_name}.csv")
            )
            seq2_dfs.append(
                pd.read_csv(f"{data_dir_shuffle}/shuffled_global_second_control_{model_name}.csv")
            )
            condition_names.append("second\nshuffled")
        except FileNotFoundError:
            logger.warning(f"Cannot find second shuffled experiment output for {model_name}")

        # Get the models repeat surprisal values for each condition
        repeat_surprisal_df = get_repeat_surprisal_df(
            seq1_dfs=seq1_dfs,
            seq2_dfs=seq2_dfs,
            condition_names=condition_names,
            use_mean=use_mean,
        )
        # Annotate with model name
        repeat_surprisal_df["model"] = model_name
        df_accumulator.append(repeat_surprisal_df)

    return pd.concat(df_accumulator)


def load_df(data_dir: str, experiment: str, model_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.info(f"Reading {data_dir}/seq1s_{experiment}_{model_name}.csv")
        seq1 = pd.read_csv(f"{data_dir}/seq1s_{experiment}_{model_name}.csv")
        logger.info(f"Reading {data_dir}/seq2s_{experiment}_{model_name}.csv")
        seq2 = pd.read_csv(f"{data_dir}/seq2s_{experiment}_{model_name}.csv")
        return seq1, seq2
    except FileNotFoundError:
        logger.warning(f"Cannot find {experiment} output for {model_name}")
        return None, None


def get_df_across_models(
    model_names: List[str],
    data_dir: str,
    experiments: List[str],
    compute_function: Callable,
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
            seq1, seq2 = load_df(
                data_dir=data_dir,
                experiment=experiment,
                model_name=model_name,
            )
            if seq1 is not None and seq2 is not None:
                seq1_dfs.append(seq1)
                seq2_dfs.append(seq2)
                condition_names.append(experiment.replace("_", "\n"))

        # Get the models repeat surprisal values for each condition
        if len(seq1_dfs):
            repeat_surprisal_df = get_repeat_surprisal_df(
                seq1_dfs=seq1_dfs,
                seq2_dfs=seq2_dfs,
                condition_names=condition_names,
                use_mean=use_mean,
                compute_function=compute_function,
            )
            # Annotate with model name
            repeat_surprisal_df["model"] = model_name
            df_accumulator.append(repeat_surprisal_df)

    return pd.concat(df_accumulator)
