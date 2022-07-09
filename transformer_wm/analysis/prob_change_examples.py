from ast import literal_eval

import pandas as pd

from transformer_wm.data.reader import load_sequences

NDIGITS = 4
NEXAMPLES = 10


def rename(df: pd.DataFrame, col: str, old_name: str, new_name: str) -> pd.DataFrame:
    df[df[col] == old_name, col] = new_name
    return df


def make_readable(cell):
    word, value = cell
    value = float(value)
    if word.startswith("Ġ"):
        pword = word[1:]
    else:
        pword = f"#{word}"
    return f"{pword} ({round(value, NDIGITS)})"


def print_example():
    model = "gpt2"
    for pos in ["noun", "verb"]:
        prob_change_data = pd.read_csv(
            f"data/output/repeat_predictions_change/pred_change_repeat_{pos}_{model}.csv"
        )
        sentences = load_sequences("data/input/repeat/seq1s_repeat.json")
        sentences = [
            {"experimentID": sent["experimentID"], "encoding sentence": sent["first"]}
            for sent in sentences
        ]
        sentences_dict = {key: [sent[key] for sent in sentences] for key in sentences[0].keys()}
        sentences_df = pd.DataFrame(sentences_dict)[["encoding sentence"]]
        sentences_neg_def = sentences_df.copy()
        # joint = sentences_df.join(
        #     prob_change_data, on="experimentID", how="inner", lsuffix="_base"
        # )
        # output_df = joint.drop(
        #     columns=["experimentID", "experimentID_base", "sequenceID", "sentenceID", "vignetteID"]
        # )

        preds_positive = (
            prob_change_data["changes_positive"].apply(literal_eval).to_frame("changes_positive")
        )
        preds_negative = (
            prob_change_data["changes_negative"].apply(literal_eval).to_frame("changes_negative")
        )
        n_predictions = len(preds_positive["changes_positive"].iloc[0])
        columns = [f"change #{num}" for num in range(1, n_predictions + 1)]

        sentences_df[columns] = pd.DataFrame(
            preds_positive.changes_positive.tolist(), index=preds_positive.index
        )
        sentences_neg_def[columns] = pd.DataFrame(
            preds_negative.changes_negative.tolist(), index=preds_negative.index
        )
        sentences_df[columns] = sentences_df[columns].applymap(make_readable)
        sentences_neg_def[columns] = sentences_neg_def[columns].applymap(make_readable)
        sentences_df = sentences_df.drop(
            columns=[f"change #{num}" for num in range(7, n_predictions + 1)]
        )[:NEXAMPLES]
        sentences_neg_def = sentences_neg_def.drop(
            columns=[f"change #{num}" for num in range(7, n_predictions + 1)]
        )[:NEXAMPLES]

        # to latex
        sentences_df.to_latex(f"data/output/tables/prob_examples_{pos}_{model}.txt")


if __name__ == "__main__":
    print_example()
