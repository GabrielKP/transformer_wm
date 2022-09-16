import os
from ast import literal_eval

import pandas as pd

NDIGITS = 4


def make_readable(cell):
    word, value = cell
    value = float(value)
    if word.startswith("Ġ"):
        pword = word[1:]
    else:
        pword = f"#{word}"
    return f"{pword} {round(value, NDIGITS)}"


def predictions_pretty(
    input_dir="data/output/repeat_predictions_change", output_dir="data/output/analysis/"
):
    experiments = [
        "pred_change_repeat_verb_gpt2",
        "pred_change_repeat_noun_gpt2",
    ]
    for experiment in experiments:
        preds_change = pd.read_csv(f"{input_dir}/{experiment}.csv")

        preds_positive = (
            preds_change["changes_positive"].apply(literal_eval).to_frame("changes_positive")
        )
        preds_negative = (
            preds_change["changes_negative"].apply(literal_eval).to_frame("changes_negative")
        )

        n_predictions = len(preds_positive["changes_positive"].iloc[0])
        columns = [f"change #{num}" for num in range(1, n_predictions + 1)]
        preds_positive[columns] = pd.DataFrame(
            preds_positive.changes_positive.tolist(), index=preds_positive.index
        )
        preds_negative[columns] = pd.DataFrame(
            preds_negative.changes_negative.tolist(), index=preds_negative.index
        )
        preds_positive = preds_positive.drop(columns="changes_positive")
        preds_negative = preds_negative.drop(columns="changes_negative")
        preds_positive[columns] = preds_positive[columns].applymap(make_readable)
        preds_negative[columns] = preds_negative[columns].applymap(make_readable)
        preds_positive.to_csv(
            os.path.join(output_dir, f"pretty_positive_{experiment}.csv"), index=False
        )
        preds_negative.to_csv(
            os.path.join(output_dir, f"pretty_negative_{experiment}.csv"), index=False
        )


if __name__ == "__main__":
    predictions_pretty()
