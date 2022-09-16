import os

import pandas as pd


def rename(df: pd.DataFrame, col: str, old_name: str, new_name: str) -> pd.DataFrame:
    df[df[col] == old_name, col] = new_name
    return df


def print_summary(output_dir=None):
    for pos in ["noun", "verb"]:
        print(f"%%%%%%%%%% {pos}")
        analysis_results = pd.read_csv(f"data/prob_change_analysis/analysis_results_{pos}.csv")
        print("-- raw")
        # print(analysis_results.groupby("category")["word"].count())
        # print(analysis_results.groupby("category")["prob_change"].sum())
        print(analysis_results.groupby("category").agg({"word": "count", "prob_change": "sum"}))

        print("-- zusammengefasst")
        analysis_results["category"] = analysis_results["category"].replace("semR_syn", "semF_syn")
        analysis_results["category"] = analysis_results["category"].replace(
            "semR_synF", "semF_synF"
        )

        zs = (
            analysis_results.groupby("category")
            .agg({"word": "count", "prob_change": "sum"})
            .transpose()
        )
        if "intervention" in zs.columns:
            zs["other"] = zs["other"] + zs["intervention"]
            zs.drop("intervention", axis=1)

        zsf = pd.DataFrame()
        zsf[
            [
                "verbatim repetition",
                "semantically correct, syntactically correct",
                "semantically correct, syntactically incorrect",
                "semantically incorrect, syntactically correct",
                "semantically incorrect, syntactically incorrect",
                "position shifted word",
                "other",
            ]
        ] = zs[
            [
                "verbatim_repeat",
                "sem_syn",
                "sem_synF",
                "semF_syn",
                "semF_synF",
                "upcoming_word",
                "other",
            ]
        ]
        zsf = zsf.transpose()
        zsf["percent"] = zsf["prob_change"] / zsf["prob_change"].sum()

        print(zsf)

        if output_dir is not None:
            zsf.to_csv(os.path.join(output_dir, f"prob_change_percentage_{pos}.csv"), index=True)


if __name__ == "__main__":
    print_summary()
