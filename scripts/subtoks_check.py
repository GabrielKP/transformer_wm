import pandas as pd

data_dir = "data/output/repeat"

model_name1 = "gpt2"
model_name2 = "bert-base-uncased"

repeats_og = pd.read_csv(f"{data_dir}/repeats_gpt2.csv")
repeats1 = pd.read_csv(f"{data_dir}/repeats_test_{model_name1}.csv")
repeats2 = pd.read_csv(f"{data_dir}/repeats_test_{model_name2}.csv")

repeats_og = repeats_og.groupby(["sequenceID", "sectionID"])
repeats1 = repeats1.groupby(["sequenceID", "sectionID"])
repeats2 = repeats2.groupby(["sequenceID", "sectionID"])

for ((_, secID), r_og), (_, r1), (_, r2) in zip(repeats_og, repeats1, repeats2):
    if secID != 1:
        continue
    print("----")
    print(f"{'Original':15}: {' '.join(r_og['word'])}")
    print(f"{model_name1:15}: {' '.join(r1['word'])}")
    print(f"{model_name2:15}: {' '.join(r2['word'])}")
    if str(input("n for next, exit for exit")) == "exit":
        break
