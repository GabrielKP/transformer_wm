from transformer_wm.data.nonce import get_correct_nonce_combinations

contexts, nouns, verbs = get_correct_nonce_combinations(
    contexts_file="data/nonce/sentential_contexts.tsv",
    return_correct_separately=True,
)

for c, n, v in zip(contexts, nouns, verbs):
    print(f"            {c[1]} & {n} & {v} \\\\")
