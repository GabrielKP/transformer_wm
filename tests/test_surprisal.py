from typing import List

from tests.expected_data import expected_input_data, expected_output_data


def nested_list_comparer(list1: List, list2: List):
    assert type(list1) == type(list2)
    assert len(list1) == len(list2)
    if isinstance(list1[0], list):
        for sub_list1, sub_list2 in zip(list1, list2):
            nested_list_comparer(sub_list1, sub_list2)
    else:
        for item1, item2 in zip(list1, list2):
            if item1 != item1:
                # expect both items to be nan
                # nan != nan == True
                assert item2 != item2
            else:
                # round to account for inaccuracies
                assert round(item1, 1) == round(item2, 1)


def test_experiment_with_gpt2(expected_input_data, expected_output_data):

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    from lm_mem.surprisal import Experiment

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = Experiment(
        model=GPT2LMHeadModel.from_pretrained("gpt2"),
        tokenizer=tokenizer,
        device=device,
    )

    output_data = experiment.start(expected_input_data["one_hots"])

    for key in expected_output_data:
        assert key in output_data
        assert len(output_data[key]) == len(expected_output_data[key])

        nested_list_comparer(expected_output_data[key], output_data[key])


def test_experiment_batched_with_gpt2(expected_input_data, expected_output_data):

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    from lm_mem.surprisal import Experiment

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = Experiment(
        model=GPT2LMHeadModel.from_pretrained("gpt2"),
        tokenizer=tokenizer,
        device=device,
        batch_size=4,
    )

    output_data = experiment.start(expected_input_data["one_hots"])

    for key in expected_output_data:
        assert key in output_data
        assert len(output_data[key]) == len(expected_output_data[key])

        nested_list_comparer(expected_output_data[key], output_data[key])
