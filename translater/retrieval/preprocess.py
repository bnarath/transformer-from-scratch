from utils.process import are_tokens_valid, is_len_valid
import logging
from typing import Literal


def get_valid_sentence_pairs(
    dataset,
    english_vocab,
    malayalam_vocab,
    max_seq_len,
    total,
    to_print=False,
    type: Literal["train", "val", "test"] = "train",
):
    """Given Hugging face dataset, return valid sentence pairs."""
    eng_mal_valid_sentence_pairs = []

    for data in dataset.select(range(total)):
        # For test we don't want to exclude cases where invalid tokens are present (Invalid tokens later will be converted into <UNK>)
        eng_valid_token_check, eng_invalid_set = True, set()
        mal_valid_token_check, mal_invalid_set = True, set()

        if type == "train":
            eng_valid_token_check, eng_invalid_set = are_tokens_valid(
                set(data["src"]),
                english_vocab,  # TBD: Need to change this if we consider words
            )
            mal_valid_token_check, mal_invalid_set = are_tokens_valid(
                set(data["tgt"]),
                malayalam_vocab,  # TBD: Need to change this if we consider words
            )

        eng_len_check = is_len_valid(data["src"])
        mal_len_check = is_len_valid(
            data["tgt"], max_seq_len=max_seq_len - 1
        )  # To account for start
        if (
            eng_valid_token_check
            and mal_valid_token_check
            and eng_len_check
            and mal_len_check
        ):
            eng_mal_valid_sentence_pairs.append((data["src"], data["tgt"]))
        elif to_print:
            if eng_invalid_set:
                logging.info(f"Invalid eng char(s) found {eng_invalid_set}")
            if mal_invalid_set:
                logging.info(
                    f"Invalid mal char(s) found {mal_invalid_set}: sentence is {data['tgt']}"
                )

    return eng_mal_valid_sentence_pairs
