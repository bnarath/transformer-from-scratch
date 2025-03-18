from config.data_dictionary import HuggingFaceData
from typing import Set, List


def are_tokens_valid(tokens: Set, vocabulary: List):
    "Check if tokens are present in vocabulary. In case of letter to letter, tokens are letters where as, in case of word by work, tokens are words."
    invalid_tokens = tokens - set(vocabulary)
    if invalid_tokens:
        return False, invalid_tokens
    return True, set()


def is_len_valid(tokens: List, max_seq_len: int = HuggingFaceData.max_length.value):
    "Check if toten len of tokens are within the limit"
    return True if len(tokens) <= max_seq_len else False
