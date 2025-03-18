"""
This is the dictionary containing parameter info for the package run
"""

from enum import Enum
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


class HuggingFaceData(Enum):
    dataset = "ai4bharat/samanantar"
    name = "ml"
    split = "train"
    remove_feature = ["idx"]
    save_train_location = "data/english_malayalam_train.arrow"
    save_test_location = "data/english_malayalam_test.arrow"
    src_column = "src"
    tgt_column = "tgt"
    test_split_ratio = 0.2
    max_length = 300  # After checking 99% percentile of sentence length
    seed = 1
    max_train_size = 1000000
    max_test_size = 200000
    preprocessor_file = "result/preprocessor.pkl"  # contains vocab, vocab <-> index map, valid eng, ml pairs


class Train(Enum):
    batch_size = 64
    seed = 1


class Training:
    framework = "torch"
