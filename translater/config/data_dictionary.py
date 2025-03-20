"""
This is the dictionary containing parameter info for the package run
"""

from enum import Enum
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

START_TOKEN = "<START>"
END_TOKEN = "<END>"
PADDING_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
NEG_INFINITY = -1e20


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
    max_train_size = 200000
    max_test_size = 20000
    preprocessor_file = "result/preprocessor.pkl"  # contains vocab, vocab <-> index map, valid eng, ml pairs


class Train(Enum):
    batch_size = 64
    seed = 1
    learning_rate = 1e-4
    num_epochs = 10
    checkpoint_dir = "result/checkpoints"


class Encoder_Enum(Enum):
    num_layers = 2
    d_model = (
        512  # the dimensionality of the model's hidden states or embeddings, q, k, v
    )
    # q_k_v_dim = 64 is deduced as d_modelnum_attention_heads as 8*64 = 512
    num_attention_heads = 8  # For self attention in both Encoder
    drop_prob = 0.1  # drop probability (10% dropout), happens after every layer norm and inside FFW
    hidden_dim = 2048  # dim of FFW nw's hidden layer


class Decoder_Enum(Enum):
    num_layers = 2
    d_model = 512
    num_attention_heads = 8
    drop_prob = 0.1
    hidden_dim = 2048
