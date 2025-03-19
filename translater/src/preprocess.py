import torch
import torch.nn as nn
from data.retrieve import Retriever
from data.create_vocabulary import CreateVocabulary
from utils.parser import parse
from data.preprocess import get_valid_sentence_pairs

from config.data_dictionary import ROOT, Train, HuggingFaceData
import logging
from typing import Literal, List, Dict


class Preprocessor:
    def __init__(self, type: Literal["letter_by_letter", "word_by_word"]):
        self.type = type
        self.vocab_type = "word" if self.type == "word_by_word" else "letter"

    def prepare_data(self):
        logging.info("1. Retrive data from hugging face")
        retriever = Retriever()
        logging.info(retriever.train_data)
        logging.info(retriever.test_data)
        logging.info("2. Retrieving vocabulary")
        create_vocab_ml = CreateVocabulary(language="ml", type=self.vocab_type)
        self.ml_vocab = create_vocab_ml.get_vocab()
        create_vocab_eng = CreateVocabulary(language="eng", type=self.vocab_type)
        self.eng_vocab = create_vocab_eng.get_vocab()
        logging.info(f"English vocab = {self.eng_vocab}")
        logging.info("-" * 30)
        logging.info(f"Malayalam vocab = {self.ml_vocab}")
        self.eng_voca_len = len(self.eng_vocab)
        self.ml_voca_len = len(self.ml_vocab)
        logging.info(
            f"English vocab length = {self.eng_voca_len} \nMalayalam vocab length = {self.ml_voca_len}"
        )
        logging.info("-" * 30)
        logging.info("3. Getting vocab to token mapping and vice versa")
        self.ml_vocab_to_index, self.ml_index_to_vocab = create_vocab_ml.get_tokens()
        self.eng_vocab_to_index, self.eng_index_to_vocab = create_vocab_eng.get_tokens()
        logging.info("4. Getting valid sentence pairs")
        self.eng_mal_valid_sentence_pairs_for_train = get_valid_sentence_pairs(
            retriever.train_data,
            create_vocab_eng.vocabulary,
            create_vocab_ml.vocabulary,
            max_seq_len=HuggingFaceData.max_length.value,
            total=HuggingFaceData.max_train_size.value,
            to_print=False,
            type="train",
        )
        self.eng_mal_valid_sentence_pairs_for_test = get_valid_sentence_pairs(
            retriever.test_data,
            create_vocab_eng.vocabulary,
            create_vocab_ml.vocabulary,
            max_seq_len=HuggingFaceData.max_length.value,
            total=HuggingFaceData.max_test_size.value,
            to_print=False,
            type="test",
        )
        logging.info(
            f"Total valid sentence pairs in train data = {len(self.eng_mal_valid_sentence_pairs_for_train)}"
        )
        logging.info(
            f"Total valid sentence pairs in test data = {len(self.eng_mal_valid_sentence_pairs_for_test)}"
        )


class SentenceEmbedding(nn.Module):
    def __init__(
        self,
        max_seq_length,
        d_model,
        vocab_to_index,
        drop_prob,
        START_TOKEN,
        END_TOKEN,
        PADDING_TOKEN,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.d_model = d_model  # Embedding dimension
        self.vocab_to_index = vocab_to_index
        self.vocab_size = len(vocab_to_index)
        self.drop_prob = drop_prob  # Drop after embedding + pos encoding
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.d_model,
            padding_idx=vocab_to_index[self.PADDING_TOKEN],
        )
        self.positional_encoder = PositionalEncoder(
            self.max_seq_length, self.d_model
        )  # TBD
        self.dropout = nn.Dropout(0.1)

    def batch_tokenize(
        self,
        batch: List[List[str]],
        start_token: bool = False,
        end_token: bool = False,
    ) -> torch.Tensor:  # (batch, max_sequence_len)
        """Tokenize in batches"""
        tokenized_batch = []
        for sentence in batch:
            tokenized_batch.append(
                self.tokenize(
                    sentence,
                    self.vocab_to_index,
                    self.max_seq_length,
                    start_token,
                    end_token,
                )
            )
        tokenized_batch = torch.stack(tokenized_batch)
        return tokenized_batch

    def tokenize(
        self,
        sentence: List[str],
        vocab_to_id: Dict[str, int],
        max_seq_len: int,
        start_token: bool = False,
        end_token: bool = False,
    ) -> List[int]:
        """Tokenize a sentence. Optionally add start and end tokens. Always pad with padding token."""
        tokens = []
        if start_token:
            tokens = [vocab_to_id[self.START_TOKEN]]
        tokens.extend([vocab_to_id[ch] for ch in sentence])
        if end_token:
            tokens.append(vocab_to_id[self.END_TOKEN])
        for _ in range(len(tokens), max_seq_len):
            tokens.append(vocab_to_id[self.PADDING_TOKEN])
        return torch.Tensor(tokens)

    def forward(self, x, start_token=False, end_token=False):
        # x: (batch, )
        x = self.batch_tokenize(
            x, start_token, end_token
        ).long()  # (batch, max_seq_len)
        x = self.embedding(x)  # (batch, max_seq_len, d_model)
        pos = self.positional_encoder()  # (batch, max_seq_len, d_model)
        x = x + pos
        x = self.dropout(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.PE = self.get_encoding()

    def forward(self):
        # x: (batch, max_seq_len, d_model)
        return self.PE

    def get_encoding(self):
        even_i = torch.arange(0, self.d_model, 2)
        denominator = torch.pow(10000, even_i / self.d_model)
        pos = torch.arange(self.max_seq_len).reshape(self.max_seq_len, 1)
        even_PE = torch.sin(pos / denominator)
        odd_PE = torch.cos(pos / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
