from data.retrieve import Retriever
from data.create_vocabulary import CreateVocabulary
from utils.parser import parse
from data.preprocess import get_valid_sentence_pairs

from config.data_dictionary import ROOT, Train, HuggingFaceData
import logging
from pathlib import Path
import os
import pickle
from typing import Literal


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
