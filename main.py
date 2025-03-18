from torch.utils.data import Dataset, DataLoader
import torch
import logging
from src.preprocess import Preprocessor
from utils.parser import parse
from utils.print import display_first_n_batch
from config.data_dictionary import ROOT, HuggingFaceData, Train
from pathlib import Path
import os
import pickle
from typing import Tuple, List, Dict

torch.manual_seed(Train.seed.value)

START_TOKEN = "<START>"
END_TOKEN = "<END>"
PADDING_TOKEN = "<PAD>"
UNKNOW_TOKEN = "<UNK>"
NEG_INFINITY = -1e20


class Translation:
    def __init__(self, framework: str = "pytorch", type: str = "letter_by_letter"):
        self.framework = framework
        self.type = type

    def build(self):
        train_dataloader, test_dataloader = (
            self.get_input()
        )  # Each batch is [[(64 src, 64 tgt)]]

    def get_input(self) -> Tuple[DataLoader]:
        """Torch dataset for training"""
        preprocessor_path = ROOT / Path(HuggingFaceData.preprocessor_file.value)
        preprocessor_dirpath = os.path.dirname(preprocessor_path)
        if not os.path.exists(preprocessor_dirpath):
            os.makedirs(preprocessor_dirpath, exist_ok=True)

        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
        else:
            preprocessor = Preprocessor(type=self.type)
            preprocessor.prepare_data()
            with open(preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)

        self.ml_vocab_to_index = preprocessor.ml_vocab_to_index
        self.ml_index_to_vocab = preprocessor.ml_index_to_vocab
        self.eng_vocab_to_index = preprocessor.eng_vocab_to_index
        self.eng_index_to_vocab = preprocessor.eng_index_to_vocab

        train_ds = TranslationDataset(
            src_trg_pairs=preprocessor.eng_mal_valid_sentence_pairs_for_train
        )
        test_ds = TranslationDataset(
            src_trg_pairs=preprocessor.eng_mal_valid_sentence_pairs_for_test
        )
        train_dataloader = DataLoader(
            train_ds, batch_size=Train.batch_size.value, shuffle=True, drop_last=True
        )
        test_dataloader = DataLoader(
            test_ds, batch_size=Train.batch_size.value, shuffle=True, drop_last=True
        )
        # n = 1
        # logging.info(f"First {n} batch from training DataLoader")
        # display_first_n_batch(train_dataloader, n)
        # logging.info(f"First {n} batch from testing DataLoader")
        # display_first_n_batch(test_dataloader, n)
        return train_dataloader, test_dataloader

    def tokenize(
        self,
        sentence: List[str],
        vocab_to_id: Dict[str, int],
        max_seq_len: int = HuggingFaceData.max_length.value,
        start_token: bool = False,
        end_token: bool = False,
    ) -> List[int]:
        tokens = []
        if start_token:
            tokens = [vocab_to_id[START_TOKEN]]
        tokens.extend([vocab_to_id[ch] for ch in sentence])
        if end_token:
            tokens.append(vocab_to_id[END_TOKEN])
        for _ in range(len(tokens), max_seq_len):
            tokens.append(vocab_to_id[PADDING_TOKEN])
        return torch.Tensor(tokens)

    def detockenize(
        self,
        tokens: List[int],
        id_to_vocab: Dict[int, str],
    ) -> str:
        sentence = []
        for token in tokens:
            sentence.append(id_to_vocab[token])
        return "".join(sentence)

    def create_tokens_for_a_batch(
        self, batch: List[Tuple[str]]
    ) -> torch.Tensor:  # shape batch_size, max_seq_len
        eng_tokens, mal_tokens = [], []
        eng_sentences, mal_sentences = batch
        for i in range(len(eng_sentences)):
            eng_tokens.append(self.tokenize(eng_sentences[i], self.eng_vocab_to_index))
            mal_tokens.append(
                self.tokenize(
                    mal_sentences[i],
                    self.ml_vocab_to_index,
                    start_token=True,
                    end_token=True,
                )
            )
        eng_tokens = torch.stack(eng_tokens)
        mal_tokens = torch.stack(mal_tokens)
        return eng_tokens, mal_tokens

    def detockenize_for_a_batch(
        self, tokens: torch.Tensor, index_to_vocab: Dict[int, str]
    ) -> List[str]:
        sentences = []
        for token in tokens:
            sentences.append(self.detockenize(token, index_to_vocab))
        return sentences

    def create_masks(batch: List[Tuple[str]], max_seq_len: int):
        eng_sentences, mal_sentences = batch
        batch_size = len(eng_sentences)
        encoder_padding_mask = torch.full(
            [batch_size, max_seq_len, max_seq_len], fill_value=False, dtype=torch.bool
        )
        decoder_padding_mask = torch.full(
            [batch_size, max_seq_len, max_seq_len], fill_value=False, dtype=torch.bool
        )
        cross_encoder_padding_mask = torch.full(
            [batch_size, max_seq_len, max_seq_len], fill_value=False, dtype=torch.bool
        )
        decoder_lookahead_mask = torch.full(
            [max_seq_len, max_seq_len], fill_value=True, dtype=torch.bool
        )
        decoder_lookahead_mask = torch.triu(decoder_lookahead_mask, diagonal=1)

        for i in range(batch_size):
            eng_sent_len = len(eng_sentences[i])
            mal_sent_len = len(mal_sentences[i])
            padded_indices_in_eng = torch.arange(eng_sent_len, max_seq_len)
            padded_indices_in_mal = torch.arange(
                mal_sent_len + 2, max_seq_len
            )  # To account for START, END
            encoder_padding_mask[i, :, padded_indices_in_eng] = (
                True  # Influence of all pads on every sentence
            )
            encoder_padding_mask[i, padded_indices_in_eng, :] = (
                True  # Pads towards all other
            )
            decoder_padding_mask[i, :, padded_indices_in_mal] = True
            decoder_padding_mask[i, padded_indices_in_mal, :] = True
            # Note: In case of cross encoder, Q is from decoder side and K, V are from encoder side. Hence weigh matrix row is decoder side and column is encoder side
            cross_encoder_padding_mask[i, :, padded_indices_in_eng] = True
            cross_encoder_padding_mask[i, padded_indices_in_mal, :] = True

        encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFINITY, 0)
        decoder_cross_attention_mask = torch.where(
            cross_encoder_padding_mask, NEG_INFINITY, 0
        )
        decoder_self_attention_mask = torch.where(
            decoder_padding_mask | decoder_lookahead_mask, NEG_INFINITY, 0
        )  # bool OR operation with broadcasting

        return (
            encoder_self_attention_mask,
            decoder_self_attention_mask,
            decoder_cross_attention_mask,
        )


class TranslationDataset(Dataset):
    def __init__(self, src_trg_pairs: list):
        super().__init__()
        self.src_trg_pairs = src_trg_pairs

    def __len__(self):
        return len(self.src_trg_pairs)

    def __getitem__(self, index):
        return self.src_trg_pairs[index]


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = parse()
    translator = Translation(framework=args.framework, type=args.type)
    translator.build()
