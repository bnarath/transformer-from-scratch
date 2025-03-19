"""Torch encoder architecture"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict

from config.data_dictionary import Encoder_Enum, Train, HuggingFaceData

START_TOKEN = "<START>"
END_TOKEN = "<END>"
PADDING_TOKEN = "<PAD>"


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_attention_heads,
        hidden_dim,
        drop_prob,
        max_seq_length,
        vocab_to_index,
        START_TOKEN,
        END_TOKEN,
        PADDING_TOKEN,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.max_seq_length = max_seq_length
        self.vocab_to_index = vocab_to_index
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.sentence_embedding = SentenceEmbedding(
            self.max_seq_length,
            self.d_model,
            self.vocab_to_index,
            self.drop_prob,
            self.START_TOKEN,
            self.END_TOKEN,
            self.PADDING_TOKEN,
        )
        self.layers = nn.Sequential(
            *[
                Encoder_Block(d_model, num_attention_heads, hidden_dim, drop_prob)
                for _ in range(self.num_layers)
            ]
        )  # Note: Sequential APPLIES the layers in order unlike modulelist layer

    def forward(self, x, mask, start_token=False, end_token=False):
        # x: 64, 300, 512
        # Sequential layer takes only one input, hence to use mask, we need to iterate
        x = self.sentence_embedding(x, start_token, end_token)
        # x: 64, 300, 512
        for layer in self.layers:
            x = layer(x, mask)
        return x  # 64, 300, 512


class Encoder_Block(nn.Module):
    def __init__(self, d_model, num_attention_heads, hidden_dim, drop_prob):
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.attention = MultiHeadAttention(self.d_model, self.num_attention_heads)
        self.norm1 = LayerNormalization(num_features=self.d_model)
        self.norm2 = LayerNormalization(num_features=self.d_model)
        self.dropout1 = nn.Dropout(self.drop_prob)
        self.dropout2 = nn.Dropout(self.drop_prob)
        self.feed_forward = FeedForward(self.d_model, self.hidden_dim, self.drop_prob)

    def forward(self, x, mask=None):
        # x: 64, 300, 512
        residual_x = x
        x = self.attention(x, mask=mask)  # 64, 300, 512
        x = self.dropout1(x)  #  64, 300, 512
        x = self.norm1(x + residual_x)  # 64, 300, 512
        residual_x = x  # 64, 300, 512
        x = self.feed_forward(x)
        x = self.dropout2(
            x
        )  # dropout makes neurons 0 w.p p and scales others with 1/(1-p)
        x = self.norm2(x + residual_x)
        return x


def scaled_dot_product_attention(q, k, v, mask=None):
    # q,k,v: 64, 8, 300, 64
    # mask: 64, 1, 300, 300
    d_k = q.size()[-1]  # 64
    scaled = torch.matmul(q, k.transpose(-1, -2)) / d_k**0.5  # 64, 8, 300, 300
    if mask is not None:
        scaled += mask  # 64, 8, 300, 300
    attention = F.softmax(scaled, dim=-1)  # 64, 8, 300, 300
    values = torch.matmul(attention, v)  # 64, 8, 300, 64
    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model  # 512
        self.num_heads = num_heads  # 8
        self.head_dim = self.d_model // self.num_heads  # 64
        self.qkv_layer = nn.Linear(
            self.d_model, 3 * self.d_model
        )  # Wq, Wk, Wv together
        self.linear_layer = nn.Linear(
            self.d_model, self.d_model
        )  # For cross interaction between multiple heads

    def forward(self, x, mask=None):
        batch_size, seq_length, d_model = x.size()
        qkv = self.qkv_layer(x)  # 64, 300, 1536
        qkv = qkv.reshape(
            batch_size, seq_length, self.num_heads, 3 * self.head_dim
        )  # 64, 300, 8, 192
        qkv = qkv.permute(0, 2, 1, 3)  # 64, 8, 300, 192
        q, k, v = qkv.chunk(3, dim=-1)  # q,k,v: 64, 8, 300, 64
        values, attention = scaled_dot_product_attention(
            q, k, v, mask
        )  # values :64, 8, 300, 64, attention: 64, 8, 300, 300
        values = values.reshape(
            batch_size, seq_length, self.num_heads * self.head_dim
        )  # 64, 300, 512
        out = self.linear_layer(values)  # 64, 300, 512
        return out


class LayerNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.num_features))  # (512,)
        self.beta = nn.Parameter(torch.zeros(self.num_features))  # (512,)

    def forward(self, x):
        # x: 64, 300, 512
        mean = x.mean(-1, keepdim=True)  # 64, 300, 1
        var = ((x - mean) ** 2).mean(-1, keepdim=True)  # 64, 300, 1
        std = (var + self.eps).sqrt()  # 64, 300, 1
        y = (x - mean) / std  # 64, 300, 512
        out = self.gamma * y + self.beta  # 64, 300, 512 #Broadcasting is used
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, drop_prob):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.linear1 = nn.Linear(self.d_model, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.drop_prob)

    def forward(self, x):
        # x - (64, 300, 512)
        x = self.linear1(x)  # (64, 300, 2048)
        x = self.relu(x)  #  (64, 300, 2048)
        x = self.dropout(x)  # (64, 300, 2048)
        x = self.linear2(x)  # (64, 300, 512)
        return x


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

    def get_encoding(self):  # (mif i % 2 == )
        even_i = torch.arange(0, self.d_model, 2)
        denominator = torch.pow(10000, even_i / self.d_model)
        pos = torch.arange(self.max_seq_len).reshape(self.max_seq_len, 1)
        even_PE = torch.sin(pos / denominator)
        odd_PE = torch.cos(pos / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE


if __name__ == "__main__":
    # Test

    import pickle
    from config.data_dictionary import ROOT
    from pathlib import Path

    fp = ROOT / Path("result/preprocessor.pkl")
    with open(fp, "rb") as f:
        preprocessor = pickle.load(f)
    print(preprocessor.eng_vocab_to_index)

    encoder = Encoder(
        num_layers=Encoder_Enum.num_layers.value,
        d_model=Encoder_Enum.d_model.value,
        num_attention_heads=Encoder_Enum.num_attention_heads.value,
        hidden_dim=Encoder_Enum.hidden_dim.value,
        drop_prob=Encoder_Enum.drop_prob.value,
        max_seq_length=HuggingFaceData.max_length.value,
        vocab_to_index=preprocessor.eng_vocab_to_index,
        START_TOKEN=START_TOKEN,
        END_TOKEN=END_TOKEN,
        PADDING_TOKEN=PADDING_TOKEN,
    )

    mask = torch.ones(
        Train.batch_size.value,
        1,
        HuggingFaceData.max_length.value,
        HuggingFaceData.max_length.value,
    )
    sentences = [
        "The cat is sleeping.",
        "She loves reading books.",
        "He plays soccer every Sunday.",
        "The sun is shining brightly.",
        "We are going to the park.",
        "They have a big house.",
        "I enjoy drinking coffee.",
        "The flowers are blooming.",
        "She writes in her journal daily.",
        "Birds are singing in the morning.",
        "My dog loves to run.",
        "He is watching a movie.",
        "She made a delicious cake.",
        "We traveled to a new city.",
        "The car needs more fuel.",
        "He studies late at night.",
        "Children love playing outside.",
        "It is raining heavily today.",
        "The wind is blowing hard.",
        "My friend is visiting soon.",
        "She bought a new laptop.",
        "The baby is crying loudly.",
        "We saw a rainbow yesterday.",
        "He is learning to swim.",
        "The train arrives at noon.",
        "She listens to classical music.",
        "We visited the museum last week.",
        "He always wakes up early.",
        "My sister is a great artist.",
        "The book was very interesting.",
        "I found a lost puppy.",
        "They are having a party tonight.",
        "She finished her homework quickly.",
        "We watched a funny movie.",
        "He enjoys hiking in the mountains.",
        "The store closes at 9 PM.",
        "She bought fresh vegetables.",
        "We are planning a trip to Japan.",
        "He is practicing the piano.",
        "I love eating chocolate cake.",
        "The teacher explained the lesson well.",
        "They decorated the house for Christmas.",
        "She solved the puzzle easily.",
        "We had a picnic at the park.",
        "He runs five miles every morning.",
        "The lake is frozen in winter.",
        "She painted a beautiful sunset.",
        "We played board games all night.",
        "The airplane landed smoothly.",
        "He fixed the broken chair.",
        "She adopted a stray kitten.",
        "They are baking cookies together.",
        "I am writing a short story.",
        "The mountain view is breathtaking.",
        "She sang a lovely song.",
        "We went on a boat ride.",
        "He is designing a new website.",
        "The dog barked at the stranger.",
        "I am learning a new language.",
        "She took amazing photographs.",
        "They built a wooden treehouse.",
        "I enjoyed the science exhibition.",
        "The coffee shop was crowded.",
        "He helped an old man cross the street.",
    ]

    print(encoder(sentences, mask, start_token=True, end_token=True).shape)
