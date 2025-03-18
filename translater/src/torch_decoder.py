"""Torch decoder architecture"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.data_dictionary import Decoder_Enum, Train, HuggingFaceData


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_attention_heads, hidden_dim, drop_prob):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.layers = nn.Sequential(
            *[
                Decoder_Block(d_model, num_attention_heads, hidden_dim, drop_prob)
                for _ in range(self.num_layers)
            ]
        )  # Note: Sequential APPLIES the layers in order unlike modulelist layer

    def forward(self, x, y, cross_mask, no_lookahead_mask):
        # x: 64, 300, 512
        # y: 64, 300, 512
        # cross_mask: 64, 300, 300
        # no_lookahead_mask: 300, 300

        # Sequential layer takes only one input, hence to use x, y and masks, we need to iterate
        for layer in self.layers:
            y = layer(x, y, cross_mask, no_lookahead_mask)  # 64, 300, 512

        return y  # 64, 300, 512


class Decoder_Block(nn.Module):
    def __init__(self, d_model, num_attention_heads, hidden_dim, drop_prob):
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.self_attention = MultiHeadAttention(self.d_model, self.num_attention_heads)
        self.encoder_decoder_attention = MultiHeadCrossAttention(
            self.d_model, self.num_attention_heads
        )
        self.norm1 = LayerNormalization(num_features=self.d_model)
        self.norm2 = LayerNormalization(num_features=self.d_model)
        self.norm3 = LayerNormalization(num_features=self.d_model)

        self.dropout1 = nn.Dropout(self.drop_prob)
        self.dropout2 = nn.Dropout(self.drop_prob)
        self.dropout3 = nn.Dropout(self.drop_prob)
        self.feed_forward = FeedForward(self.d_model, self.hidden_dim, self.drop_prob)

    def forward(self, x, y, cross_mask, no_lookahead_mask):
        # x: 64, 300, 512, y: 64, 300, 512, cross_mask: 64, 300, 300, no_lookahead_mask: 300, 300
        residual_y = y
        y = self.self_attention(y, mask=no_lookahead_mask)  # 64, 300, 512
        y = self.dropout1(y)  #  64, 300, 512
        y = self.norm1(y + residual_y)  # 64, 300, 512

        residual_y = y  # 64, 300, 512
        y = self.encoder_decoder_attention(x, y, mask=cross_mask)  # 64, 300, 512
        y = self.dropout2(y)  # 64, 300, 512
        y = self.norm2(y + residual_y)  # 64, 300, 512

        residual_y = y  # 64, 300, 512
        y = self.feed_forward(y)  # 64, 300, 512
        y = self.dropout3(y)  # 64, 300, 512
        y = self.norm3(y + residual_y)  # 64, 300, 512
        return y


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

    def forward(self, x, mask):
        # x: 64, 300, 512
        # mask: 64, 1, 300, 300
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


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model  # 512
        self.num_heads = num_heads  # 8
        self.head_dim = self.d_model // self.num_heads
        self.q_layer = nn.Linear(self.d_model, self.d_model)
        self.kv_layer = nn.Linear(self.d_model, 2 * self.d_model)
        self.linear_layer = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, y, mask):
        # x: 64, 300, 512
        # y: 64, 300, 512
        # mask: 300, 300
        batch_size, seq_length, d_model = x.size()
        q = self.q_layer(x)  # 64, 300, 512
        kv = self.kv_layer(y)  # 64, 300, 1024
        q = q.reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )  # 64, 300, 8, 64
        kv = kv.reshape(
            batch_size, seq_length, self.num_heads, 2 * self.head_dim
        )  # 64, 300, 8, 128
        q = q.permute(0, 2, 1, 3)  # 64, 8, 300, 64
        kv = kv.permute(0, 2, 1, 3)  # 64, 8, 300, 128
        k, v = kv.chunk(2, dim=-1)  # k,v: 64, 8, 300, 64
        values, attention = scaled_dot_product_attention(
            q, k, v, mask
        )  # values :64, 8, 300, 64, attention:  64, 8, 300, 300
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


if __name__ == "__main__":
    # Test
    x = torch.randn(
        Train.batch_size.value,
        HuggingFaceData.max_length.value,
        Decoder_Enum.d_model.value,
    )  # Src sentence embedding with positional encoding
    y = torch.randn(
        Train.batch_size.value,
        HuggingFaceData.max_length.value,
        Decoder_Enum.d_model.value,
    )  # Tgt sentence embedding with positional encoding
    cross_mask = torch.ones(
        Train.batch_size.value,
        1,
        HuggingFaceData.max_length.value,
        HuggingFaceData.max_length.value,
    )
    no_lookahead_mask = torch.full(
        [HuggingFaceData.max_length.value, HuggingFaceData.max_length.value],
        float("-inf"),
    )
    no_lookahead_mask = torch.triu(no_lookahead_mask, diagonal=1)

    decoder = Decoder(
        num_layers=Decoder_Enum.num_layers.value,
        d_model=Decoder_Enum.d_model.value,
        num_attention_heads=Decoder_Enum.num_attention_heads.value,
        hidden_dim=Decoder_Enum.hidden_dim.value,
        drop_prob=Decoder_Enum.drop_prob.value,
    )
    out = decoder(x, y, cross_mask, no_lookahead_mask)
