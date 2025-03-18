"""Torch transformer architecture"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.data_dictionary import Encoder_Enum, Train, HuggingFaceData


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_attention_heads, hidden_dim, drop_prob):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.layers = nn.Sequential(
            *[
                Encoder_Block(d_model, num_attention_heads, hidden_dim, drop_prob)
                for _ in range(self.num_layers)
            ]
        )  # Note: Sequential APPLIES the layers in order unlike modulelist layer

    def forward(self, x):
        return self.layers(x)


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
    # mask: 64, 300, 300
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


if __name__ == "__main__":
    # Test
    x = torch.randn(
        Train.batch_size.value,
        HuggingFaceData.max_length.value,
        Encoder_Enum.d_model.value,
    )
    mask = torch.ones(
        Train.batch_size.value,
        HuggingFaceData.max_length.value,
        HuggingFaceData.max_length.value,
    )
    encoder = Encoder(
        num_layers=Encoder_Enum.num_layers.value,
        d_model=Encoder_Enum.d_model.value,
        num_attention_heads=Encoder_Enum.num_attention_heads.value,
        hidden_dim=Encoder_Enum.hidden_dim.value,
        drop_prob=Encoder_Enum.drop_prob.value,
    )
    out = encoder(x)
