"""
This script contains the implementation of some attention classes

Author: [Haojie Zhang]
Date: [Mar 16, 2024]
"""
import math

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    # A class for computing scaled dot-product attention score for multi-head attention
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """
        compute attention score

        :param q: query, torch.Size([batch, n_heads, seq_len, head_hidden_size])
        :param k: key, torch.Size([batch, n_heads, seq_len, head_hidden_size])
        :param v: value, torch.Size([batch, n_heads, seq_len, head_hidden_size])
        :param mask: ignore some part of the attention score
        :return: torch.Size([batch, n_heads, seq_len, head_hidden_size])
        """
        head_hidden_size = k.shape[-1]

        # 1. scaled dot-product q with k^T to compute similarity
        score = (q @ k.transpose(2, 3)) / math.sqrt(head_hidden_size)
        """
        score.shape: torch.Size([batch, n_heads, seq_len(q), seq_len(k)])
        score[,,i,j] represents the relevance between the i-th position in the query sequence
        and the j-th position in the key sequence.
        """

        # 2.apply masking
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e6)

        # 3. Apply softmax function to ensure that the values are in the range [0, 1].
        score = self.softmax(score)

        # 4. get final v
        v = score @ v

        return v, score


class MultiHeadAttention(nn.Module):
    # Multi-Head Attention class
    def __init__(self, hidden_size, n_heads):
        """
        constructor of multi-head attention class
        :param hidden_size: the hidden size of model
        :param n_heads: the numer of heads
        """
        super().__init__()
        self.n_heads = n_heads
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, hidden_size)
        self.attention_weights = None

    def forward(self, q, k, v, mask=None):
        """
        compute multi head attention
        :param q: query, torch.Size([batch, seq_len, hidden_size])
        :param k: key, torch.Size([batch, seq_len, hidden_size])
        :param v: value, torch.Size([batch, seq_len, hidden_size])
        :param mask: ignore some part of the attention score
        :return: torch.Size([batch, seq_len, hidden_size])
        """

        # 1.dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)  # torch.Size([batch, seq_len, hidden_size])

        # 2.split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3.scaled dot-product attention to compute similarity
        out, self.attention_weights = self.attention(q=q, k=k, v=v, mask=mask)

        # 4.Concatenate and pass to output layer
        out = self.concat(out)
        output = self.w_o(out)
        return output

    def split(self, t: torch.Tensor):
        """
        split q/k/v by numer of head
        :param t: torch.Size([batch, seq_len, hidden_size])
        :return: torch.Size([batch, n_heads, seq_len, hidden_size//n_heads])
        """
        batch, seq_len, hidden_size = t.size()

        head_hidden_size = hidden_size // self.n_heads

        out = t.view(batch, seq_len, self.n_heads, head_hidden_size).transpose(1, 2)

        return out

    def concat(self, t: torch.Tensor):
        """
        inverse function of self.split()
        :param t: torch.Size([batch, n_heads, seq_len, head_hidden_size])
        :return: torch.Size([batch, seq_len, hidden_size])
        """
        batch, n_heads, seq_len, head_hidden_size = t.size()

        hidden_size = n_heads * head_hidden_size

        # contiguous() ensure that the transposed tensor's elements are laid out contiguously in memory
        out = t.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)

        return out


if __name__ == '__main__':
    # testing ScaledDotProductAttention
    batch, seq_len, hidden_size, n_heads = 2, 6, 6, 2
    X = torch.ones(batch, n_heads, seq_len, hidden_size // n_heads)
    mask = torch.zeros(batch, n_heads, seq_len, seq_len)
    sdp_attn = ScaledDotProductAttention()
    print('Before ScaledDotProductAttention X.shape: ', X.shape)
    X, score = sdp_attn(q=X, k=X, v=X, mask=mask)
    print('After ScaledDotProductAttention X.shape: ', X.shape)
    print('Attention score: ', score)
    print('=' * 80)

    # testing MultiHeadAttention
    X = torch.ones(batch, seq_len, hidden_size)
    attention = MultiHeadAttention(hidden_size=hidden_size, n_heads=n_heads)
    print('Before MultiHeadAttention X.shape: ', X.shape)
    X = attention(q=X, k=X, v=X, mask=mask)
    print('After MultiHeadAttention X.shape: ', X.shape)
    print('Attention score: ', attention.attention_weights)
