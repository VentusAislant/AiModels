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

    def forward(
            self,
            q: torch.Tensor,
            attention_mask: torch.Tensor = None,
            k: torch.Tensor = None,
            v: torch.Tensor = None,
    ):
        """
        compute attention score and output value
        :param q: query, torch.Size([bsz, n_heads, sql_q, head_hidden_size])
        :param k: key, torch.Size([bsz, n_heads, sql_k, head_hidden_size])
        :param v: value, torch.Size([bsz, n_heads, sql_v, head_hidden_size]), usually sql_k equals to sql_v
        :param attention_mask: ignore some part of the attention score, torch.Size([bsz, n_heads, sql_q, sql_k])
        :return: 
            output: torch.Size([bsz, n_heads, sql_v, head_hidden_size])
            attention score: torch.Size([bsz, n_heads, sql_q, sql_k])
        """
        # if self-attention
        head_hidden_size = k.shape[-1]

        # 1. scaled dot-product q with k^T to compute similarity
        score = (q @ k.transpose(2, 3)) / math.sqrt(head_hidden_size)
        """
        score.shape: torch.Size([bsz, n_heads, sql(q), sql(k)])
        score[,,i,j] represents the relevance between the i-th position in the query sequence
        and the j-th position in the key sequence.
        """

        # 2.apply masking
        if attention_mask is not None:
            score = score.masked_fill(attention_mask == 0, -1e6)

        # 3. Apply softmax function to ensure that the values are in the range [0, 1].
        score = self.softmax(score)

        # if some line is all -1e6, it will be computed attention, it is not allowed
        score = score.masked_fill(attention_mask == 0, 0)

        # 4. get final v
        v = score @ v

        return v, score


class MultiHeadAttention(nn.Module):
    # Multi-Head Attention class
    def __init__(self, hidden_size, n_heads, proj_bias=False, causal=False):
        """
        constructor of multi-head attention class
        :param hidden_size: the hidden size of model
        :param n_heads: the numer of heads
        """
        super().__init__()
        self.n_heads = n_heads
        self.attention = ScaledDotProductAttention()
        self.proj_q = nn.Linear(hidden_size, hidden_size, bias=proj_bias)
        self.proj_k = nn.Linear(hidden_size, hidden_size, bias=proj_bias)
        self.proj_v = nn.Linear(hidden_size, hidden_size, bias=proj_bias)
        self.proj_o = nn.Linear(hidden_size, hidden_size, bias=proj_bias)
        self.attention_score = None
        self.causal = causal

    def forward(
            self,
            q: torch.Tensor,
            attention_mask: torch.Tensor,
            k: torch.Tensor = None,
            k_attention_mask: torch.Tensor = None,
            v: torch.Tensor = None,
    ):
        """
        compute attention score and output value
        :param q: query, torch.Size([bsz, sql_q, hidden_size])
        :param k: key, torch.Size([bsz, sql_k, hidden_size])
        :param v: value, torch.Size([bsz, sql_v, hidden_size]), usually sql_k equals to sql_v
        :param attention_mask: ignore some part of the attention score, torch.Size([bsz, n_heads, sql_q, sql_k])
        :return: 
            output: torch.Size([bsz, n_heads, sql_v, head_hidden_size])
            attention score: torch.Size([bsz, n_heads, sql_q, sql_k])
        """
        # if self-attention
        if k is None:
            k = q
        if v is None:
            v = q
        # 1.get attention mask
        attention_mask = self.__get_attention_mask(attention_mask, k_attention_mask)

        # 2.dot product with weight matrices
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)  # torch.Size([bsz, sql, hidden_size])

        # 2.split tensor by number of heads
        q, k, v = self.__split_heads(q), self.__split_heads(k), self.__split_heads(v)

        # 3.scaled dot-product attention to compute similarity
        out, self.attention_score = self.attention(q=q, k=k, v=v, attention_mask=attention_mask)

        # 4.Concatenate and pass to output layer
        out = self.__concat_heads(out)
        output = self.proj_o(out)
        return output

    def __split_heads(self, t: torch.Tensor):
        """
        split q/k/v by numer of head
        :param t: torch.Size([bsz, sql, hidden_size])
        :return: torch.Size([bsz, n_heads, sql, hidden_size//n_heads])
        """
        bsz, sql, hidden_size = t.size()

        head_hidden_size = hidden_size // self.n_heads

        out = t.view(bsz, sql, self.n_heads, head_hidden_size).transpose(1, 2)

        return out

    def __concat_heads(self, t: torch.Tensor):
        """
        inverse function of self.split()
        :param t: torch.Size([bsz, n_heads, sql, head_hidden_size])
        :return: torch.Size([bsz, sql, hidden_size])
        """
        bsz, n_heads, sql, head_hidden_size = t.size()

        hidden_size = n_heads * head_hidden_size

        # contiguous() ensure that the transposed tensor's elements are laid out contiguously in memory
        out = t.transpose(1, 2).contiguous().view(bsz, sql, hidden_size)

        return out

    def __get_attention_mask(self, q_attention_mask: torch.Tensor, k_attention_mask: torch.Tensor = None):
        """
        get attention mask
        :param q_attention_mask: torch.Size([bsz, sql(q)])
        :param k_attention_mask: torch.Size([bsz, sql(k)]) / None
        :return: attention mask: torch.Size([bsz, n_heads, sql(q), sql(k)])
        """
        if k_attention_mask is None:
            # self attention
            k_attention_mask = q_attention_mask

        bsz, sql_q = q_attention_mask.shape
        _, sql_k = k_attention_mask.shape

        return_mask = torch.ones(size=(bsz, sql_q, sql_k)).bool()

        if self.causal:
            upper_right_triangular_mask = torch.triu(torch.ones(size=(sql_q, sql_k)), diagonal=1).bool()
            upper_right_triangular_mask = upper_right_triangular_mask.unsqueeze(0).expand(bsz, sql_q, sql_k)
            return_mask = return_mask.masked_fill(upper_right_triangular_mask, False)

        q_zeros_idx = torch.nonzero(torch.eq(q_attention_mask, 0))
        k_zeros_idx = torch.nonzero(torch.eq(k_attention_mask, 0))
        return_mask[q_zeros_idx[:, 0], q_zeros_idx[:, 1], :] = False
        return_mask[k_zeros_idx[:, 0], :, k_zeros_idx[:, 1]] = False

        return return_mask.unsqueeze(1).expand(bsz, self.n_heads, sql_q, sql_k).to(device=q_attention_mask.device)


if __name__ == '__main__':
    # testing ScaledDotProductAttention
    bsz, sql, hidden_size, n_heads = 2, 6, 6, 2
    X = torch.ones(bsz, n_heads, sql, hidden_size // n_heads)
    mask = torch.zeros(bsz, n_heads, sql, sql)
    sdp_attn = ScaledDotProductAttention()
    print('Before ScaledDotProductAttention X.shape: ', X.shape)
    X, score = sdp_attn(q=X, k=X, v=X, attention_mask=mask)
    print('After ScaledDotProductAttention X.shape: ', X.shape)
    print('Attention score: ', score)
    print('=' * 80)

    # testing MultiHeadAttention
    X = torch.ones(bsz, sql, hidden_size)
    attention_mask = torch.ones(bsz, sql)
    attention_mask[1:, -1] = False

    attention = MultiHeadAttention(hidden_size=hidden_size, n_heads=n_heads)
    print('Before MultiHeadAttention X.shape: ', X.shape)
    X = attention(q=X, attention_mask=attention_mask)
    print('After MultiHeadAttention X.shape: ', X.shape)
    print(X)
    print('Attention score: ', attention.attention_score)
