"""
This script contains the implementation of Add&Norm class

Author: [Haojie Zhang]
Date: [Mar 16, 2024]
"""

import torch
from torch import nn


class LayerNorm(nn.Module):
    # A LayerNorm class, equivalent to nn.LayerNorm
    def __init__(self, hidden_size, eps=1e-12):
        """

        :param hidden_size:
        :param eps:
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, X: torch.Tensor):
        mean = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, unbiased=False, keepdim=True)

        out = (X - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class AddNorm(nn.Module):
    # AddNorm class implement residual connection and layer norm
    def __init__(self, hidden_size, dropout=0):
        super().__init__()
        self.ln = LayerNorm(hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        return self.ln(self.dropout(Y) + X)


if __name__ == '__main__':
    # testing code
    # batch norm vs layer norm
    a = torch.arange(9).reshape(3, 3).float()
    bn = nn.BatchNorm1d(3)
    print('a: ', a)
    print('batch norm a: ', bn(a))
    ln = LayerNorm(hidden_size=3)
    print('layer norm a: ', ln(a))
    ln2 = nn.LayerNorm(3)
    print(ln2(a))
