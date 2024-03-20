"""
This script contains the implementation of position-wise feed forward network

Author: [Haojie Zhang]
Date: [Mar 16, 2024]
"""

from torch import nn


class PositionwiseFeedForward(nn.Module):
    # position-wise feed forward network, equals to MLP
    def __init__(self, hidden_size, ffn_hidden_size, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.linear2 = nn.Linear(ffn_hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
