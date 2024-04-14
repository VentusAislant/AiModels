"""
This script contains the implementation of TransformerEmbedding class

Author: [Haojie Zhang]
Date: [Mar 16, 2024]
"""

import torch
from torch import nn

from .positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    # TransformerEmbedding class
    def __init__(self, num_embeddings, embedding_dim, max_len, dropout=0, device='cpu'):
        """
        constructor of TransformerEmbedding class, normal embedding layer + positional encoding

        :param num_embeddings: usually equals to vocab_size
        :param embedding_dim:  usually equals to hidden_size of the model
        :param max_len: the max sequence length
        :param dropout: dropout probability
        :param device: move the positional encoding matrix to device
        """
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        # positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size=embedding_dim, max_len=max_len,
                                                      device=device)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X):
        """
        embed X with added positional encoding
        :param X: torch.Size([batch, seq_len])
        :return: torch.Size([batch, seq_len, hidden_size])
        """
        X = self.embedding(X)  # torch.Size([batch, seq_len, hidden_size])
        X = self.positional_encoding(X)
        return self.dropout(X)


if __name__ == '__main__':
    # testing code

    # if you want to run this script's main method
    # you need change relative import "from .positional_encoding import PositionalEncoding"
    # into "from positional_encoding import PositionalEncoding"

    vocab_size, hidden_size, max_len = 32000, 1024, 512
    embedding = TransformerEmbedding(num_embeddings=vocab_size,
                                     embedding_dim=hidden_size,
                                     max_len=max_len)

    batch, seq_len = 16, 500
    X = torch.ones(batch, seq_len).long()
    print('Before Embedding X.shape: ', X.shape)
    X = embedding(X)
    print('After Embedding X.shape: ', X.shape)
