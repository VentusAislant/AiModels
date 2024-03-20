"""
This script contains the implementation of positional encoding class

Author: [Haojie Zhang]
Date: [Mar 16, 2024]
"""

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    # Sine-Cosine Positional Encoding
    def __init__(self, hidden_size, max_len, device='cpu'):
        """
        constructor of positional encoding class, We precompute the encoding matrix based on the
        maximum sequence length to avoid computing it every time.

        :param hidden_size: the hidden_size of the model
        :param max_len: the max sequence length
        :param device: to put the encoding matrix to device
        """
        super().__init__()

        # encoding matrix, don't need to compute gradient
        self.encoding_matrix = torch.zeros(max_len, hidden_size,
                                           device=device, requires_grad=False)

        # compute encoding matrix

        # torch.Size([max_len, 1])
        i = torch.arange(0, max_len, dtype=torch.float32, device=device).reshape(-1, 1)
        # torch.Size([hidden_size//2])
        _2j = torch.arange(0, hidden_size, step=2, dtype=torch.float32, device=device)
        # even-numbered column
        self.encoding_matrix[:, 0::2] = torch.sin(i / (1e4 ** (_2j / hidden_size)))
        # odd-numbered column
        self.encoding_matrix[:, 1::2] = torch.cos(i / (1e4 ** (_2j / hidden_size)))

    def forward(self, X):
        """
        Given an embedding X, Return its value with added positional encoding

        :param X: an embedding vector, torch.Size([batch, seq_len, hidden_size])
        :return: an embedding vector with added positional encoding
        """
        seq_len = X.shape[1]
        return self.encoding_matrix[:seq_len, :].unsqueeze(0) + X


if __name__ == '__main__':
    # testing code
    hidden_size, max_len, device = 256, 256, 'cpu'
    position_encoding = PositionalEncoding(hidden_size, max_len, device)
    print('positional encoding matrix\'s shape: ', position_encoding.encoding_matrix.shape)

    # A zero embedding to facilitate visualizing the position encoding matrix.
    X = torch.zeros(1, 256, 256)
    print('After positional encoding matrix\'s shape: ', position_encoding(X).shape)

    # Display the heatmap of the positional encoding matrix
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.imshow(position_encoding.encoding_matrix, cmap='viridis', aspect='auto')
    plt.xlabel('Encoding Dimension')
    plt.ylabel('Position')
    plt.title('Heatmap of Positional Encoding Matrix')
    plt.colorbar(label='Value')
    plt.show()
