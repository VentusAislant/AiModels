"""
This script contains the implementation of Encoder parts

Author: [Haojie Zhang]
Date: [Mar 16, 2024]
"""

from .model_components import *

from torch import nn


class EncoderLayer(nn.Module):
    """A layer of Encoder"""

    def __init__(self, hidden_size, ffn_hidden_size, n_heads, dropout=0):
        """
        constructor of EncoderLayer
        :param hidden_size: the hidden_size of model
        :param ffn_hidden_size: the hidden_size of ffn
        :param n_heads: the number of heads
        :param dropout: dropout probability
        """
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size=hidden_size, n_heads=n_heads, causal=False)
        self.add_norm1 = AddNorm(hidden_size=hidden_size, dropout=dropout)
        self.ffn = PositionwiseFeedForward(hidden_size=hidden_size,
                                           ffn_hidden_size=ffn_hidden_size,
                                           dropout=dropout)
        self.add_norm2 = AddNorm(hidden_size=hidden_size, dropout=dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        """
        forward method for a single encoder layer
        :param x: torch.Size([batch, seq_len, hidden_size])
        :param attention_mask: mask some part of the score, usually for padding token, torch.Size([batch, seq_len])
        :return: torch.Size([batch, seq_len, hidden_size])
        """
        # 1.compute attention
        y = self.attention(q=x, attention_mask=attention_mask)

        # 2. add and norm
        x = self.add_norm1(x, y)

        # 3. fnn
        y = self.ffn(x)

        # 4.add and norm
        x = self.add_norm2(x, y)
        return x


class Encoder(nn.Module):
    # The Encoder part of the Transformer
    def __init__(self, vocab_size, max_len, hidden_size, ffn_hidden_size,
                 n_heads, n_layers, dropout=0, device='cpu'):
        """
        constructor of my_transformer's encoder
        :param vocab_size: the size of vocab
        :param max_len: the max sequence length for preconstruct the position encoding matrix
        :param hidden_size: the model's hidden size
        :param ffn_hidden_size: the hidden size of ffn
        :param n_heads: the number of heads
        :param n_layers: the number of Encoder layer
        :param dropout: dropout probability
        :param device: move position encoding matrix to device
        """
        super().__init__()
        self.embedding = TransformerEmbedding(num_embeddings=vocab_size,
                                              embedding_dim=hidden_size,
                                              max_len=max_len,
                                              dropout=dropout,
                                              device=device)
        self.layers = nn.Sequential()
        for i in range(n_layers):
            self.layers.add_module(
                'layer_' + str(i),
                EncoderLayer(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size,
                             n_heads=n_heads, dropout=dropout)
            )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        """
        forward method for decoder, input a batch of sequence token, output its hidden state
        :param x: torch.Size([batch, seq_len])
        :param attention_mask: mask some part of attention score
        :return: torch.Size([batch, seq_len, hidden_size])
        """
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


if __name__ == '__main__':
    # testing code
    # if you want to run this script's main method
    # you need change relative import "from .model_components import *"  into "from model_components import *"
    batch, seq_len, hidden_size = 2, 3, 4
    vocab_size, max_len, ffn_hidden_size = 32000, 512, 2048
    n_heads, n_layers = 2, 4
    encoder = Encoder(vocab_size, max_len, hidden_size, ffn_hidden_size,
                      n_heads, n_layers)

    X = torch.ones(batch, seq_len).long()
    mask = torch.zeros(batch, seq_len)
    print(X.shape)
    output = encoder(X, mask)
    print(output)
    print(output.shape)
