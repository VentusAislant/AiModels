"""
This script contains the implementation of Decoder parts

Author: [Haojie Zhang]
Date: [Mar 16, 2024]
"""

from .model_components import *

from torch import nn


class DecoderLayer(nn.Module):
    """A layer of Decoder"""

    def __init__(self, hidden_size, ffn_hidden_size, n_heads, dropout=0):
        """
        constructor of DecoderLayer
        :param hidden_size: the hidden_size of model
        :param ffn_hidden_size: the hidden_size of ffn
        :param n_heads: the number of heads
        :param dropout: dropout probability
        """
        super().__init__()
        self.masked_attention = MultiHeadAttention(hidden_size=hidden_size, n_heads=n_heads)
        self.add_norm1 = AddNorm(hidden_size=hidden_size, dropout=dropout)
        self.attention = MultiHeadAttention(hidden_size=hidden_size, n_heads=n_heads)
        self.add_norm2 = AddNorm(hidden_size=hidden_size, dropout=dropout)
        self.ffn = PositionwiseFeedForward(hidden_size=hidden_size,
                                           ffn_hidden_size=ffn_hidden_size,
                                           dropout=dropout)
        self.add_norm3 = AddNorm(hidden_size=hidden_size, dropout=dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        forward method for a single decoder layer
        :param x: decoder input, torch.Size([batch, seq_len, hidden_size])
        :param encoder_output: torch.Size([batch, seq_len, hidden_size])
        :param src_mask: mask some part of the score, usually for padding token
        :param tgt_mask: mask some part of the score, usually for preventing the model from seeing the previous time steps.
        :return: torch.Size([batch, seq_len, hidden_size])
        """
        # 1. compute attention
        y = self.masked_attention(q=x, k=x, v=x, mask=tgt_mask)

        # 2. add and norm
        x = self.add_norm1(x, y)

        # 3. attention mixed encoder output and decoder hidden_state
        y = self.attention(q=x, k=encoder_output, v=encoder_output, mask=src_mask)

        # 4. add and norm
        x = self.add_norm2(x, y)

        # 5. ffn
        y = self.ffn(x)

        # 6. add and norm
        x = self.add_norm3(x, y)
        return x


class Decoder(nn.Module):
    # The Decoder part of the Transformer
    def __init__(self, vocab_size, max_len, hidden_size, ffn_hidden_size,
                 n_heads, n_layers, dropout=0, device='cpu'):
        """
        constructor of my_transformer's decoder
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
                DecoderLayer(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size,
                             n_heads=n_heads, dropout=dropout)
            )
        self.dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, encoder_output, tgt_mask, src_mask):
        """
        forward method for decoder, input a batch of sequence token, output its hidden state
        :param x: torch.Size([batch, seq_len])
        :param encoder_output: torch.Size([batch, seq_len, hidden_size])
        :param src_mask: mask some part of attention score
        :return: torch.Size([batch, seq_len, vocab_size])
        """
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x=x, encoder_output=encoder_output,
                      src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.dense(x)
        return output


if __name__ == '__main__':
    from encoder import Encoder

    # if you want to run this script's main method
    # you need change relative import "from .model_components import *"  into "from model_components import *"

    # testing code
    batch, seq_len, hidden_size = 16, 500, 1024
    vocab_size, max_len, ffn_hidden_size = 32000, 512, 2048
    n_heads, n_layers = 8, 4
    encoder = Encoder(vocab_size, max_len, hidden_size, ffn_hidden_size,
                      n_heads, n_layers)

    decoder = Decoder(vocab_size, max_len, hidden_size, ffn_hidden_size,
                      n_heads, n_layers)

    x = torch.ones(batch, seq_len).long()
    mask = torch.zeros(batch, n_heads, seq_len, seq_len)
    encoder_output = encoder(x, mask)

    dec_x = torch.ones(batch, seq_len).long() * 9
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    print(dec_x.shape)
    output = decoder(dec_x, encoder_output, tgt_mask, mask)
    print(output.shape)
