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
        self.masked_attention = MultiHeadAttention(hidden_size=hidden_size, n_heads=n_heads, causal=True)
        self.add_norm1 = AddNorm(hidden_size=hidden_size, dropout=dropout)
        self.attention = MultiHeadAttention(hidden_size=hidden_size, n_heads=n_heads, causal=False)
        self.add_norm2 = AddNorm(hidden_size=hidden_size, dropout=dropout)
        self.ffn = PositionwiseFeedForward(hidden_size=hidden_size,
                                           ffn_hidden_size=ffn_hidden_size,
                                           dropout=dropout)
        self.add_norm3 = AddNorm(hidden_size=hidden_size, dropout=dropout)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor,
            encoder_output: torch.Tensor,
            encoder_output_mask: torch.Tensor
    ):
        """
        forward method for a single decoder layer
        :param x: decoder input, torch.Size([batch, sql(q), hidden_size])
        :param attention_mask: torch.Size([batch, sql(q)])
        :param encoder_output: torch.Size([batch, sql(k), hidden_size])
        :param encoder_output_mask: torch.Size([batch, sql(k)])
        :return: torch.Size([batch, sql(k), hidden_size])
        """
        # 1. compute attention
        y = self.masked_attention(q=x, attention_mask=attention_mask)

        # 2. add and norm
        x = self.add_norm1(x, y)

        # 3. attention mixed encoder output and decoder hidden_state
        y = self.attention(q=x, k=encoder_output, v=encoder_output,
                           attention_mask=attention_mask, k_attention_mask=encoder_output_mask)

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

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor,
            encoder_output: torch.Tensor,
            encoder_output_mask: torch.Tensor
    ):
        """
        forward method for a single decoder layer
        :param x: decoder input, torch.Size([batch, sql(q), hidden_size])
        :param attention_mask: torch.Size([batch, sql(q)])
        :param encoder_output: torch.Size([batch, sql(k), hidden_size])
        :param encoder_output_mask: torch.Size([batch, sql(k)])
        :return: torch.Size([batch, sql(k), hidden_size])
        """
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x=x, attention_mask=attention_mask,
                      encoder_output=encoder_output,
                      encoder_output_mask=encoder_output_mask)
        output = self.dense(x)
        return output


if __name__ == '__main__':
    from encoder import Encoder

    # if you want to run this script's main method
    # you need change relative import "from .model_components import *"  into "from model_components import *"

    # testing code
    batch, seq_len, hidden_size = 2, 3, 4
    vocab_size, max_len, ffn_hidden_size = 32000, 512, 2048
    n_heads, n_layers = 2, 4
    encoder = Encoder(vocab_size, max_len, hidden_size, ffn_hidden_size,
                      n_heads, n_layers)

    decoder = Decoder(vocab_size, max_len, hidden_size, ffn_hidden_size,
                      n_heads, n_layers)

    x = torch.ones(batch, seq_len).long()
    mask = torch.zeros(batch, seq_len)
    encoder_output = encoder(x, mask)

    dec_x = torch.ones(batch, seq_len).long() * 9
    tgt_mask = torch.ones(batch, seq_len)
    print(dec_x.shape)
    output = decoder(dec_x, tgt_mask, encoder_output, mask)
    print(output.shape)
