"""
This script contains the implementation of Transformer class

Author: [Haojie Zhang]
Date: [Mar 16, 2024]
"""
import torch

from .encoder import Encoder
from .decoder import Decoder
from ..tokenizer import TransformerTokenizer
from torch import nn


class Transformer(nn.Module):
    # Transformer class
    def __init__(self, tokenizer: TransformerTokenizer, max_len, hidden_size,
                 ffn_hidden_size, n_heads, n_layers, dropout=0, device='cpu'):
        """
        constructor of Transformer
        :param tokenizer: the Tokenizer class
        :param max_len: the max sequence len for preconstruct positional encoding matrix
        :param hidden_size: the hidden size of model
        :param ffn_hidden_size: the hidden size of ffn
        :param n_heads: the number of heads
        :param n_layers: the number of layers
        :param dropout: the dropout probability
        :param device: the positional encoding matrix will be moved to device
        """
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size()
        self.pad_idx = tokenizer.sp.pad_id()
        self.encoder = Encoder(vocab_size=self.vocab_size,
                               max_len=max_len,
                               hidden_size=hidden_size,
                               ffn_hidden_size=ffn_hidden_size,
                               n_heads=n_heads,
                               n_layers=n_layers,
                               dropout=dropout,
                               device=device)

        self.decoder = Decoder(vocab_size=self.vocab_size,
                               max_len=max_len,
                               hidden_size=hidden_size,
                               ffn_hidden_size=ffn_hidden_size,
                               n_heads=n_heads,
                               n_layers=n_layers,
                               dropout=dropout,
                               device=device)

    def forward(self, src, tgt):
        """
        forward for Transformer
        :param src: torch.Size([batch_size, seq_len])
        :param tgt: torch.Size([batch_size, seq_len])
        :return: torch.Size([batch_size, vocab_size])
        """
        src_mask = self.get_attention_mask(src)
        tgt_mask = self.get_attention_mask(tgt)
        enc_output = self.encoder(x=src, attention_mask=src_mask)
        output = self.decoder(
            x=tgt,
            attention_mask=tgt_mask,
            encoder_output=enc_output,
            encoder_output_mask=src_mask)

        return output

    def get_attention_mask(self, t: torch.Tensor):
        attention_mask = (t != self.pad_idx).to(self.device)
        return attention_mask
