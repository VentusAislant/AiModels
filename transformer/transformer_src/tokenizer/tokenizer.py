"""
This script contains the implementation of Tokenizer

Author: [Haojie Zhang]
Date: [Mar 16, 2024]
"""
import abc
import sentencepiece
import torch


class Tokenizer(abc.ABC):
    """An abstract class for Tokenizer."""

    @property
    @abc.abstractmethod
    def vocab_size(self):
        """Property to represent the vocabulary size."""
        pass

    @abc.abstractmethod
    def tokenize(self, text):
        """Method to tokenize the input text."""
        pass

    @abc.abstractmethod
    def encode(self, text):
        """Method to encode the input text."""
        pass

    @abc.abstractmethod
    def decode(self, tokens):
        """Method to decode the input tokens."""
        pass

    @abc.abstractmethod
    def batch_encode(self, texts, truncation_len):
        pass

    @abc.abstractmethod
    def batch_decode(self, tokens_list):
        pass


class TransformerTokenizer(Tokenizer):
    # A tokenizer class, can transform sentence to idx list, and inverse operation
    def __init__(self, model_path):
        super().__init__()
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.Load(model_path)

    def vocab_size(self):
        return self.sp.GetPieceSize()

    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)

    def encode(self, text, add_bos=False, add_eos=True):
        encoded_text = self.sp.EncodeAsIds(text)
        if add_bos:
            encoded_text = [self.sp.bos_id()] + encoded_text
        if add_eos:
            encoded_text.append(self.sp.eos_id())
        return torch.tensor(encoded_text)

    def decode(self, tokens: torch.Tensor):
        return self.sp.DecodeIds(tokens.tolist())

    def batch_encode(self, texts, add_bos=False, add_eos=True, truncation_len=None):
        encoded_texts = [self.encode(text, add_bos, add_eos) for text in texts]
        max_len = max([len(encoded_text) for encoded_text in encoded_texts])

        if truncation_len is not None:
            max_len = min(max_len, truncation_len)

        for i in range(len(encoded_texts)):
            encoded_texts[i] = encoded_texts[i].tolist()
            if len(encoded_texts[i]) > max_len:
                encoded_texts[i] = encoded_texts[i][:max_len]
            # Pad with pad_id to max_len
            encoded_texts[i] += [self.sp.pad_id()] * (max_len - len(encoded_texts[i]))

        return torch.tensor(encoded_texts)

    def batch_decode(self, tokens_list: torch.Tensor):
        return [self.decode(tokens) for tokens in tokens_list]
