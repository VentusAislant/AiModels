import json


class TransformerConfig:
    # A config class for constructing Transformer
    def __init__(
            self,
            n_heads: int = 8,
            n_layers: int = 4,
            hidden_size: int = 128,
            ffn_hidden_size: int = 256,
            dropout: float = 0.05,
            max_len: int = 4096,
    ):
        """
        constructor of TransformerConfig
        :param: n_heads: number of attention heads
        :param: n_layers: number of layers, encoder and decoder have same depth
        :param: hidden_size: the width of the Transformer, must divide n_heads
        :param: ffn_hidden_size: the width of the FeedForwardNetwork
        :param: dropout: probability of dropout
        :param: max_len: the max generation length of positional embedding
        """
        assert (hidden_size / n_heads).is_integer(), \
            "Hidden size cannot be evenly divided by the number of attention heads."
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.dropout = dropout
        self.max_len = max_len

    def load(self, json_file_path):
        with open(json_file_path, 'r') as f:
            json_cfg = json.load(f)
        for key, value in json_cfg.items():
            setattr(self, key, value)
        return self

    def save(self, json_file_path, indent=4):
        with open(json_file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=indent)

    def dict(self):
        return self.__dict__

    def __iter__(self):
        return iter(self.__dict__.items())

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)


if __name__ == '__main__':
    cfg = TransformerConfig()
