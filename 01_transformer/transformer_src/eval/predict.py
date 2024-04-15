import argparse

import torch

from transformer_src.config import TransformerConfig
from transformer_src.tokenizer.tokenizer import TransformerTokenizer
from metrics import bleu
from transformer_src.model.transformer import Transformer


def predict(model, src_sentence, tokenizer: TransformerTokenizer, max_gen_len, device):
    model.eval()
    model.to(device)
    src_tokens = tokenizer.encode(src_sentence, add_eos=True)
    src = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0)

    tgt = torch.tensor([tokenizer.sp.bos_id()], dtype=torch.long, device=device).unsqueeze(0)

    output_seq = []
    for _ in range(max_gen_len):
        # print(tgt)
        # print(tgt_vocab.to_tokens(tgt))
        output = model(src, tgt)
        pred = output.argmax(dim=2)[:, -1].item()  # get predication of the last word
        if pred == tokenizer.sp.eos_id() or len(output_seq) >= max_gen_len - 1:
            break
        output_seq.append(pred)
        tgt = torch.cat((tgt, torch.tensor([[pred]], dtype=torch.long, device=device)), dim=1)

    return tokenizer.decode(torch.tensor(output_seq))


def main(cfg):
    device = cfg.device

    tokenizer = TransformerTokenizer(model_path=cfg.tokenizer_path)

    # load config
    if args.config_path is not None:
        model_config = TransformerConfig().load(args.config_path)
    else:
        model_config = TransformerConfig(
            max_len=args.max_len,
            hidden_size=args.hidden_size,
            ffn_hidden_size=args.ffn_hidden_size,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )

    model = Transformer(tokenizer=tokenizer, device=device, **model_config.dict())

    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))

    engs = ['Let\'s go', "i lost", 'he\'s calm', 'i\'m home', 'i love you', 'we want to sleep.']
    chs = ['我们出发吧', '我输了', '他很冷静', '我回家了', '我愛你', '我們想要睡']

    if 'reverse' in args.model_path:
        test_samples = zip(chs, engs)
    else:
        test_samples = zip(engs, chs)

    for org, tgt in test_samples:
        translation = predict(
            model, org, tokenizer, args.max_gen_len, device)

        import re
        translation = re.sub(r'[.!?,。！？，]', '', translation)

        print(f'{org} => {translation}, ',
              f'GT: {tgt}, '
              f'bleu {bleu(tokenizer.encode(translation.lower(), add_eos=False), tokenizer.encode(tgt.lower(), add_eos=False), k=2):.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Transformer model using some samples')
    # device
    parser.add_argument('--device_id', type=int, default=-1, help='The index of cuda, -1 means cpu')

    # dataset config
    parser.add_argument('--tokenizer_path', type=str, default='../data/ch-eng/tokenizer.model',
                        help='Path to the tokenizer')

    # model ckp
    parser.add_argument('--model_path', type=str, default='../checkpoint/model.bin',
                        help=f'Path to the model checkpoint')

    # model config
    parser.add_argument('--config_path', type=str, default='../checkpoint/config.json',
                        help=f'Path to the model config file, if not None, then DO NOT NEED to pass the params: \
                            [n_heads, n_layers, hidden_size, ffn_hidden_size, dropout, max_len]')

    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of Transformer model')
    parser.add_argument('--ffn_hidden_size', type=int, default=256, help='FFN hidden size of Transformer model')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers in Transformer model')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout probability')
    parser.add_argument('--max_len', type=int, default=4096, help='max generation length of positional embedding')

    # generate config
    parser.add_argument('--max_gen_len', type=int, default=50, help='max generation length')

    args = parser.parse_args()

    args.device = f'cuda:{args.device_id}' if args.device_id >= 0 else 'cpu'

    main(args)
