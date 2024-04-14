import torch
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


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    tokenizer = TransformerTokenizer(model_path='./data/ch-eng/tokenizer.model')

    model = Transformer(tokenizer=tokenizer,
                        max_len=1024,
                        hidden_size=128,
                        ffn_hidden_size=256,
                        n_heads=8,
                        n_layers=4,
                        dropout=0.05,
                        device=device)
    model.load_state_dict(torch.load('./checkpoint/model.bin', map_location='cpu'))
    engs = ['go', "i lost", 'he\'s calm', 'i\'m home', 'i love you', 'we want to sleep.']
    chs = ['走', '我迷路了', '他很冷静', '我在家', '我愛你', '我們想要睡']
    for eng, ch in zip(engs, chs):
        # print(eng)
        translation = predict(
            model, eng, tokenizer, 50, device)

        # 去除标点
        import re

        translation = re.sub(r'[.!?,。！？，]', '', translation)

        print(f'{eng} => {translation}, ',
              f'bleu {bleu(tokenizer.encode(translation, add_eos=False), tokenizer.encode(ch, add_eos=False), k=2):.5f}')
