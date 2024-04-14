"""
This script contains the implementation of Ch-Eng-Dataset

Author: [Haojie Zhang]
Date: [Mar 16, 2024]
"""

from transformer_src.tokenizer import TransformerTokenizer

import sentencepiece
from torch.utils.data import Dataset, DataLoader
import os


class ChEngDataset(Dataset):
    # A Dateset class for ch-eng translation
    def __init__(self, data_path, tokenizer_path=None):
        """
        constructor of ChEngDataset, need construct tokenizer for it
        :param data_path: a path to txt file which have ch-eng text pair
        """
        self.all_txt = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                eng, ch = split_line[0], split_line[1].replace('\u200b', '').replace('\u3000', '')
                self.all_txt.append((eng, ch))

        if tokenizer_path is None:
            print('Constructing Tokenizer')
            tokenizer_save_path = os.path.dirname(data_path)
            self.construct_tokenizer(target_dir=tokenizer_save_path)
            print(f'Constructed Tokenizer in {tokenizer_save_path}')
            self.tokenizer = TransformerTokenizer(model_path=os.path.join(tokenizer_save_path, 'tokenizer.model'))
        else:
            self.tokenizer = TransformerTokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.all_txt)

    def __getitem__(self, item):
        if isinstance(item, int):
            source_ids = self.all_txt[item][0]
            target_ids = self.all_txt[item][1]
            return (source_ids, target_ids)
        elif isinstance(item, slice):
            return [self.__getitem__(i) for i in range(*item.indices(len(self.all_txt)))]
        return [self.__getitem__(i) for i in item]

    def construct_tokenizer(self, target_dir):
        eng_txt = ''
        ch_txt = ''
        for pair in self.all_txt:
            eng_txt += pair[0] + '\n'
            ch_txt += pair[1] + '\n'

        all_txt = ch_txt + eng_txt

        with open('tmp.txt', 'w', encoding='utf-8') as f:
            f.write(all_txt)

        sentencepiece.SentencePieceTrainer.Train(input='./tmp.txt', model_prefix='tokenizer',
                                                 vocab_size=8000, minloglevel=5,
                                                 unk_id=0, unk_piece='<unk>',
                                                 bos_id=1, bos_piece='<bos>',
                                                 eos_id=2, eos_piece='<eos>',
                                                 pad_id=3, pad_piece='<pad>')

        import shutil
        shutil.move('./tokenizer.model', os.path.join(target_dir, "tokenizer.model"))
        shutil.move('./tokenizer.vocab', os.path.join(target_dir, "tokenizer.vocab"))
        os.remove('tmp.txt')


class ChEngDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        # print(batch)
        sources, tagrets = [], []
        for item in batch:
            sources.append(item[0])
            tagrets.append(item[1])
        input_ids = self.dataset.tokenizer.batch_encode(sources)
        target_ids = self.dataset.tokenizer.batch_encode(tagrets)
        return input_ids, target_ids


if __name__ == '__main__':
    dataset = ChEngDataset('../../data/ch-eng/cmn.txt')
    tokenizer = dataset.tokenizer
    a = tokenizer.encode('hello. how are you, 你好阿')
    print(a)
    b = tokenizer.decode(a)
    print(b)

    print('=' * 50)
    dc = ChEngDataLoader(dataset, batch_size=5)
    for x, y in dc:
        print(x)
        print(y)
        break
