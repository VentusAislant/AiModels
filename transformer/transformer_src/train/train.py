import argparse
import os

from transformer_src.model import Transformer
from transformer_src.datasets import ChEngDataset, ChEngDataLoader

import torch
from torch import nn


def train_model(args):
    if args.device_id == -1:
        device = 'cpu'
    else:
        device = f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu'

    # Load dataset and create data loader
    dataset = ChEngDataset(data_path=args.data_path, tokenizer_path=args.tokenizer_path)
    dataloader = ChEngDataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = Transformer(tokenizer=dataset.tokenizer,
                        max_len=1024,
                        hidden_size=args.hidden_size,
                        ffn_hidden_size=args.ffn_hidden_size,
                        n_heads=args.n_heads,
                        n_layers=args.n_layers,
                        dropout=args.dropout,
                        device=device).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.sp.pad_id())

    # Train the model
    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            tgt_input = torch.cat((torch.full((tgt.shape[0], 1), model.tokenizer.sp.bos_id(),
                                              dtype=torch.long, device=device), tgt[:, :-1]), dim=1)
            output = model(src, tgt_input)
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            tgt = tgt.contiguous().view(-1)
            loss = criterion(output_reshape, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch[{epoch + 1}|{args.num_epochs}], Loss:{epoch_loss / len(dataloader)}')

    save_dir = os.path.dirname(args.model_save_path)
    os.makedirs(save_dir, exist_ok=True)
    # Save model
    torch.save(model.state_dict(), args.model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer model for Ch-Eng translation')
    parser.add_argument('--device_id', type=int, default=-1, help='The index of cuda, -1 means cpu')
    parser.add_argument('--data_path', type=str, default='../data/ch-eng/cmn.txt', help='Path to the dataset')
    parser.add_argument('--tokenizer_path', type=str, default='../data/ch-eng/tokenizer.model',
                        help='Path to the tokenizer')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of Transformer model')
    parser.add_argument('--ffn_hidden_size', type=int, default=256, help='FFN hidden size of Transformer model')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers in Transformer model')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout probability')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--model_save_path', type=str, default='../checkpoint/model.bin',
                        help='Path to save trained model')
    args = parser.parse_args()

    train_model(args)
