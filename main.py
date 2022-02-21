from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from utils import set_seed
from dataset import Code2TestDataset
import argparse
from model import load_model_and_tokenizer, Code2TestModel


if __name__ == '__main__':
    # TODO: Fix actual system for arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='./methods2test/corpus/raw/fm/', help='data directory')
    parser.add_argument('-ptm','--pretrained_model', type=str,
                        default='Salesforce/codet5-small')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/')
    parser.add_argument('--prefix', action='store_true')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)

    args = parser.parse_args()

    set_seed()

    output_dir = Path(args.output_dir) / args.pretrained_model.replace('/', '-')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_dir = Path(args.data_dir)

    print('Loading Model and Tokenizer...')
    pretrained_model, tokenizer = load_model_and_tokenizer(
        args.pretrained_model)
    model = Code2TestModel(pretrained_model, tokenizer)

    print('Loading Dataset...')
    train_data = Code2TestDataset(data_dir, 'train', tokenizer, args.prefix)
    eval_data = Code2TestDataset(data_dir, 'eval', tokenizer, args.prefix)

    print('Loading DataLoader...')
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best_model',
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor='val_loss')

    early_stop_callback = EarlyStopping('val_loss', patience=2)

    print('Training...')
    trainer = pl.Trainer(gpus=args.gpus,
                         max_epochs=10,
                         callbacks=[checkpoint_callback,
                                    early_stop_callback],
                         strategy='ddp')
    trainer.fit(model, train_loader, eval_loader)
