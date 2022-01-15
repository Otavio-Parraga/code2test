from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import Code2TestDataset
import argparse
from model import load_model_and_tokenizer, Code2TestModel
from train import train
from accelerate import Accelerator

if __name__ == '__main__':
    # TODO: Collect metrics
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./methods2test/corpus/raw/fm/', help='data directory')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--pretrained_model', type=str, default='Salesforce/codet5-small')

    args = parser.parse_args()

    accelerator = Accelerator()

    data_dir = Path(args.data_dir)

    print('Loading Model and Tokenizer...')
    pretrained_model, tokenizer = load_model_and_tokenizer(args.pretrained_model)
    model = Code2TestModel(pretrained_model, tokenizer)

    print('Loading Dataset...')
    train_data = Code2TestDataset(data_dir, 'train', tokenizer)
    eval_data = Code2TestDataset(data_dir, 'eval', tokenizer)
    test_data = Code2TestDataset(data_dir, 'test', tokenizer)

    print('Loading DataLoader...')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model, optimizer, train_loader, eval_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader, test_loader)

    print('Training...')
    train(model, tokenizer, train_loader, eval_loader, 10, optimizer, accelerator)
