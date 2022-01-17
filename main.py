from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from dataset import Code2TestDataset
import argparse
from model import load_model_and_tokenizer, Code2TestModel
from metrics import bleu
from utils import dict_to_device

class BleuCallback(pl.Callback):
    def __init__(self, dataloader, limit = 50):
        self.dataloader = dataloader
        self.limit = limit

    def on_epoch_start(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            bleu_score = 0
            pred, gt = [], []
            for i, batch in enumerate(self.dataloader):

                if i >= self.limit:
                    break

                source, target = batch
                source, target = dict_to_device(source, pl_module.device), dict_to_device(target, pl_module.device)
                outputs = pl_module.generate(input_ids=source['input_ids'],
                                    attention_mask=source['attention_mask'],
                                    do_sample=False,
                                    early_stopping=True)
                outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
                gts = [tokenizer.decode(t, skip_special_tokens=True) for t in target['input_ids']]

                pred.extend(outputs)
                gt.extend(gts)
        
        bleu_score = bleu(pred, gt)
        trainer.logger.log_metrics({"bleu": bleu_score})


if __name__ == '__main__':
    # TODO: Add support to Hugging Face's Accelerate Lib
    # TODO: Collect metrics
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='./methods2test/corpus/raw/fm/', help='data directory')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--pretrained_model', type=str,
                        default='Salesforce/codet5-small')
    parser.add_argument('--output_dir', type=str, default='./output/')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = Path(args.data_dir)

    print('Loading Model and Tokenizer...')
    pretrained_model, tokenizer = load_model_and_tokenizer(
        args.pretrained_model)
    model = Code2TestModel(pretrained_model, tokenizer)

    print('Loading Dataset...')
    train_data = Code2TestDataset(data_dir, 'train', tokenizer)
    eval_data = Code2TestDataset(data_dir, 'eval', tokenizer)
    #test_data = Code2TestDataset(data_dir, 'test', tokenizer)

    print('Loading DataLoader...')
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=False)
    #test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='model_{epoch}.ckpt',
        save_top_k=3,
        save_last=True,
        verbose=True,
        monitor='val_loss')

    bleu_callback = BleuCallback(eval_loader)

    print('Training...')
    trainer = pl.Trainer(gpus=2,
                         max_epochs=10,
                         callbacks=[checkpoint_callback, bleu_callback],
                         strategy='ddp')
    trainer.fit(model, train_loader, eval_loader)
