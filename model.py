from pathlib import Path
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RobertaTokenizer, T5ForConditionalGeneration


def load_model_and_tokenizer(model_name, tokenizer_type='roberta'):
    cache_path = Path('./pretrained_stuff')
    cache_path.mkdir(exist_ok=True, parents=True)
    if tokenizer_type == 'roberta':
        return T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='./pretrained_stuff'), \
               RobertaTokenizer.from_pretrained(model_name, cache_dir='./pretrained_stuff')
    else:
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='./pretrained_stuff'), \
               AutoTokenizer.from_pretrained(model_name, cache_dir='./pretrained_stuff')


class Code2TestModel(pl.LightningModule):
    def __init__(self, pretrained_model, tokenizer):
        super(Code2TestModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=2e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [scheduler]

    def forward(self, source_ids, source_mask, labels):
        return self.pretrained_model(source_ids,
                                     attention_mask=source_mask,
                                     labels=labels)

    def training_step(self, batch, batch_idx):
        source, target = batch
        labels = target['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self.forward(source['input_ids'],
                        source['attention_mask'],
                        labels=labels)
        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        source, target = batch
        labels = target['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self.forward(source['input_ids'],
                        source['attention_mask'],
                        labels=labels)
        loss = outputs[0]
        self.log('val_loss', loss)
        return loss

    def encode(self, source_ids, source_mask):
        return self.pretrained_model.encoder(source_ids, attention_mask=source_mask)

    def generate(self, **inputs):
        return self.pretrained_model.generate(**inputs)
