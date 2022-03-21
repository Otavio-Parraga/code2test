from pathlib import Path
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoTokenizer, RobertaTokenizer, T5ForConditionalGeneration, EncoderDecoderModel, AutoModelForCausalLM, AutoModel


def load_model_and_tokenizer(model_name):
    cache_path = Path('./pretrained_stuff')
    cache_path.mkdir(exist_ok=True, parents=True)
    if 'codet5' in model_name.lower():
        return T5ForConditionalGeneration.from_pretrained(model_name, cache_dir='./pretrained_stuff'), \
            RobertaTokenizer.from_pretrained(
                model_name, cache_dir='./pretrained_stuff')
    else:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, cache_dir='./pretrained_stuff')
        tokenizer = RobertaTokenizer.from_pretrained(model_name, cache_dir='./pretrained_stuff')
        tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer
            


class Code2TestModel(pl.LightningModule):
    def __init__(self, pretrained_model_name, pretrained_model, tokenizer):
        super(Code2TestModel, self).__init__()
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=2e-5)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        source, target = batch
        labels = target['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self.pretrained_model(input_ids=source['input_ids'],
                                        attention_mask=source['attention_mask'],
                                        labels=labels)
        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        source, target = batch
        labels = target['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = self.pretrained_model(input_ids=source['input_ids'],
                                        attention_mask=source['attention_mask'],
                                        labels=labels)
        loss = outputs[0]
        self.log('val_loss', loss)
        return loss

    def encode(self, source_ids, source_mask):
        return self.pretrained_model.encoder(source_ids, attention_mask=source_mask)

    def generate(self, **inputs):
        return self.pretrained_model.generate(**inputs)
