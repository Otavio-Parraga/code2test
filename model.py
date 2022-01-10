from pathlib import Path
import torch.nn as nn
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


class Code2TestModel(nn.Module):
    def __init__(self, pretrained_model, tokenizer):
        super(Code2TestModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer

    def forward(self, source_ids, source_mask, labels):

        return self.pretrained_model(source_ids,
                                     attention_mask=source_mask,
                                     labels=labels)

    def encode(self, source_ids, source_mask):
        return self.pretrained_model.encoder(source_ids, attention_mask=source_mask)

    def generate(self, **inputs):
        return self.pretrained_model.generate(**inputs)
