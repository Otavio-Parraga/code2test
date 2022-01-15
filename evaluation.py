from utils import dict_to_device
import torch
from tqdm import tqdm


def evaluation(model, tokenizer, test_loader):
    model.eval()
    gt = []
    pred = []
    with torch.no_grad():
        for (data, target) in tqdm(test_loader, desc='Evaluation'):
            outputs = model.generate(input_ids=data['input_ids'],
                                     attention_mask=data['attention_mask'],
                                     do_sample=True,
                                     max_length=150,
                                     early_stopping=True)
            outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            gts = [tokenizer.decode(t, skip_special_tokens=True) for t in target['input_ids']]

            pred.extend(outputs)
            gt.extend(gts)

    return pred, gt