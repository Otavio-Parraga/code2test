import torch
from utils import set_seed
from model import Code2TestModel, load_model_and_tokenizer
from dataset import Code2TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import bleu, tokenized_bleu, alpha_bleu
from utils import dict_to_device, set_seed
import json
from pathlib import Path

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels

if __name__ == '__main__':
    set_seed()
    pretrained_model = 'Salesforce/codet5-small'
    checkpoint_path = '/home/parraga/projects/_masters/code2test/checkpoints/Salesforce-codet5-small/best_model.ckpt'
    output_dir = Path('/'.join(checkpoint_path.split('/')[:-1]))
    
    pretrained_model, tokenizer = load_model_and_tokenizer(pretrained_model)
    model = Code2TestModel.load_from_checkpoint(checkpoint_path=checkpoint_path, pretrained_model=pretrained_model, tokenizer=tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Code2TestDataset(path='./methods2test/corpus/raw/fm/', split='test', tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    model.to(device)

    bleu_score, tokenized_bleu_score, alpha_bleu_score = 0, 0, 0

    with torch.no_grad():
        for i,batch in enumerate(tqdm(dataloader)):
            pred, gt = [], []
            source, target = batch
            source, target = dict_to_device(source, device), dict_to_device(target, device)

            output = model.generate(input_ids=source['input_ids'],
                                    attention_mask=source['attention_mask'],
                                    do_sample=True, max_length=128)

            outputs = tokenizer.batch_decode(output, skip_special_tokens=True)
            gts = tokenizer.batch_decode(target['input_ids'], skip_special_tokens=True)
            output, gts = postprocess_text(outputs, gts)

            pred.extend(outputs)
            gt.extend(gts)
            
            bleu_score += bleu(pred, gts)
            tokenized_bleu_score += tokenized_bleu(pred, gts, tokenizer)
            alpha_bleu_score += alpha_bleu(pred, gts)


            if i % 5 == 0:
                print(f'BLEU: {bleu_score / (i+1)} CodeBLEU: {tokenized_bleu_score / (i+1)} AlphaBLEU: {alpha_bleu_score / (i+1)}')

        
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump({'bleu': bleu_score / len(dataloader), 'code_bleu': tokenized_bleu_score / len(dataloader), 'alpha_bleu': alpha_bleu_score / len(dataloader)}, f)

        
