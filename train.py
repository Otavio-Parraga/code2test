from evaluation import evaluation
from utils import dict_to_device
from pathlib import Path
import torch
from metrics import bleu



def train(model, tokenizer,dataloader, evalloader, epochs, optimizer, accelerator):
    #model.to(device)
    model.train()
    checkpoint_path = Path('./checkpoints/')
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    for epoch in range(epochs):
        for i, (source, target) in enumerate(dataloader):
            optimizer.zero_grad()

            # replace padding token id's of the labels by -100 ->
            # https://huggingface.co/docs/transformers/model_doc/t5
            labels = target['input_ids'].clone()
            labels[labels == model.tokenizer.pad_token_id] = -100

            outputs = model(source['input_ids'],
                            source['attention_mask'],
                            labels=labels)

            loss = outputs[0]
            accelerator.backward(loss)
            optimizer.step()

            if i % 100 == 0 and i != 0:
                print('Epoch: {}/{}'.format(epoch + 1, epochs),
                      'Iteration: {}/{}'.format(i + 1, len(dataloader)),
                      'Loss: {:.4f}'.format(loss.item()))

            if i % 1000 == 0 and i != 0:
                # TODO: Evaluation is taking too long in my machine
                preds, gt = evaluation(model, tokenizer, evalloader, limitation=100)
                torch.save(model.state_dict(), checkpoint_path / f'{i+1}_epoch_{epoch+1}.ckpt')
                print('BLEU: {}'.format(bleu(preds, gt)))
