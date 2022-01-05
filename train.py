from evaluation import evaluation
from utils import dict_to_device
import torch
from metrics import bleu


def train(model, tokenizer,dataloader, evalloader, epochs, optimizer, device):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for i, (source, target) in enumerate(dataloader):
            source = dict_to_device(source, device)
            target = dict_to_device(target, device)
            optimizer.zero_grad()

            # replace padding token id's of the labels by -100 ->
            # https://huggingface.co/docs/transformers/model_doc/t5
            labels = target['input_ids'].clone()
            labels[labels == model.tokenizer.pad_token_id] = -100

            outputs = model(source['input_ids'],
                            source['attention_mask'],
                            labels=labels)

            loss = outputs[0]
            loss.backward()
            optimizer.step()

            if i % 250 == 0 and i != 0:
                # TODO: Evaluation is taking too long in my machine
                preds, gt = evaluation(model, tokenizer, evalloader, device)

                print('Epoch: {}/{}'.format(epoch + 1, epochs),
                      'Iteration: {}/{}'.format(i + 1, len(dataloader)),
                      'Loss: {:.4f}'.format(loss.item()),
                      'BLEU: {}'.format(bleu(preds, gt)))

                torch.save(model.state_dict(), f'./checkpoints/{i+1}_epoch_{epoch+1}.ckpt')
