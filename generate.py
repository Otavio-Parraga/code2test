import torch
from model import Code2TestModel, load_model_and_tokenizer

if __name__ == '__main__':
    pretrained_model = 'Salesforce/codet5-small'
    state_dict = torch.load('./checkpoints/20751_epoch_1.ckpt')

    pretrained_model, tokenizer = load_model_and_tokenizer(pretrained_model)
    model = Code2TestModel(pretrained_model, tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(state_dict)
    model.eval()

    text = 'public int sumTwoNumbers(int a, int b) {\n   return a + b;\n}'

    inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')

    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(input_ids=inputs['input_ids'],
                             attention_mask=inputs['attention_mask'],
                             do_sample=True,
                             max_length=150,
                             early_stopping=True)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
