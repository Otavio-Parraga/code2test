import torch
from models import Code2TestModel, load_model_and_tokenizer

if __name__ == '__main__':
    pretrained_model = 'Salesforce/codet5-small'

    pretrained_model, tokenizer = load_model_and_tokenizer(pretrained_model)
    model = Code2TestModel.load_from_checkpoint(checkpoint_path='/home/parraga/projects/code2test/output/last.ckpt', pretrained_model=pretrained_model, tokenizer=tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    #text = 'public int sumTwoNumbers(int a, int b) {\n   return a + b;\n}'
    text = 'public int multiply(int a, int b) {\n   return a * b;\n}'

    inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')

    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(input_ids=inputs['input_ids'],
                             attention_mask=inputs['attention_mask'],
                             do_sample=True,
                             max_length=150,
                             early_stopping=True)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
