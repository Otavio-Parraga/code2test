from nltk.translate.bleu_score import corpus_bleu
import re


def bleu(hypothesis, reference):
    hypothesis = [sentence.split() for sentence in hypothesis]
    reference = [[sentence.split()] for sentence in reference]
    return corpus_bleu(reference, hypothesis)

def tokenized_bleu(hypothesis, reference, tokenizer):
    hypothesis = [tokenizer.convert_ids_to_tokens(tokenizer(sentence)['input_ids']) for sentence in hypothesis]
    reference = [[tokenizer.convert_ids_to_tokens(tokenizer(sentence)['input_ids'])] for sentence in reference]
    return corpus_bleu(reference, hypothesis)

def alpha_bleu(hypothesis, reference):
    hypothesis = [re.split('([^a-zA-Z0-9])', sentence.strip()) for sentence in hypothesis]
    for sentence in hypothesis:
        sentence.remove("") if "" in sentence else None
        sentence.remove(" ") if " " in sentence else None
    reference = [[re.split('([^a-zA-Z0-9])', sentence.strip())] for sentence in reference]
    for sentence in reference:
        sentence[0].remove("") if "" in sentence[0] else None
        sentence[0].remove(" ") if " " in sentence[0] else None
    return corpus_bleu(reference, hypothesis)
