from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# TODO: implement bleu

def bleu(hypothesis, reference):
    hypothesis = [sentence.split() for sentence in hypothesis]
    reference = [[sentence.split() ]for sentence in reference]
    return corpus_bleu(reference, hypothesis)