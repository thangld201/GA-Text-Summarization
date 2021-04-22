import os

from rouge import Rouge 
from sacremoses import MosesTokenizer

def summarize(dictionary, corpus, threshold):

    tokenizer = MosesTokenizer(lang='en')
    summary = list()

    with open(corpus, 'r', encoding='utf-8') as file:
        corpus_lines = file.readlines()

        for line in corpus_lines:

            line_tok = tokenizer.tokenize(line)

            sentence_score = 0.0
            for word in line_tok:
                if word.lower() in dictionary:
                    sentence_score += dictionary[word.lower()]
                if sentence_score / len(line_tok) > threshold:
                    summary.append(line)
                    break
    print(summary)
    return summary

def score_summary(summary, reference_file):
    avg = 0.0
    rouge = Rouge()
    reference = open(reference_file, 'r', encoding='utf-8').readlines()

    for i in range(len(reference)):
        scores = rouge.get_scores(summary[i], reference[i])
        avg += scores[0]["rouge-1"]["f"]

    return avg / len(reference)

def evaluate(dictionary, corpus, reference, threshold):

    score = 0.0

    if corpus == "train":
        for file in os.listdir("dataset/body"):
            filename = f"dataset/body/{os.fsdecode(file)}"

            summary = summarize(dictionary, filename, 0.6)
            score += score_summary(summary, f"dataset/highlights/{os.fsdecode(file)}")
            
            break

    return {score}