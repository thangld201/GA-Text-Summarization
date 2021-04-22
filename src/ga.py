import os

from rouge import Rouge 
from sacremoses import MosesTokenizer

def summarize(dictionary, corpus, threshold):

    tokenizer = MosesTokenizer(lang='en')
    summary = list()

    with open(corpus, 'r') as file:
        corpus_lines = file.readlines()

        for line in corpus_lines:

            line_tok = tokenizer.tokenize(line)

            sentence_score = 0.0
            for word in line_tok:
                if word.lower() in dictionary:
                    sentence_score += dictionary[word.lower()]
                else:
                    sentence_score += 0.0
            if sentence_score > threshold:
                summary.append(line)

    return summary

def score(summary, reference):
    avg = 0.0
    for i in range(len(reference)):
        scores = rouge.get_scores(summary[i], reference[i])
        avg += scores["rouge-1"]["f"]

    return avg / len(reference)

def evaluate(dictionary, corpus, reference, threshold):

    if corpus == "train":
        for file in os.listdir("dataset/body"):
            filename = f"dataset/body/{os.fsdecode(file)}"

            summary = summarize(dictionary, filename, 0.0)
            print(summary)
            break

    return {0.0}