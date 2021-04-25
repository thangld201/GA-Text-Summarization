import os

from rouge import Rouge 
from sacremoses import MosesTokenizer
from dictionary import create_dictionary

def concat_sentences(arr):
    string = ""
    for line in arr:
        string = string + " " + line

    return string

def load_corpus(mode):
    articles = list()
    highlights = list()

    for in_file in os.listdir(f"dataset/{mode}/body"):

        filename = f"dataset/{mode}/body/{os.fsdecode(in_file)}"
        articles.append(open(filename, 'r', encoding='utf-8').readlines())
        break

    for in_file in os.listdir(f"dataset/{mode}/highlights"):

        filename = f"dataset/{mode}/highlights/{os.fsdecode(in_file)}"
        highlights.append(concat_sentences(open(filename, 'r', encoding='utf-8').readlines()))
        break

    return articles, highlights

def summarize(dictionary, corpus, threshold):

    tokenizer = MosesTokenizer(lang='en')
    summary = list()

    for line in corpus:

        line_tok = tokenizer.tokenize(line)

        sentence_score = 0.0
        for word in line_tok:
            if word.lower() in dictionary:
                sentence_score += dictionary[word.lower()]
            if sentence_score / len(line_tok) > threshold:
                summary.append(line)
                break

    return summary

def score_summary(summary, reference):
    avg = 0.0
    rouge = Rouge()
    length = len(reference)

    if len(summary) <= 0:
        return 0

    summary = concat_sentences(summary)

    scores = rouge.get_scores(summary, reference)
    avg += scores[0]["rouge-1"]["f"]

    return avg 

def evaluate(dictionary, articles, highlights, threshold):

    score = 0.0
    length = len(articles)
    for i in range(length):
        summary = summarize(dictionary, articles[i], threshold)
        score += score_summary(summary, highlights[i])

        break

    return {score }

def evaluate_ga(vocab, individual, articles, highlights, threshold):
    dictionary = create_dictionary(vocab, individual)

    return evaluate(dictionary, articles, highlights, threshold)