from rouge import Rouge 
from sacremoses import MosesTokenizer

def summarize(dictionary, corpus, threshold):

    tokenizer = MosesTokenizer(lang='en')
    summary = list()

    with open(corpus, 'r') as file:
        corpus_lines = file.readlines()

        for line in corpus_lines:

            line_tok = tokenizer.tokenize(line)
            line_lower = line_tok.lower()
            line_arr = line_lower.split()

            sentence_score = 0.0
            for word in line_arr:
                if word in dictionary:
                    sentence_score += dictionary[word]
                else:
                    sentence_score += 0.0
            if sentence_score > threshold:
                summary.add(line)

    return summary

def score(summary, reference):
    avg = 0.0
    for i in range(len(reference)):
        scores = rouge.get_scores(summary[i], reference[i])
        avg += scores["rouge-1"]["f"]

    return avg / len(reference)

def evaluate(dictionary, corpus, reference, threshold):