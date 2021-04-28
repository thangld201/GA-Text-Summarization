from ga import evaluate_ga, load_corpus
from dictionary import read_vocab, create_dictionary

vocab = read_vocab("all")
articles, highlights = load_corpus("test")
threshold = 0.6
print("dataset loaded")

weights = list()

with open("results/best.txt", "r", encoding='utf-8') as in_file:
        weight_lines = in_file.readlines()

        for line in weight_lines:
            weights.append(float(line.rstrip()))

print(evaluate_ga(vocab, weights, articles, highlights, threshold))