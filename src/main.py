import random

from deap import creator, base, tools, algorithms
from ga import evaluate
from dictionary import read_vocab, create_dictionary

# set up objective
creator.create("ROUGEFitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.ROUGEFitnessMax)

# read in vocab
vocab = read_vocab()

# chromosome length
IND_SIZE=len(vocab)

# set up individual
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)

# initialize an individual
ind1 = toolbox.individual()
dictionary1 = create_dictionary(vocab, ind1)


ind1.fitness.values = evaluate(dictionary1, "train", 0.6)

print(ind1.fitness)