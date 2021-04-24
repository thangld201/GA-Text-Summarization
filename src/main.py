import random

from deap import creator, base, tools, algorithms
from ga import evaluate_ga, load_corpus
from dictionary import read_vocab, create_dictionary

# set up objective
creator.create("ROUGEFitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.ROUGEFitnessMax)

# read in vocab and dataset
vocab = read_vocab()
print("vocabulary loaded")
articles, highlights = load_corpus("test")
print("dataset loaded")

# chromosome length
IND_SIZE=len(vocab)

# set up individual
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)

# set up population
POP_SIZE = 100
pop = list()
for i in range(POP_SIZE):
    pop.append(toolbox.individual())

print("Population Initialized")

# set up crossover
CXPB = 0.8
MUTPB = 0.05
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selTournament, tournsize=5)

num_generations = 100
threshold = 0.2
for i in range(num_generations):
    print(f"Generation: {i}")
    # Select the next generation individuals
    offspring = toolbox.select(pop, POP_SIZE)
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    '''
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    '''

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid_ind:
        ind.fitness.values = evaluate_ga(vocab, ind, articles, highlights, threshold)

    # The population is entirely replaced by the offspring
    pop[:] = offspring
