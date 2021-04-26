import random

import matplotlib.pyplot as plt
import numpy as np
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
    #print(pop[0] == pop[i])

print("Population Initialized")

# set up crossover
CXPB = 0.8
MUTPB = 0.05
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selTournament, tournsize=5)

num_generations = 1
threshold = 0.6
max_score = 0
max_ind = None
average_fitness_over_time = list()
best_fitness_over_time = list()

for i in range(num_generations):
    print(f"Generation: {i}")
    # Select the next generation individuals
    offspring = toolbox.select(pop, POP_SIZE)
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    print("Selection done")

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    print("Mating done")

    # Apply mutation on the offspring
    for mutant in offspring:
        if random.random() < MUTPB:
            mutant[int(random.random() * IND_SIZE)] = 0
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid_ind:
        ind.fitness.values = evaluate_ga(vocab, ind, articles, highlights, threshold)
        print(ind.fitness.values)

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    # Track average and best fitness
    fitness = 0
    best_gen = 0
    for ind in pop:
        fitness += ind.fitness.values[0]

        # save best individual of all time
        if max_score < ind.fitness.values[0]:
            max_score = ind.fitness.values[0]
            max_ind = ind

        # track fitness of best individual in generation
        if best_gen < ind.fitness.values[0]:
            best_gen = ind.fitness.values[0]

    average_fitness_over_time.append(fitness / POP_SIZE)
    best_fitness_over_time.append(best_gen)

print(f"Best Fitness: {max_score}")

with open(f"results/best.txt", 'w', encoding='utf-8') as out_file:
    for num in max_ind:
        out_file.write(f"{num}\n")

average_chart = np.array(average_fitness_over_time)
best_chart = np.array(best_fitness_over_time)

plt.plot(average_chart)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title("Average Fitness per Generation")
plt.savefig('results/average_fitness.png')

plt.plot(best_chart)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title("Best Fitness per Generation")
plt.savefig('results/best_fitness.png')