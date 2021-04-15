import random

from deap import creator, base, tools, algorithms
from ga import evaluate

# set up objective
creator.create("ROUGEFitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.ROUGEFitnessMax)

# chromosome length
IND_SIZE=10

# set up individual
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)

# initialize an individual
ind1 = toolbox.individual()

print(ind1)

ind1.fitness.values = evaluate(ind1)

print(ind1.fitness)