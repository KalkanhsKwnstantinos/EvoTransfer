import random
from deap import base, creator, tools
from src.fitness import evaluate_individual


def setup_ga(dataset):

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    filters = [16,32,64,128,256]
    # Hyperparameter ranges
    toolbox.register("lr", random.uniform(0.0005, 0.0015)
    toolbox.register("batch", random.choice(filters)
    toolbox.register("filters_1", random.choice(filters)
    toolbox.register("filters_2", random.choice(filters)
    toolbox.register("filters_3", random.choice(filters)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.lr, toolbox.batch, toolbox.filters_1,
                      toolbox.filters_2, toolbox.filters_3), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual, dataset=dataset)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_evolution(dataset, pop_size=20, generations=10):

    toolbox = setup_ga(dataset)

    population = toolbox.population(n=pop_size)

    for gen in range(generations):
        print(f"Generation {gen}")

        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(c1, c2)

        for ind in offspring:
            toolbox.mutate(ind)

        population[:] = offspring

    return population