import random
from deap import base, creator, tools
from src.fitness import evaluate_individual

filters_val = [16,32,64,128,256]

def mutate_hp(ind, indpb=0.2):
    for i in range(len(ind)+len(ind[2])-1):
        if random.random() < indpb:   # 10% of genes
            if i == 0:
                ind[i] += random.choice([-0.0001, 0.0001])
                if ind[i] < 0.0005 or ind[i] > 0.0015:
                    ind[i] = random.uniform(0.0005, 0.0015)
            elif i ==1:
                ind[i] = return_filter_neighbor(ind[i])
            else:
                ind[2][i-2] = return_filter_neighbor(ind[2][i-2])
    if random.random() < indpb:
        if len(ind[2]) == 1:
            ind[2].append(random_filter_size())
        elif len(ind[2]) == 3:
            ind[2].pop(random.randint(0,2))
        else:
            if random.random() < 0.5:
                ind[2].pop(random.randint(0,1))
            else:
                ind[2].append(random_filter_size())
    return ind,

def mate_hp(ind1, ind2):
    
    if  len(ind1[2]) == len(ind2[2]):
        for i in range(len(ind1[2])):
            if random.random() < 0.5:
                ind1[2][i], ind2[2][i] = ind2[2][i], ind1[2][i]
    else:
        if len(ind1[2]) > len(ind2[2]):
            less_layered_ind = ind2  
        else: 
            less_layered_ind = ind1 
        
        for i in range(len(less_layered_ind[2])):
            if random.random() < 0.5:
                ind1[2][i], ind2[2][i] = ind2[2][i], ind1[2][i]

        if random.random() < 0.5:
            ind1[2], ind2[2] = ind2[2], ind1[2]
                    
    for i in range(len(ind1)-2):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2

def return_filter_neighbor(value):
    index = filters_val.index(value)
    if index == 0:
        return filters_val[1]
    elif index == len(filters_val)-1:
        return filters_val[-2]
    else:
        return filters_val[index + random.choice([-1,1])]

def random_filter_size():
    return random.choice(filters_val)

def random_filters():
    return [random_filter_size() for i in range(random.randint(1,3))]
            
def setup_ga(dataset):

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    # Hyperparameter ranges
    toolbox.register("lr", random.uniform, 0.0005, 0.0015)
    toolbox.register("batch", random_filter_size)
    toolbox.register("filters", random_filters)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.lr, toolbox.batch, toolbox.filters), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual, dataset=dataset)
    toolbox.register("mate", mate_hp)
    toolbox.register("mutate", mutate_hp)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_evolution(dataset, pop_size=20, generations=10):

    toolbox = setup_ga(dataset)

    population = toolbox.population(n=pop_size)

    for gen in range(generations):
        print(f"Generation {gen+1 }")

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

