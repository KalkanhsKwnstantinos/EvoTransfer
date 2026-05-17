from deap import base, creator, tools
from src.fitness import evaluate_individual
from src.methods import mate_hp, mutate_hp, make_run_path, save_generation
            
def setup_ga(genespace, dataset):
    
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    
    
    toolbox.register("lr", genespace('lr'))
    toolbox.register("batch", genespace('batch_size'))
    toolbox.register("dropout", genespace('dropout_rate'))
    toolbox.register("kernel", genespace('kernel'))
    toolbox.register("conv_filters", genespace('conv_filters'))
    toolbox.register("dense_filters", genespace('dense_filters'))
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.lr, toolbox.batch, toolbox.dropout, toolbox.kernel, toolbox.conv_filters, toolbox.dense_filters), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual, dataset=dataset)
    toolbox.register("mate", mate_hp)
    toolbox.register("mutate", mutate_hp)
    toolbox.register("select", tools.selNSGA2)

    return toolbox


def run_evolution(genespace, dataset, pop_size=20, generations=10, matepb=0.5, mutpb=0.1, prev_pop = None):
    
    toolbox = setup_ga(genespace, dataset)
    
    population = prev_pop if prev_pop else toolbox.population(n=pop_size)

    print("Generation 0")
    
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Setup tracking
    pareto = tools.ParetoFront()
    pareto.update(population)
    results_path = make_run_path()

    save_generation(population=population, pareto=pareto, generation=0, genespace=genespace.config, path=results_path)
    
    for gen in range(generations):
        print(f"Generation {gen+1}")

        offspring = list(map(toolbox.clone, population))
        
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(c1, c2, matepb=0.1)

            del c1.fitness.values
            del c2.fitness.values

        for ind in offspring:
            toolbox.mutate(ind=ind,config_iter=iter(genespace.config.values()), mutpb=0.5)
            
            del ind.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        combined_population = population + offspring

        population = toolbox.select(combined_population, pop_size)

        pareto.update(population)
        
        save_generation(population=population, pareto=pareto, generation=gen, path=results_path)

    return population

