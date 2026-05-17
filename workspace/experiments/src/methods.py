import random, math, json, os
import tensorflow as tf
from datetime import datetime

get_current_time = lambda: datetime.now().strftime('%Y%m%d_%H%M%S')

_DISPATCH = {
    int:   lambda v: lambda: v,
    float: lambda v: lambda: v,
    tuple: lambda v: (
        lambda: random.uniform(*v) if any(isinstance(x, float) for x in v)
        else random.randint(*v)
    ),
    list:  lambda v: lambda: random.choice(v),
}

_MUTATION = {
    int: lambda v, d: v,
    float: lambda v, d: v,
    tuple: lambda v, d: max(d[0], min(d[1], v+ (random.choice([-1, 1]) if isinstance(v, int) else random.gauss(0, 0.1 * (d[1] - d[0]))))),
    list: lambda v, d: d[(index := d.index(v)) + (1 if index == 0 else -1 if index == len(d)-1 else random.choice([-1, 1]))]
    
}

def dispatch(x):
    return _DISPATCH.get(type(x))(x)

def filter_tool(x,y):
    x = int(math.log2(x))
    y = int(math.log2(y))
    
    return lambda: pow(2, random.randint(x,y))
    

def dispatch_list(filters, layers):
    return lambda: [filters() for i in range(layers())]

def mate_hp(ind1, ind2, matepb=0.5):
    for i in range(len(ind1)):
        if i<4:
            if random.random() < matepb:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        else:
            if random.random() < matepb:
                cut_point = random.randint(0, min(len(ind1[i]), len(ind2[i])))
                ind1[i], ind2[i] = ind1[i][:cut_point]+ind2[i][cut_point:], ind2[i][:cut_point]+ind1[i][cut_point:]

    return ind1, ind2



def mutate_hp(ind, config_iter, mutpb=0.1):

    for i in range(len(ind)):
        if isinstance(ind[i], list):
            current_layers = len(ind[i])
            range_layers = next(config_iter)
            range_filters = next(config_iter)
            
            if isinstance(range_filters, tuple):
                rangelog2_filters = tuple(int(math.log2(x)) for x in range_filters)
                for pos in range(current_layers):
                    if random.random()<mutpb:
                        ind[i][pos] = pow(2, _MUTATION.get(type(range_filters))(int(math.log2(ind[i][pos])), rangelog2_filters))
            else:   
                for pos in range(current_layers):
                    if random.random()<mutpb:
                        ind[i][pos] = _MUTATION.get(type(range_filters))(ind[i][pos], range_filters)

            if random.random()<mutpb:
                new_layers = _MUTATION.get(type(range_layers))(current_layers, range_layers)
                diff = new_layers - current_layers
                if diff > 0:
                    if isinstance(range_filters, tuple):
                        for lay in range(diff): ind[i].append(filter_tool(*range_filters)())
                    else:
                        for lay in range(diff): ind[i].append(_DISPATCH.get(type(range_filters))(range_filters))
                elif diff<0:
                    for lay in range(-diff): ind[i].pop(random.randrange(len(ind[i])))
            
        else:
            domain = next(config_iter)
            if random.random()<mutpb:
                ind[i] = _MUTATION.get(type(domain))(ind[i], domain)
                    
            
    return ind,

def make_run_path():
    return f"../results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def save_generation(population, pareto, generation, genespace=None, path='../results/'):

    try:
        with open(f'{path}') as f:
            history = json.load(f)
    except FileNotFoundError:
        history = {"genespace": genespace, "generations": []}

    current_gen = {
        "generation": generation,
        "population": [
            {
                "values": list(ind), 
                "fitness": list(ind.fitness.values)
            } 
            for ind in population
        ],
        "pareto": [
            {
                "values": list(ind), 
                "fitness": list(ind.fitness.values)
            } 
            for ind in pareto
        ]
    }
    
    history["generations"].append(current_gen)

    with open(f'{path}', "w") as f:
        json.dump(history, f, indent=2)

def preparation_tf():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 