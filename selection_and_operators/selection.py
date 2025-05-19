import sys
import numpy as np
sys.path.append('..')
import random
from copy import deepcopy
from library.solution import Solution


#Solution has to return me a matrix and has to have a method fitness

def ranking_selection(population: list[Solution], maximization: bool, s=1.5):
    fitness_values= [ind.fitness() for ind in population]

    if maximization:
        ranking = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)
    else:
        # Minimization
        ranking = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=False)

    population_size = len(ranking)
    #it can be tuned s
    ranks_array = np.arange(population_size)
    print('ranks_array:', ranks_array)
    probabilities = (2 - s)/population_size + (2 * ranks_array * (s - 1)) / (population_size * (population_size - 1))
    print('prob1:', probabilities)
    probabilities = probabilities[::-1]  # reverse so that the the individual with rank 0 gets the highest probability of being chosen
    print('prob2:', probabilities)

    selected_indices = np.random.choice(population_size, size=population_size, p=probabilities)
    print('selected_indexs:', selected_indices)

    parents = [deepcopy(ranking[i][0]) for i in selected_indices]
    best=parents[0]

    return deepcopy(best)
        

def tournament_selection(population: list[Solution], maximization: bool, k=2): #30 to 40 % moderated pressure
    population_size = len(population)

    for _ in range(population_size):
        tournament_participants = np.random.choice(population, size=k, replace=False)        
        if maximization:
            best = max(tournament_participants, key=lambda ind: ind.fitness())
        else:
            best = min(tournament_participants, key=lambda ind: ind.fitness())
    return deepcopy(best)
        

def fitness_proportionate_selection(population: list[Solution], maximization: bool):
    # total_fitness = sum([ind.fitness() for ind in population])

    if maximization:
        fitness_values = [ind.fitness() for ind in population]
    else:
        # Minimization: Use the inverse of the fitness value
        # Lower fitness should have higher probability of being selected
        fitness_values = [1 / ind.fitness() for ind in population]

    total_fitness = sum(fitness_values)

    # Generate random number between 0 and total
    random_nr = random.uniform(0, total_fitness)
    box_boundary = 0
    # For each individual, check if random number is inside the individual's "box"
    for ind_idx, ind in enumerate(population):
        box_boundary += fitness_values[ind_idx]
        if random_nr <= box_boundary:
            return deepcopy(ind)