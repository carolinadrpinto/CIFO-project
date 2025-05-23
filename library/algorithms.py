import random
from copy import deepcopy
import numpy as np
from library.solution import Solution
from typing import Callable

# this algorithms and functions are based in the ones provided in class with minimal changes
# to accomodate our work/problem/analysis

def get_best_ind(population: list[Solution], maximization: bool):
    fitness_list = [ind.fitness() for ind in population]
    if maximization:
        return population[fitness_list.index(max(fitness_list))]
    else:
        return population[fitness_list.index(min(fitness_list))]

def genetic_algorithm(
    initial_population: list[Solution],
    max_gen: int,
    selection_algorithm: Callable,
    s_ranking_selection = 1,
    k_tournment_selection = 2,
    maximization: bool = False,
    xo_prob: float = 0.9,
    mut_prob: float = 0.2,
    mut_max_window_size=5,
    elitism: bool = True,
    verbose: bool = False,
):
    """
    Executes a genetic algorithm to optimize a population of solutions.

    Args:
        initial_population (list[Solution]): The starting population of solutions.
        max_gen (int): The maximum number of generations to evolve.
        selection_algorithm (Callable): Function used for selecting individuals.
        maximization (bool, optional): If True, maximizes the fitness function; otherwise, minimizes. Defaults to False.
        xo_prob (float, optional): Probability of applying crossover. Defaults to 0.9.
        mut_prob (float, optional): Probability of applying mutation. Defaults to 0.2.
        elitism (bool, optional): If True, carries the best individual to the next generation. Defaults to True.
        verbose (bool, optional): If True, prints detailed logs for debugging. Defaults to False.

    Returns:
        Solution: The best solution found on the last population after evolving for max_gen generations.
    """
    # 1. Initialize a population with N individuals
    population = initial_population
    fitness_history = []

    # 2. Repeat until termination condition
    for gen in range(1, max_gen + 1):
        if verbose:
            print(f'-------------- Generation: {gen} --------------')

        # 2.1. Create an empty population P'
        new_population = []

        # 2.2. If using elitism, insert best individual from P into P'
        if elitism:
            new_population.append(deepcopy(get_best_ind(population, maximization)))
        
        # 2.3. Repeat until P' contains N individuals
        while len(new_population) < len(population):
            # 2.3.1. Choose 2 individuals from P using a selection algorithm

            if  selection_algorithm.__name__ == 'ranking_selection':
                first_ind = selection_algorithm(population, maximization, s_ranking_selection)
                second_ind = selection_algorithm(population, maximization, s_ranking_selection)
            
            elif selection_algorithm.__name__ == 'tournament_selection':
                first_ind = selection_algorithm(population, maximization, k_tournment_selection)
                second_ind = selection_algorithm(population, maximization, k_tournment_selection)

            else:
                first_ind = selection_algorithm(population, maximization)
                second_ind = selection_algorithm(population, maximization)

            if verbose:
                print(f'Selected individuals: First:\n{first_ind}\nSecond:{second_ind}')

            # 2.3.2. Choose an operator between crossover and replication
            # 2.3.3. Apply the operator to generate the offspring
            if random.random() < xo_prob:
                offspring1, offspring2 = first_ind.crossover(second_ind)
                if verbose:
                    print(f'Applied crossover')
            else:
                offspring1, offspring2 = deepcopy(first_ind), deepcopy(second_ind)
                if verbose:
                    print(f'Applied replication')
            
            if verbose:
                print(f'Offspring:\n{offspring1}\n{offspring2}')
            
            # 2.3.4. Apply mutation to the offspring
            first_new_ind = offspring1.mutation(mut_prob, mut_max_window_size)
            # 2.3.5. Insert the mutated individuals into P'
            new_population.append(first_new_ind)

            if verbose:
                print(f'First mutated individual: {first_new_ind}')
            
            if len(new_population) < len(population):
                second_new_ind = offspring2.mutation(mut_prob)
                new_population.append(second_new_ind)
                if verbose:
                    print(f'Second mutated individual: {first_new_ind}')
        
        # 2.4. Replace P with P'
        population = new_population

        fitness_history.append(get_best_ind(population, maximization).fitness())
        # if verbose:
        print(f'Final best individual in generation {gen}: {get_best_ind(population, maximization).fitness()}')

    # 3. Return the best individual in P
    return get_best_ind(population, maximization), fitness_history



def hill_climbing(initial_solution: Solution, maximization=False, max_iter=99999, verbose=False):
    """
    Implementation of the Hill Climbing optimization algorithm.  

    The algorithm iteratively explores the neighbors of the current solution, moving to a neighbor if it improves the objective function.  
    The process continues until no improvement is found or the maximum number of iterations is reached.  

    Args:
        initial_solution (Solution): The starting solution, which must implement the `fitness()` and `get_neighbors()` methods.
        maximization (bool, optional): If True, the algorithm maximizes the fitness function; otherwise, it minimizes it. Defaults to False.
        max_iter (int, optional): The maximum number of iterations allowed before stopping. Defaults to 99,999.
        verbose (bool, optional): If True, prints progress details during execution. Defaults to False.

    Returns:
        Solution: The best solution found during the search.

    Notes:
        - The initial_solution must implement a `fitness()` and `get_neighbors()` method.
        - The algorithm does not guarantee a global optimum; it only finds a local optimum.
    """

    # Run some validations to make sure initial solution is well implemented
    run_validations(initial_solution)
    fitness_history=[]

    current = initial_solution
    improved = True
    iter = 1

    while improved:
        if verbose:
            print(f'Current solution: {current} with fitness {current.fitness()}')

        improved = False
        neighbors = current.get_neighbors() # Solution must have a get_neighbors() method

        for neighbor in neighbors:

            if verbose:
                print(f'Neighbor: {neighbor} with fitness {neighbor.fitness()}')

            if maximization and (neighbor.fitness() >= current.fitness()):
                current = deepcopy(neighbor)
                improved = True
            elif not maximization and (neighbor.fitness() <= current.fitness()):
                current = deepcopy(neighbor)
                improved = True
        fitness_history.append(current.fitness())
        iter += 1
        if iter == max_iter:
            break
    
    return current, fitness_history

def run_validations(initial_solution):
    if not isinstance(initial_solution, Solution):
        raise TypeError("Initial solution must be an object of a class that inherits from Solution")
    if not hasattr(initial_solution, "get_neighbors"):
        print(f"The method 'get_neighbors' must be implemented in the initial solution.")
    neighbors = initial_solution.get_neighbors()
    if not isinstance(neighbors, list):
        raise TypeError("get_neighbors method must return a list")
    if not all([isinstance(neighbor, type(initial_solution)) for neighbor in neighbors]):
        raise TypeError(f"Neighbors must be of the same type as solution object: {type(initial_solution)}")


def simulated_annealing(
    initial_solution: Solution,
    C: float,
    L: int,
    H: float,
    maximization: bool = True,
    max_iter: int = 10,
    verbose: bool = False,
):
    """Implementation of the Simulated Annealing optimization algorithm.

    The algorithm iteratively explores the search space using a random neighbor of the
    current solution. If a better neighbor is found, the current solution is replaced by
    that neighbor. Otherwise, the solution may still be replaced by the neighbor with a certain
    probability. This probability decreases throughout the execution. The process continues until
    the maximum number of iterations is reached.  

    The convergence speed of this algorithms depends on the initial value of control parameter C,
    he speed at which C is decreased (H), and the number of iterations in which the same C is
    maitained (L).


    Params:
        - initial_solution (SASolution): Initial solution to the optimization problem
        - C (float): Probability control parameter
        - L (int): Number of iterations with same C
        - H (float): Decreasing rate of C
        - maximization (bool): Is maximization problem?
        - max_iter (int): Maximum number of iterations
        - verbose (bool): If True, prints progress details during execution. Defaults to False.
    """
    # 1. Initialize solution
    current_solution = initial_solution

    iter = 1

    if verbose:
        print(f'Initial solution: {current_solution.repr} with fitness {current_solution.fitness()}')

    fitness_history=[]
    # 2. Repeat until termination condition
    while iter <= max_iter:
    
        # 2.1 For L times
        for _ in range(L):
            # 2.1.1 Get random neighbor
            random_neighbor = current_solution.get_random_neighbor()

            neighbor_fitness = random_neighbor.fitness()
            current_fitness = current_solution.fitness()

            if verbose:
                print(f"Random neighbor {random_neighbor} with fitness: {neighbor_fitness}")

            # 2.1.2 Decide if neighbor is accepted as new solution
            # If neighbor is better, accept it
            if (
                (maximization and (neighbor_fitness >= current_fitness))
                or(not maximization and (neighbor_fitness <= current_fitness))
            ):
                current_solution = deepcopy(random_neighbor)
                if verbose:
                    print(f'Neighbor is better. Replaced current solution by neighbor.')

            # If neighbor is worse, accept it with a certain probability
            # Maximizaton: Neighbor is worse than current solution if fitness is lower
            # Minimization: Neighbor is worse than current solution if fitness is higher
            elif (
                (maximization and (neighbor_fitness < current_fitness)
                 or (not maximization and (neighbor_fitness > current_fitness)))
            ):
                # Generate random number between 0 and 1
                random_float = random.random()
                # Define probability P
                p = np.exp(-abs(current_fitness - neighbor_fitness) / C)
                if verbose:
                    print(f'Probability of accepting worse neighbor: {p}')
                # The event happens with probability P if the random number if lower than P
                if random_float < p:
                    current_solution = deepcopy(random_neighbor)
                    if verbose:
                        print(f'Neighbor is worse and was accepted.')
                else:
                    if verbose:
                        print("Neighbor is worse and was not accepted.")

            if verbose:
                print(f"New current solution {current_solution} with fitness {current_solution.fitness()}")

        fitness_history.append(current_solution.fitness())
        # 2.2 Update C
        C = C / H
        if verbose:
            print(f'Decreased C. New value: {C}')
            print('--------------')

        iter += 1

    if verbose:
        print(f'Best solution found: {current_solution.repr} with fitness {current_solution.fitness()}')
    
    # 3. Return solution
    return current_solution, fitness_history