
import numpy as np
import os
import pandas as pd

from selection_and_operators.mutation import shuffle_mutation
from selection_and_operators.crossover import cycle_crossover
from selection_and_operators.selection import tournament_selection
from library.algorithm import genetic_algorithm
from MFLUSolution import LUSolution, LUGASolution


# Function to run an experiment. It is initialized with baseline parameters

def run_experiment(
EXPERIMENT_NAME : str,
POP_SIZE = 50, 
CROSSOVER_FUNCTION = cycle_crossover,
MUTATION_FUNCTION = shuffle_mutation,
NUMBER_OF_TESTS=30,
MAX_GEN=100,
S_RANKING_SELECTION=1.5,
K_TOURNEMENT_SELECTION=5,
SELECTION_ALGORITHM=tournament_selection,
XO_PROB=0.9,
MUT_PROB=0.4,
MUT_MAX_WINDOW_SIZE=5, 
ELITISM = True,
VERBOSE = False,
):
    
    os.makedirs("results", exist_ok=True)
    folder_path = "results"
    os.makedirs(folder_path, exist_ok=True)

    initial_population = [
    LUGASolution(
        crossover_function=CROSSOVER_FUNCTION,
        mutation_function=MUTATION_FUNCTION
    )
    for _ in range(POP_SIZE)
    ]

    best_solutions = []
    fitness_histories = []

    for i in range(NUMBER_OF_TESTS):

        print(f'Iteration {i} of the genetic algorithm\n')
        
        best_solution, fitness_history = genetic_algorithm(
            initial_population=initial_population,
            max_gen=MAX_GEN,
            selection_algorithm=SELECTION_ALGORITHM,
            k_tournment_selection=K_TOURNEMENT_SELECTION,
            maximization = True,
            xo_prob = XO_PROB,
            mut_prob = MUT_PROB,
            mut_max_window_size=MUT_MAX_WINDOW_SIZE,
            elitism = ELITISM,
            verbose = VERBOSE,
        )

        best_solutions.append(best_solution)
        fitness_histories.append(fitness_history)
    
    fitness_array = np.array(fitness_histories)

    fitness_avg = np.mean(fitness_array, axis=0)
    fitness_median = np.median(fitness_array, axis=0)
    fitness_std = np.std(fitness_array, axis=0)

    stats_df = pd.DataFrame({
        "Generation": np.arange(MAX_GEN),
        "Fitness_Mean": fitness_avg,
        "Fitness_Median": fitness_median,
        "Fitness_Std": fitness_std
    })

    stats_df.to_csv(f"{folder_path}/{EXPERIMENT_NAME}.csv", index=False)


    best_final_fitnesses = [run[-1] for run in fitness_histories]
    threshold = np.percentile(best_final_fitnesses, 90)
    success_count = sum(f >= threshold for f in best_final_fitnesses)
    success_rate = success_count / NUMBER_OF_TESTS


    with open("results/success_rates_overview.txt", "a") as f:
        f.write(f"{EXPERIMENT_NAME}: {success_rate:.4f}\n")


    return {
    "name": EXPERIMENT_NAME,
    "fitness_array": fitness_array,
    "fitness_avg": fitness_avg,
    "fitness_median": fitness_median,
    "fitness_std": fitness_std,
    "success_rate": success_rate,
    "stats_df": stats_df
}
