
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time


from selection_and_operators.mutation import shuffle_mutation
from selection_and_operators.crossover import cycle_crossover
from selection_and_operators.selection import tournament_selection
from library.algorithm import *
from MFLUSolution import LUSolution, LUGASolution, LUHCSolution, LUSASolution


# Function to run an experiment. It is initialized with baseline parameters

def run_experiment_ga(
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
    elapsed_time_list=[]


    for i in range(NUMBER_OF_TESTS):
        start_time = time.time()

        print(f'\nIteration {i} of the genetic algorithm\n')
        
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
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_list.append(elapsed_time)

    fitness_array = np.array(fitness_histories)
    elapsed_time_array= np.array(elapsed_time_list)
    elapsed_time_avg = np.mean(elapsed_time_array, axis=0)
    fitness_avg = np.mean(fitness_array, axis=0)
    fitness_median = np.median(fitness_array, axis=0)
    fitness_std = np.std(fitness_array, axis=0)



    stats_df = pd.DataFrame({
        "Experiment_Name": EXPERIMENT_NAME,
        "Generation": np.arange(MAX_GEN),
        "Fitness_Mean": fitness_avg,
        "Fitness_Median": fitness_median,
        "Fitness_Std": fitness_std
    })

    final_fitness = fitness_array[:, -1]  # final generation
    final_fitness_df = pd.DataFrame({
        "Experiment_name": EXPERIMENT_NAME,
        "Final_Fitness": final_fitness})

    csv_path= "results/all_final_fitness.csv"
    if os.path.exists(csv_path):
        final_fitness_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        final_fitness_df.to_csv(csv_path, index=False)

    csv_path = "results/all_experiments.csv"
    if os.path.exists(csv_path):
        stats_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        stats_df.to_csv(csv_path, index=False)

    stats_df.to_csv(f"{folder_path}/{EXPERIMENT_NAME}.csv", index=False)

    best_final_fitnesses = [run[-1] for run in fitness_histories]

    return {
    "name": EXPERIMENT_NAME,
    "fitness_array": fitness_array,
    "fitness_avg": fitness_avg,
    "fitness_median": fitness_median,
    "fitness_std": fitness_std,
    "stats_df": stats_df,
    "elapsed_time_avg": elapsed_time_avg
}



def run_experiment_hc(
EXPERIMENT_NAME : str,
NUMBER_OF_TESTS=30,
MAX_GEN=100,
MAXIMIZATION=True,
VERBOSE = False,
):
    
    os.makedirs("results", exist_ok=True)
    folder_path = "results"
    os.makedirs(folder_path, exist_ok=True)

    initial_sol = LUHCSolution()


    best_solutions = []
    fitness_histories = []
    elapsed_time_list=[]


    for i in range(NUMBER_OF_TESTS):
        start_time = time.time()

        print(f'\nIteration {i} of the HC algorithm\n')
        
        best_solution, fitness_history= hill_climbing(
            initial_solution=initial_sol,
            maximization=MAXIMIZATION,
            max_iter=MAX_GEN,
            verbose = VERBOSE,
        )

        best_solutions.append(best_solution)
        fitness_histories.append(fitness_history)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_list.append(elapsed_time)

    fitness_array = np.array(fitness_histories)
    elapsed_time_array= np.array(elapsed_time_list)
    elapsed_time_avg = np.mean(elapsed_time_array, axis=0)
    fitness_avg = np.mean(fitness_array, axis=0)
    fitness_median = np.median(fitness_array, axis=0)
    fitness_std = np.std(fitness_array, axis=0)

    # stats_df = pd.DataFrame({
    #     "Generation": np.arange(MAX_GEN),
    #     "Fitness_Mean": fitness_avg,
    #     "Fitness_Median": fitness_median,
    #     "Fitness_Std": fitness_std
    # })

    # stats_df.to_csv(f"{folder_path}/{EXPERIMENT_NAME}.csv", index=False)
    final_fitness = fitness_array[:, -1]  # final generation
    final_fitness_df = pd.DataFrame({"Final_Fitness": final_fitness})
    final_fitness_df.to_csv(f"{folder_path}/{EXPERIMENT_NAME}_final_generation_fitness.csv", index=False)


    best_final_fitnesses = [run[-1] for run in fitness_histories]

    return {
    "name": EXPERIMENT_NAME,
    "fitness_array": fitness_array,
    "fitness_avg": fitness_avg,
    "fitness_median": fitness_median,
    "fitness_std": fitness_std,
    #"stats_df": stats_df,
    "elapsed_time_avg": elapsed_time_avg
}



def run_experiment_sa(
EXPERIMENT_NAME : str,
NUMBER_OF_TESTS=30,
MAX_GEN=100,
C=2.5,
H=4.5,
L=10,
MAXIMIZATION=True,
VERBOSE = False,
):
    
    os.makedirs("results", exist_ok=True)
    folder_path = "results"
    os.makedirs(folder_path, exist_ok=True)

    initial_sol = LUSASolution()

    best_solutions = []
    fitness_histories = []
    elapsed_time_list=[]


    for i in range(NUMBER_OF_TESTS):
        start_time = time.time()

        print(f'\nIteration {i} of the SA algorithm\n')
        
        best_solution, fitness_history = simulated_annealing(
            initial_solution=initial_sol,
            maximization=MAXIMIZATION,
            C=C,
            H=H,
            L=L,
            max_iter=MAX_GEN,
            verbose = VERBOSE,
        )

        best_solutions.append(best_solution)
        fitness_histories.append(fitness_history)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_list.append(elapsed_time)

    fitness_array = np.array(fitness_histories)
    elapsed_time_array= np.array(elapsed_time_list)
    elapsed_time_avg = np.mean(elapsed_time_array, axis=0)
    fitness_avg = np.mean(fitness_array, axis=0)
    fitness_median = np.median(fitness_array, axis=0)
    fitness_std = np.std(fitness_array, axis=0)

    stats_df = pd.DataFrame({
        'Experiment_name': EXPERIMENT_NAME,
        "Generation": np.arange(MAX_GEN),
        "Fitness_Mean": fitness_avg,
        "Fitness_Median": fitness_median,
        "Fitness_Std": fitness_std
    })

    final_fitness = fitness_array[:, -1]  # final generation
    final_fitness_df = pd.DataFrame({
        "Experiment_name": EXPERIMENT_NAME,
        "Final_Fitness": final_fitness})

    csv_path= "results/all_final_fitness"
    if os.path.exists(csv_path):
        final_fitness_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        final_fitness_df.to_csv(csv_path, index=False)

    csv_path = "results/all_experiments.csv"
    if os.path.exists(csv_path):
        stats_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        stats_df.to_csv(csv_path, index=False)

    stats_df.to_csv(f"{folder_path}/{EXPERIMENT_NAME}.csv", index=False)

    best_final_fitnesses = [run[-1] for run in fitness_histories]

    return {
    "name": EXPERIMENT_NAME,
    "fitness_array": fitness_array,
    "fitness_avg": fitness_avg,
    "fitness_median": fitness_median,
    "fitness_std": fitness_std,
    "stats_df": stats_df,
    "elapsed_time_avg": elapsed_time_avg
}



colors = [
    '#1f77b4',
    '#2ca02c',
    '#ff7f0e',
    '#9467bd',
    '#8c564b',
    '#17becf'  
    ]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)


def plot_avg_median_fit_per_generation(all_results):

    generation_order = sorted(all_results[0]['stats_df']['Generation'].unique())
    tick_step = 20  # Show every 20 generations
    tick_positions = range(0, len(generation_order), tick_step)
    tick_labels = [generation_order[i] for i in tick_positions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot average fitness
    for i, result in enumerate(all_results):
        label = f"{i+1}: {result['name']}"
        ax1.plot(result["stats_df"]["Generation"], result["fitness_avg"], label=label, color=colors[i % len(colors)], linewidth=1.2)

    ax1.set_title("Average Fitness per Generation", fontsize=13, weight='bold')
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xticks(range(len(generation_order)))
    ax1.set_xticklabels(generation_order)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    # Plot median fitness
    for i, result in enumerate(all_results):
        ax2.plot(result["stats_df"]["Generation"], result["fitness_median"], label=f"{i+1}: {result['name']}", color=colors[i % len(colors)], linewidth=1.2)

    ax2.set_title("Median Fitness per Generation", fontsize=13, weight='bold')
    ax2.set_xlabel("Generation")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xticks(range(len(generation_order)))
    ax2.set_xticklabels(generation_order)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9, frameon=False)

    plt.tight_layout()
    plt.show()



def boxplots_final_fitness(all_results):
    operator_names = []
    fitness_arrays=[]
    for result in all_results:
        fitness_arrays.append(result['fitness_array'])
        operator_names.append(result['name'])

    final_fitness_by_experiment = [arr[:, -1] for arr in fitness_arrays]

    plt.figure(figsize=(10, 6))

    positions = list(range(1, len(operator_names) + 1))
    box = plt.boxplot(final_fitness_by_experiment, patch_artist=True, positions=positions, medianprops=dict(color='black') )

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.xticks(positions, labels=[str(i) for i in positions], fontsize=10)
    plt.xlabel("Experiment number", fontsize=12)
    plt.ylabel("Final Fitness Distribution", fontsize=12)
    plt.title("Final Fitness Distribution per Operator Combination", fontsize=14, weight='bold')

    for i, name in enumerate(operator_names, start=1):
        plt.plot([], [], label=f"{i}: {name}")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=9, ncol=1, frameon=False)

    plt.tight_layout()
    plt.show()


def plot_mean_std_error(all_results):
    final_means = []
    final_stds = []
    operator_names=[]

    for result in all_results:
        df = result['stats_df']

        last_row = df.iloc[-1]
        mean = last_row['Fitness_Mean']
        std = last_row['Fitness_Std']

        final_means.append(mean)
        final_stds.append(std)
        operator_names.append(result['name'])

    # Plot
    x_pos = list(range(1, len(operator_names) + 1))
    plt.figure(figsize=(10, 6))
    for i, (x, mean, std, color) in enumerate(zip(x_pos, final_means, final_stds, colors)):
        plt.errorbar(x, mean, yerr=std, fmt='o', capsize=5, markersize=6, linestyle='None', color=color, linewidth=2)
    plt.xticks(x_pos, rotation=45, ha='right', fontsize=10)
    plt.ylabel("Fitness")
    plt.title("Final Generation Fitness: Mean ± Std Dev")
    for i, name in enumerate(operator_names, start=1):
        plt.plot([], [], label=f"{i}: {name}")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=9, ncol=1, frameon=False)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def avg_elapsed_time_table(all_results):
    data = []
    for result in all_results:
        data.append({
            "Experience Name": result['name'],
            "Elapsed Time Avg": result['elapsed_time_avg']
        })
    avg_elapsed_time = pd.DataFrame(data)
    avg_elapsed_time.index = range(1, len(all_results)+1)
    return avg_elapsed_time
