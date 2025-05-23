
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import itertools


from library.selection_and_operators.mutation import shuffle_mutation
from library.selection_and_operators.crossover import cycle_crossover
from library.selection_and_operators.selection import tournament_selection
from library.algorithms import *
from library.LUSolution import LUSolution, LUGASolution, LUHCSolution, LUSASolution, LUKAPGASolution


# Function to run an experiment of the GA algorithm
# it is initialized with baseline parameters

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
ELITISM = True,
VERBOSE = False,
):
    
    os.makedirs("results", exist_ok=True)
    folder_path = "results"
    os.makedirs(folder_path, exist_ok=True)


    best_solutions = []
    fitness_histories = []
    elapsed_time_list=[]
    df = pd.DataFrame(columns=range(MAX_GEN)) # Shape will be 30 x 200


    for i in range(NUMBER_OF_TESTS):
        start_time = time.time()

        print(f'\nIteration {i} of the genetic algorithm\n')

        initial_population = [
        LUGASolution(
            crossover_function=CROSSOVER_FUNCTION,
            mutation_function=MUTATION_FUNCTION
        )
        for _ in range(POP_SIZE)
        ]
        
        best_solution, fitness_history = genetic_algorithm(
            initial_population=initial_population,
            max_gen=MAX_GEN,
            selection_algorithm=SELECTION_ALGORITHM,
            k_tournment_selection=K_TOURNEMENT_SELECTION,
            maximization = True,
            xo_prob = XO_PROB,
            mut_prob = MUT_PROB,
            elitism = ELITISM,
            verbose = VERBOSE,
        )

        best_solutions.append(best_solution)
        fitness_histories.append(fitness_history)
        df.loc[i] = fitness_history


        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_list.append(elapsed_time)

    elapsed_time_array= np.array(elapsed_time_list)
    elapsed_time_avg = np.mean(elapsed_time_array, axis=0)
    
    
    df.to_csv(f"{folder_path}/{EXPERIMENT_NAME}_df.csv", index=False)

    
    return {
        "name": EXPERIMENT_NAME,
        "df": df, 
        "avg_elapsed_time": elapsed_time_avg}


def run_experiment_ga_KAP(
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


    best_solutions = []
    fitness_histories = []
    elapsed_time_list=[]
    df = pd.DataFrame(columns=range(MAX_GEN)) # Shape will be 30 x 200


    for i in range(NUMBER_OF_TESTS):
        start_time = time.time()

        print(f'\nIteration {i} of the genetic algorithm\n')

            
        initial_population = [
        LUKAPGASolution(
            crossover_function=CROSSOVER_FUNCTION,
            mutation_function=MUTATION_FUNCTION
        )
        for _ in range(POP_SIZE)
        ]
        
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
        df.loc[i] = fitness_history


        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_list.append(elapsed_time)

    elapsed_time_array= np.array(elapsed_time_list)
    elapsed_time_avg = np.mean(elapsed_time_array, axis=0)    
    
    df.to_csv(f"{folder_path}/{EXPERIMENT_NAME}_df.csv", index=False)

    
    return {
        "name": EXPERIMENT_NAME,
        "df": df, 
        "avg_elapsed_time": elapsed_time_avg}



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

    best_solutions = []
    fitness_histories = []
    elapsed_time_list=[]


    for i in range(NUMBER_OF_TESTS):
        start_time = time.time()
        initial_sol = LUHCSolution()


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

    elapsed_time_array= np.array(elapsed_time_list)
    elapsed_time_avg = np.mean(elapsed_time_array, axis=0)
    last_fitness_values = [history[-1] for history in fitness_histories]
    df = pd.DataFrame({
    "Fitness_max": last_fitness_values
    })

    df.to_csv(f"{folder_path}/{EXPERIMENT_NAME}.csv", index=False)


    return {
    "name": EXPERIMENT_NAME,
    "df": df,
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


    best_solutions = []
    fitness_histories = []
    elapsed_time_list=[]
    df = pd.DataFrame(columns=range(MAX_GEN))


    for i in range(NUMBER_OF_TESTS):
        start_time = time.time()

        print(f'\nIteration {i} of the SA algorithm\n')
        initial_sol = LUSASolution()

        
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
        df.loc[i] = fitness_history
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_list.append(elapsed_time)

    elapsed_time_array= np.array(elapsed_time_list)
    elapsed_time_avg = np.mean(elapsed_time_array, axis=0)
    df.to_csv(f"{folder_path}/{EXPERIMENT_NAME}_df.csv", index=False)


    return {
    "name": EXPERIMENT_NAME,
    "df": df, 
    "avg_elapsed_time": elapsed_time_avg
}



def plot_avg_median_fit_per_generation(results_dict, error_bar=False):

    colors = [
        '#1f77b4', '#2ca02c', '#ff7f0e',
        '#9467bd', '#8c564b', '#17becf',
        '#e377c2', '#7f7f7f', '#bcbd22',
        '#aec7e8', '#98df8a', '#ffbb78'
    ]

    experiments = list(results_dict.keys())
    num_experiments = len(experiments)

    any_df = list(results_dict.values())[0]
    MAX_GEN = any_df.shape[1] - 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, (exp_name, df) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        label = f"{i+1}: {exp_name}"

        df_t = df.T

        avg = df_t.mean(axis=1)
        std = df_t.std(axis=1)
        med = df_t.median(axis=1)
        generations = df_t.index

        ax1.plot(generations, avg, label=label, color=color, linewidth=2)        
        if error_bar:
            ax1.errorbar(generations, avg, yerr=std, fmt='o', capsize=4, markersize=4,
             linestyle='-', color=color, linewidth=2)
        else:
            ax1.fill_between(generations, avg - std, avg + std, color=color, alpha=0.2)
        ax2.plot(generations, med, label=label, color=color, linewidth=2)
        for ax, title in zip((ax1, ax2), ("Average Fitness per Generation", "Median Fitness per Generation")):
            ax.set_title(title, fontsize=13, weight='bold')
            ax.set_xlabel("Generation")
            ax.set_xticks(list(range(0, MAX_GEN, 20)) + [MAX_GEN])  # Ensure MAX_GEN is included
            ax.set_xlim(0, MAX_GEN + 2)  # Add a little extra padding
            ax.grid(True, linestyle='--', alpha=0.5)

    ax1.set_ylabel("Fitness")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)

    plt.tight_layout()
    plt.show()



def boxplots_final_fitness(results_input, legend=True, box_color="#4C72B0"):
    operator_names = []
    final_fitness_by_experiment = []

    if isinstance(results_input, dict):
        for name, df in results_input.items():
            df_t = df if df.shape[0] > 1 else df.T
            final_fitness = df_t.iloc[:, -1].values
            final_fitness_by_experiment.append(final_fitness)
            operator_names.append(name)
    else:
        raise ValueError("Input must be a dict of DataFrames with final fitness values.")

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(1, len(operator_names) + 1)

    box = ax.boxplot(
        final_fitness_by_experiment,
        patch_artist=True,
        positions=positions,
        medianprops=dict(color='black'),
        whiskerprops=dict(color='gray'),
        capprops=dict(color='gray'),
        boxprops=dict(facecolor=box_color, color=box_color, alpha=0.8),
        flierprops=dict(marker='o', color='gray', alpha=0.5)
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in positions], fontsize=11)
    ax.set_xlabel("Experiment number", fontsize=13)
    ax.set_ylabel("Final Fitness", fontsize=13)
    ax.set_title("Final Fitness Distribution per Parameter Combination", fontsize=15, weight='bold')

    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    legend_handles = [
        plt.Line2D([], [], linestyle='None', marker=None, color='none', label=f"{i}: {name}")
        for i, name in enumerate(operator_names, start=1)
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fontsize=10,
        ncol=2,
        frameon=False
    )

    plt.tight_layout()
    plt.show()



def plot_mean_std_error(fitness_dfs, legend=True, marker_color="tab:blue"):
    final_means = []
    final_stds = []
    operator_names = []

    for name, df in fitness_dfs.items():
        df_t = df if df.shape[0] > 1 else df.T
        final_gen_values = df_t.iloc[:, -1].values

        final_means.append(np.mean(final_gen_values))
        final_stds.append(np.std(final_gen_values, ddof=1))
        operator_names.append(name)

    x_pos = list(range(1, len(operator_names) + 1))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(
        x_pos,
        final_means,
        yerr=final_stds,
        fmt='o',
        capsize=5,
        markersize=6,
        linestyle='None',
        color=marker_color,
        linewidth=2
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i) for i in x_pos], rotation=0, fontsize=10)
    ax.set_ylabel("Fitness", fontsize=12)
    ax.set_xlabel("Experiment number", fontsize=12)
    ax.set_title("Final Generation Fitness: Mean ± Std Dev", fontsize=14, weight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_handles = [
        plt.Line2D([], [], linestyle='None', marker=None, color='none', label=f"{i}: {name}")
        for i, name in enumerate(operator_names, start=1)
    ]

    plt.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=9,
        ncol=2,              
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()




def compute_wilcoxon_pvalues(fitness_dfs):
    fitness_last_gen = {
        name: df.T.iloc[:, -1].values if df.shape[0] == 1 else df.iloc[:, -1].values
        for name, df in fitness_dfs.items()
    }

    names = list(fitness_last_gen.keys())
    p_values_table = pd.DataFrame(np.nan, index=names, columns=names)

    for i, j in itertools.combinations(range(len(names)), 2):
        stat, p = wilcoxon(fitness_last_gen[names[i]], fitness_last_gen[names[j]])
        p_values_table.loc[names[i], names[j]] = p
        p_values_table.loc[names[j], names[i]] = p

    return p_values_table.round(3)

# The code for the visualizations was generated with the helo of chatgpt