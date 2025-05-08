import pandas as pd
from copy import deepcopy
import random
from library.solution import Solution
from itertools import combinations

from df_load import artists_list, conflicts_matrix

class LUSolution(Solution):
    def __init__(
        self,
        artists: list[tuple[int]] = artists_list,
        conflicts: list[list[float]] = conflicts_matrix,
        time_slots: int = 7,
        stages: int = 5,
        repr: str = None,
    ):
        self.artists=artists
        self.conflicts=conflicts
        self.time_slots=time_slots
        self.stages=stages

        if repr:
            repr = self._validate_repr(repr) #criar função

        super().__init__(repr=repr)

    def __str__(self):
        return '\n'.join(str(row) for row in self.repr)



    def _validate_repr(self, repr):
        # Confirm repr is list
        if isinstance(repr, list):
            repr=[repr[i * self.time_slots:(i + 1) * self.time_slots] for i in range(self.stages)]
        # Make sure repr is a matrix of integers
        if not all(isinstance(idx, int) for row in repr for idx in row):
            raise TypeError('Representation must be a matrix of integers')
        # Validate matrix lenght
        if (len(repr) != (self.stages)) or (len(repr[0]) != (self.time_slots)):
            raise ValueError("The number of stages and time slots does not match the ones provided")
        # Validate matrix content
        if (set(idx for row in repr for idx in row) != set([i for i in range(len(self.artists))])):
            raise ValueError("Matrix contain repeated artists")
    
    def random_initial_representation(self):
        repr = []
        artists = list(range(35))
        random.shuffle(artists)
        repr = [artists[i * self.time_slots:(i + 1) * self.time_slots] for i in range(self.stages)]
        return repr

    def popularity_score(self):
        closers_scores = []
        for stage in self.repr:
            closer = stage[-1]        # in every stage get the last artist
            closers_scores.append(self.artists[closer][2])   # get the popularity of the artist
        optimal_pop = sorted([x[2] for x in self.artists], reverse=True)[:self.stages] # obtaining the best possible popularity
                                                                                       # across all possible combinations
        pop_score = sum(closers_scores) / sum(optimal_pop)    # divide the popularity of the closers by the optimal popularity
        return pop_score

    def diversity_score(self):
        diversity_sums = []
        for slot in zip(*self.repr):                   # * to transpose the matrix and analyze each slot
            diff_genres = set()                          # create a new set for each slot
            for artist in slot:
                diff_genres.add(self.artists[artist][1])      # a new entry for every unique genre
            diversity_sums.append(len(diff_genres) / self.stages)    # a score between 0 and 1 for each slot
                                                                   # len(diff_genres) is how many unique genres are in the slot
        diversity_score = sum(diversity_sums) / self.time_slots    
        return diversity_score

    def conflict_score(self):    
        conflict_sums = []                       # empty list to store conflicts of every slot
        for slot in zip(*self.repr):           # iterate through slots (columns)
            slot_conflicts = []                        # initialize list for every slot
            for artist1, artist2 in combinations(slot, 2):       # all unique pairs
                slot_conflicts.append(self.conflicts[artist1][artist2])  # get score
            conflict_sums.append(sum(slot_conflicts) / len(slot_conflicts))    # divide by number of combinations to get average
        worst_scenario = max(conflict_sums)
        conflict_score = 1 - (sum(conflict_sums)/ worst_scenario)/ self.time_slots    # 1 - (average all slots / worst score in this solution)
        return conflict_score
            

    def fitness(self):
        fitness = (self.popularity_score()+self.diversity_score()+self.conflict_score()) / 3
        return fitness



# class LUGASolution(LUSolution):
#     def __init__(
#         self,
#         crossover_function,
#         mutation_function,
#         artists=artists,
#         conflicts: list[list[float]] = conflicts,
#         time_slots: int = 7,
#         stages: int = 5,
#         repr: str = None,
#     ):

#         super().__init__(
#             artists=artists,
#             conflicts=conflicts,
#             time_slots=time_slots,
#             stages=stages,
#             repr=repr,
#             )
        
#         self.mutation_function=mutation_function
        # self.crossover_function=crossover_function,


